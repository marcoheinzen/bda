# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

# @title CELL CATALOG: POPULATE bda.sqlite FROM VALIDATE + cities_config + RENAME LUT
import stack_catalog
importlib.reload(stack_catalog)
from stack_catalog import BDACatalog
from stack_align_rename import rename_tif_for_convention
import os

# clean up old DB at wrong path if exists
_old_db = DATASET_ROOT / 'bda.sqlite'
if _old_db.exists() and str(_old_db) != str(CATALOG_DB):
    os.remove(str(_old_db))
    print(f"  Removed stale DB: {_old_db}")

print("=" * 70)
print("CATALOG: POPULATE bda.sqlite")
print("=" * 70)
print(f"  DB: {CATALOG_DB}")

with BDACatalog(CATALOG_DB) as cat:
    # --- 1. import ALL cities from cities_config.json (not just CITIES_TO_PROCESS) ---
    cat.upsert_cities_from_config(CITIES_CONFIG_FILE, updated_by='NB05')

    # --- 2. build RENAME LUT: scan SATELLITE_DIR sources, compute renamed filenames ---
    n_lut = 0
    lut_rows = []

    source_scan = []
    for city_name in CITIES_TO_PROCESS:
        source_scan.extend([
            (city_name, 'CARD',  'flat',               SAR_CARD_DIR / city_name),
            (city_name, 'COH',   'flat',               SAR_COH_DIR / city_name),
            (city_name, 'MS',    'flat',               MS_DIR / city_name),
            (city_name, 'CARD',  'temporal_stats',     SAR_CARD_DIR / city_name / 'temporal_stats'),
            (city_name, 'COH',   'coherence_baseline', SAR_COH_DIR / city_name / 'coherence_baseline'),
            (city_name, 'MS',    'composite',          MS_DIR / city_name / 'composites'),
            (city_name, 'MS',    'rgb',                MS_DIR / city_name / 'rgb'),
            (city_name, 'MS',    'nbr',                MS_DIR / city_name / 'nbr'),
            (city_name, 'landuse', 'landuse',          LANDUSE_DIR / city_name),
            (city_name, 'CARD',  'rolling',            TEMPORAL_ROOT / city_name / 'CARD' / 'rolling'),
            (city_name, 'COH',   'rolling',            TEMPORAL_ROOT / city_name / 'COH' / 'rolling'),
            (city_name, 'COH',   'zscore',             TEMPORAL_ROOT / city_name / 'COH' / 'zscore'),
            (city_name, 'COH',   'post_baseline',      TEMPORAL_ROOT / city_name / 'COH' / 'post_baseline'),
            (city_name, 'CARD',  'dietrich_baseline',  DIETRICH_DRIVE / city_name / 'baseline'),
            (city_name, 'CARD',  'dietrich_assessment', DIETRICH_DRIVE / city_name / 'assessment'),
        ])

    for city_name, sensor, product_type, src_dir in source_scan:
        if not src_dir.exists():
            continue
        tifs = sorted(src_dir.rglob('*.tif')) if product_type in ('composite', 'landuse') else sorted(src_dir.glob('*.tif'))
        for tif in tifs:
            original = tif.name
            renamed = rename_tif_for_convention(city_name, original)
            rel_in_src = str(tif.relative_to(src_dir))
            lut_rows.append({
                'city_name': city_name,
                'sensor': sensor,
                'product_type': product_type,
                'original_filename': original,
                'renamed_filename': renamed,
                'src_rel_path': rel_in_src,
            })
            n_lut += 1

    cat.upsert_rename_lut_batch(lut_rows)
    print(f"  Rename LUT: {n_lut} entries across {len(CITIES_TO_PROCESS)} cities")

    # --- 3. register products from data_stack (scan TIFs after ALIGN) ---
    n_products = 0
    for city_name in CITIES_TO_PROCESS:
        city_stack = STACK_ROOT / city_name
        if not city_stack.exists():
            continue
        for tif in city_stack.rglob("*.tif"):
            if tif.name in ('building_labels.tif', 'damage_mask.tif'):
                continue
            rel = tif.relative_to(city_stack)
            parts = rel.parts
            if 'SAR_CARD' in parts:
                sensor = 'CARD'
            elif 'SAR_SLC' in parts:
                sensor = 'COH'
            elif 'multispectral' in parts:
                sensor = 'MS'
            elif 'landuse' in parts:
                sensor = 'landuse'
            elif 'temporal' in parts:
                sensor = parts[parts.index('temporal') + 1] if len(parts) > parts.index('temporal') + 1 else 'temporal'
            elif 'Dietrich' in str(rel):
                sensor = 'CARD'
            else:
                sensor = 'other'

            if 'flat' in parts:
                ptype = 'flat'
            elif 'temporal_stats' in parts:
                ptype = 'temporal_stats'
            elif 'coherence_baseline' in parts:
                ptype = 'coherence_baseline'
            elif 'composites' in parts:
                ptype = 'composite'
            elif 'rolling' in parts:
                ptype = 'rolling'
            elif 'zscore' in parts:
                ptype = 'zscore'
            elif 'post_baseline' in parts:
                ptype = 'post_baseline'
            elif 'Dietrich_baseline' in parts:
                ptype = 'dietrich_baseline'
            elif 'Dietrich_assessment' in parts:
                ptype = 'dietrich_assessment'
            elif 'rgb' in parts:
                ptype = 'rgb'
            elif 'nbr' in parts:
                ptype = 'nbr'
            else:
                ptype = 'other'

            cat.upsert_product(city_name, sensor, ptype, tif.name,
                rel_path=str(rel),
                file_size_bytes=tif.stat().st_size,
                updated_by='NB05',
            )
            n_products += 1

    print(f"  Products: {n_products} TIFs registered")
    cat.summary()
