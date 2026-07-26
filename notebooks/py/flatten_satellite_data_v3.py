# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

#!/usr/bin/env python3
"""
MIGRATION SCRIPT v3: Flatten satellite data to clean directory structure.
Adds PRE/CROSS/POST labels to COH filenames based on battle dates.

Target structure:
  satellite/SAR_CARD/{city}/{city}_CARD_{VV|VH}_{YYYYMMDD}.tif
  satellite/SAR_CARD/{city}/temporal_stats/...
  satellite/SAR_COH/{city}/{city}_COH_{VV|VH}_{PRE|CROSS|POST}_{YYYYMMDD1}_{YYYYMMDD2}.tif
  satellite/SAR_COH/{city}/coherence_baseline/...
  satellite/MS/{city}/{city}_S2_{YYYYMMDD}_{BAND}_{RES}.tif
  satellite/MS/{city}/composites/...
  satellite/MS/{city}/rgb/...
  satellite/MS/metadata/...
  satellite/SAR_METADATA/*.json
  satellite/SAR_SLC_ORBIT/...    (orbit files, unchanged)
  satellite/temporal_products/... (NB04c, unchanged)
  satellite/landuse_classification/... (unchanged)

Usage:
  python flatten_satellite_data_v3.py --dry-run
  python flatten_satellite_data_v3.py
"""

import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

if os.name == 'nt':
    SAT_ROOT = Path(r'F:\PROJECTS\masterthesis\gdrive\masterthesis\data\satellite')
    DATA_ROOT = Path(r'F:\PROJECTS\masterthesis\gdrive\masterthesis\data')
elif os.path.exists('/content/drive_f'):
    SAT_ROOT = Path('/content/drive_f/masterthesis/data/satellite')
    DATA_ROOT = Path('/content/drive_f/masterthesis/data')
else:
    SAT_ROOT = Path('/mnt/f/PROJECTS/masterthesis/gdrive/masterthesis/data/satellite')
    DATA_ROOT = Path('/mnt/f/PROJECTS/masterthesis/gdrive/masterthesis/data')

DRY_RUN = '--dry-run' in sys.argv

DERIVED_SUBDIRS = {'temporal_stats', 'coherence_baseline', 'composites', 'rgb',
                   'cloud_masks', 'metadata'}

stats = {'moved': 0, 'renamed': 0, 'deleted_safe': 0, 'deleted_empty': 0,
         'deleted_dirs': 0, 'skipped': 0, 'errors': []}

print("=" * 70)
print("FLATTEN SATELLITE DATA v3 (with COH labeling)")
print("=" * 70)
print(f"  SAT_ROOT: {SAT_ROOT}")
print(f"  DRY_RUN:  {DRY_RUN}")
print(f"  Time:     {datetime.now().isoformat()}")

if not SAT_ROOT.exists():
    print(f"  ERROR: {SAT_ROOT} does not exist")
    sys.exit(1)

# ============================================================================
# LOAD BATTLE DATES FROM cities_config.json
# ============================================================================

cities_config_file = DATA_ROOT / 'cities_config.json'
if not cities_config_file.exists():
    print(f"  ERROR: {cities_config_file} not found")
    sys.exit(1)

with open(cities_config_file) as f:
    cities_config = json.load(f)

print(f"  Cities config: {len(cities_config)} cities loaded")


def classify_coh_pair(city, date1_str, date2_str):
    """Classify a COH pair as PRE/CROSS/POST based on battle dates.
    date1_str, date2_str: YYYYMMDD strings
    Returns: 'PRE', 'CROSS', or 'POST'
    """
    info = cities_config.get(city)
    if info is None:
        return 'CROSS'  # unknown city, safe default

    battle_start = info.get('battle_start', '')
    battle_stop = info.get('battle_stop', '')

    if not battle_start:
        return 'CROSS'

    bs = battle_start.replace('-', '')  # YYYYMMDD

    # ongoing conflict: no POST possible
    is_ongoing = (not battle_stop) or battle_stop == 'ongoing'

    if is_ongoing:
        if date1_str < bs and date2_str < bs:
            return 'PRE'
        else:
            return 'CROSS'

    be = battle_stop.replace('-', '')  # YYYYMMDD

    if date2_str <= bs:
        # both dates before or at battle start
        return 'PRE'
    elif date1_str >= be:
        # both dates at or after battle end
        return 'POST'
    else:
        return 'CROSS'


# ============================================================================
# HELPERS
# ============================================================================

def safe_move(src, dst):
    if DRY_RUN:
        return True
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        stats['errors'].append(f"MOVE FAIL: {src} -> {dst}: {e}")
        return False


def safe_delete(path, is_dir=False):
    if DRY_RUN:
        return True
    try:
        if is_dir:
            shutil.rmtree(str(path))
        else:
            path.unlink()
        return True
    except Exception as e:
        stats['errors'].append(f"DELETE FAIL: {path}: {e}")
        return False


def remove_desktop_ini(path):
    di = path / 'desktop.ini'
    if di.exists() and not DRY_RUN:
        di.unlink()


# ============================================================================
# STEP 1: SAR_CARD_bbox -> SAR_CARD
# ============================================================================
print(f"\n{'='*70}")
print("STEP 1: SAR_CARD_bbox -> SAR_CARD")
print(f"{'='*70}")

old_card = SAT_ROOT / 'SAR_CARD_bbox'
new_card = SAT_ROOT / 'SAR_CARD_new'

if old_card.exists():
    if not DRY_RUN:
        new_card.mkdir(parents=True, exist_ok=True)

    for city_dir in sorted(old_card.iterdir()):
        if not city_dir.is_dir() or city_dir.name == 'desktop.ini':
            continue
        city = city_dir.name
        dst_city = new_card / city
        if not DRY_RUN:
            dst_city.mkdir(parents=True, exist_ok=True)
        city_moved = 0

        for item in sorted(city_dir.iterdir()):
            if not item.is_dir():
                continue

            if item.name in DERIVED_SUBDIRS:
                dst_derived = dst_city / item.name
                if not DRY_RUN:
                    shutil.copytree(str(item), str(dst_derived), dirs_exist_ok=True)
                print(f"  {city}/{item.name}/ -> kept as subdir")
                continue

            tifs = list(item.glob('*.tif'))
            for tif in tifs:
                dst = dst_city / tif.name
                if dst.exists():
                    stats['skipped'] += 1
                    continue
                if safe_move(tif, dst):
                    city_moved += 1
                    stats['moved'] += 1

        if city_moved > 0:
            print(f"  {city:25s}: {city_moved} TIFs flattened")
else:
    print("  SAR_CARD_bbox not found, skipping")


# ============================================================================
# STEP 2: SAR_SLC_bbox -> SAR_COH (flatten + rename + label PRE/CROSS/POST)
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2: SAR_SLC_bbox -> SAR_COH (with PRE/CROSS/POST labels)")
print(f"{'='*70}")

old_slc = SAT_ROOT / 'SAR_SLC_bbox'
new_coh = SAT_ROOT / 'SAR_COH'

label_stats = {'PRE': 0, 'CROSS': 0, 'POST': 0}

if old_slc.exists():
    if not DRY_RUN:
        new_coh.mkdir(parents=True, exist_ok=True)

    for city_dir in sorted(old_slc.iterdir()):
        if not city_dir.is_dir() or city_dir.name == 'desktop.ini':
            continue
        city = city_dir.name
        dst_city = new_coh / city
        if not DRY_RUN:
            dst_city.mkdir(parents=True, exist_ok=True)
        city_moved = 0

        for period_dir in sorted(city_dir.iterdir()):
            if not period_dir.is_dir():
                continue

            if period_dir.name in DERIVED_SUBDIRS:
                dst_derived = dst_city / period_dir.name
                if not DRY_RUN:
                    shutil.copytree(str(period_dir), str(dst_derived), dirs_exist_ok=True)
                print(f"  {city}/{period_dir.name}/ -> kept as subdir")
                continue

            # collect all COH TIFs from pair subdirs and direct
            all_tifs = list(period_dir.glob('*.tif'))
            for sub in period_dir.iterdir():
                if sub.is_dir() and re.match(r'\d{8}_\d{8}', sub.name):
                    all_tifs.extend(sub.glob('*.tif'))

            for tif in all_tifs:
                m = re.match(r'(VV|VH)_coherence_(\d{4})-(\d{2})-(\d{2})_(\d{4})-(\d{2})-(\d{2})\.tif',
                             tif.name)
                if m:
                    pol = m.group(1)
                    d1 = f"{m.group(2)}{m.group(3)}{m.group(4)}"
                    d2 = f"{m.group(5)}{m.group(6)}{m.group(7)}"
                    label = classify_coh_pair(city, d1, d2)
                    label_stats[label] += 1
                    new_name = f"{city}_COH_{pol}_{label}_{d1}_{d2}.tif"
                else:
                    new_name = f"{city}_{tif.name}"
                    label = '?'

                dst = dst_city / new_name
                if dst.exists():
                    stats['skipped'] += 1
                    continue
                if safe_move(tif, dst):
                    city_moved += 1
                    stats['moved'] += 1
                    stats['renamed'] += 1

        if city_moved > 0:
            print(f"  {city:25s}: {city_moved} COH TIFs (PRE={label_stats['PRE']}, CROSS={label_stats['CROSS']}, POST={label_stats['POST']})")
            label_stats = {'PRE': 0, 'CROSS': 0, 'POST': 0}
else:
    print("  SAR_SLC_bbox not found, skipping")


# ============================================================================
# STEP 3: multispectral -> MS (flatten, delete .SAFE)
# ============================================================================
print(f"\n{'='*70}")
print("STEP 3: multispectral -> MS (flatten, delete .SAFE)")
print(f"{'='*70}")

old_ms = SAT_ROOT / 'multispectral'
new_ms = SAT_ROOT / 'MS'

if old_ms.exists():
    if not DRY_RUN:
        new_ms.mkdir(parents=True, exist_ok=True)

    old_meta = old_ms / 'metadata'
    if old_meta.exists():
        dst_meta = new_ms / 'metadata'
        if not DRY_RUN:
            shutil.copytree(str(old_meta), str(dst_meta), dirs_exist_ok=True)
        print(f"  metadata/ -> MS/metadata/")

    for city_dir in sorted(old_ms.iterdir()):
        if not city_dir.is_dir() or city_dir.name in ('metadata', 'temp', 'desktop.ini'):
            continue
        city = city_dir.name
        dst_city = new_ms / city
        if not DRY_RUN:
            dst_city.mkdir(parents=True, exist_ok=True)
        city_moved = 0
        city_safe_deleted = 0

        for item in sorted(city_dir.iterdir()):
            if not item.is_dir():
                if item.suffix == '.tif':
                    dst = dst_city / item.name
                    if not dst.exists():
                        if safe_move(item, dst):
                            city_moved += 1
                            stats['moved'] += 1
                continue

            if item.name in DERIVED_SUBDIRS:
                dst_derived = dst_city / item.name
                if not DRY_RUN:
                    shutil.copytree(str(item), str(dst_derived), dirs_exist_ok=True)
                print(f"  {city}/{item.name}/ -> kept as subdir")
                continue

            for sub_item in sorted(item.iterdir()):
                if sub_item.is_dir():
                    if sub_item.name.endswith('.SAFE'):
                        if safe_delete(sub_item, is_dir=True):
                            city_safe_deleted += 1
                            stats['deleted_safe'] += 1
                    continue

                if sub_item.suffix == '.tif':
                    dst = dst_city / sub_item.name
                    if dst.exists():
                        stats['skipped'] += 1
                        continue
                    if safe_move(sub_item, dst):
                        city_moved += 1
                        stats['moved'] += 1

        report = f"  {city:25s}: {city_moved} TIFs flattened"
        if city_safe_deleted > 0:
            report += f", {city_safe_deleted} .SAFE deleted"
        if city_moved > 0 or city_safe_deleted > 0:
            print(report)
else:
    print("  multispectral not found, skipping")


# ============================================================================
# STEP 4: SAR_SLC/metadata -> SAR_METADATA, orbits -> SAR_SLC_ORBIT
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4: SAR_SLC/metadata -> SAR_METADATA, orbits -> SAR_SLC_ORBIT")
print(f"{'='*70}")

old_slc_meta = SAT_ROOT / 'SAR_SLC' / 'metadata'
new_sar_meta = SAT_ROOT / 'SAR_METADATA'

if old_slc_meta.exists():
    if not DRY_RUN:
        shutil.copytree(str(old_slc_meta), str(new_sar_meta), dirs_exist_ok=True)
    n_files = sum(1 for f in old_slc_meta.iterdir() if f.is_file())
    print(f"  SLC metadata: {n_files} files -> SAR_METADATA/")
else:
    print("  SAR_SLC/metadata not found")

# sentinel_1_orbits -> SAR_SLC_ORBIT (rename only)
old_orbits = SAT_ROOT / 'sentinel_1_orbits'
new_orbits = SAT_ROOT / 'SAR_SLC_ORBIT'
if old_orbits.exists() and not new_orbits.exists():
    if not DRY_RUN:
        old_orbits.rename(new_orbits)
    print(f"  sentinel_1_orbits -> SAR_SLC_ORBIT")
elif old_orbits.exists() and new_orbits.exists():
    print(f"  SAR_SLC_ORBIT already exists, skipping orbit rename")
else:
    print(f"  sentinel_1_orbits not found")


# ============================================================================
# STEP 5: DELETE OLD DIRS, RENAME TEMP -> FINAL
# ============================================================================
print(f"\n{'='*70}")
print("STEP 5: CLEANUP + RENAME")
print(f"{'='*70}")

if not DRY_RUN:
    # Delete empty/obsolete dirs
    for old_name in ['SAR_CARD', 'SAR_CARD_polygon', 'SAR_SLC_polygon']:
        old_dir = SAT_ROOT / old_name
        if old_dir.exists():
            shutil.rmtree(str(old_dir))
            print(f"  Deleted empty: {old_name}/")
            stats['deleted_dirs'] += 1

    # Rename SAR_CARD_new -> SAR_CARD
    if new_card.exists():
        final_card = SAT_ROOT / 'SAR_CARD'
        if final_card.exists():
            shutil.rmtree(str(final_card))
        new_card.rename(final_card)
        print(f"  SAR_CARD_new -> SAR_CARD")

    # Delete old source dirs ONLY if new dirs have content
    deletions = [
        ('SAR_CARD_bbox', 'SAR_CARD'),
        ('SAR_SLC_bbox', 'SAR_COH'),
        ('SAR_SLC', 'SAR_METADATA'),
        ('multispectral', 'MS'),
    ]
    for old_name, new_name in deletions:
        old_dir = SAT_ROOT / old_name
        new_dir = SAT_ROOT / new_name
        if old_dir.exists() and new_dir.exists():
            new_files = list(new_dir.rglob('*.tif')) + list(new_dir.rglob('*.json')) + list(new_dir.rglob('*.pkl'))
            if len(new_files) > 0:
                shutil.rmtree(str(old_dir))
                print(f"  Deleted old: {old_name}/ (verified {len(new_files)} files in {new_name}/)")
                stats['deleted_dirs'] += 1
            else:
                print(f"  WARNING: {new_name}/ empty, NOT deleting {old_name}/")
else:
    print("  DRY RUN: would rename SAR_CARD_new -> SAR_CARD")
    print("  DRY RUN: would rename sentinel_1_orbits -> SAR_SLC_ORBIT")
    print("  DRY RUN: would delete: SAR_CARD_bbox, SAR_SLC_bbox, SAR_SLC,")
    print("           SAR_CARD (old empty), SAR_CARD_polygon, SAR_SLC_polygon, multispectral")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("MIGRATION SUMMARY")
print(f"{'='*70}")
print(f"  TIFs moved:       {stats['moved']}")
print(f"  TIFs renamed:     {stats['renamed']} (COH: city + PRE/CROSS/POST label)")
print(f"  .SAFE deleted:    {stats['deleted_safe']}")
print(f"  Old dirs deleted: {stats['deleted_dirs']}")
print(f"  Skipped (exist):  {stats['skipped']}")
print(f"  Errors:           {len(stats['errors'])}")

if stats['errors']:
    print(f"\n  ERRORS:")
    for err in stats['errors'][:20]:
        print(f"    {err}")

if DRY_RUN:
    print(f"\n  DRY RUN — no changes made. Run without --dry-run to execute.")

print(f"\n  Final structure:")
print(f"    satellite/SAR_CARD/{{city}}/{{city}}_CARD_{{pol}}_{{date}}.tif")
print(f"    satellite/SAR_COH/{{city}}/{{city}}_COH_{{pol}}_{{PRE|CROSS|POST}}_{{d1}}_{{d2}}.tif")
print(f"    satellite/MS/{{city}}/{{city}}_S2_{{date}}_{{band}}_{{res}}.tif")
print(f"    satellite/SAR_METADATA/")
print(f"    satellite/SAR_SLC_ORBIT/")
print(f"    satellite/temporal_products/  (unchanged)")
print(f"    satellite/landuse_classification/  (unchanged)")
print(f"{'='*70}")
