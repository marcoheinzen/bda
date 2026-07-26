# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
polygon_mask.py
Masks satellite products to city polygon (pixels outside polygon -> NaN).
TIF bounds/shape unchanged (still aoi_bbox). Only pixel values change.
Source: {modality}/{city}/*.tif
Output: {modality}_city_polygon/{city}/*.tif (same filenames)

Notebook usage:
    from polygon_mask import run as run_polygon_mask
    run_polygon_mask(
        modalities=dict(
            COH={'src': SAR_COH_DIR, 'dst': SAR_COH_CITY_POLYGON},
            CARD={'src': SAR_CARD_DIR, 'dst': SAR_CARD_CITY_POLYGON},
            MS={'src': MS_DIR, 'dst': MS_CITY_POLYGON},
        ),
        cities_dir=CITIES_DIR,
        city_selection=CITY_SELECTION,
        enabled=POLYGON_MASK_ENABLED,
        mask_coh=True,
        mask_card=True,
        mask_ms=False,
        force_rerun=FORCE_RERUN,
        skip_existing=SKIP_EXISTING,
        dry_run=DRY_RUN,
    )
"""

import json
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path
from shapely.geometry import shape, mapping


def _load_city_polygons(cities_dir, city_filter=None):
    """Load city_polygon geometry per city from AOI.geojson."""
    cities_dir = Path(cities_dir)
    city_polygons = {}
    for city_dir in sorted(cities_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        if city_filter and city_dir.name not in city_filter:
            continue
        aoi_file = city_dir / "AOI.geojson"
        if not aoi_file.exists():
            continue
        with open(aoi_file) as f:
            gj = json.load(f)
        for feat in gj['features']:
            if feat.get('properties', {}).get('feature_type') == 'city_polygon':
                city_polygons[city_dir.name] = shape(feat['geometry'])
                break
    return city_polygons


def _mask_modality(modality_name, source_dir, output_dir, city_polygons,
                   force_rerun=False, skip_existing=True, dry_run=False):
    """
    Mask all TIFs in source_dir/{city}/ to city polygon.
    Output: output_dir/{city}/ with same filenames.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    print(f"\n{'='*80}")
    print(f"POLYGON MASK: {modality_name}")
    print(f"{'='*80}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")

    total_files = 0
    masked_files = 0
    skipped_files = 0
    errors = []

    for city_dir in sorted(source_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        city_name = city_dir.name
        if city_name in ('metadata', 'temp', 'desktop.ini', 'city_polygon'):
            continue
        if city_name not in city_polygons:
            continue

        poly = city_polygons[city_name]

        tifs = sorted(city_dir.glob("*.tif"))
        if not tifs:
            continue

        print(f"\n  {city_name}: {len(tifs)} TIFs")

        out_city_dir = output_dir / city_name

        for tif_path in tifs:
            total_files += 1

            out_path = out_city_dir / tif_path.name

            if out_path.exists():
                if force_rerun:
                    out_path.unlink()
                elif skip_existing:
                    skipped_files += 1
                    continue
                else:
                    skipped_files += 1
                    continue

            try:
                with rasterio.open(tif_path) as src:
                    data = src.read(1)
                    profile = src.profile.copy()
                    transform = src.transform

                outside_mask = geometry_mask(
                    [mapping(poly)],
                    out_shape=data.shape,
                    transform=transform,
                    invert=False
                )

                currently_valid = ~np.isnan(data) if np.issubdtype(data.dtype, np.floating) else np.ones_like(data, dtype=bool)
                would_mask = outside_mask & currently_valid
                n_mask = int(np.sum(would_mask))

                if not dry_run:
                    masked_data = data.astype(np.float32)
                    masked_data[outside_mask] = np.nan
                    profile.update(dtype='float32', nodata=np.nan)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(out_path, 'w', **profile) as dst:
                        dst.write(masked_data, 1)

                masked_files += 1
                pct = 100 * n_mask / data.size if data.size > 0 else 0
                action = 'WOULD WRITE' if dry_run else 'WROTE'
                print(f"    {action}: {tif_path.name} ({n_mask} px masked, {pct:.1f}%)")

            except Exception as e:
                errors.append((tif_path, str(e)))
                print(f"    ERROR: {tif_path.name}: {e}")

    print(f"\n  {modality_name} summary:")
    print(f"    Total TIFs:    {total_files}")
    print(f"    {'Would write' if dry_run else 'Written'}:     {masked_files}")
    print(f"    Skipped:       {skipped_files}")
    print(f"    Errors:        {len(errors)}")

    return {'total': total_files, 'masked': masked_files, 'skipped': skipped_files, 'errors': len(errors)}


def run(modalities, cities_dir, city_selection=None,
        enabled=False, mask_coh=False, mask_card=False, mask_ms=False,
        force_rerun=False, skip_existing=True, dry_run=False):
    """
    Args:
        modalities:     dict - {'COH': {'src': Path, 'dst': Path}, 'CARD': {...}, 'MS': {...}}
        cities_dir:     Path - CITIES_DIR
        city_selection: list or None
        enabled:        bool (default False - skips cell entirely)
        mask_coh:       bool (default False)
        mask_card:      bool (default False)
        mask_ms:        bool (default False)
        force_rerun:    bool (default False - reprocess even if output exists)
        skip_existing:  bool (default True - skip files that already exist in output)
        dry_run:        bool (default False - report only, no writes)
    """
    if not enabled:
        print("POLYGON MASK: SKIPPED (enabled=False)")
        return {}

    print("=" * 80)
    print("POLYGON MASK: SATELLITE PRODUCTS")
    print("=" * 80)
    print(f"  enabled:       {enabled}")
    print(f"  mask_coh:      {mask_coh}")
    print(f"  mask_card:     {mask_card}")
    print(f"  mask_ms:       {mask_ms}")
    print(f"  force_rerun:   {force_rerun}")
    print(f"  skip_existing: {skip_existing}")
    print(f"  dry_run:       {dry_run}")
    print(f"  Cities:        {city_selection or 'ALL'}")

    active = []
    if mask_coh: active.append('COH')
    if mask_card: active.append('CARD')
    if mask_ms: active.append('MS')

    if not active:
        print("\n  No modalities enabled. Set mask_coh/mask_card/mask_ms=True.")
        return {}

    city_filter = set(city_selection) if city_selection else None
    city_polygons = _load_city_polygons(cities_dir, city_filter)
    print(f"  Loaded polygons for {len(city_polygons)} cities")

    results = {}

    for mod_name in active:
        if mod_name not in modalities:
            print(f"\n  WARNING: {mod_name} not in modalities dict, skipping")
            continue
        mod_cfg = modalities[mod_name]
        results[mod_name] = _mask_modality(
            mod_name, mod_cfg['src'], mod_cfg['dst'], city_polygons,
            force_rerun=force_rerun, skip_existing=skip_existing, dry_run=dry_run
        )

    print(f"\n{'='*80}")
    print(f"POLYGON MASK COMPLETE")
    print(f"{'='*80}")
    for mod_name, stats in results.items():
        action = 'would write' if dry_run else 'written'
        print(f"  {mod_name}: {stats['masked']} {action}, {stats['skipped']} skipped, {stats['errors']} errors")
    if dry_run:
        total_would = sum(s['masked'] for s in results.values())
        if total_would > 0:
            print(f"\n  DRY RUN - set dry_run=False to write {total_would} polygon-masked copies")

    return results
