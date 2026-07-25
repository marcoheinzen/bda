# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_rasterize.py
Rasterize building footprints + damage labels to TIF masks for fast zonal stats.

Creates per city:
  data_stack/{city}/building_labels.tif   -- int32, 1-based building index, 0=background
  data_stack/{city}/damage_mask.tif       -- int8, -1=excluded, 0=undamaged, 1=damaged, -128=nodata
  data_stack/{city}/building_raster_meta.json -- building_id <-> label mapping + stats

Downstream: stack_features.py reads these TIFs instead of GeoJSON.

Usage:
    from stack_rasterize import rasterize_city_buildings, rasterize_all_cities
    stats = rasterize_all_cities(stack_root, cities, force_rerun=True)
"""

import json
import time
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import Affine
from pathlib import Path
from datetime import datetime


def rasterize_city_buildings(city_name, stack_root, cities_dir=None, force_rerun=False):
    """Rasterize building footprints and damage labels for one city.

    Args:
        city_name:   city name
        stack_root:  output dir (data_stack/{city}/ gets building_labels.tif etc.)
        cities_dir:  if provided, read AOI.geojson + buildings from here (NB03d mode).
                     if None, read from stack_root/{city}/ (legacy NB05 mode).
        force_rerun: overwrite existing

    Returns dict with stats or None on skip/error.
    """
    city_stack = Path(stack_root) / city_name
    city_stack.mkdir(parents=True, exist_ok=True)

    labels_path = city_stack / "building_labels.tif"
    damage_path = city_stack / "damage_mask.tif"
    meta_path = city_stack / "building_raster_meta.json"

    if labels_path.exists() and damage_path.exists() and meta_path.exists() and not force_rerun:
        with open(meta_path) as f:
            existing = json.load(f)
        return existing

    # load reference grid
    if cities_dir:
        # NB03d mode: read UTM grid from AOI.geojson in CITIES_DIR
        aoi_file = Path(cities_dir) / city_name / "AOI.geojson"
        if not aoi_file.exists():
            return None
        with open(aoi_file) as f:
            aoi_gj = json.load(f)
        bbox_props = None
        for feat in aoi_gj['features']:
            if feat.get('properties', {}).get('feature_type') == 'aoi_bbox':
                bbox_props = feat['properties']
                break
        if bbox_props is None:
            return None
        ref = {
            'utm_epsg': int(bbox_props['utm_epsg']),
            'transform': [10.0, 0.0, bbox_props['utm_minx'], 0.0, -10.0, bbox_props['utm_maxy']],
            'height': int(bbox_props['height_px']),
            'width': int(bbox_props['width_px']),
        }
    else:
        # legacy mode: read from data_stack reference_grid.json
        ref_path = city_stack / "reference_grid.json"
        if not ref_path.exists():
            return None
        with open(ref_path) as f:
            ref = json.load(f)

    ref_crs = f"EPSG:{ref['utm_epsg']}"
    ref_transform = Affine(*ref['transform'][:6])
    ref_shape = (ref['height'], ref['width'])

    # load buildings
    src_dir = Path(cities_dir) / city_name if cities_dir else city_stack
    bldg_path = src_dir / f"{city_name}_buildings_overture_with_damage.geojson"
    if not bldg_path.exists():
        bldg_path = src_dir / "buildings_overture_with_damage.geojson"
    if not bldg_path.exists():
        bldg_path = src_dir / f"{city_name}_buildings_overture.geojson"
    if not bldg_path.exists():
        bldg_path = src_dir / "buildings_overture.geojson"
    if not bldg_path.exists():
        return None

    bldg_gdf = gpd.read_file(bldg_path)
    bldg_gdf = bldg_gdf.to_crs(ref_crs)

    # ensure building_id
    if 'building_id' not in bldg_gdf.columns:
        bldg_gdf['building_id'] = [f"{city_name}_{i}" for i in range(len(bldg_gdf))]

    # fix string serialization
    if 'damage_binary' in bldg_gdf.columns:
        bldg_gdf['damage_binary'] = pd.to_numeric(bldg_gdf['damage_binary'], errors='coerce')

    n_buildings = len(bldg_gdf)

    # --- building_labels.tif: each building = unique int label 1..N, background=0 ---
    shapes_labels = []
    for i, geom in enumerate(bldg_gdf.geometry):
        if geom is not None and not geom.is_empty:
            shapes_labels.append((geom, i + 1))

    label_array = rio_rasterize(
        shapes_labels,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype=np.int32,
        all_touched=True,
    )

    # pixel counts per building
    pixel_counts = np.bincount(label_array.ravel(), minlength=n_buildings + 1)[1:]

    # save building_labels.tif
    profile = {
        'driver': 'GTiff',
        'dtype': 'int32',
        'width': ref_shape[1],
        'height': ref_shape[0],
        'count': 1,
        'crs': ref_crs,
        'transform': ref_transform,
        'nodata': 0,
        'compress': 'lzw',
    }
    with rasterio.open(labels_path, 'w', **profile) as dst:
        dst.write(label_array, 1)

    # --- damage_mask.tif: -1=excluded, 0=undamaged, 1=damaged, -128=nodata ---
    # vectorized: build lookup array [0=background, 1..N=buildings], then index
    damage_lookup = np.full(n_buildings + 1, -128, dtype=np.int8)  # index 0 = background

    if 'damage_binary' in bldg_gdf.columns:
        dmg_vals = bldg_gdf['damage_binary'].values
        for i in range(n_buildings):
            if pixel_counts[i] == 0:
                continue
            if pd.notna(dmg_vals[i]):
                damage_lookup[i + 1] = int(dmg_vals[i])  # -1, 0, or 1
            # else stays -128 (no UNOSAT data)

    damage_array = damage_lookup[label_array]  # single vectorized indexing op

    profile_dmg = profile.copy()
    profile_dmg['dtype'] = 'int8'
    profile_dmg['nodata'] = -128
    with rasterio.open(damage_path, 'w', **profile_dmg) as dst:
        dst.write(damage_array, 1)

    # --- building_raster_meta.json ---
    n_with_pixels = int((pixel_counts > 0).sum())
    n_damaged = int((bldg_gdf['damage_binary'] == 1).sum()) if 'damage_binary' in bldg_gdf.columns else 0
    n_undamaged = int((bldg_gdf['damage_binary'] == 0).sum()) if 'damage_binary' in bldg_gdf.columns else 0
    n_excluded = int((bldg_gdf['damage_binary'] == -1).sum()) if 'damage_binary' in bldg_gdf.columns else 0
    n_nolabel = int(bldg_gdf['damage_binary'].isna().sum()) if 'damage_binary' in bldg_gdf.columns else n_buildings

    # building_id to label mapping (for parquet join later)
    id_to_label = {str(bldg_gdf['building_id'].iloc[i]): i + 1 for i in range(n_buildings)}

    meta = {
        'city': city_name,
        'n_buildings': n_buildings,
        'n_with_pixels': n_with_pixels,
        'n_damaged': n_damaged,
        'n_undamaged': n_undamaged,
        'n_excluded': n_excluded,
        'n_nolabel': n_nolabel,
        'ref_shape': list(ref_shape),
        'utm_epsg': ref['utm_epsg'],
        'pixel_counts_min': int(pixel_counts.min()) if len(pixel_counts) > 0 else 0,
        'pixel_counts_max': int(pixel_counts.max()) if len(pixel_counts) > 0 else 0,
        'pixel_counts_median': float(np.median(pixel_counts)) if len(pixel_counts) > 0 else 0,
        'created': datetime.now().isoformat(),
        'id_to_label': id_to_label,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def rasterize_all_cities(stack_root, cities, cities_dir=None, force_rerun=False):
    """Rasterize buildings for all cities. Returns dict of stats per city.

    Args:
        stack_root:  output dir (data_stack)
        cities:      list of city names
        cities_dir:  if provided, read AOI + buildings from here (NB03d mode)
        force_rerun: overwrite existing
    """
    stack_root = Path(stack_root)
    t0 = time.time()

    print("=" * 70)
    print("RASTERIZE BUILDINGS: VECTOR -> RASTER MASKS")
    print("=" * 70)
    print(f"  Cities: {len(cities)}")
    print(f"  Source: {cities_dir or stack_root}")
    print(f"  Output: {stack_root}")
    print(f"  Force rerun: {force_rerun}")

    results = {}
    for ci, city_name in enumerate(sorted(cities)):
        meta = rasterize_city_buildings(city_name, stack_root, cities_dir=cities_dir, force_rerun=force_rerun)
        if meta is None:
            print(f"  [{ci+1}/{len(cities)}] {city_name}: SKIP (no buildings or reference_grid)")
            continue

        results[city_name] = meta
        cached = " (cached)" if not force_rerun and meta.get('created', '') != datetime.now().isoformat()[:10] else ""
        print(f"  [{ci+1}/{len(cities)}] {city_name}: "
              f"{meta['n_buildings']} buildings ({meta['n_with_pixels']} rasterized), "
              f"dmg={meta['n_damaged']} undam={meta['n_undamaged']} excl={meta['n_excluded']}{cached}")

    elapsed = time.time() - t0
    print(f"\n  Done: {len(results)}/{len(cities)} cities ({elapsed:.1f}s)")
    print(f"  Output: building_labels.tif + damage_mask.tif per city")
    print("=" * 70)

    return results
