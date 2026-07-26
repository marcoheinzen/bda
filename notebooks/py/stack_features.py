# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_features.py
Feature extraction pipeline for building-level BDA dataset.

Functions:
  - build_temporal_lut()     observation-indexed timeblocks per city per modality
  - discover_rasters()       find all rasters in data_stack per city
  - extract_city_features()  full feature extraction for one city (uses rasterized masks)
  - compute_deltas()         pre-post deltas + merge + save parquet

Requires: stack_rasterize.py must have created building_labels.tif + damage_mask.tif first.

Notebook usage:
    from stack_features import build_temporal_lut, extract_city_features, compute_deltas
"""

import re
import json
import time
import gc
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from datetime import datetime


# =========================================================================
# GeoJSON COLUMN WHITELIST (everything else dropped at load time)
# Prevents Overture string columns from leaking into parquet
# =========================================================================

# columns to KEEP from buildings_overture_with_damage.geojson
GEOJSON_KEEP_COLS = {
    # labels (TARGET - never in ML feature set!)
    'damage', 'damage_label', 'damage_binary', 'ems98_grade',
    # label traceability (never in ML feature set)
    'unosat_id', 'unosat_date', 'unosat_ep',
    'match_method', 'match_distance', 'in_aoi',
    # building attributes (potential features)
    'height', 'num_floors', 'roof_height',
    # geometry (dropped before parquet save)
    'geometry',
}

# columns to cast to numeric (prevent str-as-float in parquet)
NUMERIC_CAST_COLS = [
    'damage_binary', 'damage_label', 'damage', 'ems98_grade',
    'height', 'num_floors', 'roof_height', 'match_distance', 'in_aoi',
]


# =========================================================================
# TEMPORAL LUT
# =========================================================================

def build_temporal_lut(stack_root, city_meta, lut_path):
    """Build observation index per city per modality.
    obs_idx: negative = pre-battle, positive = post-battle.
    Date stored as metadata, NOT as ML feature.

    Args:
        stack_root:  Path to STACK_ROOT
        city_meta:   dict {city_name: {'battle_start': str, ...}}
        lut_path:    Path to save temporal_lut.json

    Returns:
        temporal_lut dict
    """
    stack_root = Path(stack_root)
    lut_path = Path(lut_path)

    print("=" * 70)
    print("TEMPORAL-LUT: OBSERVATION INDEX PER CITY PER MODALITY")
    print("=" * 70)

    temporal_lut = {}

    for city_name, meta in city_meta.items():
        battle_start_str = meta.get('battle_start', '')
        if not battle_start_str:
            continue
        battle_dt = datetime.strptime(str(battle_start_str)[:10], '%Y-%m-%d')
        city_stack = stack_root / city_name
        city_lut = {'battle_start': str(battle_start_str)[:10], 'modalities': {}}

        # SAR dates (flat dirs only - skip derived products)
        sar_dates = set()
        for subdir in ['SAR_CARD/flat', 'SAR_SLC/flat']:
            d = city_stack / subdir
            if d.exists():
                for f in d.glob("*.tif"):
                    # findall: COH files have 2 dates, CARD files have 1
                    for ds in re.findall(r'(\d{8})', f.name):
                        sar_dates.add(ds)

        if sar_dates:
            sar_obs = []
            for ds in sorted(sar_dates):
                obs_dt = datetime.strptime(ds, '%Y%m%d')
                sar_obs.append({'date': ds, 'days_from_battle': (obs_dt - battle_dt).days})
            pre_obs = [o for o in sar_obs if o['days_from_battle'] < 0]
            post_obs = [o for o in sar_obs if o['days_from_battle'] >= 0]
            for i, o in enumerate(pre_obs):
                o['obs_idx'] = -(len(pre_obs) - i)
            for i, o in enumerate(post_obs):
                o['obs_idx'] = i + 1
            city_lut['modalities']['SAR'] = pre_obs + post_obs

        # MS dates (flat dir only - skip composites/rgb/nbr)
        ms_dates = set()
        ms_flat_dir = city_stack / "multispectral" / "flat"
        if ms_flat_dir.exists():
            for f in ms_flat_dir.glob("*.tif"):
                m = re.search(r'[_](\d{8})[_.]', f.name) or re.search(r'__(\d{8})\.', f.name)
                if m:
                    ms_dates.add(m.group(1))

        if ms_dates:
            ms_obs = []
            for ds in sorted(ms_dates):
                obs_dt = datetime.strptime(ds, '%Y%m%d')
                ms_obs.append({'date': ds, 'days_from_battle': (obs_dt - battle_dt).days})
            pre_obs = [o for o in ms_obs if o['days_from_battle'] < 0]
            post_obs = [o for o in ms_obs if o['days_from_battle'] >= 0]
            for i, o in enumerate(pre_obs):
                o['obs_idx'] = -(len(pre_obs) - i)
            for i, o in enumerate(post_obs):
                o['obs_idx'] = i + 1
            city_lut['modalities']['MS'] = pre_obs + post_obs

        temporal_lut[city_name] = city_lut

        n_sar = len(city_lut['modalities'].get('SAR', []))
        n_ms = len(city_lut['modalities'].get('MS', []))
        print(f"  {city_name:<22s} SAR: {n_sar}  MS: {n_ms}")

    with open(lut_path, 'w') as f:
        json.dump(temporal_lut, f, indent=2, default=str)
    print(f"\n  Saved: {lut_path}")

    return temporal_lut


# =========================================================================
# RASTER DISCOVERY
# =========================================================================

def discover_rasters(city_stack_dir):
    """Find all rasters in data_stack for a city, organized by group.
    Returns dict: {group_name: [(path, feature_prefix), ...]}
    """
    city_stack_dir = Path(city_stack_dir)
    city_name = city_stack_dir.name
    groups = {}

    def clean_stem(stem):
        """Strip city name prefix from TIF filename for cross-city column compatibility.
        NB03d creates TIFs as City_product_pol_stat.tif — stripping city ensures
        Mariupol and Lysychansk get identical column names after pd.concat().
        """
        if stem.startswith(f"{city_name}_"):
            return stem[len(city_name) + 1:]
        return stem

    # SAR CARD: temporal stats
    ts_dir = city_stack_dir / "SAR_CARD" / "temporal_stats"
    if ts_dir.exists():
        groups["card_stats"] = [(f, clean_stem(f.stem)) for f in sorted(ts_dir.glob("*.tif"))]

    # SAR SLC: coherence baseline
    bl_dir = city_stack_dir / "SAR_SLC" / "coherence_baseline"
    if bl_dir.exists():
        groups["coh_baseline"] = [(f, clean_stem(f.stem)) for f in sorted(bl_dir.glob("*.tif"))]

    # Composites (period subdirs + root-level dNBR/RBR + cloud_freq + obs_count)
    comp_dir = city_stack_dir / "multispectral" / "composites"
    if comp_dir.exists():
        for period_dir in sorted(comp_dir.iterdir()):
            if period_dir.is_dir():
                period = period_dir.name
                # composite band TIFs
                comp_tifs = [f for f in sorted(period_dir.glob("*.tif"))
                             if f.stem.startswith(('composite_', 's2__')) and not f.stem.startswith('qa__')]
                if comp_tifs:
                    def _comp_pfx(stem, per):
                        s = clean_stem(stem)
                        if '__' in s:
                            return f"{s}__{per}"
                        return f"comp_{per}_{s.replace('composite_','')}"
                    groups[f"comp_{period}"] = [(f, _comp_pfx(f.stem, period)) for f in comp_tifs]
                # cloud frequency (NB03d P5)
                cf = period_dir / 'qa__cloud_freq.tif'
                if not cf.exists():
                    cf = period_dir / 'cloud_frequency.tif'
                if cf.exists():
                    groups.setdefault("cloud_freq", []).append((cf, f"cloud_freq_{period}"))
                # observation count (NB03d P2)
                oc = period_dir / 'qa__obs_count.tif'
                if not oc.exists():
                    oc = period_dir / 'obs_count.tif'
                if oc.exists():
                    groups.setdefault("obs_count", []).append((oc, f"obs_count_{period}"))

        # dNBR + RBR change maps at composites root
        change_tifs = sorted(comp_dir.glob("dNBR_*.tif")) + sorted(comp_dir.glob("RBR_*.tif")) + sorted(comp_dir.glob("cd__*.tif"))
        if change_tifs:
            def _cd_pfx(stem):
                s = clean_stem(stem)
                if s.startswith('cd__'):
                    return s
                return f"cd_{s}"
            groups["change_detection"] = [(f, _cd_pfx(f.stem)) for f in change_tifs]

    # Per-scene NBR
    nbr_dir = city_stack_dir / "multispectral" / "nbr"
    if nbr_dir.exists():
        nbr_tifs = sorted(nbr_dir.glob("*.tif"))
        if nbr_tifs:
            def _nbr_pfx(stem):
                s = clean_stem(stem)
                if s.startswith('s2__'):
                    return s
                return f"nbr_{s.replace('NBR_', '')}"
            groups["scene_nbr"] = [(f, _nbr_pfx(f.stem)) for f in nbr_tifs]

    # Landuse: classification files (categorical, mode extraction)
    # and spectral indices (continuous, mean/std extraction) - separate groups
    lu_dir = city_stack_dir / "landuse"
    if lu_dir.exists():
        for period_dir in sorted(lu_dir.iterdir()):
            if period_dir.is_dir():
                period = period_dir.name
                # classification TIFs only (is_landuse=True in extract)
                lu_class_tifs = sorted(period_dir.rglob("landuse_classification.tif")) + sorted(period_dir.rglob("lulc__class.tif"))
                if lu_class_tifs:
                    def _lu_pfx(stem, per):
                        s = clean_stem(stem)
                        if s.startswith('lulc__'):
                            return f"{s}__{per}"
                        return f"lu_{per}_{s}"
                    groups[f"lu_{period}"] = [(f, _lu_pfx(f.stem, period)) for f in lu_class_tifs]
                # spectral index TIFs in indices/ subdirs (continuous, is_landuse=False)
                idx_tifs = []
                for date_dir in sorted(period_dir.iterdir()):
                    idx_dir = date_dir / "indices" if date_dir.is_dir() else None
                    if idx_dir and idx_dir.exists():
                        idx_tifs.extend(sorted(idx_dir.glob("*.tif")))
                if idx_tifs:
                    def _idx_pfx(stem, per):
                        s = clean_stem(stem)
                        if s.startswith('s2__'):
                            return f"{s}__{per}"
                        return f"idx_{per}_{s}"
                    groups[f"idx_{period}"] = [(f, _idx_pfx(f.stem, period)) for f in idx_tifs]
                # fire detection TIFs (binary per-scene)
                fire_tifs = []
                for date_dir in sorted(period_dir.iterdir()):
                    if not date_dir.is_dir():
                        continue
                    for fire_name in ['active_fire.tif', 'burn_scar.tif', 'fire__active.tif', 'fire__burn_scar.tif']:
                        fp = date_dir / fire_name
                        if fp.exists():
                            fire_tifs.append((fp, f"fire_{period}_{date_dir.name}_{fp.stem}"))
                if fire_tifs:
                    groups.setdefault("fire", []).extend(fire_tifs)

    # Temporal products
    temp_dir = city_stack_dir / "temporal"
    if temp_dir.exists():
        for sensor_dir in sorted(temp_dir.iterdir()):
            if sensor_dir.is_dir():
                for product_dir in sorted(sensor_dir.iterdir()):
                    if product_dir.is_dir():
                        tifs = sorted(product_dir.glob("*.tif"))
                        if tifs:
                            gname = f"temp_{sensor_dir.name}_{product_dir.name}"
                            def _tmp_pfx(stem):
                                s = clean_stem(stem)
                                if '__' in s:
                                    return s
                                return f"t_{s}"
                            groups[gname] = [(f, _tmp_pfx(f.stem)) for f in tifs]

    # Dietrich features
    for diet_name in ['Dietrich_baseline', 'Dietrich_assessment']:
        diet_dir = city_stack_dir / diet_name
        if diet_dir.exists():
            tifs = sorted(diet_dir.glob("*.tif"))
            if tifs:
                def _diet_pfx(stem, dn):
                    s = clean_stem(stem)
                    if '__' in s:
                        label = 'baseline' if 'baseline' in dn.lower() else 'assessment'
                        return f"d_{label}__{s}"
                    return f"d_{dn.lower().replace('dietrich_','')}_{s}"
                groups[f"d_{diet_name.lower()}"] = [(f, _diet_pfx(f.stem, diet_name)) for f in tifs]

    return groups


# =========================================================================
# ZONAL STATS -- PURE NUMPY (reads rasterized building_labels.tif)
# =========================================================================

def extract_zonal_stats_from_labels(label_array, raster_data, feature_prefix,
                                     n_buildings, is_landuse=False, min_pixels=3):
    """Extract per-building stats using pre-loaded label array + raster.
    Returns dict of {col_name: numpy_array}.

    Non-landuse: {prefix}_mean, {prefix}_std, {prefix}_max
    Landuse:     {prefix}_mode
    """
    results = {}

    valid_mask = np.isfinite(raster_data)
    data_flat = raster_data.ravel()
    label_flat = label_array.ravel()
    valid_flat = valid_mask.ravel()

    if is_landuse:
        col = f"{feature_prefix}_mode"
        modes = np.full(n_buildings, np.nan)

        # vectorized mode: bincount per (building, class) pair
        use = valid_flat & (label_flat > 0)
        if np.any(use):
            class_vals = data_flat[use].astype(np.int32)
            labels_used = label_flat[use]
            n_classes = int(class_vals.max()) + 1 if len(class_vals) > 0 else 1

            # combined key: label_id * n_classes + class_value
            keys = labels_used * n_classes + class_vals
            counts = np.bincount(keys, minlength=(n_buildings + 1) * n_classes)
            counts_2d = counts.reshape(n_buildings + 1, n_classes)  # [0] = background, [1:] = buildings
            bldg_counts = counts_2d[1:]  # (n_buildings, n_classes)

            # total valid pixels per building
            total_per_bldg = bldg_counts.sum(axis=1)
            good = total_per_bldg >= min_pixels
            if good.any():
                modes[good] = bldg_counts[good].argmax(axis=1).astype(np.float64)

        results[col] = modes
    else:
        from scipy.ndimage import maximum as ndimage_maximum

        col_mean = f"{feature_prefix}_mean"
        col_std = f"{feature_prefix}_std"
        col_max = f"{feature_prefix}_max"
        means = np.full(n_buildings, np.nan)
        stds = np.full(n_buildings, np.nan)
        maxs = np.full(n_buildings, np.nan)

        data_zeroed = np.where(valid_flat, data_flat, 0.0)
        sums = np.bincount(label_flat, weights=data_zeroed, minlength=n_buildings + 1)[1:]
        valid_counts = np.bincount(label_flat, weights=valid_flat.astype(float), minlength=n_buildings + 1)[1:]

        good = valid_counts >= min_pixels
        means[good] = sums[good] / valid_counts[good]

        # max via scipy.ndimage.maximum (O(N) single pass over label array)
        data_for_max = np.where(valid_mask, raster_data, -np.inf)
        label_ids = np.arange(1, n_buildings + 1)
        if good.any():
            raw_max = ndimage_maximum(data_for_max, labels=label_array, index=label_ids)
            raw_max = np.asarray(raw_max, dtype=np.float64)
            raw_max[~good] = np.nan
            raw_max[np.isinf(raw_max)] = np.nan
            maxs = raw_max

        if good.any():
            label_means = np.zeros(n_buildings + 1, dtype=np.float64)
            label_means[1:] = np.where(good, means, 0.0)
            pixel_means = label_means[label_flat]
            sq_diff = np.where(valid_flat & (label_flat > 0), (data_flat - pixel_means) ** 2, 0.0)
            sum_sq = np.bincount(label_flat, weights=sq_diff, minlength=n_buildings + 1)[1:]
            stds[good] = np.sqrt(sum_sq[good] / valid_counts[good])

        results[col_mean] = means
        results[col_std] = stds
        results[col_max] = maxs

    return results


# =========================================================================
# CITY-LEVEL FEATURE EXTRACTION
# =========================================================================

def extract_city_features(city_name, stack_root, min_pixels=3):
    """Extract all features for one city using rasterized building masks.

    Requires building_labels.tif + building_raster_meta.json from stack_rasterize.py.
    Returns DataFrame or None.
    """
    city_stack_dir = Path(stack_root) / city_name

    # load rasterized building labels (created by stack_rasterize.py)
    labels_path = city_stack_dir / "building_labels.tif"
    meta_path = city_stack_dir / "building_raster_meta.json"

    if not labels_path.exists() or not meta_path.exists():
        print(f"    ERROR: building_labels.tif or building_raster_meta.json missing. Run RASTERIZE cell first.")
        return None

    t0 = time.time()
    with rasterio.open(labels_path) as src:
        label_array = src.read(1)
        ref_shape = (src.height, src.width)

    with open(meta_path) as f:
        bldg_meta = json.load(f)

    n_buildings = bldg_meta['n_buildings']
    n_with_pixels = bldg_meta['n_with_pixels']
    print(f"    Loaded building_labels.tif: {n_buildings} buildings ({n_with_pixels} rasterized) in {time.time()-t0:.1f}s")

    # load buildings GeoJSON — WHITELIST columns to prevent string pollution
    bldg_path = city_stack_dir / "buildings_overture_with_damage.geojson"
    if not bldg_path.exists():
        bldg_path = city_stack_dir / "buildings_overture.geojson"
    if not bldg_path.exists():
        return None

    import geopandas as gpd
    bldg_gdf_raw = gpd.read_file(bldg_path)

    # keep only whitelisted columns
    keep = [c for c in GEOJSON_KEEP_COLS if c in bldg_gdf_raw.columns]
    bldg_gdf = bldg_gdf_raw[keep].copy()

    ref_path = city_stack_dir / "reference_grid.json"
    with open(ref_path) as f:
        ref = json.load(f)
    bldg_gdf = bldg_gdf.to_crs(f"EPSG:{ref['utm_epsg']}")

    # add computed columns
    bldg_gdf['city'] = city_name
    bldg_gdf['building_id'] = [f"{city_name}_{i}" for i in range(len(bldg_gdf))]
    bldg_gdf['area_m2'] = bldg_gdf.geometry.area
    bldg_gdf['centroid_x'] = bldg_gdf.geometry.centroid.x
    bldg_gdf['centroid_y'] = bldg_gdf.geometry.centroid.y

    # cast all numeric columns properly (prevent str-as-object in parquet)
    for col in NUMERIC_CAST_COLS:
        if col in bldg_gdf.columns:
            bldg_gdf[col] = pd.to_numeric(bldg_gdf[col], errors='coerce')

    # pixel count per building from label array
    pixel_counts = np.bincount(label_array.ravel(), minlength=n_buildings + 1)[1:]
    bldg_gdf['n_pixels'] = pixel_counts

    # discover rasters
    raster_groups = discover_rasters(city_stack_dir)
    total_rasters = sum(len(v) for v in raster_groups.values())
    print(f"    Rasters: {total_rasters} across {len(raster_groups)} groups")

    # extract features from each raster
    raster_count = 0
    for gname, raster_list in raster_groups.items():
        is_lu = gname.startswith('lu_')
        for raster_path, prefix in raster_list:
            try:
                with rasterio.open(raster_path) as src:
                    raster_data = src.read(1).astype(np.float32)
                    # handle shape mismatch
                    if raster_data.shape != ref_shape:
                        padded = np.full(ref_shape, np.nan, dtype=np.float32)
                        h = min(raster_data.shape[0], ref_shape[0])
                        w = min(raster_data.shape[1], ref_shape[1])
                        padded[:h, :w] = raster_data[:h, :w]
                        raster_data = padded
                    # replace nodata
                    nd = src.nodata
                    if nd is not None:
                        try:
                            nd_f = float(nd)
                            if not np.isnan(nd_f):
                                raster_data[raster_data == nd_f] = np.nan
                        except (ValueError, TypeError):
                            pass

                stats = extract_zonal_stats_from_labels(
                    label_array, raster_data, prefix, n_buildings,
                    is_landuse=is_lu, min_pixels=min_pixels,
                )
                for col_name, values in stats.items():
                    bldg_gdf[col_name] = values
                raster_count += 1
            except Exception as e:
                print(f"    WARN: {raster_path.name}: {e}")

    print(f"    Extracted {raster_count} rasters -> {len(bldg_gdf.columns)} columns ({time.time()-t0:.1f}s)")
    gc.collect()
    return bldg_gdf


# =========================================================================
# DELTA COMPUTATION + PARQUET SAVE
# =========================================================================

def compute_deltas(all_city_dfs, parquet_path, meta_path=None):
    """Compute pre-post deltas and save merged parquet.

    For each feature pair comp_prebattle_baseline_X_mean / comp_post_winter_baseline_X_mean,
    computes delta_X = post - pre.

    Args:
        all_city_dfs: list of DataFrames from extract_city_features
        parquet_path: output path for parquet
        meta_path: optional path for metadata json
    """
    if not all_city_dfs:
        print("  No city dataframes to merge")
        return None

    print("=" * 70)
    print("DELTA: COMPUTE DELTAS + MERGE + SAVE PARQUET")
    print("=" * 70)

    t0 = time.time()

    # drop geometry for parquet
    dfs = []
    for df in all_city_dfs:
        df_flat = df.drop(columns=['geometry'], errors='ignore').copy()
        dfs.append(df_flat)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Merged: {len(merged)} buildings, {len(merged.columns)} columns")

    # compute deltas for paired pre/post columns (_mean and _max)
    delta_count = 0
    for suffix in ['_mean', '_max']:
        pre_cols = [c for c in merged.columns if 'prebattle_baseline' in c and c.endswith(suffix)]
        for pre_col in pre_cols:
            # find matching post column
            post_col = pre_col.replace('prebattle_baseline', 'post_winter_baseline')
            if post_col not in merged.columns:
                post_col = pre_col.replace('prebattle_baseline', 'postbattle')
            if post_col in merged.columns:
                delta_name = pre_col.replace('comp_prebattle_baseline_', 'delta_').replace(suffix, '')
                if suffix == '_max':
                    delta_name = delta_name + '_max'
                if delta_name not in merged.columns:
                    merged[delta_name] = merged[post_col] - merged[pre_col]
                    delta_count += 1

    # winter baseline deltas
    for suffix in ['_mean', '_max']:
        winter_cols = [c for c in merged.columns if 'winter_baseline' in c and 'post_winter' not in c and c.endswith(suffix)]
        for win_col in winter_cols:
            post_col = win_col.replace('winter_baseline', 'post_winter_baseline')
            if post_col in merged.columns:
                delta_name = win_col.replace('comp_winter_baseline_', 'delta_winter_').replace(suffix, '')
                if suffix == '_max':
                    delta_name = delta_name + '_max'
                if delta_name not in merged.columns:
                    merged[delta_name] = merged[post_col] - merged[win_col]
                    delta_count += 1

    # index deltas (idx_prebattle vs idx_postbattle)
    for suffix in ['_mean', '_max']:
        idx_pre_cols = [c for c in merged.columns if c.startswith('idx_prebattle_') and c.endswith(suffix)]
        for pre_col in idx_pre_cols:
            post_col = pre_col.replace('idx_prebattle_', 'idx_postbattle_')
            if post_col in merged.columns:
                delta_name = pre_col.replace('idx_prebattle_', 'delta_idx_').replace(suffix, '')
                if suffix == '_max':
                    delta_name = delta_name + '_max'
                if delta_name not in merged.columns:
                    merged[delta_name] = merged[post_col] - merged[pre_col]
                    delta_count += 1

    print(f"  Deltas computed: {delta_count}")

    # save
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(parquet_path, index=False)
    print(f"  Saved: {parquet_path} ({len(merged)} rows, {len(merged.columns)} cols)")

    if meta_path:
        meta = {
            'n_buildings': len(merged),
            'n_cities': len(all_city_dfs),
            'cities': sorted(merged['city'].unique().tolist()) if 'city' in merged.columns else [],
            'n_features': len(merged.columns),
            'n_deltas': delta_count,
            'damage_distribution': merged['damage_binary'].value_counts().to_dict() if 'damage_binary' in merged.columns else {},
            'created': datetime.now().isoformat(),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Meta: {meta_path}")

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")
    print("=" * 70)

    return merged
