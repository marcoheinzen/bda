# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_loader.py
Unified data_stack consumer for NB06-NB10.

Single entry point for loading the NB05 dataset: parquet, manifest,
temporal LUT, GroupKFold assignments, and per-city catalogs.

Notebook usage:
    from stack_loader import load_dataset, load_manifest, load_temporal_lut
    from stack_loader import load_groupkfold, get_feature_groups, get_kfold_splitter
    from stack_loader import assign_periods, load_city_catalog

    # Full dataset
    df, meta = load_dataset(STACK_ROOT)

    # Filtered
    df, meta = load_dataset(STACK_ROOT, tiers=[0,1], require_damage=True)

    # With kfold splits
    gkf = load_groupkfold(STACK_ROOT)
    for fold_id, train_idx, test_idx in get_kfold_splitter(df, gkf):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    # Temporal obs indexing
    lut = load_temporal_lut(STACK_ROOT)
    periods = assign_periods(dates=['20220304','20220528'], battle_start='2022-02-24')
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Central column-role filter. Uses is_metadata() pattern rules to drop
# observability (was_observed_*, scenes_observed_*), data-quality
# (qa__cloud_freq, visibility__*__freq, obs_count) and raw-pixel-count
# (block/rolling __count_*) columns from the ML feature set.
# Loaded lazily so older notebooks that don't have metadata_filter.py still
# import stack_loader.py (include_metadata defaults to False; falling back
# to legacy behavior when the module is unavailable).
try:
    from metadata_filter import is_metadata, is_non_feature, select_feature_columns
    _METADATA_FILTER_AVAILABLE = True
except ImportError:
    _METADATA_FILTER_AVAILABLE = False
    def is_metadata(col): return False
    def is_non_feature(col): return False
    def select_feature_columns(df):
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]



# =========================================================================
# MANIFEST + CATALOG
# =========================================================================

def load_manifest(stack_root):
    """Load data_stack_manifest.json.
    Returns dict: {cities: {city_name: {tier, ready_ml, has_card, ...}}, ...}
    """
    stack_root = Path(stack_root)
    manifest_path = stack_root / "data_stack_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}\nRun NB05 VALIDATE first.")
    with open(manifest_path) as f:
        return json.load(f)


def load_city_catalog(stack_root, city_name):
    """Load per-city data_catalog.json from data_stack/{city}/.
    Returns dict with SAR_CARD, SAR_SLC, multispectral_scenes, composites,
    landuse, temporal, readiness, vectors, etc.
    """
    stack_root = Path(stack_root)
    cat_path = stack_root / city_name / "data_catalog.json"
    if not cat_path.exists():
        raise FileNotFoundError(f"Catalog not found: {cat_path}\nRun NB05 VALIDATE first.")
    with open(cat_path) as f:
        return json.load(f)


# =========================================================================
# DATASET (PARQUET)
# =========================================================================

def load_dataset(stack_root, parquet_path=None,
                 tiers=None, cities=None,
                 require_damage=False, require_ml=False,
                 drop_geometry_cols=True,
                 include_metadata=False):
    """Load NB05 building-level parquet dataset.

    Args:
        stack_root:     Path to STACK_ROOT
        parquet_path:   override path (default: STACK_ROOT/dataset/bda_dataset.parquet)
        tiers:          list of tier ints to keep, or None for all
        cities:         list of city names to keep, or None for all
        require_damage: if True, keep only cities with damage labels (n_damage > 0)
        require_ml:     if True, keep only cities flagged ready_ml in manifest
        drop_geometry_cols: drop lon/lat/centroid columns that aren't ML features
        include_metadata: if False (default), drop LEAKY metadata columns
                        (was_observed_*, scenes_observed_*, qa__cloud_freq__*,
                        visibility__*__freq__*, obs_count__*, block __count_*)
                        while keeping essential id/label/grouping columns
                        (building_id, city, tier, damage_binary, date, ...).
                        if True, return the full DataFrame untouched.
                        Set to True for analysis/validation cells that
                        explicitly want to inspect metadata columns (e.g.
                        NB08c C10 leakage delta, NB13 QA reports).

    Returns:
        (df, meta) tuple. meta is dict from bda_dataset_meta.json or {}.
    """
    stack_root = Path(stack_root)
    if parquet_path is None:
        parquet_path = stack_root / "dataset" / "bda_dataset.parquet"
    else:
        parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}\nRun NB05 DELTA cell first.")

    df = pd.read_parquet(parquet_path)

    # --- auto-merge building metadata if missing ---
    # Feature parquets (product_prepost, rolling_stats, block_stats) contain
    # only extracted raster features + building_id + city. Building metadata
    # (damage_binary, damage_label, tier, area_m2, etc.) lives in
    # bda_buildings.parquet. Merge at load time per design spec:
    # "Building metadata separate. Joined at ML time via building_id."
    _bldg_merge_cols = ['damage_binary', 'damage_label', 'damage', 'tier', 'ems98_grade', 'area_m2', 'height', 'num_floors']
    _missing = [c for c in _bldg_merge_cols if c not in df.columns]
    if _missing and 'building_id' in df.columns and 'city' in df.columns:
        _bldg_path = parquet_path.parent / 'bda_buildings.parquet'
        if _bldg_path.exists():
            _df_bldg = pd.read_parquet(_bldg_path)
            _avail = [c for c in _bldg_merge_cols if c in _df_bldg.columns and c not in df.columns]
            if _avail:
                df = df.merge(_df_bldg[['building_id', 'city'] + _avail],
                              on=['building_id', 'city'], how='left')


    # load meta
    meta_path = parquet_path.parent / "bda_dataset_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # city filter
    if cities is not None:
        if isinstance(cities, str):
            cities = [cities]
        df = df[df['city'].isin(cities)].reset_index(drop=True)

    # tier filter
    if tiers is not None and 'tier' in df.columns:
        df = df[df['tier'].isin(tiers)].reset_index(drop=True)

    # damage filter
    if require_damage and 'damage_binary' in df.columns:
        city_has_damage = df.groupby('city')['damage_binary'].apply(lambda s: s.eq(1).any())
        valid_cities = city_has_damage[city_has_damage].index.tolist()
        df = df[df['city'].isin(valid_cities)].reset_index(drop=True)

    # ML readiness filter
    if require_ml:
        manifest = load_manifest(stack_root)
        ml_cities = [c for c, info in manifest.get('cities', {}).items() if info.get('ready_ml', False)]
        df = df[df['city'].isin(ml_cities)].reset_index(drop=True)

    if drop_geometry_cols:
        geo_cols = [c for c in df.columns if c in ('geometry', 'centroid_wkt')]
        if geo_cols:
            df = df.drop(columns=geo_cols, errors='ignore')

    # Default behavior: drop leaky metadata columns (was_observed_*,
    # scenes_observed_*, qa__cloud_freq__*, visibility__*__freq__*,
    # obs_count__*, block/rolling __count_*) that NB03e wrote as TIFs and
    # NB05b surfaced as zonal-stat columns. These are data-presence / data-
    # quality / pixel-count indicators, not damage signal; keeping them
    # during training acts as a city-identifier leakage vector (measured
    # +0.137 AUC inflation on F1/F5 fusion in NB08c C10 diagnostic).
    # Essential id/label/grouping columns (building_id, city, tier, damage_*,
    # date, battle_*, etc.) are preserved via META_COLS_SET whitelist in
    # metadata_filter.py -- they are metadata but not pattern-metadata.
    if not include_metadata and _METADATA_FILTER_AVAILABLE:
        pattern_meta = [c for c in df.columns
                        if is_metadata(c) and c not in ('city', 'building_id', 'tier',
                                                        'date', 'date1', 'date2',
                                                        'timestep', 'period_label',
                                                        'battle_start', 'battle_stop',
                                                        'conflict_ongoing',
                                                        'has_card', 'has_coh', 'has_ms',
                                                        'unosat_id', 'unosat_date', 'unosat_ep',
                                                        'match_method', 'match_distance', 'in_aoi',
                                                        'height', 'num_floors', 'roof_height',
                                                        'area_m2', 'centroid_x', 'centroid_y',
                                                        'n_pixels',
                                                        'pre_date_card', 'post_date_card',
                                                        'dataset')]
        if pattern_meta:
            df = df.drop(columns=pattern_meta, errors='ignore')

    return df, meta


# =========================================================================
# TEMPORAL LUT
# =========================================================================

def load_temporal_lut(stack_root, lut_path=None):
    """Load observation-indexed temporal LUT from NB05.
    Returns dict: {city: {battle_start, modalities: {SAR: [{date, obs_idx, days_from_battle}], MS: [...]}}}
    """
    stack_root = Path(stack_root)
    if lut_path is None:
        lut_path = stack_root / "dataset" / "temporal_lut.json"
    else:
        lut_path = Path(lut_path)

    if not lut_path.exists():
        raise FileNotFoundError(f"Temporal LUT not found: {lut_path}\nRun NB05 TEMPORAL-LUT cell first.")

    with open(lut_path) as f:
        return json.load(f)


def assign_periods(dates, battle_start, battle_stop=None):
    """Assign period labels to a list of date strings (YYYYMMDD or YYYY-MM-DD).

    Args:
        dates:         list of date strings
        battle_start:  str 'YYYY-MM-DD'
        battle_stop:   str 'YYYY-MM-DD' or None

    Returns:
        list of period labels: 'prebattle_baseline', 'prebattle', 'battle',
        'postbattle', 'post_winter_baseline', etc.
    """
    from dateutil.relativedelta import relativedelta

    bs = datetime.strptime(str(battle_start)[:10], '%Y-%m-%d')
    if battle_stop and str(battle_stop).lower() not in ('', 'none', 'nat', 'ongoing'):
        be = datetime.strptime(str(battle_stop)[:10], '%Y-%m-%d')
    else:
        be = None

    # prebattle_baseline: > 30 days before battle_start
    # prebattle: 0-30 days before battle_start
    # battle: between start and stop
    # postbattle: 0-180 days after battle_stop (or battle_start if no stop)
    # post_winter_baseline: > 180 days after battle_stop
    bl_cutoff = bs - relativedelta(days=30)
    post_ref = be if be else bs
    pwb_cutoff = post_ref + relativedelta(days=180)

    periods = []
    for d in dates:
        d_clean = str(d).replace('-', '')[:8]
        dt = datetime.strptime(d_clean, '%Y%m%d')
        if dt < bl_cutoff:
            periods.append('prebattle_baseline')
        elif dt < bs:
            periods.append('prebattle')
        elif be and dt <= be:
            periods.append('battle')
        elif dt <= pwb_cutoff:
            periods.append('postbattle')
        else:
            periods.append('post_winter_baseline')

    return periods


# =========================================================================
# GROUPKFOLD
# =========================================================================

def load_groupkfold(stack_root, gkf_path=None):
    """Load GroupKFold assignments from NB05.
    Returns dict: {fold_mode, n_folds, city_assignments: {city: {fold_id, ...}}, fold_stats, ...}
    """
    stack_root = Path(stack_root)
    if gkf_path is None:
        gkf_path = stack_root / "groupkfold_assignments.json"
    else:
        gkf_path = Path(gkf_path)

    if not gkf_path.exists():
        raise FileNotFoundError(f"GroupKFold not found: {gkf_path}\nRun NB05 GROUPKFOLD cell first.")

    with open(gkf_path) as f:
        return json.load(f)


def get_kfold_splitter(df, gkf, val_fold=0):
    """Yield (fold_id, train_indices, test_indices) for each fold.

    Args:
        df:        DataFrame with 'city' column
        gkf:       dict from load_groupkfold()
        val_fold:  fold_id to reserve as validation (excluded from all train/test).
                   Set to None to use all folds.

    Yields:
        (fold_id, train_idx_array, test_idx_array) for each non-val fold
    """
    assignments = gkf['city_assignments']
    n_folds = gkf['n_folds']

    # map city -> fold_id
    city_fold = {city: info['fold_id'] for city, info in assignments.items()}

    # assign fold_id to each row
    fold_ids = df['city'].map(city_fold)

    # validation mask
    if val_fold is not None:
        val_mask = fold_ids == val_fold
    else:
        val_mask = pd.Series(False, index=df.index)

    non_val = ~val_mask

    for fold_id in range(n_folds):
        if fold_id == val_fold:
            continue
        test_mask = (fold_ids == fold_id) & non_val
        train_mask = (fold_ids != fold_id) & non_val
        if test_mask.sum() == 0:
            continue
        yield fold_id, np.where(train_mask)[0], np.where(test_mask)[0]


def add_split_column(df, gkf, test_fold, val_fold=0):
    """Add 'split' column to df: 'train', 'test', 'val'.
    Convenience for notebooks that expect a single train/test/val split.

    Args:
        df:         DataFrame with 'city' column
        gkf:        dict from load_groupkfold()
        test_fold:  which fold_id is test
        val_fold:   which fold_id is val (default 0)

    Returns:
        df with 'split' column added
    """
    assignments = gkf['city_assignments']
    city_fold = {city: info['fold_id'] for city, info in assignments.items()}
    fold_ids = df['city'].map(city_fold)

    df = df.copy()
    df['split'] = 'train'
    df.loc[fold_ids == test_fold, 'split'] = 'test'
    if val_fold is not None:
        df.loc[fold_ids == val_fold, 'split'] = 'val'
    return df


# =========================================================================
# FEATURE GROUP DETECTION
# =========================================================================

def get_feature_groups(df):
    """Auto-detect feature groups from column name prefixes.

    Returns dict: {group_name: [col_list]}
    Groups: card_baseline, card_assessment, card_blk{NN}, card_blk_pre{NN},
    coh_baseline, comp_prebattle_baseline, comp_post_winter_baseline,
    lu_*, delta_*, cd_*, nbr_*, temp_*, d_baseline, d_assessment, meta
    """
    groups = {}
    # all non-feature columns: identifiers, labels, metadata, Overture building attrs
    meta_cols = {
        # identifiers
        'city', 'building_id', 'id',
        # damage labels (TARGET - never in feature set, leakage!)
        'damage_binary', 'damage_label', 'damage', 'ems98_grade',
        # UNOSAT metadata (traceability, never features)
        'unosat_id', 'unosat_date', 'unosat_ep', 'match_method', 'match_distance', 'in_aoi',
        # spatial coordinates (used for spatial analysis, not ML features)
        'centroid_x', 'centroid_y', 'lon', 'lat',
        # city/conflict metadata
        'tier', 'battle_start', 'battle_stop', 'conflict_ongoing',
        # modality flags
        'has_card', 'has_coh', 'has_ms',
        # observation counts + obs index ranges
        'n_obs_sar_pre', 'n_obs_sar_post', 'n_obs_ms_pre', 'n_obs_ms_post',
        'n_obs_sar_total', 'n_obs_ms_total',
        'obs_idx_sar_min', 'obs_idx_sar_max', 'obs_idx_ms_min', 'obs_idx_ms_max',
        'obs_span_sar_pre_days', 'obs_span_sar_post_days',
        'obs_span_ms_pre_days', 'obs_span_ms_post_days',
        # ML split
        'split', 'fold_id',
        # Overture building attributes (string/categorical - NOT features)
        'names', 'sources', 'is_underground',
        'facade_material', 'subtype', 'class',
        'facade_color', 'roof_material', 'roof_shape', 'roof_color',
        'has_parts', 'version', 'level', 'min_height', 'min_floor',
    }

    # building attributes that ARE numeric features (not in meta)
    # area_m2, height, num_floors, roof_height, n_pixels -> go to 'building_attrs' group
    building_attr_cols = {'area_m2', 'height', 'num_floors', 'roof_height', 'n_pixels'}


    for col in df.columns:
        if col in meta_cols or col.startswith('meta__'):
            groups.setdefault('meta', []).append(col)
            continue
        if col in building_attr_cols:
            groups.setdefault('building_attrs', []).append(col)
            continue

        # === __ convention prefix matching (from stack_align_rename) ===
        if col.startswith('s1__') and '__assessment__' in col:
            groups.setdefault('card_assessment', []).append(col)
        elif col.startswith('s1__') and '__baseline__' in col and 'coh' not in col:
            groups.setdefault('card_baseline', []).append(col)
        elif col.startswith('s1__coh__baseline__'):
            groups.setdefault('coh_baseline', []).append(col)
        elif col.startswith('s1__coh_v') and '__roll' in col:
            m_roll = re.search(r'__roll(\d+)__', col)
            if m_roll:
                groups.setdefault(f'coh_roll{m_roll.group(1)}', []).append(col)
            else:
                groups.setdefault('temporal_other', []).append(col)
        elif col.startswith('s1__coh_v') and '__zscore__' in col:
            groups.setdefault('coh_zscore', []).append(col)
        elif col.startswith('s1__coh_v') and '__post_baseline__' in col:
            groups.setdefault('coh_post_baseline', []).append(col)
        elif col.startswith('s1__') and '__roll' in col:
            m_roll = re.search(r'__roll(\d+)__', col)
            if m_roll:
                groups.setdefault(f'card_roll{m_roll.group(1)}', []).append(col)
            else:
                groups.setdefault('temporal_other', []).append(col)
        elif col.startswith('s1__') and '__blk' in col and 'coh' not in col:
            m_blk = re.search(r'__(blk(?:_pre)?\d+)__', col)
            if m_blk:
                groups.setdefault(f'card_{m_blk.group(1)}', []).append(col)
            else:
                groups.setdefault('card_block_other', []).append(col)
        elif col.startswith('s1__') and col.count('__') == 2 and re.search(r'__\d{8}', col):
            groups.setdefault('card_scenes', []).append(col)
        elif col.startswith('s2__') and '__prebattle_baseline' in col:
            groups.setdefault('comp_prebattle_baseline', []).append(col)
        elif col.startswith('s2__') and '__post_winter_baseline' in col:
            groups.setdefault('comp_post_winter_baseline', []).append(col)
        elif col.startswith('s2__') and '__winter_baseline' in col and '__post_winter' not in col:
            groups.setdefault('comp_winter_baseline', []).append(col)
        elif col.startswith('s2__nbr__'):
            groups.setdefault('scene_nbr', []).append(col)
        elif col.startswith('s2__') and re.search(r'__\d{8}', col):
            groups.setdefault('ms_scenes', []).append(col)
        elif col.startswith('s2__') and '__' in col:
            groups.setdefault('comp_other', []).append(col)
        elif col.startswith('cd__'):
            groups.setdefault('change_detection', []).append(col)
        elif col.startswith('lulc__'):
            groups.setdefault('landuse', []).append(col)
        elif col.startswith('fire__'):
            groups.setdefault('fire', []).append(col)
        elif col.startswith('qa__'):
            groups.setdefault('qa', []).append(col)
        elif col.startswith('d_baseline__') or col.startswith('d_assessment__'):
            if 'baseline' in col:
                groups.setdefault('dietrich_baseline', []).append(col)
            else:
                groups.setdefault('dietrich_assessment', []).append(col)
        elif col.startswith('delta__') or col.startswith('delta_'):
            groups.setdefault('deltas', []).append(col)
        # === OLD prefix matching (fallback for legacy parquets) ===
        elif col.startswith('card_') and 'assessment' in col:
            groups.setdefault('card_assessment', []).append(col)
        elif col.startswith('card_') and ('baseline' in col or 'prebattle' in col):
            groups.setdefault('card_baseline', []).append(col)
        elif col.startswith('card_'):
            groups.setdefault('card_stats', []).append(col)
        elif col.startswith('coh_bl_') or col.startswith('coh_baseline'):
            groups.setdefault('coh_baseline', []).append(col)
        elif col.startswith('comp_prebattle_baseline_'):
            groups.setdefault('comp_prebattle_baseline', []).append(col)
        elif col.startswith('comp_post_winter_baseline_'):
            groups.setdefault('comp_post_winter_baseline', []).append(col)
        elif col.startswith('comp_winter_baseline_'):
            groups.setdefault('comp_winter_baseline', []).append(col)
        elif col.startswith('comp_'):
            # other composite periods
            m = re.match(r'comp_([a-z_]+?)_', col)
            if m:
                groups.setdefault(f'comp_{m.group(1)}', []).append(col)
            else:
                groups.setdefault('comp_other', []).append(col)
        elif col.startswith('delta_'):
            groups.setdefault('deltas', []).append(col)
        elif col.startswith('cd_'):
            groups.setdefault('change_detection', []).append(col)
        elif col.startswith('cloud_freq_'):
            groups.setdefault('cloud_freq', []).append(col)
        elif col.startswith('obs_count_'):
            groups.setdefault('obs_count', []).append(col)
        elif col.startswith('fire_'):
            groups.setdefault('fire', []).append(col)
        elif col.startswith('nbr_'):
            groups.setdefault('scene_nbr', []).append(col)
        elif col.startswith('lu_'):
            m = re.match(r'lu_([a-z_]+?)_', col)
            if m:
                groups.setdefault(f'lu_{m.group(1)}', []).append(col)
            else:
                groups.setdefault('landuse', []).append(col)
        elif col.startswith('idx_'):
            m = re.match(r'idx_([a-z_]+?)_', col)
            if m:
                groups.setdefault(f'idx_{m.group(1)}', []).append(col)
            else:
                groups.setdefault('spectral_indices', []).append(col)
        elif col.startswith('t_CARD_') or col.startswith('t_coh_') or col.startswith('t_COH_'):
            # split temporal products by sensor + product + window size
            col_lower = col.lower()
            m_roll = re.search(r'roll(\d+)', col_lower)
            if 'card' in col_lower and m_roll:
                groups.setdefault(f'card_roll{m_roll.group(1)}', []).append(col)
            elif 'coh' in col_lower and m_roll:
                groups.setdefault(f'coh_roll{m_roll.group(1)}', []).append(col)
            elif 'zscore' in col_lower:
                groups.setdefault('coh_zscore', []).append(col)
            elif 'post_baseline' in col_lower:
                groups.setdefault('coh_post_baseline', []).append(col)
            else:
                groups.setdefault('temporal_other', []).append(col)
        elif col.startswith('d_baseline_'):
            groups.setdefault('dietrich_baseline', []).append(col)
        elif col.startswith('d_assessment_'):
            groups.setdefault('dietrich_assessment', []).append(col)
        else:
            # non-numeric unknowns -> meta (prevents string columns crashing np.var)
            if not pd.api.types.is_numeric_dtype(df[col]):
                groups.setdefault('meta', []).append(col)
            else:
                groups.setdefault('other', []).append(col)

    return groups


def get_ml_features(df, exclude_groups=None, include_metadata=False):
    """Return list of ML feature columns (excludes meta, geometry, identifiers).

    Args:
        df:              DataFrame
        exclude_groups:  list of group names to exclude (e.g. ['dietrich', 'temporal'])
        include_metadata: if False (default), additionally exclude pattern-metadata
                         columns caught by metadata_filter.is_metadata()
                         (was_observed_*, scenes_observed_*, qa__cloud_freq__*,
                         visibility__*__freq__*, obs_count__*, block __count_*).
                         if True, only exclude_groups is applied (legacy behavior).

    Returns:
        list of column names
    """
    groups = get_feature_groups(df)
    if exclude_groups is None:
        exclude_groups = set()
    else:
        exclude_groups = set(exclude_groups)

    exclude_groups.add('meta')
    feature_cols = []
    for gname, cols in groups.items():
        if gname not in exclude_groups:
            feature_cols.extend(cols)

    # default-on pattern filter: drops was_observed_*, scenes_observed_*,
    # qa__cloud_freq__*, visibility__*__freq__*, obs_count__*, block __count_*
    # columns that pass the group-prefix filter but are metadata per
    # metadata_filter.is_metadata()
    if not include_metadata and _METADATA_FILTER_AVAILABLE:
        feature_cols = [c for c in feature_cols if not is_metadata(c)]

    return sorted(feature_cols)


def get_target(df, target_col='damage_binary'):
    """Return target array. Convenience wrapper."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in DataFrame")
    return df[target_col].values
