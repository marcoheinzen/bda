# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
metadata_filter.py -- single source of truth for column-role classification.

bda -- Building Damage Assessment using satellite imagery
Copyright (C) 2024-2026 Marco Heinzen
SPDX-License-Identifier: AGPL-3.0-or-later

Purpose
-------
Every notebook in the BDA pipeline needs to decide which columns in a parquet
are FEATURES (satellite signal, used for training) vs METADATA (row descriptors,
never used as features) vs LABEL (training target).

Classification rule (end-to-end, from disk to parquet)
------------------------------------------------------
LABEL:
  - damage_binary, damage_label, damage, ems98_grade

FEATURES (damage signal):
  Every column derived via stack_features.extract_zonal_stats_from_labels()
  from a PRIMARY (NB03a/b/c) or DERIVED (NB03e) product TIF on data_stack.
  Column names always end with one of _mean, _std, _max, _mode.

  Primary products (NB03a/b/c):
    * CARD scenes: s1__{vv|vh}__*
    * COH pairs  : s1__coh_{vv|vh}__*
    * MS bands   : s2__{b02..b12, b8a, scl, cloud_mask, visibility}__*

  Derived products (NB03e):
    * Composites : s2__composite__{band|index}__*
    * Indices    : s2__{ndvi|bsi|savi|mndwi|ndbi|ndsi|ibi|baei|ui|nbr}__*
    * Block stats: s1__{vv|vh|coh}__{baseline|post_baseline|blk*|blk_pre*}__{stat}_*
    * Rolling    : s1__*__roll{3|7|13}__{assessment}__{stat}_*
    * Drop accum : s1__{coh|vv}__{max_drop|max_z_drop|running_min|drop_count|date_*_drop}_*
    * MS accum   : s2__{swir_*|mahalanobis_*|nbr_anomaly_count|dnbr|rbr|date_first_*|date_worst_*}_*
    * Landuse chg: s2__lu__{loss_count|loss_fraction|urban_retained|date_*_loss|final_class|modal_*}_*
    * Pre/post   : pre_vv_*, post_vv_*, delta_vv_*, pre_vh_*, post_vh_*, delta_vh_*
    * Simple     : landuse__*_mode, landuse_changed, scene_s2__*, s2__landuse_mode

METADATA (never features), even when prefixed s1__/s2__:
  - Row descriptors enumerated in META_COLS_SET (city, date, battle timing, ...)
  - All was_observed_* flags (pattern rule).
  - Observability rasters from NB03e (data-presence, not damage):
      * s1__{vv|vh|coh}__scenes_observed_* -- per-modality scene count
      * s2__scenes_observed_*              -- MS scene count
      * s2__lu__scenes_observed_*          -- landuse scene count
      * s2__obs_count__*                   -- observation count per composite period
  - Data-quality rasters from NB03e (scene-fraction metrics):
      * s2__qa__cloud_freq__*              -- fraction of cloudy scenes
      * s2__visibility__{cloud|fire|smoke|clear}__freq__* -- SCL-derived quality
  - Raw pixel-count rasters from NB03e block/rolling/baseline processing:
      * s1__{pol|coh}__{baseline|post_baseline|blk*|roll*__assessment}__count_*
    These are pixel counts inside the building footprint (dependent on bldg size,
    not damage state).

  Rationale: NB08c C10 diagnostic confirmed was_observed_* inflates F1 AUC by
  +0.137 and F5 by +0.136 by acting as city-identifier. Analogous logic applies
  to scene_observed / obs_count / qa__cloud_freq / visibility__*__freq / count_*:
  they systematically differ across cities and induce the same leakage vector.

Usage
-----
    from metadata_filter import (
        ID_COLS, LABEL_COLS, META_COLS_SET,
        is_metadata, is_non_feature, select_feature_columns,
        build_role_overrides, classify_manifest_features,
    )

    feat_cols = select_feature_columns(df)
    role_map  = build_role_overrides(df.columns)
    assert all(not is_metadata(c) for c in feat_cols)
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping


# ---------------------------------------------------------------------------
# Column role sets
# ---------------------------------------------------------------------------

ID_COLS: frozenset[str] = frozenset({
    'building_id',          # V2 sample-unit identifier (Overture footprint)
    'point_id',             # V3-V7 sample-unit identifier (UNOSAT point or sampled centroid)
})

LABEL_COLS: frozenset[str] = frozenset({
    'damage_binary', 'damage_label', 'damage', 'ems98_grade',
    # V3-V7 label proxies from bda_points_t{tier}.parquet:
    'ep',                   # UNOSAT EMS-98 grade as numeric (1-6); damage_binary derived from this
    't_unosat',             # UNOSAT survey date as int32 YYYYMMDD; -1 for sampled negatives = perfect target proxy
})

META_COLS_SET: frozenset[str] = frozenset({
    # geographic / admin
    'city', 'tier',
    # battle timing
    'battle_start', 'battle_stop', 'conflict_ongoing',
    # sensor availability hints (building-level, set in NB05b from manifest)
    'has_card', 'has_coh', 'has_ms',
    # UNOSAT linkage
    'unosat_id', 'unosat_date', 'unosat_ep',
    'match_method', 'match_distance', 'in_aoi',
    # V3-V7 per-point identity and location columns (from bda_points):
    'point_source',         # unosat_pos / unosat_undamaged / sampled_neg (encodes label)
    'lon', 'lat',           # WGS84 coordinates (encode city identity perfectly)
    'x_utm', 'y_utm',      # UTM coordinates in per-city CRS (encode city identity)
    'row', 'col',           # AOI raster pixel indices (encode city + spatial location)
    # Overture building attributes (descriptive, not damage signal)
    'height', 'num_floors', 'roof_height',
    # geometry-derived (descriptive)
    'area_m2', 'centroid_x', 'centroid_y', 'n_pixels',
    # temporal axis for long-format parquets
    'date', 'date1', 'date2', 'timestep', 'period_label',
    # pre/post date metadata (CARD per-pair)
    'pre_date_card', 'post_date_card',
    # dataset-origin tag
    'dataset',
})


# ---------------------------------------------------------------------------
# Pattern-based metadata detection
# ---------------------------------------------------------------------------
#
# Every column in a wide BDA parquet has the shape
#     {feature_prefix}_{mean|std|max|mode}
# produced by stack_features.extract_zonal_stats_from_labels(). The zonal-stat
# suffix is NOT a separator on a word boundary because underscore is a word
# character; patterns below include the explicit suffix to anchor correctly.

_STAT_SUFFIX = r'(?:_(?:mean|std|max|mode))'


_METADATA_PATTERNS: tuple[re.Pattern, ...] = (
    # was_observed_* flags (every wide parquet has >=1)
    re.compile(r'^was_observed_'),

    # observability: per-modality scene-count rasters (NB03e)
    re.compile(r'^s1__(?:vv|vh|coh)__scenes_observed' + _STAT_SUFFIX + r'$'),
    re.compile(r'^s2__scenes_observed' + _STAT_SUFFIX + r'$'),
    re.compile(r'^s2__lu__scenes_observed' + _STAT_SUFFIX + r'$'),

    # observation count per composite period (NB03e P2: qa__obs_count.tif)
    re.compile(r'^s2__obs_count__[a-z_]+' + _STAT_SUFFIX + r'$'),

    # cloud / visibility frequency maps (NB03e P5: qa__cloud_freq.tif, visibility__*__freq.tif)
    re.compile(r'^s2__qa__cloud_freq__[a-z_]+' + _STAT_SUFFIX + r'$'),
    re.compile(r'^s2__visibility__(?:cloud|fire|smoke|clear)__freq__[a-z_]+' + _STAT_SUFFIX + r'$'),

    # raw pixel counts from block/rolling/baseline processing (NB03e block_stats and rolling_stats).
    # Column shape: s1__{pol|coh}__{window_kind}__count_{stat}
    # Does NOT match event-count features like drop_count, loss_count, swir_rise_count,
    # nbr_anomaly_count, mahalanobis_exceedance_count -- those are damage signal.
    re.compile(
        r'^s1__(?:vv|vh|coh)__'
        r'(?:baseline|post_baseline|blk\d+|blk_pre\d+|roll(?:3|7|13)__assessment)__'
        r'count' + _STAT_SUFFIX + r'$'
    ),
)


def is_metadata_by_pattern(col: str) -> bool:
    """Return True if `col` is metadata by name-pattern rule.

    Catches:
      - was_observed_* flags (observation-presence indicators)
      - scenes_observed rasters per modality (observation density)
      - obs_count, qa__cloud_freq, visibility__*__freq (data quality)
      - raw pixel-count columns from block/rolling/baseline processing

    Extending the rule set: add a regex to _METADATA_PATTERNS above.
    """
    return any(pat.match(col) for pat in _METADATA_PATTERNS)


def is_metadata(col: str) -> bool:
    """Return True if `col` is metadata by explicit name OR by pattern."""
    return col in META_COLS_SET or is_metadata_by_pattern(col)


def is_id(col: str) -> bool:
    return col in ID_COLS


def is_label(col: str) -> bool:
    return col in LABEL_COLS


def is_non_feature(col: str) -> bool:
    """Return True if `col` must NOT be treated as a feature (id|label|meta)."""
    return is_id(col) or is_label(col) or is_metadata(col)


# ---------------------------------------------------------------------------
# Column-list helpers
# ---------------------------------------------------------------------------

_NUMERIC_DTYPES: frozenset[str] = frozenset({'f', 'i', 'u', 'b'})


def select_feature_columns(df) -> list[str]:
    """Return numeric feature columns in df.columns order, excluding id/label/meta."""
    return [
        c for c in df.columns
        if not is_non_feature(c) and getattr(df[c], 'dtype', None) is not None
        and df[c].dtype.kind in _NUMERIC_DTYPES
    ]


def split_columns(df) -> dict[str, list[str]]:
    """Partition df.columns into {'id','label','metadata','feature','unclassified'}."""
    out: dict[str, list[str]] = {
        'id': [], 'label': [], 'metadata': [], 'feature': [], 'unclassified': [],
    }
    for c in df.columns:
        if is_id(c):
            out['id'].append(c)
        elif is_label(c):
            out['label'].append(c)
        elif is_metadata(c):
            out['metadata'].append(c)
        elif getattr(df[c], 'dtype', None) is not None and df[c].dtype.kind in _NUMERIC_DTYPES:
            out['feature'].append(c)
        else:
            out['unclassified'].append(c)
    return out


def build_role_overrides(columns: Iterable[str]) -> Mapping[str, str]:
    """Return role mapping for catalog registration: id|label|metadata; features omitted."""
    overrides: dict[str, str] = {}
    for col in columns:
        if is_id(col):
            overrides[col] = 'id'
        elif is_label(col):
            overrides[col] = 'label'
        elif is_metadata(col):
            overrides[col] = 'metadata'
    return overrides


# ---------------------------------------------------------------------------
# Manifest audit helper
# ---------------------------------------------------------------------------

def classify_manifest_features(feature_columns: Iterable[str]) -> dict[str, list[str]]:
    """Partition a manifest's feature_columns list into signal vs metadata_leak.

    Returns:
      {'signal': [...], 'metadata_leak': [...]}

    Use to audit whether parquet_manifest.json lists any metadata columns in
    `feature_columns`. With metadata_filter.py adopted, downstream notebooks
    exclude metadata_leak entries via is_non_feature() so the manifest
    doesn't need to be regenerated -- but cleaning up the manifest is tidier.
    """
    signal: list[str] = []
    metadata_leak: list[str] = []
    for col in feature_columns:
        if is_metadata(col):
            metadata_leak.append(col)
        else:
            signal.append(col)
    return {'signal': signal, 'metadata_leak': metadata_leak}


# ---------------------------------------------------------------------------
# Legacy API compatibility
# ---------------------------------------------------------------------------
#
# Older notebooks reference a single `_NON_FEATURE` set. _NON_FEATURE alone is
# INSUFFICIENT for full metadata protection -- it does NOT cover pattern-based
# metadata. Use is_non_feature(col) or select_feature_columns(df) instead.

_NON_FEATURE: frozenset[str] = ID_COLS | LABEL_COLS | META_COLS_SET


__all__ = [
    'ID_COLS', 'LABEL_COLS', 'META_COLS_SET',
    'is_id', 'is_label', 'is_metadata', 'is_metadata_by_pattern',
    'is_non_feature',
    'select_feature_columns', 'split_columns', 'build_role_overrides',
    'classify_manifest_features',
    '_NON_FEATURE',
]
