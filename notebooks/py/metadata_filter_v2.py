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
  Column names always end with one of _mean, _std, _max, _mode (legacy v28
  format) OR __mean, __std, __p10, __p50, __p90, __min, __max,
  __max_abs_delta, __mean_abs_delta, __mode_raw, __mode_corrected,
  __urban_or_bare_frac, __{classname}_frac (v29 format).

  Primary products (NB03a/b/c):
    * CARD scenes: s1__{vv|vh}__*
    * COH pairs  : s1__coh_{vv|vh}__*
    * MS bands   : s2__{b02..b12, b8a, scl, cloud_mask, visibility}__*

  Derived products (NB03e):
    * Composites : s2__composite__{band|index}__*
    * Indices    : s2__{ndvi|bsi|savi|mndwi|ndbi|ndsi|ibi|baei|ui|nbr}__*
    * Block stats: s1__{vv|vh|coh}__{baseline|post_baseline|blk*|blk_pre*}__{stat}_*
    * Rolling    : s1__*__roll{3|7|13}__{assessment}__{stat}_*
    * Drop accum : s1__{coh|vv}__{max_drop|max_z_drop|running_min|drop_count}_*
    * MS accum   : s2__{swir_*|mahalanobis_*|nbr_anomaly_count|dnbr|rbr}_*
    * Landuse chg: s2__lu__{loss_count|loss_fraction|urban_retained|final_class|modal_*}_*
    * Pre/post   : pre_vv_*, post_vv_*, delta_vv_*, pre_vh_*, post_vh_*, delta_vh_*
    * Simple     : landuse__*_mode, landuse_changed, scene_s2__*, s2__landuse_mode

  v29 R2b/P1d products (rolling_accum + block_accum, NB03e v47):
    * Rolling accum (long): s1__coh_vv__roll{N}__{op}__*, s1__{vv|vh}__roll{N}__{op}__*,
                            s2__{b11|b12|b08|b8a|nbr}__roll{N}__{op}__*
                            where op in {running_min, running_max, max_abs_delta}
    * Block accum (wide):   s1__coh_vv__blk{NN}__{op}__*, s1__{vv|vh}__blk{NN}__{op}__*,
                            s2__{b11|b12|b08|b8a|nbr}__blk{NN}__{op}__*
                            where op in {running_min, running_max, max_abs_delta,
                                         drop_count, rise_count}
    * v29 categorical: {prefix}__mode_raw, __mode_corrected, __urban_or_bare_frac,
                       __{classname}_frac for landuse/composite_landuse

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

  v29 additions (battle-calendar leakage + geometry leakage):
  - Date-of-event columns (battle calendar): any column whose prefix contains
      date_first_drop, date_worst_drop,
      date_first_exceedance, date_worst_exceedance,
      date_first_swir_rise, date_worst_swir_rise,
      date_first_loss, date_persistent_loss,
      pre_date_ms, post_date_ms, pre_date_coh, post_date_coh
    -> their reductions (__min_nonzero, __max, __count_unique) are METADATA.
       The __max suffix here is a date max, not a value max.
  - Geometry-dependent pixel count from v29 aggregation_helpers:
      * any column ending in __n_pixels_valid

  Rationale: NB08c C10 diagnostic confirmed was_observed_* inflates F1 AUC by
  +0.137 and F5 by +0.136 by acting as city-identifier. Analogous logic applies
  to scene_observed / obs_count / qa__cloud_freq / visibility__*__freq / count_*:
  they systematically differ across cities and induce the same leakage vector.
  v29 strict-EO rule: dates encode the battle calendar (city-specific), pixel
  counts encode building geometry -- both leak when used as features.

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

ID_COLS: frozenset[str] = frozenset({'building_id'})

LABEL_COLS: frozenset[str] = frozenset({
    'damage_binary', 'damage_label', 'damage', 'ems98_grade',
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
# v29 substring patterns: battle-calendar leakage
# ---------------------------------------------------------------------------
#
# In v29, date-of-event rasters are reduced via extract_zonal_dates() into
# columns named e.g. `s1__coh_vv__date_first_drop__min_nonzero`. A substring
# rule is used because the prefix carries the discriminator (date_first_*).
# These reductions encode the battle calendar -- city-specific, NOT pixel signal.

META_DATE_SUBSTRINGS: tuple[str, ...] = (
    'date_first_drop',
    'date_worst_drop',
    'date_first_exceedance',
    'date_worst_exceedance',
    'date_first_swir_rise',
    'date_worst_swir_rise',
    'date_first_loss',
    'date_persistent_loss',
    'pre_date_ms', 'post_date_ms',
    'pre_date_coh', 'post_date_coh',
    # legacy variants
    'pre_date_card', 'post_date_card',
)


# ---------------------------------------------------------------------------
# Pattern-based metadata detection
# ---------------------------------------------------------------------------
#
# Two column-shape eras are supported:
#   v28 era: {prefix}_{mean|std|max|mode}     (single-underscore stat suffix)
#   v29 era: {prefix}__{mean|std|p10|p50|p90|min|max|max_abs_delta|mean_abs_delta|
#                       n_pixels_valid|mode_raw|mode_corrected|...|{class}_frac}
#                                              (double-underscore separator)

_STAT_SUFFIX_V28 = r'(?:_(?:mean|std|max|mode))'

_METADATA_PATTERNS: tuple[re.Pattern, ...] = (
    # was_observed_* flags (every wide parquet has >=1)
    re.compile(r'^was_observed_'),

    # observability: per-modality scene-count rasters (NB03e)
    re.compile(r'^s1__(?:vv|vh|coh)__scenes_observed' + _STAT_SUFFIX_V28 + r'$'),
    re.compile(r'^s2__scenes_observed' + _STAT_SUFFIX_V28 + r'$'),
    re.compile(r'^s2__lu__scenes_observed' + _STAT_SUFFIX_V28 + r'$'),
    # v29 era variants (double-underscore aggregator suffix)
    re.compile(r'^s1__(?:vv|vh|coh)__scenes_observed__\w+$'),
    re.compile(r'^s2__scenes_observed__\w+$'),
    re.compile(r'^s2__lu__scenes_observed__\w+$'),

    # observation count per composite period (NB03e P2: qa__obs_count.tif)
    re.compile(r'^s2__obs_count__[a-z_]+' + _STAT_SUFFIX_V28 + r'$'),

    # cloud / visibility frequency maps (NB03e P5: qa__cloud_freq.tif, visibility__*__freq.tif)
    re.compile(r'^s2__qa__cloud_freq__[a-z_]+' + _STAT_SUFFIX_V28 + r'$'),
    re.compile(r'^s2__visibility__(?:cloud|fire|smoke|clear)__freq__[a-z_]+' + _STAT_SUFFIX_V28 + r'$'),

    # raw pixel counts from block/rolling/baseline processing (NB03e block_stats and rolling_stats).
    # Column shape: s1__{pol|coh}__{window_kind}__count_{stat}
    # Does NOT match event-count features like drop_count, loss_count, swir_rise_count,
    # nbr_anomaly_count, mahalanobis_exceedance_count -- those are damage signal.
    re.compile(
        r'^s1__(?:vv|vh|coh)__'
        r'(?:baseline|post_baseline|blk\d+|blk_pre\d+|roll(?:3|7|13)__assessment)__'
        r'count' + _STAT_SUFFIX_V28 + r'$'
    ),

    # v29: any column ending in __n_pixels_valid (geometry-dependent count emitted
    # by every aggregation_helpers helper alongside its feature reductions)
    re.compile(r'__n_pixels_valid$'),

    # v29 fusion: was_observed_* with double-underscore prefix variant
    # (some fusion code may emit was_observed_<source_name> with non-standard separators)
    re.compile(r'^was_observed__'),
)


def is_metadata_by_substring(col: str) -> bool:
    """Return True if `col` is metadata by substring rule (v29 date columns)."""
    return any(s in col for s in META_DATE_SUBSTRINGS)


def is_metadata_by_pattern(col: str) -> bool:
    """Return True if `col` is metadata by name-pattern rule.

    Catches:
      - was_observed_* flags (observation-presence indicators)
      - scenes_observed rasters per modality (observation density, v28 + v29)
      - obs_count, qa__cloud_freq, visibility__*__freq (data quality, v28)
      - raw pixel-count columns from block/rolling/baseline processing (v28)
      - __n_pixels_valid suffix from any v29 aggregation helper
      - any column whose prefix contains a date_*_* substring (v29 dates)

    Extending the rule set: add a regex to _METADATA_PATTERNS or a substring
    to META_DATE_SUBSTRINGS above.
    """
    return any(pat.search(col) if pat.pattern.startswith('__') or pat.pattern.endswith('$') and not pat.pattern.startswith('^')
               else pat.match(col) for pat in _METADATA_PATTERNS) or is_metadata_by_substring(col)


def is_metadata(col: str) -> bool:
    """Return True if `col` is metadata by explicit name OR by pattern OR by substring."""
    if col in META_COLS_SET:
        return True
    # check substring rule first (cheaper) before regex sweep
    if is_metadata_by_substring(col):
        return True
    # full regex sweep (uses .search() for trailing patterns like __n_pixels_valid$)
    for pat in _METADATA_PATTERNS:
        if pat.search(col):
            return True
    return False


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
    'META_DATE_SUBSTRINGS',
    'is_id', 'is_label', 'is_metadata', 'is_metadata_by_pattern',
    'is_metadata_by_substring', 'is_non_feature',
    'select_feature_columns', 'split_columns', 'build_role_overrides',
    'classify_manifest_features',
    '_NON_FEATURE',
]
