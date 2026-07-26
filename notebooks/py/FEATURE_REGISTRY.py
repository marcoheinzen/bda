# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
FEATURE_REGISTRY.py
Single source of truth for ML experiment feature lists.

Reads from bda.sqlite features table (populated by NB05 FEATURES cell).
If bda.sqlite not available, falls back to hardcoded lists.

Usage in NB09:
    from FEATURE_REGISTRY import get_features, DIETRICH_28, CARD_ALL, COH_ALL, MS_ALL

    X = df[DIETRICH_28]                                  # hardcoded shortcut
    X = df[get_features('dietrich_baseline')]             # from DB by group
    X = df[get_features(sensor='s1', is_ml=True)]        # from DB by sensor
"""

import re
from pathlib import Path


# =========================================================================
# DB READER
# =========================================================================

def get_features(group_name=None, sensor=None, is_ml=True, db_path=None):
    """Read feature list from bda.sqlite features table.

    Args:
        group_name: filter by group_name (e.g. 'card_assessment', 'dietrich_baseline')
        sensor:     filter by sensor (e.g. 's1', 's2', 'cd')
        is_ml:      True = ML features only, False = all, None = no filter
        db_path:    Path to bda.sqlite. None = auto-detect from DATASET_ROOT.

    Returns:
        sorted list of column name strings
    """
    if db_path is None:
        # try common locations
        candidates = [
            Path('F:/PROJECTS/masterthesis/data_stack/dataset/bda.sqlite'),
            Path('/content/nvme_masterthesis/data_stack/dataset/bda.sqlite'),
        ]
        for p in candidates:
            if p.exists():
                db_path = p
                break

    if db_path is None or not Path(db_path).exists():
        raise FileNotFoundError(
            f"bda.sqlite not found. Run NB05 FEATURES cell first, or pass db_path explicitly."
        )

    from stack_catalog import BDACatalog
    with BDACatalog(db_path, readonly=True) as cat:
        rows = cat.query_features(group_name=group_name, is_ml=is_ml, sensor=sensor)
        return sorted([r['feature_name'] for r in rows])


def get_feature_groups_from_db(db_path=None):
    """Get all group_name values from bda.sqlite. Returns dict {group: [cols]}."""
    if db_path is None:
        candidates = [
            Path('F:/PROJECTS/masterthesis/data_stack/dataset/bda.sqlite'),
            Path('/content/nvme_masterthesis/data_stack/dataset/bda.sqlite'),
        ]
        for p in candidates:
            if p.exists():
                db_path = p
                break

    if db_path is None or not Path(db_path).exists():
        return {}

    from stack_catalog import BDACatalog
    with BDACatalog(db_path, readonly=True) as cat:
        rows = cat.raw_sql("SELECT feature_name, group_name FROM features WHERE is_ml_feature = 1 AND group_name IS NOT NULL")
        groups = {}
        for r in rows:
            groups.setdefault(r['group_name'], []).append(r['feature_name'])
        return {k: sorted(v) for k, v in groups.items()}


# =========================================================================
# HARDCODED FALLBACK LISTS (populated after first NB05 run)
# These are the EXPECTED column names after __ rename convention.
# If a column is missing at runtime -> KeyError -> immediate debugging.
# =========================================================================

# Dietrich 2025 replication: 28 features (7 stats x 2 pol x 2 periods)
_STATS = ['count', 'kurtosis', 'max', 'mean', 'median', 'min', 'skewness', 'std']
_POLS = ['vv', 'vh']

DIETRICH_BASELINE = [f"s1__{pol}__baseline__{stat}_mean" for pol in _POLS for stat in _STATS]
DIETRICH_ASSESSMENT = [f"s1__{pol}__assessment__{stat}_mean" for pol in _POLS for stat in _STATS]
DIETRICH_28 = sorted(DIETRICH_BASELINE + DIETRICH_ASSESSMENT)

# COH baseline: 6 stats x 1 (VV only in our pipeline)
COH_BASELINE = [f"s1__coh__baseline__{stat}_mean" for stat in ['count', 'max', 'mean', 'median', 'min', 'std']]

# Building attributes
BUILDING_ATTRS = ['area_m2', 'n_pixels', 'height', 'num_floors', 'roof_height']

# Convenience aggregates (populated dynamically from parquet at import time if available)
CARD_ALL = sorted(DIETRICH_BASELINE + DIETRICH_ASSESSMENT)
COH_ALL = sorted(COH_BASELINE)


def validate_columns(df, feature_list, label=""):
    """Check that all features in list exist in DataFrame. Raises KeyError if not."""
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise KeyError(f"FEATURE_REGISTRY[{label}]: {len(missing)} missing columns: {missing[:10]}")
    return True
