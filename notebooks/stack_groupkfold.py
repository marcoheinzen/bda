# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_groupkfold.py
GroupKFold cross-validation assignment by city.

Prevents geographic memorization by ensuring no city appears in both
train and test within the same fold.

Notebook usage:
    from stack_groupkfold import run as run_groupkfold
    GKF_RESULT = run_groupkfold(stack_root=STACK_ROOT, n_folds=None)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def run(stack_root, n_folds=None, output_path=None):
    """
    Args:
        stack_root:   Path to STACK_ROOT (must contain data_stack_manifest.json)
        n_folds:      int or None. None = Leave-One-City-Out.
        output_path:  Path for output JSON. None = stack_root / 'groupkfold_assignments.json'

    Returns:
        dict with fold assignments
    """
    stack_root = Path(stack_root)

    print("=" * 70)
    print("GROUPKFOLD: CROSS-VALIDATION FOLD ASSIGNMENT BY CITY")
    print("=" * 70)

    manifest_path = stack_root / "data_stack_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run VALIDATE first: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    cities = manifest.get("cities", {})
    ml_ready = {c: info for c, info in cities.items() if info.get("ready_ml", False)}

    print(f"\n  Total cities: {len(cities)}")
    print(f"  ML-ready cities: {len(ml_ready)}")

    if not ml_ready:
        print("  No ML-ready cities.")
        return {}

    city_list = sorted(ml_ready.keys(), key=lambda c: ml_ready[c].get("n_buildings", 0), reverse=True)

    if n_folds is None or n_folds >= len(city_list):
        actual_folds = len(city_list)
        fold_mode = "leave_one_city_out"
    else:
        actual_folds = n_folds
        fold_mode = f"{n_folds}_fold_grouped"

    print(f"  Fold mode: {fold_mode}")
    print(f"  N folds: {actual_folds}")

    fold_assignments = {}
    for idx, city in enumerate(city_list):
        fold_id = idx if fold_mode == "leave_one_city_out" else idx % actual_folds
        info = ml_ready[city]
        fold_assignments[city] = {
            "fold_id": fold_id,
            "n_buildings": info.get("n_buildings", 0),
            "n_damage_labels": info.get("n_damage_labels", 0),
            "tier": info.get("tier", 99),
            "has_card": info.get("has_card", False),
            "has_coh": info.get("has_coh", False),
            "has_ms": info.get("has_ms", False),
        }

    fold_stats = {}
    for fold_id in range(actual_folds):
        fold_cities = [c for c, a in fold_assignments.items() if a["fold_id"] == fold_id]
        fold_stats[fold_id] = {
            "cities": fold_cities,
            "n_buildings": sum(fold_assignments[c]["n_buildings"] for c in fold_cities),
            "n_damage": sum(fold_assignments[c]["n_damage_labels"] for c in fold_cities),
        }

    print(f"\n  {'City':<22s} Fold  {'Bldg':>6s}  {'Dmg':>5s}  CARD  COH   MS")
    print(f"  {'-'*22} ----  {'-'*6}  {'-'*5}  ----  ----  ----")
    for city in city_list:
        a = fold_assignments[city]
        print(f"  {city:<22s}  {a['fold_id']:>2d}   {a['n_buildings']:>6d}  {a['n_damage_labels']:>5d}   "
              f"{'Y' if a['has_card'] else '-':>2s}    {'Y' if a['has_coh'] else '-':>2s}    {'Y' if a['has_ms'] else '-':>2s}")

    output = {
        "created": datetime.now().isoformat(),
        "fold_mode": fold_mode,
        "n_folds": actual_folds,
        "n_cities": len(city_list),
        "total_buildings": sum(a["n_buildings"] for a in fold_assignments.values()),
        "total_damage_labels": sum(a["n_damage_labels"] for a in fold_assignments.values()),
        "city_assignments": fold_assignments,
        "fold_stats": {str(k): v for k, v in fold_stats.items()},
    }

    if output_path is None:
        output_path = stack_root / "groupkfold_assignments.json"
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"GROUPKFOLD COMPLETE")
    print(f"{'='*70}")
    print(f"  Mode: {fold_mode}, Cities: {len(city_list)}, Folds: {actual_folds}")
    print(f"  Saved: {output_path}")

    return output
