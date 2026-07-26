# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

# bda -- Building Damage Assessment using Sentinel-1/2 satellite imagery
#
# This program is free software: you can redistribute it and/or modify
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
PATCH for scene_plan.py
Apply AFTER the existing code. Adds:
  1. _compact_date() - strips hyphens from YYYY-MM-DD -> YYYYMMDD
  2. _normalize_plan_dates() - walks entire plan dict, converts all date fields
  3. Monkey-patches build_from_verified to normalize dates before saving

Usage in NB02a:
    import scene_plan
    importlib.reload(scene_plan)

    # Apply the patch
    exec(open(NOTEBOOKS_DIR / 'scene_plan_patch.py').read())

    from scene_plan import build_from_verified, compute_reuse_report
    PLAN = build_from_verified(...)  # dates will be YYYYMMDD
"""

import scene_plan as _sp
import json
from pathlib import Path


def _compact_date(d):
    """YYYY-MM-DD -> YYYYMMDD. Already compact strings pass through."""
    if not d or not isinstance(d, str):
        return d
    return d.replace('-', '')


def _normalize_plan_dates(plan):
    """Walk entire plan dict, convert all date fields to YYYYMMDD."""
    DATE_KEYS = {'date', 'battle_start', 'battle_stop'}

    for city, cdata in plan.items():
        # top-level city fields
        for key in DATE_KEYS:
            if key in cdata and cdata[key]:
                cdata[key] = _compact_date(cdata[key])

        # scene entries (slc, card, ms)
        for modality in ('slc', 'card', 'ms'):
            for entry in cdata.get(modality, []):
                if 'date' in entry:
                    entry['date'] = _compact_date(entry['date'])
                # SLC biweekly periods contain date suffix: "biweekly_2022-03-04"
                period = entry.get('period', '')
                if period.startswith('biweekly_') and '-' in period:
                    parts = period.split('_', 1)
                    entry['period'] = parts[0] + '_' + _compact_date(parts[1])

    return plan


# Monkey-patch: wrap build_from_verified to normalize dates
_original_build = _sp.build_from_verified

def _patched_build(*args, **kwargs):
    plan = _original_build(*args, **kwargs)
    _normalize_plan_dates(plan)

    # Re-save with compact dates
    outputs_dir = kwargs.get('outputs_dir') or args[3]
    plan_file = Path(outputs_dir) / 'nb02a_scene_plan.json'
    with open(plan_file, 'w') as f:
        json.dump(plan, f, indent=2, default=str)
    print(f"  Plan re-saved with YYYYMMDD dates: {plan_file}")

    return plan

_sp.build_from_verified = _patched_build

print("scene_plan_patch.py: applied. build_from_verified will output YYYYMMDD dates.")
