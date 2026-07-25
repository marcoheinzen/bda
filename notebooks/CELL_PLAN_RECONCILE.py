# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

# @title CELL PLAN-RECONCILE: SYNC PLAN STATUS WITH DISK REALITY
# =============================================================================
# Run this cell in ANY notebook after global_setup to fix plan status.
# Scans disk for actual TIFs, updates status: to_download -> on_disk -> extracted
# Also normalizes all dates to YYYYMMDD (no hyphens).
#
# Safe to run multiple times. Only changes status when disk disagrees with plan.
# =============================================================================
import sys, importlib
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import plan_reconcile
importlib.reload(plan_reconcile)

# locate plan
_plan_candidates = [
    STACK_DIR / 'nb02a_scene_plan.json' if 'STACK_DIR' in dir() else None,
    OUTPUTS_DIR / 'nb02a_scene_plan.json' if 'OUTPUTS_DIR' in dir() else None,
    NOTEBOOKS_DIR / 'nb02a_scene_plan.json' if 'NOTEBOOKS_DIR' in dir() else None,
]
PLAN_PATH = None
for _p in _plan_candidates:
    if _p and _p.exists():
        PLAN_PATH = _p
        break
assert PLAN_PATH is not None, f'Plan not found. Tried: {[str(p) for p in _plan_candidates if p]}'

PLAN = plan_reconcile.reconcile(
    plan_path=PLAN_PATH,
    ms_dir=MS_DIR if 'MS_DIR' in dir() else None,
    sar_card_dir=SAR_CARD_DIR if 'SAR_CARD_DIR' in dir() else None,
    sar_coh_dir=SAR_COH_DIR if 'SAR_COH_DIR' in dir() else None,
    save=True,
)
