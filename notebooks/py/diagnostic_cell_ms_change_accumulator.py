# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

# @title DIAGNOSTIC: inspect ms_change_accumulator state on disk (read-only)
# Paste in a new cell AFTER global_setup + PLAN-LOAD have run.
# Produces a matrix of city x product showing what's on disk and file sizes.
# Does NOT modify anything.

from pathlib import Path

_EXPECTED = [
    's2__swir_z_running_max.tif',
    's2__nbr_z_abs_running_max.tif',
    's2__swir_rise_count.tif',
    's2__nbr_anomaly_count.tif',
    's2__date_first_swir_rise.tif',
    's2__date_worst_swir_rise.tif',
    's2__scenes_observed.tif',
    's2__lu_transition.tif',
    's2__swir_baseline_mean.tif',
    's2__swir_baseline_std.tif',
    's2__nbr_baseline_mean.tif',
    's2__nbr_baseline_std.tif',
]

print("=" * 110)
print("DIAGNOSTIC: ms_change_accumulator disk state per city")
print("=" * 110)
print(f"  TEMPORAL_ROOT = {TEMPORAL_ROOT}")
print(f"  subdir        = {MS_CHANGE_ACCUMULATOR_SUBDIR}")
print()

# header
_labels = ['swirZmax', 'nbrZabs', 'swirRise', 'nbrAnom', 'dFirst', 'dWorst',
           'scenes', 'luTrans', 'swirBm', 'swirBs', 'nbrBm', 'nbrBs']
print(f"  {'city':<20s} {'dir':<4s}  " + ' '.join(f"{l:<9s}" for l in _labels) + f" {'meta':<5s}")
print("  " + "-" * 140)

_totals = {'ok': 0, 'partial': 0, 'missing_dir': 0, 'no_meta': 0}
_problem_cities = []

for CITY in CITIES_TO_PROCESS:
    _d = TEMPORAL_ROOT / CITY / 'MS' / MS_CHANGE_ACCUMULATOR_SUBDIR
    if not _d.exists():
        print(f"  {CITY:<20s} {'NO':<4s}  " + ' '.join(f"{'-':<9s}" for _ in _EXPECTED))
        _totals['missing_dir'] += 1
        _problem_cities.append((CITY, 'DIR_MISSING'))
        continue

    cells = []
    n_ok = 0
    n_missing = 0
    n_zero = 0
    for p in _EXPECTED:
        _f = _d / p
        if not _f.exists():
            cells.append(f"{'MISS':<9s}")
            n_missing += 1
        else:
            _sz = _f.stat().st_size
            if _sz < 200:
                cells.append(f"{'0B':<9s}")
                n_zero += 1
            else:
                # show KB for brevity
                cells.append(f"{_sz/1024:>7.1f}KB")
                n_ok += 1

    _meta_exists = (_d / 'ms_change_accumulator_meta.json').exists()
    print(f"  {CITY:<20s} {'YES':<4s}  " + ' '.join(cells) + f" {'YES' if _meta_exists else 'NO':<5s}")

    if n_ok == 12:
        _totals['ok'] += 1
        if not _meta_exists:
            _totals['no_meta'] += 1
            _problem_cities.append((CITY, 'NO_META_JSON'))
    else:
        _totals['partial'] += 1
        _problem_cities.append((CITY, f'INCOMPLETE({n_ok}/12 ok, {n_missing} miss, {n_zero} zero)'))

print()
print(f"  Summary: {_totals['ok']} complete, {_totals['partial']} partial, "
      f"{_totals['missing_dir']} dir-missing, {_totals['no_meta']} complete-but-no-meta")

if _problem_cities:
    print(f"\n  Cities needing rerun ({len(_problem_cities)}):")
    for c, status in _problem_cities:
        print(f"    {c:<25s} {status}")

# Upstream input check for problem cities: is the input data there?
print(f"\n{'='*110}")
print("UPSTREAM INPUT CHECK for problem cities")
print("=" * 110)

for CITY, _status in _problem_cities:
    print(f"\n  {CITY} [{_status}]")
    # MS bands
    _ms = MS_DIR / CITY
    if not _ms.exists():
        print(f"    MS_DIR missing: {_ms}")
        continue
    _b11 = list(_ms.glob('*B11*.tif')) + list(_ms.glob('s2__b11__*.tif'))
    _b12 = list(_ms.glob('*B12*.tif')) + list(_ms.glob('s2__b12__*.tif'))
    _b08 = list(_ms.glob('*B08*.tif')) + list(_ms.glob('s2__b08__*.tif'))
    _b8a = list(_ms.glob('*B8A*.tif')) + list(_ms.glob('s2__b8a__*.tif'))
    print(f"    MS bands: B11={len(_b11)} B12={len(_b12)} B08={len(_b08)} B8A={len(_b8a)}")
    # cloud masks
    _cm = list(_ms.glob('*cloud_mask*.tif'))
    print(f"    cloud masks: {len(_cm)}")
    # NBR scenes
    _nbr_d = _ms / MS_NBR_SUBDIR
    _nbr_n = len(list(_nbr_d.glob('*.tif'))) if _nbr_d.exists() else 0
    print(f"    NBR scenes: {_nbr_n}")
    # landuse
    _lu = LANDUSE_DIR / CITY
    if not _lu.exists():
        print(f"    LANDUSE_DIR missing: {_lu}")
    else:
        _lu_tifs = list(_lu.rglob('s2__landuse.tif'))
        print(f"    landuse scenes: {len(_lu_tifs)}")
    # plan dates
    _plan_n = len(PLAN_DATES_MS.get(CITY, set()))
    print(f"    plan MS dates: {_plan_n}")

print(f"\n{'='*110}")
