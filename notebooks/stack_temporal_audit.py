# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_temporal_audit.py
Quarantine late post-battle scenes in data_stack.

For cities with ENDED battles (scans flat dirs, uses date extraction):
  - CARD flat: quarantine post-battle scenes beyond MAX_CARD_POST_MONTHS
  - MS flat: quarantine post-battle scenes beyond post-winter window + buffer
  - Landuse postbattle date dirs: quarantine beyond MS cutoff

NEVER quarantines: pre-battle scenes, composites/, temporal_stats/,
coherence_baseline/, derived products, ongoing conflicts.

Notebook usage:
    from stack_temporal_audit import run as run_temporal_audit
    AUDIT_RESULT = run_temporal_audit(
        stack_root=STACK_ROOT,
        cities=CITIES_TO_PROCESS,
        load_aoi_fn=load_aoi,
        logs_dir=LOGS_DIR,
    )
"""

import re
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def extract_date_from_filename(filename):
    m = re.search(r'(\d{8})', filename)
    if m:
        try:
            return datetime.strptime(m.group(1), '%Y%m%d')
        except Exception:
            pass
    return None


def extract_date_from_dirname(dirname):
    if re.match(r'^\d{8}$', dirname):
        try:
            return datetime.strptime(dirname, '%Y%m%d')
        except Exception:
            pass
    return None


def _load_battle_dates(city_name, load_aoi_fn):
    try:
        aoi_row = load_aoi_fn(city_name)
    except (FileNotFoundError, ValueError):
        return None, None, True

    battle_start = str(aoi_row.get('battle_start', '') or '')
    battle_stop = str(aoi_row.get('battle_stop', '') or '')

    if not battle_start:
        return None, None, True

    try:
        bs = datetime.strptime(battle_start[:10], '%Y-%m-%d')
    except Exception:
        return None, None, True

    ongoing = battle_stop in (None, '', 'ongoing')
    if not ongoing:
        try:
            be = datetime.strptime(battle_stop[:10], '%Y-%m-%d')
        except Exception:
            return bs, None, True
        return bs, be, False

    return bs, None, True


def compute_postwinter_end(battle_stop_dt):
    """Compute end of first post-battle winter window.
    Mirrors NB02 search_ms_post_winter_baseline() logic.
    """
    month = battle_stop_dt.month
    if month in (10, 11, 12):
        pw_end = datetime(battle_stop_dt.year + 1, 3, 31)
    elif month in (1, 2):
        pw_end = datetime(battle_stop_dt.year, 3, 31)
    elif month in (3, 4, 5, 6, 7, 8, 9):
        pw_end = datetime(battle_stop_dt.year + 1, 3, 31)
    else:
        pw_end = battle_stop_dt + relativedelta(months=12)
    return pw_end


def run(stack_root, cities, load_aoi_fn, logs_dir=None,
        dry_run=False, max_card_post_months=6, postwinter_buffer_months=1):
    """
    Args:
        stack_root:               Path to STACK_ROOT
        cities:                   list of city names
        load_aoi_fn:              callable(city_name) -> Series with battle_start/stop
        logs_dir:                 Path or None
        dry_run:                  bool
        max_card_post_months:     int
        postwinter_buffer_months: int

    Returns:
        dict with audit results
    """
    stack_root = Path(stack_root)

    print("=" * 70)
    print("TEMPORAL-AUDIT: SCAN + QUARANTINE LATE POST-BATTLE SCENES")
    print("=" * 70)
    print(f"  STACK_ROOT:           {stack_root}")
    print(f"  MAX_CARD_POST_MONTHS: {max_card_post_months}")
    print(f"  POSTWINTER_BUFFER:    {postwinter_buffer_months} months")
    print(f"  DRY_RUN:              {dry_run}")
    print(f"  Cities:               {len(cities)}")

    city_dirs = [stack_root / c for c in cities if (stack_root / c / "reference_grid.json").exists()]
    print(f"  Cities with ref grid: {len(city_dirs)}")

    summary_rows = []
    total_quarantine = 0
    total_ok = 0
    total_composite_candidates = 0

    for ci, city_dir in enumerate(city_dirs):
        city_name = city_dir.name
        battle_start, battle_stop, ongoing = _load_battle_dates(city_name, load_aoi_fn)

        if battle_start is None or ongoing:
            continue

        card_cutoff = battle_stop + relativedelta(months=max_card_post_months)
        pw_end = compute_postwinter_end(battle_stop)
        ms_cutoff = pw_end + relativedelta(months=postwinter_buffer_months)

        city_quarantine = 0
        city_ok = 0
        city_issues = []
        city_composite_candidates = []

        # 1. CARD flat dir — quarantine post-battle scenes beyond cutoff
        card_flat_dir = city_dir / "SAR_CARD" / "flat"
        if card_flat_dir.exists():
            for tif in sorted(card_flat_dir.glob("*.tif")):
                fd = extract_date_from_filename(tif.name)
                if fd is None:
                    continue
                if fd <= battle_stop:
                    city_ok += 1
                    continue
                if fd > card_cutoff:
                    city_quarantine += 1
                    city_issues.append({
                        'file': str(tif.relative_to(city_dir)),
                        'date': fd.strftime('%Y-%m-%d'),
                        'reason': f'CARD too_late (cutoff={card_cutoff.strftime("%Y-%m-%d")})',
                        'group': 'SAR_CARD',
                    })
                    if not dry_run:
                        q = city_dir / "quarantine" / "SAR_CARD"
                        q.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(tif), str(q / tif.name))
                else:
                    city_ok += 1

        # 2. MS flat dir — quarantine post-battle scenes beyond cutoff
        ms_flat_dir = city_dir / "multispectral" / "flat"
        if ms_flat_dir.exists():
            for tif in sorted(ms_flat_dir.glob("*.tif")):
                fd = extract_date_from_filename(tif.name)
                if fd is None:
                    continue
                if fd <= battle_stop:
                    city_ok += 1
                    continue
                if fd > ms_cutoff:
                    city_quarantine += 1
                    city_issues.append({
                        'file': str(tif.relative_to(city_dir)),
                        'date': fd.strftime('%Y-%m-%d'),
                        'reason': f'MS too_late (cutoff={ms_cutoff.strftime("%Y-%m-%d")})',
                        'group': 'multispectral',
                    })
                    if not dry_run:
                        q = city_dir / "quarantine" / "multispectral"
                        q.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(tif), str(q / tif.name))
                else:
                    city_ok += 1

        # 3. Landuse postbattle date dirs
        lu_post_dir = city_dir / "landuse" / "postbattle"
        if lu_post_dir.exists():
            for date_subdir in sorted(lu_post_dir.iterdir()):
                if not date_subdir.is_dir():
                    continue
                subdir_date = extract_date_from_dirname(date_subdir.name)
                if subdir_date and subdir_date > ms_cutoff:
                    n_files = len(list(date_subdir.rglob("*")))
                    city_issues.append({
                        'file': str(date_subdir.relative_to(city_dir)),
                        'date': subdir_date.strftime('%Y-%m-%d'),
                        'reason': 'landuse dir too_late',
                        'group': 'landuse',
                        'n_files': n_files,
                    })
                    city_quarantine += n_files
                    if not dry_run:
                        q = city_dir / "quarantine" / "landuse" / "postbattle"
                        q.mkdir(parents=True, exist_ok=True)
                        dst = q / date_subdir.name
                        if dst.exists():
                            shutil.rmtree(str(dst))
                        shutil.move(str(date_subdir), str(dst))

        # 4. Safety: don't leave 0 post-battle CARD in flat dir
        if card_flat_dir.exists() and city_quarantine > 0:
            card_post_ok = sum(1 for tif in card_flat_dir.glob("*.tif")
                               if extract_date_from_filename(tif.name) and
                               extract_date_from_filename(tif.name) > battle_stop and
                               extract_date_from_filename(tif.name) <= card_cutoff)
            if card_post_ok == 0:
                print(f"\n  {city_name}: WARNING - quarantine would leave 0 post-battle CARD!")
                city_issues = [iss for iss in city_issues if iss['group'] != 'SAR_CARD']

        total_quarantine += city_quarantine
        total_ok += city_ok
        total_composite_candidates += len(city_composite_candidates)

        if city_issues or city_composite_candidates:
            print(f"\n  {city_name} (battle: {battle_start.strftime('%Y-%m-%d')} to {battle_stop.strftime('%Y-%m-%d')})")
            print(f"    CARD cutoff: {card_cutoff.strftime('%Y-%m-%d')}, MS cutoff: {ms_cutoff.strftime('%Y-%m-%d')}")
            print(f"    OK: {city_ok}, Quarantine: {city_quarantine}")
            for iss in city_issues[:5]:
                action = '[DRY]' if dry_run else '[MOVED]'
                print(f"    {action} {iss['file']}  {iss['reason']}")
            if len(city_issues) > 5:
                print(f"    ... and {len(city_issues) - 5} more")

            summary_rows.append({
                'city': city_name,
                'battle_stop': battle_stop.strftime('%Y-%m-%d'),
                'card_cutoff': card_cutoff.strftime('%Y-%m-%d'),
                'ms_cutoff': ms_cutoff.strftime('%Y-%m-%d'),
                'n_ok': city_ok,
                'n_quarantine': city_quarantine,
                'n_composite_candidates': len(city_composite_candidates),
            })

    # summary
    print(f"\n{'='*70}")
    print(f"TEMPORAL AUDIT {'(DRY RUN)' if dry_run else 'COMPLETE'}")
    print(f"{'='*70}")
    print(f"  Total OK: {total_ok}, Quarantine: {total_quarantine}, Cities affected: {len(summary_rows)}")

    if total_quarantine > 0:
        print(f"\n  {'City':<25s} {'Stop':>12s} {'CARD cut':>12s} {'MS cut':>12s} {'OK':>5s} {'Quar':>5s}")
        print(f"  {'-'*70}")
        for row in summary_rows:
            print(f"  {row['city']:<25s} {row['battle_stop']:>12s} {row['card_cutoff']:>12s} {row['ms_cutoff']:>12s} {row['n_ok']:>5d} {row['n_quarantine']:>5d}")

    result = {
        'dry_run': dry_run,
        'total_ok': total_ok,
        'total_quarantine': total_quarantine,
        'total_composite_candidates': total_composite_candidates,
        'cities': summary_rows,
        'timestamp': datetime.now().isoformat(),
    }

    if logs_dir:
        audit_path = Path(logs_dir) / "temporal_audit.json"
        with open(audit_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Log: {audit_path}")

    return result
