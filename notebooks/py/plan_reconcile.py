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
plan_reconcile.py
Reconcile nb02a_scene_plan.json status fields against disk reality.

Problem: NB03a/b/c set status='extracted' only at notebook end or every N successes.
If notebook crashes, plan says 'on_disk' but TIFs actually exist.
This module scans disk and fixes the plan.

Usage (paste in any notebook after global_setup):
    import plan_reconcile
    importlib.reload(plan_reconcile)
    PLAN = plan_reconcile.reconcile(
        plan_path=PLAN_PATH,
        ms_dir=MS_DIR,
        sar_card_dir=SAR_CARD_DIR,
        sar_coh_dir=SAR_COH_DIR,
        ms_zip_dir=MS_ZIP_DIR,
        card_zip_dir=CARD_ZIP_DIR,
        raw_slc_zip=RAW_SLC_ZIP,
        save=True,
    )

Or use the one-liner cell (see CELL below).
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def _compact(d):
    """YYYY-MM-DD -> YYYYMMDD. Pass-through if already compact."""
    if not d or not isinstance(d, str):
        return d
    return d.replace('-', '')


def _scan_ms_tifs(ms_dir, city):
    """Check which MS dates have clipped TIFs on disk."""
    city_dir = Path(ms_dir) / city
    if not city_dir.is_dir():
        return set()
    dates = set()
    pat_old = re.compile(r'_(\d{8})_B\d{2}')
    pat_new = re.compile(r'^s2__b\d{2}__(\d{8})\.tif$', re.IGNORECASE)
    for f in city_dir.iterdir():
        if not f.is_file() or not f.name.endswith('.tif'):
            continue
        m = pat_old.search(f.name)
        if not m:
            m = pat_new.match(f.name)
        if m:
            dates.add(m.group(1))
    return dates


def _scan_card_tifs(sar_card_dir, city):
    """Check which CARD dates have VV+VH TIFs on disk."""
    city_dir = Path(sar_card_dir) / city
    if not city_dir.is_dir():
        return set()
    date_pols = defaultdict(set)
    pat_old = re.compile(r'CARD_(VV|VH)_(\d{8})\.tif$', re.IGNORECASE)
    pat_new = re.compile(r'^s1__(vv|vh)__(?:o\d{3}__)?(\d{8})\.tif$', re.IGNORECASE)
    for f in city_dir.iterdir():
        if not f.is_file() or not f.name.endswith('.tif'):
            continue
        m = pat_old.search(f.name) or pat_new.match(f.name)
        if m:
            pol = m.group(1).upper()
            date_str = m.group(2)
            date_pols[date_str].add(pol)
    return {d for d, pols in date_pols.items() if 'VV' in pols and 'VH' in pols}


def _scan_coh_tifs(sar_coh_dir, city):
    """Check which COH date-pairs have VV TIFs on disk."""
    city_dir = Path(sar_coh_dir) / city
    if not city_dir.is_dir():
        return set()
    pairs = set()
    pat_old = re.compile(r'COH_VV_\w+_(\d{8})_(\d{8})\.tif$')
    pat_new = re.compile(r'^s1__coh_vv__(?:o\d{3}__)?(\d{8})_(\d{8})\.tif$', re.IGNORECASE)
    for f in city_dir.rglob('*.tif'):
        m = pat_old.search(f.name) or pat_new.match(f.name)
        if m:
            d1, d2 = sorted([m.group(1), m.group(2)])
            pairs.add((d1, d2))
    return pairs


def _scan_zips(zip_dir, pattern_func):
    """Generic zip scanner. Returns set of date strings found."""
    zip_dir = Path(zip_dir)
    if not zip_dir.is_dir():
        return set()
    dates = set()
    for f in zip_dir.iterdir():
        if f.suffix != '.zip':
            continue
        d = pattern_func(f.name)
        if d:
            dates.add(d)
    return dates


def reconcile(plan_path, ms_dir=None, sar_card_dir=None, sar_coh_dir=None,
              ms_zip_dir=None, card_zip_dir=None, raw_slc_zip=None,
              save=True, verbose=True):
    """Reconcile plan status against disk reality.

    Status lifecycle:
        to_download -> on_disk (zip downloaded) -> extracted (TIFs on disk)

    Returns updated plan dict.
    """
    plan_path = Path(plan_path)
    with open(plan_path) as f:
        plan = json.load(f)

    changes = defaultdict(lambda: defaultdict(int))  # {city: {change_type: count}}
    total_changes = 0

    for city, cdata in plan.items():
        # --- MS ---
        if ms_dir:
            ms_tif_dates = _scan_ms_tifs(ms_dir, city)
            for entry in cdata.get('ms', []):
                d = _compact(entry.get('date', ''))
                old_status = entry.get('status', '')
                if d in ms_tif_dates and old_status != 'extracted':
                    entry['status'] = 'extracted'
                    changes[city]['ms_to_extracted'] += 1
                    total_changes += 1
                elif old_status == 'extracted' and d not in ms_tif_dates:
                    # marked extracted but TIFs gone (pruned?)
                    entry['status'] = 'on_disk'
                    changes[city]['ms_extracted_lost'] += 1
                    total_changes += 1

        # --- CARD ---
        if sar_card_dir:
            card_tif_dates = _scan_card_tifs(sar_card_dir, city)
            for entry in cdata.get('card', []):
                d = _compact(entry.get('date', ''))
                old_status = entry.get('status', '')
                if d in card_tif_dates and old_status != 'extracted':
                    entry['status'] = 'extracted'
                    changes[city]['card_to_extracted'] += 1
                    total_changes += 1
                elif old_status == 'extracted' and d not in card_tif_dates:
                    entry['status'] = 'on_disk'
                    changes[city]['card_extracted_lost'] += 1
                    total_changes += 1

        # --- SLC -> COH ---
        if sar_coh_dir:
            coh_pairs_on_disk = _scan_coh_tifs(sar_coh_dir, city)
            slc_dates = [_compact(e.get('date', '')) for e in cdata.get('slc', [])]
            # build expected pairs from consecutive SLC dates
            sorted_slc = sorted(slc_dates)
            for i in range(len(sorted_slc) - 1):
                pair = (sorted_slc[i], sorted_slc[i+1])
                if pair in coh_pairs_on_disk:
                    # mark both SLC scenes as extracted (COH product exists)
                    for entry in cdata.get('slc', []):
                        d = _compact(entry.get('date', ''))
                        if d in pair and entry.get('status') != 'extracted':
                            entry['status'] = 'extracted'
                            changes[city]['slc_to_extracted'] += 1
                            total_changes += 1

    if verbose:
        print('=' * 70)
        print('PLAN RECONCILE: DISK REALITY vs PLAN STATUS')
        print('=' * 70)
        print(f'  Plan: {plan_path}')
        print(f'  Total changes: {total_changes}')
        if total_changes > 0:
            for city in sorted(changes.keys()):
                parts = ', '.join(f'{k}={v}' for k, v in changes[city].items())
                print(f'    {city:<22s}: {parts}')

    # also normalize dates to YYYYMMDD while we are at it
    DATE_KEYS = {'date', 'battle_start', 'battle_stop'}
    for city, cdata in plan.items():
        for key in DATE_KEYS:
            if key in cdata and cdata[key]:
                cdata[key] = _compact(cdata[key])
        for modality in ('slc', 'card', 'ms'):
            for entry in cdata.get(modality, []):
                if 'date' in entry:
                    entry['date'] = _compact(entry['date'])
                period = entry.get('period', '')
                if period.startswith('biweekly_') and '-' in period:
                    parts = period.split('_', 1)
                    entry['period'] = parts[0] + '_' + _compact(parts[1])

    if save:
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2, default=str)
        if verbose:
            print(f'  Plan saved: {plan_path}')

    if verbose:
        # summary counts
        for mod in ('ms', 'card', 'slc'):
            statuses = defaultdict(int)
            for city in plan:
                for e in plan[city].get(mod, []):
                    statuses[e.get('status', '?')] += 1
            print(f'  {mod.upper():4s}: {dict(statuses)}')

        print('=' * 70)

    return plan
