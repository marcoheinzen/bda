# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
product_verify.py
Product completeness verification for flat satellite directory structure.
Extracted from Cell 14D-VERIFY.

Scans: SAR_COH/{city}/, SAR_CARD/{city}/, MS/{city}/ (all flat, no period subdirs).
COH filenames: s1__coh_{vv|vh}__o{orbit}__{YYYYMMDD}_{YYYYMMDD}.tif
CARD filenames: s1__{vv|vh}__o{orbit}__{YYYYMMDD}.tif
MS filenames:   s2__{band}__{YYYYMMDD}.tif  (band = b01..b12, b8a, cloud_mask, scl, visibility)

Notebook usage:
    from product_verify import run as run_product_verify
    VERIFICATION_RESULTS, VERIFICATION_SUMMARY = run_product_verify(
        sar_coh_dir=SAR_COH_DIR,
        sar_card_dir=SAR_CARD_DIR,
        ms_dir=MS_DIR,
        cities_dir=CITIES_DIR,
        insar_tracker_file=TRACKER_FILE,
        tier_selection=TIER_SELECTION,
        city_selection=CITY_SELECTION,
        cities_to_process=CITIES_TO_PROCESS if 'CITIES_TO_PROCESS' in dir() else None,
    )
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def run(sar_coh_dir, sar_card_dir, ms_dir, cities_dir, insar_tracker_file,
        tier_selection=None, city_selection=None, cities_to_process=None):
    """
    Args:
        sar_coh_dir:        Path - SAR_COH_DIR (flat COH TIFs)
        sar_card_dir:       Path - SAR_CARD_DIR (flat CARD TIFs)
        ms_dir:             Path - MS_DIR (flat MS TIFs)
        cities_dir:         Path - CITIES_DIR
        insar_tracker_file: Path - TRACKER_FILE
        tier_selection:     list or None
        city_selection:     list or None
        cities_to_process:  list or None (fallback city list)

    Returns:
        (VERIFICATION_RESULTS, VERIFICATION_SUMMARY)
    """
    sar_coh_dir = Path(sar_coh_dir)
    sar_card_dir = Path(sar_card_dir)
    ms_dir = Path(ms_dir)
    cities_dir = Path(cities_dir)
    insar_tracker_file = Path(insar_tracker_file)

    print("=" * 90)
    print("CELL 14D-VERIFY: PRODUCT COMPLETENESS VERIFICATION")
    print("=" * 90)
    print(f"Timestamp: {datetime.now().isoformat()}")

    print(f"\nVerification scope:")
    print(f"  TIER_SELECTION:  {tier_selection}")
    print(f"  CITY_SELECTION:  {city_selection}")
    print(f"  SAR_COH_DIR:     {sar_coh_dir}")
    print(f"  SAR_CARD_DIR:    {sar_card_dir}")
    print(f"  MS_DIR:          {ms_dir}")

    # ========================================================================
    # DETERMINE CITIES TO VERIFY
    # ========================================================================

    cities_to_verify = []

    if city_selection:
        cities_to_verify = list(city_selection)
    elif cities_to_process:
        cities_to_verify = list(cities_to_process)
    else:
        if cities_dir.exists():
            for city_dir in sorted(cities_dir.iterdir()):
                if city_dir.is_dir() and city_dir.name != 'metadata':
                    cities_to_verify.append(city_dir.name)

    if not cities_to_verify:
        print("\nERROR: No cities to verify.")
        return [], {}

    print(f"\nCities to verify: {len(cities_to_verify)}")

    # ========================================================================
    # LOAD INSAR TRACKER
    # ========================================================================

    insar_tracker = {}
    if insar_tracker_file.exists():
        with open(insar_tracker_file, 'r') as f:
            raw = f.read()
            raw = raw.replace(': NaN', ': null')
            insar_tracker = json.loads(raw)
        print(f"  InSAR tracker loaded: {len(insar_tracker.get('cities', {}))} cities")
    else:
        print(f"  InSAR tracker not found: {insar_tracker_file}")

    # ========================================================================
    # HELPERS — regex patterns matching double-underscore naming convention
    # ========================================================================

    # COH: s1__coh_{vv|vh}__o{orbit}__{YYYYMMDD}_{YYYYMMDD}.tif
    # No period label in filename — all pairs counted together
    coh_pattern = re.compile(r'^s1__coh_(vv|vh)__o(\d+)__(\d{8})_(\d{8})\.tif$')

    # CARD: s1__{vv|vh}__o{orbit}__{YYYYMMDD}.tif
    card_pattern = re.compile(r'^s1__(vv|vh)__o(\d+)__(\d{8})\.tif$')

    # MS spectral bands: s2__{band}__{YYYYMMDD}.tif  (b01..b12, b8a)
    ms_spectral_pattern = re.compile(r'^s2__(b\d{2}|b8a)__(\d{8})\.tif$')

    # MS all products: spectral + cloud_mask, scl, visibility
    ms_any_pattern = re.compile(r'^s2__(\w+)__(\d{8})\.tif$')

    def scan_coh_city(city):
        """Scan flat COH dir. Returns dict with pair counts."""
        city_dir = sar_coh_dir / city
        if not city_dir.exists():
            return None

        tifs = list(city_dir.glob('*.tif'))
        result = {
            'exists': True,
            'total_tifs': len(tifs),
            'pairs': set(),
            'orbits': set(),
            'vv_count': 0,
            'vh_count': 0,
            'matched': 0,
            'unmatched': 0,
        }

        for f in tifs:
            m = coh_pattern.match(f.name)
            if m:
                pol = m.group(1)
                orbit = m.group(2)
                d1 = m.group(3)
                d2 = m.group(4)
                pair_key = f"{d1}_{d2}"
                result['pairs'].add(pair_key)
                result['orbits'].add(orbit)
                result['matched'] += 1
                if pol == 'vv':
                    result['vv_count'] += 1
                else:
                    result['vh_count'] += 1
            else:
                result['unmatched'] += 1

        # coherence_baseline subdir: s1__coh__baseline__*.tif
        bl_dir = city_dir / 'coherence_baseline'
        result['bl_tifs'] = 0
        if bl_dir.exists():
            bl_tifs = list(bl_dir.glob('s1__coh__baseline__*.tif'))
            result['bl_tifs'] = len(bl_tifs)

        return result

    def scan_card_city(city):
        """Scan flat CARD dir. Returns dict with counts."""
        city_dir = sar_card_dir / city
        if not city_dir.exists():
            return None

        tifs = [f for f in city_dir.glob('*.tif') if f.is_file()]
        result = {
            'exists': True,
            'total_tifs': len(tifs),
            'dates': set(),
            'orbits': set(),
            'vv_count': 0,
            'vh_count': 0,
            'matched': 0,
            'unmatched': 0,
        }

        for f in tifs:
            m = card_pattern.match(f.name)
            if m:
                pol = m.group(1)
                orbit = m.group(2)
                date_str = m.group(3)
                result['dates'].add(date_str)
                result['orbits'].add(orbit)
                result['matched'] += 1
                if pol == 'vv':
                    result['vv_count'] += 1
                else:
                    result['vh_count'] += 1
            else:
                result['unmatched'] += 1

        # temporal_stats subdir
        ts_dir = city_dir / 'temporal_stats'
        result['temporal_stats'] = len(list(ts_dir.glob('*.tif'))) if ts_dir.exists() else 0

        return result

    def scan_ms_city(city):
        """Scan flat MS dir. Returns dict with counts."""
        city_dir = ms_dir / city
        if not city_dir.exists():
            return None

        tifs = [f for f in city_dir.glob('*.tif') if f.is_file()]
        result = {
            'exists': True,
            'total_tifs': len(tifs),
            'dates': set(),
            'bands_per_date': defaultdict(set),
            'matched': 0,
            'unmatched': 0,
        }

        for f in tifs:
            m = ms_spectral_pattern.match(f.name)
            if m:
                band = m.group(1)
                date_str = m.group(2)
                result['dates'].add(date_str)
                result['bands_per_date'][date_str].add(band)
                result['matched'] += 1
            elif ms_any_pattern.match(f.name):
                # non-spectral product (cloud_mask, scl, visibility)
                result['matched'] += 1
            else:
                result['unmatched'] += 1

        # composites subdirs: prebattle, postbattle, prebattle_baseline,
        # winter_baseline, post_winter_baseline, crossbattle
        comp_dir = city_dir / 'composites'
        result['composites'] = {}
        if comp_dir.exists():
            for sub in sorted(comp_dir.iterdir()):
                if sub.is_dir():
                    result['composites'][sub.name] = len(list(sub.glob('*.tif')))

        return result

    # ========================================================================
    # SCAN ALL CITIES
    # ========================================================================

    results = []

    for city in sorted(cities_to_verify):
        r = {'city': city}

        # --- COH ---
        coh = scan_coh_city(city)
        if coh:
            r['coh_exists'] = True
            r['coh_pairs'] = len(coh['pairs'])
            r['coh_vv'] = coh['vv_count']
            r['coh_vh'] = coh['vh_count']
            r['coh_orbits'] = len(coh['orbits'])
            r['coh_bl_tifs'] = coh['bl_tifs']
            r['coh_matched'] = coh['matched']
            r['coh_unmatched'] = coh['unmatched']
            r['coh_total'] = coh['total_tifs']
        else:
            r['coh_exists'] = False
            for k in ['coh_pairs', 'coh_vv', 'coh_vh', 'coh_orbits',
                       'coh_bl_tifs', 'coh_matched', 'coh_unmatched', 'coh_total']:
                r[k] = 0

        # --- CARD ---
        card = scan_card_city(city)
        if card:
            r['card_exists'] = True
            r['card_total_tifs'] = card['total_tifs']
            r['card_dates'] = len(card['dates'])
            r['card_vv'] = card['vv_count']
            r['card_vh'] = card['vh_count']
            r['card_orbits'] = len(card['orbits'])
            r['card_temporal_stats'] = card['temporal_stats']
        else:
            r['card_exists'] = False
            for k in ['card_total_tifs', 'card_dates', 'card_vv', 'card_vh', 'card_orbits', 'card_temporal_stats']:
                r[k] = 0

        # --- MS ---
        ms = scan_ms_city(city)
        if ms:
            r['ms_exists'] = True
            r['ms_total_tifs'] = ms['total_tifs']
            r['ms_dates'] = len(ms['dates'])
            r['ms_composites'] = ms['composites']
            r['ms_complete_scenes'] = sum(1 for d, bands in ms['bands_per_date'].items() if len(bands) >= 10)
        else:
            r['ms_exists'] = False
            for k in ['ms_total_tifs', 'ms_dates', 'ms_complete_scenes']:
                r[k] = 0
            r['ms_composites'] = {}

        # --- TRACKER STATUS ---
        tracker_city = insar_tracker.get('cities', {}).get(city, {})
        r['tracker_exists'] = bool(tracker_city)
        if tracker_city:
            for phase in ['prebattle_baseline', 'prebattle', 'postbattle', 'crossbattle', 'biweekly', 'monthly']:
                phase_data = tracker_city.get(phase, {})
                if isinstance(phase_data, dict) and 'status' in phase_data:
                    r[f'tracker_{phase}'] = phase_data['status']
                elif isinstance(phase_data, dict):
                    has_success = False
                    for key, run_data in phase_data.items():
                        if isinstance(run_data, dict) and run_data.get('status') == 'success':
                            has_success = True
                            break
                    r[f'tracker_{phase}'] = 'success' if has_success else ('attempted' if phase_data else 'pending')
                else:
                    r[f'tracker_{phase}'] = 'pending'
        else:
            for phase in ['prebattle_baseline', 'prebattle', 'postbattle', 'crossbattle', 'biweekly', 'monthly']:
                r[f'tracker_{phase}'] = 'no_tracker'

        results.append(r)

    # ========================================================================
    # DISPLAY: COH
    # ========================================================================

    print(f"\n{'='*90}")
    print(f"COH (COHERENCE) PRODUCTS — flat: SAR_COH/{{city}}/")
    print(f"{'='*90}")
    print(f"{'City':<22} {'Pairs':>6} {'VV':>6} {'VH':>6} {'Orbit':>6} {'BL':>6} {'Total':>6} {'Status':>8}")
    print("-" * 90)

    coh_ok = 0
    coh_partial = 0
    coh_missing = 0

    for r in results:
        city = r['city']
        if not r['coh_exists']:
            print(f"  {city:<20} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'---':>6} {'MISSING':>8}")
            coh_missing += 1
            continue

        pairs = r['coh_pairs']
        vv = r['coh_vv']
        vh = r['coh_vh']
        orbits = r['coh_orbits']
        bl = r['coh_bl_tifs']
        total = r['coh_total']

        if pairs >= 2 and bl >= 1:
            status = 'OK'
            coh_ok += 1
        elif pairs >= 1 or bl >= 1:
            status = 'PARTIAL'
            coh_partial += 1
        else:
            status = 'EMPTY'
            coh_missing += 1

        print(f"  {city:<20} {pairs:>6} {vv:>6} {vh:>6} {orbits:>6} {bl:>6} {total:>6} {status:>8}")

    print(f"\n  COH Summary: {coh_ok} OK, {coh_partial} partial, {coh_missing} missing/empty")

    # ========================================================================
    # DISPLAY: CARD
    # ========================================================================

    print(f"\n{'='*90}")
    print(f"CARD-BS PRODUCTS — flat: SAR_CARD/{{city}}/")
    print(f"{'='*90}")
    print(f"{'City':<22} {'Dates':>8} {'VV':>6} {'VH':>6} {'Orbit':>6} {'Total':>8} {'TempSt':>8} {'Status':>8}")
    print("-" * 90)

    card_ok = 0
    card_partial = 0
    card_missing = 0

    for r in results:
        city = r['city']
        if not r['card_exists']:
            print(f"  {city:<20} {'---':>8} {'---':>6} {'---':>6} {'---':>6} {'---':>8} {'---':>8} {'MISSING':>8}")
            card_missing += 1
            continue

        dates = r['card_dates']
        vv = r['card_vv']
        vh = r['card_vh']
        orbits = r['card_orbits']
        total = r['card_total_tifs']
        ts = r['card_temporal_stats']

        if dates >= 2 and vv >= 2 and vh >= 2:
            status = 'OK'
            card_ok += 1
        elif dates >= 1:
            status = 'PARTIAL'
            card_partial += 1
        else:
            status = 'EMPTY'
            card_missing += 1

        print(f"  {city:<20} {dates:>8} {vv:>6} {vh:>6} {orbits:>6} {total:>8} {ts:>8} {status:>8}")

    print(f"\n  CARD Summary: {card_ok} OK, {card_partial} partial, {card_missing} missing")

    # ========================================================================
    # DISPLAY: MS
    # ========================================================================

    print(f"\n{'='*90}")
    print(f"MULTISPECTRAL (Sentinel-2) PRODUCTS — flat: MS/{{city}}/")
    print(f"{'='*90}")
    print(f"{'City':<22} {'Dates':>8} {'Complete':>8} {'TIFs':>8} {'Composites':>12} {'Status':>8}")
    print(f"{'':<22} {'':>8} {'>=10band':>8} {'':>8} {'subdirs':>12}")
    print("-" * 90)

    ms_ok = 0
    ms_partial = 0
    ms_missing = 0

    for r in results:
        city = r['city']
        if not r['ms_exists']:
            print(f"  {city:<20} {'---':>8} {'---':>8} {'---':>8} {'---':>12} {'MISSING':>8}")
            ms_missing += 1
            continue

        dates = r['ms_dates']
        complete = r['ms_complete_scenes']
        total = r['ms_total_tifs']
        comp = r['ms_composites']
        comp_count = len(comp) if comp else 0
        comp_str = str(comp_count) if comp_count else "-"

        if complete >= 2:
            status = 'OK'
            ms_ok += 1
        elif dates >= 1:
            status = 'PARTIAL'
            ms_partial += 1
        else:
            status = 'EMPTY'
            ms_missing += 1

        print(f"  {city:<20} {dates:>8} {complete:>8} {total:>8} {comp_str:>12} {status:>8}")

    print(f"\n  MS Summary: {ms_ok} OK, {ms_partial} partial, {ms_missing} missing")

    # ========================================================================
    # DISPLAY: TRACKER
    # ========================================================================

    print(f"\n{'='*90}")
    print(f"INSAR TRACKER STATUS")
    print(f"{'='*90}")
    print(f"{'City':<22} {'PreBL':>10} {'Pre':>10} {'Post':>10} {'Cross':>10} {'Biwk':>10} {'Month':>10}")
    print("-" * 90)

    for r in results:
        city = r['city']
        if not r['tracker_exists']:
            print(f"  {city:<20} {'---':>10} {'---':>10} {'---':>10} {'---':>10} {'---':>10} {'---':>10}")
            continue

        bl = r.get('tracker_prebattle_baseline', '-')
        pre = r.get('tracker_prebattle', '-')
        post = r.get('tracker_postbattle', '-')
        cross = r.get('tracker_crossbattle', '-')
        bw = r.get('tracker_biweekly', '-')
        mo = r.get('tracker_monthly', '-')

        print(f"  {city:<20} {bl:>10} {pre:>10} {post:>10} {cross:>10} {bw:>10} {mo:>10}")

    # ========================================================================
    # OVERALL READINESS MATRIX
    # ========================================================================

    print(f"\n{'='*90}")
    print(f"OVERALL READINESS MATRIX")
    print(f"{'='*90}")
    print(f"{'City':<22} {'COH':>8} {'CARD':>8} {'MS':>8} {'Ready':>8}")
    print("-" * 90)

    total_ready = 0
    total_partial = 0
    total_not_ready = 0

    for r in results:
        city = r['city']

        # COH readiness
        if r['coh_exists'] and r['coh_pairs'] >= 2 and r['coh_bl_tifs'] >= 1:
            coh_status = 'OK'
        elif r['coh_exists'] and (r['coh_pairs'] >= 1 or r['coh_bl_tifs'] >= 1):
            coh_status = 'PARTIAL'
        elif not r['coh_exists']:
            coh_status = '-'
        else:
            coh_status = 'EMPTY'

        # CARD readiness
        if r['card_exists'] and r['card_dates'] >= 2:
            card_status = 'OK'
        elif r['card_exists'] and r['card_dates'] >= 1:
            card_status = 'PARTIAL'
        elif not r['card_exists']:
            card_status = '-'
        else:
            card_status = 'EMPTY'

        # MS readiness
        if r['ms_exists'] and r['ms_complete_scenes'] >= 2:
            ms_status = 'OK'
        elif r['ms_exists'] and r['ms_dates'] >= 1:
            ms_status = 'PARTIAL'
        elif not r['ms_exists']:
            ms_status = '-'
        else:
            ms_status = 'EMPTY'

        has_sar = (coh_status == 'OK' or card_status == 'OK')
        has_ms = (ms_status == 'OK')

        if has_sar and has_ms:
            overall = 'READY'
            total_ready += 1
        elif (coh_status in ('OK', 'PARTIAL') or card_status in ('OK', 'PARTIAL')) and ms_status in ('OK', 'PARTIAL'):
            overall = 'PARTIAL'
            total_partial += 1
        else:
            overall = 'NOT READY'
            total_not_ready += 1

        print(f"  {city:<20} {coh_status:>8} {card_status:>8} {ms_status:>8} {overall:>8}")

    print(f"\n{'='*90}")
    print(f"SUMMARY")
    print(f"{'='*90}")
    print(f"  Total cities verified: {len(results)}")
    print(f"  READY (SAR+MS OK):     {total_ready}")
    print(f"  PARTIAL:                {total_partial}")
    print(f"  NOT READY:              {total_not_ready}")
    print(f"{'='*90}")

    VERIFICATION_SUMMARY = {
        'tier': tier_selection,
        'total': len(results),
        'ready': total_ready,
        'partial': total_partial,
        'not_ready': total_not_ready,
        'timestamp': datetime.now().isoformat()
    }

    return results, VERIFICATION_SUMMARY
