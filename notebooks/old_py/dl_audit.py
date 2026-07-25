# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
dl_audit.py
NB02 product audit: scans SLC zips, CARD TIFs, MS scenes, orbits on disk.
Extracted from Cell DL-AUDIT.

Notebook usage:
    from dl_audit import run as run_dl_audit
    NB02_AUDIT = run_dl_audit(
        raw_slc_zip=RAW_SLC_ZIP,
        sar_card_dir=SAR_CARD_DIR,
        multispectral_root=MULTISPECTRAL_ROOT,
        orbits_dir=ORBITS_DIR,
        outputs_dir=OUTPUTS_DIR,
        insar_tracker_file=INSAR_TRACKER_FILE,
        card_tracker_file=CARD_TRACKER_FILE,
        min_slc_size=MIN_SLC_SIZE,
    )
"""

import os
import re
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def run(raw_slc_zip, sar_card_dir, multispectral_root, orbits_dir,
        outputs_dir, insar_tracker_file, card_tracker_file, min_slc_size=3e9):
    """
    Args:
        raw_slc_zip:         Path - RAW_SLC_ZIP
        sar_card_dir:        Path - SAR_CARD_DIR
        multispectral_root:  Path - MULTISPECTRAL_ROOT
        orbits_dir:          Path - ORBITS_DIR
        outputs_dir:         Path - OUTPUTS_DIR
        insar_tracker_file:  Path - INSAR_TRACKER_FILE
        card_tracker_file:   Path - CARD_TRACKER_FILE
        min_slc_size:        float

    Returns:
        NB02_AUDIT dict
    """
    raw_slc_zip = Path(raw_slc_zip)
    sar_card_dir = Path(sar_card_dir)
    multispectral_root = Path(multispectral_root)
    orbits_dir = Path(orbits_dir)
    outputs_dir = Path(outputs_dir)
    insar_tracker_file = Path(insar_tracker_file)
    card_tracker_file = Path(card_tracker_file)

    print("=" * 80)
    print("CELL DL-AUDIT: NB02 PRODUCT AUDIT")
    print("=" * 80)
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # =========================================================================
    # 1. SLC ZIP INVENTORY
    # =========================================================================
    print(f"\n{'='*80}")
    print("1. SLC ZIP INVENTORY")
    print(f"{'='*80}")
    print(f"  RAW_SLC_ZIP: {raw_slc_zip}")

    def parse_slc_filename(filename):
        name = filename.replace('.SAFE.zip', '').replace('.zip', '')
        m = re.search(
            r'(S1[ABC])_IW_SLC__1SDV_(\d{8})T(\d{6})_(\d{8})T(\d{6})_(\d{6})_([A-F0-9]+)_([A-F0-9]+)',
            name
        )
        if m:
            return {
                'satellite': m.group(1),
                'date': f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:8]}",
                'date_raw': m.group(2),
                'time': m.group(3),
                'orbit_abs': int(m.group(6)),
                'product_id': m.group(7),
                'base_name': name,
                'filename': filename,
            }
        return None

    slc_inventory = {}
    slc_corrupt = []
    slc_undersized = []
    slc_total_gb = 0

    if raw_slc_zip.exists():
        for f in sorted(raw_slc_zip.iterdir()):
            if f.suffix != '.zip':
                continue
            size = f.stat().st_size
            size_gb = size / 1e9
            parsed = parse_slc_filename(f.name)

            if parsed is None:
                print(f"  WARN: unparseable filename: {f.name}")
                continue

            parsed['size_gb'] = size_gb
            parsed['path'] = str(f)

            if size < min_slc_size:
                parsed['issue'] = 'undersized'
                slc_undersized.append(parsed)
            else:
                slc_inventory[parsed['base_name']] = parsed
                slc_total_gb += size_gb
    else:
        print(f"  WARNING: RAW_SLC_ZIP not found: {raw_slc_zip}")

    # group by satellite
    sat_counts = defaultdict(int)
    for s in slc_inventory.values():
        sat_counts[s['satellite']] += 1

    # group by date for quick lookup
    slc_by_date = defaultdict(list)
    for name, info in slc_inventory.items():
        slc_by_date[info['date']].append(info)

    # date range
    all_dates = sorted(set(s['date'] for s in slc_inventory.values()))
    date_range = f"{all_dates[0]} to {all_dates[-1]}" if all_dates else "none"

    # group by time-of-day (proxy for orbit track)
    time_groups = defaultdict(int)
    for s in slc_inventory.values():
        t = s['time'][:4]
        time_groups[t] += 1

    print(f"\n  Valid SLC zips:    {len(slc_inventory)}")
    print(f"  Undersized (<3GB): {len(slc_undersized)}")
    print(f"  Total size:        {slc_total_gb:.0f} GB ({slc_total_gb/1024:.2f} TB)")
    print(f"  Date range:        {date_range}")
    print(f"  Unique dates:      {len(all_dates)}")
    print(f"\n  By satellite:")
    for sat in sorted(sat_counts.keys()):
        print(f"    {sat}: {sat_counts[sat]} scenes")
    print(f"\n  By pass time (orbit proxy):")
    for t in sorted(time_groups.keys()):
        print(f"    ~{t[:2]}:{t[2:]} UTC: {time_groups[t]} scenes")

    if slc_undersized:
        print(f"\n  Undersized zips (possible corrupt):")
        for s in slc_undersized:
            print(f"    {s['filename']}: {s['size_gb']:.2f} GB")

    # =========================================================================
    # 2. CARD-BS PRODUCT INVENTORY
    # =========================================================================
    print(f"\n{'='*80}")
    print("2. CARD-BS PRODUCT INVENTORY")
    print(f"{'='*80}")
    print(f"  SAR_CARD_DIR: {sar_card_dir}")

    card_inventory = {}

    if sar_card_dir.exists():
        for city_dir in sorted(sar_card_dir.iterdir()):
            if not city_dir.is_dir() or city_dir.name.startswith('.') or city_dir.name == 'desktop.ini':
                continue
            city = city_dir.name
            city_data = {}

            for period_dir in sorted(city_dir.iterdir()):
                if not period_dir.is_dir():
                    continue
                period = period_dir.name
                tifs = sorted(period_dir.glob('*.tif'))
                dates = set()
                pols = set()
                for t in tifs:
                    m = re.search(r'CARD_(V[VH])_(\d{8})', t.name)
                    if m:
                        pols.add(m.group(1))
                        dates.add(m.group(2))
                city_data[period] = {
                    'tif_count': len(tifs),
                    'dates': sorted(dates),
                    'polarizations': sorted(pols),
                }

            card_inventory[city] = city_data

        print(f"\n  Cities with CARD products: {len(card_inventory)}")
        print(f"\n  {'City':25s} {'Pre':>5s} {'Post':>5s} {'BW':>5s} {'Cross':>5s} {'Month':>5s} {'TempSt':>6s} {'PreBL':>5s}")
        print(f"  {'-'*82}")
        for city in sorted(card_inventory.keys()):
            cd = card_inventory[city]
            pre = cd.get('prebattle', {}).get('tif_count', 0)
            post = cd.get('postbattle', {}).get('tif_count', 0)
            bw = sum(v.get('tif_count', 0) for k, v in cd.items() if k.startswith('biweekly'))
            cross = cd.get('crossbattle', {}).get('tif_count', 0)
            month = cd.get('monthly', {}).get('tif_count', 0)
            ts = cd.get('temporal_stats', {}).get('tif_count', 0)
            bl = cd.get('prebattle_baseline', {}).get('tif_count', 0)
            print(f"  {city:25s} {pre:5d} {post:5d} {bw:5d} {cross:5d} {month:5d} {ts:6d} {bl:5d}")
    else:
        print(f"  WARNING: SAR_CARD_DIR not found: {sar_card_dir}")

    # =========================================================================
    # 3. MULTISPECTRAL SCENE INVENTORY
    # =========================================================================
    print(f"\n{'='*80}")
    print("3. MULTISPECTRAL SCENE INVENTORY")
    print(f"{'='*80}")
    print(f"  MULTISPECTRAL_ROOT: {multispectral_root}")

    ms_inventory = {}

    if multispectral_root.exists():
        for city_dir in sorted(multispectral_root.iterdir()):
            if not city_dir.is_dir() or city_dir.name in ('metadata', 'temp', 'desktop.ini'):
                continue
            city = city_dir.name
            city_data = {}

            for period_dir in sorted(city_dir.iterdir()):
                if not period_dir.is_dir():
                    continue
                period = period_dir.name

                safe_dirs = [d for d in period_dir.iterdir()
                             if d.is_dir() and d.name.endswith('.SAFE')]
                clipped_tifs = list(period_dir.glob(f'{city}_S2_*.tif'))
                other_tifs = [t for t in period_dir.glob('*.tif') if t not in clipped_tifs]

                city_data[period] = {
                    'safe_count': len(safe_dirs),
                    'clipped_tifs': len(clipped_tifs),
                    'other_tifs': len(other_tifs),
                    'safe_names': sorted([d.name for d in safe_dirs]),
                }

            ms_inventory[city] = city_data

        print(f"\n  Cities with MS data: {len(ms_inventory)}")
        print(f"\n  {'City':25s} {'Pre':>5s} {'Post':>5s} {'BW':>5s} {'PreBL':>5s} {'WinBL':>5s} {'PWinBL':>6s} {'Mo':>5s}")
        print(f"  {'-'*82}")
        for city in sorted(ms_inventory.keys()):
            cd = ms_inventory[city]
            def sc(period):
                d = cd.get(period, {})
                return d.get('safe_count', 0) + (1 if d.get('clipped_tifs', 0) > 0 and d.get('safe_count', 0) == 0 else 0)
            pre = sc('prebattle')
            post = sc('postbattle')
            bw = sum(sc(p) for p in cd if p.startswith('biweekly'))
            bl = sc('prebattle_baseline')
            wbl = sc('winter_baseline')
            pwbl = sc('post_winter_baseline')
            mo = sc('monthly')
            if any([pre, post, bw, bl, wbl, pwbl, mo]):
                print(f"  {city:25s} {pre:5d} {post:5d} {bw:5d} {bl:5d} {wbl:5d} {pwbl:6d} {mo:5d}")
    else:
        print(f"  WARNING: MULTISPECTRAL_ROOT not found: {multispectral_root}")

    # =========================================================================
    # 4. ORBIT FILES INVENTORY
    # =========================================================================
    print(f"\n{'='*80}")
    print("4. ORBIT FILES INVENTORY")
    print(f"{'='*80}")
    print(f"  ORBITS_DIR: {orbits_dir}")

    orbit_count = 0
    orbit_sats = defaultdict(int)
    if orbits_dir.exists():
        for eof in orbits_dir.rglob('*.EOF*'):
            orbit_count += 1
            sat = eof.name[:3]
            orbit_sats[sat] += 1

    print(f"  Total orbit files: {orbit_count}")
    for sat in sorted(orbit_sats.keys()):
        print(f"    {sat}: {orbit_sats[sat]}")

    # =========================================================================
    # 5. TRACKER FILES STATUS
    # =========================================================================
    print(f"\n{'='*80}")
    print("5. TRACKER FILES STATUS")
    print(f"{'='*80}")

    trackers = {
        'SLC DL tracker': raw_slc_zip / 'download_tracker.json',
        'InSAR processing tracker': insar_tracker_file,
        'CARD download tracker': card_tracker_file,
        'MS download tracking v2': outputs_dir / 'ms_download_tracking_v2.json',
        'MS shopping list': outputs_dir / 'ms_shopping_list.csv',
    }

    for name, path in trackers.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            if path.suffix == '.json':
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        if 'downloads' in data:
                            entries = len(data['downloads'])
                        elif 'targets' in data:
                            entries = len(data['targets'])
                        elif 'cities' in data:
                            entries = len(data['cities'])
                        else:
                            entries = len(data)
                    else:
                        entries = len(data)
                    print(f"  {name:35s} OK  {size_kb:8.1f} KB  {entries:4d} entries  {mtime}")
                except:
                    print(f"  {name:35s} CORRUPT  {size_kb:8.1f} KB  {mtime}")
            else:
                with open(path) as f:
                    rows = sum(1 for _ in csv.reader(f)) - 1
                print(f"  {name:35s} OK  {size_kb:8.1f} KB  {rows:4d} rows  {mtime}")
        else:
            print(f"  {name:35s} MISSING")

    # =========================================================================
    # EXPORT
    # =========================================================================
    NB02_AUDIT = {
        'slc_inventory': slc_inventory,
        'slc_by_date': dict(slc_by_date),
        'slc_undersized': slc_undersized,
        'slc_total_gb': slc_total_gb,
        'card_inventory': card_inventory,
        'ms_inventory': ms_inventory,
        'orbit_count': orbit_count,
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n{'='*80}")
    print(f"AUDIT COMPLETE")
    print(f"{'='*80}")
    print(f"  SLC zips:     {len(slc_inventory)} valid + {len(slc_undersized)} undersized ({slc_total_gb:.0f} GB)")
    print(f"  CARD cities:  {len(card_inventory)}")
    print(f"  MS cities:    {len(ms_inventory)}")
    print(f"  Orbit files:  {orbit_count}")
    print(f"\n  Returned: NB02_AUDIT dict")
    print(f"{'='*80}")

    return NB02_AUDIT
