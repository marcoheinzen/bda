# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
dl_audit_zips.py
Zip-only audit: SLC zips, CARD zips, MS zips, orbit files, tracker status.
No product TIF scanning (that's dl_audit_products.py).

Notebook usage:
    from dl_audit_zips import run as run_audit_zips
    NB02_AUDIT = run_audit_zips(
        raw_slc_zip=RAW_SLC_ZIP,
        card_zip_dir=CARD_ZIP_DIR,
        ms_zip_dir=MS_ZIP_DIR,
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


def run(raw_slc_zip, card_zip_dir, ms_zip_dir, orbits_dir,
        outputs_dir, insar_tracker_file, card_tracker_file, min_slc_size=3e9):
    raw_slc_zip = Path(raw_slc_zip)
    card_zip_dir = Path(card_zip_dir)
    ms_zip_dir = Path(ms_zip_dir)
    orbits_dir = Path(orbits_dir)
    outputs_dir = Path(outputs_dir)
    insar_tracker_file = Path(insar_tracker_file)
    card_tracker_file = Path(card_tracker_file)

    print("=" * 80)
    print("DL-AUDIT-ZIPS: ZIP INVENTORY + TRACKER STATUS")
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

    sat_counts = defaultdict(int)
    for s in slc_inventory.values():
        sat_counts[s['satellite']] += 1
    slc_by_date = defaultdict(list)
    for name, info in slc_inventory.items():
        slc_by_date[info['date']].append(info)
    all_dates = sorted(set(s['date'] for s in slc_inventory.values()))
    date_range = f"{all_dates[0]} to {all_dates[-1]}" if all_dates else "none"

    print(f"\n  Valid SLC zips:    {len(slc_inventory)}")
    print(f"  Undersized (<3GB): {len(slc_undersized)}")
    print(f"  Total size:        {slc_total_gb:.0f} GB ({slc_total_gb/1024:.2f} TB)")
    print(f"  Date range:        {date_range}")
    print(f"  Unique dates:      {len(all_dates)}")
    print(f"\n  By satellite:")
    for sat in sorted(sat_counts.keys()):
        print(f"    {sat}: {sat_counts[sat]} scenes")

    if slc_undersized:
        print(f"\n  Undersized zips:")
        for s in slc_undersized:
            print(f"    {s['filename']}: {s['size_gb']:.2f} GB")

    # =========================================================================
    # 2. CARD ZIP INVENTORY
    # =========================================================================
    print(f"\n{'='*80}")
    print("2. CARD ZIP INVENTORY")
    print(f"{'='*80}")
    print(f"  CARD_ZIP_DIR: {card_zip_dir}")

    card_zips = {}
    card_zip_gb = 0
    if card_zip_dir.exists():
        for f in sorted(card_zip_dir.iterdir()):
            if f.suffix == '.zip':
                size_gb = f.stat().st_size / 1e9
                card_zips[f.stem] = {'filename': f.name, 'size_gb': size_gb, 'path': str(f)}
                card_zip_gb += size_gb
    print(f"  Total CARD zips: {len(card_zips)} ({card_zip_gb:.1f} GB)")

    # =========================================================================
    # 3. MS ZIP INVENTORY
    # =========================================================================
    print(f"\n{'='*80}")
    print("3. MS ZIP INVENTORY")
    print(f"{'='*80}")
    print(f"  MS_ZIP_DIR: {ms_zip_dir}")

    ms_zips = {}
    ms_zip_gb = 0
    if ms_zip_dir.exists():
        for f in sorted(ms_zip_dir.iterdir()):
            if f.suffix == '.zip':
                size_gb = f.stat().st_size / 1e9
                ms_zips[f.stem] = {'filename': f.name, 'size_gb': size_gb, 'path': str(f)}
                ms_zip_gb += size_gb
    print(f"  Total MS zips: {len(ms_zips)} ({ms_zip_gb:.1f} GB)")

    # =========================================================================
    # 4. ORBIT FILES
    # =========================================================================
    print(f"\n{'='*80}")
    print("4. ORBIT FILES")
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
    # 5. TRACKER STATUS
    # =========================================================================
    print(f"\n{'='*80}")
    print("5. TRACKER STATUS")
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
                        entries = len(data.get('downloads', data.get('targets', data.get('cities', data))))
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
        'card_zips': card_zips,
        'card_zip_gb': card_zip_gb,
        'ms_zips': ms_zips,
        'ms_zip_gb': ms_zip_gb,
        'orbit_count': orbit_count,
        'timestamp': datetime.now().isoformat(),
    }

    total_gb = slc_total_gb + card_zip_gb + ms_zip_gb
    print(f"\n{'='*80}")
    print(f"ZIP AUDIT COMPLETE")
    print(f"  SLC zips:  {len(slc_inventory)} ({slc_total_gb:.0f} GB)")
    print(f"  CARD zips: {len(card_zips)} ({card_zip_gb:.1f} GB)")
    print(f"  MS zips:   {len(ms_zips)} ({ms_zip_gb:.1f} GB)")
    print(f"  Total:     {total_gb:.0f} GB")
    print(f"  Orbits:    {orbit_count}")
    print(f"{'='*80}")

    return NB02_AUDIT
