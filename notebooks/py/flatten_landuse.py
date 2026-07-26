# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
flatten_landuse.py
One-time script to flatten landuse/{period}/{date}/ structure to landuse/flat/.

Before: {city}/landuse/{period}/{YYYYMMDD}/lulc__class.tif
        {city}/landuse/{period}/{YYYYMMDD}/indices/s2__ndvi.tif
After:  {city}/landuse/flat/lulc__class__YYYYMMDD.tif
        {city}/landuse/flat/s2__ndvi__YYYYMMDD.tif

Run once. Copies (not moves) so old structure preserved until verified.
After verification, delete old period dirs manually.

Usage:
    python flatten_landuse.py                     # dry run
    python flatten_landuse.py --execute           # actually copy
    python flatten_landuse.py --execute --delete  # copy + delete old period dirs
"""

import shutil
import re
import sys
from pathlib import Path
from datetime import datetime

DRY_RUN = '--execute' not in sys.argv
DELETE_OLD = '--delete' in sys.argv

STACK_ROOT = Path(r'F:\PROJECTS\masterthesis\data_stack')
if not STACK_ROOT.exists():
    STACK_ROOT = Path('/mnt/f/PROJECTS/masterthesis/data_stack')
if not STACK_ROOT.exists():
    print(f"STACK_ROOT not found: {STACK_ROOT}")
    sys.exit(1)

DATE_RE = re.compile(r'^\d{8}$')

print("=" * 70)
print(f"FLATTEN LANDUSE: {'DRY RUN' if DRY_RUN else 'EXECUTE'}")
print(f"  STACK_ROOT: {STACK_ROOT}")
print(f"  Delete old: {DELETE_OLD}")
print("=" * 70)

total_copied = 0
total_skipped = 0
total_cities = 0
cities_processed = []

for city_dir in sorted(STACK_ROOT.iterdir()):
    if not city_dir.is_dir():
        continue
    lu_dir = city_dir / 'landuse'
    if not lu_dir.is_dir():
        continue

    flat_dir = lu_dir / 'flat'
    city_name = city_dir.name
    city_copied = 0
    city_skipped = 0

    period_dirs = sorted([d for d in lu_dir.iterdir()
                          if d.is_dir() and d.name != 'flat'])

    for period_dir in period_dirs:
        date_dirs = sorted([d for d in period_dir.iterdir()
                            if d.is_dir() and DATE_RE.match(d.name)])

        for date_dir in date_dirs:
            date_str = date_dir.name

            tifs = list(date_dir.glob('*.tif'))
            idx_dir = date_dir / 'indices'
            if idx_dir.is_dir():
                tifs.extend(idx_dir.glob('*.tif'))

            for tif in tifs:
                stem = tif.stem
                new_name = f"{stem}__{date_str}.tif"
                dst = flat_dir / new_name

                if dst.exists():
                    city_skipped += 1
                    continue

                if DRY_RUN:
                    print(f"  WOULD COPY: {tif.relative_to(city_dir)} -> landuse/flat/{new_name}")
                else:
                    flat_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(tif), str(dst))

                city_copied += 1

    if city_copied > 0 or city_skipped > 0:
        total_copied += city_copied
        total_skipped += city_skipped
        total_cities += 1
        action = "would copy" if DRY_RUN else "copied"
        print(f"  {city_name:<25s} {action}: {city_copied}, skipped: {city_skipped}")
        cities_processed.append(city_name)

    if DELETE_OLD and not DRY_RUN and city_copied > 0:
        for period_dir in period_dirs:
            if period_dir.name == 'flat':
                continue
            shutil.rmtree(str(period_dir))
            print(f"    DELETED: {period_dir.relative_to(city_dir)}")

print(f"\n{'=' * 70}")
print(f"FLATTEN COMPLETE")
print(f"{'=' * 70}")
action = "Would copy" if DRY_RUN else "Copied"
print(f"  {action}: {total_copied} files across {total_cities} cities")
print(f"  Skipped: {total_skipped} (already exist)")
if DRY_RUN:
    print(f"\n  Run with --execute to actually copy files")
    print(f"  Run with --execute --delete to also remove old period dirs")
