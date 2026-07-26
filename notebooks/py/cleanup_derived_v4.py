# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

#!/usr/bin/env python3
"""
CLEANUP SCRIPT v4: Remove old derived products with period subdirs.

Run AFTER flatten_satellite_data_v3.py.

Deletes:
  1. SAR_CARD_bbox/  (leftover from v3 if not fully cleaned)
  2. MS/{city}/composites/  (old period subdirs - NB03d recreates with pre/post)
  3. MS/{city}/rgb/  (old period-baked filenames - NB03d recreates as {city}_RGB_{date}.tif)
  4. landuse_classification/  (old period/date structure - NB03d recreates flat)

Keeps untouched:
  - SAR_CARD/{city}/temporal_stats/  (clean naming)
  - SAR_COH/{city}/coherence_baseline/  (clean naming)
  - temporal_products/  (clean naming, NB04c output)
  - SAR_METADATA/
  - SAR_SLC_ORBIT/
  - MS/{city}/*.tif  (flat scene TIFs from v3)
  - SAR_CARD/{city}/*.tif  (flat CARD TIFs from v3)
  - SAR_COH/{city}/*.tif  (flat COH TIFs from v3)

Usage:
  python cleanup_derived_v4.py --dry-run
  python cleanup_derived_v4.py
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

if os.name == 'nt':
    SAT_ROOT = Path(r'F:\PROJECTS\masterthesis\gdrive\masterthesis\data\satellite')
elif os.path.exists('/content/drive_f'):
    SAT_ROOT = Path('/content/drive_f/masterthesis/data/satellite')
else:
    SAT_ROOT = Path('/mnt/f/PROJECTS/masterthesis/gdrive/masterthesis/data/satellite')

DRY_RUN = '--dry-run' in sys.argv

stats = {'dirs_deleted': 0, 'files_deleted': 0, 'errors': []}

print("=" * 70)
print("CLEANUP v4: REMOVE OLD DERIVED PRODUCTS")
print("=" * 70)
print(f"  SAT_ROOT: {SAT_ROOT}")
print(f"  DRY_RUN:  {DRY_RUN}")
print(f"  Time:     {datetime.now().isoformat()}")

if not SAT_ROOT.exists():
    print(f"  ERROR: {SAT_ROOT} does not exist")
    sys.exit(1)


def count_contents(path):
    if not path.exists():
        return 0, 0
    n_files = sum(1 for f in path.rglob('*') if f.is_file() and f.name != 'desktop.ini')
    n_dirs = sum(1 for d in path.rglob('*') if d.is_dir())
    return n_files, n_dirs


def safe_rmtree(path, label):
    n_files, n_dirs = count_contents(path)
    if DRY_RUN:
        print(f"    DELETE: {label} ({n_files} files, {n_dirs} subdirs)")
    else:
        try:
            shutil.rmtree(str(path))
            print(f"    DELETED: {label} ({n_files} files, {n_dirs} subdirs)")
            stats['dirs_deleted'] += 1
            stats['files_deleted'] += n_files
        except Exception as e:
            stats['errors'].append(f"DELETE FAIL {path}: {e}")
            print(f"    ERROR: {label}: {e}")


# ============================================================================
# STEP 1: Delete leftover SAR_CARD_bbox
# ============================================================================
print(f"\n{'='*70}")
print("STEP 1: DELETE LEFTOVER OLD DIRS")
print(f"{'='*70}")

for old_name in ['SAR_CARD_bbox', 'SAR_SLC_bbox', 'SAR_SLC', 'SAR_CARD_polygon',
                  'SAR_SLC_polygon', 'multispectral']:
    old_dir = SAT_ROOT / old_name
    if old_dir.exists():
        safe_rmtree(old_dir, old_name)
    else:
        print(f"    {old_name}: already gone")


# ============================================================================
# STEP 2: Delete MS/{city}/composites/ (old period subdirs)
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2: DELETE MS/{city}/composites/ (old period structure)")
print(f"{'='*70}")

ms_root = SAT_ROOT / 'MS'
if ms_root.exists():
    for city_dir in sorted(ms_root.iterdir()):
        if not city_dir.is_dir() or city_dir.name in ('metadata', 'desktop.ini'):
            continue

        comp_dir = city_dir / 'composites'
        if comp_dir.exists():
            safe_rmtree(comp_dir, f"MS/{city_dir.name}/composites")
else:
    print("    MS/ not found")


# ============================================================================
# STEP 3: Delete MS/{city}/rgb/ (old period-baked filenames)
# ============================================================================
print(f"\n{'='*70}")
print("STEP 3: DELETE MS/{city}/rgb/ (old period-baked filenames)")
print(f"{'='*70}")

if ms_root.exists():
    for city_dir in sorted(ms_root.iterdir()):
        if not city_dir.is_dir() or city_dir.name in ('metadata', 'desktop.ini'):
            continue

        rgb_dir = city_dir / 'rgb'
        if rgb_dir.exists():
            safe_rmtree(rgb_dir, f"MS/{city_dir.name}/rgb")
else:
    print("    MS/ not found")


# ============================================================================
# STEP 4: Delete landuse_classification/ (old period/date structure)
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4: DELETE landuse_classification/ (old period/date structure)")
print(f"{'='*70}")

landuse_dir = SAT_ROOT / 'landuse_classification'
if landuse_dir.exists():
    safe_rmtree(landuse_dir, "landuse_classification")
else:
    print("    landuse_classification: already gone")


# ============================================================================
# STEP 5: Verify final structure
# ============================================================================
print(f"\n{'='*70}")
print("STEP 5: VERIFY FINAL STRUCTURE")
print(f"{'='*70}")

expected_dirs = ['SAR_CARD', 'SAR_COH', 'MS', 'SAR_METADATA', 'SAR_SLC_ORBIT',
                 'temporal_products']
should_not_exist = ['SAR_CARD_bbox', 'SAR_SLC_bbox', 'SAR_SLC', 'SAR_CARD_polygon',
                    'SAR_SLC_polygon', 'multispectral', 'landuse_classification']

print("  Expected dirs:")
for d in expected_dirs:
    p = SAT_ROOT / d
    exists = p.exists() if not DRY_RUN else '(check after run)'
    print(f"    {d:30s} {'OK' if exists is True else exists}")

print("  Should NOT exist:")
for d in should_not_exist:
    p = SAT_ROOT / d
    if DRY_RUN:
        print(f"    {d:30s} (check after run)")
    elif p.exists():
        print(f"    {d:30s} STILL EXISTS!")
    else:
        print(f"    {d:30s} gone OK")

# spot-check MS cities have no composites/rgb subdirs
if not DRY_RUN and ms_root.exists():
    dirty_cities = []
    for city_dir in ms_root.iterdir():
        if not city_dir.is_dir():
            continue
        if (city_dir / 'composites').exists() or (city_dir / 'rgb').exists():
            dirty_cities.append(city_dir.name)
    if dirty_cities:
        print(f"\n  WARNING: {len(dirty_cities)} cities still have composites/rgb subdirs:")
        for c in dirty_cities[:5]:
            print(f"    {c}")
    else:
        print(f"\n  All MS cities clean (no composites/rgb subdirs)")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("CLEANUP SUMMARY")
print(f"{'='*70}")
print(f"  Directories deleted: {stats['dirs_deleted']}")
print(f"  Files deleted:       {stats['files_deleted']}")
print(f"  Errors:              {len(stats['errors'])}")

if stats['errors']:
    for err in stats['errors']:
        print(f"    {err}")

if DRY_RUN:
    print(f"\n  DRY RUN — no changes made.")

print(f"\n  Clean structure:")
print(f"    satellite/SAR_CARD/{{city}}/{{city}}_CARD_{{pol}}_{{date}}.tif")
print(f"    satellite/SAR_CARD/{{city}}/temporal_stats/              (kept)")
print(f"    satellite/SAR_COH/{{city}}/{{city}}_COH_{{pol}}_{{label}}_{{d1}}_{{d2}}.tif")
print(f"    satellite/SAR_COH/{{city}}/coherence_baseline/           (kept)")
print(f"    satellite/MS/{{city}}/{{city}}_S2_{{date}}_{{band}}_{{res}}.tif")
print(f"    satellite/MS/{{city}}/                                   (NO composites/rgb)")
print(f"    satellite/temporal_products/                             (kept)")
print(f"    satellite/SAR_METADATA/                                 (kept)")
print(f"    satellite/SAR_SLC_ORBIT/                                (kept)")
print(f"\n  NB03d will recreate: composites, rgb, landuse from flat scene data.")
print(f"{'='*70}")
