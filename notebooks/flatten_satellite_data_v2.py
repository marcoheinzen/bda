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
MIGRATION SCRIPT v2: Flatten satellite data to clean directory structure.

Target structure:
  satellite/SAR_CARD/{city}/{city}_CARD_{VV|VH}_{YYYYMMDD}.tif
  satellite/SAR_CARD/{city}/temporal_stats/...           (derived, keep subdir)
  satellite/SAR_COH/{city}/{city}_COH_{VV|VH}_{YYYYMMDD1}_{YYYYMMDD2}.tif
  satellite/SAR_COH/{city}/coherence_baseline/...        (derived, keep subdir)
  satellite/MS/{city}/{city}_S2_{YYYYMMDD}_{BAND}_{RES}.tif
  satellite/MS/{city}/composites/...                     (derived, keep subdir)
  satellite/MS/{city}/rgb/...                            (derived, keep subdir)
  satellite/MS/metadata/...                              (MS scene metadata)
  satellite/SAR_METADATA/{city}_scene_metadata.json      (SLC discovery metadata)
  satellite/sentinel_1_orbits/...                        (unchanged)
  satellite/temporal_products/...                        (NB04c, unchanged)
  satellite/landuse_classification/...                   (unchanged)

Deletions after migration:
  satellite/SAR_CARD_bbox/      (migrated to SAR_CARD)
  satellite/SAR_CARD/           (old empty dir, replaced)
  satellite/SAR_CARD_polygon/   (empty, delete)
  satellite/SAR_SLC_bbox/       (migrated to SAR_COH)
  satellite/SAR_SLC/            (metadata moved to SAR_METADATA)
  satellite/SAR_SLC_polygon/    (only Mariupol, delete)
  satellite/multispectral/      (migrated to MS)

.SAFE directories are DELETED (extracted zip sources, regenerable from ms_zip).

Usage:
  python flatten_satellite_data_v2.py --dry-run     # preview
  python flatten_satellite_data_v2.py               # execute
"""

import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

if os.name == 'nt':
    SAT_ROOT = Path(r'F:\PROJECTS\masterthesis\gdrive\masterthesis\data\satellite')
elif os.path.exists('/content/drive_f'):
    SAT_ROOT = Path('/content/drive_f/masterthesis/data/satellite')
else:
    SAT_ROOT = Path('/mnt/f/PROJECTS/masterthesis/gdrive/masterthesis/data/satellite')

DRY_RUN = '--dry-run' in sys.argv

# derived product subdirs to KEEP as subdirs (not flatten)
DERIVED_SUBDIRS = {'temporal_stats', 'coherence_baseline', 'composites', 'rgb',
                   'cloud_masks', 'metadata'}

stats = {'moved': 0, 'renamed': 0, 'deleted_safe': 0, 'deleted_empty': 0,
         'deleted_dirs': 0, 'skipped': 0, 'errors': []}

print("=" * 70)
print("FLATTEN SATELLITE DATA v2")
print("=" * 70)
print(f"  SAT_ROOT: {SAT_ROOT}")
print(f"  DRY_RUN:  {DRY_RUN}")
print(f"  Time:     {datetime.now().isoformat()}")

if not SAT_ROOT.exists():
    print(f"  ERROR: {SAT_ROOT} does not exist")
    sys.exit(1)


def safe_move(src, dst):
    if DRY_RUN:
        return True
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        stats['errors'].append(f"MOVE FAIL: {src} -> {dst}: {e}")
        return False


def safe_delete(path, is_dir=False):
    if DRY_RUN:
        return True
    try:
        if is_dir:
            shutil.rmtree(str(path))
        else:
            path.unlink()
        return True
    except Exception as e:
        stats['errors'].append(f"DELETE FAIL: {path}: {e}")
        return False


def remove_empty_dirs(path):
    if not path.is_dir():
        return
    for child in list(path.iterdir()):
        if child.is_dir():
            remove_empty_dirs(child)
    remaining = [f for f in path.iterdir() if f.name != 'desktop.ini']
    if not remaining:
        di = path / 'desktop.ini'
        if di.exists() and not DRY_RUN:
            di.unlink()
        if not DRY_RUN:
            try:
                path.rmdir()
                stats['deleted_empty'] += 1
            except OSError:
                pass


# ============================================================================
# STEP 1: SAR_CARD_bbox -> SAR_CARD (flatten period subdirs)
# ============================================================================
print(f"\n{'='*70}")
print("STEP 1: SAR_CARD_bbox -> SAR_CARD")
print(f"{'='*70}")

old_card = SAT_ROOT / 'SAR_CARD_bbox'
new_card = SAT_ROOT / 'SAR_CARD_new'  # temporary name to avoid collision

if old_card.exists():
    if not DRY_RUN:
        new_card.mkdir(parents=True, exist_ok=True)

    for city_dir in sorted(old_card.iterdir()):
        if not city_dir.is_dir() or city_dir.name == 'desktop.ini':
            continue
        city = city_dir.name
        dst_city = new_card / city
        if not DRY_RUN:
            dst_city.mkdir(parents=True, exist_ok=True)
        city_moved = 0

        for item in sorted(city_dir.iterdir()):
            if not item.is_dir():
                continue

            # keep derived subdirs as-is
            if item.name in DERIVED_SUBDIRS:
                dst_derived = dst_city / item.name
                if not DRY_RUN:
                    if item.exists():
                        shutil.copytree(str(item), str(dst_derived), dirs_exist_ok=True)
                print(f"  {city}/{item.name}/ -> kept as subdir")
                continue

            # period subdir (prebattle, postbattle, biweekly_YYYY-MM-DD, etc)
            # flatten all TIFs to city root
            tifs = list(item.glob('*.tif'))
            for tif in tifs:
                dst = dst_city / tif.name
                if dst.exists():
                    stats['skipped'] += 1
                    continue
                if safe_move(tif, dst):
                    city_moved += 1
                    stats['moved'] += 1

        if city_moved > 0:
            print(f"  {city:25s}: {city_moved} TIFs flattened")
else:
    print("  SAR_CARD_bbox not found, skipping")


# ============================================================================
# STEP 2: SAR_SLC_bbox -> SAR_COH (flatten pair subdirs, rename TIFs)
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2: SAR_SLC_bbox -> SAR_COH")
print(f"{'='*70}")

old_slc = SAT_ROOT / 'SAR_SLC_bbox'
new_coh = SAT_ROOT / 'SAR_COH'

if old_slc.exists():
    if not DRY_RUN:
        new_coh.mkdir(parents=True, exist_ok=True)

    for city_dir in sorted(old_slc.iterdir()):
        if not city_dir.is_dir() or city_dir.name == 'desktop.ini':
            continue
        city = city_dir.name
        dst_city = new_coh / city
        if not DRY_RUN:
            dst_city.mkdir(parents=True, exist_ok=True)
        city_moved = 0

        for period_dir in sorted(city_dir.iterdir()):
            if not period_dir.is_dir():
                continue

            # keep coherence_baseline as subdir
            if period_dir.name in DERIVED_SUBDIRS:
                dst_derived = dst_city / period_dir.name
                if not DRY_RUN:
                    if period_dir.exists():
                        shutil.copytree(str(period_dir), str(dst_derived), dirs_exist_ok=True)
                print(f"  {city}/{period_dir.name}/ -> kept as subdir")
                continue

            # period subdir: may contain pair subdirs (20220208_20220304/) or direct TIFs
            # check for pair subdirs first
            pair_dirs = [d for d in period_dir.iterdir()
                         if d.is_dir() and re.match(r'\d{8}_\d{8}', d.name)]
            direct_tifs = list(period_dir.glob('*.tif'))

            # handle pair subdirs
            for pair_dir in pair_dirs:
                for tif in pair_dir.glob('*.tif'):
                    # rename: VV_coherence_2022-02-08_2022-03-04.tif
                    #      -> {city}_COH_VV_20220208_20220304.tif
                    m = re.match(r'(VV|VH)_coherence_(\d{4})-(\d{2})-(\d{2})_(\d{4})-(\d{2})-(\d{2})\.tif',
                                 tif.name)
                    if m:
                        pol = m.group(1)
                        d1 = f"{m.group(2)}{m.group(3)}{m.group(4)}"
                        d2 = f"{m.group(5)}{m.group(6)}{m.group(7)}"
                        new_name = f"{city}_COH_{pol}_{d1}_{d2}.tif"
                    else:
                        # fallback: prepend city name
                        new_name = f"{city}_{tif.name}"

                    dst = dst_city / new_name
                    if dst.exists():
                        stats['skipped'] += 1
                        continue
                    if safe_move(tif, dst):
                        city_moved += 1
                        stats['moved'] += 1
                        stats['renamed'] += 1

            # handle direct TIFs (e.g. postbattle/ has TIFs directly)
            for tif in direct_tifs:
                m = re.match(r'(VV|VH)_coherence_(\d{4})-(\d{2})-(\d{2})_(\d{4})-(\d{2})-(\d{2})\.tif',
                             tif.name)
                if m:
                    pol = m.group(1)
                    d1 = f"{m.group(2)}{m.group(3)}{m.group(4)}"
                    d2 = f"{m.group(5)}{m.group(6)}{m.group(7)}"
                    new_name = f"{city}_COH_{pol}_{d1}_{d2}.tif"
                else:
                    new_name = f"{city}_{tif.name}"

                dst = dst_city / new_name
                if dst.exists():
                    stats['skipped'] += 1
                    continue
                if safe_move(tif, dst):
                    city_moved += 1
                    stats['moved'] += 1
                    stats['renamed'] += 1

        if city_moved > 0:
            print(f"  {city:25s}: {city_moved} COH TIFs flattened + renamed")
else:
    print("  SAR_SLC_bbox not found, skipping")


# ============================================================================
# STEP 3: multispectral -> MS (flatten, delete .SAFE dirs)
# ============================================================================
print(f"\n{'='*70}")
print("STEP 3: multispectral -> MS")
print(f"{'='*70}")

old_ms = SAT_ROOT / 'multispectral'
new_ms = SAT_ROOT / 'MS'

if old_ms.exists():
    if not DRY_RUN:
        new_ms.mkdir(parents=True, exist_ok=True)

    # move metadata dir first
    old_meta = old_ms / 'metadata'
    if old_meta.exists():
        dst_meta = new_ms / 'metadata'
        if not DRY_RUN:
            shutil.copytree(str(old_meta), str(dst_meta), dirs_exist_ok=True)
        print(f"  metadata/ -> MS/metadata/")

    for city_dir in sorted(old_ms.iterdir()):
        if not city_dir.is_dir() or city_dir.name in ('metadata', 'temp', 'desktop.ini'):
            continue
        city = city_dir.name
        dst_city = new_ms / city
        if not DRY_RUN:
            dst_city.mkdir(parents=True, exist_ok=True)
        city_moved = 0
        city_safe_deleted = 0

        for item in sorted(city_dir.iterdir()):
            if not item.is_dir():
                # root-level file in city dir (shouldn't happen, but handle)
                if item.suffix == '.tif':
                    dst = dst_city / item.name
                    if not dst.exists():
                        if safe_move(item, dst):
                            city_moved += 1
                            stats['moved'] += 1
                continue

            # keep derived subdirs
            if item.name in DERIVED_SUBDIRS:
                dst_derived = dst_city / item.name
                if not DRY_RUN:
                    if item.exists():
                        shutil.copytree(str(item), str(dst_derived), dirs_exist_ok=True)
                print(f"  {city}/{item.name}/ -> kept as subdir")
                continue

            # period subdir: flatten TIFs, delete .SAFE dirs
            for sub_item in sorted(item.iterdir()):
                if sub_item.is_dir():
                    if sub_item.name.endswith('.SAFE'):
                        # delete .SAFE directory
                        if safe_delete(sub_item, is_dir=True):
                            city_safe_deleted += 1
                            stats['deleted_safe'] += 1
                    continue

                if sub_item.suffix == '.tif':
                    dst = dst_city / sub_item.name
                    if dst.exists():
                        stats['skipped'] += 1
                        continue
                    if safe_move(sub_item, dst):
                        city_moved += 1
                        stats['moved'] += 1

        report = f"  {city:25s}: {city_moved} TIFs flattened"
        if city_safe_deleted > 0:
            report += f", {city_safe_deleted} .SAFE dirs deleted"
        if city_moved > 0 or city_safe_deleted > 0:
            print(report)
else:
    print("  multispectral not found, skipping")


# ============================================================================
# STEP 4: Move SAR_SLC/metadata -> SAR_METADATA
# ============================================================================
print(f"\n{'='*70}")
print("STEP 4: SAR_SLC/metadata -> SAR_METADATA")
print(f"{'='*70}")

old_slc_meta = SAT_ROOT / 'SAR_SLC' / 'metadata'
new_sar_meta = SAT_ROOT / 'SAR_METADATA'

if old_slc_meta.exists():
    if not DRY_RUN:
        shutil.copytree(str(old_slc_meta), str(new_sar_meta), dirs_exist_ok=True)
    n_files = sum(1 for f in old_slc_meta.iterdir() if f.is_file())
    print(f"  Moved {n_files} metadata files")
else:
    print("  SAR_SLC/metadata not found")


# ============================================================================
# STEP 5: Rename new dirs to final names
# ============================================================================
print(f"\n{'='*70}")
print("STEP 5: RENAME TO FINAL STRUCTURE")
print(f"{'='*70}")

if not DRY_RUN:
    # Delete old empty dirs
    for old_dir_name in ['SAR_CARD', 'SAR_CARD_polygon', 'SAR_SLC_polygon']:
        old_dir = SAT_ROOT / old_dir_name
        if old_dir.exists():
            shutil.rmtree(str(old_dir))
            print(f"  Deleted: {old_dir_name}/")
            stats['deleted_dirs'] += 1

    # Rename SAR_CARD_new -> SAR_CARD (after deleting old SAR_CARD)
    if new_card.exists():
        final_card = SAT_ROOT / 'SAR_CARD'
        if final_card.exists():
            shutil.rmtree(str(final_card))
        new_card.rename(final_card)
        print(f"  SAR_CARD_new -> SAR_CARD")

    # SAR_COH already has correct name

    # Delete old source dirs (SAR_CARD_bbox, SAR_SLC_bbox, SAR_SLC, multispectral)
    # Only after confirming new dirs exist and have content
    for old_name, new_name in [('SAR_CARD_bbox', 'SAR_CARD'), ('SAR_SLC_bbox', 'SAR_COH'),
                                ('multispectral', 'MS'), ('SAR_SLC', 'SAR_METADATA')]:
        old_dir = SAT_ROOT / old_name
        new_dir = SAT_ROOT / new_name
        if old_dir.exists() and new_dir.exists():
            # verify new dir has content
            new_files = list(new_dir.rglob('*.tif')) + list(new_dir.rglob('*.json'))
            if len(new_files) > 0:
                shutil.rmtree(str(old_dir))
                print(f"  Deleted old: {old_name}/ ({len(new_files)} files verified in {new_name}/)")
                stats['deleted_dirs'] += 1
            else:
                print(f"  WARNING: {new_name}/ is empty, NOT deleting {old_name}/")
else:
    print("  DRY RUN: would rename SAR_CARD_new -> SAR_CARD")
    print("  DRY RUN: would delete SAR_CARD_bbox, SAR_SLC_bbox, SAR_SLC, SAR_CARD_polygon, SAR_SLC_polygon, multispectral")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("MIGRATION SUMMARY")
print(f"{'='*70}")
print(f"  TIFs moved:       {stats['moved']}")
print(f"  TIFs renamed:     {stats['renamed']} (COH: added city name)")
print(f"  .SAFE deleted:    {stats['deleted_safe']}")
print(f"  Empty dirs gone:  {stats['deleted_empty']}")
print(f"  Old dirs deleted: {stats['deleted_dirs']}")
print(f"  Skipped (exist):  {stats['skipped']}")
print(f"  Errors:           {len(stats['errors'])}")

if stats['errors']:
    print(f"\n  ERRORS:")
    for err in stats['errors'][:20]:
        print(f"    {err}")

if DRY_RUN:
    print(f"\n  DRY RUN — no changes made. Run without --dry-run to execute.")

print(f"\n  Final structure:")
print(f"    satellite/SAR_CARD/{{city}}/{{city}}_CARD_{{pol}}_{{date}}.tif")
print(f"    satellite/SAR_COH/{{city}}/{{city}}_COH_{{pol}}_{{date1}}_{{date2}}.tif")
print(f"    satellite/MS/{{city}}/{{city}}_S2_{{date}}_{{band}}_{{res}}.tif")
print(f"    satellite/SAR_METADATA/*.json")
print(f"    satellite/sentinel_1_orbits/  (unchanged)")
print(f"    satellite/temporal_products/  (unchanged)")
print(f"    satellite/landuse_classification/  (unchanged)")
print(f"{'='*70}")
