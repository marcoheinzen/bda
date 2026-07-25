# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
dl_prune_zips.py
Zip pruning: determines which SLC/CARD/MS zips are safe to delete.

DESIGN:
  1. NO tier filtering - always checks ALL cities across ALL tiers
  2. NO scene selector - uses discovery metadata directly
  3. Time window: keeps zips within pre_battle_start to post_battle_end per city
  4. Cross-city aware: a zip needed by ANY city is protected
  5. Wrong-tile: zips not in any city's discovery metadata

Protection built from:
  - SAR_METADATA_DIR/*_scene_metadata.json (all SAR scenes per city)
  - MS_METADATA_DIR/*_ms_scene_metadata.json (all MS scenes per city)
  - InSAR processing tracker (all processed pairs)

Notebook usage:
    from dl_prune_zips import run as run_prune_zips
    PRUNE_RESULTS = run_prune_zips(
        raw_slc_zip=RAW_SLC_ZIP,
        card_zip_dir=CARD_ZIP_DIR,
        ms_zip_dir=MS_ZIP_DIR,
        sar_metadata_dir=SAR_METADATA_DIR,
        ms_metadata_dir=MS_METADATA_DIR,
        cities_dir=CITIES_DIR,
        card_tracker_file=CARD_TRACKER_FILE,
        insar_tracker_file=INSAR_TRACKER_FILE,
        nb02_audit=NB02_AUDIT,
        dry_run=False,
    )
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def run(raw_slc_zip, card_zip_dir, ms_zip_dir,
        sar_metadata_dir, ms_metadata_dir, cities_dir,
        card_tracker_file, insar_tracker_file,
        nb02_audit=None, dry_run=False, min_slc_size=3e9, verbose=True,
        prune_selector='ALL'):

    raw_slc_zip = Path(raw_slc_zip)
    card_zip_dir = Path(card_zip_dir)
    ms_zip_dir = Path(ms_zip_dir)
    sar_metadata_dir = Path(sar_metadata_dir)
    ms_metadata_dir = Path(ms_metadata_dir)
    cities_dir = Path(cities_dir)
    card_tracker_file = Path(card_tracker_file)
    insar_tracker_file = Path(insar_tracker_file)

    # normalize prune_selector
    if isinstance(prune_selector, str) and prune_selector.upper() == 'ALL':
        prune_categories = None  # delete everything
    elif isinstance(prune_selector, (list, tuple)):
        prune_categories = set(s.lower() for s in prune_selector)
    else:
        prune_categories = None

    print("=" * 80)
    print("DL-PRUNE-ZIPS: ZIP PRUNING (ALL CITIES, METADATA-BASED)")
    print("=" * 80)
    print(f"  DRY_RUN: {dry_run}")
    print(f"  PRUNE_SELECTOR: {prune_selector}")
    print(f"  SAR_METADATA_DIR: {sar_metadata_dir}")
    print(f"  MS_METADATA_DIR:  {ms_metadata_dir}")

    # =========================================================================
    # 1. DISK INVENTORIES
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 1: DISK INVENTORIES")
    print(f"{'='*80}")

    # SLC zips
    if nb02_audit and 'slc_inventory' in nb02_audit:
        slc_inventory = nb02_audit['slc_inventory']
        print(f"  SLC zips (from audit): {len(slc_inventory)}")
    else:
        slc_inventory = {}
        if raw_slc_zip.exists():
            for f in raw_slc_zip.iterdir():
                if f.suffix != '.zip' or f.stat().st_size < min_slc_size:
                    continue
                name = f.name.replace('.SAFE.zip', '').replace('.zip', '')
                m = re.search(r'(S1[ABC])_IW_SLC.*?_(\d{8})T(\d{6}).*?_(\d{6})_', f.name)
                if m:
                    slc_inventory[name] = {
                        'satellite': m.group(1),
                        'date': f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:8]}",
                        'date_raw': m.group(2),
                        'orbit_abs': int(m.group(4)),
                        'filename': f.name,
                        'size_gb': f.stat().st_size / 1e9,
                        'path': str(f),
                    }
        print(f"  SLC zips (scanned): {len(slc_inventory)}")

    # CARD zips
    card_on_disk = {}
    if card_zip_dir.exists():
        for f in card_zip_dir.iterdir():
            if f.suffix == '.zip':
                card_on_disk[f.stem] = {'filename': f.name, 'size_gb': f.stat().st_size / 1e9, 'path': str(f)}
    print(f"  CARD zips: {len(card_on_disk)}")

    # MS zips
    ms_on_disk = {}
    if ms_zip_dir.exists():
        for f in ms_zip_dir.iterdir():
            if f.suffix == '.zip':
                ms_on_disk[f.stem] = {'filename': f.name, 'size_gb': f.stat().st_size / 1e9, 'path': str(f)}
    print(f"  MS zips: {len(ms_on_disk)}")

    # =========================================================================
    # 2. BUILD SLC PROTECTION SET FROM SAR METADATA
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: SLC PROTECTION SET (SAR METADATA, ALL CITIES)")
    print(f"{'='*80}")

    slc_needed = set()
    slc_scene_to_cities = defaultdict(set)
    sar_meta_count = 0

    if sar_metadata_dir.exists():
        for meta_file in sorted(sar_metadata_dir.glob('*_scene_metadata.json')):
            city_name = meta_file.stem.replace('_scene_metadata', '')
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
            except Exception:
                continue
            sar_meta_count += 1

            for orbit_key, orbit_data in meta.get('orbits', {}).items():
                for list_key in ('pre_scenes', 'post_scenes', 'battle_scenes'):
                    for s in orbit_data.get(list_key, []):
                        name = s.get('name', s.get('Name', ''))
                        stem = name.replace('.SAFE', '').replace('.zip', '')
                        if stem:
                            slc_needed.add(stem)
                            slc_scene_to_cities[stem].add(city_name)

        print(f"  SAR metadata files: {sar_meta_count} cities")
        print(f"  SLC scenes in metadata: {len(slc_needed)}")
    else:
        print(f"  FATAL: SAR_METADATA_DIR not found: {sar_metadata_dir}")
        print(f"  ABORTING.")
        return {'aborted': True, 'reason': 'sar_metadata_missing'}

    # add InSAR tracker scenes
    if insar_tracker_file.exists():
        with open(insar_tracker_file) as f:
            insar_tracker = json.load(f)
        before = len(slc_needed)
        for city_name, city_data in insar_tracker.get('cities', {}).items():
            for phase in ['prebattle', 'postbattle', 'crossbattle', 'prebattle_baseline']:
                runs = city_data.get(phase, {})
                if not isinstance(runs, dict):
                    continue
                for key, run_data in runs.items():
                    if not isinstance(run_data, dict):
                        continue
                    for sn in run_data.get('scenes', []):
                        base = sn.replace('.SAFE', '')
                        slc_needed.add(base)
                        slc_scene_to_cities[base].add(city_name)
            for section in ['biweekly', 'monthly']:
                section_data = city_data.get(section, {})
                if not isinstance(section_data, dict):
                    continue
                for pk, pi in section_data.get('pairs', {}).items():
                    if not isinstance(pi, dict):
                        continue
                    for sn in pi.get('scenes', []):
                        base = sn.replace('.SAFE', '')
                        slc_needed.add(base)
                        slc_scene_to_cities[base].add(city_name)
        print(f"  Added {len(slc_needed) - before} from InSAR tracker")

    print(f"  TOTAL SLC protected: {len(slc_needed)}")

    # =========================================================================
    # 3. SLC MATCH + CATEGORIZE SURPLUS
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: SLC SURPLUS")
    print(f"{'='*80}")

    slc_matched = set()
    for name in slc_inventory:
        if name in slc_needed:
            slc_matched.add(name)
        else:
            prefix = name[:40]
            for needed in slc_needed:
                if needed[:40] == prefix:
                    slc_matched.add(name)
                    break

    slc_surplus = set(slc_inventory.keys()) - slc_matched
    slc_surplus_gb = sum(slc_inventory[n]['size_gb'] for n in slc_surplus)

    print(f"  On disk:   {len(slc_inventory)} ({sum(v['size_gb'] for v in slc_inventory.values()):.0f} GB)")
    print(f"  Protected: {len(slc_matched)} ({sum(slc_inventory[n]['size_gb'] for n in slc_matched):.0f} GB)")
    print(f"  Surplus:   {len(slc_surplus)} ({slc_surplus_gb:.0f} GB)")

    if slc_surplus:
        print(f"\n  SLC SURPLUS ({len(slc_surplus)}):")
        for name in sorted(slc_surplus, key=lambda n: slc_inventory[n]['date']):
            info = slc_inventory[name]
            # determine reason
            cities_using = slc_scene_to_cities.get(name, set())
            if cities_using:
                reason = 'in_metadata_but_not_matched'
                city_str = ','.join(sorted(cities_using))
            else:
                reason = 'not_in_any_metadata'
                city_str = '-'
            if verbose or len(slc_surplus) <= 30:
                print(f"    {info['date']}  {info['size_gb']:5.1f}GB  {reason:30s}  cities={city_str}  {name[:80]}")
        if not verbose and len(slc_surplus) > 30:
            print(f"    ... {len(slc_surplus)} total (use verbose=True to show all)")

    # =========================================================================
    # 4. MS PROTECTION SET FROM MS METADATA
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 4: MS PROTECTION SET (MS METADATA, ALL CITIES)")
    print(f"{'='*80}")

    ms_needed = set()  # scene stems
    ms_scene_to_cities = defaultdict(set)
    ms_tile_to_cities = defaultdict(set)
    # per-city date ranges: city -> (min_date_str, max_date_str)
    ms_city_date_range = {}
    # per-city tiles
    ms_city_tiles = defaultdict(set)
    ms_meta_count = 0

    if ms_metadata_dir.exists():
        for meta_file in sorted(ms_metadata_dir.glob('*_ms_scene_metadata.json')):
            city_name = meta_file.stem.replace('_ms_scene_metadata', '')
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
            except Exception:
                continue
            ms_meta_count += 1

            city_dates = []
            # MS metadata has windows: pre_window, post_window, battle_window, etc.
            for window_key in meta:
                window = meta[window_key]
                if not isinstance(window, dict):
                    continue
                for s in window.get('scenes', []):
                    name = s.get('name', '')
                    stem = name.replace('.SAFE', '').replace('.zip', '')
                    if stem:
                        ms_needed.add(stem)
                        ms_scene_to_cities[stem].add(city_name)
                        tile_m = re.search(r'_T(\d{2}[A-Z]{3})_', stem)
                        if tile_m:
                            ms_tile_to_cities[tile_m.group(1)].add(city_name)
                            ms_city_tiles[city_name].add(tile_m.group(1))
                        date_m = re.search(r'_(\d{8})T\d{6}_', stem)
                        if date_m:
                            city_dates.append(date_m.group(1))
            if city_dates:
                ms_city_date_range[city_name] = (min(city_dates), max(city_dates))

        print(f"  MS metadata files: {ms_meta_count} cities")
        print(f"  MS scenes in metadata: {len(ms_needed)}")
        print(f"  Tiles in metadata: {sorted(ms_tile_to_cities.keys())}")
    else:
        print(f"  WARNING: MS_METADATA_DIR not found: {ms_metadata_dir}")

    # build per-tile date ranges (union of all cities using that tile)
    ms_tile_date_range = {}
    for tile, cities_set in ms_tile_to_cities.items():
        tile_min = None
        tile_max = None
        for c in cities_set:
            if c in ms_city_date_range:
                cmin, cmax = ms_city_date_range[c]
                if tile_min is None or cmin < tile_min:
                    tile_min = cmin
                if tile_max is None or cmax > tile_max:
                    tile_max = cmax
        if tile_min and tile_max:
            ms_tile_date_range[tile] = (tile_min, tile_max)

    # =========================================================================
    # 5. MS MATCH + SURPLUS
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 5: MS SURPLUS")
    print(f"{'='*80}")

    ms_matched = set()
    for name in ms_on_disk:
        stem = name.replace('.SAFE', '')
        if stem in ms_needed:
            ms_matched.add(name)
        # NO fuzzy prefix for MS: tile ID (T37TCN) starts at char 38,
        # so 40-char prefix cuts mid-tile and matches wrong tiles.
        # MS matching must be EXACT on full scene name including tile.

    ms_surplus = set(ms_on_disk.keys()) - ms_matched
    ms_surplus_gb = sum(ms_on_disk[n]['size_gb'] for n in ms_surplus)

    print(f"  On disk:   {len(ms_on_disk)} ({sum(v['size_gb'] for v in ms_on_disk.values()):.1f} GB)")
    print(f"  Protected: {len(ms_matched)} ({sum(ms_on_disk[n]['size_gb'] for n in ms_matched):.1f} GB)")
    print(f"  Surplus:   {len(ms_surplus)} ({ms_surplus_gb:.1f} GB)")

    # classify each MS surplus scene
    def classify_ms_surplus(name):
        tile_m = re.search(r'_T(\d{2}[A-Z]{3})_', name)
        tile = tile_m.group(1) if tile_m else '?'
        date_m = re.search(r'_(\d{8})T\d{6}_', name)
        date_raw = date_m.group(1) if date_m else None
        date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}" if date_raw else '?'
        if tile not in ms_tile_to_cities:
            return tile, date_str, date_raw, 'wrong_tile', '-'
        city_str = ','.join(sorted(ms_tile_to_cities[tile]))
        if date_raw and tile in ms_tile_date_range:
            tmin, tmax = ms_tile_date_range[tile]
            if date_raw < tmin or date_raw > tmax:
                return tile, date_str, date_raw, 'date_excess', city_str
        return tile, date_str, date_raw, 'in_window_not_selected', city_str

    ms_surplus_classified = {}
    if ms_surplus:
        for name in ms_surplus:
            tile, date_str, date_raw, reason, city_str = classify_ms_surplus(name)
            ms_surplus_classified[name] = (tile, date_str, date_raw, reason, city_str)

        # count by category
        cat_counts = defaultdict(lambda: [0, 0.0])
        for name, (tile, date_str, date_raw, reason, city_str) in ms_surplus_classified.items():
            cat_counts[reason][0] += 1
            cat_counts[reason][1] += ms_on_disk[name]['size_gb']

        print(f"\n  MS SURPLUS BY CATEGORY:")
        for cat in ['wrong_tile', 'date_excess', 'in_window_not_selected']:
            if cat in cat_counts:
                print(f"    {cat:30s}: {cat_counts[cat][0]:4d} zips ({cat_counts[cat][1]:6.1f} GB)")

        # tile summary
        surplus_by_tile = defaultdict(list)
        for name, (tile, date_str, date_raw, reason, city_str) in ms_surplus_classified.items():
            surplus_by_tile[tile].append((name, date_str, reason))

        print(f"\n  MS SURPLUS TILE SUMMARY:")
        all_disk_tiles = set()
        for name in ms_on_disk:
            tile_m = re.search(r'_T(\d{2}[A-Z]{3})_', name)
            if tile_m:
                all_disk_tiles.add(tile_m.group(1))
        metadata_tiles = set(ms_tile_to_cities.keys())
        wrong_tiles = all_disk_tiles - metadata_tiles
        print(f"    Tiles on disk:       {sorted(all_disk_tiles)}")
        print(f"    Tiles in metadata:   {sorted(metadata_tiles)}")
        print(f"    Wrong tiles on disk: {sorted(wrong_tiles)}")
        for tile in sorted(surplus_by_tile.keys()):
            items = surplus_by_tile[tile]
            count = len(items)
            total_gb = sum(ms_on_disk[n]['size_gb'] for n, _, _ in items)
            reasons = set(r for _, _, r in items)
            reason_str = '+'.join(sorted(reasons))
            if tile in ms_tile_to_cities:
                city_str = ','.join(sorted(ms_tile_to_cities[tile]))
                if tile in ms_tile_date_range:
                    tmin, tmax = ms_tile_date_range[tile]
                    rng = f"  window={tmin[:4]}-{tmin[4:6]}-{tmin[6:]}..{tmax[:4]}-{tmax[4:6]}-{tmax[6:]}"
                else:
                    rng = ''
            else:
                city_str = '-'
                rng = ''
            print(f"    T{tile}: {count:3d} zips ({total_gb:6.1f} GB)  {reason_str:35s}  cities={city_str}{rng}")

        # full list (only verbose for date_excess and wrong_tile; in_window_not_selected only if verbose)
        print(f"\n  MS SURPLUS FULL LIST ({len(ms_surplus)}):")
        for name in sorted(ms_surplus):
            info = ms_on_disk[name]
            tile, date_str, date_raw, reason, city_str = ms_surplus_classified[name]
            show = verbose or reason in ('wrong_tile', 'date_excess') or len(ms_surplus) <= 30
            if show:
                print(f"    {date_str}  T{tile}  {info['size_gb']:.1f}GB  {reason:30s}  cities={city_str}  {name[:80]}")
        hidden = sum(1 for n in ms_surplus if not (verbose or ms_surplus_classified[n][3] in ('wrong_tile', 'date_excess') or len(ms_surplus) <= 30))
        if hidden > 0:
            print(f"    ... +{hidden} in_window_not_selected hidden (use verbose=True to show all)")

    # =========================================================================
    # 6. CARD ZIP PRUNING
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 6: CARD SURPLUS")
    print(f"{'='*80}")

    card_needed_stems = set()
    if card_tracker_file.exists():
        with open(card_tracker_file) as f:
            ct = json.load(f)
        for entry in ct.values():
            sn = entry.get('scene_name', '') if isinstance(entry, dict) else ''
            if sn:
                card_needed_stems.add(sn.replace('.zip', ''))
        print(f"  From CARD tracker: {len(card_needed_stems)} stems")

    card_surplus = set(card_on_disk.keys()) - card_needed_stems
    card_surplus_gb = sum(card_on_disk[n]['size_gb'] for n in card_surplus)

    print(f"  On disk:   {len(card_on_disk)} ({sum(v['size_gb'] for v in card_on_disk.values()):.1f} GB)")
    print(f"  Protected: {len(card_on_disk) - len(card_surplus)}")
    print(f"  Surplus:   {len(card_surplus)} ({card_surplus_gb:.1f} GB)")

    # =========================================================================
    # 7. FILTER SURPLUS BY PRUNE_SELECTOR
    # =========================================================================
    def should_prune(reason):
        if prune_categories is None:
            return True
        return reason.lower() in prune_categories

    # filter MS surplus by category
    ms_to_delete = set()
    ms_kept = set()
    for name in ms_surplus:
        if ms_surplus_classified:
            reason = ms_surplus_classified[name][3]
        else:
            reason = 'unknown'
        if should_prune(reason):
            ms_to_delete.add(name)
        else:
            ms_kept.add(name)

    # SLC surplus: always prune (selector only applies to MS categories)
    slc_to_delete = set(slc_surplus)

    # CARD surplus: always prune (selector only applies to MS categories)
    card_to_delete = set(card_surplus)

    if prune_categories is not None:
        print(f"\n{'='*80}")
        print(f"PRUNE SELECTOR FILTER: {sorted(prune_categories)}")
        print(f"{'='*80}")
        print(f"  MS to delete:   {len(ms_to_delete)} / {len(ms_surplus)} surplus")
        print(f"  MS kept:        {len(ms_kept)} (categories not in selector)")
        if ms_kept:
            kept_cats = defaultdict(int)
            for n in ms_kept:
                kept_cats[ms_surplus_classified[n][3]] += 1
            for cat, cnt in sorted(kept_cats.items()):
                print(f"    {cat}: {cnt}")
        print(f"  SLC to delete:  {len(slc_to_delete)} / {len(slc_surplus)} surplus")
        print(f"  CARD to delete: {len(card_to_delete)} / {len(card_surplus)} surplus")

    # =========================================================================
    # 8. EXECUTE
    # =========================================================================
    slc_deleted = 0
    slc_deleted_gb = 0
    ms_deleted = 0
    ms_deleted_gb = 0
    card_deleted = 0
    card_deleted_gb = 0

    if not dry_run:
        print(f"\n{'='*80}")
        print("EXECUTING PRUNE")
        print(f"{'='*80}")

        for name in sorted(slc_to_delete):
            p = Path(slc_inventory[name]['path'])
            if p.exists():
                try:
                    p.unlink()
                    slc_deleted += 1
                    slc_deleted_gb += slc_inventory[name]['size_gb']
                except Exception as ex:
                    print(f"  FAILED SLC: {p.name}: {ex}")
        if slc_deleted:
            print(f"  SLC deleted: {slc_deleted} ({slc_deleted_gb:.0f} GB)")

        for name in sorted(ms_to_delete):
            p = Path(ms_on_disk[name]['path'])
            if p.exists():
                try:
                    p.unlink()
                    ms_deleted += 1
                    ms_deleted_gb += ms_on_disk[name]['size_gb']
                except Exception as ex:
                    print(f"  FAILED MS: {p.name}: {ex}")
        if ms_deleted:
            print(f"  MS deleted: {ms_deleted} ({ms_deleted_gb:.1f} GB)")

        for name in sorted(card_to_delete):
            p = Path(card_on_disk[name]['path'])
            if p.exists():
                try:
                    p.unlink()
                    card_deleted += 1
                    card_deleted_gb += card_on_disk[name]['size_gb']
                except Exception as ex:
                    print(f"  FAILED CARD: {p.name}: {ex}")
        if card_deleted:
            print(f"  CARD deleted: {card_deleted} ({card_deleted_gb:.1f} GB)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    ms_to_delete_gb = sum(ms_on_disk[n]['size_gb'] for n in ms_to_delete)
    slc_to_delete_gb = sum(slc_inventory[n]['size_gb'] for n in slc_to_delete)
    card_to_delete_gb = sum(card_on_disk[n]['size_gb'] for n in card_to_delete)

    print(f"\n{'='*80}")
    if dry_run:
        print(f"DRY RUN - nothing deleted")
        print(f"  SLC surplus:  {len(slc_surplus)} ({slc_surplus_gb:.0f} GB)  -> {len(slc_to_delete)} to delete ({slc_to_delete_gb:.0f} GB)")
        print(f"  MS surplus:   {len(ms_surplus)} ({ms_surplus_gb:.1f} GB)  -> {len(ms_to_delete)} to delete ({ms_to_delete_gb:.1f} GB)")
        print(f"  CARD surplus: {len(card_surplus)} ({card_surplus_gb:.1f} GB)  -> {len(card_to_delete)} to delete ({card_to_delete_gb:.1f} GB)")
        total = slc_to_delete_gb + ms_to_delete_gb + card_to_delete_gb
        print(f"  TOTAL TO DELETE: {total:.0f} GB")
        if prune_categories is not None:
            kept_total = slc_surplus_gb + ms_surplus_gb + card_surplus_gb - total
            print(f"  KEPT (not in selector): {kept_total:.0f} GB")
    else:
        print(f"PRUNE COMPLETE")
        print(f"  SLC deleted:  {slc_deleted} ({slc_deleted_gb:.0f} GB)")
        print(f"  MS deleted:   {ms_deleted} ({ms_deleted_gb:.1f} GB)")
        print(f"  CARD deleted: {card_deleted} ({card_deleted_gb:.1f} GB)")
    print(f"{'='*80}")

    return {
        'slc_surplus': sorted(slc_surplus),
        'ms_surplus': sorted(ms_surplus),
        'ms_surplus_classified': ms_surplus_classified,
        'ms_to_delete': sorted(ms_to_delete),
        'ms_kept': sorted(ms_kept),
        'card_surplus': sorted(card_surplus),
        'slc_protected': len(slc_matched),
        'ms_protected': len(ms_matched),
        'slc_deleted': slc_deleted,
        'ms_deleted': ms_deleted,
        'card_deleted': card_deleted,
        'prune_selector': prune_selector,
        'dry_run': dry_run,
        'timestamp': datetime.now().isoformat(),
    }
