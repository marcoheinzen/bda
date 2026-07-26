# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
dl_sync.py
Pre-download sync: scene selection + disk diff + download plans.
Extracted from Cell DL-SYNC.

Notebook usage:
    from dl_sync import run as run_dl_sync
    DL_SYNC = run_dl_sync(
        cities_df=cities_df,
        sar_filtered=sar_filtered,
        raw_slc_zip=RAW_SLC_ZIP,
        card_zip_dir=CARD_ZIP_DIR,
        ms_zip_dir=MS_ZIP_DIR,
        sar_card_dir=SAR_CARD_DIR,
        outputs_dir=OUTPUTS_DIR,
        card_tracker_file=CARD_TRACKER_FILE,
        slc_plan_file=SLC_PLAN_FILE,
        card_plan_file=CARD_PLAN_FILE,
        n_prebattle_baseline=N_PREBATTLE_BASELINE,
        short_conflict_threshold_days=SHORT_CONFLICT_THRESHOLD_DAYS,
        min_slc_size=MIN_SLC_SIZE,
        min_card_size=MIN_CARD_SIZE,
    )
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from scene_selection import (
    select_scene_pair,
    select_biweekly_chain,
    select_prebattle_baseline,
)


def run(cities_df, sar_filtered, raw_slc_zip, card_zip_dir, ms_zip_dir,
        sar_card_dir, outputs_dir, card_tracker_file, slc_plan_file, card_plan_file,
        n_prebattle_baseline=5, short_conflict_threshold_days=30,
        min_slc_size=3e9, min_card_size=1e9):
    """
    Args:
        cities_df:          pd.DataFrame - ALL cities (for protection set)
        sar_filtered:       pd.DataFrame - filtered cities (download targets)
        raw_slc_zip:        Path - RAW_SLC_ZIP
        card_zip_dir:       Path - CARD_ZIP_DIR
        ms_zip_dir:         Path - MS_ZIP_DIR
        sar_card_dir:       Path - SAR_CARD_DIR
        outputs_dir:        Path - OUTPUTS_DIR
        card_tracker_file:  Path - CARD_TRACKER_FILE
        slc_plan_file:      Path - SLC_PLAN_FILE
        card_plan_file:     Path - CARD_PLAN_FILE
        n_prebattle_baseline: int
        short_conflict_threshold_days: int
        min_slc_size:       float
        min_card_size:      float

    Returns:
        DL_SYNC dict
    """
    raw_slc_zip = Path(raw_slc_zip)
    card_zip_dir = Path(card_zip_dir)
    ms_zip_dir = Path(ms_zip_dir)
    sar_card_dir = Path(sar_card_dir)
    outputs_dir = Path(outputs_dir)
    card_tracker_file = Path(card_tracker_file)
    slc_plan_file = Path(slc_plan_file)
    card_plan_file = Path(card_plan_file)

    print("=" * 80)
    print("CELL DL-SYNC: PRE-DOWNLOAD SYNC")
    print("=" * 80)
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  N_PREBATTLE_BASELINE: {n_prebattle_baseline}")
    print(f"  SHORT_CONFLICT_THRESHOLD: {short_conflict_threshold_days}d")

    # Ensure output dirs exist
    for d in [raw_slc_zip, card_zip_dir, ms_zip_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"  cities_df:          {len(cities_df)} cities (ALL tiers)")
    print(f"  sar_filtered:       {len(sar_filtered)} cities (filtered tiers)")

    # =========================================================================
    # STEP 1A: SCAN SLC ZIPS ON DISK
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 1A: SCANNING SLC ZIPS ON DISK")
    print(f"{'='*80}")
    print(f"  RAW_SLC_ZIP: {raw_slc_zip}")

    def parse_slc_name(filename):
        name = filename.replace('.SAFE.zip', '').replace('.zip', '')
        m = re.search(r'(S1[ABC])_IW_SLC.*?_(\d{8})T(\d{6})_(\d{8})T(\d{6})_(\d{6})_([A-F0-9]+)', name)
        if m:
            sat = m.group(1)
            abs_orbit = int(m.group(6))
            if sat == 'S1B':
                rel_orbit = ((abs_orbit - 27) % 175) + 1
            else:
                rel_orbit = ((abs_orbit - 73) % 175) + 1
            return {
                'satellite': sat,
                'date': f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:8]}",
                'date_raw': m.group(2),
                'orbit_number': abs_orbit,
                'relative_orbit': rel_orbit,
                'base_name': name,
                'filename': filename
            }
        return None

    slc_on_disk = {}
    slc_disk_total_gb = 0

    for f in raw_slc_zip.iterdir():
        if f.suffix == '.zip' and f.stat().st_size > min_slc_size:
            parsed = parse_slc_name(f.name)
            if parsed:
                parsed['size_gb'] = f.stat().st_size / 1e9
                parsed['path'] = str(f)
                slc_on_disk[parsed['base_name']] = parsed
                slc_disk_total_gb += parsed['size_gb']

    slc_disk_by_date = defaultdict(list)
    for name, info in slc_on_disk.items():
        slc_disk_by_date[info['date']].append(info)

    print(f"  Found {len(slc_on_disk)} valid SLC zips ({slc_disk_total_gb:.0f} GB)")
    print(f"  Unique dates: {len(slc_disk_by_date)}")

    # =========================================================================
    # STEP 1B: SCAN CARD ZIPS ON DISK
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 1B: SCANNING CARD ZIPS ON DISK")
    print(f"{'='*80}")
    print(f"  CARD_ZIP_DIR: {card_zip_dir}")

    card_zips_on_disk = {}

    for f in card_zip_dir.iterdir():
        if f.suffix == '.zip' and f.stat().st_size > min_card_size:
            name = f.name.replace('.zip', '')
            m = re.search(r'(\d{8})T(\d{6})', name)
            if m:
                date_str = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
                card_zips_on_disk[name] = {
                    'date': date_str,
                    'size_gb': f.stat().st_size / 1e9,
                    'path': str(f),
                    'filename': f.name
                }

    print(f"  Found {len(card_zips_on_disk)} valid CARD zips")

    # =========================================================================
    # STEP 1C: SCAN CARD TIFS ON DISK
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 1C: SCANNING CARD TIFS ON DISK")
    print(f"{'='*80}")
    print(f"  SAR_CARD_DIR: {sar_card_dir}")

    from product_scan import scan_card_tifs_for_sync
    card_tifs_on_disk = scan_card_tifs_for_sync(sar_card_dir)

    card_tif_cities = len(card_tifs_on_disk)
    card_tif_total = sum(len(dates) for dates in card_tifs_on_disk.values())
    print(f"  Found CARD TIFs for {card_tif_cities} cities, {card_tif_total} city-dates")

    # =========================================================================
    # STEP 1D: SCAN MS ZIPS ON DISK
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 1D: SCANNING MS ZIPS ON DISK")
    print(f"{'='*80}")
    print(f"  MS_ZIP_DIR: {ms_zip_dir}")

    ms_zips_on_disk = {}

    if ms_zip_dir.exists():
        for f in ms_zip_dir.iterdir():
            if f.suffix == '.zip':
                ms_zips_on_disk[f.stem] = {
                    'size_gb': f.stat().st_size / 1e9,
                    'path': str(f),
                    'filename': f.name
                }

    print(f"  Found {len(ms_zips_on_disk)} MS zips")

    # =========================================================================
    # STEP 2: LOAD EXISTING TRACKERS
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: LOADING EXISTING TRACKERS")
    print(f"{'='*80}")

    slc_tracker_file = raw_slc_zip / 'download_tracker.json'
    slc_rejected = defaultdict(set)

    if slc_tracker_file.exists():
        with open(slc_tracker_file, 'r') as f:
            old_slc_tracker = json.load(f)
        print(f"  SLC tracker: {len(old_slc_tracker.get('downloads', {}))} entries")
        for scene_name, info in old_slc_tracker.get('downloads', {}).items():
            if info.get('status') == 'rejected':
                city = info.get('city', '')
                if city:
                    slc_rejected[city].add(scene_name)
        if slc_rejected:
            total_rej = sum(len(v) for v in slc_rejected.values())
            print(f"  SLC rejected: {total_rej} city-scene pairs")
    else:
        old_slc_tracker = {}
        print("  SLC tracker: not found (fresh start)")

    if card_tracker_file.exists():
        with open(card_tracker_file, 'r') as f:
            old_card_tracker = json.load(f)
        print(f"  CARD tracker: {len(old_card_tracker)} entries")
    else:
        old_card_tracker = {}
        print("  CARD tracker: not found (fresh start)")

    ms_tracker_file = outputs_dir / 'ms_download_tracking_v2.json'
    if ms_tracker_file.exists():
        with open(ms_tracker_file, 'r') as f:
            old_ms_tracker = json.load(f)
        print(f"  MS tracker: {len(old_ms_tracker)} entries")
    else:
        old_ms_tracker = {}
        print("  MS tracker: not found")

    # =========================================================================
    # STEP 3: SCENE SELECTION - ALL TIERS (PROTECTION SET)
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: SCENE SELECTION - ALL TIERS (PROTECTION SET)")
    print(f"{'='*80}")

    protection_set = set()

    def scene_base_name(scene):
        name = scene.get('name', scene.get('Name', ''))
        if name.endswith('.SAFE'):
            name = name[:-5]
        return name

    def select_city_scenes(city_name, row, rejected_set=None):
        scenes = []
        recommended_orbit = row.get('recommended_orbit')
        common_orbits = row.get('common_orbits', [])
        is_ongoing = row.get('conflict_ongoing', False)
        if not is_ongoing:
            battle_end = row.get('battle_end')
            is_ongoing = pd.isna(battle_end) if hasattr(pd, 'isna') else (battle_end is None)

        battle_start = row.get('battle_start')
        battle_end = row.get('battle_end')
        if is_ongoing:
            conflict_days = (datetime.now() - pd.Timestamp(battle_start).to_pydatetime()).days if pd.notna(battle_start) else 999
        elif pd.notna(battle_start) and pd.notna(battle_end):
            conflict_days = (pd.Timestamp(battle_end) - pd.Timestamp(battle_start)).days
        else:
            conflict_days = 999
        is_short_conflict = conflict_days < short_conflict_threshold_days
        if is_short_conflict:
            print(f"    {city_name}: SHORT CONFLICT ({conflict_days}d < {short_conflict_threshold_days}d) - using all available scenes")

        orbits_to_try = [recommended_orbit] if recommended_orbit else []
        for o in common_orbits:
            if o != recommended_orbit and o not in orbits_to_try:
                orbits_to_try.append(o)

        selected_orbit = None
        exclude = list(rejected_set) if rejected_set else None

        for try_orbit in orbits_to_try:
            try_orbit = int(try_orbit) if try_orbit else None
            if not try_orbit:
                continue

            try:
                pre_result = select_scene_pair(city_name, 'prebattle', orbit=try_orbit, exclude_scenes=exclude)
                if pre_result[0] is None:
                    continue
                pre_s1, pre_s2, pre_baseline = pre_result
            except Exception as e:
                continue

            post_pair = None
            if not is_ongoing:
                try:
                    post_result = select_scene_pair(city_name, 'postbattle', orbit=try_orbit, exclude_scenes=exclude)
                    if post_result[0] is None:
                        continue
                    post_s1, post_s2, post_baseline = post_result
                    post_pair = (post_s1, post_s2)
                except Exception:
                    continue

            try:
                chain = select_biweekly_chain(city_name, orbit=try_orbit, use_all_scenes=is_short_conflict)
            except Exception:
                chain = []

            if len(chain) < 2:
                continue

            try:
                baseline_chain = select_prebattle_baseline(city_name, n_scenes=n_prebattle_baseline, orbit=try_orbit)
            except Exception:
                baseline_chain = []

            selected_orbit = try_orbit

            for s in [pre_s1, pre_s2]:
                scenes.append((s, 'prebattle'))
            for s in chain:
                sd = s.get('scene_date', s.get('date', ''))
                if hasattr(sd, 'strftime'):
                    sd = sd.strftime('%Y-%m-%d')
                else:
                    sd = str(sd)[:10]
                scenes.append((s, f'biweekly_{sd}'))
            if post_pair:
                for s in post_pair:
                    scenes.append((s, 'postbattle'))
            for s in baseline_chain:
                scenes.append((s, 'prebattle_baseline'))

            break

        return scenes, selected_orbit

    all_cities_scenes = {}

    print(f"  Processing {len(cities_df)} cities for protection set...")
    for _, row in cities_df.iterrows():
        city = row['city']
        try:
            scenes, orbit = select_city_scenes(city, row, rejected_set=None)
            if scenes:
                seen = set()
                deduped = []
                for s, purpose in scenes:
                    bn = scene_base_name(s)
                    if bn and bn not in seen:
                        seen.add(bn)
                        deduped.append((s, purpose))
                        protection_set.add(bn)
                all_cities_scenes[city] = {'orbit': orbit, 'scenes': deduped}
        except Exception as e:
            print(f"    {city}: ERROR in scene selection: {e}")

    print(f"\n  Protection set: {len(protection_set)} unique SLC scenes across {len(all_cities_scenes)} cities")

    # =========================================================================
    # STEP 4: SCENE SELECTION - FILTERED TIERS (DOWNLOAD TARGETS)
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 4: SCENE SELECTION - FILTERED TIERS (DOWNLOAD TARGETS)")
    print(f"{'='*80}")

    filtered_cities = list(sar_filtered['city'].values)
    print(f"  Processing {len(filtered_cities)} filtered cities...")

    download_targets = {}
    city_orbit_map = {}

    for _, row in sar_filtered.iterrows():
        city = row['city']
        tier = int(row['tier'])
        rejected_set = slc_rejected.get(city, set())

        try:
            scenes, orbit = select_city_scenes(city, row, rejected_set=rejected_set)
        except Exception as e:
            print(f"    {city}: ERROR: {e}")
            continue

        if not scenes or not orbit:
            print(f"    {city}: WARNING - no valid orbit found")
            continue

        city_orbit_map[city] = orbit

        seen = set()
        deduped = []
        for s, purpose in scenes:
            bn = scene_base_name(s)
            if bn and bn not in seen:
                seen.add(bn)
                deduped.append((s, purpose))

        scene_entries = []
        for s, purpose in deduped:
            bn = scene_base_name(s)
            date_str = s.get('date', '')[:10]
            sid = s.get('id', '')
            src = s.get('source', 'copernicus')

            status = 'missing'
            zip_path = None

            if bn in slc_on_disk:
                status = 'on_disk'
                zip_path = slc_on_disk[bn].get('path')
            elif f"{bn}.SAFE" in slc_on_disk:
                status = 'on_disk'
                zip_path = slc_on_disk[f"{bn}.SAFE"].get('path')
            elif date_str in slc_disk_by_date:
                for disk_info in slc_disk_by_date[date_str]:
                    if disk_info.get('relative_orbit') == orbit:
                        status = 'on_disk'
                        zip_path = disk_info.get('path')
                        break

            if bn in rejected_set:
                status = 'rejected'

            scene_entries.append({
                'name': bn,
                'date': date_str,
                'id': sid,
                'source': src,
                'purpose': purpose,
                'status': status,
                'zip_path': zip_path
            })

        download_targets[city] = {
            'orbit': orbit,
            'tier': tier,
            'scenes': scene_entries
        }

    # =========================================================================
    # STEP 5: SLC DOWNLOAD PLAN
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 5: SLC DOWNLOAD PLAN")
    print(f"{'='*80}")

    slc_download_queue = []
    slc_summary = {}

    for city, data in sorted(download_targets.items()):
        on_disk = sum(1 for s in data['scenes'] if s['status'] == 'on_disk')
        missing = [s for s in data['scenes'] if s['status'] == 'missing']
        rejected = sum(1 for s in data['scenes'] if s['status'] == 'rejected')
        total = len(data['scenes'])

        slc_summary[city] = {
            'orbit': data['orbit'],
            'tier': data['tier'],
            'total': total,
            'on_disk': on_disk,
            'missing': len(missing),
            'rejected': rejected
        }

        status_str = "COMPLETE" if len(missing) == 0 else f"{len(missing)} to download"
        rej_str = f" ({rejected} rejected)" if rejected > 0 else ""
        print(f"  {city:<22} T{data['tier']} orbit {data['orbit']:3d}: {on_disk:3d}/{total:3d} on disk -> {status_str}{rej_str}")

        for s in missing:
            slc_download_queue.append({
                'name': s['name'],
                'id': s['id'],
                'city': city,
                'purpose': s['purpose'],
                'source': s['source']
            })

    seen_names = set()
    unique_slc_queue = []
    for item in slc_download_queue:
        if item['name'] not in seen_names:
            seen_names.add(item['name'])
            unique_slc_queue.append(item)

    print(f"\n  Total SLC to download: {len(unique_slc_queue)} unique scenes ({len(slc_download_queue)} before dedup)")

    # =========================================================================
    # STEP 6: CARD DOWNLOAD PLAN
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 6: CARD DOWNLOAD PLAN")
    print(f"{'='*80}")

    card_plan = {}

    for city, data in sorted(download_targets.items()):
        orbit = data['orbit']
        tier = data['tier']
        dates = {}

        for s in data['scenes']:
            date_str = s['date']
            if not date_str:
                continue

            tif_exists = False
            orbit_match = True
            city_card = card_tifs_on_disk.get(city, {})
            if date_str in city_card:
                pols = city_card[date_str].get('pols', [])
                tif_orbit = city_card[date_str].get('orbit')
                if 'VV' in pols and 'VH' in pols:
                    tif_exists = True
                    if tif_orbit is not None and tif_orbit != orbit:
                        print(f"    WARNING: {city} {date_str} CARD TIF orbit mismatch: tif=o{tif_orbit:03d} plan=o{orbit:03d}")
                        orbit_match = False

            if tif_exists:
                status = 'tif_exists'
            else:
                status = 'missing'

            if date_str not in dates:
                dates[date_str] = {
                    'purpose': s['purpose'],
                    'slc_scene': s['name'],
                    'status': status,
                    'tif_exists': tif_exists
                }

        card_plan[city] = {
            'orbit': orbit,
            'tier': tier,
            'dates': dates
        }

        existing = sum(1 for d in dates.values() if d['tif_exists'])
        missing_count = sum(1 for d in dates.values() if not d['tif_exists'])
        total = len(dates)
        status_str = "COMPLETE" if missing_count == 0 else f"{missing_count} to process"
        print(f"  {city:<22} T{tier} orbit {orbit:3d}: {existing:3d}/{total:3d} TIFs exist -> {status_str}")

    card_total_missing = sum(
        sum(1 for d in data['dates'].values() if not d['tif_exists'])
        for data in card_plan.values()
    )
    print(f"\n  Total CARD city-dates to process: {card_total_missing}")

    # =========================================================================
    # STEP 7: WRITE PLANS TO DISK
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 7: WRITING PLANS")
    print(f"{'='*80}")

    slc_plan = {
        'created': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'tier_selection': list(sar_filtered['tier'].unique()),
        'n_prebattle_baseline': n_prebattle_baseline,
        'cities': {},
        'protection_set': sorted(list(protection_set)),
        'download_queue': unique_slc_queue,
        'summary': slc_summary,
        'city_orbits': {k: int(v) for k, v in city_orbit_map.items()}
    }

    for city, data in download_targets.items():
        slc_plan['cities'][city] = {
            'orbit': data['orbit'],
            'tier': data['tier'],
            'scenes': {s['name']: {
                'date': s['date'],
                'purpose': s['purpose'],
                'status': s['status'],
                'id': s['id'],
                'source': s['source']
            } for s in data['scenes']}
        }

    with open(slc_plan_file, 'w') as f:
        json.dump(slc_plan, f, indent=2, default=str)
    print(f"  SLC plan: {slc_plan_file}")
    print(f"    {len(slc_plan['cities'])} cities, {len(unique_slc_queue)} to download, {len(protection_set)} in protection set")

    card_plan_out = {
        'created': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'cities': card_plan
    }

    with open(card_plan_file, 'w') as f:
        json.dump(card_plan_out, f, indent=2, default=str)
    print(f"  CARD plan: {card_plan_file}")
    print(f"    {len(card_plan)} cities, {card_total_missing} city-dates to process")

    # =========================================================================
    # STEP 8: BUILD DL_SYNC DICT
    # =========================================================================
    DL_SYNC = {
        'slc_plan': slc_plan,
        'card_plan': card_plan_out,
        'slc_on_disk': slc_on_disk,
        'card_zips_on_disk': card_zips_on_disk,
        'card_tifs_on_disk': dict(card_tifs_on_disk),
        'ms_zips_on_disk': ms_zips_on_disk,
        'protection_set': protection_set,
        'city_orbit_map': city_orbit_map,
        'slc_rejected': dict(slc_rejected),
        'download_queue': unique_slc_queue,
        'download_targets': download_targets,
        'timestamp': datetime.now().isoformat()
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("DL-SYNC SUMMARY")
    print(f"{'='*80}")

    total_slc_needed = sum(d['total'] for d in slc_summary.values())
    total_slc_disk = sum(d['on_disk'] for d in slc_summary.values())
    total_slc_missing = sum(d['missing'] for d in slc_summary.values())
    total_slc_rejected = sum(d['rejected'] for d in slc_summary.values())

    print(f"\n  SLC SCENES:")
    print(f"    Cities (filtered):   {len(download_targets)}")
    print(f"    Cities (all/protect):{len(all_cities_scenes)}")
    print(f"    Needed (filtered):   {total_slc_needed}")
    print(f"    On disk:             {total_slc_disk}")
    print(f"    To download:         {len(unique_slc_queue)} unique ({total_slc_missing} city-refs)")
    print(f"    Rejected:            {total_slc_rejected}")
    print(f"    Protection set:      {len(protection_set)} scenes")

    print(f"\n  CARD PRODUCTS:")
    print(f"    Cities:              {len(card_plan)}")
    print(f"    City-dates to proc:  {card_total_missing}")

    print(f"\n  DISK:")
    print(f"    SLC zips:            {len(slc_on_disk)} ({slc_disk_total_gb:.0f} GB)")
    print(f"    CARD zips:           {len(card_zips_on_disk)}")
    print(f"    CARD TIFs:           {card_tif_total} city-dates")
    print(f"    MS zips:             {len(ms_zips_on_disk)}")

    print(f"\n  FILES:")
    print(f"    {slc_plan_file}")
    print(f"    {card_plan_file}")
    print(f"\n  DL_SYNC dict returned for downstream cells")
    print("=" * 80)

    return DL_SYNC
