# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
dl_reconcile.py
Tracker reconciler: fixes NB02 trackers vs actual disk state.
Extracted from Cell DL-RECONCILE.

Notebook usage:
    from dl_reconcile import run as run_dl_reconcile
    RECONCILE_RESULTS = run_dl_reconcile(
        raw_slc_zip=RAW_SLC_ZIP,
        sar_card_dir=SAR_CARD_DIR,
        ms_zip_dir=MS_ZIP_DIR,
        outputs_dir=OUTPUTS_DIR,
        insar_tracker_file=INSAR_TRACKER_FILE,
        card_tracker_file=CARD_TRACKER_FILE,
        nb02_audit=NB02_AUDIT,
        fix_mode=FIX_MODE,
        backup_before_fix=BACKUP_BEFORE_FIX,
    )
"""

import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from scene_selection import load_discovered_scenes


def run(raw_slc_zip, sar_card_dir, ms_zip_dir, outputs_dir,
        insar_tracker_file, card_tracker_file,
        nb02_audit=None, fix_mode=True, backup_before_fix=True):
    """
    Args:
        raw_slc_zip:        Path - RAW_SLC_ZIP
        sar_card_dir:       Path - SAR_CARD_DIR
        ms_zip_dir:         Path - MS_ZIP_DIR
        outputs_dir:        Path - OUTPUTS_DIR
        insar_tracker_file: Path - INSAR_TRACKER_FILE
        card_tracker_file:  Path - CARD_TRACKER_FILE
        nb02_audit:         dict or None - NB02_AUDIT from dl_audit.run()
        fix_mode:           bool
        backup_before_fix:  bool

    Returns:
        RECONCILE_RESULTS dict
    """
    raw_slc_zip = Path(raw_slc_zip)
    sar_card_dir = Path(sar_card_dir)
    ms_zip_dir = Path(ms_zip_dir)
    outputs_dir = Path(outputs_dir)
    insar_tracker_file = Path(insar_tracker_file)
    card_tracker_file = Path(card_tracker_file)

    print("=" * 80)
    print("CELL DL-RECONCILE: TRACKER RECONCILER")
    print("=" * 80)
    print(f"  FIX_MODE: {fix_mode}")
    print(f"  BACKUP_BEFORE_FIX: {backup_before_fix}")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    total_issues = 0
    total_fixes = 0

    def backup_file(path):
        if backup_before_fix and path.exists():
            bak = path.parent / f"{path.stem}_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
            shutil.copy2(path, bak)
            print(f"    Backup: {bak.name}")

    # =========================================================================
    # LOAD AUDIT DATA
    # =========================================================================
    if nb02_audit is not None:
        slc_inventory = nb02_audit['slc_inventory']
        print(f"  Using NB02_AUDIT: {len(slc_inventory)} SLC zips")
    else:
        print("  WARNING: NB02_AUDIT not provided - building minimal inventories...")
        slc_inventory = {}
        if raw_slc_zip.exists():
            for f in raw_slc_zip.iterdir():
                if f.suffix == '.zip' and f.stat().st_size > 3e9:
                    name = f.name.replace('.SAFE.zip', '').replace('.zip', '')
                    slc_inventory[name] = {'filename': f.name, 'size_gb': f.stat().st_size / 1e9, 'path': str(f)}

    # CARD TIF scan for reconciliation (flat structure)
    from product_scan import scan_card_tifs_for_sync
    card_tifs_on_disk = scan_card_tifs_for_sync(sar_card_dir)

    # Build SLC lookup by base_name prefix (first 40 chars) for fuzzy matching
    slc_prefixes = {}
    for name in slc_inventory:
        slc_prefixes[name[:40]] = name

    def find_zip_on_disk(scene_name):
        base = scene_name[:-5] if scene_name.endswith('.SAFE') else scene_name
        if base in slc_inventory:
            return True
        if base[:40] in slc_prefixes:
            return True
        return False

    # =========================================================================
    # 1. SLC DOWNLOAD TRACKER
    # =========================================================================
    print(f"\n{'='*80}")
    print("1. SLC DOWNLOAD TRACKER")
    print(f"{'='*80}")

    DL_TRACKER_FILE = raw_slc_zip / 'download_tracker.json'
    dl_issues = {'tracker_no_disk': [], 'disk_no_tracker': [], 'status_mismatch': []}

    if DL_TRACKER_FILE.exists():
        with open(DL_TRACKER_FILE) as f:
            dl_tracker = json.load(f)

        dl_downloads = dl_tracker.get('downloads', {})
        print(f"  Tracker entries: {len(dl_downloads)}")
        print(f"  Zips on disk:    {len(slc_inventory)}")

        # Check tracker entries vs disk
        for base_name, entry in dl_downloads.items():
            status = entry.get('status', '')
            on_disk = base_name in slc_inventory or base_name[:40] in slc_prefixes
            if status == 'complete' and not on_disk:
                dl_issues['tracker_no_disk'].append(base_name)
            elif status != 'complete' and on_disk:
                dl_issues['status_mismatch'].append(base_name)

        # Check disk vs tracker
        tracked_names = set(dl_downloads.keys())
        for name in slc_inventory:
            if name not in tracked_names:
                if not any(name[:40] == t[:40] for t in tracked_names):
                    dl_issues['disk_no_tracker'].append(name)

        print(f"\n  Issues:")
        print(f"    Tracker=complete but zip missing: {len(dl_issues['tracker_no_disk'])}")
        print(f"    Tracker!=complete but zip exists: {len(dl_issues['status_mismatch'])}")
        print(f"    Zip on disk but not tracked:      {len(dl_issues['disk_no_tracker'])}")
        total_issues += sum(len(v) for v in dl_issues.values())

        if dl_issues['tracker_no_disk']:
            print(f"\n    Tracker=complete but missing:")
            for n in dl_issues['tracker_no_disk'][:10]:
                print(f"      {n[:70]}")

        if dl_issues['disk_no_tracker']:
            print(f"\n    Untracked zips on disk:")
            for n in dl_issues['disk_no_tracker'][:10]:
                print(f"      {n[:70]}")
            if len(dl_issues['disk_no_tracker']) > 10:
                print(f"      ... and {len(dl_issues['disk_no_tracker'])-10} more")

        # FIX
        if fix_mode and (dl_issues['disk_no_tracker'] or dl_issues['status_mismatch']):
            backup_file(DL_TRACKER_FILE)
            fixes = 0

            for name in dl_issues['disk_no_tracker']:
                info = slc_inventory[name]
                dl_tracker['downloads'][name] = {
                    'status': 'complete',
                    'size_gb': info.get('size_gb', 0),
                    'path': info.get('path', ''),
                    'added_by': 'reconciler',
                    'timestamp': datetime.now().isoformat(),
                }
                fixes += 1

            for name in dl_issues['status_mismatch']:
                dl_tracker['downloads'][name]['status'] = 'complete'
                dl_tracker['downloads'][name]['fixed_by'] = 'reconciler'
                fixes += 1

            dl_tracker['last_reconciled'] = datetime.now().isoformat()
            with open(DL_TRACKER_FILE, 'w') as f:
                json.dump(dl_tracker, f, indent=2)
            print(f"\n    FIXED: {fixes} entries updated in download_tracker.json")
            total_fixes += fixes
    else:
        print(f"  download_tracker.json not found at {DL_TRACKER_FILE}")

    # =========================================================================
    # 2. CARD DOWNLOAD TRACKER
    # =========================================================================
    print(f"\n{'='*80}")
    print("2. CARD DOWNLOAD TRACKER")
    print(f"{'='*80}")

    card_issues = {'tracker_no_disk': [], 'disk_no_tracker': []}

    if card_tracker_file.exists():
        with open(card_tracker_file) as f:
            card_tracker = json.load(f)
        print(f"  Tracker entries: {len(card_tracker)}")
        print(f"  CARD cities on disk: {len(card_tifs_on_disk)}")

        # Check tracker entries vs disk
        for tracker_key, entry in card_tracker.items():
            if entry.get('status') not in ('success', 'exists'):
                continue
            files = entry.get('files', [])
            city = tracker_key.split('_')[0]
            all_exist = True
            for rel_path in files:
                full_path = sar_card_dir / rel_path
                if not full_path.exists():
                    all_exist = False
                    break
            if not all_exist:
                card_issues['tracker_no_disk'].append(tracker_key)

        # Check disk vs tracker (flat scan)
        tracked_card_keys = set(card_tracker.keys())
        for city, dates in card_tifs_on_disk.items():
            for date_str in dates:
                key = f"{city}_{date_str}"
                if key not in tracked_card_keys:
                    card_issues['disk_no_tracker'].append({
                        'key': key, 'city': city, 'date': date_str
                    })

        print(f"\n  Issues:")
        print(f"    Tracker=success but TIFs missing: {len(card_issues['tracker_no_disk'])}")
        print(f"    TIFs on disk but not tracked:     {len(card_issues['disk_no_tracker'])}")
        total_issues += len(card_issues['tracker_no_disk']) + len(card_issues['disk_no_tracker'])

        if card_issues['tracker_no_disk']:
            print(f"\n    Tracker=success but missing:")
            for k in card_issues['tracker_no_disk'][:10]:
                print(f"      {k}")

        if card_issues['disk_no_tracker']:
            print(f"\n    Untracked CARD TIFs:")
            for item in card_issues['disk_no_tracker'][:10]:
                print(f"      {item['key']}")

        # FIX ghost entries: tracker=success but TIFs gone -> reset to zip_ready
        # NB03b v14 disk-is-truth will re-extract; reconciler just fixes tracker status
        if fix_mode and card_issues['tracker_no_disk']:
            backup_file(card_tracker_file)
            card_fixes = 0
            for tracker_key in card_issues['tracker_no_disk']:
                entry = card_tracker[tracker_key]
                zip_path = entry.get('zip_path', '')
                if zip_path and Path(zip_path).exists():
                    entry['status'] = 'zip_ready'
                else:
                    entry['status'] = 'zip_ready'  # NB03b will skip if zip not found
                entry['_reset_by_reconciler'] = datetime.now().isoformat()
                entry['_reset_reason'] = 'ghost: TIFs missing on disk'
                card_fixes += 1
            with open(card_tracker_file, 'w') as f:
                json.dump(card_tracker, f, indent=2)
            print(f"\n    FIXED: {card_fixes} ghost entries reset to zip_ready")
            total_fixes += card_fixes

        # NOTE: disk_no_tracker CARD TIFs are NOT auto-added to tracker.
        # They may be stale dates, wrong tiles, or bonus cities from extraction.
        if card_issues['disk_no_tracker']:
            print(f"\n    NOT auto-adding {len(card_issues['disk_no_tracker'])} untracked CARD TIFs")
            print(f"    (may be stale/bonus - use NB03e PRUNE to clean up)")
    else:
        print(f"  card_download_tracker.json not found at {card_tracker_file}")

    # =========================================================================
    # 3. MS DOWNLOAD TRACKER
    # =========================================================================
    print(f"\n{'='*80}")
    print("3. MS DOWNLOAD TRACKER")
    print(f"{'='*80}")

    MS_TRACKING_V2 = outputs_dir / 'ms_download_tracking_v2.json'
    ms_issues = {'tracker_no_disk': [], 'disk_no_tracker': []}

    if MS_TRACKING_V2.exists():
        with open(MS_TRACKING_V2) as f:
            ms_tracker = json.load(f)

        targets = ms_tracker.get('targets', ms_tracker)
        print(f"  Tracker entries: {len(targets)}")

        # Check tracker entries vs disk
        for target_key, entry in targets.items():
            if entry.get('status') != 'success':
                continue
            city = entry.get('city', '')
            period = entry.get('period', '')
            scene_name = entry.get('scene_name', '')

            if not city or not scene_name:
                continue

            # check ms_zip_dir for the zip file
            zip_name = scene_name if scene_name.endswith('.zip') else f"{scene_name}.zip"
            zip_path = ms_zip_dir / zip_name
            found = zip_path.exists()
            if not found:
                # fuzzy: check by date prefix
                scene_prefix = scene_name[:40]
                for zf in ms_zip_dir.iterdir() if ms_zip_dir.exists() else []:
                    if zf.name.startswith(scene_prefix) and zf.suffix == '.zip':
                        found = True
                        break

            if not found:
                ms_issues['tracker_no_disk'].append({
                    'key': target_key, 'city': city, 'period': period, 'scene': scene_name
                })

        # Check disk vs tracker
        tracked_scenes = set()
        for entry in targets.values():
            sn = entry.get('scene_name', '')
            if sn:
                tracked_scenes.add(sn)
                tracked_scenes.add(sn.replace('.zip', ''))

        # scan ms_zip_dir for untracked zips
        if ms_zip_dir.exists():
            for zf in sorted(ms_zip_dir.iterdir()):
                if zf.suffix != '.zip':
                    continue
                base = zf.stem
                if base not in tracked_scenes:
                    ms_issues['disk_no_tracker'].append({
                        'scene': base, 'filename': zf.name
                    })

        print(f"\n  Issues:")
        print(f"    Tracker=success but SAFE missing: {len(ms_issues['tracker_no_disk'])}")
        print(f"    SAFE on disk but not tracked:     {len(ms_issues['disk_no_tracker'])}")
        total_issues += len(ms_issues['tracker_no_disk']) + len(ms_issues['disk_no_tracker'])

        if ms_issues['tracker_no_disk']:
            print(f"\n    Tracker=success but missing:")
            for item in ms_issues['tracker_no_disk'][:10]:
                print(f"      {item['city']:20s} {item['period']:12s} {item['scene'][:50]}")
            if len(ms_issues['tracker_no_disk']) > 10:
                print(f"      ... and {len(ms_issues['tracker_no_disk'])-10} more")

        if ms_issues['disk_no_tracker']:
            print(f"\n    Untracked MS zips on disk:")
            for item in ms_issues['disk_no_tracker'][:10]:
                print(f"      {item['filename']}")
            if len(ms_issues['disk_no_tracker']) > 10:
                print(f"      ... and {len(ms_issues['disk_no_tracker'])-10} more")

        # NOTE: disk_no_tracker MS zips are NOT auto-added to tracker.
        # They may be wrong-tile downloads or leftovers from before NB02c fix.
        # Use DL-PRUNE-ZIPS to clean up surplus MS zips instead.
        if ms_issues['disk_no_tracker']:
            print(f"\n    NOT auto-fixing {len(ms_issues['disk_no_tracker'])} untracked MS zips")
            print(f"    (may be wrong-tile - use DL-PRUNE-ZIPS to clean up)")
    else:
        print(f"  ms_download_tracking_v2.json not found at {MS_TRACKING_V2}")

    # =========================================================================
    # 4. INSAR TRACKER: SCENE ZIP AVAILABILITY
    # =========================================================================
    print(f"\n{'='*80}")
    print("4. INSAR TRACKER: SCENE ZIP AVAILABILITY")
    print(f"{'='*80}")

    insar_issues = {'scenes_missing_zips': [], 'cities_reprocessable': []}

    if insar_tracker_file.exists():
        with open(insar_tracker_file) as f:
            insar_tracker = json.load(f)

        cities_data = insar_tracker.get('cities', {})
        print(f"  Tracker cities: {len(cities_data)}")

        print(f"\n  {'City':25s} {'Pre':>5s} {'Post':>5s} {'Cross':>5s} {'Biwk':>5s} {'Zip':>5s}")
        print(f"  {'-'*70}")

        for city_name in sorted(cities_data.keys()):
            city_data = cities_data[city_name]
            city_scenes = set()
            city_missing = set()

            for phase in ['prebattle', 'postbattle', 'crossbattle']:
                runs = city_data.get(phase, {})
                for key, run_data in runs.items():
                    for scene_name in run_data.get('scenes', []):
                        base = scene_name[:-5] if scene_name.endswith('.SAFE') else scene_name
                        city_scenes.add(base)
                        if not find_zip_on_disk(scene_name):
                            city_missing.add(base)

            bw_data = city_data.get('biweekly', {})
            for pk, pi in bw_data.get('pairs', {}).items():
                for sn in pi.get('scenes', []):
                    base = sn[:-5] if sn.endswith('.SAFE') else sn
                    city_scenes.add(base)
                    if not find_zip_on_disk(sn):
                        city_missing.add(base)

            mo_data = city_data.get('monthly', {})
            for pk, pi in mo_data.get('pairs', {}).items():
                for sn in pi.get('scenes', []):
                    base = sn[:-5] if sn.endswith('.SAFE') else sn
                    city_scenes.add(base)
                    if not find_zip_on_disk(sn):
                        city_missing.add(base)

            pre_ok = any(v.get('status') == 'success' for v in city_data.get('prebattle', {}).values())
            post_ok = any(v.get('status') == 'success' for v in city_data.get('postbattle', {}).values())
            cross_ok = any(v.get('status') == 'success' for v in city_data.get('crossbattle', {}).values())
            bw_ok = bw_data.get('completed_pairs', 0) > 0

            ok_str = lambda x: "OK" if x else "-"
            zip_str = f"{len(city_scenes)-len(city_missing)}/{len(city_scenes)}" if city_scenes else "-"

            print(f"  {city_name:25s} {ok_str(pre_ok):>5s} {ok_str(post_ok):>5s} {ok_str(cross_ok):>5s} {ok_str(bw_ok):>5s} {zip_str:>5s}")

            if city_missing:
                insar_issues['scenes_missing_zips'].append({
                    'city': city_name, 'missing': list(city_missing), 'total': len(city_scenes)
                })

            pre_failed = any(v.get('status') in ('failed', 'attempted')
                             for v in city_data.get('prebattle', {}).values())
            if pre_failed and not pre_ok:
                discovered = None
                try:
                    discovered = load_discovered_scenes(city_name)
                except:
                    pass
                if discovered:
                    orbit = city_data.get('orbit') or discovered.get('recommended_orbit')
                    if orbit:
                        orbit_data = discovered.get('orbits', {}).get(str(orbit), {})
                        pre_scenes = orbit_data.get('pre_scenes', [])
                        available = 0
                        for s in pre_scenes:
                            name = s.get('name', s.get('Name', ''))
                            if find_zip_on_disk(name):
                                available += 1
                        if available >= 2:
                            insar_issues['cities_reprocessable'].append({
                                'city': city_name,
                                'available_pre_zips': available,
                                'total_pre_scenes': len(pre_scenes),
                            })

        print(f"\n  Cities with missing zips: {len(insar_issues['scenes_missing_zips'])}")
        total_issues += len(insar_issues['scenes_missing_zips'])

        if insar_issues['scenes_missing_zips']:
            print(f"\n  Missing zip details:")
            for item in insar_issues['scenes_missing_zips']:
                print(f"    {item['city']:25s} missing {len(item['missing'])}/{item['total']} zips")

        if insar_issues['cities_reprocessable']:
            print(f"\n  Cities with failed prebattle but zips available for retry:")
            for item in insar_issues['cities_reprocessable']:
                print(f"    {item['city']:25s} {item['available_pre_zips']}/{item['total_pre_scenes']} prebattle zips on disk")
    else:
        print(f"  insar_processing_tracker.json not found at {insar_tracker_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("RECONCILIATION SUMMARY")
    print(f"{'='*80}")

    print(f"\n  SLC DL tracker:")
    print(f"    Tracker=complete but zip missing: {len(dl_issues.get('tracker_no_disk', []))}")
    print(f"    Zip on disk but not tracked:      {len(dl_issues.get('disk_no_tracker', []))}")

    print(f"\n  CARD tracker:")
    print(f"    Tracker=success but TIF missing:  {len(card_issues.get('tracker_no_disk', []))}")
    print(f"    TIF on disk but not tracked:      {len(card_issues.get('disk_no_tracker', []))}")

    print(f"\n  MS tracker:")
    print(f"    Tracker=success but SAFE missing: {len(ms_issues.get('tracker_no_disk', []))}")
    print(f"    SAFE on disk but not tracked:     {len(ms_issues.get('disk_no_tracker', []))}")

    print(f"\n  InSAR tracker:")
    print(f"    Cities with missing scene zips:   {len(insar_issues.get('scenes_missing_zips', []))}")
    print(f"    Cities reprocessable (have zips):  {len(insar_issues.get('cities_reprocessable', []))}")

    print(f"\n  Total issues found: {total_issues}")
    if fix_mode:
        print(f"  Total fixes applied: {total_fixes}")
    else:
        print(f"  FIX_MODE = False (report only). Set FIX_MODE = True to apply fixes.")

    RECONCILE_RESULTS = {
        'dl_issues': dl_issues,
        'card_issues': card_issues,
        'ms_issues': ms_issues,
        'insar_issues': insar_issues,
        'total_issues': total_issues,
        'total_fixes': total_fixes,
        'timestamp': datetime.now().isoformat(),
    }
    print(f"\n  Returned: RECONCILE_RESULTS dict")
    print(f"{'='*80}")

    return RECONCILE_RESULTS
