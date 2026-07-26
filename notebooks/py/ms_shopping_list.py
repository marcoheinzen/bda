# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
ms_shopping_list.py
Audit + fill gaps in per-city MS scene metadata.

Used by BOTH NB02a and NB02c. Reads existing *_ms_scene_metadata.json,
checks which windows are missing or have too few scenes, and runs
Copernicus API searches to fill gaps. Never overwrites existing windows.

SAR-independent windows (baseline, post_baseline) always run.
SAR-dependent windows (biweekly, prebattle_baseline, battle) only run
if SAR metadata exists on disk.

Notebook usage (NB02a or NB02c):
    from ms_shopping_list import audit_metadata, fill_gaps

    # 1. Audit: see what's missing
    audit = audit_metadata(MS_METADATA_DIR)

    # 2. Fill: run API searches for missing windows, update JSONs
    fill_gaps(
        cities_dir=CITIES_DIR,
        ms_metadata_dir=MS_METADATA_DIR,
        sar_metadata_dir=SAR_METADATA_DIR,
        tier_selection=TIER_SELECTION,
        city_selection=CITY_SELECTION,
    )
"""

import json
import sys
from pathlib import Path
from datetime import datetime


# =========================================================================
# REQUIRED WINDOWS PER CITY
# =========================================================================

REQUIRED_WINDOWS = {
    # window_key: (min_scenes, needs_sar, description)
    'pre_window':                    (2,  True,  'SAR-aligned pre-battle'),
    'post_window':                   (2,  True,  'SAR-aligned post-battle'),
    'baseline_window':               (3,  False, 'pre-battle winter baseline'),
    'post_baseline_window':          (3,  False, 'post-battle winter baseline'),
    'battle_window':                 (0,  True,  'during-battle (T0 only)'),
    'biweekly_window':               (0,  True,  'SAR-aligned biweekly'),
    'prebattle_baseline_window':     (0,  True,  'SAR-aligned prebattle baseline'),
}

# windows that should exist for ALL cities regardless of tier
MANDATORY_WINDOWS = ['pre_window', 'post_window', 'baseline_window', 'post_baseline_window']


# =========================================================================
# AUDIT: CHECK EXISTING METADATA
# =========================================================================

def audit_metadata(ms_metadata_dir, tier_selection=None, city_selection=None):
    """Audit all *_ms_scene_metadata.json files for completeness.

    Returns:
        dict: {city: {window: {status, n_scenes, min_needed, ...}}}
    """
    ms_metadata_dir = Path(ms_metadata_dir)

    print("=" * 70)
    print("MS SHOPPING LIST: METADATA AUDIT")
    print("=" * 70)

    if city_selection:
        if isinstance(city_selection, str):
            city_selection = [city_selection]

    audit = {}
    total_gaps = 0
    total_cities = 0

    for metadata_file in sorted(ms_metadata_dir.glob("*_ms_scene_metadata.json")):
        city_name = metadata_file.stem.replace('_ms_scene_metadata', '')

        if city_selection and city_name not in city_selection:
            continue

        with open(metadata_file) as f:
            meta = json.load(f)

        tier = int(meta.get('tier', 99))
        if tier_selection and tier not in tier_selection:
            continue

        total_cities += 1
        conflict_ongoing = meta.get('conflict_ongoing', False)
        city_audit = {'tier': tier, 'conflict_ongoing': conflict_ongoing, 'windows': {}}
        city_gaps = 0

        for window_key, (min_scenes, needs_sar, desc) in REQUIRED_WINDOWS.items():
            window = meta.get(window_key, {})
            n_scenes = len(window.get('scenes', []))

            # skip post_baseline for ongoing conflicts
            if window_key == 'post_baseline_window' and conflict_ongoing:
                city_audit['windows'][window_key] = {
                    'status': 'skipped', 'reason': 'ongoing_conflict',
                    'n_scenes': 0, 'min_needed': 0,
                }
                continue

            # determine if this window is mandatory for this city
            is_mandatory = window_key in MANDATORY_WINDOWS
            effective_min = min_scenes if is_mandatory else 0

            if window_key not in meta:
                status = 'missing'
                city_gaps += 1
            elif n_scenes < effective_min:
                status = 'insufficient'
                city_gaps += 1
            else:
                status = 'ok'

            city_audit['windows'][window_key] = {
                'status': status,
                'n_scenes': n_scenes,
                'min_needed': effective_min,
                'needs_sar': needs_sar,
                'description': desc,
            }

        audit[city_name] = city_audit
        total_gaps += city_gaps

    # print summary
    print(f"\n  Cities audited: {total_cities}")
    print(f"  Total gaps: {total_gaps}")

    if total_gaps > 0:
        print(f"\n  {'City':<25s} {'Tier':>4s}  {'pre':>4s} {'post':>4s} {'bl':>4s} {'pbl':>4s} {'bat':>4s} {'bw':>4s} {'preb':>4s}")
        print(f"  {'-'*25} {'-'*4}  {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
        for city_name, ca in sorted(audit.items()):
            if not any(w['status'] in ('missing', 'insufficient') for w in ca['windows'].values()):
                continue
            print(f"  {city_name:<25s} T{ca['tier']:>2d} ", end="")
            for wk in ['pre_window', 'post_window', 'baseline_window', 'post_baseline_window',
                        'battle_window', 'biweekly_window', 'prebattle_baseline_window']:
                w = ca['windows'].get(wk, {})
                s = w.get('status', '?')
                n = w.get('n_scenes', 0)
                if s == 'ok':
                    print(f" {n:>3d}+", end="")
                elif s == 'missing':
                    print(f"  ---", end="")
                elif s == 'insufficient':
                    print(f" {n:>3d}!", end="")
                elif s == 'skipped':
                    print(f"  skp", end="")
                else:
                    print(f"    ?", end="")
            print()

    # count cities that are fully complete for mandatory windows
    complete = sum(1 for ca in audit.values()
                   if all(ca['windows'].get(wk, {}).get('status', 'missing') in ('ok', 'skipped')
                          for wk in MANDATORY_WINDOWS))
    print(f"\n  Fully complete (mandatory windows): {complete}/{total_cities}")

    return audit


# =========================================================================
# FILL GAPS: RUN COPERNICUS API SEARCHES FOR MISSING WINDOWS
# =========================================================================

def fill_gaps(cities_dir, ms_metadata_dir, sar_metadata_dir,
              outputs_dir=None, tier_selection=None, city_selection=None,
              baseline_n_scenes=5, baseline_max_cloud=15,
              winter_months=None, dry_run=False):
    """Fill missing windows in existing ms_scene_metadata.json files.

    Only adds missing windows — never overwrites existing ones.
    SAR-independent searches (baseline, post_baseline) always run.
    SAR-dependent searches only run if SAR metadata exists.

    Args:
        cities_dir:       Path to CITIES_DIR
        ms_metadata_dir:  Path to MS_METADATA_DIR
        sar_metadata_dir: Path to SAR_METADATA_DIR
        tier_selection:   list of tier ints or None
        city_selection:   list of city names or None
        baseline_n_scenes: target scenes for baseline windows
        baseline_max_cloud: max cloud % for baseline
        winter_months:    months considered winter
        dry_run:          if True, print what would be done but don't search/save
    """
    # import and initialize ms_scene_discovery
    import importlib
    notebooks_dir = str(Path(ms_metadata_dir).parent.parent.parent / 'notebooks')
    if notebooks_dir not in sys.path:
        sys.path.insert(0, notebooks_dir)

    import ms_scene_discovery
    importlib.reload(ms_scene_discovery)

    # set ms_scene_discovery globals
    ms_scene_discovery._CITIES_DIR = Path(cities_dir)
    ms_scene_discovery._SAR_METADATA_DIR = Path(sar_metadata_dir)
    ms_scene_discovery._MS_METADATA_DIR = Path(ms_metadata_dir)
    if outputs_dir is not None:
        ms_scene_discovery._OUTPUTS_DIR = Path(outputs_dir)
    elif ms_scene_discovery._OUTPUTS_DIR is None:
        ms_scene_discovery._OUTPUTS_DIR = Path(ms_metadata_dir).parent.parent.parent / 'data' / 'outputs'
    ms_scene_discovery.BASELINE_N_SCENES = baseline_n_scenes
    ms_scene_discovery.POST_BASELINE_N_SCENES = baseline_n_scenes
    ms_scene_discovery.BASELINE_MAX_CLOUD = baseline_max_cloud
    ms_scene_discovery.WINTER_MONTHS = winter_months if winter_months else [10, 11, 12, 1, 2]

    ms_metadata_dir = Path(ms_metadata_dir)
    sar_metadata_dir = Path(sar_metadata_dir)

    if city_selection and isinstance(city_selection, str):
        city_selection = [city_selection]

    print("=" * 70)
    print("MS SHOPPING LIST: FILL GAPS")
    print("=" * 70)
    print(f"  baseline target:   {baseline_n_scenes} scenes")
    print(f"  baseline max cloud: {baseline_max_cloud}%")
    print(f"  winter months:     {ms_scene_discovery.WINTER_MONTHS}")
    print(f"  dry_run:           {dry_run}")

    # first audit
    audit = audit_metadata(ms_metadata_dir, tier_selection, city_selection)

    cities_with_gaps = {c: a for c, a in audit.items()
                        if any(w['status'] in ('missing', 'insufficient')
                               for w in a['windows'].values())}

    if not cities_with_gaps:
        print("\n  No gaps found. All metadata complete.")
        return audit

    print(f"\n  Cities with gaps: {len(cities_with_gaps)}")

    filled_count = 0
    failed_count = 0

    for ci, (city_name, city_audit) in enumerate(sorted(cities_with_gaps.items())):
        print(f"\n{'='*60}")
        print(f"  [{ci+1}/{len(cities_with_gaps)}] {city_name} (T{city_audit['tier']})")
        print(f"{'='*60}")

        # load existing metadata
        metadata_file = ms_metadata_dir / f"{city_name}_ms_scene_metadata.json"
        with open(metadata_file) as f:
            meta = json.load(f)

        # load city geometry + dates
        try:
            gdf, battle_start, battle_stop, conflict_ongoing, tier = \
                ms_scene_discovery.load_city_boundary_with_dates(city_name)
            city_geom = gdf.geometry.iloc[0]
            city_bounds = city_geom.bounds
        except Exception as e:
            print(f"    FAILED to load city boundary: {e}")
            failed_count += 1
            continue

        # check SAR metadata availability
        sar_meta_file = sar_metadata_dir / f"{city_name}_scene_metadata.json"
        has_sar = sar_meta_file.exists()
        if has_sar:
            print(f"    SAR metadata: available")
        else:
            print(f"    SAR metadata: NOT available (SAR-dependent windows will be skipped)")

        changed = False
        windows = city_audit['windows']

        # --- baseline_window (SAR-independent) ---
        w = windows.get('baseline_window', {})
        if w.get('status') in ('missing', 'insufficient'):
            print(f"\n    Searching: winter baseline (need >={REQUIRED_WINDOWS['baseline_window'][0]} scenes)")
            if dry_run:
                print(f"    [DRY RUN] Would search winter baseline")
            else:
                try:
                    scenes = ms_scene_discovery.search_ms_winter_baseline(
                        city_geom, city_bounds, battle_start, n_scenes=baseline_n_scenes
                    )
                    if scenes:
                        meta['baseline_window'] = {
                            'label': 'winter_baseline',
                            'cloud_threshold_used': float(min(s['cloud_cover'] for s in scenes)),
                            'scenes_found': len(scenes),
                            'scenes': scenes,
                        }
                        changed = True
                        print(f"    -> Added {len(scenes)} winter baseline scenes")
                    else:
                        print(f"    -> No winter baseline scenes found")
                except Exception as e:
                    print(f"    -> FAILED: {e}")

        # --- post_baseline_window (SAR-independent, skip if ongoing) ---
        w = windows.get('post_baseline_window', {})
        if w.get('status') in ('missing', 'insufficient') and not conflict_ongoing:
            print(f"\n    Searching: post-winter baseline (need >={REQUIRED_WINDOWS['post_baseline_window'][0]} scenes)")
            if dry_run:
                print(f"    [DRY RUN] Would search post-winter baseline")
            else:
                try:
                    scenes = ms_scene_discovery.search_ms_post_winter_baseline(
                        city_geom, city_bounds, battle_stop, n_scenes=baseline_n_scenes
                    )
                    if scenes:
                        meta['post_baseline_window'] = {
                            'label': 'post_winter_baseline',
                            'cloud_threshold_used': float(min(s['cloud_cover'] for s in scenes)),
                            'scenes_found': len(scenes),
                            'scenes': scenes,
                        }
                        changed = True
                        print(f"    -> Added {len(scenes)} post-winter baseline scenes")
                    else:
                        print(f"    -> No post-winter baseline scenes found")
                except Exception as e:
                    print(f"    -> FAILED: {e}")

        # --- pre_window (SAR-dependent) ---
        w = windows.get('pre_window', {})
        if w.get('status') in ('missing', 'insufficient') and has_sar:
            print(f"\n    Searching: pre-battle SAR-aligned")
            if dry_run:
                print(f"    [DRY RUN] Would search pre-battle SAR-aligned")
            else:
                try:
                    sar_pre_dates, sar_post_dates, _ = ms_scene_discovery.load_sar_scene_dates(city_name)
                    if sar_pre_dates:
                        scenes, threshold = ms_scene_discovery.search_ms_for_sar_dates(
                            city_geom, city_bounds, sar_pre_dates, "Pre-battle", battle_start, battle_stop
                        )
                        if scenes:
                            meta['pre_window'] = {
                                'label': 'sar_aligned',
                                'sar_dates': [d.strftime('%Y-%m-%d') for d in sar_pre_dates],
                                'cloud_threshold_used': float(threshold),
                                'scenes_found': len(scenes),
                                'scenes': scenes,
                            }
                            changed = True
                            print(f"    -> Added {len(scenes)} pre-battle scenes")
                    else:
                        print(f"    -> No SAR pre-battle dates found")
                except Exception as e:
                    print(f"    -> FAILED: {e}")

        # --- post_window (SAR-dependent) ---
        w = windows.get('post_window', {})
        if w.get('status') in ('missing', 'insufficient') and has_sar:
            print(f"\n    Searching: post-battle SAR-aligned")
            if dry_run:
                print(f"    [DRY RUN] Would search post-battle SAR-aligned")
            else:
                try:
                    sar_pre_dates, sar_post_dates, _ = ms_scene_discovery.load_sar_scene_dates(city_name)
                    if sar_post_dates:
                        scenes, threshold = ms_scene_discovery.search_ms_for_sar_dates(
                            city_geom, city_bounds, sar_post_dates, "Post-battle", battle_start, battle_stop
                        )
                        if scenes:
                            meta['post_window'] = {
                                'label': 'sar_aligned',
                                'sar_dates': [d.strftime('%Y-%m-%d') for d in sar_post_dates],
                                'cloud_threshold_used': float(threshold),
                                'scenes_found': len(scenes),
                                'scenes': scenes,
                            }
                            changed = True
                            print(f"    -> Added {len(scenes)} post-battle scenes")
                    else:
                        print(f"    -> No SAR post-battle dates found")
                except Exception as e:
                    print(f"    -> FAILED: {e}")

        # --- battle_window (SAR-dependent, all tiers now) ---
        w = windows.get('battle_window', {})
        if w.get('status') == 'missing' and has_sar:
            try:
                _, _, sar_battle_dates = ms_scene_discovery.load_sar_scene_dates(city_name)
                if sar_battle_dates and len(sar_battle_dates) > 0:
                    print(f"\n    Searching: battle-period ({len(sar_battle_dates)} SAR dates)")
                    if dry_run:
                        print(f"    [DRY RUN] Would search battle-period")
                    else:
                        # sample monthly
                        battle_dates_sorted = sorted(sar_battle_dates)
                        sampled = []
                        last_month = None
                        for d in battle_dates_sorted:
                            mk = (d.year, d.month)
                            if mk != last_month:
                                sampled.append(d)
                                last_month = mk

                        all_candidates = []
                        seen_ids = set()
                        for sar_date in sampled:
                            for ct in [ms_scene_discovery.CLOUD_COVER_STRICT,
                                       ms_scene_discovery.CLOUD_COVER_RELAXED,
                                       ms_scene_discovery.CLOUD_COVER_MAX]:
                                scenes = ms_scene_discovery.search_sentinel2_directional(
                                    city_geom, city_bounds, sar_date,
                                    ms_scene_discovery.SAR_DATE_WINDOW_DAYS, ct, 'both'
                                )
                                if scenes:
                                    for s in scenes:
                                        if s['id'] not in seen_ids:
                                            seen_ids.add(s['id'])
                                            all_candidates.append(s)
                                    break

                        if all_candidates:
                            all_candidates.sort(key=lambda x: x['date'])
                            meta['battle_window'] = {
                                'label': f'sar_aligned_tier{tier}',
                                'cloud_threshold_used': float(min(s.get('cloud_threshold_used', 99) for s in all_candidates)),
                                'scenes_found': len(all_candidates),
                                'scenes': all_candidates,
                            }
                            changed = True
                            print(f"    -> Added {len(all_candidates)} battle-period scenes")
                        else:
                            print(f"    -> No battle-period scenes found")
            except Exception as e:
                print(f"    -> Battle search FAILED: {e}")

        # --- biweekly_window (SAR-dependent) ---
        w = windows.get('biweekly_window', {})
        if w.get('status') == 'missing' and has_sar:
            try:
                biweekly_pairs = ms_scene_discovery.load_sar_biweekly_dates(city_name)
                if biweekly_pairs:
                    print(f"\n    Searching: biweekly MS ({len(biweekly_pairs)} SAR pairs)")
                    if dry_run:
                        print(f"    [DRY RUN] Would search biweekly")
                    else:
                        scenes = ms_scene_discovery.search_ms_for_biweekly(
                            city_geom, city_bounds, biweekly_pairs
                        )
                        if scenes:
                            meta['biweekly_window'] = {
                                'label': 'sar_aligned_biweekly',
                                'scenes_found': len(scenes),
                                'scenes': scenes,
                            }
                            changed = True
                            print(f"    -> Added {len(scenes)} biweekly scenes")
            except Exception as e:
                print(f"    -> Biweekly search FAILED: {e}")

        # --- prebattle_baseline_window (SAR-dependent) ---
        w = windows.get('prebattle_baseline_window', {})
        if w.get('status') == 'missing' and has_sar:
            try:
                bl_sar_dates = ms_scene_discovery.load_sar_prebattle_baseline_dates(city_name)
                if bl_sar_dates:
                    print(f"\n    Searching: prebattle baseline MS ({len(bl_sar_dates)} SAR dates)")
                    if dry_run:
                        print(f"    [DRY RUN] Would search prebattle baseline")
                    else:
                        bl_seen = set()
                        bl_ms_scenes = []
                        for sar_date in bl_sar_dates:
                            for ct in [ms_scene_discovery.CLOUD_COVER_STRICT,
                                       ms_scene_discovery.CLOUD_COVER_RELAXED,
                                       ms_scene_discovery.CLOUD_COVER_MAX]:
                                scenes = ms_scene_discovery.search_sentinel2_directional(
                                    city_geom, city_bounds, sar_date,
                                    ms_scene_discovery.SAR_DATE_WINDOW_DAYS, ct, 'both'
                                )
                                if scenes:
                                    best = scenes[0]
                                    if best['id'] not in bl_seen:
                                        bl_seen.add(best['id'])
                                        best['sar_anchor_date'] = sar_date.strftime('%Y-%m-%d')
                                        bl_ms_scenes.append(best)
                                    break
                        if bl_ms_scenes:
                            meta['prebattle_baseline_window'] = {
                                'label': 'sar_aligned_prebattle_baseline',
                                'scenes_found': len(bl_ms_scenes),
                                'scenes': bl_ms_scenes,
                            }
                            changed = True
                            print(f"    -> Added {len(bl_ms_scenes)} prebattle baseline scenes")
            except Exception as e:
                print(f"    -> Prebattle baseline search FAILED: {e}")

        # save updated metadata
        if changed and not dry_run:
            # recount total
            total = 0
            for wk in REQUIRED_WINDOWS:
                total += len(meta.get(wk, {}).get('scenes', []))
            meta['total_scenes'] = total
            meta['timestamp'] = datetime.now().isoformat()
            meta['last_gap_fill'] = datetime.now().isoformat()

            with open(metadata_file, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"\n    Saved: {metadata_file.name} ({total} total scenes)")
            filled_count += 1
        elif changed and dry_run:
            print(f"\n    [DRY RUN] Would save updated metadata")
        else:
            print(f"\n    No new scenes found")
            failed_count += 1

    # final summary
    print(f"\n{'='*70}")
    print(f"FILL GAPS {'(DRY RUN) ' if dry_run else ''}COMPLETE")
    print(f"{'='*70}")
    print(f"  Cities updated: {filled_count}")
    print(f"  Cities with no new scenes: {failed_count}")

    return audit
