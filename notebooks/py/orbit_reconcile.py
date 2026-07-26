# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
orbit_reconcile.py
Scans existing CARD and COH TIFs on disk independently, determines the best
orbit per modality per city, and overrides cities_df so that dl_sync and
scene_plan use disk-truth orbits.

Key design: CARD and COH/SLC orbits are INDEPENDENT.
- COH is computed from SLC pairs -> same orbit by necessity -> recommended_orbit
- CARD is downloaded independently -> can be different orbit -> recommended_orbit_card

Runs AFTER scene_loader + tier_filter, BEFORE dl_sync.

Notebook usage (NB02a Cell 14):
    import orbit_reconcile
    importlib.reload(orbit_reconcile)
    from orbit_reconcile import run as run_orbit_reconcile
    cities_df = run_orbit_reconcile(
        cities_df=cities_df,
        sar_card_dir=SAR_CARD_DIR,
        sar_coh_dir=SAR_COH_DIR,
    )
"""

import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ORBIT_PAT = re.compile(r'__o(\d{3})__(\d{8})')


def scan_orbits_on_disk(base_dir):
    """Scan TIF filenames, return {city: {orbit: set_of_dates}}."""
    results = {}
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return results
    for city_dir in base_dir.iterdir():
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        orbit_dates = defaultdict(set)
        for tif in city_dir.glob('*.tif'):
            m = ORBIT_PAT.search(tif.name)
            if m:
                orbit_dates[int(m.group(1))].add(m.group(2))
        if orbit_dates:
            results[city] = dict(orbit_dates)
    return results


def _score_orbits(orbit_dates_dict, battle_start, battle_end):
    """Score each orbit by date count and period coverage."""
    if not orbit_dates_dict:
        return {}
    bs = battle_start.replace('-', '') if battle_start else None
    be = battle_end.replace('-', '') if battle_end else None
    scores = {}
    for orb, dates in orbit_dates_dict.items():
        n_pre = 0
        n_post = 0
        n_during = 0
        for d in dates:
            if bs:
                if d < bs:
                    n_pre += 1
                elif be and d > be:
                    n_post += 1
                else:
                    n_during += 1
            else:
                n_pre += 1
        has_both = (n_pre > 0 and (n_post > 0 or battle_end is None))
        scores[orb] = {
            'n_total': len(dates), 'n_pre': n_pre,
            'n_during': n_during, 'n_post': n_post, 'has_both': has_both,
        }
    return scores


def _pick_best(scores):
    """Pick best orbit: prefer has_both, then most total dates."""
    if not scores:
        return None, {}
    qualified = {o: s for o, s in scores.items() if s['has_both']}
    pool = qualified if qualified else scores
    best = max(pool, key=lambda o: pool[o]['n_total'])
    return best, scores[best]


def run(cities_df, sar_card_dir, sar_coh_dir, dry_run=True, sar_filtered=None):
    """Reconcile plan orbit vs disk reality, per modality independently.

    Sets two orbit columns in cities_df:
    - recommended_orbit      : best for SLC/COH (from COH TIFs on disk)
    - recommended_orbit_card : best for CARD (from CARD TIFs on disk)

    If sar_filtered is passed, overrides are applied there too (dl_sync uses it).

    Returns cities_df (modified in-place if dry_run=False).
    """
    print("=" * 80)
    print("ORBIT RECONCILE: DISK TRUTH vs PLAN (per-modality)")
    print("=" * 80)
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  DRY_RUN: {dry_run}")
    if dry_run:
        print("  (set dry_run=False to apply overrides)")

    print(f"\n  Scanning CARD TIFs in {sar_card_dir} ...")
    card_disk = scan_orbits_on_disk(sar_card_dir)
    print(f"    {len(card_disk)} cities with CARD TIFs")
    print(f"  Scanning COH TIFs in {sar_coh_dir} ...")
    coh_disk = scan_orbits_on_disk(sar_coh_dir)
    print(f"    {len(coh_disk)} cities with COH TIFs")

    if 'recommended_orbit_card' not in cities_df.columns:
        cities_df['recommended_orbit_card'] = None

    rows_out = []
    overrides = {}
    stats = {'ok': 0, 'override': 0, 'no_data': 0}

    for idx, row in cities_df.iterrows():
        city = row.get('city', row.name if isinstance(row.name, str) else str(row.name))
        tier = row.get('tier', '?')
        plan_orbit = row.get('recommended_orbit')
        plan_card_orbit = row.get('recommended_orbit_card')
        if plan_orbit is not None:
            plan_orbit = int(plan_orbit)
        if plan_card_orbit is not None and pd.notna(plan_card_orbit):
            plan_card_orbit = int(plan_card_orbit)
        else:
            plan_card_orbit = plan_orbit

        battle_start = row.get('battle_start')
        battle_end = row.get('battle_end')
        if pd.notna(battle_start):
            battle_start = str(battle_start)[:10]
        else:
            battle_start = None
        if pd.notna(battle_end):
            battle_end = str(battle_end)[:10]
        else:
            battle_end = None

        common_orbits = row.get('common_orbits', [])
        common_set = set()
        if isinstance(common_orbits, list):
            common_set = {int(o) for o in common_orbits if o}

        card_orbits = card_disk.get(city, {})
        coh_orbits = coh_disk.get(city, {})
        card_scores = _score_orbits(card_orbits, battle_start, battle_end)
        coh_scores = _score_orbits(coh_orbits, battle_start, battle_end)
        card_best, card_info = _pick_best(card_scores)
        coh_best, coh_info = _pick_best(coh_scores)

        # Format columns
        p_slc = f"o{plan_orbit:03d}" if plan_orbit else "?"
        d_card = f"o{card_best:03d}" if card_best else "-"
        d_coh = f"o{coh_best:03d}" if coh_best else "-"
        cn = card_info.get('n_total', 0) if card_info else 0
        cp = card_info.get('n_pre', 0) if card_info else 0
        cpo = card_info.get('n_post', 0) if card_info else 0
        hn = coh_info.get('n_total', 0) if coh_info else 0
        hp = coh_info.get('n_pre', 0) if coh_info else 0
        hpo = coh_info.get('n_post', 0) if coh_info else 0

        if not card_best and not coh_best:
            rows_out.append((tier, city, p_slc, '-', '-', '-', '-', '-', '-', '-', '-', 'NO_DATA', ''))
            stats['no_data'] += 1
            continue

        # Determine actions
        parts = []
        warns = []

        # CARD orbit check
        if card_best and card_best != plan_card_orbit:
            if common_set and card_best not in common_set:
                parts.append(f"CARD:!SUSPECT")
            else:
                parts.append(f"CARD->o{card_best:03d}")
        # COH orbit check
        if coh_best and coh_best != plan_orbit:
            if common_set and coh_best not in common_set:
                parts.append(f"COH:!SUSPECT")
            else:
                parts.append(f"COH->o{coh_best:03d}")

        # Period warnings
        if card_best and battle_start and cp == 0:
            warns.append("CARD:NO_PRE!")
        if card_best and battle_end and cpo == 0:
            warns.append("CARD:NO_POST!")
        if coh_best and battle_start and hp == 0:
            warns.append("COH:NO_PRE!")
        if coh_best and battle_end and hpo == 0:
            warns.append("COH:NO_POST!")

        if not parts:
            action_str = "OK"
            stats['ok'] += 1
        else:
            action_str = "*** " + ", ".join(parts)
            stats['override'] += 1
        warn_str = " " + " ".join(warns) if warns else ""

        rows_out.append((tier, city, p_slc, d_card, cn, cp, cpo, d_coh, hn, hp, hpo, action_str, warn_str))

        # Record overrides
        if card_best and card_best != plan_card_orbit:
            suspicious = bool(common_set and card_best not in common_set)
            if city not in overrides:
                overrides[city] = {'idx': idx, 'card': None, 'coh': None}
            overrides[city]['card'] = {'old': plan_card_orbit, 'new': card_best, 'n': cn, 'pre': cp, 'post': cpo, 'suspicious': suspicious}
        if coh_best and coh_best != plan_orbit:
            suspicious = bool(common_set and coh_best not in common_set)
            if city not in overrides:
                overrides[city] = {'idx': idx, 'card': None, 'coh': None}
            overrides[city]['coh'] = {'old': plan_orbit, 'new': coh_best, 'n': hn, 'pre': hp, 'post': hpo, 'suspicious': suspicious}

    # 2b. Print grouped by tier
    rows_out.sort(key=lambda r: (r[0] if isinstance(r[0], (int, float)) else 99, r[1]))
    hdr = (f"  {'tier':>4s} {'city':<22s} {'plan':>5s} "
           f"{'CARD_o':>6s} {'n':>4s} {'pre':>4s} {'pst':>4s} "
           f"{'COH_o':>6s} {'n':>4s} {'pre':>4s} {'pst':>4s}  {'action'}")
    sep = f"  {'-'*4} {'-'*22} {'-'*5} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*4} {'-'*4} {'-'*4}  {'-'*25}"
    current_tier = None
    print(f"\n{hdr}")
    print(sep)
    for item in rows_out:
        tier_val, city, p_slc = item[0], item[1], item[2]
        d_card, cn, cp, cpo = item[3], item[4], item[5], item[6]
        d_coh, hn, hp, hpo = item[7], item[8], item[9], item[10]
        action_str, warn_str = item[11], item[12]
        t = f"T{tier_val}" if isinstance(tier_val, (int, float)) else str(tier_val)
        if tier_val != current_tier:
            if current_tier is not None:
                print()
            current_tier = tier_val
        def _f(v):
            return f"{v:>4d}" if isinstance(v, int) else f"{v:>4s}"
        print(f"  {t:>4s} {city:<22s} {p_slc:>5s} "
              f"{d_card:>6s} {_f(cn)} {_f(cp)} {_f(cpo)} "
              f"{d_coh:>6s} {_f(hn)} {_f(hp)} {_f(hpo)}  {action_str}{warn_str}")

    # 3. Summary
    print(f"\n{'='*80}")
    print("RECONCILIATION SUMMARY")
    print(f"{'='*80}")
    print(f"  OK (matches disk):       {stats['ok']}")
    print(f"  OVERRIDE needed:         {stats['override']}")
    print(f"  NO_DATA (no TIFs):       {stats['no_data']}")

    if overrides:
        safe_card = [(c, i) for c, i in overrides.items() if i['card'] and not i['card']['suspicious']]
        safe_coh = [(c, i) for c, i in overrides.items() if i['coh'] and not i['coh']['suspicious']]
        suspect = [(c, i) for c, i in overrides.items()
                   if (i['card'] and i['card']['suspicious']) or (i['coh'] and i['coh']['suspicious'])]
        if safe_card:
            print(f"\n  CARD ORBIT OVERRIDES ({len(safe_card)} cities):")
            for city, info in safe_card:
                c = info['card']
                old = f"o{c['old']:03d}" if c['old'] else "None"
                print(f"    {city:<22s}  {old} -> o{c['new']:03d}  (n:{c['n']}, pre:{c['pre']}, post:{c['post']})")
        if safe_coh:
            print(f"\n  COH/SLC ORBIT OVERRIDES ({len(safe_coh)} cities):")
            for city, info in safe_coh:
                c = info['coh']
                old = f"o{c['old']:03d}" if c['old'] else "None"
                print(f"    {city:<22s}  {old} -> o{c['new']:03d}  (n:{c['n']}, pre:{c['pre']}, post:{c['post']})")
        if suspect:
            print(f"\n  SUSPICIOUS ({len(suspect)} - orbit not in common_orbits, SKIPPED):")
            for city, info in suspect:
                for mod, label in [('card', 'CARD'), ('coh', 'COH')]:
                    c = info[mod]
                    if c and c['suspicious']:
                        print(f"    {city:<22s}  {label}: o{c['new']:03d} not in discovery")

    # 4. Apply overrides
    if not dry_run and overrides:
        applied = 0
        for city, info in overrides.items():
            idx = info['idx']
            if info['card'] and not info['card']['suspicious']:
                cities_df.at[idx, 'recommended_orbit_card'] = info['card']['new']
                applied += 1
                print(f"    {city}: recommended_orbit_card = {info['card']['new']}")
            if info['coh'] and not info['coh']['suspicious']:
                new_orbit = info['coh']['new']
                cities_df.at[idx, 'recommended_orbit'] = new_orbit
                existing = cities_df.at[idx, 'common_orbits']
                if not isinstance(existing, list):
                    existing = []
                updated = list(existing)
                if new_orbit not in updated:
                    updated.insert(0, new_orbit)
                cities_df.at[idx, 'common_orbits'] = updated
                applied += 1
                print(f"    {city}: recommended_orbit = {new_orbit}")
        # default: cities without CARD override keep plan orbit for CARD too
        for idx, row in cities_df.iterrows():
            if pd.isna(cities_df.at[idx, 'recommended_orbit_card']) or cities_df.at[idx, 'recommended_orbit_card'] is None:
                cities_df.at[idx, 'recommended_orbit_card'] = row.get('recommended_orbit')

        # propagate overrides to sar_filtered (dl_sync uses this, not cities_df)
        if sar_filtered is not None:
            if 'recommended_orbit_card' not in sar_filtered.columns:
                sar_filtered['recommended_orbit_card'] = None
            for city, info in overrides.items():
                mask = sar_filtered['city'] == city
                if not mask.any():
                    continue
                filt_idx = sar_filtered.index[mask][0]
                if info['card'] and not info['card']['suspicious']:
                    sar_filtered.at[filt_idx, 'recommended_orbit_card'] = info['card']['new']
                if info['coh'] and not info['coh']['suspicious']:
                    sar_filtered.at[filt_idx, 'recommended_orbit'] = info['coh']['new']
                    existing = sar_filtered.at[filt_idx, 'common_orbits']
                    if not isinstance(existing, list):
                        existing = []
                    updated = list(existing)
                    if info['coh']['new'] not in updated:
                        updated.insert(0, info['coh']['new'])
                    sar_filtered.at[filt_idx, 'common_orbits'] = updated
            # default card orbit for filtered too
            for filt_idx, row in sar_filtered.iterrows():
                if pd.isna(sar_filtered.at[filt_idx, 'recommended_orbit_card']) or sar_filtered.at[filt_idx, 'recommended_orbit_card'] is None:
                    sar_filtered.at[filt_idx, 'recommended_orbit_card'] = row.get('recommended_orbit')
            print(f"  Propagated overrides to sar_filtered ({len(sar_filtered)} cities).")

        print(f"\n  Applied {applied} overrides.")
    elif dry_run and overrides:
        print(f"\n  DRY_RUN=True: no changes applied. Set dry_run=False to override.")
    else:
        print(f"\n  No overrides needed.")

    return cities_df
