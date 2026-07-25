# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
scene_plan.py
Thin plan builder that creates a unified download plan from VERIFIED module outputs.

Does NOT query CDSE. Does NOT validate coverage (discovery modules already did that).
Reads FROM: dl_sync results, ms_scene_metadata.json, scene_metadata.json
Produces: nb02a_scene_plan.json with lifecycle status tracking.

Notebook usage:
    from scene_plan import build_from_verified, compute_reuse_report

    PLAN = build_from_verified(
        dl_sync=DL_SYNC,
        ms_metadata_dir=MS_METADATA_DIR,
        cities_dir=CITIES_DIR,
        outputs_dir=OUTPUTS_DIR,
        catalog_db=CATALOG_DB,
    )
    compute_reuse_report(PLAN)
"""

import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ============================================================================
# PHASE 2: DISK SCAN (kept from v1 — proven)
# ============================================================================

def scan_slc_zips(raw_slc_zip, min_size=3e9):
    """Scan SLC zip directory. Returns dict: orbit -> [{date, filename, size_gb, abs_orbit, rel_orbit}]."""
    raw_slc_zip = Path(raw_slc_zip)
    if not raw_slc_zip.exists():
        return {}
    orbits = defaultdict(list)
    for f in raw_slc_zip.iterdir():
        if f.suffix != '.zip' or f.stat().st_size < min_size:
            continue
        m = re.search(
            r'(S1[ABC])_IW_SLC.*?_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{6})_([A-F0-9]+)',
            f.name)
        if not m:
            continue
        sat = m.group(1)
        date_raw = m.group(2)
        abs_orbit = int(m.group(3))
        if sat == "S1B":
            rel_orbit = ((abs_orbit - 27) % 175) + 1
        else:
            rel_orbit = ((abs_orbit - 73) % 175) + 1
        date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        orbits[rel_orbit].append({
            "date": date_str, "filename": f.name,
            "size_gb": round(f.stat().st_size / 1e9, 2),
            "abs_orbit": abs_orbit, "rel_orbit": rel_orbit,
        })
    return dict(orbits)


def scan_card_zips(card_zip_dir):
    """Scan CARD zip directory. Returns dict: date_str -> [{filename, size_gb}]."""
    card_zip_dir = Path(card_zip_dir)
    if not card_zip_dir.exists():
        return {}
    by_date = defaultdict(list)
    for f in card_zip_dir.iterdir():
        if f.suffix != '.zip' or 'CARD_BS' not in f.name:
            continue
        m = re.search(r'_(\d{8})T', f.name)
        if m:
            d = m.group(1)
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            by_date[date_str].append({"filename": f.name, "size_gb": round(f.stat().st_size / 1e9, 2)})
    return dict(by_date)


def scan_ms_zips(ms_zip_dir):
    """Scan MS zip directory. Returns set of date strings on disk."""
    ms_zip_dir = Path(ms_zip_dir)
    if not ms_zip_dir.exists():
        return set()
    dates = set()
    for f in ms_zip_dir.iterdir():
        if f.suffix != '.zip':
            continue
        m = re.search(r'_(\d{8})T', f.name)
        if m:
            d = m.group(1)
            dates.add(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
    return dates


def scan_card_tifs(sar_card_dir):
    """Scan CARD TIF directory. Returns dict: city -> set of date strings with VV+VH."""
    sar_card_dir = Path(sar_card_dir)
    if not sar_card_dir.exists():
        return {}
    result = {}
    for city_dir in sar_card_dir.iterdir():
        if not city_dir.is_dir():
            continue
        dates = defaultdict(set)
        for tif in city_dir.rglob("*.tif"):
            m = re.search(r'(\d{8})', tif.stem)
            pol_m = re.search(r'(VV|VH)', tif.stem, re.IGNORECASE)
            if m and pol_m:
                d = m.group(1)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                dates[date_str].add(pol_m.group(1).upper())
        result[city_dir.name] = {d for d, pols in dates.items() if 'VV' in pols and 'VH' in pols}
    return result


# ============================================================================
# MS METADATA READER (reads coverage-verified scenes from ms_scene_discovery)
# ============================================================================

WINDOW_PERIOD_MAP = {
    'pre_window': 'pre_battle',
    'post_window': 'post_battle',
    'battle_window': 'biweekly',
    'baseline_window': 'winter_baseline_pre',
    'post_baseline_window': 'winter_baseline_post',
    'biweekly_window': 'biweekly',
    'prebattle_baseline_window': 'prebattle_baseline',
}


def _load_ms_metadata_for_city(city_name, ms_metadata_dir):
    """Read ms_scene_metadata.json for a city.
    Returns list of scene dicts with: id, name, date, cloud_cover, tile_id, coverage_pct, period.
    These scenes were already coverage-verified by ms_scene_discovery.
    """
    ms_metadata_dir = Path(ms_metadata_dir)
    meta_file = ms_metadata_dir / f"{city_name}_ms_scene_metadata.json"
    if not meta_file.exists():
        return []

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    scenes = []
    seen_ids = set()
    for window_key, period in WINDOW_PERIOD_MAP.items():
        window = meta.get(window_key, {})
        for scene in window.get('scenes', []):
            sid = scene.get('id', '')
            if sid in seen_ids:
                continue
            seen_ids.add(sid)

            name = scene.get('name', '')
            date_str = scene.get('date', '')[:10]

            # extract tile_id from scene name
            tile_id = ''
            tile_m = re.search(r'_(T\d{2}[A-Z]{3})_', name)
            if tile_m:
                tile_id = tile_m.group(1)

            scenes.append({
                'id': sid,
                'name': name,
                'date': date_str,
                'cloud_cover': scene.get('cloud_cover'),
                'coverage_pct': scene.get('coverage_pct'),
                'tile_id': tile_id,
                'period': period,
            })
    return scenes


# ============================================================================
# BUILD UNIFIED PLAN FROM VERIFIED MODULE OUTPUTS
# ============================================================================

def build_from_verified(dl_sync, ms_metadata_dir, cities_dir, outputs_dir, catalog_db,
                        ms_max_cloud=20, sar_card_dir=None, cities_df=None):
    """Build unified download plan from verified discovery + dl_sync outputs.

    Args:
        dl_sync:         dict from dl_sync.run() — has slc_plan, card_plan, disk scan, city_orbit_map
        ms_metadata_dir: Path — ms_scene_metadata.json files (coverage-verified by ms_scene_discovery)
        cities_dir:      Path — cities directory
        outputs_dir:     Path — where to save plan JSON
        catalog_db:      Path — bda.sqlite
        ms_max_cloud:    float — hard cap on MS cloud cover (default 20%)

    Returns:
        dict: {city_name: plan_dict, ...}
    """
    ms_metadata_dir = Path(ms_metadata_dir)
    outputs_dir = Path(outputs_dir)

    download_targets = dl_sync.get('download_targets', {})
    card_plan_raw = dl_sync.get('card_plan', {})
    city_orbit_map = dl_sync.get('city_orbit_map', {})
    ms_zips_on_disk = dl_sync.get('ms_zips_on_disk', {})

    # card_plan from dl_sync may be nested under 'cities'
    if 'cities' in card_plan_raw:
        card_cities = card_plan_raw['cities']
    else:
        card_cities = card_plan_raw

    print("=" * 80)
    print("BUILDING UNIFIED PLAN FROM VERIFIED DATA")
    print("=" * 80)

    # Scan actual CARD TIFs on disk for ground truth count
    card_tifs_actual = {}
    if sar_card_dir:
        card_tifs_actual = scan_card_tifs(Path(sar_card_dir))
        print(f"  CARD TIFs on disk: {sum(len(v) for v in card_tifs_actual.values())} dates across {len(card_tifs_actual)} cities")

    all_plans = {}

    for city, slc_data in sorted(download_targets.items()):
        orbit = slc_data.get('orbit')
        tier = slc_data.get('tier', 99)

        # get CARD orbit from cities_df (independent of SLC orbit)
        orbit_card = orbit  # default: same as SLC
        if cities_df is not None:
            city_row = cities_df[cities_df['city'] == city]
            if not city_row.empty:
                oc = city_row.iloc[0].get('recommended_orbit_card')
                if oc is not None and not (isinstance(oc, float) and str(oc) == 'nan'):
                    orbit_card = int(oc)
            # DEBUG: print first 3 cities
            if city in ('Avdiivka', 'Sievierodonetsk', 'Mykolaiv'):
                print(f"  DEBUG {city}: cities_df cols={list(cities_df.columns)[:5]}, "
                      f"recommended_orbit_card={'recommended_orbit_card' in cities_df.columns}, "
                      f"oc={city_row.iloc[0].get('recommended_orbit_card') if not city_row.empty else 'EMPTY'}, "
                      f"orbit_card={orbit_card}")
        else:
            print(f"  DEBUG: cities_df is None!")

        # --- SLC entries (from dl_sync download_targets) ---
        slc_entries = []
        for s in slc_data.get('scenes', []):
            slc_entries.append({
                "date": s.get('date', ''),
                "period": s.get('purpose', ''),
                "status": "on_disk" if s.get('status') == 'on_disk' else "to_download",
                "modality": "slc",
                "orbit": orbit,
                "scene_id": s.get('id', ''),
                "scene_name": s.get('name', ''),
            })

        # --- CARD entries (from dl_sync card_plan) ---
        card_entries = []
        card_data = card_cities.get(city, {})
        for date_str, info in card_data.get('dates', {}).items():
            card_entries.append({
                "date": date_str,
                "period": info.get('purpose', ''),
                "status": "on_disk" if info.get('tif_exists') else "to_download",
                "modality": "card",
                "orbit": card_data.get('orbit', orbit),
                "scene_id": "",  # CARD scene_ids found at download time via CDSE search
                "scene_name": "",
                "slc_scene": info.get('slc_scene', ''),
            })

        # --- MS entries (from coverage-verified ms_scene_metadata.json) ---
        ms_entries = []
        ms_scenes = _load_ms_metadata_for_city(city, ms_metadata_dir)

        # determine which MS scenes are on disk (match by scene name, not just date)
        # ms_zips_on_disk keys are filename stems like S2B_MSIL2A_20220315T...
        # ms_scene_metadata names are like S2B_MSIL2A_20220315T..._20220315T....SAFE
        # normalize both: strip .SAFE, .zip suffixes for comparison
        ms_stems_on_disk = set()
        for name_stem in ms_zips_on_disk.keys():
            norm = name_stem.replace('.SAFE', '').replace('.zip', '')
            ms_stems_on_disk.add(norm)

        def _ms_scene_on_disk(scene_name):
            if not scene_name:
                return False
            norm = scene_name.replace('.SAFE', '').replace('.zip', '')
            if norm in ms_stems_on_disk:
                return True
            # also check partial match (zip stem might be truncated)
            for stem in ms_stems_on_disk:
                if norm.startswith(stem) or stem.startswith(norm):
                    return True
            return False

        for s in ms_scenes:
            cc = s.get('cloud_cover')
            if cc is not None and cc > ms_max_cloud:
                continue
            date = s.get('date', '')
            scene_name = s.get('name', '')
            ms_entries.append({
                "date": date,
                "period": s.get('period', ''),
                "status": "on_disk" if _ms_scene_on_disk(scene_name) else "to_download",
                "modality": "ms",
                "scene_id": s.get('id', ''),
                "scene_name": scene_name,
                "cloud_cover": cc,
                "tile_id": s.get('tile_id', ''),
                "coverage_pct": s.get('coverage_pct'),
            })

        # sort by date
        slc_entries.sort(key=lambda x: x["date"])
        card_entries.sort(key=lambda x: x["date"])
        ms_entries.sort(key=lambda x: x["date"])

        # counts
        def _counts(entries):
            on = sum(1 for e in entries if e["status"] == "on_disk")
            dl = sum(1 for e in entries if e["status"] == "to_download")
            return len(entries), on, dl

        slc_t, slc_on, slc_dl = _counts(slc_entries)
        card_t, card_on, card_dl = _counts(card_entries)
        ms_t, ms_on, ms_dl = _counts(ms_entries)

        # actual CARD TIFs on disk (any orbit) for this city
        card_disk_dates = card_tifs_actual.get(city, set())
        card_disk_n = len(card_disk_dates)

        # Get battle dates from cities_dir
        battle_start = None
        battle_stop = None
        try:
            aoi_file = Path(cities_dir) / city / "AOI.geojson"
            if aoi_file.exists():
                with open(aoi_file) as f:
                    gj = json.load(f)
                for feat in gj.get('features', []):
                    props = feat.get('properties', {})
                    if not battle_start and props.get('battle_start'):
                        battle_start = props['battle_start']
                    if not battle_stop and props.get('battle_stop'):
                        battle_stop = props['battle_stop']
        except Exception:
            pass

        all_plans[city] = {
            "city": city,
            "tier": tier,
            "orbit": orbit,
            "orbit_card": orbit_card,
            "battle_start": battle_start,
            "battle_stop": battle_stop,
            "slc": slc_entries,
            "card": card_entries,
            "ms": ms_entries,
            "counts": {
                "slc_total": slc_t, "slc_on_disk": slc_on, "slc_to_download": slc_dl,
                "card_total": card_t, "card_on_disk": card_on, "card_to_download": card_dl,
                "card_disk_actual": card_disk_n,
                "ms_total": ms_t, "ms_on_disk": ms_on, "ms_to_download": ms_dl,
            },
        }

        card_note = f"  (disk:{card_disk_n})" if card_disk_n != card_on else ""
        print(f"  {city:22s} T{tier} orbit={orbit}  "
              f"SLC={slc_on}/{slc_t}  CARD={card_on}/{card_t}{card_note}  MS={ms_on}/{ms_t}")

    # Save plan
    plan_file = outputs_dir / 'nb02a_scene_plan.json'
    with open(plan_file, 'w') as f:
        json.dump(all_plans, f, indent=2, default=str)
    print(f"\n  Plan saved: {plan_file} ({plan_file.stat().st_size // 1024} KB)")

    # Save to sqlite
    try:
        _save_plan_sqlite(all_plans, catalog_db)
    except Exception as e:
        print(f"  SQLite save failed: {e}")

    print("=" * 80)
    return all_plans


# ============================================================================
# SQLITE SAVE
# ============================================================================

def _save_plan_sqlite(all_plans, db_path):
    """Save plan to bda.sqlite."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scene_coordination (
            city TEXT PRIMARY KEY, tier INTEGER, orbit INTEGER,
            battle_start TEXT, battle_stop TEXT, conflict_ongoing INTEGER,
            slc_total INTEGER, slc_on_disk INTEGER, slc_to_download INTEGER,
            card_total INTEGER, card_on_disk INTEGER, card_to_download INTEGER,
            ms_total INTEGER, ms_on_disk INTEGER, ms_to_download INTEGER,
            biweekly_count INTEGER, updated_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scene_plan (
            city TEXT, modality TEXT, date TEXT, period TEXT,
            status TEXT, orbit INTEGER, scene_id TEXT, scene_name TEXT,
            cloud_cover REAL, tile_id TEXT,
            PRIMARY KEY (city, modality, date)
        )
    """)

    for city, plan in all_plans.items():
        c = plan["counts"]
        cur.execute("""
            INSERT OR REPLACE INTO scene_coordination VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (city, plan["tier"], plan.get("orbit"),
              plan.get("battle_start"), plan.get("battle_stop"), 0,
              c["slc_total"], c["slc_on_disk"], c["slc_to_download"],
              c["card_total"], c["card_on_disk"], c["card_to_download"],
              c["ms_total"], c["ms_on_disk"], c["ms_to_download"],
              0, datetime.now().isoformat()))

        for modality in ["slc", "card", "ms"]:
            for entry in plan[modality]:
                cur.execute("""
                    INSERT OR REPLACE INTO scene_plan VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (city, modality, entry["date"], entry.get("period", ""),
                      entry["status"], entry.get("orbit"),
                      entry.get("scene_id", ""), entry.get("scene_name", ""),
                      entry.get("cloud_cover"), entry.get("tile_id", "")))

    conn.commit()
    conn.close()
    print(f"  SQLite: {db_path}")


# ============================================================================
# REUSE REPORT
# ============================================================================

def compute_reuse_report(all_plans):
    """Print reuse report with ASCII progress bars."""
    print("=" * 80)
    print("REUSE REPORT: EXISTING DATA vs DOWNLOAD NEEDED")
    print("=" * 80)

    report = {"grand": {}, "per_period": {}, "cities": {}}

    for mod in ["slc", "card", "ms"]:
        total = sum(p["counts"][f"{mod}_total"] for p in all_plans.values())
        on_disk = sum(p["counts"][f"{mod}_on_disk"] for p in all_plans.values())
        to_dl = sum(p["counts"][f"{mod}_to_download"] for p in all_plans.values())
        pct = round(100 * on_disk / total, 1) if total else 0

        bar_len = 40
        filled = int(bar_len * on_disk / total) if total else 0
        bar = "#" * filled + "." * (bar_len - filled)

        size_mult = {"slc": 8, "card": 4, "ms": 1}[mod]
        dl_gb = to_dl * size_mult

        print(f"\n  {mod.upper():4s} [{bar}]  {pct}% reusable")
        print(f"       on_disk={on_disk}  to_download={to_dl}  total={total}")
        print(f"       estimated download: ~{dl_gb} GB")

        report["grand"][mod] = {"total": total, "on_disk": on_disk, "to_download": to_dl, "reuse_pct": pct}

        # per-period
        period_counts = defaultdict(lambda: {"on_disk": 0, "to_download": 0, "total": 0})
        for plan in all_plans.values():
            for entry in plan[mod]:
                p = entry.get("period", "unknown")
                period_counts[p]["total"] += 1
                if entry["status"] == "on_disk":
                    period_counts[p]["on_disk"] += 1
                elif entry["status"] == "to_download":
                    period_counts[p]["to_download"] += 1
        report["per_period"][mod] = dict(period_counts)

    # per-period detail
    grand = report["grand"]
    total_dl_gb = (grand["slc"]["to_download"] * 8 +
                   grand["card"]["to_download"] * 4 +
                   grand["ms"]["to_download"] * 1)
    print(f"\n  Total estimated download: ~{total_dl_gb} GB")

    print(f"\n  Per-period breakdown:")
    for mod in ["slc", "card", "ms"]:
        print(f"\n    {mod.upper()}:")
        for period, counts in sorted(report["per_period"][mod].items()):
            t = counts["total"]
            pct = round(100 * counts["on_disk"] / t, 1) if t > 0 else 0.0
            print(f"      {period:<25s} {counts['on_disk']:>4d}/{t:<4d} on disk ({pct:5.1f}%)  dl={counts['to_download']}")

    # per-city
    city_scores = []
    for city, plan in all_plans.items():
        cr = {}
        for mod in ["slc", "card", "ms"]:
            t = plan["counts"][f"{mod}_total"]
            on = plan["counts"][f"{mod}_on_disk"]
            cr[mod] = {"reuse_pct": round(100 * on / t, 1) if t else 0}
        avg = sum(cr[m]["reuse_pct"] for m in ["slc", "card", "ms"]) / 3
        city_scores.append((city, avg, cr))
        report["cities"][city] = cr
    city_scores.sort(key=lambda x: x[1])

    print(f"\n  Cities with lowest reuse (most rework):")
    print(f"  {'city':<22s} {'SLC%':>6s} {'CARD%':>6s} {'MS%':>6s} {'avg%':>6s}")
    for city, avg, cr in city_scores[:15]:
        print(f"  {city:<22s} {cr['slc']['reuse_pct']:>5.1f}% {cr['card']['reuse_pct']:>5.1f}% {cr['ms']['reuse_pct']:>5.1f}% {avg:>5.1f}%")
    if len(city_scores) > 15:
        print(f"  ... and {len(city_scores) - 15} more cities")

    print("=" * 80)
    return report
