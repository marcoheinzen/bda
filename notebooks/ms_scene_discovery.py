# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
ms_scene_discovery.py
Discovers Sentinel-2 L2A scenes aligned with SAR scene dates.
Extracted from Cell 14A1-MS.

Notebook usage:
    from ms_scene_discovery import run as run_ms_discovery
    run_ms_discovery(
        cities_dir=CITIES_DIR,
        logs_dir=LOGS_DIR,
        sar_metadata_dir=SAR_METADATA_DIR,
        ms_metadata_dir=MS_METADATA_DIR,
        ms_progress_file=MS_PROGRESS_FILE,
        outputs_dir=OUTPUTS_DIR,
        force_rerun=FORCE_RERUN,
        city_selection=CITY_SELECTION,
        sar_date_window_days=MS_SAR_WINDOW_DAYS,
        cloud_cover_strict=MS_CLOUD_STRICT,
        cloud_cover_relaxed=MS_CLOUD_RELAXED,
        cloud_cover_max=MS_CLOUD_MAX,
        min_coverage_pct=MS_MIN_COVERAGE_PCT,
        scenes_per_period=MS_SCENES_PER_PERIOD,
        winter_months=MS_WINTER_MONTHS,
        baseline_n_scenes=MS_BASELINE_N_SCENES,
        baseline_max_cloud=MS_BASELINE_MAX_CLOUD,
    )
"""

import json
import sys
import requests
from pathlib import Path
from datetime import datetime, timedelta

import geopandas as gpd
from shapely.geometry import shape
from shapely import wkt as shapely_wkt
import warnings
warnings.filterwarnings('ignore')

from aoi_date_extend_loader import (
    load_aoi_gdf as _loader_load_aoi_gdf,
    load_aoi_bbox as _loader_load_aoi_bbox,
    discover_cities as _loader_discover_cities,
)


# ---------------------------------------------------------------------------
# Module globals - set by run()
# ---------------------------------------------------------------------------
_CITIES_DIR = None
_SAR_METADATA_DIR = None
_MS_METADATA_DIR = None
_OUTPUTS_DIR = None

SAR_DATE_WINDOW_DAYS = 30
CLOUD_COVER_STRICT = 1
CLOUD_COVER_RELAXED = 5
CLOUD_COVER_MAX = 10
MIN_COVERAGE_PCT = 90
SCENES_PER_PERIOD = 2
WINTER_MONTHS = [10, 11, 12, 1, 2]
BASELINE_N_SCENES = 5
BASELINE_MAX_CLOUD = 15
POST_BASELINE_N_SCENES = 5

TEMPORAL_THRESHOLD_MONTHS = 6
BASELINE_DATES = {
    'pre_2022': {'start': '2021-09-01', 'end': '2022-02-22', 'ideal_start': '2022-01-01', 'ideal_end': '2022-02-22'},
    'post_2025': {'start': '2025-01-01', 'end': '2025-12-31', 'ideal_start': '2025-09-01', 'ideal_end': '2025-12-31'}
}


# ---------------------------------------------------------------------------
# DualLogger
# ---------------------------------------------------------------------------
class DualLogger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
def load_progress(progress_file):
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress, progress_file):
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
def parse_date(date_str):
    if isinstance(date_str, datetime):
        return date_str
    if date_str == 'ongoing' or date_str is None:
        return None
    if 'T' in str(date_str):
        return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    return datetime.strptime(date_str, "%Y-%m-%d")


# ---------------------------------------------------------------------------
# AOI loader (wraps aoi_date_extend_loader)
# ---------------------------------------------------------------------------
def load_city_boundary_with_dates(city_name):
    aoi_gdf = _loader_load_aoi_gdf(city_name, _CITIES_DIR)
    gdf = aoi_gdf

    # load aoi_bbox for coverage checks (NB03a clips to aoi_bbox, not city_polygon)
    aoi_bbox_geom = _loader_load_aoi_bbox(city_name, _CITIES_DIR)

    battle_start = gdf.iloc[0].get('battle_start', None)
    battle_stop = gdf.iloc[0].get('battle_stop', None)
    tier = int(gdf.iloc[0].get('tier', 99))

    if not battle_start:
        raise ValueError(f"Missing battle_start in {city_name}")

    pre_battle_date = parse_date(battle_start)
    post_battle_date = parse_date(battle_stop) if battle_stop and battle_stop != 'ongoing' else None
    conflict_ongoing = battle_stop == 'ongoing' or battle_stop is None

    return gdf, pre_battle_date, post_battle_date, conflict_ongoing, tier, aoi_bbox_geom


# ---------------------------------------------------------------------------
# SAR metadata loaders
# ---------------------------------------------------------------------------
def load_sar_scene_dates(city_name):
    """Load SAR scene dates from metadata JSON. Returns pre, post, battle dates."""
    sar_metadata_file = _SAR_METADATA_DIR / f"{city_name}_scene_metadata.json"

    if not sar_metadata_file.exists():
        print(f"    No SAR metadata found: {sar_metadata_file.name}")
        return None, None, None

    with open(sar_metadata_file, 'r') as f:
        metadata = json.load(f)

    rec_orbit = str(metadata.get('recommended_orbit'))
    orbits = metadata.get('orbits', {})

    if rec_orbit in orbits:
        orbit_data = orbits[rec_orbit]
        pre_scenes = orbit_data.get('pre_scenes', [])
        post_scenes = orbit_data.get('post_scenes', [])
        battle_scenes = orbit_data.get('battle_scenes', [])
    else:
        pre_scenes = []
        post_scenes = []
        battle_scenes = []

    pre_dates = []
    for s in pre_scenes[:SCENES_PER_PERIOD]:
        try:
            date_str = s.get('date', '')[:10]
            pre_dates.append(parse_date(date_str))
        except:
            pass

    post_dates = []
    for s in post_scenes[:SCENES_PER_PERIOD]:
        try:
            date_str = s.get('date', '')[:10]
            post_dates.append(parse_date(date_str))
        except:
            pass

    battle_dates = []
    for s in battle_scenes:
        try:
            date_str = s.get('date', '')[:10]
            battle_dates.append(parse_date(date_str))
        except:
            pass

    return pre_dates, post_dates, battle_dates


def load_sar_biweekly_dates(city_name):
    """Load biweekly SAR pair dates from InSAR tracker. Returns list of (date_a, date_b) tuples."""
    tracker_file = _OUTPUTS_DIR / 'insar_processing_tracker.json'
    if not tracker_file.exists():
        return []

    with open(tracker_file, 'r') as f:
        tracker = json.load(f)

    city_data = tracker.get('cities', {}).get(city_name, {})
    biweekly = city_data.get('biweekly', {})
    pairs = biweekly.get('pairs', {})

    result = []
    for pair_key, pair_data in pairs.items():
        if pair_data.get('status') != 'success':
            continue
        dates = pair_key.split('_')
        if len(dates) == 2:
            try:
                d1 = parse_date(dates[0])
                d2 = parse_date(dates[1])
                result.append((d1, d2))
            except:
                pass

    result.sort(key=lambda x: x[0])
    return result


def load_sar_prebattle_baseline_dates(city_name):
    """Load prebattle baseline SAR dates from InSAR tracker. Returns list of dates."""
    tracker_file = _OUTPUTS_DIR / 'insar_processing_tracker.json'
    if not tracker_file.exists():
        return []

    with open(tracker_file, 'r') as f:
        tracker = json.load(f)

    city_data = tracker.get('cities', {}).get(city_name, {})
    bl_data = city_data.get('prebattle_baseline', {})
    pairs = bl_data.get('pairs', {})

    all_dates = set()
    for pair_key, pair_data in pairs.items():
        if pair_data.get('status') != 'success':
            continue
        dates = pair_key.split('_')
        for d in dates:
            try:
                all_dates.add(parse_date(d))
            except:
                pass

    return sorted(all_dates)


# ---------------------------------------------------------------------------
# Sentinel-2 search
# ---------------------------------------------------------------------------
def calculate_boundary_coverage(footprint, city_geom):
    try:
        scene_geom = None

        if isinstance(footprint, dict):
            scene_geom = shape(footprint)
        else:
            footprint_clean = str(footprint)
            if 'POLYGON' in footprint_clean.upper():
                polygon_pos = footprint_clean.upper().find('POLYGON')
                footprint_clean = footprint_clean[polygon_pos:]
            footprint_clean = footprint_clean.replace("'", "").replace('"', '')
            scene_geom = shapely_wkt.loads(footprint_clean)

        if scene_geom is None or not scene_geom.is_valid:
            return 0.0

        if not scene_geom.intersects(city_geom):
            return 0.0

        intersection = scene_geom.intersection(city_geom)
        coverage_pct = (intersection.area / city_geom.area) * 100

        return float(coverage_pct)
    except Exception:
        return 0.0


def search_sentinel2_directional(city_geom, city_bounds, anchor_date, window_days, cloud_threshold, direction='both'):
    """
    Search Sentinel-2 scenes with directional control.
    direction: 'both' (+-window), 'before' (anchor-window to anchor), 'after' (anchor to anchor+window)
    """
    if direction == 'before':
        start_date = anchor_date - timedelta(days=window_days)
        end_date = anchor_date
    elif direction == 'after':
        start_date = anchor_date
        end_date = anchor_date + timedelta(days=window_days)
    else:
        start_date = anchor_date - timedelta(days=window_days)
        end_date = anchor_date + timedelta(days=window_days)

    minx, miny, maxx, maxy = city_bounds
    aoi_wkt = f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}))"

    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    filter_parts = [
        f"Collection/Name eq 'SENTINEL-2'",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')",
        f"ContentDate/Start ge {start_date.strftime('%Y-%m-%d')}T00:00:00.000Z",
        f"ContentDate/Start le {end_date.strftime('%Y-%m-%d')}T23:59:59.999Z",
        f"contains(Name,'L2A')"
    ]
    filter_query = " and ".join(filter_parts)

    params = {
        '$filter': filter_query,
        '$orderby': 'ContentDate/Start desc',
        '$top': 100,
        '$expand': 'Attributes'
    }

    response = None
    for _retry in range(3):
        try:
            response = requests.get(base_url, params=params, timeout=120)
            if response.status_code == 200:
                break
            print(f"    S2 search HTTP {response.status_code}, retry {_retry+1}")
        except Exception as _e:
            _wait = 30 * (_retry + 1)
            print(f"    S2 search error retry {_retry+1}: {_e}")
            if _retry < 2:
                import time
                time.sleep(_wait)

    if response is None or response.status_code != 200:
        return []

    try:
        data = response.json()
        products = data.get('value', [])

        valid_scenes = []
        for product in products:
            product_name = product.get('Name', 'Unknown')

            cloud_cover = None
            for attr in product.get('Attributes', []):
                if attr.get('Name') == 'cloudCover':
                    cloud_cover = attr.get('Value')
                    break

            if cloud_cover is None or cloud_cover > cloud_threshold:
                continue

            footprint = None
            geofootprint = product.get('GeoFootprint', {})
            if isinstance(geofootprint, dict) and geofootprint.get('type') and geofootprint.get('coordinates'):
                footprint = geofootprint
            elif geofootprint.get('Geography'):
                footprint = geofootprint.get('Geography')

            if not footprint:
                continue

            coverage_pct = calculate_boundary_coverage(footprint, city_geom)

            if coverage_pct < MIN_COVERAGE_PCT:
                continue

            scene_date = datetime.fromisoformat(product['ContentDate']['Start'].replace('Z', '+00:00')).replace(tzinfo=None)
            date_diff = abs((scene_date - anchor_date).days)

            scene_info = {
                'name': product_name,
                'id': product['Id'],
                'date': product['ContentDate']['Start'],
                'cloud_cover': float(cloud_cover),
                'coverage_pct': float(coverage_pct),
                'anchor_date': anchor_date.strftime('%Y-%m-%d'),
                'date_diff_days': int(date_diff),
                'cloud_threshold_used': float(cloud_threshold),
                'download_url': f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value"
            }
            valid_scenes.append(scene_info)

        valid_scenes.sort(key=lambda x: (x['cloud_cover'], x['date_diff_days']))

        return valid_scenes

    except Exception as e:
        print(f"        Search error: {type(e).__name__}: {e}")
        return []


# ---------------------------------------------------------------------------
# MS search strategies
# ---------------------------------------------------------------------------
def search_ms_for_sar_dates(city_geom, city_bounds, sar_dates, period_label, battle_start, battle_end):
    """Search MS scenes for SAR dates with battle-date fallback."""
    print(f"\n  {period_label}: {len(sar_dates)} SAR dates to match")

    is_pre_battle = 'pre' in period_label.lower()

    all_candidates = []

    # Stage 1: Standard search +-30 days around SAR dates
    for i, sar_date in enumerate(sar_dates):
        print(f"    SAR date {i+1}: {sar_date.strftime('%Y-%m-%d')} (+-{SAR_DATE_WINDOW_DAYS}d)")

        for cloud_threshold in [CLOUD_COVER_STRICT, CLOUD_COVER_RELAXED, CLOUD_COVER_MAX]:
            scenes = search_sentinel2_directional(
                city_geom, city_bounds, sar_date, SAR_DATE_WINDOW_DAYS, cloud_threshold, 'both'
            )

            if scenes:
                print(f"      Cloud <{cloud_threshold}%: {len(scenes)} scenes")
                all_candidates.extend(scenes)
                break
            else:
                print(f"      Cloud <{cloud_threshold}%: 0 scenes")

    # Remove duplicates
    seen_ids = set()
    unique_candidates = []
    for s in all_candidates:
        if s['id'] not in seen_ids:
            seen_ids.add(s['id'])
            unique_candidates.append(s)

    # Stage 2: Battle date fallback if <2 scenes
    if len(unique_candidates) < SCENES_PER_PERIOD:
        if is_pre_battle:
            anchor_date = battle_start
            direction = 'before'
            print(f"    Fallback: from battle_start ({anchor_date.strftime('%Y-%m-%d')}) going back...")
        else:
            anchor_date = battle_end if battle_end else datetime.now()
            direction = 'after'
            print(f"    Fallback: from battle_end ({anchor_date.strftime('%Y-%m-%d')}) going forward...")

        # cloud first (5% steps up to 25%), then extend timeframe
        fallback_attempts = []
        for window_days in [60, 90, 120, 180]:
            for cloud_pct in [5, 10, 15, 20, 25]:
                fallback_attempts.append((window_days, cloud_pct))

        for window_days, cloud_threshold in fallback_attempts:
            if len(unique_candidates) >= SCENES_PER_PERIOD:
                break

            print(f"      {direction} {window_days}d, cloud <{cloud_threshold}%")

            scenes = search_sentinel2_directional(
                city_geom, city_bounds, anchor_date, window_days, cloud_threshold, direction
            )

            for s in scenes:
                if s['id'] not in seen_ids:
                    seen_ids.add(s['id'])
                    unique_candidates.append(s)
                    print(f"        +1: {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}%")

                if len(unique_candidates) >= SCENES_PER_PERIOD:
                    break

    if not unique_candidates:
        print(f"    WARNING: No MS scenes found for {period_label}")
        return [], CLOUD_COVER_MAX

    unique_candidates.sort(key=lambda x: (x['cloud_cover'], x['date_diff_days']))
    best_threshold = min(s['cloud_threshold_used'] for s in unique_candidates[:SCENES_PER_PERIOD])
    print(f"    Found {len(unique_candidates)} unique candidates")

    return unique_candidates[:SCENES_PER_PERIOD], best_threshold


def search_ms_fallback(city_geom, city_bounds, pre_battle_date, post_battle_date, conflict_ongoing):
    """Fallback search when no SAR metadata available."""
    print("\n  Using fallback temporal windows (no SAR alignment)")

    # Pre-battle: search backwards from battle_start
    print(f"\n  Pre-battle (fallback): from {pre_battle_date.strftime('%Y-%m-%d')} going back...")
    pre_scenes = []
    seen_ids = set()

    fallback_attempts = [
        (60, CLOUD_COVER_STRICT),
        (60, CLOUD_COVER_RELAXED),
        (90, CLOUD_COVER_RELAXED),
        (90, CLOUD_COVER_MAX),
        (120, CLOUD_COVER_MAX),
    ]

    for window_days, cloud_threshold in fallback_attempts:
        if len(pre_scenes) >= SCENES_PER_PERIOD:
            break

        scenes = search_sentinel2_directional(
            city_geom, city_bounds, pre_battle_date, window_days, cloud_threshold, 'before'
        )

        for s in scenes:
            if s['id'] not in seen_ids:
                seen_ids.add(s['id'])
                pre_scenes.append(s)
                print(f"    +1: {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}%")

            if len(pre_scenes) >= SCENES_PER_PERIOD:
                break

    # Post-battle: search forwards from battle_end or use 2025 baseline
    if conflict_ongoing or post_battle_date is None:
        post_anchor = datetime.strptime(BASELINE_DATES['post_2025']['ideal_start'], '%Y-%m-%d')
        post_label = '2025_baseline'
    else:
        post_anchor = post_battle_date
        post_label = 'postbattle'

    print(f"\n  Post-battle (fallback): from {post_anchor.strftime('%Y-%m-%d')} going forward...")
    post_scenes = []
    seen_ids = set()

    for window_days, cloud_threshold in fallback_attempts:
        if len(post_scenes) >= SCENES_PER_PERIOD:
            break

        scenes = search_sentinel2_directional(
            city_geom, city_bounds, post_anchor, window_days, cloud_threshold, 'after'
        )

        for s in scenes:
            if s['id'] not in seen_ids:
                seen_ids.add(s['id'])
                post_scenes.append(s)
                print(f"    +1: {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}%")

            if len(post_scenes) >= SCENES_PER_PERIOD:
                break

    pre_threshold = CLOUD_COVER_MAX if not pre_scenes else min(s['cloud_threshold_used'] for s in pre_scenes)
    post_threshold = CLOUD_COVER_MAX if not post_scenes else min(s['cloud_threshold_used'] for s in post_scenes)

    return pre_scenes[:SCENES_PER_PERIOD], post_scenes[:SCENES_PER_PERIOD], '2022_baseline', post_label


def search_ms_winter_baseline(city_geom, city_bounds, battle_start, n_scenes=None):
    if n_scenes is None:
        n_scenes = BASELINE_N_SCENES

    # Search window: 1.5 years before battle, only winter months
    search_end = battle_start - timedelta(days=1)
    search_start = search_end - timedelta(days=540)

    print(f"\n  Winter baseline search: {search_start.strftime('%Y-%m-%d')} to {search_end.strftime('%Y-%m-%d')}")
    print(f"    Winter months: {WINTER_MONTHS}, cloud <{BASELINE_MAX_CLOUD}%, need {n_scenes} scenes")

    all_candidates = []
    seen_ids = set()

    # try increasing cloud thresholds and wider windows
    for max_cloud, search_days in [(BASELINE_MAX_CLOUD, 540), (20, 540), (25, 540), (25, 730)]:
        if len(all_candidates) >= n_scenes:
            break

        scenes = search_sentinel2_directional(
            city_geom, city_bounds, battle_start, search_days, max_cloud, 'before'
        )

        for s in scenes:
            scene_date_str = s.get('date', '')[:10]
            try:
                scene_dt = parse_date(scene_date_str)
            except:
                continue

            if scene_dt.month not in WINTER_MONTHS:
                continue

            if s['id'] not in seen_ids:
                seen_ids.add(s['id'])
                all_candidates.append(s)

        if len(all_candidates) >= n_scenes:
            break

    all_candidates.sort(key=lambda x: x.get('cloud_cover', 99))

    print(f"    Found {len(all_candidates)} winter scenes, using top {min(n_scenes, len(all_candidates))}")
    for i, s in enumerate(all_candidates[:n_scenes]):
        print(f"      {i+1}. {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}%")

    return all_candidates[:n_scenes]


def search_ms_post_winter_baseline(city_geom, city_bounds, battle_stop, n_scenes=None):
    if n_scenes is None:
        n_scenes = POST_BASELINE_N_SCENES

    if battle_stop is None:
        print("\n  Post-winter baseline: skipped (conflict ongoing)")
        return []

    # determine first post-battle winter window
    year = battle_stop.year
    month = battle_stop.month
    if month in [10, 11, 12]:
        pw_start = battle_stop
        pw_end = datetime(year + 1, 3, 1)
    elif month in [1, 2]:
        pw_start = battle_stop
        pw_end = datetime(year, 3, 1)
    else:
        pw_start = datetime(year, 10, 1)
        pw_end = datetime(year + 1, 3, 1)

    search_days = (pw_end - battle_stop).days
    if search_days < 30:
        search_days = 30

    print(f"\n  Post-winter baseline search: {pw_start.strftime('%Y-%m-%d')} to {pw_end.strftime('%Y-%m-%d')}")
    print(f"    Winter months: {WINTER_MONTHS}, cloud <{BASELINE_MAX_CLOUD}%, need {n_scenes} scenes")

    all_candidates = []
    seen_ids = set()

    # try increasing cloud thresholds; if still insufficient, extend to 2nd winter
    search_configs = [
        (pw_start, pw_end, BASELINE_MAX_CLOUD),
        (pw_start, pw_end, 20),
        (pw_start, pw_end, 25),
    ]
    # 2nd post-battle winter (year+1)
    pw2_start = datetime(pw_end.year, 10, 1)
    pw2_end = datetime(pw_end.year + 1, 3, 1)
    search_configs.append((pw2_start, pw2_end, BASELINE_MAX_CLOUD))
    search_configs.append((pw2_start, pw2_end, 25))

    for pw_s, pw_e, max_cloud in search_configs:
        if len(all_candidates) >= n_scenes:
            break

        search_days_cfg = max(30, (pw_e - battle_stop).days)
        scenes = search_sentinel2_directional(
            city_geom, city_bounds, battle_stop, search_days_cfg, max_cloud, 'after'
        )

        for s in scenes:
            scene_date_str = s.get('date', '')[:10]
            try:
                scene_dt = parse_date(scene_date_str)
            except:
                continue

            if scene_dt.month not in WINTER_MONTHS:
                continue

            if scene_dt < pw_s or scene_dt >= pw_e:
                continue

            if s['id'] not in seen_ids:
                seen_ids.add(s['id'])
                all_candidates.append(s)

    all_candidates.sort(key=lambda x: x.get('cloud_cover', 99))

    print(f"    Found {len(all_candidates)} post-winter scenes, using top {min(n_scenes, len(all_candidates))}")
    for i, s in enumerate(all_candidates[:n_scenes]):
        print(f"      {i+1}. {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}%")

    return all_candidates[:n_scenes]


def search_ms_for_biweekly(city_geom, city_bounds, biweekly_pairs):
    if not biweekly_pairs:
        return []

    print(f"\n  Biweekly MS search: {len(biweekly_pairs)} SAR pairs")

    all_scenes = []
    seen_ids = set()

    for pair_idx, (date_a, date_b) in enumerate(biweekly_pairs):
        for sar_date in [date_a, date_b]:
            for cloud_threshold in [CLOUD_COVER_STRICT, CLOUD_COVER_RELAXED, CLOUD_COVER_MAX]:
                scenes = search_sentinel2_directional(
                    city_geom, city_bounds, sar_date, SAR_DATE_WINDOW_DAYS, cloud_threshold, 'both'
                )
                if scenes:
                    best = scenes[0]
                    if best['id'] not in seen_ids:
                        seen_ids.add(best['id'])
                        best['biweekly_pair_idx'] = pair_idx
                        best['sar_anchor_date'] = sar_date.strftime('%Y-%m-%d')
                        all_scenes.append(best)
                    break

    all_scenes.sort(key=lambda x: x['date'])
    print(f"    Found {len(all_scenes)} unique biweekly MS scenes")

    return all_scenes


# ---------------------------------------------------------------------------
# Per-city discovery
# ---------------------------------------------------------------------------
def discover_scenes_for_city(city_name):
    print(f"\n{'='*80}")
    print(f"CITY: {city_name}")
    print(f"{'='*80}")

    try:
        gdf, pre_battle_date, post_battle_date, conflict_ongoing, tier, aoi_bbox_geom = load_city_boundary_with_dates(city_name)

        city_geom = aoi_bbox_geom  # use aoi_bbox for coverage checks (NB03a clips to this)
        city_bounds = city_geom.bounds

        print(f"\n  Tier: {tier}")
        print(f"  Battle: {pre_battle_date.date()} to {post_battle_date.date() if post_battle_date else 'ongoing'}")

        # Load SAR scene dates
        sar_pre_dates, sar_post_dates, sar_battle_dates = load_sar_scene_dates(city_name)

        if sar_pre_dates and sar_post_dates:
            print(f"\n  SAR alignment mode:")
            print(f"    Pre-battle SAR dates:  {[d.strftime('%Y-%m-%d') for d in sar_pre_dates]}")
            print(f"    Post-battle SAR dates: {[d.strftime('%Y-%m-%d') for d in sar_post_dates]}")
            if sar_battle_dates:
                print(f"    Battle SAR dates:      {len(sar_battle_dates)} dates")

            pre_scenes, pre_threshold = search_ms_for_sar_dates(
                city_geom, city_bounds, sar_pre_dates, "Pre-battle", pre_battle_date, post_battle_date
            )

            post_scenes, post_threshold = search_ms_for_sar_dates(
                city_geom, city_bounds, sar_post_dates, "Post-battle", pre_battle_date, post_battle_date
            )

            # Tier 0: search MS scenes for battle-period SAR dates
            battle_scenes = []
            battle_threshold = CLOUD_COVER_MAX
            if tier == 0 and sar_battle_dates:
                battle_dates_sorted = sorted(sar_battle_dates)
                sampled_battle_dates = []
                last_month = None
                for d in battle_dates_sorted:
                    month_key = (d.year, d.month)
                    if month_key != last_month:
                        sampled_battle_dates.append(d)
                        last_month = month_key

                print(f"\n  [TIER 0] Battle-period MS search: {len(sampled_battle_dates)} monthly samples from {len(sar_battle_dates)} SAR dates")

                battle_all_candidates = []
                seen_ids = set()

                for i, sar_date in enumerate(sampled_battle_dates):
                    print(f"    Battle SAR date {i+1}/{len(sampled_battle_dates)}: {sar_date.strftime('%Y-%m-%d')}")
                    for cloud_threshold in [CLOUD_COVER_STRICT, CLOUD_COVER_RELAXED, CLOUD_COVER_MAX]:
                        scenes = search_sentinel2_directional(
                            city_geom, city_bounds, sar_date, SAR_DATE_WINDOW_DAYS, cloud_threshold, 'both'
                        )
                        if scenes:
                            print(f"      Cloud <{cloud_threshold}%: {len(scenes)} scenes")
                            for s in scenes:
                                if s['id'] not in seen_ids:
                                    seen_ids.add(s['id'])
                                    battle_all_candidates.append(s)
                            break
                        else:
                            print(f"      Cloud <{cloud_threshold}%: 0 scenes")

                battle_all_candidates.sort(key=lambda x: x['date'])
                battle_scenes = battle_all_candidates
                battle_threshold = min(s['cloud_threshold_used'] for s in battle_scenes) if battle_scenes else CLOUD_COVER_MAX
                print(f"  [TIER 0] Found {len(battle_scenes)} unique battle-period MS scenes")

            pre_label = 'sar_aligned'
            post_label = 'sar_aligned'

            baseline_scenes = search_ms_winter_baseline(city_geom, city_bounds, pre_battle_date)
            post_baseline_scenes = search_ms_post_winter_baseline(city_geom, city_bounds, post_battle_date)
            biweekly_pairs = load_sar_biweekly_dates(city_name)
            biweekly_scenes = search_ms_for_biweekly(city_geom, city_bounds, biweekly_pairs)

            bl_sar_dates = load_sar_prebattle_baseline_dates(city_name)
            bl_ms_scenes = []
            if bl_sar_dates:
                print(f"\n  Prebattle baseline MS search: {len(bl_sar_dates)} SAR dates")
                bl_seen = set()
                for sar_date in bl_sar_dates:
                    for cloud_threshold in [CLOUD_COVER_STRICT, CLOUD_COVER_RELAXED, CLOUD_COVER_MAX]:
                        scenes = search_sentinel2_directional(
                            city_geom, city_bounds, sar_date, SAR_DATE_WINDOW_DAYS, cloud_threshold, 'both'
                        )
                        if scenes:
                            best = scenes[0]
                            if best['id'] not in bl_seen:
                                bl_seen.add(best['id'])
                                best['sar_anchor_date'] = sar_date.strftime('%Y-%m-%d')
                                bl_ms_scenes.append(best)
                            break
                print(f"    Found {len(bl_ms_scenes)} prebattle baseline MS scenes")

        else:
            pre_scenes, post_scenes, pre_label, post_label = search_ms_fallback(
                city_geom, city_bounds, pre_battle_date, post_battle_date, conflict_ongoing
            )
            pre_threshold = CLOUD_COVER_MAX if not pre_scenes else min(s['cloud_threshold_used'] for s in pre_scenes)
            post_threshold = CLOUD_COVER_MAX if not post_scenes else min(s['cloud_threshold_used'] for s in post_scenes)
            battle_scenes = []
            battle_threshold = CLOUD_COVER_MAX
            baseline_scenes = []
            post_baseline_scenes = []
            biweekly_scenes = []
            bl_ms_scenes = []
            biweekly_pairs = []
            bl_sar_dates = []

        # Build metadata
        scene_metadata = {
            'city': city_name,
            'tier': int(tier),
            'battle_start': pre_battle_date.isoformat(),
            'battle_end': post_battle_date.isoformat() if post_battle_date else None,
            'conflict_ongoing': conflict_ongoing,
            'sar_aligned': sar_pre_dates is not None and sar_post_dates is not None,
            'pre_window': {
                'label': pre_label,
                'sar_dates': [d.strftime('%Y-%m-%d') for d in sar_pre_dates] if sar_pre_dates else None,
                'cloud_threshold_used': float(pre_threshold),
                'scenes_found': len(pre_scenes),
                'scenes': pre_scenes
            },
            'post_window': {
                'label': post_label,
                'sar_dates': [d.strftime('%Y-%m-%d') for d in sar_post_dates] if sar_post_dates else None,
                'cloud_threshold_used': float(post_threshold),
                'scenes_found': len(post_scenes),
                'scenes': post_scenes
            },
            'total_scenes': len(pre_scenes) + len(post_scenes) + len(battle_scenes) + len(baseline_scenes) + len(post_baseline_scenes) + len(biweekly_scenes) + len(bl_ms_scenes),
            'timestamp': datetime.now().isoformat()
        }

        if battle_scenes:
            scene_metadata['battle_window'] = {
                'label': 'sar_aligned_tier0',
                'cloud_threshold_used': float(battle_threshold),
                'scenes_found': len(battle_scenes),
                'scenes': battle_scenes
            }

        if baseline_scenes:
            scene_metadata['baseline_window'] = {
                'label': 'winter_baseline',
                'cloud_threshold_used': float(min(s['cloud_cover'] for s in baseline_scenes)),
                'scenes_found': len(baseline_scenes),
                'scenes': baseline_scenes
            }

        if post_baseline_scenes:
            scene_metadata['post_baseline_window'] = {
                'label': 'post_winter_baseline',
                'cloud_threshold_used': float(min(s['cloud_cover'] for s in post_baseline_scenes)),
                'scenes_found': len(post_baseline_scenes),
                'scenes': post_baseline_scenes
            }

        if biweekly_scenes:
            scene_metadata['biweekly_window'] = {
                'label': 'sar_aligned_biweekly',
                'scenes_found': len(biweekly_scenes),
                'scenes': biweekly_scenes
            }

        if bl_ms_scenes:
            scene_metadata['prebattle_baseline_window'] = {
                'label': 'sar_aligned_prebattle_baseline',
                'scenes_found': len(bl_ms_scenes),
                'scenes': bl_ms_scenes
            }

        metadata_file = _MS_METADATA_DIR / f"{city_name}_ms_scene_metadata.json"

        with open(metadata_file, 'w') as f:
            json.dump(scene_metadata, f, indent=2)

        print(f"\n  SUMMARY:")
        print(f"    Pre-battle:  {len(pre_scenes)} scenes (cloud <{pre_threshold}%)")
        print(f"    Post-battle: {len(post_scenes)} scenes (cloud <{post_threshold}%)")
        if battle_scenes:
            print(f"    Battle:      {len(battle_scenes)} scenes (cloud <{battle_threshold}%)")
        if baseline_scenes:
            print(f"    Baseline:    {len(baseline_scenes)} scenes (winter)")
        if post_baseline_scenes:
            print(f"    Post BL:     {len(post_baseline_scenes)} scenes (post-winter)")
        if bl_ms_scenes:
            print(f"    Pre BL MS:   {len(bl_ms_scenes)} scenes (SAR-aligned)")
        if biweekly_scenes:
            print(f"    Biweekly:    {len(biweekly_scenes)} scenes")
        print(f"    SAR aligned: {scene_metadata['sar_aligned']}")
        print(f"    Saved: {metadata_file.name}")

        if pre_scenes:
            print(f"\n    Best pre-battle MS scenes:")
            for i, s in enumerate(pre_scenes, 1):
                diff_info = f"+{s['date_diff_days']}d from anchor" if 'date_diff_days' in s else ""
                print(f"      {i}. {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}% | {diff_info}")

        if post_scenes:
            print(f"\n    Best post-battle MS scenes:")
            for i, s in enumerate(post_scenes, 1):
                diff_info = f"+{s['date_diff_days']}d from anchor" if 'date_diff_days' in s else ""
                print(f"      {i}. {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}% | {diff_info}")

        if battle_scenes:
            print(f"\n    Battle-period MS scenes ({len(battle_scenes)} total):")
            for i, s in enumerate(battle_scenes[:5], 1):
                diff_info = f"+{s['date_diff_days']}d from anchor" if 'date_diff_days' in s else ""
                print(f"      {i}. {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}% | {diff_info}")
            if len(battle_scenes) > 5:
                print(f"      ... and {len(battle_scenes) - 5} more")

        if baseline_scenes:
            print(f"\n    Winter baseline MS scenes:")
            for i, s in enumerate(baseline_scenes, 1):
                print(f"      {i}. {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}%")
        if post_baseline_scenes:
            print(f"\n    Post-winter baseline MS scenes:")
            for i, s in enumerate(post_baseline_scenes, 1):
                print(f"      {i}. {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}%")
        if biweekly_scenes:
            print(f"\n    Biweekly MS scenes ({len(biweekly_scenes)} total):")
            for i, s in enumerate(biweekly_scenes[:10], 1):
                anchor = s.get('sar_anchor_date', '?')
                print(f"      {i}. {s['date'][:10]} | Cloud: {s['cloud_cover']:.1f}% | SAR: {anchor}")
            if len(biweekly_scenes) > 10:
                print(f"      ... and {len(biweekly_scenes) - 10} more")

        return True

    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# run() - called from notebook cell
# ---------------------------------------------------------------------------
def run(cities_dir, logs_dir, sar_metadata_dir, ms_metadata_dir, ms_progress_file,
        outputs_dir, force_rerun, city_selection=None,
        sar_date_window_days=30, cloud_cover_strict=1, cloud_cover_relaxed=5,
        cloud_cover_max=10, min_coverage_pct=90, scenes_per_period=2,
        winter_months=None, baseline_n_scenes=5, baseline_max_cloud=15):

    global _CITIES_DIR, _SAR_METADATA_DIR, _MS_METADATA_DIR, _OUTPUTS_DIR
    global SAR_DATE_WINDOW_DAYS, CLOUD_COVER_STRICT, CLOUD_COVER_RELAXED, CLOUD_COVER_MAX
    global MIN_COVERAGE_PCT, SCENES_PER_PERIOD, WINTER_MONTHS
    global BASELINE_N_SCENES, BASELINE_MAX_CLOUD, POST_BASELINE_N_SCENES

    _CITIES_DIR = Path(cities_dir)
    _SAR_METADATA_DIR = Path(sar_metadata_dir)
    _MS_METADATA_DIR = Path(ms_metadata_dir)
    _OUTPUTS_DIR = Path(outputs_dir)
    ms_progress_file = Path(ms_progress_file)

    SAR_DATE_WINDOW_DAYS = sar_date_window_days
    CLOUD_COVER_STRICT = cloud_cover_strict
    CLOUD_COVER_RELAXED = cloud_cover_relaxed
    CLOUD_COVER_MAX = cloud_cover_max
    MIN_COVERAGE_PCT = min_coverage_pct
    SCENES_PER_PERIOD = scenes_per_period
    WINTER_MONTHS = winter_months if winter_months is not None else [10, 11, 12, 1, 2]
    BASELINE_N_SCENES = baseline_n_scenes
    BASELINE_MAX_CLOUD = baseline_max_cloud
    POST_BASELINE_N_SCENES = baseline_n_scenes

    _MS_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    CELL_ID = "cell_14a1_ms_scene_discovery"
    logs_dir = Path(logs_dir)
    CELL_LOG_FILE = logs_dir / f"{CELL_ID}.json"
    CELL_OUTPUT_LOG = logs_dir / f"{CELL_ID}_output.log"

    logger = DualLogger(CELL_OUTPUT_LOG)
    sys.stdout = logger

    try:
        print("\n" + "=" * 80)
        print("CELL 14A1-MS: MULTISPECTRAL SCENE DISCOVERY (SAR-ALIGNED + FALLBACK)")
        print("=" * 80)
        print(f"Start time: {datetime.now().isoformat()}")
        print(f"SAR date window: +-{SAR_DATE_WINDOW_DAYS} days")
        print(f"Scenes per period: {SCENES_PER_PERIOD}")
        print(f"Cloud thresholds: {CLOUD_COVER_STRICT}% / {CLOUD_COVER_RELAXED}% / {CLOUD_COVER_MAX}%")
        print(f"Min coverage: {MIN_COVERAGE_PCT}%")
        print(f"SAR metadata: {_SAR_METADATA_DIR}")
        print(f"MS metadata:  {_MS_METADATA_DIR}")
        print(f"\nFallback logic:")
        print(f"  If <{SCENES_PER_PERIOD} scenes around SAR dates:")
        print(f"    Pre-battle:  search backwards from battle_start (60d/90d/120d)")
        print(f"    Post-battle: search forwards from battle_end (60d/90d/120d)")

        if force_rerun:
            print(f"\n  FORCE_RERUN = True - Clearing progress...")
            progress = {}
            save_progress(progress, ms_progress_file)

        progress = load_progress(ms_progress_file)
        progress.setdefault('scenes_discovered', {})

        all_cities = _loader_discover_cities(_CITIES_DIR)

        if not all_cities:
            raise ValueError(f"No cities found in {_CITIES_DIR}")

        print(f"\n  Found {len(all_cities)} cities")

        sar_count = sum(1 for name in all_cities if (_SAR_METADATA_DIR / f"{name}_scene_metadata.json").exists())
        print(f"  SAR metadata available: {sar_count}/{len(all_cities)} cities")

        cities_to_process = all_cities

        if city_selection:
            if isinstance(city_selection, str):
                city_selection = [city_selection]
            cities_to_process = [c for c in cities_to_process if c in city_selection]
            print(f"  City filter: {city_selection} ({len(cities_to_process)} cities)")

        success_count = 0
        failed_count = 0
        sar_aligned_count = 0

        for idx, city_name in enumerate(cities_to_process, 1):
            print(f"\n{'#'*80}")
            print(f"# CITY {idx}/{len(cities_to_process)}: {city_name}")
            print(f"{'#'*80}")

            if not force_rerun and city_name in progress['scenes_discovered'] and progress['scenes_discovered'][city_name].get('complete'):
                print(f"\n  Skipping {city_name} (already completed)")
                success_count += 1
                continue

            success = discover_scenes_for_city(city_name)

            if success:
                progress['scenes_discovered'][city_name] = {
                    'complete': True,
                    'timestamp': datetime.now().isoformat()
                }
                save_progress(progress, ms_progress_file)
                success_count += 1

                ms_meta_file = _MS_METADATA_DIR / f"{city_name}_ms_scene_metadata.json"
                if ms_meta_file.exists():
                    with open(ms_meta_file) as f:
                        ms_meta = json.load(f)
                    if ms_meta.get('sar_aligned'):
                        sar_aligned_count += 1
            else:
                failed_count += 1

            print(f"\n  Progress: {idx}/{len(cities_to_process)} | Success: {success_count} | Failed: {failed_count} | SAR-aligned: {sar_aligned_count}")

        log_data = {
            'cell_id': CELL_ID,
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'cities_processed': len(cities_to_process),
            'success_count': success_count,
            'failed_count': failed_count,
            'sar_aligned_count': sar_aligned_count,
            'metadata_location': str(_MS_METADATA_DIR)
        }

        with open(CELL_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n{'='*80}")
        print(f"MULTISPECTRAL SCENE DISCOVERY COMPLETE")
        print(f"{'='*80}")
        print(f"Cities processed: {len(cities_to_process)}")
        print(f"Success: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"SAR-aligned: {sar_aligned_count}")
        print(f"Metadata: {_MS_METADATA_DIR}")

    except Exception as e:
        log_data = {
            'cell_id': CELL_ID,
            'status': 'FAILED',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        with open(CELL_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        sys.stdout = logger.terminal
        logger.close()

    print("Cell 14A1-MS: Multispectral scene discovery complete")
