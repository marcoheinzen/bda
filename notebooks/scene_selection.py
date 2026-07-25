# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
scene_selection.py
Scene selection functions for InSAR pair selection, biweekly chains, baselines.
Extracted from Cell 14B1.

Notebook usage:
    from scene_selection import init as init_scene_selection
    init_scene_selection(
        cities_dir=CITIES_DIR,
        sar_metadata_dir=SAR_METADATA_DIR,
        data_root=DATA_ROOT,
        min_temporal_baseline=MIN_TEMPORAL_BASELINE,
        max_temporal_baseline=MAX_TEMPORAL_BASELINE,
        prebattle_tolerance_days=PREBATTLE_TOLERANCE_DAYS,
    )

    # Then use functions directly:
    from scene_selection import select_scene_pair, select_biweekly_chain, ...
"""

import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from dateutil.relativedelta import relativedelta

from aoi_date_extend_loader import load_aoi

try:
    import asf_search as asf
    ASF_AVAILABLE = True
except ImportError:
    ASF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module globals - set by init()
# ---------------------------------------------------------------------------
_CITIES_DIR = None
_SAR_METADATA_DIR = None
_DATA_ROOT = None
_MIN_TEMPORAL_BASELINE = 10
_MAX_TEMPORAL_BASELINE = 24
_PREBATTLE_TOLERANCE_DAYS = 180
_PRE_MONTHS = 14


# ---------------------------------------------------------------------------
# Helper: replaces get_data_start_date()
# ---------------------------------------------------------------------------
def _get_pre_start_date(battle_start_str):
    """Dynamic pre-battle start: battle_start - _PRE_MONTHS months."""
    if isinstance(battle_start_str, str):
        dt = datetime.strptime(battle_start_str[:10], '%Y-%m-%d')
    else:
        dt = battle_start_str
    return (dt - relativedelta(months=_PRE_MONTHS)).strftime('%Y-%m-%d')


# ---------------------------------------------------------------------------
# City info / scene loading
# ---------------------------------------------------------------------------
def get_city_info(city_name):
    """Load city boundary and battle dates from AOI.geojson"""

    props = load_aoi(city_name, _CITIES_DIR)
    city_geom = props.geometry

    battle_start = None
    battle_stop = None
    tier = props.get('tier', 3)

    if 'battle_start' in props and props['battle_start']:
        val = str(props['battle_start'])
        try:
            if 'T' in val:
                battle_start = datetime.fromisoformat(val.replace('Z', '+00:00')).replace(tzinfo=None)
            else:
                battle_start = datetime.strptime(val[:10], '%Y-%m-%d')
        except:
            pass

    if 'battle_stop' in props and props['battle_stop']:
        val = str(props['battle_stop'])
        if val == 'ongoing':
            battle_stop = None
        else:
            try:
                if 'T' in val:
                    battle_stop = datetime.fromisoformat(val.replace('Z', '+00:00')).replace(tzinfo=None)
                else:
                    battle_stop = datetime.strptime(val[:10], '%Y-%m-%d')
            except:
                pass

    return {
        'name': city_name,
        'geometry': city_geom,
        'bounds': city_geom.bounds,
        'centroid': city_geom.centroid,
        'battle_start': battle_start,
        'battle_stop': battle_stop,
        'tier': tier
    }


def load_discovered_scenes(city_name):
    """Load scenes discovered by Cell 14A1 for a city"""
    metadata_file = _SAR_METADATA_DIR / f"{city_name}_scene_metadata.json"

    if not metadata_file.exists():
        return None

    with open(metadata_file, 'r') as f:
        data = json.load(f)

    return data


# ---------------------------------------------------------------------------
# Copernicus search
# ---------------------------------------------------------------------------
def search_copernicus_scenes(start_date, end_date, bounds, product_type="SLC", orbit_direction=None, relative_orbit=None):
    """
    Search Copernicus Data Space API for Sentinel-1 scenes.
    EXACT copy of working logic from Cell 14A1.
    """
    from shapely.geometry import shape

    minx, miny, maxx, maxy = bounds
    bbox_wkt = f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"

    catalog_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    if isinstance(start_date, str):
        start_str = start_date
    else:
        start_str = start_date.strftime('%Y-%m-%d')

    if isinstance(end_date, str):
        end_str = end_date
    else:
        end_str = end_date.strftime('%Y-%m-%d')

    filter_query = (
        f"Collection/Name eq 'SENTINEL-1' "
        f"and ContentDate/Start ge {start_str}T00:00:00.000Z "
        f"and ContentDate/Start le {end_str}T23:59:59.999Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')"
    )

    params = {
        '$filter': filter_query,
        '$top': 1000,
        '$orderby': 'ContentDate/Start desc',
        '$expand': 'Attributes'
    }

    print(f"    Copernicus query: {start_str} to {end_str}")

    try:
        response = requests.get(catalog_url, params=params, timeout=120)

        if response.status_code != 200:
            print(f"    ERROR: HTTP {response.status_code}")
            return []

        results = response.json()
        products = results.get('value', [])
        print(f"    Found {len(products)} raw Sentinel-1 products")

        scenes = []
        for product in products:
            name = product.get('Name', '')

            if product_type == "SLC":
                if '_SLC_' not in name.upper() or 'IW' not in name:
                    continue
            elif product_type == "GRD":
                if '_GRD' not in name.upper():
                    continue
            elif product_type == "CARD":
                if 'CARD-BS' not in name.upper():
                    continue

            attributes = product.get('Attributes', [])
            rel_orbit = None
            orbit_dir = None

            for attr in attributes:
                attr_name = attr.get('Name')
                attr_value = attr.get('Value')

                if attr_name == 'relativeOrbitNumber':
                    rel_orbit = int(attr_value) if attr_value else None
                elif attr_name == 'orbitDirection':
                    orbit_dir = attr_value

            if orbit_direction and orbit_dir != orbit_direction:
                continue
            if relative_orbit and rel_orbit != relative_orbit:
                continue

            geofootprint = product.get('GeoFootprint')

            scene_info = {
                'name': name,
                'id': product.get('Id'),
                'date': product.get('ContentDate', {}).get('Start', ''),
                'relative_orbit': rel_orbit,
                'orbit_direction': orbit_dir,
                'size': product.get('ContentLength', 0),
                'geofootprint': geofootprint,
                'source': 'copernicus'
            }

            scenes.append(scene_info)

        print(f"    Filtered to {len(scenes)} {product_type} scenes")
        return scenes

    except requests.exceptions.Timeout:
        print(f"    ERROR: Request timed out")
        return []
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        return []


def filter_scenes_by_footprint(scenes_list, city_geometry, verbose=True):
    """Filter scenes that fully contain the city geometry (from Cell 14A1)"""
    if not scenes_list:
        return []

    if verbose:
        print(f"\n    Checking footprint coverage for {len(scenes_list)} scenes...")

    from shapely.geometry import shape

    filtered_scenes = []

    for scene in scenes_list:
        geofootprint = scene.get('geofootprint')

        if not geofootprint:
            continue

        try:
            footprint_geom = shape(geofootprint)

            if footprint_geom.contains(city_geometry):
                filtered_scenes.append(scene)

        except Exception as e:
            continue

    if verbose:
        print(f"    Footprint filtering: {len(filtered_scenes)}/{len(scenes_list)} scenes valid")

    return filtered_scenes


# ---------------------------------------------------------------------------
# ASF search
# ---------------------------------------------------------------------------
def query_asf_scenes_for_city(city_info, start_date, end_date, orbit=None):
    """Query ASF (Alaska Satellite Facility) for SLC scenes - fallback"""

    if not ASF_AVAILABLE:
        print(f"    ASF not available")
        return []

    bounds = city_info['bounds']

    if isinstance(start_date, str):
        start_str = start_date
    else:
        start_str = start_date.strftime('%Y-%m-%d')

    if isinstance(end_date, str):
        end_str = end_date
    else:
        end_str = end_date.strftime('%Y-%m-%d')

    print(f"    ASF query: {start_str} to {end_str}")

    try:
        results = asf.geo_search(
            platform=[asf.PLATFORM.SENTINEL1],
            processingLevel=[asf.PRODUCT_TYPE.SLC],
            start=start_str,
            end=end_str,
            intersectsWith=f"POLYGON(({bounds[0]} {bounds[1]},{bounds[2]} {bounds[1]},{bounds[2]} {bounds[3]},{bounds[0]} {bounds[3]},{bounds[0]} {bounds[1]}))"
        )

        scenes = []
        for r in results:
            props = r.properties

            if orbit and props.get('pathNumber') != orbit:
                continue

            scenes.append({
                'id': props.get('fileID', ''),
                'name': props.get('sceneName', props.get('fileID', '')),
                'date': props.get('startTime', '')[:10],
                'relative_orbit': int(props.get('pathNumber', 0)) if props.get('pathNumber') else None,
                'orbit_direction': props.get('flightDirection', '').upper(),
                'size': props.get('bytes', 0),
                'url': props.get('url', ''),
                'source': 'asf'
            })

        print(f"    Found {len(scenes)} scenes from ASF")
        return scenes

    except Exception as e:
        print(f"    ASF query error: {e}")
        return []


def query_scenes_for_city(city_info, start_date, end_date, orbit=None):
    """Query Copernicus (primary) then ASF (fallback) for scenes"""

    print(f"  Querying catalogs...")

    bounds = city_info['bounds']
    city_geom = city_info['geometry']

    scenes = search_copernicus_scenes(start_date, end_date, bounds, "SLC", relative_orbit=orbit)

    if scenes:
        scenes = filter_scenes_by_footprint(scenes, city_geom, verbose=True)

    if not scenes and ASF_AVAILABLE:
        print(f"    Copernicus returned 0 valid scenes - trying ASF...")
        scenes = query_asf_scenes_for_city(city_info, start_date, end_date, orbit)

    return scenes


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------
def deduplicate_scenes_by_date(scenes):
    """Keep only largest scene per date to prevent PyGMTSAR multi-scene issues"""
    by_date = {}
    for s in scenes:
        date = s.get('date', s.get('beginningDateTime', ''))[:10]
        size = s.get('size', 0)
        if date not in by_date or size > by_date[date].get('size', 0):
            by_date[date] = s
    return list(by_date.values())


# ---------------------------------------------------------------------------
# Orbit file check
# ---------------------------------------------------------------------------
def check_orbit_available(scene_name, orbit_source_dir=None):
    """
    Check if orbit file exists for a scene BEFORE downloading.

    Returns:
        True if orbit available, False otherwise
    """
    if orbit_source_dir is None:
        orbit_source_dir = _DATA_ROOT / 'satellite' / 'sentinel_1_orbits'

    orbit_source_dir = Path(orbit_source_dir)

    parts = scene_name.split('_')
    if len(parts) < 5:
        return False

    satellite = parts[0]  # S1A, S1B, or S1C

    # Parse acquisition time
    acq_time_str = None
    for part in parts:
        if len(part) == 15 and 'T' in part:
            acq_time_str = part
            break

    if not acq_time_str:
        return False

    try:
        acq_time = datetime.strptime(acq_time_str, '%Y%m%dT%H%M%S')
    except:
        return False

    # Search for matching orbit file
    narrow_dir = orbit_source_dir / satellite / str(acq_time.year) / f'{acq_time.month:02d}'
    if narrow_dir.exists():
        orbit_files = list(narrow_dir.glob(f'{satellite}_OPER_AUX_POEORB_*.EOF.zip'))
    else:
        orbit_files = []

    for orbit_file in orbit_files:
        orbit_name = orbit_file.name
        try:
            validity_start_str = orbit_name.split('_V')[1].split('_')[0]
            validity_end_str = orbit_name.split('_V')[1].split('_')[1].split('.')[0]

            validity_start = datetime.strptime(validity_start_str, '%Y%m%dT%H%M%S')
            validity_end = datetime.strptime(validity_end_str, '%Y%m%dT%H%M%S')

            if validity_start <= acq_time <= validity_end:
                return True
        except:
            continue

    return False


# ---------------------------------------------------------------------------
# Scene selection
# ---------------------------------------------------------------------------
def select_scene_for_city(city_name, period, scene_index, orbit=None, orbit_direction=None, exclude_scenes=None):
    """
    Select a single scene for a city.

    Args:
        city_name: Name of the city
        period: 'prebattle' or 'postbattle'
        scene_index: 1 (anchor scene closest to battle) or 2 (pair scene 10-24 days away)
        orbit: Specific orbit number to use (None = auto-select best)
        orbit_direction: 'ASCENDING' or 'DESCENDING' (None = any, used for crossbattle consistency)
        exclude_scenes: List of scene IDs to exclude (already tried and failed)

    Returns:
        scene_metadata dict or None if no suitable scene found
    """
    exclude_scenes = exclude_scenes or []

    city_info = get_city_info(city_name)

    if city_info['battle_start'] is None:
        print(f"  ERROR: No battle start date for {city_name}")
        return None

    discovered = load_discovered_scenes(city_name)
    scenes = []
    available_orbits = []

    if discovered:
        orbits_data = discovered.get('orbits', {})
        common_orbits = discovered.get('common_orbits', [])

        if orbits_data and common_orbits:
            print(f"  Using discovered scenes from Cell 14A1")
            print(f"    Common orbits: {common_orbits}")

            if orbit:
                orbit_key = str(orbit)
                if orbit_key in orbits_data:
                    orbit_data = orbits_data[orbit_key]
                    if period == 'prebattle':
                        scenes = orbit_data.get('pre_scenes', [])
                    else:
                        scenes = orbit_data.get('post_scenes', [])
                    print(f"    Found {len(scenes)} {period} scenes for orbit {orbit}")
            else:
                for orbit_key in [str(o) for o in common_orbits]:
                    if orbit_key in orbits_data:
                        orbit_data = orbits_data[orbit_key]
                        if period == 'prebattle':
                            scenes.extend(orbit_data.get('pre_scenes', []))
                        else:
                            scenes.extend(orbit_data.get('post_scenes', []))
                print(f"    Found {len(scenes)} {period} scenes total")

            # DEDUP FIX: Remove duplicate scenes per date
            scenes = deduplicate_scenes_by_date(scenes)
            print(f"    After dedup: {len(scenes)} unique dates")

            available_orbits = common_orbits

    if not scenes:
        print(f"  No discovered scenes - querying catalogs...")

        if period == 'prebattle':
            start_date = datetime.strptime(_get_pre_start_date(city_info['battle_start']), '%Y-%m-%d')
            end_date = city_info['battle_start']
        elif period == 'postbattle':
            if city_info['battle_stop']:
                start_date = city_info['battle_stop']
                end_date = datetime.now()
            else:
                start_date = city_info['battle_start']
                end_date = datetime.now()
        else:
            print(f"    ERROR: Unknown period '{period}'")
            return None

        print(f"    DEBUG query: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        scenes = query_scenes_for_city(city_info, start_date, end_date, orbit)
        scenes = deduplicate_scenes_by_date(scenes)  # DEDUP FIX
        available_orbits = list(set(s.get('relative_orbit') for s in scenes if s.get('relative_orbit')))
        print(f"  Found {len(scenes)} scenes, orbits: {available_orbits}")

    scenes = [s for s in scenes if s['id'] not in exclude_scenes]

    # Filter to scenes with orbit files available
    scenes_with_orbits = [s for s in scenes if check_orbit_available(s.get('Name', s.get('name', '')))]
    if scenes_with_orbits:
        print(f"    Filtered to scenes with orbits: {len(scenes_with_orbits)}/{len(scenes)}")
        scenes = scenes_with_orbits
    elif scenes:
        print(f"    WARNING: No scenes have orbit files available ({len(scenes)} scenes)")

    if orbit:
        scenes = [s for s in scenes if s.get('relative_orbit') == orbit]

    if orbit_direction:
        before_filter = len(scenes)
        scenes = [s for s in scenes if s.get('orbit_direction') == orbit_direction]
        print(f"    Orbit direction filter ({orbit_direction}): {len(scenes)}/{before_filter} scenes")

    if not scenes:
        print(f"  No scenes available for {city_name} {period} (orbit={orbit}, direction={orbit_direction})")
        return None

    if period == 'prebattle':
        target_date = city_info['battle_start']
        min_date = datetime.strptime(_get_pre_start_date(city_info['battle_start']), '%Y-%m-%d')
        max_date = target_date
    elif period == 'postbattle':
        if city_info['battle_stop']:
            target_date = city_info['battle_stop']
            min_date = target_date
            max_date = datetime.now()
        else:
            target_date = datetime.now()
            min_date = city_info['battle_start']
            max_date = datetime.now()
    else:
        print(f"    ERROR: Unknown period '{period}'")
        return None

    print(f"    DEBUG filter: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}, target={target_date.strftime('%Y-%m-%d')}")

    valid_scenes = []
    for scene in scenes:
        scene_date_str = scene['date'][:10]
        scene_date = datetime.strptime(scene_date_str, '%Y-%m-%d')

        if min_date <= scene_date <= max_date:
            days_from_target = abs((scene_date - target_date).days)
            valid_scenes.append({
                **scene,
                'scene_date': scene_date,
                'days_from_target': days_from_target
            })

    if not valid_scenes:
        print(f"  No scenes in valid date range for {city_name} {period}")
        print(f"    Expected: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        return None

    valid_scenes.sort(key=lambda x: x['days_from_target'])

    if scene_index == 1:
        selected = valid_scenes[0]
        print(f"  Selected anchor scene: {selected['name'][:40]}...")
        print(f"    Date: {selected['scene_date'].strftime('%Y-%m-%d')}, {selected['days_from_target']} days from battle")
        print(f"    Orbit direction: {selected.get('orbit_direction', 'unknown')}")
        return selected

    elif scene_index == 2:
        print(f"  ERROR: scene_index=2 requires anchor_scene parameter")
        print(f"    Use select_pair_scene() instead")
        return None

    return None


def select_pair_scene(city_name, period, anchor_scene, orbit=None, orbit_direction=None, exclude_scenes=None):
    """
    Select a pair scene (scene_index=2) that forms a valid InSAR pair with anchor.
    """
    exclude_scenes = exclude_scenes or []

    anchor_date = anchor_scene['scene_date']
    anchor_orbit = anchor_scene.get('relative_orbit')
    anchor_direction = anchor_scene.get('orbit_direction')

    if orbit is None:
        orbit = anchor_orbit
    elif orbit != anchor_orbit:
        print(f"  WARNING: Orbit mismatch - anchor={anchor_orbit}, requested={orbit}")
        orbit = anchor_orbit

    if orbit_direction is None:
        orbit_direction = anchor_direction
    elif orbit_direction != anchor_direction:
        print(f"  WARNING: Direction mismatch - anchor={anchor_direction}, requested={orbit_direction}")
        orbit_direction = anchor_direction

    city_info = get_city_info(city_name)

    discovered = load_discovered_scenes(city_name)
    scenes = []

    if discovered:
        orbits_data = discovered.get('orbits', {})
        orbit_key = str(orbit)

        if orbit_key in orbits_data:
            orbit_data = orbits_data[orbit_key]
            if period == 'prebattle':
                scenes = orbit_data.get('pre_scenes', [])
            else:
                scenes = orbit_data.get('post_scenes', [])

    # DEDUP FIX
    if scenes:
        scenes = deduplicate_scenes_by_date(scenes)

    if not scenes:
        if period == 'prebattle':
            start_date = datetime.strptime(_get_pre_start_date(city_info['battle_start']), '%Y-%m-%d')
            end_date = city_info['battle_start']
        elif period == 'postbattle':
            if city_info['battle_stop']:
                start_date = city_info['battle_stop']
                end_date = datetime.now()
            else:
                start_date = city_info['battle_start']
                end_date = datetime.now()

        scenes = query_scenes_for_city(city_info, start_date, end_date, orbit)
        scenes = deduplicate_scenes_by_date(scenes)  # DEDUP FIX

    scenes = [s for s in scenes if s['id'] not in exclude_scenes]
    scenes = [s for s in scenes if s['id'] != anchor_scene['id']]

    if orbit:
        scenes = [s for s in scenes if s.get('relative_orbit') == orbit]

    if orbit_direction:
        before_filter = len(scenes)
        scenes = [s for s in scenes if s.get('orbit_direction') == orbit_direction]
        if before_filter != len(scenes):
            print(f"    Orbit direction filter ({orbit_direction}): {len(scenes)}/{before_filter} scenes")

    valid_pairs = []
    for scene in scenes:
        scene_date_str = scene['date'][:10]
        scene_date = datetime.strptime(scene_date_str, '%Y-%m-%d')

        temporal_baseline = abs((scene_date - anchor_date).days)

        if _MIN_TEMPORAL_BASELINE <= temporal_baseline <= _MAX_TEMPORAL_BASELINE:
            valid_pairs.append({
                **scene,
                'scene_date': scene_date,
                'temporal_baseline': temporal_baseline
            })

    if not valid_pairs:
        print(f"  No valid pair scenes found (need {_MIN_TEMPORAL_BASELINE}-{_MAX_TEMPORAL_BASELINE} days from anchor)")
        return None

    valid_pairs.sort(key=lambda x: abs(x['temporal_baseline'] - 12))

    selected = valid_pairs[0]
    print(f"  Selected pair scene: {selected['name'][:40]}...")
    print(f"    Date: {selected['scene_date'].strftime('%Y-%m-%d')}, temporal baseline: {selected['temporal_baseline']} days")
    print(f"    Orbit direction: {selected.get('orbit_direction', 'unknown')}")

    return selected


def select_scene_pair(city_name, period, orbit=None, orbit_direction=None, exclude_scenes=None):
    """
    Select a complete scene pair (anchor + pair) for a city/period.

    Returns:
        (anchor_scene, pair_scene, temporal_baseline) or (None, None, None)
    """
    exclude_scenes = exclude_scenes or []

    print(f"\n  Selecting {period} pair for {city_name} (orbit={orbit or 'auto'}, direction={orbit_direction or 'any'})...")

    anchor = select_scene_for_city(city_name, period, scene_index=1, orbit=orbit, orbit_direction=orbit_direction, exclude_scenes=exclude_scenes)

    if anchor is None:
        return None, None, None

    selected_orbit = anchor.get('relative_orbit')
    selected_direction = anchor.get('orbit_direction')

    pair = select_pair_scene(city_name, period, anchor, orbit=selected_orbit, orbit_direction=selected_direction, exclude_scenes=exclude_scenes)

    if pair is None:
        return None, None, None

    if anchor['scene_date'] < pair['scene_date']:
        scene1, scene2 = anchor, pair
    else:
        scene1, scene2 = pair, anchor

    temporal_baseline = abs((scene2['scene_date'] - scene1['scene_date']).days)

    print(f"  Pair selected:")
    print(f"    Scene 1: {scene1['scene_date'].strftime('%Y-%m-%d')}")
    print(f"    Scene 2: {scene2['scene_date'].strftime('%Y-%m-%d')}")
    print(f"    Temporal baseline: {temporal_baseline} days")
    print(f"    Orbit: {selected_orbit}")
    print(f"    Direction: {selected_direction}")

    return scene1, scene2, temporal_baseline


def get_available_orbits_for_city(city_name):
    """Get list of orbits that have scene coverage for a city"""
    discovered = load_discovered_scenes(city_name)

    if discovered:
        common_orbits = discovered.get('common_orbits', [])
        if common_orbits:
            return [int(o) for o in common_orbits]

    print(f"    No discovered orbits - querying catalog...")

    city_info = get_city_info(city_name)

    battle_start = city_info.get('battle_start')
    if battle_start:
        search_start = battle_start - timedelta(days=_PREBATTLE_TOLERANCE_DAYS)
        search_end = battle_start
    else:
        search_start = datetime.strptime(_get_pre_start_date('2022-02-24'), '%Y-%m-%d')
        search_end = datetime.now()

    scenes = query_scenes_for_city(city_info, search_start, search_end, orbit=None)

    if scenes:
        orbits = list(set(int(s.get('relative_orbit')) for s in scenes if s.get('relative_orbit')))
        orbits.sort()
        return orbits

    return []


def select_biweekly_chain(city_name, orbit=None, orbit_direction=None, start_date=None, end_date=None, use_all_scenes=False):
    """
    Select scenes every ~12 days within a date range.
    Defaults to battle_start - 24d to battle_end + 24d (captures bridge pairs).
    For ongoing conflicts, end_date defaults to now.
    Uses discovered scenes from Cell 14A1, recommended orbit if not specified.
    Returns ordered list of scenes for consecutive-pair coherence computation.
    """
    discovered = load_discovered_scenes(city_name)
    if not discovered:
        print(f"  ERROR: No discovered scenes for {city_name}")
        return []

    if orbit is None:
        orbit = discovered.get('recommended_orbit')
    orbit_key = str(orbit)

    orbits_data = discovered.get('orbits', {})
    if orbit_key not in orbits_data:
        print(f"  ERROR: Orbit {orbit} not found in discovery metadata")
        return []

    orbit_data = orbits_data[orbit_key]

    # Default date range: battle period with 24d buffer for bridge pairs
    if start_date is None or end_date is None:
        city_info = get_city_info(city_name)
        battle_start = city_info.get('battle_start')
        battle_stop = city_info.get('battle_stop')
        if start_date is None:
            if battle_start:
                start_date = battle_start - timedelta(days=24)
            else:
                start_date = datetime.strptime(_get_pre_start_date('2022-02-24'), '%Y-%m-%d')
        if end_date is None:
            if battle_stop:
                end_date = battle_stop + timedelta(days=24)
            else:
                end_date = datetime.now()

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date[:10], '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date[:10], '%Y-%m-%d')

    # collect all scenes across periods
    all_scenes = []
    all_scenes.extend(orbit_data.get('pre_scenes', []))
    all_scenes.extend(orbit_data.get('battle_scenes', []))
    all_scenes.extend(orbit_data.get('post_scenes', []))

    if not all_scenes:
        print(f"  ERROR: No scenes for orbit {orbit}")
        return []

    # dedup and parse dates
    all_scenes = deduplicate_scenes_by_date(all_scenes)

    for s in all_scenes:
        s['scene_date'] = datetime.strptime(s['date'][:10], '%Y-%m-%d')

    all_scenes.sort(key=lambda x: x['scene_date'])

    # filter to date range
    all_scenes = [s for s in all_scenes if start_date <= s['scene_date'] <= end_date]
    print(f"  Date filter: {start_date.date()} to {end_date.date()} -> {len(all_scenes)} scenes")

    # filter orbit direction if specified
    if orbit_direction:
        all_scenes = [s for s in all_scenes if s.get('orbit_direction') == orbit_direction]
    elif all_scenes:
        # auto-detect dominant direction
        directions = [s.get('orbit_direction') for s in all_scenes if s.get('orbit_direction')]
        if directions:
            dominant = max(set(directions), key=directions.count)
            all_scenes = [s for s in all_scenes if s.get('orbit_direction') == dominant]
            orbit_direction = dominant

    # filter to scenes with orbit files
    scenes_with_orbits = [s for s in all_scenes if check_orbit_available(s.get('name', s.get('Name', '')))]
    if scenes_with_orbits:
        print(f"  Scenes with orbits: {len(scenes_with_orbits)}/{len(all_scenes)}")
        all_scenes = scenes_with_orbits
    else:
        print(f"  WARNING: No orbit files found - using all {len(all_scenes)} scenes")

    if len(all_scenes) < 2:
        print(f"  ERROR: Need at least 2 scenes, got {len(all_scenes)}")
        return []

    if use_all_scenes:
        # short conflict mode: use every available scene, no gap filtering
        chain = list(all_scenes)
        print(f"  Short conflict mode: using ALL {len(chain)} scenes (no 12-day subsampling)")
    else:
        # greedy chain: pick scenes ~12 days apart
        # start from first scene, always pick next scene closest to +12 days
        chain = [all_scenes[0]]
        remaining = all_scenes[1:]

        while remaining:
            last_date = chain[-1]['scene_date']
            target_date = last_date + timedelta(days=12)

            # find scene closest to target (accept 6-18 day gap)
            best = None
            best_diff = float('inf')

            for s in remaining:
                gap = (s['scene_date'] - last_date).days
                if gap < 6:
                    continue
                diff = abs(gap - 12)
                if diff < best_diff:
                    best_diff = diff
                    best = s

            if best is None:
                break

            chain.append(best)
            remaining.remove(best)

    # summary
    dates = [s['scene_date'] for s in chain]
    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]

    print(f"\n  Biweekly chain for {city_name}:")
    print(f"    Orbit: {orbit} ({orbit_direction})")
    print(f"    Scenes: {len(chain)}")
    print(f"    Pairs:  {len(chain)-1}")
    print(f"    Range:  {dates[0].date()} to {dates[-1].date()}")
    if gaps:
        print(f"    Gaps:   min={min(gaps)}d, max={max(gaps)}d, mean={sum(gaps)/len(gaps):.0f}d")

    return chain


def select_prebattle_baseline(city_name, n_scenes=5, orbit=None, orbit_direction=None):
    """
    Select n_scenes prebattle scenes forming a backward chain from battle_start.
    Scene closest to battle_start is the anchor, then we step backward ~12d each.
    Returns ordered list (oldest first) for consecutive-pair coherence baseline.
    Compatible with existing 2-scene prebattle pair (those 2 are the last 2 in chain).
    """
    discovered = load_discovered_scenes(city_name)
    if not discovered:
        print(f"  ERROR: No discovered scenes for {city_name}")
        return []

    if orbit is None:
        orbit = discovered.get('recommended_orbit')
    orbit_key = str(orbit)

    orbits_data = discovered.get('orbits', {})
    if orbit_key not in orbits_data:
        print(f"  ERROR: Orbit {orbit} not found in discovery metadata")
        return []

    orbit_data = orbits_data[orbit_key]
    city_info = get_city_info(city_name)
    battle_start = city_info.get('battle_start')

    if not battle_start:
        print(f"  ERROR: No battle_start for {city_name}")
        return []

    # collect prebattle scenes only
    pre_scenes = list(orbit_data.get('pre_scenes', []))
    if not pre_scenes:
        print(f"  ERROR: No prebattle scenes for orbit {orbit}")
        return []

    # dedup and parse dates
    pre_scenes = deduplicate_scenes_by_date(pre_scenes)
    for s in pre_scenes:
        s['scene_date'] = datetime.strptime(s['date'][:10], '%Y-%m-%d')
    pre_scenes.sort(key=lambda x: x['scene_date'])

    # filter orbit direction
    if orbit_direction:
        pre_scenes = [s for s in pre_scenes if s.get('orbit_direction') == orbit_direction]
    elif pre_scenes:
        directions = [s.get('orbit_direction') for s in pre_scenes if s.get('orbit_direction')]
        if directions:
            dominant = max(set(directions), key=directions.count)
            pre_scenes = [s for s in pre_scenes if s.get('orbit_direction') == dominant]
            orbit_direction = dominant

    # filter to scenes with orbit files
    scenes_with_orbits = [s for s in pre_scenes if check_orbit_available(s.get('name', s.get('Name', '')))]
    if scenes_with_orbits:
        print(f"  Prebattle scenes with orbits: {len(scenes_with_orbits)}/{len(pre_scenes)}")
        pre_scenes = scenes_with_orbits
    else:
        print(f"  WARNING: No orbit files found - using all {len(pre_scenes)} prebattle scenes")

    if len(pre_scenes) < 2:
        print(f"  ERROR: Need at least 2 prebattle scenes, got {len(pre_scenes)}")
        return []

    # greedy backward chain from anchor (scene closest to battle_start)
    # anchor = latest prebattle scene
    pre_scenes_sorted_desc = sorted(pre_scenes, key=lambda x: x['scene_date'], reverse=True)
    anchor = pre_scenes_sorted_desc[0]
    remaining = pre_scenes_sorted_desc[1:]

    chain = [anchor]

    while len(chain) < n_scenes and remaining:
        last_date = chain[-1]['scene_date']
        target_date = last_date - timedelta(days=12)

        best = None
        best_diff = float('inf')

        for s in remaining:
            gap = (last_date - s['scene_date']).days
            if gap < 6:
                continue
            diff = abs(gap - 12)
            if diff < best_diff:
                best_diff = diff
                best = s

        if best is None:
            break

        chain.append(best)
        remaining.remove(best)

    # reverse so oldest first (consistent with biweekly chain)
    chain.reverse()

    # summary
    dates = [s['scene_date'] for s in chain]
    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]

    print(f"\n  Prebattle baseline for {city_name}:")
    print(f"    Orbit: {orbit} ({orbit_direction})")
    print(f"    Scenes: {len(chain)} (requested {n_scenes})")
    print(f"    Pairs:  {len(chain)-1}")
    print(f"    Range:  {dates[0].date()} to {dates[-1].date()}")
    print(f"    Days before battle: {(battle_start - dates[-1]).days}")
    if gaps:
        print(f"    Gaps:   min={min(gaps)}d, max={max(gaps)}d, mean={sum(gaps)/len(gaps):.0f}d")

    return chain


# ---------------------------------------------------------------------------
# init() - called from notebook cell to set globals
# ---------------------------------------------------------------------------
def init(cities_dir, sar_metadata_dir, data_root,
         min_temporal_baseline=10, max_temporal_baseline=24,
         prebattle_tolerance_days=180, pre_months=14):
    """
    Args:
        cities_dir:               Path - CITIES_DIR
        sar_metadata_dir:         Path - SAR_METADATA_DIR
        data_root:                Path - DATA_ROOT (for orbit file lookup)
        min_temporal_baseline:    int  - MIN_TEMPORAL_BASELINE
        max_temporal_baseline:    int  - MAX_TEMPORAL_BASELINE
        prebattle_tolerance_days: int  - PREBATTLE_TOLERANCE_DAYS
        pre_months:               int  - months before battle_start for lookback
    """
    global _CITIES_DIR, _SAR_METADATA_DIR, _DATA_ROOT
    global _MIN_TEMPORAL_BASELINE, _MAX_TEMPORAL_BASELINE
    global _PREBATTLE_TOLERANCE_DAYS, _PRE_MONTHS

    _CITIES_DIR = Path(cities_dir)
    _SAR_METADATA_DIR = Path(sar_metadata_dir)
    _DATA_ROOT = Path(data_root)
    _MIN_TEMPORAL_BASELINE = min_temporal_baseline
    _MAX_TEMPORAL_BASELINE = max_temporal_baseline
    _PREBATTLE_TOLERANCE_DAYS = prebattle_tolerance_days
    _PRE_MONTHS = pre_months

    print("\n" + "=" * 80)
    print("CELL 14B1: SCENE SELECTION FUNCTIONS")
    print("=" * 80)
    print(f"  asf_search: {'AVAILABLE' if ASF_AVAILABLE else 'NOT INSTALLED'}")

    print("\n  Functions defined:")
    print("    get_city_info(city_name)")
    print("    load_discovered_scenes(city_name)")
    print("    search_copernicus_scenes(start_date, end_date, bounds, ...)")
    print("    filter_scenes_by_footprint(scenes_list, city_geometry, verbose)")
    print("    query_asf_scenes_for_city(city_info, start_date, end_date, orbit)")
    print("    query_scenes_for_city(city_info, start_date, end_date, orbit)")
    print("    deduplicate_scenes_by_date(scenes)")
    print("    select_scene_for_city(city_name, period, scene_index, orbit, orbit_direction, exclude_scenes)")
    print("    select_pair_scene(city_name, period, anchor_scene, orbit, orbit_direction, exclude_scenes)")
    print("    select_scene_pair(city_name, period, orbit, orbit_direction, exclude_scenes)")
    print("    get_available_orbits_for_city(city_name)")
    print("    select_biweekly_chain(city_name, orbit, orbit_direction, start_date, end_date)")
    print("    select_prebattle_baseline(city_name, n_scenes, orbit, orbit_direction)")

    print("\n" + "=" * 80)
    print("CELL 14B1 COMPLETE")
    print("=" * 80)
