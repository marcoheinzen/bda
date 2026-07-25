# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
slc_scene_discovery.py
Discovers ALL Sentinel-1 SLC scenes for all cities, tests all orbits.
Stores metadata as {city}_scene_metadata.json.

Notebook usage:
    from slc_scene_discovery import run as run_scene_discovery
    run_scene_discovery(
        cities_dir=CITIES_DIR,
        logs_dir=LOGS_DIR,
        output_dir=SAR_METADATA_DIR,
        progress_file=SLC_PROGRESS_FILE,
        product_types=PRODUCT_TYPES,
        force_rerun=FORCE_RERUN,
    )
"""

import json
import math
import sys
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from aoi_date_extend_loader import (
    load_city_boundary_with_dates,
    compute_temporal_windows,
    discover_cities,
)


# ---------------------------------------------------------------------------
# DualLogger (same as global_setup.py)
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
# Copernicus scene search
# ---------------------------------------------------------------------------
def search_copernicus_scenes(start_date, end_date, bounds, product_type, orbit_direction=None, relative_orbit=None):
    minx, miny, maxx, maxy = bounds
    bbox_wkt = f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"

    catalog_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    filter_query = (
        f"Collection/Name eq 'SENTINEL-1' "
        f"and ContentDate/Start ge {start_date.strftime('%Y-%m-%d')}T00:00:00.000Z "
        f"and ContentDate/Start le {end_date.strftime('%Y-%m-%d')}T23:59:59.999Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')"
    )

    all_scenes = []
    seen_ids = set()

    PAGE_SIZE = 1000
    MAX_SKIP = 10000

    for order in ['asc', 'desc']:
        skip = 0
        page = 0

        try:
            while skip <= MAX_SKIP:
                page += 1
                params = {
                    '$filter': filter_query,
                    '$top': PAGE_SIZE,
                    '$skip': skip,
                    '$orderby': f'ContentDate/Start {order}',
                    '$expand': 'Attributes'
                }

                if page == 1:
                    print(f"    Querying Copernicus ({order})...")
                else:
                    print(f"    Copernicus page {page} ({order}, skip={skip})...")

                response = None
                for _retry in range(3):
                    try:
                        response = requests.get(catalog_url, params=params, timeout=180)
                        response.raise_for_status()
                        break
                    except Exception as _e:
                        _wait = 30 * (_retry + 1)
                        print(f"    Copernicus retry {_retry+1}: {_e}")
                        if _retry < 2:
                            print(f"    Waiting {_wait}s...")
                            import time
                            time.sleep(_wait)
                if response is None:
                    print(f"    Copernicus ({order}): all retries failed at page {page}")
                    break

                results = response.json()

                products = results.get('value', [])

                if not products:
                    break

                for product in products:
                    pid = product.get('Id')
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

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
                        'geofootprint': geofootprint
                    }

                    all_scenes.append(scene_info)

                # stop if last page was not full
                if len(products) < PAGE_SIZE:
                    break

                skip += PAGE_SIZE

            print(f"    Copernicus ({order}): {page} pages fetched")

        except Exception as e:
            print(f"    Copernicus search error ({order}): {type(e).__name__}: {e}")

    print(f"  Total unique Copernicus {product_type} scenes: {len(all_scenes)}")

    return all_scenes


# ---------------------------------------------------------------------------
# ASF scene search
# ---------------------------------------------------------------------------
def search_asf_scenes(start_date, end_date, bounds, product_type):
    try:
        import asf_search as asf
    except ImportError:
        print("    ASF search not available (asf_search not installed)")
        return []

    minx, miny, maxx, maxy = bounds
    wkt = f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}))"

    all_scenes = []
    seen_ids = set()

    for sort_order in ['asc', 'desc']:
        try:
            print(f"    Querying ASF ({sort_order})...")

            if product_type == "SLC":
                processing_level = [asf.PRODUCT_TYPE.SLC]
            elif product_type == "GRD":
                processing_level = [asf.PRODUCT_TYPE.GRD_HD, asf.PRODUCT_TYPE.GRD_MD]
            else:
                processing_level = None

            opts = {
                'platform': [asf.PLATFORM.SENTINEL1],
                'intersectsWith': wkt,
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'maxResults': 5000,
            }

            if processing_level:
                opts['processingLevel'] = processing_level

            results = asf.search(**opts)

            if sort_order == 'desc':
                results = sorted(results, key=lambda x: x.properties.get('startTime', ''), reverse=True)
            else:
                results = sorted(results, key=lambda x: x.properties.get('startTime', ''))

            print(f"    Found {len(results)} products ({sort_order})")

            for result in results:
                props = result.properties
                scene_id = props.get('sceneName', props.get('fileID', ''))

                if scene_id in seen_ids:
                    continue
                seen_ids.add(scene_id)

                scene_info = {
                    'name': scene_id,
                    'id': props.get('fileID', scene_id),
                    'date': props.get('startTime', '')[:19],
                    'relative_orbit': int(props.get('pathNumber')) if props.get('pathNumber') else None,
                    'orbit_direction': props.get('flightDirection', '').upper(),
                    'size': props.get('bytes', 0),
                    'url': props.get('url', ''),
                    'source': 'asf',
                    'type': product_type,
                    'geofootprint': result.geometry if hasattr(result, 'geometry') else None
                }

                all_scenes.append(scene_info)

        except Exception as e:
            print(f"    ASF search error ({sort_order}): {type(e).__name__}: {e}")

    print(f"  Total unique ASF {product_type} scenes: {len(all_scenes)}")

    return all_scenes


# ---------------------------------------------------------------------------
# Combined search
# ---------------------------------------------------------------------------
def search_all_sources(start_date, end_date, bounds, product_type, orbit_direction=None, relative_orbit=None):
    print(f"\n  Searching all sources: {start_date.date()} to {end_date.date()}")

    all_scenes = []
    seen_ids = set()

    copernicus_scenes = search_copernicus_scenes(start_date, end_date, bounds, product_type, orbit_direction, relative_orbit)

    for scene in copernicus_scenes:
        sid = scene.get('id') or scene.get('name')
        if sid not in seen_ids:
            seen_ids.add(sid)
            scene['source'] = 'copernicus'
            scene.setdefault('type', product_type)
            all_scenes.append(scene)

    asf_scenes = search_asf_scenes(start_date, end_date, bounds, product_type)

    for scene in asf_scenes:
        sid = scene.get('id') or scene.get('name')
        name_base = scene.get('name', '')[:50]

        already_have = False
        for existing in all_scenes:
            existing_name = existing.get('name', '')[:50]
            if name_base and existing_name and name_base == existing_name:
                already_have = True
                break

        if not already_have and sid not in seen_ids:
            seen_ids.add(sid)
            all_scenes.append(scene)

    print(f"  Combined total unique scenes: {len(all_scenes)}")

    return all_scenes


# ---------------------------------------------------------------------------
# Footprint filter
# ---------------------------------------------------------------------------
def filter_scenes_by_footprint(scenes_list, city_geometry, verbose=True):
    if not scenes_list:
        return []

    if verbose:
        print(f"\n  Checking footprint coverage for {len(scenes_list)} scenes...")

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
        print(f"  Footprint filtering: {len(filtered_scenes)}/{len(scenes_list)} scenes valid")

    return filtered_scenes


# ---------------------------------------------------------------------------
# Per-city discovery
# ---------------------------------------------------------------------------
def discover_scenes_for_city(city_name, cities_dir, output_dir, product_types, proximity_days=14, pre_months=14):
    print(f"\n{'='*80}")
    print(f"CITY: {city_name}")
    print(f"{'='*80}")

    try:
        gdf, pre_battle_date, post_battle_date, conflict_ongoing, tier = \
            load_city_boundary_with_dates(city_name, cities_dir)

        windows = compute_temporal_windows(
            pre_battle_date, post_battle_date, conflict_ongoing,
            proximity_days=proximity_days, pre_months=pre_months
        )

        city_geom = gdf.geometry.iloc[0]
        city_bounds = city_geom.bounds
        city_center = city_geom.centroid

        print(f"\n  City bounds: [{city_bounds[0]:.3f}, {city_bounds[1]:.3f}, {city_bounds[2]:.3f}, {city_bounds[3]:.3f}]")
        print(f"  City center: ({city_center.y:.3f}, {city_center.x:.3f})")

        # check bbox width vs S1 swath (250km) - warn if multi-scene needed
        _lat_mid = (city_bounds[1] + city_bounds[3]) / 2
        _lon_scale = math.cos(math.radians(_lat_mid)) * 111.32
        _lat_scale = 111.32
        _bbox_width_km = (city_bounds[2] - city_bounds[0]) * _lon_scale
        _bbox_height_km = (city_bounds[3] - city_bounds[1]) * _lat_scale
        print(f"  Bbox size: {_bbox_width_km:.1f} x {_bbox_height_km:.1f} km")
        if _bbox_width_km > 200 or _bbox_height_km > 200:
            print(f"  WARNING: Bbox exceeds 200km ({_bbox_width_km:.0f}x{_bbox_height_km:.0f}km).")
            print(f"           S1 IW swath width is ~250km. This AOI may require")
            print(f"           multi-scene mosaicking (not yet implemented).")
            print(f"           Future: NB03b/c for multi-swath/multi-scene processing.")

        print(f"\n  Tier: {tier}")
        print(f"  Battle dates:")
        print(f"    Start: {pre_battle_date.date()}")
        if post_battle_date:
            print(f"    End: {post_battle_date.date()}")
        elif conflict_ongoing:
            print(f"    End: ongoing")

        pre_battle_start = windows['pre_battle_start']
        pre_battle_end = windows['pre_battle_end']
        post_battle_start = windows['post_battle_start']
        post_battle_end = windows['post_battle_end']
        crossbattle_start = windows['crossbattle_start']
        crossbattle_end = windows['crossbattle_end']
        battle_duration_days = windows['battle_duration_days']
        max_temporal_baseline = windows['max_temporal_baseline_days']

        print(f"\n  Search windows (proximity: {proximity_days} days, pre: {pre_months} months):")
        print(f"    Pre-battle:  {pre_battle_start.date()} to {pre_battle_end.date()}")
        print(f"    Post-battle: {post_battle_start.date()} to {post_battle_end.date()}")
        print(f"    Battle duration: {battle_duration_days} days")

        pre_scenes_raw = []
        post_scenes_raw = []
        battle_scenes_raw = []

        for product_type in product_types:
            print(f"\n  Searching pre-battle {product_type} scenes...")
            pre_scenes_raw += search_all_sources(
                pre_battle_start,
                pre_battle_end,
                city_bounds,
                product_type
            )

            print(f"\n  Searching post-battle {product_type} scenes...")
            post_scenes_raw += search_all_sources(
                post_battle_start,
                post_battle_end,
                city_bounds,
                product_type
            )

            # Also discover all scenes during the battle period (crossbattle)
            print(f"\n  Searching crossbattle {product_type} scenes ({crossbattle_start.date()} to {crossbattle_end.date()})...")
            battle_raw = search_all_sources(
                crossbattle_start,
                crossbattle_end,
                city_bounds,
                product_type
            )
            battle_scenes_raw += battle_raw
            print(f"  Found {len(battle_raw)} crossbattle {product_type} scenes")

        print(f"\n  Total: {len(pre_scenes_raw)} pre, {len(post_scenes_raw)} post, {len(battle_scenes_raw)} battle scenes across {product_types}")

        if not pre_scenes_raw or not post_scenes_raw:
            raise ValueError("No scenes found in pre or post-battle period")

        pre_df = pd.DataFrame(pre_scenes_raw)
        post_df = pd.DataFrame(post_scenes_raw)

        pre_orbits = set(pre_df['relative_orbit'].dropna().unique())
        post_orbits = set(post_df['relative_orbit'].dropna().unique())
        common_orbits = pre_orbits & post_orbits

        battle_df = pd.DataFrame(battle_scenes_raw) if battle_scenes_raw else pd.DataFrame()

        print(f"\n  Pre-battle orbits: {sorted(pre_orbits)}")
        print(f"  Post-battle orbits: {sorted(post_orbits)}")
        if not battle_df.empty:
            battle_orbits = set(battle_df['relative_orbit'].dropna().unique())
            print(f"  Battle-period orbits: {sorted(battle_orbits)}")
        print(f"  Common orbits: {sorted(common_orbits)}")

        if not common_orbits:
            raise ValueError(f"No common orbits found between pre and post-battle periods")

        orbit_metadata = {}

        for orbit in sorted(common_orbits):
            print(f"\n  Testing orbit {orbit}...")

            pre_orbit_scenes = pre_df[pre_df['relative_orbit'] == orbit].to_dict('records')
            post_orbit_scenes = post_df[post_df['relative_orbit'] == orbit].to_dict('records')

            pre_valid = filter_scenes_by_footprint(pre_orbit_scenes, city_geom, verbose=False)
            post_valid = filter_scenes_by_footprint(post_orbit_scenes, city_geom, verbose=False)

            # Filter crossbattle scenes for this orbit
            battle_valid = []
            if not battle_df.empty:
                battle_orbit_scenes = battle_df[battle_df['relative_orbit'] == orbit].to_dict('records')
                battle_valid = filter_scenes_by_footprint(battle_orbit_scenes, city_geom, verbose=False)

            for scene in pre_valid:
                if 'geofootprint' in scene:
                    del scene['geofootprint']

            for scene in post_valid:
                if 'geofootprint' in scene:
                    del scene['geofootprint']

            for scene in battle_valid:
                if 'geofootprint' in scene:
                    del scene['geofootprint']

            pre_valid_sorted = sorted(pre_valid, key=lambda x: x['date'])
            post_valid_sorted = sorted(post_valid, key=lambda x: x['date'])
            battle_valid_sorted = sorted(battle_valid, key=lambda x: x['date'])

            battle_msg = f", {len(battle_valid_sorted)} battle" if battle_valid_sorted else ""
            print(f"    Orbit {orbit}: {len(pre_valid_sorted)} pre, {len(post_valid_sorted)} post{battle_msg} valid")

            orbit_entry = {
                'orbit_number': int(orbit),
                'pre_scenes_total': len(pre_orbit_scenes),
                'post_scenes_total': len(post_orbit_scenes),
                'pre_scenes_valid': len(pre_valid_sorted),
                'post_scenes_valid': len(post_valid_sorted),
                'pre_scenes': pre_valid_sorted,
                'post_scenes': post_valid_sorted,
                'total_valid_scenes': len(pre_valid_sorted) + len(post_valid_sorted)
            }

            if battle_valid_sorted:
                orbit_entry['battle_scenes_total'] = len(battle_orbit_scenes)
                orbit_entry['battle_scenes_valid'] = len(battle_valid_sorted)
                orbit_entry['battle_scenes'] = battle_valid_sorted
                orbit_entry['total_valid_scenes'] += len(battle_valid_sorted)

            orbit_metadata[str(orbit)] = orbit_entry

        best_orbit = max(orbit_metadata.keys(), key=lambda o: orbit_metadata[o]['total_valid_scenes'])
        best_total = orbit_metadata[best_orbit]['total_valid_scenes']

        print(f"\n  Recommended orbit: {best_orbit} ({best_total} total valid scenes)")

        scene_metadata = {
            'city': city_name,
            'tier': tier,
            'product_types': product_types,
            'battle_start': pre_battle_date.isoformat(),
            'battle_end': post_battle_date.isoformat() if post_battle_date else None,
            'conflict_ongoing': conflict_ongoing,
            'battle_duration_days': battle_duration_days,
            'proximity_days': proximity_days,
            'pre_months': pre_months,
            'pre_battle_start': pre_battle_start.isoformat(),
            'pre_battle_end': pre_battle_end.isoformat(),
            'post_battle_start': post_battle_start.isoformat(),
            'post_battle_end': post_battle_end.isoformat(),
            'max_temporal_baseline_days': max_temporal_baseline,
            'common_orbits': [int(o) for o in sorted(common_orbits)],
            'recommended_orbit': int(best_orbit),
            'orbits': orbit_metadata,
            'timestamp': datetime.now().isoformat()
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        scene_metadata_file = output_path / f"{city_name}_scene_metadata.json"

        with open(scene_metadata_file, 'w') as f:
            json.dump(scene_metadata, f, indent=2)

        print(f"\n  Scene discovery complete for {city_name}")
        print(f"  Metadata saved to: {scene_metadata_file}")

        return True

    except Exception as e:
        print(f"\n  Failed for {city_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# run() - called from notebook cell
# ---------------------------------------------------------------------------
def run(cities_dir, logs_dir, output_dir, progress_file, product_types, force_rerun,
        proximity_days=14, pre_months=14):
    """
    Args:
        cities_dir:     Path - CITIES_DIR (each subdir has AOI.geojson)
        logs_dir:       Path - LOGS_DIR (for cell log JSON + output log)
        output_dir:     Path - SAR_METADATA_DIR (where scene_metadata.json go)
        progress_file:  Path - SLC_PROGRESS_FILE
        product_types:  list - e.g. ["SLC"]
        force_rerun:    bool - FORCE_RERUN
        proximity_days: int  - buffer days around battle dates (default 14)
        pre_months:     int  - months before battle_start to search (default 14)
    """
    CELL_ID = "cell_14a1_scene_discovery"

    cities_dir = Path(cities_dir)
    logs_dir = Path(logs_dir)
    output_dir = Path(output_dir)
    progress_file = Path(progress_file)

    CELL_LOG_FILE = logs_dir / f"{CELL_ID}.json"
    CELL_OUTPUT_LOG = logs_dir / f"{CELL_ID}_output.log"

    logger = DualLogger(CELL_OUTPUT_LOG)
    sys.stdout = logger

    try:
        print("\n" + "=" * 80)
        print("CELL 14A1: SCENE DISCOVERY AND TEMPORAL WINDOWS - MULTI-ORBIT")
        print("=" * 80)
        print(f"Start time: {datetime.now().isoformat()}")
        print(f"Product types: {product_types}")
        print(f"Storage location: {output_dir}")
        print(f"Pre-months: {pre_months}")
        print(f"Proximity days: {proximity_days}")

        if force_rerun:
            print(f"\n  FORCE_RERUN = True - Clearing progress...")
            progress = {}
            save_progress(progress, progress_file)

        progress = load_progress(progress_file)
        progress.setdefault('scenes_discovered', {})

        all_cities = discover_cities(cities_dir)

        if not all_cities:
            raise ValueError(f"No cities found in {cities_dir}")

        print(f"\n  Found {len(all_cities)} cities")

        success_count = 0
        failed_count = 0

        for idx, city_name in enumerate(all_cities, 1):
            print(f"\n{'#'*80}")
            print(f"# CITY {idx}/{len(all_cities)}: {city_name}")
            print(f"{'#'*80}")

            if not force_rerun and city_name in progress['scenes_discovered'] and progress['scenes_discovered'][city_name].get('complete'):
                print(f"\n  Skipping {city_name} (already completed)")
                success_count += 1
                continue

            success = discover_scenes_for_city(
                city_name, cities_dir, output_dir, product_types,
                proximity_days=proximity_days, pre_months=pre_months
            )

            if success:
                progress['scenes_discovered'][city_name] = {
                    'complete': True,
                    'timestamp': datetime.now().isoformat()
                }
                save_progress(progress, progress_file)
                success_count += 1
            else:
                failed_count += 1

            print(f"\n  Progress: {idx}/{len(all_cities)} | Success: {success_count} | Failed: {failed_count}")

        log_data = {
            'cell_id': CELL_ID,
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'product_types': product_types,
            'cities_processed': len(all_cities),
            'success_count': success_count,
            'failed_count': failed_count,
            'metadata_location': str(output_dir)
        }

        with open(CELL_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n{'='*80}")
        print(f"SCENE DISCOVERY COMPLETE")
        print(f"{'='*80}")
        print(f"Product types: {product_types}")
        print(f"Cities processed: {len(all_cities)}")
        print(f"Success: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Metadata location: {output_dir}")

    except Exception as e:
        log_data = {
            'cell_id': CELL_ID,
            'status': 'FAILED',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

        with open(CELL_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n  Failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        sys.stdout = logger.terminal
        logger.close()

    print("Cell 14A1: Scene discovery complete")
