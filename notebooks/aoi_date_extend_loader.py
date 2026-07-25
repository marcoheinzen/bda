# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
aoi_date_extend_loader.py
AOI loader with battle date extraction and temporal window computation.
Extracted from global_setup / notebook functions used across NB02a/b/c, NB03, NB05, NB10a.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape


_aoi_cache = {}

def _load_city_polygon_feature(city_name, cities_dir):
    cache_key = f"{cities_dir}/{city_name}"
    if cache_key in _aoi_cache:
        return _aoi_cache[cache_key]
    aoi_file = Path(cities_dir) / city_name / "AOI.geojson"
    if not aoi_file.exists():
        raise FileNotFoundError(f"AOI.geojson not found: {aoi_file}")
    with open(aoi_file) as f:
        gj = json.load(f)
    for feat in gj['features']:
        if feat.get('properties', {}).get('feature_type') == 'city_polygon':
            _aoi_cache[cache_key] = feat
            return feat
    raise ValueError(f"No city_polygon feature in {aoi_file}")


def load_aoi(city_name, cities_dir):
    feat = _load_city_polygon_feature(city_name, cities_dir)
    props = dict(feat['properties'])
    props['geometry'] = shape(feat['geometry'])
    return pd.Series(props)


def load_aoi_gdf(city_name, cities_dir):
    feat = _load_city_polygon_feature(city_name, cities_dir)
    props = dict(feat['properties'])
    geom = shape(feat['geometry'])
    return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")


def load_aoi_bbox(city_name, cities_dir):
    aoi_file = Path(cities_dir) / city_name / "AOI.geojson"
    if not aoi_file.exists():
        raise FileNotFoundError(f"AOI.geojson not found: {aoi_file}")
    with open(aoi_file) as f:
        gj = json.load(f)
    for feat in gj['features']:
        if feat.get('properties', {}).get('feature_type') == 'aoi_bbox':
            return shape(feat['geometry'])
    raise ValueError(f"No aoi_bbox feature in {aoi_file}")


def parse_date(date_str):
    if isinstance(date_str, datetime):
        return date_str
    if date_str == 'ongoing' or date_str is None:
        return None
    if 'T' in str(date_str):
        return datetime.fromisoformat(str(date_str).replace('Z', '+00:00')).replace(tzinfo=None)
    return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")


def load_city_boundary_with_dates(city_name, cities_dir):
    """
    Returns:
        (gdf, pre_battle_date, post_battle_date, conflict_ongoing, tier)
    """
    aoi_gdf = load_aoi_gdf(city_name, cities_dir)

    battle_start = aoi_gdf.iloc[0].get('battle_start', None)
    battle_stop = aoi_gdf.iloc[0].get('battle_stop', None)

    if not battle_start:
        raise ValueError(f"Missing battle_start in {city_name}")

    pre_battle_date = parse_date(battle_start)
    post_battle_date = parse_date(battle_stop) if battle_stop and str(battle_stop) != 'ongoing' else None
    conflict_ongoing = str(battle_stop) == 'ongoing' or battle_stop is None

    tier = int(aoi_gdf.iloc[0].get('tier', 1))

    return aoi_gdf, pre_battle_date, post_battle_date, conflict_ongoing, tier


def compute_temporal_windows(battle_start, battle_stop, conflict_ongoing,
                             proximity_days=14, pre_months=14):
    """
    Compute pre-battle, post-battle and crossbattle search windows.
    Replaces get_data_start_date() with dynamic battle_start - pre_months.
    """
    pre_battle_start = battle_start - relativedelta(months=pre_months)
    pre_battle_end = battle_start - timedelta(days=proximity_days)

    if battle_stop:
        post_battle_start = battle_stop + timedelta(days=proximity_days)
        post_battle_end = battle_stop + relativedelta(months=pre_months)
        battle_duration_days = (battle_stop - battle_start).days
    else:
        post_battle_end = datetime.now()
        post_battle_start = post_battle_end - timedelta(days=90)
        battle_duration_days = (datetime.now() - battle_start).days

    crossbattle_start = battle_start
    crossbattle_end = battle_stop if battle_stop else datetime.now()

    max_temporal_baseline = battle_duration_days + (2 * proximity_days)

    return {
        'pre_battle_start': pre_battle_start,
        'pre_battle_end': pre_battle_end,
        'post_battle_start': post_battle_start,
        'post_battle_end': post_battle_end,
        'crossbattle_start': crossbattle_start,
        'crossbattle_end': crossbattle_end,
        'battle_duration_days': battle_duration_days,
        'max_temporal_baseline_days': max_temporal_baseline,
        'proximity_days': proximity_days,
    }


def discover_cities(cities_dir):
    """Find all cities with AOI.geojson under cities_dir."""
    cities = []
    cities_path = Path(cities_dir)
    for city_dir in sorted(cities_path.iterdir()):
        if city_dir.is_dir():
            aoi_file = city_dir / "AOI.geojson"
            if aoi_file.exists():
                cities.append(city_dir.name)
    return cities
