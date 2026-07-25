# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
bonus_city_filter.py
Shared module for NB03a/b/c bonus city cells.
Provides spatial (bbox overlap) + temporal (date range) filtering.

A bonus city only gets clipped if:
  1. Zip/scene footprint overlaps city AOI bbox >= min_overlap_pct
  2. Scene date falls within [battle_start - MARGIN_MONTHS, battle_stop + MARGIN_MONTHS]

Usage NB03b/NB03c (zip-based bonus cells):
    from bonus_city_filter import load_bonus_city_index, filter_covering_cities

    city_index = load_bonus_city_index(CITIES_DIR, load_aoi_bbox)
    covering = filter_covering_cities(footprint_wgs84, date_str, city_index)
    # city_index[city_name]['geom'] for clipping

Usage NB03a (COH bonus in process_insar_pair):
    from bonus_city_filter import load_city_dates, is_date_in_range

    city_dates = load_city_dates(CITIES_DIR)  # once at cell level
    # in bonus city loop:
    bs, be = city_dates.get(bonus_city, (None, None))
    if not is_date_in_range(dates_str[0], bs, be):
        print(f"    SKIP {bonus_city}: outside date window")
        continue
"""

import re
import json
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shapely.geometry import box


MARGIN_MONTHS = 14
MIN_OVERLAP_PCT = 50.0


def _parse_date(val):
    """Parse date string to datetime. Returns None for ongoing/missing."""
    if val is None:
        return None
    val = str(val).strip()
    if not val or val.lower() == 'ongoing':
        return None
    try:
        if 'T' in val:
            return datetime.fromisoformat(val.replace('Z', '+00:00')).replace(tzinfo=None)
        return datetime.strptime(val[:10], '%Y-%m-%d')
    except Exception:
        return None


def _extract_dates_from_aoi(aoi_path):
    """Extract battle_start, battle_stop from AOI.geojson using regex.
    Fast method for large single-line files on drvfs."""
    try:
        with open(aoi_path, 'r') as f:
            chunk = f.read(50000)
    except Exception:
        return None, None

    battle_start = None
    battle_stop = None

    m = re.search(r'"battle_start"\s*:\s*"([^"]+)"', chunk)
    if m:
        battle_start = _parse_date(m.group(1))

    m = re.search(r'"battle_stop"\s*:\s*"([^"]+)"', chunk)
    if m:
        raw = m.group(1)
        if raw.lower() == 'ongoing':
            battle_stop = None
        else:
            battle_stop = _parse_date(raw)

    return battle_start, battle_stop


def load_city_dates(cities_dir):
    """Load battle dates for all cities. Lightweight: no geometry loading.

    For NB03a where spatial overlap is checked separately via load_aoi().
    Uses regex on AOI.geojson header (fast on drvfs).

    Args:
        cities_dir: Path to data/cities/

    Returns:
        dict: {city_name: (battle_start, battle_stop)}
              battle_start = datetime or None
              battle_stop = datetime or None (ongoing)
    """
    cities_dir = Path(cities_dir)
    city_dates = {}

    for city_dir in sorted(cities_dir.iterdir()):
        if not city_dir.is_dir():
            continue

        aoi_path = city_dir / 'AOI.geojson'
        if not aoi_path.exists():
            aoi_path = city_dir / 'boundary.geojson'
        if not aoi_path.exists():
            continue

        battle_start, battle_stop = _extract_dates_from_aoi(aoi_path)
        city_dates[city_dir.name] = (battle_start, battle_stop)

    return city_dates


def load_bonus_city_index(cities_dir, load_aoi_bbox_fn):
    """Load all cities with their AOI bbox geometry and battle dates.

    Args:
        cities_dir: Path to data/cities/
        load_aoi_bbox_fn: callable(city_name, cities_dir) -> shapely geometry

    Returns:
        dict: {city_name: {'geom': shapely_geom, 'battle_start': datetime|None,
                           'battle_stop': datetime|None}}
    """
    cities_dir = Path(cities_dir)
    city_index = {}

    for city_dir in sorted(cities_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        city_name = city_dir.name

        # load bbox geometry
        try:
            geom = load_aoi_bbox_fn(city_name, cities_dir)
        except (FileNotFoundError, ValueError):
            continue

        # load battle dates from AOI.geojson
        aoi_path = city_dir / 'AOI.geojson'
        if not aoi_path.exists():
            # fallback: try boundary.geojson (legacy)
            aoi_path = city_dir / 'boundary.geojson'
        if not aoi_path.exists():
            continue

        battle_start, battle_stop = _extract_dates_from_aoi(aoi_path)

        city_index[city_name] = {
            'geom': geom,
            'battle_start': battle_start,
            'battle_stop': battle_stop,
        }

    return city_index


def is_date_in_range(date_str, battle_start, battle_stop, margin_months=MARGIN_MONTHS):
    """Check if a scene date falls within the allowed window.

    Window: [battle_start - margin_months, battle_stop + margin_months]
    If battle_start is None, city has no dates -> skip (return False).
    If battle_stop is None (ongoing), use datetime.now() as battle_stop.

    Args:
        date_str: scene date as YYYYMMDD or YYYY-MM-DD string
        battle_start: datetime or None
        battle_stop: datetime or None (ongoing)
        margin_months: int, months of margin on each side

    Returns:
        bool
    """
    if battle_start is None:
        return False

    try:
        s = str(date_str).replace('-', '')[:8]
        scene_date = datetime.strptime(s, '%Y%m%d')
    except (ValueError, TypeError):
        return False

    effective_stop = battle_stop if battle_stop is not None else datetime.now()

    window_start = battle_start - relativedelta(months=margin_months)
    window_end = effective_stop + relativedelta(months=margin_months)

    return window_start <= scene_date <= window_end


def filter_covering_cities(footprint_wgs84, date_str, city_index,
                           margin_months=MARGIN_MONTHS,
                           min_overlap_pct=MIN_OVERLAP_PCT):
    """Find cities that a zip covers both spatially AND temporally.

    Args:
        footprint_wgs84: tuple (minx, miny, maxx, maxy) in WGS84
        date_str: scene date as YYYYMMDD string
        city_index: dict from load_bonus_city_index()
        margin_months: int
        min_overlap_pct: float, minimum overlap percentage of city area

    Returns:
        list of city names that pass both spatial and temporal filters
    """
    if footprint_wgs84 is None:
        return []

    swath_box = box(footprint_wgs84[0], footprint_wgs84[1],
                    footprint_wgs84[2], footprint_wgs84[3])

    covering = []
    for city_name, info in city_index.items():
        city_geom = info['geom']
        city_box = box(*city_geom.bounds)

        # spatial check
        if not swath_box.intersects(city_box):
            continue
        overlap_area = swath_box.intersection(city_box).area
        city_area = city_box.area
        if city_area <= 0:
            continue
        overlap_pct = 100 * overlap_area / city_area
        if overlap_pct < min_overlap_pct:
            continue

        # temporal check
        if not is_date_in_range(date_str, info['battle_start'], info['battle_stop'],
                                margin_months):
            continue

        covering.append(city_name)

    return covering
