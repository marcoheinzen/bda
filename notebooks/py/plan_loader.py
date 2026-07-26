# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
plan_loader.py - Plan-based scene source for NB03c InSAR processing.
Reads nb02a_scene_plan.json (written by NB02a).

Provides:
  - SLC scene chains with gap-aware pair building
  - Orbit info per city (from plan, not discovery metadata)
  - Write-back after COH processing (mark coh_processed, add coh_pairs)

Usage in NB03c:
    import plan_loader
    plan_loader.init(OUTPUTS_DIR, RAW_SLC_ZIP, MIN_SLC_SIZE)
    plan_loader.load_plan()

    orbit = plan_loader.get_city_orbit("Mariupol")
    pairs = plan_loader.build_pairs_with_gaps("Mariupol", period_filter="pre")
    for scene_a, scene_b in pairs:
        ...
    plan_loader.mark_coh_processed("Mariupol", "2022-01-01", "2022-01-13")
"""

import json
from pathlib import Path
from datetime import datetime

_PLAN = None
_PLAN_FILE = None
_RAW_SLC_ZIP = None
_MIN_SLC_SIZE = 3e9
_CITIES_DIR = None


def init(outputs_dir, raw_slc_zip=None, min_slc_size=3e9, cities_dir=None):
    global _PLAN_FILE, _RAW_SLC_ZIP, _MIN_SLC_SIZE, _CITIES_DIR
    _PLAN_FILE = Path(outputs_dir) / "nb02a_scene_plan.json"
    if raw_slc_zip is not None:
        _RAW_SLC_ZIP = Path(raw_slc_zip)
    _MIN_SLC_SIZE = min_slc_size
    if cities_dir is not None:
        _CITIES_DIR = Path(cities_dir)


def load_plan():
    global _PLAN
    if _PLAN_FILE and _PLAN_FILE.exists():
        with open(_PLAN_FILE) as f:
            _PLAN = json.load(f)
        print(f"  Plan loaded: {len(_PLAN)} cities from {_PLAN_FILE.name}")
    else:
        _PLAN = {}
        print(f"  WARNING: Plan file not found: {_PLAN_FILE}")
    return _PLAN


def save_plan():
    if _PLAN is not None and _PLAN_FILE:
        with open(_PLAN_FILE, "w") as f:
            json.dump(_PLAN, f, indent=2, default=str)


def get_city_plan(city_name):
    if _PLAN is None:
        load_plan()
    return _PLAN.get(city_name, {})


def get_city_orbit(city_name):
    plan = get_city_plan(city_name)
    return int(plan.get("orbit", 0))


def _classify_period(date_str, battle_start, battle_stop):
    if battle_start and date_str >= battle_start:
        if battle_stop and date_str > battle_stop:
            return "post"
        return "battle"
    return "pre"


def _zip_on_disk(scene_name):
    """Check if SLC zip exists in RAW_SLC_ZIP. Returns True/False."""
    if _RAW_SLC_ZIP is None:
        return False
    base = scene_name[:-5] if scene_name.endswith(".SAFE") else scene_name
    for suffix in [".zip", ".SAFE.zip"]:
        zp = _RAW_SLC_ZIP / f"{base}{suffix}"
        if zp.exists() and zp.stat().st_size > _MIN_SLC_SIZE:
            return True
    prefix = base[:40]
    candidates = list(_RAW_SLC_ZIP.glob(f"{prefix}*.zip"))
    return any(c.stat().st_size > _MIN_SLC_SIZE for c in candidates)


def build_pairs_with_gaps(city_name, period_filter=None):
    """
    Build COH pairs from plan SLC entries with gap handling.

    Iterates ALL planned SLC dates (sorted). Scenes that are on disk
    (status in on_disk/extracted/coh_processed AND zip verified) form
    continuous sub-chains. A scene that is NOT on disk breaks the chain.
    Each sub-chain of N scenes produces N-1 consecutive pairs.

    Args:
        city_name: City name
        period_filter: None (all), "pre", "battle", or "post"

    Returns:
        (pairs, orbit) where pairs is list of (scene_a, scene_b) tuples.
        Each scene dict has keys: name, date, period, scene_id.
    """
    plan = get_city_plan(city_name)
    if not plan:
        print(f"    WARNING: {city_name} not in plan")
        return [], 0

    orbit = int(plan.get("orbit", 0))
    battle_start = plan.get("battle_start", "")
    battle_stop = plan.get("battle_stop", "")

    all_entries = sorted(plan.get("slc", []), key=lambda e: e.get("date", "")[:10])

    # Apply period filter to the full planned list
    if period_filter:
        all_entries = [
            e for e in all_entries
            if _classify_period(e.get("date", "")[:10], battle_start, battle_stop) == period_filter
        ]

    if not all_entries:
        print(f"    No SLC entries in plan for {city_name}" +
              (f" (period={period_filter})" if period_filter else ""))
        return [], orbit

    # Build sub-chains: consecutive on-disk entries
    sub_chains = []
    current_chain = []
    skipped = 0

    for entry in all_entries:
        status = entry.get("status", "")
        name = entry.get("scene_name", "")
        date_str = entry.get("date", "")[:10]
        is_available = status in ("on_disk", "extracted", "coh_processed") and _zip_on_disk(name)

        if is_available:
            period = _classify_period(date_str, battle_start, battle_stop)
            current_chain.append({
                "name": name,
                "date": date_str,
                "period": period,
                "scene_id": entry.get("scene_id", ""),
            })
        else:
            # Gap: flush current chain
            if len(current_chain) >= 2:
                sub_chains.append(current_chain)
            elif len(current_chain) == 1:
                skipped += 1  # isolated scene, cannot form pair
            current_chain = []
            skipped += 1

    # Flush last chain
    if len(current_chain) >= 2:
        sub_chains.append(current_chain)

    # Flatten to pairs
    pairs = []
    for chain in sub_chains:
        for i in range(len(chain) - 1):
            pairs.append((chain[i], chain[i + 1]))

    n_scenes = sum(len(c) for c in sub_chains)
    n_gaps = max(0, len(sub_chains) - 1)
    filter_str = f" [{period_filter}]" if period_filter else ""
    print(f"    Plan chain{filter_str} for {city_name} orbit {orbit}: "
          f"{n_scenes} scenes, {len(sub_chains)} sub-chain(s), "
          f"{n_gaps} gap(s), {len(pairs)} pair(s)"
          + (f", {skipped} skipped" if skipped else ""))

    for chain in sub_chains:
        for s in chain:
            print(f"      [{s['period']:6s}] {s['date']} - {s['name'][:50]}")
        if chain is not sub_chains[-1]:
            print(f"      --- gap ---")

    return pairs, orbit


def get_all_scenes_on_disk(city_name, period_filter=None):
    """
    Get flat list of on-disk SLC scenes (no gap handling).
    Useful for build_scene_chain_from_disk compatibility.
    """
    plan = get_city_plan(city_name)
    if not plan:
        return [], 0

    orbit = int(plan.get("orbit", 0))
    battle_start = plan.get("battle_start", "")
    battle_stop = plan.get("battle_stop", "")

    scenes = []
    for entry in sorted(plan.get("slc", []), key=lambda e: e.get("date", "")[:10]):
        status = entry.get("status", "")
        name = entry.get("scene_name", "")
        date_str = entry.get("date", "")[:10]

        if status not in ("on_disk", "extracted", "coh_processed"):
            continue
        if not _zip_on_disk(name):
            continue

        period = _classify_period(date_str, battle_start, battle_stop)
        if period_filter and period != period_filter:
            continue

        scenes.append({
            "name": name,
            "date": date_str,
            "period": period,
            "scene_id": entry.get("scene_id", ""),
        })

    return scenes, orbit


def mark_coh_processed(city_name, date_a, date_b):
    """
    Write-back to plan after successful COH pair.
    Marks both SLC entries as coh_processed and appends to coh_pairs list.
    """
    plan = get_city_plan(city_name)
    if not plan:
        return

    for entry in plan.get("slc", []):
        d = entry.get("date", "")[:10]
        if d == date_a or d == date_b:
            entry["coh_processed"] = True

    if "coh_pairs" not in plan:
        plan["coh_pairs"] = []
    plan["coh_pairs"].append({
        "date_a": date_a,
        "date_b": date_b,
        "timestamp": datetime.now().isoformat(),
    })
    save_plan()


# =========================================================================
# AOI.geojson readers (single source of truth for vector data)
# =========================================================================
# AOI.geojson per city contains:
#   feature_type="aoi_bbox"      -> 64-aligned rectangular bbox (Polygon)
#   feature_type="city_polygon"  -> admin boundary + battle dates + tier
#   feature_type="building"      -> Overture buildings + UNOSAT damage labels
# =========================================================================

def _load_aoi_geojson(city_name, cities_dir=None):
    """Load and return parsed AOI.geojson for a city."""
    d = Path(cities_dir) if cities_dir else _CITIES_DIR
    if d is None:
        raise ValueError("cities_dir not set. Call plan_loader.init(cities_dir=...)")
    aoi_file = d / city_name / "AOI.geojson"
    if not aoi_file.exists():
        raise FileNotFoundError(f"AOI.geojson not found: {aoi_file}")
    with open(aoi_file) as f:
        return json.load(f)


def load_aoi_bbox(city_name, cities_dir=None):
    """
    Load aoi_bbox polygon from AOI.geojson as a shapely geometry.

    Args:
        city_name: City name (subfolder of cities_dir)
        cities_dir: Path to CITIES_DIR (uses module default if None)

    Returns:
        shapely.geometry.shape with .bounds = (minx, miny, maxx, maxy)
    """
    from shapely.geometry import shape

    gj = _load_aoi_geojson(city_name, cities_dir)
    for feat in gj["features"]:
        if feat.get("properties", {}).get("feature_type") == "aoi_bbox":
            return shape(feat["geometry"])
    raise ValueError(f"No aoi_bbox feature in AOI.geojson for {city_name}")


def get_city_info(city_name, cities_dir=None):
    """
    Load city metadata from the city_polygon feature in AOI.geojson.

    Returns dict with keys: battle_start, battle_stop, tier, oblast,
    admin_level, city_name, and the full properties dict.
    """
    gj = _load_aoi_geojson(city_name, cities_dir)
    for feat in gj["features"]:
        props = feat.get("properties", {})
        if props.get("feature_type") == "city_polygon":
            return props
    raise ValueError(f"No city_polygon feature in AOI.geojson for {city_name}")
