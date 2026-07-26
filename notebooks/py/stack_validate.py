# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_validate.py
Data stack completeness audit + catalog builder.
Scans all products per city, writes data_catalog.json and global manifest.

Notebook usage:
    from stack_validate import run as run_validate
    VALIDATE_RESULT = run_validate(
        stack_root=STACK_ROOT,
        cities=CITIES_TO_PROCESS,
        load_aoi_fn=load_aoi,
    )
"""

import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import json
import re
import time


# =========================================================================
# HELPERS
# =========================================================================

def extract_dates_from_filename(filename):
    dates = re.findall(r'(\d{8})', filename)
    return dates

def count_tifs(directory, pattern="*.tif", recursive=False):
    if not directory.exists():
        return 0, []
    if recursive:
        files = sorted(directory.rglob(pattern))
    else:
        files = sorted(directory.glob(pattern))
    return len(files), [f.name for f in files]

def get_tif_dates(directory, recursive=False):
    if not directory.exists():
        return set()
    pattern = "*.tif"
    files = sorted(directory.rglob(pattern)) if recursive else sorted(directory.glob(pattern))
    dates = set()
    for f in files:
        for d in extract_dates_from_filename(f.name):
            dates.add(d)
    return dates

def check_raster_grid(tif_path, ref_width, ref_height):
    try:
        with rasterio.open(tif_path) as src:
            return src.width == ref_width and src.height == ref_height
    except Exception:
        return False

def _load_battle_dates_from_aoi(city_name, load_aoi_fn):
    try:
        aoi_row = load_aoi_fn(city_name)
        battle_start = str(aoi_row.get('battle_start', ''))[:10] if aoi_row.get('battle_start') else None
        battle_stop_raw = aoi_row.get('battle_stop', '')
        battle_stop = str(battle_stop_raw)[:10] if battle_stop_raw and str(battle_stop_raw) not in ('', 'ongoing') else None
        conflict_ongoing = str(battle_stop_raw) == 'ongoing' or battle_stop_raw is None or str(battle_stop_raw) == ''
        tier = int(aoi_row.get('tier', 99))
        return battle_start, battle_stop, conflict_ongoing, tier
    except Exception:
        return None, None, True, 99


# =========================================================================
# MAIN
# =========================================================================

def run(stack_root, cities, load_aoi_fn):
    stack_root = Path(stack_root)
    t0 = time.time()

    print("=" * 70)
    print("VALIDATE: DATA_STACK COMPLETENESS AUDIT")
    print("=" * 70)
    print(f"  STACK_ROOT: {stack_root}")
    print(f"  Cities:     {len(cities)}")

    city_dirs = sorted([stack_root / c for c in cities
                        if (stack_root / c / "reference_grid.json").exists()])
    print(f"  With reference_grid.json: {len(city_dirs)}")

    all_catalogs = {}
    city_summary = []

    for ci, city_stack_dir in enumerate(city_dirs):
        city_name = city_stack_dir.name
        ref_path = city_stack_dir / "reference_grid.json"
        with open(ref_path) as f:
            ref_info = json.load(f)
        ref_width = ref_info['width']
        ref_height = ref_info['height']

        battle_start, battle_stop, conflict_ongoing, tier = _load_battle_dates_from_aoi(city_name, load_aoi_fn)

        catalog = {
            "city": city_name, "generated": datetime.now().isoformat(),
            "reference_grid": ref_info, "tier": tier,
            "battle_start": battle_start, "battle_stop": battle_stop,
            "conflict_ongoing": conflict_ongoing,
        }

        vectors = {}
        for vname in ["AOI.geojson", "buildings_overture.geojson",
                       "buildings_overture_with_damage.geojson", "unosat_damage.geojson"]:
            vpath = city_stack_dir / vname
            if vpath.exists():
                try:
                    gdf = gpd.read_file(vpath)
                    vectors[vname] = {"n_features": len(gdf), "exists": True}
                except Exception:
                    vectors[vname] = {"exists": True, "n_features": -1}
            else:
                vectors[vname] = {"exists": False}
        catalog["vectors"] = vectors

        n_buildings = vectors.get("buildings_overture_with_damage.geojson", {}).get("n_features", 0)
        if n_buildings <= 0:
            n_buildings = vectors.get("buildings_overture.geojson", {}).get("n_features", 0)
        n_damage = 0
        dmg_path = city_stack_dir / "buildings_overture_with_damage.geojson"
        if dmg_path.exists():
            try:
                gdf = gpd.read_file(dmg_path)
                import pandas as _pd
                if 'damage_binary' in gdf.columns:
                    db = _pd.to_numeric(gdf['damage_binary'], errors='coerce')
                    n_damage = int(db.eq(1).sum())
                elif 'damage_label' in gdf.columns:
                    dl = _pd.to_numeric(gdf['damage_label'], errors='coerce')
                    n_damage = int(dl.eq(1).sum())
                elif 'damage' in gdf.columns:
                    dl = _pd.to_numeric(gdf['damage'], errors='coerce')
                    n_damage = int((dl > 0).sum())
            except Exception:
                pass

        card_dir = city_stack_dir / "SAR_CARD"
        card_flat_dir = card_dir / "flat"
        card_flat_n, card_flat_fnames = count_tifs(card_flat_dir)
        card_flat_dates = get_tif_dates(card_flat_dir)
        ts_dir = card_dir / "temporal_stats"
        ts_n, _ = count_tifs(ts_dir)
        catalog["SAR_CARD"] = {
            "flat": {"n_tifs": card_flat_n, "n_dates": len(card_flat_dates), "dates": sorted(card_flat_dates)},
            "temporal_stats": ts_n,
        }

        slc_dir = city_stack_dir / "SAR_SLC"
        slc_flat_dir = slc_dir / "flat"
        slc_flat_n, _ = count_tifs(slc_flat_dir)
        slc_flat_dates = get_tif_dates(slc_flat_dir)
        bl_dir = slc_dir / "coherence_baseline"
        bl_n, _ = count_tifs(bl_dir)
        catalog["SAR_SLC"] = {
            "flat": {"n_tifs": slc_flat_n, "n_dates": len(slc_flat_dates)},
            "coherence_baseline": bl_n,
        }

        ms_dir = city_stack_dir / "multispectral"
        ms_flat_dir = ms_dir / "flat"
        ms_flat_n, _ = count_tifs(ms_flat_dir)
        ms_flat_dates = get_tif_dates(ms_flat_dir)
        ms_nbr_dir = ms_dir / "nbr"
        ms_nbr_n, _ = count_tifs(ms_nbr_dir)
        catalog["multispectral_scenes"] = {
            "flat": {"n_tifs": ms_flat_n, "n_dates": len(ms_flat_dates)},
            "nbr": {"n_tifs": ms_nbr_n},
        }

        comp_dir = ms_dir / "composites"
        composites = {}
        if comp_dir.exists():
            for period_dir in sorted(comp_dir.iterdir()):
                if period_dir.is_dir():
                    n, fnames = count_tifs(period_dir)
                    composites[period_dir.name] = {"n_tifs": n, "files": fnames}
        catalog["composites"] = composites

        rgb_dir = ms_dir / "rgb"
        rgb_n, _ = count_tifs(rgb_dir)
        catalog["rgb"] = {"n_tifs": rgb_n}

        lu_dir = city_stack_dir / "landuse"
        lu_periods = {}
        if lu_dir.exists():
            for period_dir in sorted(lu_dir.iterdir()):
                if period_dir.is_dir():
                    n, _ = count_tifs(period_dir, recursive=True)
                    lu_periods[period_dir.name] = {"n_tifs": n}
        catalog["landuse"] = lu_periods

        temp_dir = city_stack_dir / "temporal"
        temporal_products = {}
        if temp_dir.exists():
            for sensor_dir in sorted(temp_dir.iterdir()):
                if sensor_dir.is_dir():
                    sensor = sensor_dir.name
                    temporal_products[sensor] = {}
                    for product_dir in sorted(sensor_dir.iterdir()):
                        if product_dir.is_dir():
                            n, _ = count_tifs(product_dir)
                            temporal_products[sensor][product_dir.name] = {"n_tifs": n}
        catalog["temporal_products"] = temporal_products

        diet_bl_n, _ = count_tifs(city_stack_dir / "Dietrich_baseline")
        diet_as_n, _ = count_tifs(city_stack_dir / "Dietrich_assessment")
        catalog["dietrich"] = {"baseline": diet_bl_n, "assessment": diet_as_n}

        all_tifs = sorted(city_stack_dir.rglob("*.tif"))
        sample = all_tifs[:min(20, len(all_tifs))]
        grid_ok = sum(1 for t in sample if check_raster_grid(t, ref_width, ref_height))
        catalog["grid_check"] = {"sampled": len(sample), "ok": grid_ok, "total_tifs": len(all_tifs)}

        has_card = card_flat_n > 0
        has_coh = slc_flat_n > 0
        has_ms = ms_flat_n > 0
        has_composites = sum(c.get("n_tifs", 0) for c in composites.values()) > 0
        has_buildings = n_buildings > 0
        has_damage = n_damage > 0
        ready_ml = has_buildings and has_damage and (has_card or has_coh or has_ms)
        ready_dl = ready_ml and has_composites

        readiness = {
            "has_card": has_card, "has_coh": has_coh, "has_ms": has_ms,
            "has_composites": has_composites, "has_buildings": has_buildings,
            "has_damage": has_damage, "n_buildings": n_buildings,
            "n_damage_labels": n_damage, "ready_ml": ready_ml, "ready_dl": ready_dl,
        }
        catalog["readiness"] = readiness

        cat_path = city_stack_dir / "data_catalog.json"
        with open(cat_path, 'w') as f:
            json.dump(catalog, f, indent=2, default=str)

        all_catalogs[city_name] = catalog
        status = "ML-READY" if ready_ml else ("DL-READY" if ready_dl else "incomplete")
        print(f"  {city_name:<22s} tier={tier} bldg={n_buildings:>5d} dmg={n_damage:>4d} "
              f"CARD={'Y' if has_card else '-'} COH={'Y' if has_coh else '-'} "
              f"MS={'Y' if has_ms else '-'} comp={'Y' if has_composites else '-'} [{status}]")

        city_summary.append({'city': city_name, 'n_buildings': n_buildings, 'n_damage': n_damage, **readiness})

    manifest = {"generated": datetime.now().isoformat(), "n_cities": len(all_catalogs), "cities": {}}
    for city_name, cat in all_catalogs.items():
        manifest["cities"][city_name] = cat.get("readiness", {})
        manifest["cities"][city_name]["tier"] = cat.get("tier", 99)

    manifest_path = stack_root / "data_stack_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    elapsed = time.time() - t0
    n_ml = sum(1 for s in city_summary if s.get('ready_ml'))
    n_dl = sum(1 for s in city_summary if s.get('ready_dl'))

    print(f"\n{'='*70}")
    print(f"VALIDATE COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Cities: {len(all_catalogs)}, ML-ready: {n_ml}, DL-ready: {n_dl}")
    print(f"  Manifest: {manifest_path}")

    return manifest
