# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_align.py
Align NB03/NB03d products to common UTM reference grid in data_stack.

Shared functions:
  - load_canonical_grid()     read UTM grid from AOI.geojson (NB01b)
  - load_reference_grid()     read UTM grid from data_stack/reference_grid.json
  - align_raster_to_grid()    reproject single raster to reference grid
  - run_align()               align a set of ALIGN_GROUPS for multiple cities
  - verify_bbox()             check all data_stack rasters match reference grid

Notebook usage:
    from stack_align import load_canonical_grid, run_align, verify_bbox

    # Primary products (NB03a/b/c)
    run_align(
        cities=CITIES_TO_PROCESS,
        stack_root=STACK_ROOT,
        cities_dir=CITIES_DIR,
        align_groups_fn=build_primary_groups,
        create_grid=True,
        force_rerun=False,
    )

    # Derived products (NB03d/NB04c)
    run_align(
        cities=CITIES_TO_PROCESS,
        stack_root=STACK_ROOT,
        cities_dir=CITIES_DIR,
        align_groups_fn=build_derived_groups,
        create_grid=False,
        force_rerun=True,
    )
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from affine import Affine
from shapely.geometry import shape
from pathlib import Path
from datetime import datetime
import json
import time
import shutil


# =========================================================================
# LOAD CANONICAL GRID FROM AOI.geojson (NB01b output)
# =========================================================================

def load_canonical_grid(city_name, cities_dir):
    """Read 256-aligned UTM grid params from AOI.geojson.
    Returns dict with crs, transform, width, height, utm_epsg, utm_bounds, poly_geom.
    """
    aoi_file = Path(cities_dir) / city_name / "AOI.geojson"
    with open(aoi_file) as f:
        gj = json.load(f)

    bbox_props = None
    poly_geom = None
    for feat in gj['features']:
        ft = feat.get('properties', {}).get('feature_type', '')
        if ft == 'aoi_bbox':
            bbox_props = feat['properties']
        elif ft == 'city_polygon':
            poly_geom = shape(feat['geometry'])
        if bbox_props and poly_geom:
            break
    del gj

    if bbox_props is None:
        raise ValueError(f"No aoi_bbox feature in AOI.geojson for {city_name}")

    p = bbox_props
    utm_epsg = int(p['utm_epsg'])
    utm_minx = p['utm_minx']
    utm_miny = p['utm_miny']
    utm_maxx = p['utm_maxx']
    utm_maxy = p['utm_maxy']
    width_px = int(p['width_px'])
    height_px = int(p['height_px'])

    ref_crs = rasterio.crs.CRS.from_epsg(utm_epsg)
    ref_transform = Affine(10.0, 0.0, utm_minx, 0.0, -10.0, utm_maxy)

    expected_w = int(round((utm_maxx - utm_minx) / 10.0))
    expected_h = int(round((utm_maxy - utm_miny) / 10.0))
    assert expected_w == width_px, f"Width mismatch: {expected_w} vs {width_px}"
    assert expected_h == height_px, f"Height mismatch: {expected_h} vs {height_px}"
    assert width_px % 64 == 0, f"Width {width_px} not divisible by 64"
    assert height_px % 64 == 0, f"Height {height_px} not divisible by 64"

    return {
        'crs': ref_crs,
        'transform': ref_transform,
        'width': width_px,
        'height': height_px,
        'utm_epsg': utm_epsg,
        'utm_bounds': [utm_minx, utm_miny, utm_maxx, utm_maxy],
        'poly_geom': poly_geom,
    }


# =========================================================================
# LOAD REFERENCE GRID FROM EXISTING data_stack
# =========================================================================

def load_reference_grid(city_name, stack_root, cities_dir):
    """Read reference grid from data_stack/{city}/reference_grid.json.
    Returns (ref_transform, ref_crs, ref_width, ref_height, polygon_mask) or None.
    """
    ref_path = Path(stack_root) / city_name / "reference_grid.json"
    if not ref_path.exists():
        return None

    with open(ref_path) as f:
        info = json.load(f)

    ref_transform = Affine(*info['transform'])
    ref_crs = rasterio.crs.CRS.from_string(info['crs'])
    ref_width = int(info['width'])
    ref_height = int(info['height'])

    from aoi_date_extend_loader import load_aoi_gdf
    try:
        poly_gdf = load_aoi_gdf(city_name, cities_dir).to_crs(ref_crs)
    except (FileNotFoundError, ValueError):
        return None

    geom = poly_gdf.geometry.unary_union
    polygon_mask = ~geometry_mask(
        [geom],
        out_shape=(ref_height, ref_width),
        transform=ref_transform,
        invert=False
    )

    return ref_transform, ref_crs, ref_width, ref_height, polygon_mask


# =========================================================================
# SINGLE RASTER ALIGNMENT
# =========================================================================

def align_raster_to_grid(src_path, dst_path, ref_transform, ref_crs,
                         ref_width, ref_height, polygon_mask,
                         force_rerun=False):
    """Reproject single raster to reference grid.
    Returns status string: 'ok:{valid_px}', 'skipped', or 'error:{msg}'.
    """
    if dst_path.exists() and not force_rerun:
        return "skipped"

    try:
        with rasterio.open(src_path) as src:
            n_bands = src.count
            src_dtype = src.dtypes[0]
            src_nodata = src.nodata

            _path_lower = str(src_path).lower()
            is_classification = ("landuse" in _path_lower or
                                 "count" in _path_lower or
                                 "cloud_mask" in _path_lower or
                                 "visibility" in _path_lower or
                                 "severity" in _path_lower or
                                 "_SCL_" in str(src_path))
            is_rgb = (n_bands == 3 and 'rgb' in _path_lower)
            resamp = Resampling.nearest if (is_classification or is_rgb) else Resampling.bilinear

            dst_data = np.full((n_bands, ref_height, ref_width), np.nan, dtype=np.float32)

            for b in range(n_bands):
                src_band = src.read(b + 1).astype(np.float32)

                if src_nodata is not None:
                    try:
                        nd = float(src_nodata)
                        if not np.isnan(nd):
                            src_band[src_band == nd] = np.nan
                    except (ValueError, TypeError):
                        pass

                if np.issubdtype(np.dtype(src_dtype), np.unsignedinteger):
                    src_band[src_band == 0] = np.nan

                reproject(
                    source=src_band,
                    destination=dst_data[b],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=resamp,
                )

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if is_rgb:
                rgb_data = np.clip(np.nan_to_num(dst_data, nan=0), 0, 255).astype(np.uint8)
                dst_meta = {
                    'driver': 'GTiff', 'crs': ref_crs, 'transform': ref_transform,
                    'width': ref_width, 'height': ref_height,
                    'dtype': 'uint8', 'nodata': 0, 'count': n_bands,
                    'compress': 'deflate', 'tiled': True, 'blockxsize': 256, 'blockysize': 256,
                }
                with rasterio.open(dst_path, 'w', **dst_meta) as dst:
                    dst.write(rgb_data)
            else:
                dst_meta = {
                    'driver': 'GTiff', 'crs': ref_crs, 'transform': ref_transform,
                    'width': ref_width, 'height': ref_height,
                    'dtype': 'float32', 'nodata': float('nan'), 'count': n_bands,
                    'compress': 'deflate', 'predictor': 2, 'tiled': True,
                    'blockxsize': 256, 'blockysize': 256,
                }
                with rasterio.open(dst_path, 'w', **dst_meta) as dst:
                    dst.write(dst_data)

            valid_inside = int(np.sum(np.isfinite(dst_data[0]) & polygon_mask))
            return f"ok:{valid_inside}"

    except Exception as e:
        return f"error:{str(e)[:120]}"


# =========================================================================
# ALIGN GROUPS FOR A SINGLE CITY
# =========================================================================

def _align_city_groups(city_name, align_groups, ref_transform, ref_crs,
                       ref_width, ref_height, polygon_mask, stack_city_dir,
                       force_rerun=False, copy_jsons=True):
    """Process all ALIGN_GROUPS for one city. Returns (ok, skip, err, stats)."""
    city_ok = 0
    city_skip = 0
    city_err = 0
    city_stats = {}

    for gname, gconf in align_groups.items():
        src_dir = Path(gconf["src_dir"])
        dst_subdir = gconf["dst_subdir"]
        recursive = gconf.get("recursive", True)

        if not src_dir.exists():
            city_stats[gname] = {"status": "missing", "n_total": 0}
            continue

        tifs = sorted(src_dir.rglob("*.tif")) if recursive else sorted(src_dir.glob("*.tif"))
        tifs = [t for t in tifs if t.name != "desktop.ini"]

        if len(tifs) == 0:
            city_stats[gname] = {"status": "empty", "n_total": 0}
            continue

        rename_fn = gconf.get("rename_fn", None)

        g_ok = 0
        g_skip = 0
        g_err = 0
        g_errors = []

        for src_path in tifs:
            rel = src_path.relative_to(src_dir)
            if rename_fn:
                new_name = rename_fn(city_name, rel.name)
                rel = rel.parent / new_name if rel.parent != Path('.') else Path(new_name)
            dst_path = stack_city_dir / dst_subdir / rel

            result = align_raster_to_grid(
                src_path, dst_path, ref_transform, ref_crs,
                ref_width, ref_height, polygon_mask,
                force_rerun=force_rerun,
            )

            if result.startswith("ok"):
                g_ok += 1
                city_ok += 1
            elif result == "skipped":
                g_skip += 1
                city_skip += 1
            else:
                g_err += 1
                city_err += 1
                g_errors.append(f"{rel}: {result}")

        # copy metadata jsons
        if copy_jsons:
            jsons = sorted(src_dir.rglob("*.json")) if recursive else sorted(src_dir.glob("*.json"))
            for src_json in jsons:
                rel = src_json.relative_to(src_dir)
                dst_json = stack_city_dir / dst_subdir / rel
                if not dst_json.exists() or force_rerun:
                    dst_json.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_json, dst_json)

        city_stats[gname] = {
            "n_total": len(tifs),
            "n_ok": g_ok,
            "n_skipped": g_skip,
            "n_errors": g_err,
            "errors": g_errors[:5],
        }
        if g_ok > 0 or g_err > 0:
            print(f"    {gname:<25s}: {g_ok:>4d} ok, {g_skip:>4d} skip, {g_err:>4d} err  (of {len(tifs)})")
        elif g_skip > 0:
            print(f"    {gname:<25s}: {g_skip:>4d} skip (all up to date)")

    return city_ok, city_skip, city_err, city_stats


# =========================================================================
# COPY VECTOR FILES TO DATA_STACK
# =========================================================================

def _copy_vectors(city_name, stack_city_dir, cities_dir, unosat_cities_dir=None,
                  force_rerun=False):
    """Copy vector files (geojsons) to data_stack city dir."""
    cities_dir = Path(cities_dir)
    vector_candidates = {
        "AOI.geojson": cities_dir / city_name / "AOI.geojson",
        "buildings_overture.geojson": cities_dir / city_name / f"{city_name}_buildings_overture.geojson",
        "buildings_overture_with_damage.geojson": cities_dir / city_name / f"{city_name}_buildings_overture_with_damage.geojson",
    }
    if unosat_cities_dir:
        unosat_cities_dir = Path(unosat_cities_dir)
        vector_candidates["unosat_damage.geojson"] = unosat_cities_dir / city_name / "unosat_damage.geojson"
        vector_candidates["unosat_aoi.geojson"] = unosat_cities_dir / city_name / "aoi.geojson"

    copied = []
    for vname, vpath in vector_candidates.items():
        dst_path = stack_city_dir / vname
        if dst_path.exists() and not force_rerun:
            copied.append(vname)
            continue
        if not vpath.exists():
            continue
        try:
            shutil.copy2(vpath, dst_path)
            copied.append(vname)
        except Exception as e:
            print(f"      vector error {vname}: {e}")
    return copied


# =========================================================================
# MAIN RUN FUNCTION
# =========================================================================

def run_align(cities, stack_root, cities_dir,
              align_groups_fn, create_grid=True,
              force_rerun=False, unosat_cities_dir=None,
              copy_vectors=True, logs_dir=None,
              label="ALIGN"):
    """
    Align products for multiple cities.

    Args:
        cities:            list of city names
        stack_root:        Path to STACK_ROOT
        cities_dir:        Path to CITIES_DIR
        align_groups_fn:   callable(city_name) -> dict of ALIGN_GROUPS
                           Each group: {'src_dir': Path, 'dst_subdir': str, 'recursive': bool}
        create_grid:       bool - True for primary (create reference_grid.json),
                                  False for derived (read existing)
        force_rerun:       bool
        unosat_cities_dir: Path or None
        copy_vectors:      bool - copy geojson files (only for primary)
        logs_dir:          Path or None
        label:             str - label for print output

    Returns:
        list of (city, ok, skip, err, valid) tuples
    """
    stack_root = Path(stack_root)
    cities_dir = Path(cities_dir)

    t0 = time.time()
    print("=" * 70)
    print(f"STACK {label}: ALIGN TO UTM REFERENCE GRID")
    print("=" * 70)
    print(f"  Stack root:    {stack_root}")
    print(f"  Cities:        {len(cities)}")
    print(f"  Create grid:   {create_grid}")
    print(f"  Force rerun:   {force_rerun}")

    city_summary = []

    for ci, city_name in enumerate(cities):
        stack_city_dir = stack_root / city_name

        print(f"\n{'='*70}")
        print(f"  [{ci+1}/{len(cities)}] {city_name}")
        print(f"{'='*70}")

        # ----- load or create grid -----
        if create_grid:
            try:
                grid = load_canonical_grid(city_name, cities_dir)
            except Exception as e:
                print(f"    SKIP: {e}")
                continue

            ref_crs = grid['crs']
            ref_transform = grid['transform']
            ref_width = grid['width']
            ref_height = grid['height']

            # polygon mask
            poly_geom_utm = gpd.GeoDataFrame(
                geometry=[grid['poly_geom']], crs="EPSG:4326"
            ).to_crs(grid['crs']).geometry.iloc[0]

            polygon_mask = ~geometry_mask(
                [poly_geom_utm],
                out_shape=(ref_height, ref_width),
                transform=ref_transform,
                invert=False
            )

            # save reference_grid.json
            polygon_px = int(polygon_mask.sum())
            total_px = ref_width * ref_height
            ref_grid_info = {
                "crs": f"EPSG:{grid['utm_epsg']}",
                "utm_epsg": grid['utm_epsg'],
                "target_res_m": 10.0,
                "width": ref_width,
                "height": ref_height,
                "utm_bounds": grid['utm_bounds'],
                "transform": list(ref_transform)[:6],
                "n_patches_64": (ref_width // 64) * (ref_height // 64),
                "polygon_pixels": polygon_px,
                "total_pixels": total_px,
            }
            stack_city_dir.mkdir(parents=True, exist_ok=True)
            with open(stack_city_dir / "reference_grid.json", "w") as f:
                json.dump(ref_grid_info, f, indent=2)

            print(f"    Grid: {ref_width}x{ref_height} EPSG:{grid['utm_epsg']} "
                  f"({100*polygon_px/total_px:.1f}% polygon fill)")

        else:
            ref = load_reference_grid(city_name, stack_root, cities_dir)
            if ref is None:
                print(f"    No reference grid, skipping")
                city_summary.append((city_name, 0, 0, 0, False))
                continue
            ref_transform, ref_crs, ref_width, ref_height, polygon_mask = ref

        # ----- build align groups -----
        align_groups = align_groups_fn(city_name)

        # ----- align -----
        city_ok, city_skip, city_err, city_stats = _align_city_groups(
            city_name, align_groups, ref_transform, ref_crs,
            ref_width, ref_height, polygon_mask, stack_city_dir,
            force_rerun=force_rerun,
        )

        # ----- copy vectors -----
        if copy_vectors and create_grid:
            vectors = _copy_vectors(city_name, stack_city_dir, cities_dir,
                                    unosat_cities_dir, force_rerun)
            print(f"    Vectors: {len(vectors)} copied")

        # ----- validation -----
        all_tifs = sorted(stack_city_dir.rglob("*.tif"))
        shapes_found = set()
        for tif in all_tifs:
            try:
                with rasterio.open(tif) as src:
                    shapes_found.add((src.width, src.height))
            except Exception:
                pass
        valid = len(shapes_found) <= 1
        if valid:
            print(f"    VALID: {len(all_tifs)} rasters all {ref_width}x{ref_height}")
        else:
            print(f"    MISMATCH: shapes found: {shapes_found}")

        city_summary.append((city_name, city_ok, city_skip, city_err, valid))

    # ----- global summary -----
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"STACK {label} COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")

    print(f"\n  {'City':<25s} {'OK':>5s} {'Skip':>5s} {'Err':>5s} {'Valid'}")
    print(f"  {'-'*50}")
    for city, ok, skip, err, valid in city_summary:
        print(f"  {city:<25s} {ok:>5d} {skip:>5d} {err:>5d} {'OK' if valid else 'FAIL'}")

    total_ok = sum(r[1] for r in city_summary)
    total_skip = sum(r[2] for r in city_summary)
    total_err = sum(r[3] for r in city_summary)
    total_valid = sum(1 for r in city_summary if r[4])
    print(f"\n  Total: {total_ok} ok, {total_skip} skip, {total_err} err "
          f"across {len(city_summary)} cities ({total_valid} valid)")

    # save log
    if logs_dir:
        logs_dir = Path(logs_dir)
        log_path = logs_dir / f"stack_{label.lower()}.json"
        log_data = {
            "label": label,
            "started": t0,
            "elapsed_s": round(elapsed, 1),
            "cities": {c: {"ok": o, "skip": s, "err": e, "valid": v}
                       for c, o, s, e, v in city_summary},
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"  Log: {log_path}")

    return city_summary


# =========================================================================
# BBOX VERIFY
# =========================================================================

def verify_bbox(cities, stack_root):
    """Check all data_stack rasters match reference_grid.json dimensions.
    Returns (total_ok, total_mismatch, mismatches_list).
    """
    stack_root = Path(stack_root)
    t0 = time.time()

    print("=" * 70)
    print("BBOX-VERIFY: CHECK RASTER DIMENSIONS vs REFERENCE GRID")
    print("=" * 70)

    # filter to cities with reference_grid.json
    valid_cities = [c for c in cities if (stack_root / c / "reference_grid.json").exists()]
    print(f"  Cities with reference_grid.json: {len(valid_cities)}")

    total_ok = 0
    total_mismatch = 0
    total_checked = 0
    mismatches = []

    for city_name in valid_cities:
        stack_city_dir = stack_root / city_name
        ref_path = stack_city_dir / "reference_grid.json"
        with open(ref_path) as f:
            ref = json.load(f)
        expected_w = ref['width']
        expected_h = ref['height']

        all_tifs = sorted(stack_city_dir.rglob("*.tif"))
        if not all_tifs:
            continue

        city_ok = 0
        city_mismatch = 0

        for tif_path in all_tifs:
            try:
                with rasterio.open(tif_path) as src:
                    w, h = src.width, src.height
            except Exception:
                continue

            total_checked += 1
            if w == expected_w and h == expected_h:
                city_ok += 1
            else:
                city_mismatch += 1
                if city_mismatch <= 3:
                    rel = tif_path.relative_to(stack_city_dir)
                    mismatches.append((city_name, str(rel), w, h, expected_w, expected_h))

        total_ok += city_ok
        total_mismatch += city_mismatch

        status = "OK" if city_mismatch == 0 else f"MISMATCH ({city_mismatch})"
        print(f"  {city_name:<22s} expected={expected_w}x{expected_h}  "
              f"checked={len(all_tifs):>4d}  ok={city_ok:>4d}  [{status}]")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"BBOX-VERIFY COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Checked: {total_checked}, OK: {total_ok}, Mismatch: {total_mismatch}")

    if mismatches:
        print(f"\n  MISMATCHED FILES:")
        for city, rel, w, h, ew, eh in mismatches:
            print(f"    {city}/{rel}: {w}x{h} (expected {ew}x{eh})")
    else:
        print(f"  ALL RASTERS MATCH REFERENCE GRID.")

    return total_ok, total_mismatch, mismatches
