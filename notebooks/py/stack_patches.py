# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_patches.py
Patch catalog generation + 64x64 patch extraction.

Functions:
  - generate_patch_catalog()  non-overlapping grid, AOI coverage, building stats
  - extract_patches()         per-building pre/post patches (xBD format)
  - generate_masks()          _mask.tif + _weight.tif for every existing patch

Notebook usage:
    from stack_patches import generate_patch_catalog, extract_patches
    generate_patch_catalog(stack_root=STACK_ROOT, cities=CITIES_TO_PROCESS,
                           load_aoi_fn=load_aoi, patch_sizes=[64], min_valid_pct=5.0)
    extract_patches(stack_root=STACK_ROOT, parquet_path=PARQUET_PATH, ...)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import Window
from shapely.geometry import box, mapping
from pathlib import Path
from datetime import datetime
import json
import time


def generate_patch_catalog(stack_root, cities, load_aoi_fn,
                           patch_sizes=None, min_valid_pct=5.0,
                           force_rerun=True):
    """Generate patch grid + AOI coverage catalog per city.
    Output: stack_root/{city}/patch_catalog_{size}.json
    """
    stack_root = Path(stack_root)
    if patch_sizes is None:
        patch_sizes = [64]

    print("=" * 70)
    print("PATCH-CATALOG: GENERATE PATCH GRID + AOI COVERAGE")
    print("=" * 70)

    city_dirs = [stack_root / c for c in cities
                 if (stack_root / c / "reference_grid.json").exists()]
    print(f"  Cities: {len(city_dirs)}")
    print(f"  Patch sizes: {patch_sizes}")
    print(f"  Min valid: {min_valid_pct}%")

    global_stats = []

    for ci, city_dir in enumerate(city_dirs):
        city_name = city_dir.name
        print(f"\n  [{ci+1}/{len(city_dirs)}] {city_name}")

        with open(city_dir / "reference_grid.json") as f:
            grid = json.load(f)

        ref_width = grid['width']
        ref_height = grid['height']
        utm_epsg = grid['utm_epsg']
        transform_list = grid['transform']
        from affine import Affine
        ref_transform = Affine(*transform_list[:6])

        try:
            aoi_row = load_aoi_fn(city_name)
            poly_geom = aoi_row.get('geometry', None)
        except Exception:
            poly_geom = None

        if poly_geom is not None:
            poly_gdf = gpd.GeoDataFrame(geometry=[poly_geom], crs="EPSG:4326").to_crs(f"EPSG:{utm_epsg}")
            polygon_mask = ~geometry_mask(
                poly_gdf.geometry.tolist(),
                out_shape=(ref_height, ref_width),
                transform=ref_transform,
                invert=False
            )
        else:
            polygon_mask = np.ones((ref_height, ref_width), dtype=bool)

        bldg_path = city_dir / "buildings_overture_with_damage.geojson"
        if not bldg_path.exists():
            bldg_path = city_dir / "buildings_overture.geojson"
        buildings = None
        if bldg_path.exists():
            try:
                buildings = gpd.read_file(bldg_path).to_crs(f"EPSG:{utm_epsg}")
            except Exception:
                pass

        for patch_size in patch_sizes:
            cat_path = city_dir / f"patch_catalog_{patch_size}.json"
            if cat_path.exists() and not force_rerun:
                print(f"    patch_catalog_{patch_size}.json exists, skipping")
                continue

            n_cols = ref_width // patch_size
            n_rows = ref_height // patch_size
            patches = []

            for row in range(n_rows):
                for col in range(n_cols):
                    r0 = row * patch_size
                    c0 = col * patch_size
                    patch_mask = polygon_mask[r0:r0+patch_size, c0:c0+patch_size]
                    valid_pct = 100.0 * patch_mask.sum() / (patch_size * patch_size)

                    if valid_pct < min_valid_pct:
                        continue

                    patch_info = {
                        "row": row, "col": col,
                        "r0": r0, "c0": c0,
                        "valid_pct": round(valid_pct, 1),
                    }

                    if buildings is not None and len(buildings) > 0:
                        x0 = ref_transform.c + c0 * ref_transform.a
                        y0 = ref_transform.f + r0 * ref_transform.e
                        x1 = x0 + patch_size * ref_transform.a
                        y1 = y0 + patch_size * ref_transform.e
                        patch_box = box(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
                        bldg_in = buildings[buildings.geometry.intersects(patch_box)]
                        patch_info["n_buildings"] = len(bldg_in)
                        if 'damage_binary' in bldg_in.columns:
                            db = pd.to_numeric(bldg_in['damage_binary'], errors='coerce')
                            patch_info["n_damaged"] = int(db.eq(1).sum())
                            patch_info["n_undamaged"] = int(db.eq(0).sum())
                            patch_info["n_excluded"] = int(db.eq(-1).sum() + db.isna().sum())
                        elif 'damage_label' in bldg_in.columns:
                            dl = pd.to_numeric(bldg_in['damage_label'], errors='coerce')
                            patch_info["n_damaged"] = int(dl.eq(1).sum())
                            patch_info["n_undamaged"] = int(dl.eq(0).sum())
                    else:
                        patch_info["n_buildings"] = 0

                    patches.append(patch_info)

            catalog = {
                "city": city_name,
                "patch_size": patch_size,
                "grid": f"{n_cols}x{n_rows}",
                "total_patches": n_cols * n_rows,
                "valid_patches": len(patches),
                "min_valid_pct": min_valid_pct,
                "generated": datetime.now().isoformat(),
                "patches": patches,
            }

            with open(cat_path, 'w') as f:
                json.dump(catalog, f, indent=2)

            n_with_bldg = sum(1 for p in patches if p.get("n_buildings", 0) > 0)
            print(f"    {patch_size}px: {len(patches)} valid patches ({n_with_bldg} with buildings)")
            global_stats.append((city_name, patch_size, len(patches), n_with_bldg))

    print(f"\n{'='*70}")
    print(f"PATCH-CATALOG COMPLETE")
    print(f"{'='*70}")
    return global_stats


def _resolve_composite_path(comp_dir, band):
    """Find composite TIF for band, handles both old and __ naming."""
    # __ convention first: s2__b02.tif
    p = comp_dir / f"s2__{band.lower()}.tif"
    if p.exists():
        return p
    # old convention: composite_B02.tif
    p = comp_dir / f"composite_{band}.tif"
    if p.exists():
        return p
    return None


def _resolve_card_path(flat_dir, pol, date_str):
    """Find CARD TIF for polarization+date, handles both old and __ naming."""
    # __ convention: s1__vv__20220103.tif
    p = flat_dir / f"s1__{pol.lower()}__{date_str}.tif"
    if p.exists():
        return p
    return None


def _load_city_buildings(stack_root, city_name, ref_crs, only_labeled=True):
    """Load buildings from GeoJSON (no parquet dependency)."""
    city_dir = stack_root / city_name
    bldg_path = city_dir / "buildings_overture_with_damage.geojson"
    if not bldg_path.exists():
        bldg_path = city_dir / "buildings_overture.geojson"
    if not bldg_path.exists():
        return None
    gdf = gpd.read_file(bldg_path).to_crs(ref_crs)
    gdf['building_id'] = [f"{city_name}_{i}" for i in range(len(gdf))]
    gdf['centroid_x'] = gdf.geometry.centroid.x
    gdf['centroid_y'] = gdf.geometry.centroid.y
    if only_labeled and 'damage_binary' in gdf.columns:
        db = pd.to_numeric(gdf['damage_binary'], errors='coerce')
        gdf = gdf[db.isin([0, 1])].copy()
    return gdf


def _get_battle_dates(city_dir):
    """Read battle_start and battle_stop from AOI.geojson city_polygon."""
    aoi_path = city_dir / "AOI.geojson"
    bs_str, be_str = '2022-02-24', ''
    if aoi_path.exists():
        with open(aoi_path) as f:
            gj = json.load(f)
        for feat in gj.get('features', []):
            props = feat.get('properties', {})
            if props.get('feature_type') == 'city_polygon':
                bs_str = str(props.get('battle_start', bs_str))[:10]
                be_str = str(props.get('battle_stop', ''))[:10]
                break
    from datetime import datetime as _dt
    try:
        bs = _dt.strptime(bs_str, '%Y-%m-%d')
    except ValueError:
        bs = _dt(2022, 2, 24)
    ongoing = be_str in ('', 'None', 'none', 'ongoing', 'NaT')
    try:
        be = None if ongoing else _dt.strptime(be_str, '%Y-%m-%d')
    except ValueError:
        be = None
    return bs, be, ongoing


def extract_patches(stack_root, cities, patch_dir,
                    patch_bands=None, patch_size=64,
                    only_labeled=True, force_rerun=False,
                    modality='ms', min_buildings=1, **kwargs):
    """Extract grid-based 64x64 pre/post patches from patch_catalog_64.json.

    Reads patch_catalog_{patch_size}.json per city (from generate_patch_catalog).
    For each valid grid cell with >= min_buildings:
      - Reads a 64x64 window from the closest pre-battle and post-battle scene
      - Writes {city}_r{row:03d}_c{col:03d}_pre.tif and _post.tif

    modality='ms':   RGB from multispectral/flat/ (B04, B03, B02)
    modality='card': VV+VH from SAR_CARD/flat/
    """
    stack_root = Path(stack_root)
    patch_dir = Path(patch_dir)
    if patch_bands is None:
        patch_bands = ['b04', 'b03', 'b02']

    print("=" * 70)
    print(f"PATCHES: {patch_size}x{patch_size} {modality.upper()} GRID PATCHES")
    print("=" * 70)

    t0 = time.time()
    total_patches = 0
    skipped = 0

    for city_name in sorted(cities):
        city_dir = stack_root / city_name
        ref_path = city_dir / "reference_grid.json"
        cat_path = city_dir / f"patch_catalog_{patch_size}.json"
        if not ref_path.exists():
            continue
        if not cat_path.exists():
            print(f"\n  {city_name}: no patch_catalog_{patch_size}.json, run PATCH-CATALOG first")
            continue

        with open(ref_path) as f:
            grid = json.load(f)
        from affine import Affine
        ref_transform = Affine(*grid['transform'][:6])
        ref_crs = f"EPSG:{grid['utm_epsg']}"

        with open(cat_path) as f:
            catalog = json.load(f)
        patches = [p for p in catalog['patches'] if p.get('n_buildings', 0) >= min_buildings]
        if not patches:
            continue

        print(f"\n  {city_name}: {len(patches)} grid patches (of {catalog['valid_patches']} valid)")

        city_patch_dir = patch_dir / city_name
        city_patch_dir.mkdir(parents=True, exist_ok=True)

        battle_start, battle_stop, ongoing = _get_battle_dates(city_dir)

        import re as _re
        from datetime import datetime as _dt

        # ---- resolve pre/post scenes per modality ----
        if modality == 'ms':
            flat_dir = city_dir / "multispectral" / "flat"
            if not flat_dir.exists():
                print(f"    No MS flat dir, skipping")
                continue

            ms_dates = set()
            for f in flat_dir.glob("s2__b02__*.tif"):
                m = _re.search(r'__(\d{8})\.tif$', f.name)
                if m:
                    ms_dates.add(m.group(1))

            if not ms_dates:
                print(f"    No MS dates found, skipping")
                continue

            # pre = last scene before battle_start (closest to war)
            pre_candidates = sorted([d for d in ms_dates
                                     if _dt.strptime(d, '%Y%m%d') < battle_start], reverse=True)
            # post = first scene after battle_stop; ongoing = last available scene
            if ongoing or battle_stop is None:
                post_candidates = sorted([d for d in ms_dates
                                          if _dt.strptime(d, '%Y%m%d') >= battle_start], reverse=True)
            else:
                post_candidates = sorted([d for d in ms_dates
                                          if _dt.strptime(d, '%Y%m%d') > battle_stop])

            if not pre_candidates or not post_candidates:
                print(f"    No pre or post MS dates, skipping")
                continue

            pre_date = pre_candidates[0]
            post_date = post_candidates[0]
            print(f"    MS pre={pre_date} post={post_date}")

            band_keys = list(patch_bands)
            pre_srcs = {}
            post_srcs = {}
            for band in band_keys:
                pp = flat_dir / f"s2__{band}__{pre_date}.tif"
                if pp.exists():
                    pre_srcs[band] = rasterio.open(pp)
                pp = flat_dir / f"s2__{band}__{post_date}.tif"
                if pp.exists():
                    post_srcs[band] = rasterio.open(pp)

            if len(pre_srcs) < len(band_keys) or len(post_srcs) < len(band_keys):
                print(f"    Missing bands, skipping")
                for s in pre_srcs.values(): s.close()
                for s in post_srcs.values(): s.close()
                continue

        elif modality == 'card':
            flat_dir = city_dir / "SAR_CARD" / "flat"
            if not flat_dir.exists():
                print(f"    No CARD flat dir, skipping")
                continue

            vv_dates = set()
            for f in flat_dir.glob("s1__vv__*.tif"):
                m = _re.search(r'__(\d{8})\.tif$', f.name)
                if m:
                    vv_dates.add(m.group(1))

            if not vv_dates:
                print(f"    No CARD dates, skipping")
                continue

            # pre = last scene before battle_start
            pre_dates = sorted([d for d in vv_dates if _dt.strptime(d, '%Y%m%d') < battle_start], reverse=True)
            # post = first scene after battle_stop; ongoing = last available scene
            if ongoing or battle_stop is None:
                post_dates = sorted([d for d in vv_dates if _dt.strptime(d, '%Y%m%d') >= battle_start], reverse=True)
            else:
                post_dates = sorted([d for d in vv_dates if _dt.strptime(d, '%Y%m%d') > battle_stop])

            if not pre_dates or not post_dates:
                print(f"    No pre or post CARD dates, skipping")
                continue

            pre_date = pre_dates[0]
            post_date = post_dates[0]
            print(f"    CARD pre={pre_date} post={post_date}")

            band_keys = ['vv', 'vh']
            pre_srcs = {}
            post_srcs = {}
            for pol in band_keys:
                pp = _resolve_card_path(flat_dir, pol, pre_date)
                if pp:
                    pre_srcs[pol] = rasterio.open(pp)
                pp = _resolve_card_path(flat_dir, pol, post_date)
                if pp:
                    post_srcs[pol] = rasterio.open(pp)

            if not pre_srcs or not post_srcs:
                print(f"    Missing CARD TIFs, skipping")
                continue
        else:
            raise ValueError(f"Unknown modality: {modality}")

        ref_src = list(pre_srcs.values())[0]

        city_extracted = 0
        for p in patches:
            r0 = p['r0']
            c0 = p['c0']
            row = p['row']
            col = p['col']
            patch_name = f"{city_name}_r{row:03d}_c{col:03d}"
            out_pre = city_patch_dir / f"{patch_name}_pre.tif"
            out_post = city_patch_dir / f"{patch_name}_post.tif"

            if out_pre.exists() and out_post.exists() and not force_rerun:
                skipped += 1
                continue

            window = Window(c0, r0, patch_size, patch_size)

            try:
                pre_data = np.stack([pre_srcs[b].read(1, window=window) for b in band_keys if b in pre_srcs])
                post_data = np.stack([post_srcs[b].read(1, window=window) for b in band_keys if b in post_srcs])
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"    READ FAIL {patch_name}: {e}")
                continue

            if pre_data.shape[1] != patch_size or pre_data.shape[2] != patch_size:
                skipped += 1
                if skipped <= 3:
                    print(f"    SHAPE FAIL {patch_name}: {pre_data.shape}")
                continue

            win_transform = rasterio.windows.transform(window, ref_transform)
            meta = {
                'driver': 'GTiff', 'crs': ref_src.crs, 'transform': win_transform,
                'width': patch_size, 'height': patch_size,
                'dtype': 'float32', 'count': pre_data.shape[0],
                'compress': 'deflate',
            }
            with rasterio.open(out_pre, 'w', **meta) as dst:
                dst.write(pre_data)
            with rasterio.open(out_post, 'w', **meta) as dst:
                dst.write(post_data)

            city_extracted += 1
            total_patches += 1

        for src in pre_srcs.values():
            src.close()
        for src in post_srcs.values():
            src.close()
        print(f"    Extracted: {city_extracted}, Skipped: {skipped}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PATCHES COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Extracted: {total_patches}, Skipped: {skipped}")
    return total_patches, skipped


def generate_masks(stack_root, cities, patch_dirs, patch_size=64,
                   only_labeled=True, force_rerun=False):
    """Generate _mask.tif for every existing patch.

    For each {name}_pre.tif in patch_dirs, writes:
      {name}_mask.tif (uint8) — 0=background, 1=undamaged building, 2=damaged building

    Consistent with xBD and BRIGHT collapsed class convention.
    NB09d derives from this single mask:
      weight = (mask > 0)   i.e. all building pixels
      target = (mask == 2)  i.e. damaged only

    Args:
        stack_root:   Path to data_stack root
        cities:       list of city names to process
        patch_dirs:   list of patch directories to scan (e.g. [MS_PATCH_DIR, CARD_PATCH_DIR])
        patch_size:   64
        only_labeled: if True, only rasterize buildings with damage_binary in {0, 1}
        force_rerun:  overwrite existing mask files
    """
    stack_root = Path(stack_root)
    if isinstance(patch_dirs, (str, Path)):
        patch_dirs = [Path(patch_dirs)]
    else:
        patch_dirs = [Path(d) for d in patch_dirs]

    print("=" * 70)
    print("MASKS: GENERATE 3-CLASS BUILDING DAMAGE MASKS")
    print("=" * 70)

    t0 = time.time()
    total_masks = 0
    total_skipped = 0

    for city_name in sorted(cities):
        city_dir = stack_root / city_name
        ref_path = city_dir / "reference_grid.json"
        if not ref_path.exists():
            continue

        with open(ref_path) as f:
            grid = json.load(f)
        from affine import Affine
        ref_transform = Affine(*grid['transform'][:6])
        ref_crs = f"EPSG:{grid['utm_epsg']}"

        buildings = _load_city_buildings(stack_root, city_name, ref_crs,
                                         only_labeled=only_labeled)
        if buildings is None or len(buildings) == 0:
            print(f"\n  {city_name}: no buildings, skipping masks")
            continue

        has_damage = 'damage_binary' in buildings.columns
        if has_damage:
            buildings['_dmg'] = pd.to_numeric(buildings['damage_binary'], errors='coerce').fillna(0).astype(int)
            labeled = buildings[buildings['_dmg'].isin([0, 1])].copy()
            # rasterize value: undamaged(0)->1, damaged(1)->2
            labeled['_rast_val'] = labeled['_dmg'].map({0: 1, 1: 2}).astype(np.uint8)
        else:
            labeled = buildings.copy()
            labeled['_rast_val'] = np.uint8(1)  # all undamaged if no labels

        n_dmg = int((labeled['_rast_val'] == 2).sum()) if len(labeled) > 0 else 0
        n_lbl = len(labeled)

        city_masks = 0
        city_skipped = 0

        for patch_dir in patch_dirs:
            city_patch_dir = patch_dir / city_name
            if not city_patch_dir.exists():
                continue

            pre_files = sorted(city_patch_dir.glob("*_pre.tif"))
            if not pre_files:
                continue

            for pre_path in pre_files:
                patch_stem = pre_path.stem.replace("_pre", "")
                out_mask = city_patch_dir / f"{patch_stem}_mask.tif"

                if out_mask.exists() and not force_rerun:
                    city_skipped += 1
                    continue

                with rasterio.open(pre_path) as src:
                    win_transform = src.transform
                    win_crs = src.crs
                    win_h = src.height
                    win_w = src.width

                meta_1band = {
                    'driver': 'GTiff', 'crs': win_crs, 'transform': win_transform,
                    'width': win_w, 'height': win_h,
                    'dtype': 'uint8', 'count': 1,
                    'compress': 'deflate', 'nodata': 255,
                }

                # rasterize all labeled buildings: 0=bg, 1=undamaged, 2=damaged
                # damaged (val=2) rasterized AFTER undamaged (val=1) so it overwrites
                # where buildings overlap
                if len(labeled) > 0:
                    shapes = [(mapping(row.geometry), int(row._rast_val))
                              for _, row in labeled.iterrows()
                              if row.geometry is not None and not row.geometry.is_empty]
                else:
                    shapes = []

                if shapes:
                    mask_arr = rasterize(
                        shapes,
                        out_shape=(win_h, win_w),
                        transform=win_transform,
                        fill=0,
                        dtype='uint8',
                    )
                else:
                    mask_arr = np.zeros((win_h, win_w), dtype=np.uint8)

                with rasterio.open(out_mask, 'w', **meta_1band) as dst:
                    dst.write(mask_arr[np.newaxis])

                city_masks += 1

            total_masks += city_masks
            total_skipped += city_skipped

        if city_masks > 0 or city_skipped > 0:
            print(f"\n  {city_name}: {city_masks} masks written, {city_skipped} skipped "
                  f"({n_lbl} labeled buildings, {n_dmg} damaged)")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"MASKS COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Written: {total_masks}, Skipped: {total_skipped}")
    return total_masks, total_skipped
