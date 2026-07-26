# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
product_prune.py
Derived product pruning for COH, CARD, MS TIFs.
Identifies surplus, stale, wrong-extent, superseded, and corrupt products.

Categories (for prune_selector):
  - superseded_coh:  COH pairs replaced by finer-grained pairs
  - stale_date:      products outside temporal windows
  - wrong_extent:    TIF bbox outside city AOI
  - surplus_city:    city not in valid_cities set
  - corrupt:         unreadable TIF files
  - empty_dir:       empty directories

Notebook usage:
    from product_prune import run as run_product_prune
    PRUNE_RESULTS = run_product_prune(
        coh_raw=NB04_AUDIT['coh_raw'],
        card_raw=NB04_AUDIT['card_raw'],
        ms_raw=NB04_AUDIT['ms_raw'],
        coh_superseded=NB04_AUDIT.get('coh_superseded', {}),
        cities_dir=CITIES_DIR,
        valid_cities=CITIES_TO_PROCESS,
        product_dirs=[SAR_COH_DIR, SAR_CARD_DIR, MS_DIR, LANDUSE_DIR],
        dry_run=True,
        prune_selector='ALL',
    )
"""

import re
import json
import rasterio
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from aoi_date_extend_loader import load_city_boundary_with_dates, compute_temporal_windows


# =========================================================================
# HELPERS
# =========================================================================

def _get_city_bbox_native(city_name, cities_dir):
    """Load AOI bbox from AOI.geojson, return (minx,miny,maxx,maxy) in WGS84."""
    aoi_file = Path(cities_dir) / city_name / "AOI.geojson"
    if not aoi_file.exists():
        return None
    gdf = gpd.read_file(aoi_file)
    aoi_row = gdf[gdf['feature_type'] == 'aoi_bbox']
    if len(aoi_row) == 0:
        return None
    return tuple(aoi_row.total_bounds)


def _check_tif_extent(tif_path, expected_bbox_wgs84, tolerance=50):
    """Check if TIF bbox overlaps expected AOI bbox. tolerance in CRS units (meters)."""
    try:
        with rasterio.open(tif_path) as src:
            from rasterio.warp import transform_bounds
            exp_native = transform_bounds('EPSG:4326', src.crs, *expected_bbox_wgs84)
            b = src.bounds
            if b.right < exp_native[0] - tolerance or b.left > exp_native[2] + tolerance:
                return False
            if b.top < exp_native[1] - tolerance or b.bottom > exp_native[3] + tolerance:
                return False
            return True
    except Exception:
        return None  # corrupt


def _load_temporal_windows(valid_cities, cities_dir):
    """Load temporal windows for all valid cities."""
    city_windows = {}
    for city_name in valid_cities:
        try:
            _, bs, be, ongoing, _ = load_city_boundary_with_dates(city_name, cities_dir)
            city_windows[city_name] = compute_temporal_windows(bs, be, ongoing)
        except Exception:
            pass
    return city_windows


# =========================================================================
# MAIN
# =========================================================================

def run(coh_raw=None, card_raw=None, ms_raw=None,
        coh_superseded=None,
        cities_dir=None,
        valid_cities=None,
        product_dirs=None,
        dry_run=True, verbose=True,
        prune_selector='ALL',
        stale_date_buffer_days=60):
    """
    Args:
        coh_raw:          dict from product_scan.scan_coh_products (or audit)
        card_raw:         dict from product_scan.scan_card_products (or audit)
        ms_raw:           dict from product_scan.scan_ms_products (or audit)
        coh_superseded:   dict from product_pre_audit.detect_superseded_coh (or audit)
        cities_dir:       Path to CITIES_DIR
        valid_cities:     list/set of city names in scope (for surplus check)
        product_dirs:     list of Paths to scan for empty dirs [SAR_COH_DIR, SAR_CARD_DIR, MS_DIR, LANDUSE_DIR]
        dry_run:          bool
        verbose:          bool
        prune_selector:   'ALL' or list of category strings

    Returns:
        dict with prune results
    """
    cities_dir = Path(cities_dir) if cities_dir else None
    coh_raw = coh_raw or {}
    card_raw = card_raw or {}
    ms_raw = ms_raw or {}
    coh_superseded = coh_superseded or {}
    product_dirs = [Path(d) for d in (product_dirs or [])]
    valid_cities = set(valid_cities) if valid_cities else set()

    # normalize prune_selector
    if isinstance(prune_selector, str) and prune_selector.upper() == 'ALL':
        prune_categories = None  # delete everything
    elif isinstance(prune_selector, (list, tuple)):
        prune_categories = set(s.lower() for s in prune_selector)
    else:
        prune_categories = None

    def should_prune(category):
        if prune_categories is None:
            return True
        return category.lower() in prune_categories

    print("=" * 80)
    print("PRODUCT PRUNE: IDENTIFY AND REMOVE STALE/ORPHAN DERIVED PRODUCTS")
    print("=" * 80)
    print(f"  DRY_RUN: {dry_run}")
    print(f"  PRUNE_SELECTOR: {prune_selector}")
    print(f"  STALE_DATE_BUFFER: {stale_date_buffer_days} days")
    print(f"  Valid cities: {len(valid_cities)}")

    # load temporal windows
    city_windows = _load_temporal_windows(valid_cities, cities_dir) if valid_cities and cities_dir else {}
    print(f"  Cities with temporal windows: {len(city_windows)}")

    # =====================================================================
    # COLLECT PRUNE CANDIDATES: (path_str, category, size_mb, detail)
    # =====================================================================
    candidates = []

    # --- 1. SURPLUS CITIES ---
    print(f"\n--- 1. Surplus city check ---")
    surplus_count = 0
    for label, raw in [('coh', coh_raw), ('card', card_raw), ('ms', ms_raw)]:
        if not valid_cities:
            continue
        for city in raw:
            if city not in valid_cities:
                for f in raw[city]:
                    candidates.append((str(f['file']), 'surplus_city', f['size_mb'],
                                       f"city={city} sensor={label}"))
                    surplus_count += 1
    print(f"  Surplus city files: {surplus_count}")

    # --- 2. SUPERSEDED COH PAIRS ---
    print(f"\n--- 2. Superseded COH pairs ---")
    superseded_count = 0
    for city, items in coh_superseded.items():
        for d1, d3, pol, d2, fpath in items:
            mb = fpath.stat().st_size / (1024*1024) if Path(fpath).exists() else 0
            candidates.append((str(fpath), 'superseded_coh', round(mb, 2),
                               f"{city} {pol} {d1}_{d3} -> {d1}_{d2}+{d2}_{d3}"))
            superseded_count += 1
    print(f"  Superseded COH pairs: {superseded_count}")

    # --- 3. STALE DATES ---
    print(f"\n--- 3. Stale date check (buffer={stale_date_buffer_days}d) ---")
    stale_count = 0
    from datetime import timedelta
    buffer = timedelta(days=stale_date_buffer_days)

    def _check_stale(dt, w):
        return dt < (w['pre_battle_start'] - buffer) or dt > (w['post_battle_end'] + buffer)

    def _window_str(w):
        return f"{(w['pre_battle_start'] - buffer).strftime('%Y%m%d')}..{(w['post_battle_end'] + buffer).strftime('%Y%m%d')}"

    for city, files in card_raw.items():
        if city not in city_windows:
            continue
        w = city_windows[city]
        for f in files:
            if f['product_type'] != 'card_bs' or not f.get('date1'):
                continue
            dt = datetime.strptime(f['date1'], '%Y%m%d')
            if _check_stale(dt, w):
                candidates.append((str(f['file']), 'stale_date', f['size_mb'],
                                   f"CARD {city} {f['date1']} outside {_window_str(w)}"))
                stale_count += 1

    for city, files in ms_raw.items():
        if city not in city_windows:
            continue
        w = city_windows[city]
        for f in files:
            if f['product_type'] != 'clipped_band' or not f.get('date1'):
                continue
            dt = datetime.strptime(f['date1'], '%Y%m%d')
            if _check_stale(dt, w):
                candidates.append((str(f['file']), 'stale_date', f['size_mb'],
                                   f"MS {city} {f['date1']} outside {_window_str(w)}"))
                stale_count += 1

    # COH stale date
    for city, files in coh_raw.items():
        if city not in city_windows:
            continue
        w = city_windows[city]
        for f in files:
            if f['product_type'] != 'coherence' or not f.get('date1'):
                continue
            dt = datetime.strptime(f['date1'], '%Y%m%d')
            if _check_stale(dt, w):
                candidates.append((str(f['file']), 'stale_date', f['size_mb'],
                                   f"COH {city} {f['date1']} outside {_window_str(w)}"))
                stale_count += 1

    print(f"  Stale date files: {stale_count}")

    # --- 4. WRONG EXTENT ---
    print(f"\n--- 4. Wrong extent check ---")
    extent_count = 0
    corrupt_count = 0

    bbox_cache = {}
    for label, raw in [('coh', coh_raw), ('card', card_raw)]:
        for city, files in raw.items():
            if city not in bbox_cache:
                bbox_cache[city] = _get_city_bbox_native(city, cities_dir) if cities_dir else None
            bbox = bbox_cache[city]
            if bbox is None:
                continue
            for f in files:
                if f.get('subdir'):
                    continue
                result = _check_tif_extent(f['file'], bbox)
                if result is None:
                    candidates.append((str(f['file']), 'corrupt', f['size_mb'],
                                       f"{label.upper()} {city} unreadable"))
                    corrupt_count += 1
                elif result is False:
                    candidates.append((str(f['file']), 'wrong_extent', f['size_mb'],
                                       f"{label.upper()} {city} bbox outside AOI"))
                    extent_count += 1

    print(f"  Wrong extent: {extent_count}")
    print(f"  Corrupt: {corrupt_count}")

    # --- 5. EMPTY DIRECTORIES ---
    print(f"\n--- 5. Empty directory check ---")
    empty_dirs = []
    for base_dir in product_dirs:
        if not base_dir.exists():
            continue
        for city_dir in base_dir.iterdir():
            if not city_dir.is_dir():
                continue
            for subdir in city_dir.iterdir():
                if subdir.is_dir() and not any(subdir.rglob('*')):
                    empty_dirs.append(str(subdir))
    print(f"  Empty directories: {len(empty_dirs)}")

    # =====================================================================
    # DEDUPLICATE
    # =====================================================================
    seen = set()
    unique = []
    for item in candidates:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    candidates = unique

    # =====================================================================
    # CATEGORY BREAKDOWN
    # =====================================================================
    cat_counts = defaultdict(lambda: {'count': 0, 'mb': 0.0})
    for _, cat, mb, _ in candidates:
        cat_counts[cat]['count'] += 1
        cat_counts[cat]['mb'] += mb

    print(f"\n{'='*80}")
    print(f"PRUNE CANDIDATE SUMMARY")
    print(f"{'='*80}")
    print(f"  Total candidates: {len(candidates)}")
    total_mb = sum(mb for _, _, mb, _ in candidates)
    print(f"  Total size: {total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
    for cat in sorted(cat_counts):
        c = cat_counts[cat]
        marker = " *" if should_prune(cat) else "  (skip)"
        print(f"    {cat:30s}: {c['count']:5d} files ({c['mb']:8.0f} MB){marker}")

    if empty_dirs:
        print(f"    {'empty_dir':30s}: {len(empty_dirs):5d} dirs")

    # =====================================================================
    # FILTER BY PRUNE_SELECTOR
    # =====================================================================
    to_delete = [(p, cat, mb, detail) for p, cat, mb, detail in candidates if should_prune(cat)]
    kept = [(p, cat, mb, detail) for p, cat, mb, detail in candidates if not should_prune(cat)]

    delete_mb = sum(mb for _, _, mb, _ in to_delete)
    kept_mb = sum(mb for _, _, mb, _ in kept)

    if prune_categories is not None:
        print(f"\n{'='*80}")
        print(f"PRUNE SELECTOR: {sorted(prune_categories)}")
        print(f"{'='*80}")
        print(f"  To delete: {len(to_delete)} files ({delete_mb:.0f} MB)")
        print(f"  Kept:      {len(kept)} files ({kept_mb:.0f} MB)")
        if kept:
            kept_cats = defaultdict(int)
            for _, cat, _, _ in kept:
                kept_cats[cat] += 1
            for cat, cnt in sorted(kept_cats.items()):
                print(f"    {cat}: {cnt}")

    # =====================================================================
    # VERBOSE LIST
    # =====================================================================
    if to_delete and verbose:
        print(f"\n  FILES TO DELETE ({len(to_delete)}):")
        for p, cat, mb, detail in sorted(to_delete, key=lambda x: (x[1], x[0])):
            print(f"    [{cat:20s}] {mb:7.1f} MB  {detail}")

    # =====================================================================
    # EXECUTE
    # =====================================================================
    deleted_count = 0
    deleted_mb = 0.0
    import shutil

    if not dry_run and to_delete:
        print(f"\n{'='*80}")
        print(f"EXECUTING PRUNE")
        print(f"{'='*80}")

        for p, cat, mb, detail in to_delete:
            pp = Path(p)
            try:
                if pp.is_file():
                    pp.unlink()
                    deleted_count += 1
                    deleted_mb += mb
                elif pp.is_dir():
                    shutil.rmtree(pp)
                    deleted_count += 1
                    deleted_mb += mb
            except Exception as e:
                print(f"  FAILED: {pp.name}: {e}")

        # empty dirs
        if should_prune('empty_dir'):
            for d in empty_dirs:
                dp = Path(d)
                try:
                    if dp.exists() and dp.is_dir() and not any(dp.rglob('*')):
                        dp.rmdir()
                except Exception:
                    pass

        print(f"  Deleted: {deleted_count} files ({deleted_mb:.0f} MB, {deleted_mb/1024:.1f} GB)")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*80}")
    if dry_run:
        print(f"DRY RUN - nothing deleted")
        print(f"  Total candidates: {len(candidates)} ({total_mb:.0f} MB)")
        print(f"  To delete:        {len(to_delete)} ({delete_mb:.0f} MB, {delete_mb/1024:.1f} GB)")
        if kept:
            print(f"  Kept (not in selector): {len(kept)} ({kept_mb:.0f} MB)")
    else:
        print(f"PRUNE COMPLETE")
        print(f"  Deleted: {deleted_count} files ({deleted_mb:.0f} MB, {deleted_mb/1024:.1f} GB)")
    print(f"{'='*80}")

    return {
        'candidates': candidates,
        'to_delete': to_delete,
        'kept': kept,
        'empty_dirs': empty_dirs,
        'cat_counts': dict(cat_counts),
        'deleted_count': deleted_count,
        'deleted_mb': round(deleted_mb, 1),
        'prune_selector': prune_selector,
        'dry_run': dry_run,
        'timestamp': datetime.now().isoformat(),
    }
