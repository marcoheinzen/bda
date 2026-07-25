# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
dl_prune_products.py
Product TIF pruning: identifies orphan, stale, surplus TIFs.
Uses product_scan.py for flat-aware scanning.
Uses aoi_date_extend_loader for temporal window stale detection.

Notebook usage:
    from dl_prune_products import run as run_prune_products
    PRUNE_RESULTS = run_prune_products(
        sar_coh_dir=SAR_COH_DIR,
        sar_card_dir=SAR_CARD_DIR,
        ms_dir=MS_DIR,
        landuse_dir=LANDUSE_DIR,
        cities_dir=CITIES_DIR,
        outputs_dir=OUTPUTS_DIR,
        cities_filter=CITIES_TO_PROCESS,
        dry_run=True,
    )
"""

import json
import shutil
import rasterio
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from product_scan import scan_coh_products, scan_card_products, scan_ms_products
from aoi_date_extend_loader import load_city_boundary_with_dates, compute_temporal_windows


def run(sar_coh_dir, sar_card_dir, ms_dir, landuse_dir, cities_dir, outputs_dir,
        cities_filter=None, dry_run=True,
        prune_stale_dates=True, prune_surplus_cities=True, prune_wrong_extent=True):

    sar_coh_dir = Path(sar_coh_dir)
    sar_card_dir = Path(sar_card_dir)
    ms_dir = Path(ms_dir)
    landuse_dir = Path(landuse_dir)
    cities_dir = Path(cities_dir)
    outputs_dir = Path(outputs_dir)

    print("=" * 80)
    print("DL-PRUNE-PRODUCTS: PRODUCT TIF PRUNING")
    print("=" * 80)
    print(f"  DRY_RUN: {dry_run}")
    print(f"  PRUNE_STALE_DATES: {prune_stale_dates}")
    print(f"  PRUNE_SURPLUS_CITIES: {prune_surplus_cities}")
    print(f"  PRUNE_WRONG_EXTENT: {prune_wrong_extent}")

    valid_cities = set(cities_filter) if cities_filter else None

    # load temporal windows
    city_windows = {}
    for city_dir in cities_dir.iterdir():
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        try:
            _, bs, be, ongoing, _ = load_city_boundary_with_dates(city, cities_dir)
            city_windows[city] = compute_temporal_windows(bs, be, ongoing)
        except Exception:
            pass

    print(f"  Valid cities: {len(valid_cities) if valid_cities else 'ALL'}")
    print(f"  Temporal windows: {len(city_windows)} cities")

    # AOI bbox loader
    def _get_bbox(city_name):
        try:
            import geopandas as gpd
            aoi = cities_dir / city_name / "AOI.geojson"
            if not aoi.exists():
                return None
            gdf = gpd.read_file(aoi)
            row = gdf[gdf['layer'] == 'aoi_bbox']
            return row.total_bounds if len(row) > 0 else None
        except Exception:
            return None

    def _check_extent(tif_path, bbox, tol=50):
        try:
            with rasterio.open(tif_path) as src:
                b = src.bounds
                if b.right < bbox[0] - tol or b.left > bbox[2] + tol:
                    return False
                if b.top < bbox[1] - tol or b.bottom > bbox[3] + tol:
                    return False
                return True
        except Exception:
            return None  # corrupt

    # =========================================================================
    # SCAN
    # =========================================================================
    print(f"\n--- Scanning products ---")
    coh_raw = scan_coh_products(sar_coh_dir)
    card_raw = scan_card_products(sar_card_dir)
    ms_raw = scan_ms_products(ms_dir)

    print(f"  COH: {len(coh_raw)} cities")
    print(f"  CARD: {len(card_raw)} cities")
    print(f"  MS: {len(ms_raw)} cities")

    prune_candidates = []

    # =========================================================================
    # 1. SURPLUS CITIES
    # =========================================================================
    if prune_surplus_cities and valid_cities:
        print(f"\n--- Surplus city check ---")
        n = 0
        for label, raw in [('COH', coh_raw), ('CARD', card_raw), ('MS', ms_raw)]:
            for city in raw:
                if city not in valid_cities:
                    for f in raw[city]:
                        prune_candidates.append((str(f['file']), f'surplus_city_{label}', f['size_mb']))
                        n += 1
        print(f"  Surplus files: {n}")

    # =========================================================================
    # 2. STALE DATES
    # =========================================================================
    if prune_stale_dates and city_windows:
        print(f"\n--- Stale date check ---")
        n = 0
        for city, files in card_raw.items():
            if city not in city_windows:
                continue
            w = city_windows[city]
            for f in files:
                if f['product_type'] != 'card_bs' or not f.get('date1'):
                    continue
                dt = datetime.strptime(f['date1'], '%Y%m%d')
                if dt < w['pre_battle_start'] or dt > w['post_battle_end']:
                    prune_candidates.append((str(f['file']), 'stale_card', f['size_mb']))
                    n += 1

        for city, files in ms_raw.items():
            if city not in city_windows:
                continue
            w = city_windows[city]
            for f in files:
                if f['product_type'] != 'clipped_band' or not f.get('date1'):
                    continue
                dt = datetime.strptime(f['date1'], '%Y%m%d')
                if dt < w['pre_battle_start'] or dt > w['post_battle_end']:
                    prune_candidates.append((str(f['file']), 'stale_ms', f['size_mb']))
                    n += 1
        print(f"  Stale files: {n}")

    # =========================================================================
    # 3. WRONG EXTENT
    # =========================================================================
    if prune_wrong_extent:
        print(f"\n--- Wrong extent check ---")
        n_extent = 0
        n_corrupt = 0
        for label, raw in [('COH', coh_raw), ('CARD', card_raw)]:
            for city, files in raw.items():
                bbox = _get_bbox(city)
                if bbox is None:
                    continue
                for f in files:
                    if f.get('subdir'):
                        continue
                    result = _check_extent(f['file'], bbox)
                    if result is None:
                        prune_candidates.append((str(f['file']), f'corrupt_{label}', f['size_mb']))
                        n_corrupt += 1
                    elif result is False:
                        prune_candidates.append((str(f['file']), f'wrong_extent_{label}', f['size_mb']))
                        n_extent += 1
        print(f"  Wrong extent: {n_extent}, Corrupt: {n_corrupt}")

    # =========================================================================
    # DEDUP + SUMMARY
    # =========================================================================
    seen = set()
    unique = []
    for p, r, s in prune_candidates:
        if p not in seen:
            seen.add(p)
            unique.append((p, r, s))
    prune_candidates = unique

    total_mb = sum(s for _, _, s in prune_candidates)

    reason_counts = defaultdict(lambda: {'count': 0, 'mb': 0})
    for _, reason, size in prune_candidates:
        reason_counts[reason]['count'] += 1
        reason_counts[reason]['mb'] += size

    print(f"\n{'='*80}")
    print(f"PRUNE SUMMARY")
    print(f"{'='*80}")
    print(f"  Candidates: {len(prune_candidates)} ({total_mb:.0f} MB)")
    for reason in sorted(reason_counts):
        rc = reason_counts[reason]
        print(f"    {reason:30s}: {rc['count']:5d} ({rc['mb']:.0f} MB)")

    if prune_candidates:
        print(f"\n  First 20:")
        for p, r, s in prune_candidates[:20]:
            print(f"    [{r:20s}] {Path(p).name} ({s:.1f} MB)")

    # =========================================================================
    # EXECUTE
    # =========================================================================
    deleted_count = 0
    deleted_mb = 0

    if not dry_run and prune_candidates:
        print(f"\n--- EXECUTING ---")
        for path_str, reason, size in prune_candidates:
            p = Path(path_str)
            try:
                if p.is_file():
                    p.unlink()
                    deleted_count += 1
                    deleted_mb += size
                elif p.is_dir():
                    shutil.rmtree(p)
                    deleted_count += 1
                    deleted_mb += size
            except Exception as e:
                print(f"  FAILED: {p.name}: {e}")
        print(f"  Deleted: {deleted_count} ({deleted_mb:.0f} MB)")

    # =========================================================================
    # REPORT
    # =========================================================================
    report = {
        'timestamp': datetime.now().isoformat(),
        'dry_run': dry_run,
        'candidates': len(prune_candidates),
        'total_mb': round(total_mb, 1),
        'deleted': deleted_count,
        'deleted_mb': round(deleted_mb, 1),
        'reasons': {r: v for r, v in reason_counts.items()},
    }
    report_path = outputs_dir / 'product_prune_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*80}")
    if dry_run:
        print(f"DRY RUN - nothing deleted. Set dry_run=False to execute.")
    else:
        print(f"PRUNE COMPLETE - deleted {deleted_count} ({deleted_mb:.0f} MB)")
    print(f"  Report: {report_path}")
    print(f"{'='*80}")

    return report
