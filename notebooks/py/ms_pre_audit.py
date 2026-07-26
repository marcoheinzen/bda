# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
ms_pre_audit.py
MS product scanner: validates clipped TIFs on disk vs tracker.
Extracted from Cell PRE-AUDIT (NB03c). Updated for flat directory structure.

Notebook usage:
    from ms_pre_audit import run as run_ms_pre_audit
    run_ms_pre_audit(
        ms_dir=MS_DIR,
        cities_dir=CITIES_DIR,
        ms_tracking_file=MS_TRACKING_FILE,
        outputs_dir=OUTPUTS_DIR,
        cities_to_process=CITIES_TO_PROCESS,
        delete_bad_tifs=DELETE_BAD_TIFS,
        sample_bands=SAMPLE_BANDS,
        rasterio_sample_per_city=RASTERIO_SAMPLE_PER_CITY,
        min_file_bytes=MIN_FILE_BYTES,
        min_pixel_dim=MIN_PIXEL_DIM,
        nan_threshold=NAN_THRESHOLD,
        bbox_tolerance=BBOX_TOLERANCE,
    )
"""

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import json
import re
import time
import random
from pathlib import Path
from datetime import datetime

from aoi_date_extend_loader import load_aoi_bbox


def run(ms_dir, cities_dir, ms_tracking_file, outputs_dir,
        cities_to_process=None, delete_bad_tifs=False, sample_bands=None,
        rasterio_sample_per_city=5, min_file_bytes=1000,
        min_pixel_dim=10, nan_threshold=0.95, bbox_tolerance=0.01):
    """
    Args:
        ms_dir:                  Path - MS_DIR (flat: MS/{city}/*.tif)
        cities_dir:              Path - CITIES_DIR
        ms_tracking_file:        Path - MS_TRACKING_FILE
        outputs_dir:             Path - OUTPUTS_DIR
        cities_to_process:       list or None
        delete_bad_tifs:         bool
        sample_bands:            list or None
        rasterio_sample_per_city: int (0=skip deep check)
        min_file_bytes:          int
        min_pixel_dim:           int
        nan_threshold:           float
        bbox_tolerance:          float
    """
    ms_dir = Path(ms_dir)
    cities_dir = Path(cities_dir)
    ms_tracking_file = Path(ms_tracking_file)
    outputs_dir = Path(outputs_dir)

    print("=" * 70)
    print("CELL PRE-AUDIT: MS PRODUCT SCANNER")
    print("=" * 70)

    print(f"  MS dir:         {ms_dir}")
    print(f"  DELETE_BAD:     {delete_bad_tifs}")
    print(f"  Sample bands:   {sample_bands}")

    # =========================================================================
    # PART 1: VALIDATE EXISTING CLIPPED TIFS ON DISK (FLAT STRUCTURE)
    # =========================================================================

    t0 = time.time()
    total_ok = 0
    total_bad_bbox = 0
    total_bad_nan = 0
    total_bad_size = 0
    total_bad_dim = 0
    total_deleted = 0

    aoi_cache = {}
    def get_aoi_bounds(city_name):
        if city_name not in aoi_cache:
            try:
                geom = load_aoi_bbox(city_name, cities_dir)
                aoi_cache[city_name] = geom.bounds  # (minx, miny, maxx, maxy) in WGS84
            except (FileNotFoundError, ValueError):
                aoi_cache[city_name] = None
        return aoi_cache[city_name]

    city_summary = {}
    band_pattern = re.compile(r'_(\d{8})_(B\d{2}|B8A)_(\d+m)\.tif$')

    # flat structure: MS/{city}/*.tif (no period subdirs)
    city_dirs = sorted([d for d in ms_dir.iterdir()
                        if d.is_dir() and d.name not in ('metadata', 'temp', 'desktop.ini')])
    if cities_to_process:
        city_dirs = [d for d in city_dirs if d.name in cities_to_process]

    print(f"  Cities on disk: {len(city_dirs)}")
    print(f"  Rasterio sample: {rasterio_sample_per_city} per city (0=skip deep check)")

    for city_dir in city_dirs:
        city_name = city_dir.name
        city_ok = 0
        city_bad = 0
        city_issues = []
        city_dates = set()
        all_tif_paths = []

        exp = get_aoi_bounds(city_name)

        # flat: TIFs directly in city dir
        tifs = sorted(city_dir.glob('*.tif'))

        # also check legacy period subdirs if they still exist
        for period_dir in sorted(city_dir.iterdir()):
            if period_dir.is_dir() and period_dir.name not in ('composites', 'rgb', 'metadata', 'cloud_masks'):
                tifs.extend(sorted(period_dir.glob('*.tif')))

        for f in tifs:
            m = band_pattern.search(f.name)
            if m:
                city_dates.add(m.group(1))

            stat = f.stat()
            if stat.st_size < min_file_bytes:
                city_bad += 1
                total_bad_size += 1
                city_issues.append((f.parent.name, f.name, f"TINY({stat.st_size}B)"))
                if delete_bad_tifs:
                    f.unlink()
                    total_deleted += 1
            else:
                city_ok += 1
                all_tif_paths.append(f)

        # deep check: sample a few TIFs per city for bbox/NaN validation
        if rasterio_sample_per_city > 0 and all_tif_paths:
            random.seed(42)
            sample = random.sample(all_tif_paths, min(rasterio_sample_per_city, len(all_tif_paths)))
            for tif_path in sample:
                deep_issues = []
                try:
                    with rasterio.open(tif_path) as src:
                        w, h = src.width, src.height
                        if w < min_pixel_dim or h < min_pixel_dim:
                            deep_issues.append(f"SMALL({w}x{h})")
                            total_bad_dim += 1

                        data = src.read(1)
                        valid = np.isfinite(data) if np.issubdtype(data.dtype, np.floating) else (data != 0)
                        nan_frac = 1.0 - (np.sum(valid) / data.size) if data.size > 0 else 1.0
                        if nan_frac > nan_threshold:
                            deep_issues.append(f"NaN={nan_frac:.1%}")
                            total_bad_nan += 1

                        if exp is not None:
                            b = src.bounds
                            # transform AOI bounds (WGS84) to TIF native CRS
                            try:
                                exp_native = transform_bounds('EPSG:4326', src.crs, *exp)
                                # use pixel-aware tolerance: 1 pixel in native CRS units
                                res_x = abs(src.transform.a)
                                tol = max(res_x * 2, bbox_tolerance)
                                if (abs(b.left - exp_native[0]) > tol or abs(b.bottom - exp_native[1]) > tol or
                                    abs(b.right - exp_native[2]) > tol or abs(b.top - exp_native[3]) > tol):
                                    deep_issues.append(f"BBOX_MISMATCH(tol={tol:.1f})")
                                    total_bad_bbox += 1
                            except Exception as e:
                                deep_issues.append(f"BBOX_CRS_ERR({str(e)[:30]})")
                except Exception as e:
                    deep_issues.append(f"CORRUPT({str(e)[:40]})")

                if deep_issues:
                    city_issues.append(("SAMPLE", tif_path.name, ', '.join(deep_issues)))

        total_ok += city_ok

        if city_name not in city_summary:
            city_summary[city_name] = {'ok': city_ok, 'bad': city_bad, 'n_dates': len(city_dates), 'issues': city_issues}

    # =========================================================================
    # PART 2: CHECK DOWNLOAD TRACKER vs CLIPPED FILES
    # =========================================================================

    missing_clips = {}

    _ms_track_path = None
    if ms_tracking_file.exists():
        _ms_track_path = ms_tracking_file

    # also check v2 tracker
    _ms_v2 = outputs_dir / 'ms_download_tracking_v2.json'
    if _ms_v2.exists():
        _ms_track_path = _ms_v2

    if _ms_track_path and _ms_track_path.exists():
        print(f"\n  MS tracker: {_ms_track_path.name}")
        with open(_ms_track_path) as f:
            ms_tracker_raw = json.load(f)

        # v2 tracker nests entries under 'targets'
        if 'targets' in ms_tracker_raw:
            ms_tracker = ms_tracker_raw['targets']
        elif 'cities' in ms_tracker_raw:
            # v1 tracker: flatten cities -> periods -> entries
            ms_tracker = {}
            for _city, _cdata in ms_tracker_raw['cities'].items():
                if not isinstance(_cdata, dict):
                    continue
                for _period, _pdata in _cdata.items():
                    if not isinstance(_pdata, dict):
                        continue
                    for _dk, _entry in _pdata.items():
                        if isinstance(_entry, dict):
                            _entry.setdefault('city', _city)
                            _entry.setdefault('period', _period)
                            _entry.setdefault('date', _dk)
                            ms_tracker[f"{_city}_{_period}_{_dk}"] = _entry
        else:
            ms_tracker = ms_tracker_raw

        for tracker_key, entry in ms_tracker.items():
            if not isinstance(entry, dict):
                continue
            if entry.get('status') != 'success':
                continue

            city = entry.get('city', '')
            date_str = entry.get('date', entry.get('target_date', ''))
            period = entry.get('period', '')
            if not city or not date_str:
                continue

            if cities_to_process and city not in cities_to_process:
                continue

            date_compact = date_str.replace('-', '')
            scene_date = entry.get('scene_date', date_str).replace('-', '')

            # flat structure: check directly in city dir
            city_ms_dir = ms_dir / city
            b04_pattern_scene = f"{city}_S2_{scene_date}_B04_10m.tif"
            b04_pattern_target = f"{city}_S2_{date_compact}_B04_10m.tif"
            b04_file = city_ms_dir / b04_pattern_scene
            if not b04_file.exists():
                b04_file = city_ms_dir / b04_pattern_target

            # also check legacy period subdir
            if not b04_file.exists() and period:
                period_dir = city_ms_dir / period
                b04_file = period_dir / b04_pattern_scene
                if not b04_file.exists():
                    b04_file = period_dir / b04_pattern_target

            if not b04_file.exists():
                if city not in missing_clips:
                    missing_clips[city] = []
                missing_clips[city].append((period, date_str))

    # =========================================================================
    # SUMMARY
    # =========================================================================

    for city in sorted(city_summary.keys()):
        s = city_summary[city]
        n_missing = len(missing_clips.get(city, []))
        status_parts = []
        if s['bad'] > 0: status_parts.append(f"bad={s['bad']}")
        if n_missing > 0: status_parts.append(f"unclipped={n_missing}")
        status = "OK" if not status_parts else ', '.join(status_parts)
        if s['bad'] > 0 or n_missing > 0:
            print(f"  {city:<22s}  dates={s['n_dates']:>3d}  ok={s['ok']:>3d}  bad={s['bad']:>3d}  unclipped={n_missing:>3d}  [{status}]")
            for item in s['issues'][:3]:
                if len(item) == 3:
                    print(f"    {item[0]}/{item[1]}: {item[2]}")
                else:
                    print(f"    {item}")
            if len(s['issues']) > 3:
                print(f"    ... and {len(s['issues'])-3} more")
        else:
            print(f"  {city:<22s}  dates={s['n_dates']:>3d}  ok={s['ok']:>3d}  [OK]")

    if missing_clips:
        print(f"\n  UNCLIPPED SCENES (downloaded but no TIFs on disk):")
        for city in sorted(missing_clips.keys()):
            dates = missing_clips[city]
            print(f"    {city}: {len(dates)} scenes ({dates[0][0]}/{dates[0][1]} ... {dates[-1][0]}/{dates[-1][1]})")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"MS AUDIT COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Clipped TIFs OK:   {total_ok}")
    print(f"  Bad bbox:          {total_bad_bbox}")
    print(f"  Excessive NaN:     {total_bad_nan}")
    print(f"  Tiny/empty files:  {total_bad_size}")
    print(f"  Bad dimensions:    {total_bad_dim}")
    if delete_bad_tifs:
        print(f"  Files deleted:     {total_deleted}")
    print(f"  Unclipped scenes:  {sum(len(v) for v in missing_clips.values())}")
