# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
product_pre_audit.py
Pre-processing product validator for COH, CARD, and MS TIFs.
Runs BEFORE NB03 computation to detect bad/stale/superseded products.

Checks per sensor:
  ALL:  tiny files, corrupt TIFs, NaN-heavy, bbox mismatch
  COH:  superseded pairs (wide pair replaced by two narrow pairs)
  CARD: (standard checks only)
  MS:   tracker vs disk cross-check

Notebook usage:
    from product_pre_audit import run as run_pre_audit
    run_pre_audit(
        sensors=['coh', 'card', 'ms'],
        sar_coh_dir=SAR_COH_DIR,
        sar_card_dir=SAR_CARD_DIR,
        ms_dir=MS_DIR,
        cities_dir=CITIES_DIR,
        cities_to_process=CITIES_TO_PROCESS,
        delete_bad=False,
        rasterio_sample_per_city=3,
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
from collections import defaultdict

from aoi_date_extend_loader import load_aoi_bbox
from product_scan import scan_coh_products, scan_card_products, scan_ms_products


# =========================================================================
# SHARED HELPERS
# =========================================================================

_aoi_bounds_cache = {}

def _get_aoi_bounds(city_name, cities_dir):
    key = f"{cities_dir}/{city_name}"
    if key not in _aoi_bounds_cache:
        try:
            geom = load_aoi_bbox(city_name, cities_dir)
            _aoi_bounds_cache[key] = geom.bounds
        except (FileNotFoundError, ValueError):
            _aoi_bounds_cache[key] = None
    return _aoi_bounds_cache[key]


def _deep_check_tif(tif_path, expected_bounds_wgs84, min_pixel_dim=10,
                     nan_threshold=0.90, bbox_tolerance=0.001):
    """Deep rasterio check: dimensions, NaN fraction, bbox match.
    Returns list of issue strings (empty = OK)."""
    issues = []
    try:
        with rasterio.open(tif_path) as src:
            w, h = src.width, src.height
            if w < min_pixel_dim or h < min_pixel_dim:
                issues.append(f"SMALL({w}x{h})")

            data = src.read(1)
            if np.issubdtype(data.dtype, np.floating):
                valid = np.isfinite(data)
            else:
                valid = (data != 0)
            nan_frac = 1.0 - (np.sum(valid) / data.size) if data.size > 0 else 1.0
            if nan_frac > nan_threshold:
                issues.append(f"NaN={nan_frac:.1%}")

            if expected_bounds_wgs84 is not None:
                try:
                    exp_native = transform_bounds('EPSG:4326', src.crs, *expected_bounds_wgs84)
                    res_x = abs(src.transform.a)
                    tol = max(res_x * 2, bbox_tolerance)
                    b = src.bounds
                    if (abs(b.left - exp_native[0]) > tol or abs(b.bottom - exp_native[1]) > tol or
                        abs(b.right - exp_native[2]) > tol or abs(b.top - exp_native[3]) > tol):
                        issues.append(f"BBOX_MISMATCH(tol={tol:.1f})")
                except Exception as e:
                    issues.append(f"BBOX_CRS_ERR({str(e)[:30]})")
    except Exception as e:
        issues.append(f"CORRUPT({str(e)[:40]})")
    return issues


def _validate_tifs(label, city_files_dict, cities_dir,
                   delete_bad=False, min_file_bytes=512,
                   rasterio_sample_per_city=3,
                   min_pixel_dim=10, nan_threshold=0.90, bbox_tolerance=0.001):
    """Generic TIF validation for any sensor. Returns (city_summary, totals)."""
    totals = {'ok': 0, 'bad_size': 0, 'bad_nan': 0, 'bad_bbox': 0,
              'bad_dim': 0, 'corrupt': 0, 'deleted': 0}
    city_summary = {}

    for city, files in sorted(city_files_dict.items()):
        exp = _get_aoi_bounds(city, cities_dir)
        ok = 0
        bad = 0
        issues = []
        valid_paths = []

        for f in files:
            fpath = f['file']
            if not fpath.exists():
                continue
            stat = fpath.stat()
            if stat.st_size < min_file_bytes:
                bad += 1
                totals['bad_size'] += 1
                issues.append((fpath.name, f"TINY({stat.st_size}B)"))
                if delete_bad:
                    fpath.unlink()
                    totals['deleted'] += 1
            else:
                ok += 1
                valid_paths.append(fpath)

        # deep check: sample
        if rasterio_sample_per_city > 0 and valid_paths:
            random.seed(42)
            sample = random.sample(valid_paths, min(rasterio_sample_per_city, len(valid_paths)))
            for tif_path in sample:
                deep_issues = _deep_check_tif(tif_path, exp, min_pixel_dim, nan_threshold, bbox_tolerance)
                for iss in deep_issues:
                    if 'NaN' in iss:
                        totals['bad_nan'] += 1
                    elif 'BBOX' in iss:
                        totals['bad_bbox'] += 1
                    elif 'SMALL' in iss:
                        totals['bad_dim'] += 1
                    elif 'CORRUPT' in iss:
                        totals['corrupt'] += 1
                if deep_issues:
                    issues.append((tif_path.name, ', '.join(deep_issues)))

        totals['ok'] += ok
        city_summary[city] = {'ok': ok, 'bad': bad, 'n_files': len(files), 'issues': issues}

    return city_summary, totals


def _print_sensor_summary(label, city_summary, totals):
    print(f"\n  {'City':25s} {'Files':>6s} {'OK':>5s} {'Bad':>5s} {'Issues'}")
    print(f"  {'-'*70}")
    for city in sorted(city_summary):
        s = city_summary[city]
        status = "OK" if not s['issues'] and s['bad'] == 0 else f"{len(s['issues'])} issues"
        if s['bad'] > 0 or s['issues']:
            print(f"  {city:25s} {s['n_files']:>6d} {s['ok']:>5d} {s['bad']:>5d} {status}")
            for item in s['issues'][:5]:
                if isinstance(item, tuple):
                    print(f"    {item[0]}: {item[1]}")
                else:
                    print(f"    {item}")
            if len(s['issues']) > 5:
                print(f"    ... +{len(s['issues'])-5} more")
        else:
            print(f"  {city:25s} {s['n_files']:>6d} {s['ok']:>5d} {s['bad']:>5d} OK")

    print(f"\n  {label} totals: OK={totals['ok']}, tiny={totals['bad_size']}, "
          f"NaN={totals['bad_nan']}, bbox={totals['bad_bbox']}, "
          f"dim={totals['bad_dim']}, corrupt={totals['corrupt']}, deleted={totals['deleted']}")


# =========================================================================
# COH: SUPERSEDED PAIR DETECTION
# =========================================================================

def detect_superseded_coh(coh_raw):
    """Detect COH pairs that are superseded by finer-grained pairs.

    A pair (d1, d3) is superseded if there exists d2 such that:
      d1 < d2 < d3 AND (d1,d2) exists AND (d2,d3) exists
    Only for same polarization. Only flags if BOTH replacement pairs exist.

    Returns: {city: [(d1, d3, pol, d2, file_path), ...]}
    """
    superseded = {}

    for city, files in coh_raw.items():
        # collect pairs by pol: {pol: {(d1,d2): file_path}}
        pairs_by_pol = defaultdict(dict)
        for f in files:
            if f['product_type'] != 'coherence':
                continue
            if f.get('date1') and f.get('date2') and f.get('pol'):
                pairs_by_pol[f['pol']][(f['date1'], f['date2'])] = f['file']

        city_superseded = []
        for pol, pairs in pairs_by_pol.items():
            pair_set = set(pairs.keys())
            # collect all dates that appear as endpoints
            all_dates = set()
            for d1, d2 in pair_set:
                all_dates.add(d1)
                all_dates.add(d2)

            for d1, d3 in list(pair_set):
                # only check pairs spanning more than 12 days
                if d3 <= d1:
                    continue
                # look for any intermediate date
                for d2 in sorted(all_dates):
                    if d1 < d2 < d3:
                        if (d1, d2) in pair_set and (d2, d3) in pair_set:
                            city_superseded.append((d1, d3, pol, d2, pairs[(d1, d3)]))
                            break  # one witness is enough

        if city_superseded:
            superseded[city] = city_superseded

    return superseded


# =========================================================================
# MAIN RUN FUNCTION
# =========================================================================

def run(sensors=None,
        sar_coh_dir=None, sar_card_dir=None, ms_dir=None,
        cities_dir=None,
        ms_tracking_file=None, outputs_dir=None,
        cities_to_process=None,
        delete_bad=False,
        rasterio_sample_per_city=3,
        min_file_bytes=512, min_pixel_dim=10,
        nan_threshold=0.90, bbox_tolerance=0.001,
        verbose=True):
    """
    Args:
        sensors:     list of 'coh', 'card', 'ms' (None = all)
        sar_coh_dir: Path to SAR_COH_DIR
        sar_card_dir: Path to SAR_CARD_DIR
        ms_dir:      Path to MS_DIR
        cities_dir:  Path to CITIES_DIR
        ms_tracking_file: Path to MS_TRACKING_FILE (for MS tracker cross-check)
        outputs_dir: Path to OUTPUTS_DIR (for v2 tracker check)
        cities_to_process: list of city names or None
        delete_bad:  bool - delete tiny/corrupt files
        rasterio_sample_per_city: int (0=skip deep check)
        min_file_bytes: int
        min_pixel_dim:  int
        nan_threshold:  float
        bbox_tolerance: float
        verbose:        bool

    Returns:
        dict with keys per sensor: coh_summary, card_summary, ms_summary,
        coh_superseded, totals_coh, totals_card, totals_ms
    """
    if sensors is None:
        sensors = ['coh', 'card', 'ms']
    sensors = [s.lower() for s in sensors]

    cities_dir = Path(cities_dir) if cities_dir else None

    t0 = time.time()
    print("=" * 80)
    print("PRODUCT PRE-AUDIT")
    print("=" * 80)
    print(f"  Sensors: {sensors}")
    print(f"  DELETE_BAD: {delete_bad}")
    print(f"  Sample per city: {rasterio_sample_per_city}")
    if cities_to_process:
        print(f"  Cities filter: {len(cities_to_process)} cities")

    result = {'timestamp': datetime.now().isoformat()}

    # =====================================================================
    # COH
    # =====================================================================
    if 'coh' in sensors and sar_coh_dir:
        sar_coh_dir = Path(sar_coh_dir)
        print(f"\n{'='*80}")
        print(f"COHERENCE PRODUCTS (SAR_COH_DIR)")
        print(f"{'='*80}")
        print(f"  Dir: {sar_coh_dir}")

        coh_raw = scan_coh_products(sar_coh_dir, cities_to_process or None)
        coh_total = sum(len(f) for f in coh_raw.values())
        print(f"  Cities: {len(coh_raw)}, TIFs: {coh_total}")

        # standard validation
        coh_summary, totals_coh = _validate_tifs(
            'COH', coh_raw, cities_dir,
            delete_bad=delete_bad, min_file_bytes=min_file_bytes,
            rasterio_sample_per_city=rasterio_sample_per_city,
            min_pixel_dim=min_pixel_dim, nan_threshold=nan_threshold,
            bbox_tolerance=bbox_tolerance,
        )
        _print_sensor_summary('COH', coh_summary, totals_coh)

        # superseded pair detection
        coh_superseded = detect_superseded_coh(coh_raw)
        n_superseded = sum(len(v) for v in coh_superseded.values())
        superseded_mb = 0
        if coh_superseded:
            print(f"\n  SUPERSEDED COH PAIRS: {n_superseded}")
            print(f"  (wide pair replaced by two narrow pairs - safe to prune)")
            for city in sorted(coh_superseded):
                for d1, d3, pol, d2, fpath in coh_superseded[city]:
                    mb = fpath.stat().st_size / (1024*1024) if fpath.exists() else 0
                    superseded_mb += mb
                    if verbose:
                        print(f"    {city:25s} {pol} {d1}_{d3} superseded by {d1}_{d2} + {d2}_{d3}  ({mb:.1f} MB)")
            print(f"  Total superseded: {n_superseded} files ({superseded_mb:.0f} MB)")
        else:
            print(f"\n  Superseded COH pairs: 0 (clean)")

        result['coh_raw'] = coh_raw
        result['coh_summary'] = coh_summary
        result['totals_coh'] = totals_coh
        result['coh_superseded'] = coh_superseded

    # =====================================================================
    # CARD
    # =====================================================================
    if 'card' in sensors and sar_card_dir:
        sar_card_dir = Path(sar_card_dir)
        print(f"\n{'='*80}")
        print(f"CARD-BS PRODUCTS (SAR_CARD_DIR)")
        print(f"{'='*80}")
        print(f"  Dir: {sar_card_dir}")

        card_raw = scan_card_products(sar_card_dir, cities_to_process or None)
        card_total = sum(len(f) for f in card_raw.values())
        print(f"  Cities: {len(card_raw)}, TIFs: {card_total}")

        card_summary, totals_card = _validate_tifs(
            'CARD', card_raw, cities_dir,
            delete_bad=delete_bad, min_file_bytes=min_file_bytes,
            rasterio_sample_per_city=rasterio_sample_per_city,
            min_pixel_dim=min_pixel_dim, nan_threshold=nan_threshold,
            bbox_tolerance=bbox_tolerance,
        )
        _print_sensor_summary('CARD', card_summary, totals_card)

        result['card_raw'] = card_raw
        result['card_summary'] = card_summary
        result['totals_card'] = totals_card

    # =====================================================================
    # MS
    # =====================================================================
    if 'ms' in sensors and ms_dir:
        ms_dir = Path(ms_dir)
        print(f"\n{'='*80}")
        print(f"MULTISPECTRAL PRODUCTS (MS_DIR)")
        print(f"{'='*80}")
        print(f"  Dir: {ms_dir}")

        ms_raw = scan_ms_products(ms_dir, cities_to_process or None)
        ms_total = sum(len(f) for f in ms_raw.values())
        ms_clipped = sum(1 for files in ms_raw.values() for f in files if f['product_type'] == 'clipped_band')
        ms_composites = sum(1 for files in ms_raw.values() for f in files if f['product_type'] == 'composite')
        print(f"  Cities: {len(ms_raw)}, TIFs: {ms_total} (clipped={ms_clipped}, composites={ms_composites})")

        ms_summary, totals_ms = _validate_tifs(
            'MS', ms_raw, cities_dir,
            delete_bad=delete_bad, min_file_bytes=min_file_bytes,
            rasterio_sample_per_city=rasterio_sample_per_city,
            min_pixel_dim=min_pixel_dim, nan_threshold=nan_threshold,
            bbox_tolerance=bbox_tolerance,
        )
        _print_sensor_summary('MS', ms_summary, totals_ms)

        # MS tracker cross-check
        ms_missing_clips = {}
        if ms_tracking_file and outputs_dir:
            ms_tracking_file = Path(ms_tracking_file)
            outputs_dir = Path(outputs_dir)
            _ms_track_path = None
            if ms_tracking_file.exists():
                _ms_track_path = ms_tracking_file
            _ms_v2 = outputs_dir / 'ms_download_tracking_v2.json'
            if _ms_v2.exists():
                _ms_track_path = _ms_v2

            if _ms_track_path and _ms_track_path.exists():
                print(f"\n  MS tracker cross-check: {_ms_track_path.name}")
                with open(_ms_track_path) as f:
                    ms_tracker_raw = json.load(f)

                if 'targets' in ms_tracker_raw:
                    ms_tracker = ms_tracker_raw['targets']
                elif 'cities' in ms_tracker_raw:
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
                    city_ms_dir = ms_dir / city
                    # new convention first
                    b04_file = city_ms_dir / f"s2__b04__{scene_date}.tif"
                    if not b04_file.exists():
                        b04_file = city_ms_dir / f"s2__b04__{date_compact}.tif"
                    # fallback: old convention
                    if not b04_file.exists():
                        b04_file = city_ms_dir / f"{city}_S2_{scene_date}_B04_10m.tif"
                    if not b04_file.exists():
                        b04_file = city_ms_dir / f"{city}_S2_{date_compact}_B04_10m.tif"
                    if not b04_file.exists() and period:
                        period_dir = city_ms_dir / period
                        b04_file = period_dir / f"{city}_S2_{scene_date}_B04_10m.tif"
                        if not b04_file.exists():
                            b04_file = period_dir / f"{city}_S2_{date_compact}_B04_10m.tif"
                    if not b04_file.exists():
                        if city not in ms_missing_clips:
                            ms_missing_clips[city] = []
                        ms_missing_clips[city].append((period, date_str))

                if ms_missing_clips:
                    n_miss = sum(len(v) for v in ms_missing_clips.values())
                    print(f"  Unclipped scenes (downloaded but no TIFs): {n_miss}")
                    for city in sorted(ms_missing_clips):
                        dates = ms_missing_clips[city]
                        print(f"    {city}: {len(dates)} scenes")
                else:
                    print(f"  Tracker cross-check: all downloaded scenes have clipped TIFs")

        result['ms_raw'] = ms_raw
        result['ms_summary'] = ms_summary
        result['totals_ms'] = totals_ms
        result['ms_missing_clips'] = ms_missing_clips

    # =====================================================================
    # OVERALL SUMMARY
    # =====================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"PRE-AUDIT COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*80}")

    for sensor in sensors:
        key = f"totals_{sensor}"
        if key in result:
            t = result[key]
            print(f"  {sensor.upper():5s}: OK={t['ok']}, tiny={t['bad_size']}, NaN={t['bad_nan']}, "
                  f"bbox={t['bad_bbox']}, dim={t['bad_dim']}, corrupt={t['corrupt']}, deleted={t['deleted']}")

    if 'coh_superseded' in result:
        n = sum(len(v) for v in result['coh_superseded'].values())
        if n > 0:
            print(f"  COH superseded pairs: {n} (prunable via NB04a)")

    if 'ms_missing_clips' in result:
        n = sum(len(v) for v in result['ms_missing_clips'].values())
        if n > 0:
            print(f"  MS unclipped scenes: {n}")

    if delete_bad:
        total_deleted = sum(result.get(f'totals_{s}', {}).get('deleted', 0) for s in sensors)
        print(f"  Total files deleted: {total_deleted}")

    return result
