# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
product_scan.py
Shared flat-aware product scanning for SAR_COH, SAR_CARD, MS directories.
Used by: NB03d (Cell 14D inventory), NB04a (product audit), NB04b (product prune),
         NB02e (dl_sync CARD TIF scan).

Supports both old ({city}_COH_VV_PRE_...) and new (s1__coh_vv__...) naming conventions.

Notebook usage:
    from product_scan import scan_coh_products, scan_card_products, scan_ms_products

All scanners return:
    {city_name: [{'file': Path, 'date1': str, 'date2': str|None, 'pol': str|None,
                  'period': str, 'product_type': str, 'size_mb': float}, ...]}
"""

import re
from pathlib import Path

SKIP_DIRS = frozenset({'metadata', 'temp', 'desktop.ini', 'composites', 'rgb',
                        'coherence_baseline', 'temporal_stats'})


def scan_coh_products(coh_dir, cities_filter=None):
    """Scan SAR_COH_DIR flat structure.
    Matches new convention: s1__coh_{vv|vh}__{d1}_{d2}.tif
    Matches old convention: {city}_COH_{VV|VH}_{PRE|CROSS|POST}_{d1}_{d2}.tif
    Also picks up coherence_baseline/ and temporal_stats/ subdirs.
    """
    results = {}
    if not coh_dir.exists():
        return results

    for city_dir in sorted(coh_dir.iterdir()):
        if not city_dir.is_dir() or city_dir.name in SKIP_DIRS:
            continue
        if cities_filter and city_dir.name not in cities_filter:
            continue

        city = city_dir.name
        city_files = []

        for item in city_dir.rglob('*.tif'):
            fname = item.name
            size_mb = item.stat().st_size / (1024 * 1024)
            rel = item.relative_to(city_dir)
            subdir = rel.parts[0] if len(rel.parts) > 1 else None

            # new convention with orbit: s1__coh_vv__o029__20220103_20220115.tif
            m_new_orbit = re.match(r'^s1__coh_(vv|vh)__o(\d{3})__(\d{8})_(\d{8})\.tif$', fname)
            if m_new_orbit and not subdir:
                city_files.append({
                    'file': item,
                    'date1': m_new_orbit.group(3),
                    'date2': m_new_orbit.group(4),
                    'pol': m_new_orbit.group(1).upper(),
                    'orbit': int(m_new_orbit.group(2)),
                    'period': 'unknown',
                    'product_type': 'coherence',
                    'subdir': subdir,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # new convention without orbit (legacy): s1__coh_vv__20220103_20220115.tif
            m_new = re.match(r'^s1__coh_(vv|vh)__(\d{8})_(\d{8})\.tif$', fname)
            if m_new and not subdir:
                city_files.append({
                    'file': item,
                    'date1': m_new.group(2),
                    'date2': m_new.group(3),
                    'pol': m_new.group(1).upper(),
                    'period': 'unknown',
                    'product_type': 'coherence',
                    'subdir': subdir,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # old convention: {city}_COH_VV_PRE_20220103_20220115.tif
            m = re.match(
                rf'^{re.escape(city)}_COH_(VV|VH)_(PRE|CROSS|POST)_(\d{{8}})_(\d{{8}})\.tif$',
                fname
            )
            if m:
                period_map = {'PRE': 'prebattle', 'CROSS': 'crossbattle', 'POST': 'postbattle'}
                city_files.append({
                    'file': item,
                    'date1': m.group(3),
                    'date2': m.group(4),
                    'pol': m.group(1),
                    'period': period_map[m.group(2)],
                    'product_type': 'coherence',
                    'subdir': subdir,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # old convention with dashed dates: {city}_COH_VV_PRE_2022-01-03_2022-01-15.tif
            m_dash = re.match(
                rf'^{re.escape(city)}_COH_(VV|VH)_(PRE|CROSS|POST)_(\d{{4}})-(\d{{2}})-(\d{{2}})_(\d{{4}})-(\d{{2}})-(\d{{2}})\.tif$',
                fname
            )
            if m_dash:
                period_map = {'PRE': 'prebattle', 'CROSS': 'crossbattle', 'POST': 'postbattle'}
                d1 = m_dash.group(3) + m_dash.group(4) + m_dash.group(5)
                d2 = m_dash.group(6) + m_dash.group(7) + m_dash.group(8)
                city_files.append({
                    'file': item,
                    'date1': d1,
                    'date2': d2,
                    'pol': m_dash.group(1),
                    'period': period_map[m_dash.group(2)],
                    'product_type': 'coherence',
                    'subdir': subdir,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # subdir products (coherence_baseline/, temporal_stats/)
            if subdir:
                d1, d2 = _extract_dates(fname)
                city_files.append({
                    'file': item,
                    'date1': d1,
                    'date2': d2,
                    'pol': _extract_pol(fname),
                    'period': subdir,
                    'product_type': subdir,
                    'subdir': subdir,
                    'size_mb': round(size_mb, 2),
                })

        if city_files:
            results[city] = city_files

    return results


def scan_card_products(card_dir, cities_filter=None):
    """Scan SAR_CARD_DIR flat structure.
    Matches new convention: s1__{vv|vh}__{YYYYMMDD}.tif
    Matches old convention: {city}_CARD_{VV|VH}_{YYYYMMDD}.tif
    Also picks up temporal_stats/ subdir.
    """
    results = {}
    if not card_dir.exists():
        return results

    for city_dir in sorted(card_dir.iterdir()):
        if not city_dir.is_dir() or city_dir.name in SKIP_DIRS:
            continue
        if cities_filter and city_dir.name not in cities_filter:
            continue

        city = city_dir.name
        city_files = []

        for item in city_dir.rglob('*.tif'):
            fname = item.name
            size_mb = item.stat().st_size / (1024 * 1024)
            rel = item.relative_to(city_dir)
            subdir = rel.parts[0] if len(rel.parts) > 1 else None

            # new convention with orbit: s1__vv__o029__20220103.tif
            m_new_orbit = re.match(r'^s1__(vv|vh)__o(\d{3})__(\d{8})\.tif$', fname)
            if m_new_orbit and not subdir:
                city_files.append({
                    'file': item,
                    'date1': m_new_orbit.group(3),
                    'date2': None,
                    'pol': m_new_orbit.group(1).upper(),
                    'orbit': int(m_new_orbit.group(2)),
                    'period': 'unknown',
                    'product_type': 'card_bs',
                    'subdir': None,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # new convention without orbit (legacy): s1__vv__20220103.tif
            m_new = re.match(r'^s1__(vv|vh)__(\d{8})\.tif$', fname)
            if m_new and not subdir:
                city_files.append({
                    'file': item,
                    'date1': m_new.group(2),
                    'date2': None,
                    'pol': m_new.group(1).upper(),
                    'period': 'unknown',
                    'product_type': 'card_bs',
                    'subdir': None,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # old convention: {city}_CARD_{VV|VH}_{YYYYMMDD}.tif
            m = re.match(
                rf'^{re.escape(city)}_CARD_(VV|VH)_(\d{{8}})\.tif$',
                fname
            )
            if m:
                city_files.append({
                    'file': item,
                    'date1': m.group(2),
                    'date2': None,
                    'pol': m.group(1),
                    'period': 'unknown',
                    'product_type': 'card_bs',
                    'subdir': None,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # subdir products (temporal_stats/)
            if subdir:
                city_files.append({
                    'file': item,
                    'date1': None,
                    'date2': None,
                    'pol': _extract_pol(fname),
                    'period': subdir,
                    'product_type': subdir,
                    'subdir': subdir,
                    'size_mb': round(size_mb, 2),
                })

        if city_files:
            results[city] = city_files

    return results


def scan_card_tifs_for_sync(card_dir, cities_filter=None):
    """Lightweight CARD scan for dl_sync: returns {city: {date_str: {pols: [VV,VH]}}}.
    Flat structure, compact YYYYMMDD dates."""
    from collections import defaultdict
    result = defaultdict(dict)
    if not card_dir.exists():
        return result

    for city_dir in sorted(card_dir.iterdir()):
        if not city_dir.is_dir() or city_dir.name in SKIP_DIRS:
            continue
        if cities_filter and city_dir.name not in cities_filter:
            continue

        city = city_dir.name
        for tif in city_dir.glob('*.tif'):
            # new convention with orbit: s1__vv__o029__20220103.tif
            m_new_orbit = re.match(r'^s1__(vv|vh)__o(\d{3})__(\d{8})\.tif$', tif.name)
            if m_new_orbit:
                d = m_new_orbit.group(3)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                if date_str not in result[city]:
                    result[city][date_str] = {'pols': [], 'orbit': int(m_new_orbit.group(2))}
                result[city][date_str]['pols'].append(m_new_orbit.group(1).upper())
                continue
            # new convention without orbit: s1__vv__20220103.tif (legacy)
            m_new = re.match(r'^s1__(vv|vh)__(\d{8})\.tif$', tif.name)
            if m_new:
                d = m_new.group(2)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                if date_str not in result[city]:
                    result[city][date_str] = {'pols': []}
                result[city][date_str]['pols'].append(m_new.group(1).upper())
                continue
            # old convention
            m = re.search(r'CARD_(VV|VH)_(\d{8})', tif.name)
            if m:
                d = m.group(2)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                if date_str not in result[city]:
                    result[city][date_str] = {'pols': []}
                result[city][date_str]['pols'].append(m.group(1))

    return result


def scan_ms_products(ms_dir, cities_filter=None):
    """Scan MS_DIR flat structure.
    Matches new convention: s2__{band}__{YYYYMMDD}.tif
    Matches old convention: {city}_S2_{YYYYMMDD}_{band}_{res}.tif
    Also picks up composites/, rgb/ subdirs.
    """
    results = {}
    if not ms_dir.exists():
        return results

    for city_dir in sorted(ms_dir.iterdir()):
        if not city_dir.is_dir() or city_dir.name in ('metadata', 'temp', 'desktop.ini'):
            continue
        if cities_filter and city_dir.name not in cities_filter:
            continue

        city = city_dir.name
        city_files = []

        for item in city_dir.rglob('*.tif'):
            fname = item.name
            size_mb = item.stat().st_size / (1024 * 1024)
            rel = item.relative_to(city_dir)
            subdir = rel.parts[0] if len(rel.parts) > 1 else None

            # new convention: s2__b02__20220108.tif, s2__scl__20220108.tif
            m_new_band = re.match(r'^s2__(b\d{2}|b8a|scl)__(\d{8})\.tif$', fname)
            if m_new_band and not subdir:
                band_upper = m_new_band.group(1).upper()
                city_files.append({
                    'file': item,
                    'date1': m_new_band.group(2),
                    'date2': None,
                    'band': band_upper,
                    'resolution': None,
                    'period': 'all',
                    'product_type': 'clipped_band',
                    'subdir': None,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # new convention derived: s2__cloud_mask__20220108.tif, s2__visibility__20220108.tif
            m_new_derived = re.match(r'^s2__(cloud_mask|visibility)__(\d{8})\.tif$', fname)
            if m_new_derived and not subdir:
                city_files.append({
                    'file': item,
                    'date1': m_new_derived.group(2),
                    'date2': None,
                    'band': None,
                    'resolution': None,
                    'period': 'all',
                    'product_type': m_new_derived.group(1),
                    'subdir': None,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # old convention band: {city}_S2_{YYYYMMDD}_{band}_{res}.tif
            m_band = re.search(r'_(\d{8})_(B\d{2}|B8A|SCL)_(\d+m)\.tif$', fname)
            if m_band and not subdir:
                city_files.append({
                    'file': item,
                    'date1': m_band.group(1),
                    'date2': None,
                    'band': m_band.group(2),
                    'resolution': m_band.group(3),
                    'period': 'all',
                    'product_type': 'clipped_band',
                    'subdir': None,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # old convention derived: {city}_S2_{YYYYMMDD}_cloud_mask.tif
            m_derived = re.search(r'_(\d{8})_(cloud_mask|visibility)\.tif$', fname)
            if m_derived and not subdir:
                city_files.append({
                    'file': item,
                    'date1': m_derived.group(1),
                    'date2': None,
                    'band': None,
                    'resolution': None,
                    'period': 'all',
                    'product_type': m_derived.group(2),
                    'subdir': None,
                    'size_mb': round(size_mb, 2),
                })
                continue

            # subdir products (composites/, rgb/, nbr/)
            if subdir:
                d1, _ = _extract_dates(fname)
                ptype = 'composite' if subdir == 'composites' else subdir
                city_files.append({
                    'file': item,
                    'date1': d1,
                    'date2': None,
                    'band': _extract_band(fname),
                    'resolution': None,
                    'period': rel.parts[1] if len(rel.parts) > 2 else subdir,
                    'product_type': ptype,
                    'subdir': subdir,
                    'size_mb': round(size_mb, 2),
                })

        if city_files:
            results[city] = city_files

    return results


# =============================================================================
# Summary helpers (for audit/inventory display)
# =============================================================================

def summarize_by_city_period(products):
    """Aggregate product list into {city: {period: {n_files, total_mb, dates, pols}}}."""
    summary = {}
    for city, files in products.items():
        city_summary = {}
        for f in files:
            period = f['period']
            if period not in city_summary:
                city_summary[period] = {'n_files': 0, 'total_mb': 0.0, 'dates': set(), 'pols': set()}
            s = city_summary[period]
            s['n_files'] += 1
            s['total_mb'] += f['size_mb']
            if f.get('date1'):
                s['dates'].add(f['date1'])
            if f.get('pol'):
                s['pols'].add(f['pol'])
        # convert sets
        for period, s in city_summary.items():
            s['dates'] = sorted(s['dates'])
            s['pols'] = sorted(s['pols'])
            s['total_mb'] = round(s['total_mb'], 1)
        summary[city] = city_summary
    return summary


def count_by_type(products):
    """Count files by product_type across all cities."""
    from collections import Counter
    counts = Counter()
    for city, files in products.items():
        for f in files:
            counts[f['product_type']] += 1
    return dict(counts)


# =============================================================================
# Internal helpers
# =============================================================================

def _extract_dates(filename):
    """Extract up to 2 YYYYMMDD dates from filename."""
    parts = filename.replace('.tif', '').split('_')
    found = []
    for p in parts:
        if len(p) == 8 and p.isdigit():
            found.append(p)
    d1 = found[0] if len(found) >= 1 else None
    d2 = found[1] if len(found) >= 2 else None
    return d1, d2


def _extract_pol(filename):
    """Extract polarization from filename (case-insensitive)."""
    fn_upper = filename.upper()
    if '_VV_' in fn_upper or '_VV.' in fn_upper or fn_upper.startswith('S1__VV'):
        return 'VV'
    elif '_VH_' in fn_upper or '_VH.' in fn_upper or fn_upper.startswith('S1__VH'):
        return 'VH'
    # new convention: s1__coh_vv__ or s1__coh_vh__
    if '_COH_VV' in fn_upper or 'COH_VV' in fn_upper:
        return 'VV'
    if '_COH_VH' in fn_upper or 'COH_VH' in fn_upper:
        return 'VH'
    return None


def _extract_band(filename):
    """Extract band from filename (case-insensitive)."""
    m = re.search(r'(B\d{2}|B8A|SCL)', filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None
