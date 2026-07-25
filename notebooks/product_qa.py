# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
product_qa.py  --  Reusable QA/QC functions for BDA satellite products.

Used by NB04a. All functions accept paths/config from global_setup.py,
nothing hardcoded. CRS derived per-city from AOI.geojson aoi_bbox feature.

Directory structure scanned:
    SAR_CARD_DIR/{city}/                    CARD flat  (s1__{pol}__{date}.tif)
    SAR_CARD_DIR/{city}/temporal_stats/     CARD stats (s1__{pol}__{phase}__{stat}.tif)
    SAR_COH_DIR/{city}/                     COH flat   (s1__coh_{pol}__{d1}_{d2}.tif)
    SAR_COH_DIR/{city}/coherence_baseline/  COH stats  (s1__coh__baseline__{stat}.tif)
    SAR_COH_DIR/{city}/post_baseline/       COH post   (s1__coh__post_baseline__{stat}.tif)
    MS_DIR/{city}/                          MS flat    (s2__{band}__{date}.tif)
    MS_DIR/{city}/composites/{period}/      MS comps   (composite_{band}.tif)
    MS_DIR/{city}/nbr/                      NBR        (s2__nbr__{date}.tif)
    MS_DIR/{city}/rgb/                      RGB        (s2__rgb__{date}.tif)
    TEMPORAL_ROOT/{city}/COH/rolling/       COH roll   (s1__coh_vv__roll{N}__{date}.tif)
    TEMPORAL_ROOT/{city}/COH/zscore/        COH zscore (s1__coh_vv__zscore__{date}.tif)
    TEMPORAL_ROOT/{city}/CARD/rolling/      CARD roll  (s1__{pol}__roll{N}__{date}.tif)
    TEMPORAL_ROOT/{city}/CARD/rolling_stats/ CARD rstats
    TEMPORAL_ROOT/{city}/CARD/block_stats/  CARD blk   (s1__{pol}__blk{N}__{stat}.tif)
    STACK_DIR/{city}/                       masks      (building_labels.tif, damage_mask.tif)
"""

import re
import json
import time
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ============================================================================
# DEFAULT THRESHOLDS
# ============================================================================

DEFAULT_THRESHOLDS = {
    "card_db_min": -65.0,
    "card_db_max": 15.0,
    "card_db_nesz": -22.0,
    "card_nodata_leak": -99.0,
    "card_jump_db": 5.0,

    "coh_min": -0.15,
    "coh_max": 1.1,
    "coh_decorrelated": 0.2,
    "coh_suspect_high": 0.95,
    "coh_all_zero_pct": 85.0,

    "ms_dn_min": -2000,
    "ms_dn_max": 20000,
    "ms_dn_typical_max": 12000,
    "ms_max_cloud_pct": 50.0,
    "ms_scl_cloud_classes": [8, 9, 10],

    "ndvi_range": (-1.0, 1.0),
    "nbr_range": (-1.0, 1.0),
    "dnbr_range": (-2.0, 2.0),

    "stat_min_obs": 3,
    "zscore_extreme": 3.0,
    "landuse_nodata": 255,

    "damage_mask_valid": [-1, 0, 1],
    "building_labels_min": 0,

    "max_bbox_diff_deg": 0.001,

    "s1_revisit_days": 12,
    "s2_revisit_days": 5,
    "gap_warn_factor": 2.5,
    "gap_fail_factor": 5.0,
}


# ============================================================================
# CRS UTILITIES
# ============================================================================

def get_city_expected_crs(city_name, cities_dir):
    aoi_file = Path(cities_dir) / city_name / "AOI.geojson"
    if not aoi_file.exists():
        return {"utm_epsg": None, "wgs84_epsg": 4326}
    with open(aoi_file) as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        if props.get("feature_type") == "aoi_bbox":
            utm = props.get("utm_epsg")
            if utm is not None:
                utm = int(utm)
            return {"utm_epsg": utm, "wgs84_epsg": 4326}
    return {"utm_epsg": None, "wgs84_epsg": 4326}


def get_city_aoi_bbox_wgs84(city_name, cities_dir):
    aoi_file = Path(cities_dir) / city_name / "AOI.geojson"
    if not aoi_file.exists():
        return None
    with open(aoi_file) as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        if props.get("feature_type") == "aoi_bbox":
            geom = feat.get("geometry")
            if geom and geom.get("coordinates"):
                from shapely.geometry import shape
                return shape(geom).bounds
    return None


def crs_matches_city(raster_crs, city_name, cities_dir):
    expected = get_city_expected_crs(city_name, cities_dir)
    raster_epsg = None
    try:
        raster_epsg = CRS(raster_crs).to_epsg()
    except Exception:
        pass
    if raster_epsg is None:
        return False, "CRS_UNREADABLE"
    valid_set = {expected["wgs84_epsg"]}
    if expected["utm_epsg"] is not None:
        valid_set.add(expected["utm_epsg"])
        # accept adjacent UTM zones (cities on zone boundaries)
        base_zone = expected["utm_epsg"]
        if 32600 < base_zone < 32661:
            valid_set.add(base_zone - 1)
            valid_set.add(base_zone + 1)
        elif 32700 < base_zone < 32761:
            valid_set.add(base_zone - 1)
            valid_set.add(base_zone + 1)
    if raster_epsg in valid_set:
        return True, raster_epsg
    return False, raster_epsg


# ============================================================================
# FILENAME CLASSIFIERS
# ============================================================================

_RE_CARD_NEW = re.compile(r"^s1__(vv|vh)(?:__o\d{3})?__(\d{8})\.tif$", re.IGNORECASE)
_RE_CARD_OLD = re.compile(r"^(.+?)_CARD_(VV|VH)_(\d{8})\.tif$", re.IGNORECASE)

_RE_COH_NEW = re.compile(r"^s1__coh_(vv|vh)(?:__o\d{3})?__(\d{8})_(\d{8})\.tif$", re.IGNORECASE)
_RE_COH_OLD = re.compile(
    r"^(.+?)_COH_(VV|VH)_(PRE|CROSS|POST|BIWEEKLY|MONTHLY)_(\d{8})_(\d{8})\.tif$",
    re.IGNORECASE)

_RE_MS_NEW = re.compile(r"^s2__([a-z0-9]+)__(\d{8})\.tif$", re.IGNORECASE)
_RE_MS_OLD = re.compile(
    r"^(.+?)_S2_(\d{8})_(B\d{2}|B8A|SCL)_(\d+m?)\.tif$", re.IGNORECASE)

_RE_CLOUD = re.compile(r"^s2__cloud_mask__(\d{8})\.tif$", re.IGNORECASE)
_RE_VISIBILITY = re.compile(r"^s2__visibility__(\d{8})\.tif$", re.IGNORECASE)
_RE_NBR = re.compile(r"^s2__nbr__(\d{8})\.tif$", re.IGNORECASE)
_RE_RGB = re.compile(r"^s2__rgb__", re.IGNORECASE)
_RE_COMPOSITE = re.compile(r"^composite_(\w+)\.tif$", re.IGNORECASE)
_RE_COMPOSITE_NEW = re.compile(r"^s2__composite__(\w+)\.tif$", re.IGNORECASE)

_RE_CARD_STAT = re.compile(r"^s1__(vv|vh)(?:__o\d{3})?__(baseline|assessment)__(\w+)\.tif$", re.IGNORECASE)
_RE_COH_STAT = re.compile(r"^s1__coh__(?:o\d{3}__)?(baseline|post_baseline)__(\w+)\.tif$", re.IGNORECASE)

_RE_ROLLING = re.compile(r"^s1__(?:coh_)?(vv|vh)(?:__o\d{3})?__roll(\d+)__(\d{8})\.tif$", re.IGNORECASE)
_RE_ZSCORE = re.compile(r"^s1__coh_(vv|vh)(?:__o\d{3})?__zscore__(\d{8})\.tif$", re.IGNORECASE)
_RE_BLOCK_STAT = re.compile(r"^s1__(vv|vh)(?:__o\d{3})?__blk[_a-z]*(\d+)__(\w+)\.tif$", re.IGNORECASE)
_RE_ROLLING_STAT = re.compile(r"^s1__(?:coh_)?(vv|vh)(?:__o\d{3})?__roll(\d+)__(\w+)__(\w+)\.tif$", re.IGNORECASE)

_RE_LANDUSE = re.compile(r"landuse", re.IGNORECASE)

# OLD RGB without s2__ prefix: {YYYYMMDD}_RGB.tif
_RE_RGB_OLD = re.compile(r"^(\d{8})_RGB\.tif$", re.IGNORECASE)
# OLD index files: {index_name}.tif in indices/ subdirs  
_RE_INDEX_FILE = re.compile(r"^(ndvi|nbr|ndbi|bsi|mndwi|ndsi|baei|ui|savi|evi)\.tif$", re.IGNORECASE)
# OLD s2__ index: s2__{index}.tif
_RE_S2_INDEX = re.compile(r"^s2__(ndvi|nbr|ndbi|bsi|mndwi|ndsi|baei|ui|savi|evi)\.tif$", re.IGNORECASE)
# lulc class
_RE_LULC = re.compile(r"^lulc__class\.tif$", re.IGNORECASE)


def classify_tif(filepath, city_name):
    fname = filepath.name
    parent = filepath.parent.name
    grandparent = filepath.parent.parent.name if filepath.parent.parent else ""
    r = {"modality": "UNKNOWN", "product_type": "UNKNOWN",
         "polarization": None, "band": None,
         "date1": None, "date2": None,
         "period_label": None, "window_size": None}

    if fname in ("building_labels.tif", "damage_mask.tif"):
        r["modality"] = "MASK"; r["product_type"] = fname.replace(".tif", ""); return r

    m = _RE_COH_NEW.match(fname)
    if m:
        r["modality"]="COH"; r["product_type"]="flat"; r["polarization"]=m.group(1).upper()
        r["date1"]=m.group(2); r["date2"]=m.group(3); return r
    m = _RE_COH_OLD.match(fname)
    if m:
        r["modality"]="COH"; r["product_type"]="flat"; r["polarization"]=m.group(2).upper()
        r["period_label"]=m.group(3).upper(); r["date1"]=m.group(4); r["date2"]=m.group(5); return r

    m = _RE_ZSCORE.match(fname)
    if m:
        r["modality"]="COH"; r["product_type"]="zscore"; r["polarization"]=m.group(1).upper()
        r["date1"]=m.group(2); return r
    m = _RE_COH_STAT.match(fname)
    if m:
        r["modality"]="COH"; r["product_type"]="temporal_stats"; r["period_label"]=m.group(1).upper(); return r

    m = _RE_ROLLING.match(fname)
    if m:
        is_coh = "coh" in fname.lower()
        r["modality"]="COH" if is_coh else "CARD"; r["product_type"]="rolling"
        r["polarization"]=m.group(1).upper(); r["window_size"]=int(m.group(2)); r["date1"]=m.group(3); return r
    m = _RE_ROLLING_STAT.match(fname)
    if m:
        is_coh = "coh" in fname.lower()
        r["modality"]="COH" if is_coh else "CARD"; r["product_type"]="rolling_stats"
        r["polarization"]=m.group(1).upper(); r["window_size"]=int(m.group(2)); return r
    m = _RE_BLOCK_STAT.match(fname)
    if m:
        r["modality"]="CARD"; r["product_type"]="block_stats"; r["polarization"]=m.group(1).upper(); return r

    m = _RE_CARD_STAT.match(fname)
    if m:
        r["modality"]="CARD"; r["product_type"]="temporal_stats"
        r["polarization"]=m.group(1).upper(); r["period_label"]=m.group(2).upper(); return r
    m = _RE_CARD_NEW.match(fname)
    if m:
        r["modality"]="CARD"; r["product_type"]="flat"; r["polarization"]=m.group(1).upper()
        r["date1"]=m.group(2); return r
    m = _RE_CARD_OLD.match(fname)
    if m:
        r["modality"]="CARD"; r["product_type"]="flat"; r["polarization"]=m.group(2).upper()
        r["date1"]=m.group(3); return r

    m = _RE_CLOUD.match(fname)
    if m:
        r["modality"]="MS"; r["product_type"]="cloud_mask"; r["date1"]=m.group(1); return r
    m = _RE_VISIBILITY.match(fname)
    if m:
        r["modality"]="MS"; r["product_type"]="fire_visibility"; r["date1"]=m.group(1); return r
    m = _RE_NBR.match(fname)
    if m:
        r["modality"]="MS"; r["product_type"]="nbr"; r["date1"]=m.group(1); return r
    if _RE_RGB.match(fname):
        r["modality"]="MS"; r["product_type"]="rgb"; return r
    m = _RE_COMPOSITE.match(fname)
    if m:
        r["modality"]="MS"; r["product_type"]="composite"; r["band"]=m.group(1); return r
    m = _RE_COMPOSITE_NEW.match(fname)
    if m:
        r["modality"]="MS"; r["product_type"]="composite"; r["band"]=m.group(1); return r

    # dNBR / RBR files
    if fname.lower().startswith("dnbr") or fname.lower().startswith("rbr"):
        r["modality"]="MS"; r["product_type"]="nbr"; return r

    # obs_count, cloud_frequency, etc
    if "obs_count" in fname.lower() or "cloud_freq" in fname.lower():
        r["modality"]="MS"; r["product_type"]="qa_layer"; return r

    m = _RE_MS_NEW.match(fname)
    if m:
        band_raw = m.group(1).lower()
        if band_raw not in ("cloud_mask", "nbr", "rgb", "visibility"):
            r["modality"]="MS"; r["product_type"]="flat"; r["band"]=m.group(1).upper()
            r["date1"]=m.group(2); return r
    m = _RE_MS_OLD.match(fname)
    if m:
        r["modality"]="MS"; r["product_type"]="flat"; r["date1"]=m.group(2)
        r["band"]=m.group(3); return r

    # change detection products: cd__dnbr__*, cd__rbr__*
    if fname.startswith('cd__'):
        r['modality'] = 'MS'
        r['product_type'] = 'change_detection'
        return r

    # visibility frequency maps: visibility_*_frequency.tif
    if fname.startswith('visibility_') and 'frequency' in fname:
        r['modality'] = 'MS'
        r['product_type'] = 'qa_layer'
        return r

    # composite band without date: s2__b02.tif (in composites/ subdir)
    _m_comp_band = re.match(r'^s2__([a-z0-9]+)\.tif$', fname, re.IGNORECASE)
    if _m_comp_band and 'composit' in str(filepath).lower():
        r['modality'] = 'MS'
        r['product_type'] = 'composite'
        r['band'] = _m_comp_band.group(1).upper()
        return r

    # s2__{band}.tif without date in non-composite dirs (indices)
    if _m_comp_band and ('indices' in str(filepath).lower() or 'landuse' in str(filepath).lower()):
        r['modality'] = 'MS'
        r['product_type'] = 'index'
        r['band'] = _m_comp_band.group(1).upper()
        return r

    # old RGB without s2__ prefix
    m = _RE_RGB_OLD.match(fname)
    if m:
        r["modality"]="MS"; r["product_type"]="rgb"; r["date1"]=m.group(1); return r

    # s2__ index files (ndvi, nbr, ndbi, bsi, etc.)
    m = _RE_S2_INDEX.match(fname)
    if m:
        r["modality"]="MS"; r["product_type"]="index"; r["band"]=m.group(1).upper(); return r

    # bare index files in indices/ subdirs
    m = _RE_INDEX_FILE.match(fname)
    if m and "indices" in str(filepath):
        r["modality"]="MS"; r["product_type"]="index"; r["band"]=m.group(1).upper(); return r

    # lulc__class.tif
    if _RE_LULC.match(fname):
        r["modality"]="MS"; r["product_type"]="landuse"; return r

    if _RE_LANDUSE.search(fname) or _RE_LANDUSE.search(parent):
        r["modality"]="MS"; r["product_type"]="landuse"; return r

    if "temporal_stats" in str(filepath):
        r["product_type"] = "temporal_stats"
    elif "coherence_baseline" in str(filepath) or "post_baseline" in str(filepath):
        r["modality"]="COH"; r["product_type"]="temporal_stats"
    elif "rolling" in parent or "rolling" in grandparent:
        r["product_type"] = "rolling"
    elif "block_stats" in parent:
        r["product_type"] = "block_stats"

    return r


# ============================================================================
# INVENTORY SCANNER
# ============================================================================

def scan_inventory(sar_card_dir, sar_coh_dir, ms_dir, stack_dir,
                   temporal_root, cities, battle_dates, cities_dir):
    rows = []

    for city in cities:
        expected_crs = get_city_expected_crs(city, cities_dir)
        base = {"city": city, "utm_epsg": expected_crs["utm_epsg"]}

        scan_dirs = [
            Path(sar_card_dir) / city,
            Path(sar_coh_dir) / city,
            Path(ms_dir) / city,
            Path(temporal_root) / city,
        ]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for tif in sorted(scan_dir.rglob("*.tif")):
                info = classify_tif(tif, city)
                rows.append({
                    **base,
                    "filepath": str(tif), "filename": tif.name,
                    "file_size_bytes": tif.stat().st_size,
                    "parent_dir": tif.parent.name,
                    **info,
                })

        stack_city = Path(stack_dir) / city
        if stack_city.exists():
            for mask_name in ("building_labels.tif", "damage_mask.tif"):
                mask_path = stack_city / mask_name
                if mask_path.exists():
                    info = classify_tif(mask_path, city)
                    rows.append({
                        **base,
                        "filepath": str(mask_path), "filename": mask_name,
                        "file_size_bytes": mask_path.stat().st_size,
                        "parent_dir": stack_city.name,
                        **info,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["file_size_mb"] = df["file_size_bytes"] / (1024 * 1024)
    return df


# ============================================================================
# PER-RASTER QA
# ============================================================================

def qa_read_raster_stats(filepath, max_pixels=500000):
    result = {
        "crs_epsg": None, "width": None, "height": None,
        "res_x": None, "res_y": None,
        "origin_x": None, "origin_y": None,
        "dtype": None, "nodata": None, "n_bands": None,
        "bounds_left": None, "bounds_bottom": None,
        "bounds_right": None, "bounds_top": None,
        "val_min": None, "val_max": None, "val_mean": None, "val_std": None,
        "val_p01": None, "val_p99": None,
        "pct_nodata": None, "pct_zero": None, "pct_nodata_leak": None,
        "read_error": None,
    }
    try:
        with rasterio.open(str(filepath)) as ds:
            result["crs_epsg"] = ds.crs.to_epsg() if ds.crs else None
            result["width"] = ds.width
            result["height"] = ds.height
            result["res_x"] = abs(ds.transform[0])
            result["res_y"] = abs(ds.transform[4])
            result["origin_x"] = ds.transform[2]
            result["origin_y"] = ds.transform[5]
            result["dtype"] = str(ds.dtypes[0])
            result["nodata"] = ds.nodata
            result["n_bands"] = ds.count
            result["bounds_left"] = ds.bounds.left
            result["bounds_bottom"] = ds.bounds.bottom
            result["bounds_right"] = ds.bounds.right
            result["bounds_top"] = ds.bounds.top

            total_px = ds.width * ds.height
            if total_px > max_pixels:
                step = max(1, int(np.sqrt(total_px / max_pixels)))
                data = ds.read(1, out_shape=(ds.height // step, ds.width // step))
            else:
                data = ds.read(1)

            data = data.astype(np.float64)
            if ds.nodata is not None and np.isfinite(ds.nodata):
                mask = np.isfinite(data) & ~np.isclose(data, ds.nodata)
            else:
                mask = np.isfinite(data)

            total = data.size
            n_valid = mask.sum()
            n_nodata = total - n_valid

            result["pct_nodata"] = 100.0 * n_nodata / total if total > 0 else 0
            result["pct_zero"] = 100.0 * np.sum(data[mask] == 0) / total if total > 0 else 0
            result["pct_nodata_leak"] = 100.0 * np.sum(data[mask] <= -99.0) / total if total > 0 else 0

            if n_valid > 0:
                valid = data[mask]
                result["val_min"] = float(np.nanmin(valid))
                result["val_max"] = float(np.nanmax(valid))
                result["val_mean"] = float(np.nanmean(valid))
                result["val_std"] = float(np.nanstd(valid))
                result["val_p01"] = float(np.nanpercentile(valid, 1))
                result["val_p99"] = float(np.nanpercentile(valid, 99))
    except Exception as e:
        result["read_error"] = str(e)

    return result


# ============================================================================
# MODALITY-SPECIFIC CHECKS
# ============================================================================

def qa_check_card_row(stats, thresholds):
    t = thresholds
    findings = []
    if stats["read_error"]:
        findings.append(("FAIL", "READ_ERROR", stats["read_error"])); return findings
    if stats["pct_nodata_leak"] is not None and stats["pct_nodata_leak"] > 0.1:
        findings.append(("FAIL", "NODATA_LEAK",
                         f"{stats['pct_nodata_leak']:.1f}% pixels <=-99dB (min={stats['val_min']:.0f})"))
    if stats["val_p01"] is not None and stats["val_p01"] < t["card_db_min"]:
        findings.append(("WARN", "BELOW_RANGE", f"p01={stats['val_p01']:.1f} < {t['card_db_min']}"))
    if stats["val_p99"] is not None and stats["val_p99"] > t["card_db_max"]:
        findings.append(("WARN", "ABOVE_RANGE", f"p99={stats['val_p99']:.1f} > {t['card_db_max']}"))
    if stats["val_mean"] is not None and stats["val_mean"] < t["card_db_nesz"]:
        findings.append(("WARN", "BELOW_NESZ", f"mean={stats['val_mean']:.1f} < NESZ {t['card_db_nesz']}"))
    if stats["pct_nodata"] is not None and stats["pct_nodata"] > 90:
        findings.append(("FAIL", "MOSTLY_NODATA", f"nodata={stats['pct_nodata']:.1f}%"))
    if stats["val_std"] is not None and stats["val_std"] < 0.01:
        findings.append(("WARN", "CONSTANT_RASTER", f"std={stats['val_std']:.4f}"))
    if not findings:
        findings.append(("PASS", "OK", ""))
    return findings


def qa_check_coh_row(stats, thresholds):
    t = thresholds
    findings = []
    if stats["read_error"]:
        findings.append(("FAIL", "READ_ERROR", stats["read_error"])); return findings
    if stats["val_min"] is not None and stats["val_min"] < t["coh_min"] - 0.001:
        findings.append(("FAIL", "BELOW_ZERO", f"min={stats['val_min']:.4f}"))
    if stats["val_max"] is not None and stats["val_max"] > t["coh_max"] + 0.001:
        findings.append(("FAIL", "ABOVE_ONE", f"max={stats['val_max']:.4f}"))
    if stats["pct_zero"] is not None and stats["pct_zero"] > t["coh_all_zero_pct"]:
        findings.append(("FAIL", "MOSTLY_ZERO", f"zero={stats['pct_zero']:.1f}%"))
    if stats["val_std"] is not None and stats["val_std"] < 0.01:
        findings.append(("WARN", "CONSTANT_COH", f"std={stats['val_std']:.4f}"))
    if stats["val_mean"] is not None and stats["val_mean"] > t["coh_suspect_high"]:
        findings.append(("WARN", "SUSPECT_HIGH", f"mean={stats['val_mean']:.3f}"))
    if stats["pct_nodata"] is not None and stats["pct_nodata"] > 90:
        findings.append(("FAIL", "MOSTLY_NODATA", f"nodata={stats['pct_nodata']:.1f}%"))
    if not findings:
        findings.append(("PASS", "OK", ""))
    return findings


def qa_check_ms_row(stats, thresholds):
    t = thresholds
    findings = []
    if stats["read_error"]:
        findings.append(("FAIL", "READ_ERROR", stats["read_error"])); return findings
    if stats["val_min"] is not None and stats["val_min"] < t["ms_dn_min"]:
        findings.append(("WARN", "NEGATIVE_DN", f"min={stats['val_min']:.0f}"))
    if stats["val_max"] is not None and stats["val_max"] > t["ms_dn_max"]:
        findings.append(("FAIL", "SATURATED", f"max={stats['val_max']:.0f}"))
    if stats["pct_nodata"] is not None and stats["pct_nodata"] > 90:
        findings.append(("FAIL", "MOSTLY_NODATA", f"nodata={stats['pct_nodata']:.1f}%"))
    if stats["val_std"] is not None and stats["val_std"] < 0.1:
        findings.append(("WARN", "CONSTANT_RASTER", f"std={stats['val_std']:.4f}"))
    if not findings:
        findings.append(("PASS", "OK", ""))
    return findings


def qa_check_mask_row(stats, thresholds, mask_type):
    findings = []
    if stats["read_error"]:
        findings.append(("FAIL", "READ_ERROR", stats["read_error"])); return findings
    if mask_type == "damage_mask":
        valid = set(thresholds["damage_mask_valid"])
        if stats["val_min"] is not None and stats["val_min"] < min(valid):
            findings.append(("WARN", "UNEXPECTED_VALUE", f"min={stats['val_min']}"))
        if stats["val_max"] is not None and stats["val_max"] > max(valid):
            findings.append(("WARN", "UNEXPECTED_VALUE", f"max={stats['val_max']}"))
    elif mask_type == "building_labels":
        if stats["val_min"] is not None and stats["val_min"] < 0:
            findings.append(("FAIL", "NEGATIVE_LABEL", f"min={stats['val_min']}"))
    if stats["pct_nodata"] is not None and stats["pct_nodata"] > 95:
        findings.append(("FAIL", "MOSTLY_NODATA", f"nodata={stats['pct_nodata']:.1f}%"))
    if not findings:
        findings.append(("PASS", "OK", ""))
    return findings


def qa_check_stats_row(stats, modality, thresholds):
    findings = []
    if stats["read_error"]:
        findings.append(("FAIL", "READ_ERROR", stats["read_error"])); return findings
    if modality == "CARD":
        if stats["pct_nodata_leak"] is not None and stats["pct_nodata_leak"] > 0.1:
            findings.append(("FAIL", "STAT_NODATA_LEAK", f"{stats['pct_nodata_leak']:.1f}% pixels <=-99"))
    if modality == "COH":
        if stats["val_min"] is not None and stats["val_min"] < -0.001:
            findings.append(("FAIL", "COH_STAT_NEGATIVE", f"min={stats['val_min']:.4f}"))
        if stats["val_max"] is not None and stats["val_max"] > 1.001:
            findings.append(("FAIL", "COH_STAT_ABOVE_1", f"max={stats['val_max']:.4f}"))
    if stats["pct_nodata"] is not None and stats["pct_nodata"] > 90:
        findings.append(("FAIL", "MOSTLY_NODATA", f"nodata={stats['pct_nodata']:.1f}%"))
    if not findings:
        findings.append(("PASS", "OK", ""))
    return findings


# ============================================================================
# CRS + ALIGNMENT (bbox-based)
# ============================================================================

def qa_check_crs(inventory_df, cities_dir):
    findings = []
    for city, grp in inventory_df.groupby("city"):
        sampled = grp.groupby("modality").apply(
            lambda g: g.sample(min(5, len(g)), random_state=42)
        ).reset_index(drop=True)
        invalid_found = False
        city_crs_set = set()
        for _, row in sampled.iterrows():
            try:
                with rasterio.open(row["filepath"]) as ds:
                    epsg = ds.crs.to_epsg() if ds.crs else None
                    city_crs_set.add(epsg)
                    ok, actual = crs_matches_city(ds.crs, city, cities_dir)
                    if not ok:
                        invalid_found = True
                        findings.append({
                            "city": city, "check": "CRS_INVALID",
                            "status": "FAIL",
                            "detail": f"{row['filename']}: EPSG:{actual} not valid",
                            "filepath": row["filepath"],
                        })
            except Exception as e:
                findings.append({
                    "city": city, "check": "CRS_READ_ERROR",
                    "status": "FAIL", "detail": str(e),
                    "filepath": row["filepath"],
                })
        city_crs_set.discard(None)
        if not invalid_found:
            findings.append({
                "city": city, "check": "CRS_OK",
                "status": "PASS", "detail": f"All CRS valid: {city_crs_set}",
                "filepath": "",
            })
    return findings


def qa_check_alignment(inventory_df, cities_dir, thresholds):
    findings = []
    tol = thresholds["max_bbox_diff_deg"]

    for city, grp in inventory_df.groupby("city"):
        aoi_bounds = get_city_aoi_bbox_wgs84(city, cities_dir)
        if aoi_bounds is None:
            findings.append({
                "city": city, "check": "ALIGN_NO_AOI",
                "status": "WARN", "detail": "No AOI.geojson aoi_bbox found",
            })
            continue

        aoi_left, aoi_bottom, aoi_right, aoi_top = aoi_bounds

        for mod in ["CARD", "COH", "MS"]:
            flat = grp[(grp["modality"] == mod) & (grp["product_type"] == "flat")]
            if flat.empty:
                continue
            sample_path = flat.iloc[0]["filepath"]
            try:
                with rasterio.open(sample_path) as ds:
                    b = ds.bounds
                    rl, rb, rr, rt = b.left, b.bottom, b.right, b.top

                    epsg = ds.crs.to_epsg() if ds.crs else None
                    if epsg and epsg != 4326:
                        import pyproj
                        transformer = pyproj.Transformer.from_crs(
                            f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
                        rl, rb = transformer.transform(b.left, b.bottom)
                        rr, rt = transformer.transform(b.right, b.top)

                    if rl > aoi_left + tol or rb > aoi_bottom + tol:
                        findings.append({
                            "city": city, "check": "ALIGN_RASTER_SMALLER",
                            "status": "WARN",
                            "detail": f"{mod}: raster misses AOI left/bottom",
                        })
                    if rr < aoi_right - tol or rt < aoi_top - tol:
                        findings.append({
                            "city": city, "check": "ALIGN_RASTER_SMALLER",
                            "status": "WARN",
                            "detail": f"{mod}: raster misses AOI right/top",
                        })
            except Exception:
                pass

    return findings


# ============================================================================
# ORBIT CONTAMINATION
# ============================================================================

def qa_detect_orbit_contamination(inventory_df, card_tracker_file=None):
    findings = []
    card_flat = inventory_df[
        (inventory_df["modality"] == "CARD") & (inventory_df["product_type"] == "flat")]

    for city, grp in card_flat.groupby("city"):
        extents = {}
        sample = grp.sample(min(20, len(grp)), random_state=42)
        for _, row in sample.iterrows():
            try:
                with rasterio.open(row["filepath"]) as ds:
                    bounds = ds.bounds
                    key = (round(bounds.left, 3), round(bounds.bottom, 3),
                           round(bounds.right, 3), round(bounds.top, 3))
                    extents.setdefault(key, []).append(row["filename"])
            except Exception:
                pass

        if len(extents) > 1:
            findings.append({
                "city": city, "check": "ORBIT_MULTI_EXTENT",
                "status": "FAIL",
                "detail": f"{len(extents)} distinct extents (mixed orbits)",
            })
        else:
            findings.append({
                "city": city, "check": "ORBIT_CARD_OK",
                "status": "PASS", "detail": "Single extent/orbit",
            })

    return findings


def qa_detect_zigzag(inventory_df, sar_card_dir, cities):
    findings = []
    card_flat = inventory_df[
        (inventory_df["modality"] == "CARD") &
        (inventory_df["product_type"] == "flat") &
        (inventory_df["polarization"] == "VV")].copy()

    for city in cities:
        city_files = card_flat[card_flat["city"] == city].sort_values("date1")
        if len(city_files) < 6:
            continue
        means = []
        for _, row in city_files.iterrows():
            try:
                with rasterio.open(row["filepath"]) as ds:
                    cy, cx = ds.height // 2, ds.width // 2
                    win = rasterio.windows.Window(max(0, cx - 5), max(0, cy - 5), 10, 10)
                    data = ds.read(1, window=win)
                    valid = data[(np.isfinite(data)) & (data > -99)]
                    if len(valid) > 0:
                        means.append(float(np.nanmean(valid)))
                    else:
                        means.append(np.nan)
            except Exception:
                means.append(np.nan)

        means = np.array(means)
        valid_mask = np.isfinite(means)
        if valid_mask.sum() < 6:
            continue
        valid_means = means[valid_mask]
        diffs = np.diff(valid_means)
        if len(diffs) < 5:
            continue
        signs = np.sign(diffs)
        sign_changes = np.diff(signs)
        max_alt = 0
        cur = 0
        for sc in sign_changes:
            if sc != 0:
                cur += 1
                max_alt = max(max_alt, cur)
            else:
                cur = 0

        if max_alt >= 4:
            findings.append({
                "city": city, "check": "ZIGZAG_CARD",
                "status": "FAIL",
                "detail": f"Alternating length={max_alt} (orbit mixing suspected)",
            })
        else:
            findings.append({
                "city": city, "check": "ZIGZAG_CARD",
                "status": "PASS", "detail": f"Max alternation={max_alt}",
            })
    return findings


# ============================================================================
# TEMPORAL GAP ANALYSIS
# ============================================================================

def qa_temporal_gaps(inventory_df, battle_dates, thresholds):
    findings = []
    t = thresholds

    for modality, revisit_key in [("CARD", "s1_revisit_days"), ("COH", "s1_revisit_days"), ("MS", "s2_revisit_days")]:
        revisit = t[revisit_key]
        flat = inventory_df[
            (inventory_df["modality"] == modality) & (inventory_df["product_type"] == "flat")]

        for city, grp in flat.groupby("city"):
            dates = []
            for d in grp["date1"].dropna().unique():
                try:
                    dates.append(datetime.strptime(str(d), "%Y%m%d"))
                except Exception:
                    pass
            if len(dates) < 2:
                findings.append({
                    "city": city, "modality": modality,
                    "check": "TEMPORAL_FEW_SCENES", "status": "WARN",
                    "detail": f"Only {len(dates)} dates",
                })
                continue

            dates = sorted(dates)
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            max_gap = max(gaps)
            median_gap = float(np.median(gaps))
            n_warn = sum(1 for g in gaps if g > revisit * t["gap_warn_factor"])
            n_fail = sum(1 for g in gaps if g > revisit * t["gap_fail_factor"])

            bd = battle_dates.get(city, {})
            bs = bd.get("battle_start")
            be = bd.get("battle_stop")
            n_pre = 0
            n_post = 0
            if bs:
                try:
                    bs_dt = datetime.strptime(bs, "%Y-%m-%d")
                    n_pre = sum(1 for d in dates if d < bs_dt)
                except Exception:
                    pass
            if be and str(be) not in ("", "None", "ongoing", "NaT"):
                try:
                    be_dt = datetime.strptime(be, "%Y-%m-%d")
                    n_post = sum(1 for d in dates if d > be_dt)
                except Exception:
                    pass

            status = "PASS"
            if n_fail > 0:
                status = "FAIL"
            elif n_warn > 0:
                status = "WARN"

            findings.append({
                "city": city, "modality": modality,
                "check": "TEMPORAL_GAPS", "status": status,
                "detail": f"n={len(dates)} max_gap={max_gap}d median={median_gap:.0f}d warn={n_warn} fail={n_fail} pre={n_pre} post={n_post}",
            })
    return findings


# ============================================================================
# TEMPORAL PROFILE EXTRACTION
# ============================================================================

def extract_temporal_profile(inventory_df, city, modality, polarization="VV",
                             product_type="flat", roi_radius_px=20):
    sub = inventory_df[
        (inventory_df["city"] == city) &
        (inventory_df["modality"] == modality) &
        (inventory_df["product_type"] == product_type)]
    if polarization:
        sub = sub[sub["polarization"] == polarization]
    sub = sub.sort_values("date1")

    results = []
    for _, row in sub.iterrows():
        date_str = row["date1"]
        if date_str is None:
            continue
        try:
            with rasterio.open(row["filepath"]) as ds:
                cy, cx = ds.height // 2, ds.width // 2
                r = roi_radius_px
                win = rasterio.windows.Window(max(0, cx - r), max(0, cy - r), 2*r, 2*r)
                data = ds.read(1, window=win).astype(np.float64)
                if ds.nodata is not None:
                    data[np.isclose(data, ds.nodata)] = np.nan
                val = float(np.nanmean(data))
                results.append((str(date_str), val))
        except Exception:
            pass
    return results



# ============================================================================
# VISUAL: EXTENT DIAGNOSTIC
# ============================================================================

def qa_plot_extents(inventory_df, cities, cities_dir, sar_card_dir,
                    sar_coh_dir, ms_dir, output_dir=None):
    """Plot actual raster extents vs AOI bbox per city per modality."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    import rasterio
    import pyproj

    n_cities = len(cities)
    fig, axes = plt.subplots(n_cities, 1, figsize=(14, 6 * n_cities), squeeze=False)

    for idx, city in enumerate(cities):
        ax = axes[idx, 0]
        aoi_bounds = get_city_aoi_bbox_wgs84(city, cities_dir)

        if aoi_bounds:
            aoi_l, aoi_b, aoi_r, aoi_t = aoi_bounds
            aoi_rect = Rectangle((aoi_l, aoi_b), aoi_r - aoi_l, aoi_t - aoi_b,
                                 linewidth=3, edgecolor='black', facecolor='none',
                                 linestyle='-', label='AOI bbox', zorder=10)
            ax.add_patch(aoi_rect)

        colors = {"CARD": "steelblue", "COH": "forestgreen", "MS": "darkorange"}
        legend_entries = {}
        all_coords = []
        if aoi_bounds:
            all_coords.extend([(aoi_l, aoi_b), (aoi_r, aoi_t)])

        city_inv = inventory_df[
            (inventory_df["city"] == city) &
            (inventory_df["product_type"] == "flat")
        ]

        for mod, color in colors.items():
            mod_files = city_inv[city_inv["modality"] == mod]
            if mod_files.empty:
                continue
            sample = mod_files.sample(min(15, len(mod_files)), random_state=42)
            extent_groups = {}
            for _, row in sample.iterrows():
                try:
                    with rasterio.open(row["filepath"]) as ds:
                        b = ds.bounds
                        epsg = ds.crs.to_epsg() if ds.crs else None
                        rl, rb, rr, rt = b.left, b.bottom, b.right, b.top
                        if epsg and epsg != 4326:
                            transformer = pyproj.Transformer.from_crs(
                                f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
                            rl, rb = transformer.transform(b.left, b.bottom)
                            rr, rt = transformer.transform(b.right, b.top)
                        key = (round(rl, 3), round(rb, 3), round(rr, 3), round(rt, 3))
                        extent_groups.setdefault(key, []).append(row["filename"])
                except Exception:
                    pass

            n_ext = len(extent_groups)
            for gi, ((el, eb, er, et), fnames) in enumerate(extent_groups.items()):
                alpha = 0.25 if gi == 0 else 0.12
                ls = '-' if gi == 0 else '--'
                lw = 2.0 if gi == 0 else 1.5
                rect = Rectangle((el, eb), er - el, et - eb,
                                 linewidth=lw, edgecolor=color, facecolor=color,
                                 alpha=alpha, linestyle=ls)
                ax.add_patch(rect)
                all_coords.extend([(el, eb), (er, et)])
                # annotate extent group
                cx = (el + er) / 2
                cy = et + 0.002 * (gi + 1)
                ax.text(cx, cy, f"{mod} ext{gi+1} ({len(fnames)}f)",
                        fontsize=7, ha='center', color=color, alpha=0.8)

            legend_entries[mod] = mpatches.Patch(
                color=color, alpha=0.4,
                label=f"{mod} ({n_ext} extent{'s' if n_ext > 1 else ''}, {len(mod_files)} files)")

        if all_coords:
            xs = [c[0] for c in all_coords]
            ys = [c[1] for c in all_coords]
            pad_x = (max(xs) - min(xs)) * 0.08 or 0.01
            pad_y = (max(ys) - min(ys)) * 0.08 or 0.01
            ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
            ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

        ax.set_aspect('equal')
        ax.set_title(f"{city} -- Raster extents vs AOI bbox (WGS84)", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        handles = [mpatches.Patch(facecolor='none', edgecolor='black', linewidth=2, label='AOI bbox')]
        handles.extend(legend_entries.values())
        ax.legend(handles=handles, loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        from pathlib import Path
        out_path = Path(output_dir) / "qa_extent_map.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show()

# ============================================================================
# SQLITE
# ============================================================================

QA_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    city TEXT NOT NULL, modality TEXT, product_type TEXT,
    check_name TEXT NOT NULL, status TEXT NOT NULL,
    detail TEXT, filepath TEXT,
    val_min REAL, val_max REAL, val_mean REAL, val_std REAL,
    pct_nodata REAL, pct_nodata_leak REAL,
    crs_epsg INTEGER, width_px INTEGER, height_px INTEGER
);"""

QA_SUMMARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_run_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    n_cities INTEGER, n_files_scanned INTEGER,
    n_pass INTEGER, n_warn INTEGER, n_fail INTEGER,
    tier_selection TEXT, duration_seconds REAL
);"""


def ensure_qa_schema(catalog_db):
    conn = sqlite3.connect(str(catalog_db))
    # check if existing table has all required columns
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(qa_results)").fetchall()]
        if cols and "pct_nodata_leak" not in cols:
            conn.execute("DROP TABLE IF EXISTS qa_results")
            conn.execute("DROP TABLE IF EXISTS qa_run_summary")
            print("  Dropped old qa_results/qa_run_summary tables (schema changed)")
    except Exception:
        pass
    conn.execute(QA_TABLE_SCHEMA)
    conn.execute(QA_SUMMARY_SCHEMA)
    conn.commit()
    conn.close()


def register_qa_results(catalog_db, results_df, run_timestamp):
    ensure_qa_schema(catalog_db)
    conn = sqlite3.connect(str(catalog_db))
    for _, row in results_df.iterrows():
        conn.execute(
            """INSERT INTO qa_results
               (run_timestamp, city, modality, product_type, check_name, status,
                detail, filepath, val_min, val_max, val_mean, val_std,
                pct_nodata, pct_nodata_leak, crs_epsg, width_px, height_px)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_timestamp, row.get("city",""), row.get("modality",""),
             row.get("product_type",""), row.get("check_name", row.get("check","")),
             row.get("status",""), row.get("detail",""), row.get("filepath",""),
             row.get("val_min"), row.get("val_max"), row.get("val_mean"), row.get("val_std"),
             row.get("pct_nodata"), row.get("pct_nodata_leak"),
             row.get("crs_epsg"), row.get("width_px"), row.get("height_px")))
    conn.commit()
    conn.close()


def register_qa_summary(catalog_db, run_timestamp, n_cities, n_files,
                        n_pass, n_warn, n_fail, tier_selection, duration_s):
    ensure_qa_schema(catalog_db)
    conn = sqlite3.connect(str(catalog_db))
    conn.execute(
        """INSERT INTO qa_run_summary
           (run_timestamp, n_cities, n_files_scanned, n_pass, n_warn, n_fail,
            tier_selection, duration_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_timestamp, n_cities, n_files, n_pass, n_warn, n_fail,
         str(tier_selection), duration_s))
    conn.commit()
    conn.close()
