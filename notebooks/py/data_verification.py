# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
data_verification.py
Data verification: scans all directories, checks completeness, disk space.
Extracted from Cell 4B. Updated for flat satellite directory structure.

Notebook usage:
    from data_verification import run as run_data_verification
    VERIFICATION_STATUS = run_data_verification(
        satellite_dirs=dict(
            SAR_COH_DIR=SAR_COH_DIR,
            SAR_CARD_DIR=SAR_CARD_DIR,
            MS_DIR=MS_DIR,
            MS_METADATA_DIR=MS_METADATA_DIR,
            SAR_METADATA_DIR=SAR_METADATA_DIR,
            LANDUSE_DIR=LANDUSE_DIR,
            TRANSITION_DIR=TRANSITION_DIR,
        ),
        reference_dirs=dict(
            CITIES_DIR=CITIES_DIR,
            UKR_BOUNDARIES=UKR_BOUNDARIES,
            OSM_2022=OSM_2022,
            OSM_2025=OSM_2025,
            OSM_OUTPUT_DIR=OSM_OUTPUT_DIR,
            UNOSAT_DIR=UNOSAT_DIR,
            UNOSAT_RAW_DIR=UNOSAT_RAW_DIR,
            XBD_ROOT=XBD_ROOT,
        ),
        ml_dirs=dict(
            CHECKPOINTS_DIR=CHECKPOINTS_DIR,
            PRETRAINED_DIR=PRETRAINED_DIR,
            TRAINED_DIR=TRAINED_DIR,
        ),
        result_dirs=dict(
            PREDICTIONS_DIR=PREDICTIONS_DIR,
            VALIDATION_DIR=VALIDATION_DIR,
            VISUALIZATIONS_DIR=VISUALIZATIONS_DIR,
            OUTPUTS_DIR=OUTPUTS_DIR,
            LOGS_DIR=LOGS_DIR,
        ),
        local_dirs=dict(
            RAW_SLC_ZIP=RAW_SLC_ZIP,
            SAR_SLC_ORBIT_DIR=SAR_SLC_ORBIT_DIR,
            DEM_DIR=DEM_DIR,
        ),
        snap_dirs=dict(
            SNAP_WORKDIR=SNAP_WORKDIR,
            TEMP_DOWNLOAD_DIR=TEMP_DOWNLOAD_DIR,
            CROSSBATTLE_STASH_DIR=CROSSBATTLE_STASH_DIR,
            SNAP_GRAPH_DIR=SNAP_GRAPH_DIR,
        ),
        tracker_files=dict(
            INSAR_TRACKER_FILE=INSAR_TRACKER_FILE,
            MS_TRACKING_FILE=MS_TRACKING_FILE,
            CARD_TRACKER_FILE=CARD_TRACKER_FILE,
            PROGRESS_FILE=PROGRESS_FILE,
            CITIES_PKL_FILE=CITIES_PKL_FILE,
            MS_CITIES_PKL_FILE=MS_CITIES_PKL_FILE,
        ),
        gpt_path=GPT_PATH,
        content_local=CONTENT_LOCAL,
        get_disk_space_fn=get_disk_space,
    )
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime


def _scan_dir(path):
    """Count files and total size under path."""
    path = Path(path)
    if not path.exists():
        return None, 0, 0
    all_files = [f for f in path.rglob('*') if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in all_files) if all_files else 0
    return all_files, len(all_files), total_bytes


def _print_dir_status(name, path):
    path = Path(path)
    if not path.exists():
        print(f"  [ MISSING ] {name:35s} {path}")
        return False
    all_files, count, total_bytes = _scan_dir(path)
    if count == 0:
        print(f"  [ EMPTY  ] {name:35s} {path}")
        return True
    total_gb = total_bytes / (1024**3)
    total_mb = total_bytes / (1024**2)
    size_str = f"{total_gb:.1f} GB" if total_gb >= 1 else f"{total_mb:.1f} MB"
    print(f"  [   OK   ] {name:35s} {count:6d} files, {size_str}")
    return True


def run(satellite_dirs, reference_dirs, ml_dirs, result_dirs,
        local_dirs, snap_dirs, tracker_files,
        gpt_path, content_local, get_disk_space_fn=None):
    """
    Args:
        satellite_dirs:   dict - SAR_COH_DIR, SAR_CARD_DIR, MS_DIR, MS_METADATA_DIR, SAR_METADATA_DIR, LANDUSE_DIR, TRANSITION_DIR
        reference_dirs:   dict - CITIES_DIR, UKR_BOUNDARIES, OSM_2022, OSM_2025, OSM_OUTPUT_DIR, UNOSAT_DIR, UNOSAT_RAW_DIR, XBD_ROOT
        ml_dirs:          dict - CHECKPOINTS_DIR, PRETRAINED_DIR, TRAINED_DIR
        result_dirs:      dict - PREDICTIONS_DIR, VALIDATION_DIR, VISUALIZATIONS_DIR, OUTPUTS_DIR, LOGS_DIR
        local_dirs:       dict - RAW_SLC_ZIP, SAR_SLC_ORBIT_DIR, DEM_DIR
        snap_dirs:        dict - SNAP_WORKDIR, TEMP_DOWNLOAD_DIR, CROSSBATTLE_STASH_DIR, SNAP_GRAPH_DIR
        tracker_files:    dict - INSAR_TRACKER_FILE, MS_TRACKING_FILE, CARD_TRACKER_FILE, PROGRESS_FILE, CITIES_PKL_FILE, MS_CITIES_PKL_FILE
        gpt_path:         str/Path
        content_local:    Path
        get_disk_space_fn: callable or None

    Returns:
        VERIFICATION_STATUS dict
    """
    # resolve all paths
    for d in [satellite_dirs, reference_dirs, ml_dirs, result_dirs, local_dirs, snap_dirs, tracker_files]:
        for k, v in d.items():
            d[k] = Path(v)
    gpt_path = Path(gpt_path)
    content_local = Path(content_local)

    print("=" * 70)
    print("DATA VERIFICATION")
    print("=" * 70)

    # =========================================================================
    # DIRECTORY INVENTORY
    # =========================================================================

    data_locations = {
        # Core satellite data (flat structure)
        "SAR COH (coherence products)":  satellite_dirs['SAR_COH_DIR'],
        "SAR CARD-BS (calibrated)":      satellite_dirs['SAR_CARD_DIR'],
        "Multispectral (Sentinel-2)":    satellite_dirs['MS_DIR'],
        "MS Metadata":                   satellite_dirs['MS_METADATA_DIR'],
        "SAR Metadata":                  satellite_dirs['SAR_METADATA_DIR'],
        "Landuse Classification":        satellite_dirs['LANDUSE_DIR'],
        "Transition Matrix":             satellite_dirs['TRANSITION_DIR'],

        # Reference data
        "City Boundaries":               reference_dirs['CITIES_DIR'],
        "Ukraine Admin Boundaries":      reference_dirs['UKR_BOUNDARIES'],
        "OSM Feb 2022 (pre-conflict)":   reference_dirs['OSM_2022'],
        "OSM Oct 2025 (current)":        reference_dirs['OSM_2025'],
        "OSM Buildings (extracted)":     reference_dirs['OSM_OUTPUT_DIR'],
        "UNOSAT Damage Assessments":     reference_dirs['UNOSAT_DIR'],
        "xBD Dataset":                   reference_dirs['XBD_ROOT'],

        # ML
        "Model Checkpoints":             ml_dirs['CHECKPOINTS_DIR'],
        "Pretrained Models":             ml_dirs['PRETRAINED_DIR'],
        "Trained Models":                ml_dirs['TRAINED_DIR'],

        # Results
        "Predictions":                   result_dirs['PREDICTIONS_DIR'],
        "Validation Results":            result_dirs['VALIDATION_DIR'],
        "Visualizations":                result_dirs['VISUALIZATIONS_DIR'],

        # Outputs & tracking
        "Outputs":                       result_dirs['OUTPUTS_DIR'],
        "Logs":                          result_dirs['LOGS_DIR'],
    }

    for name, path in data_locations.items():
        _print_dir_status(name, path)

    # =========================================================================
    # LOCAL HDD STORAGE
    # =========================================================================

    print(f"\nLocal HDD storage:")
    print("-" * 70)

    local_locations = {
        "RAW SLC ZIPs (prestage)":  local_dirs['RAW_SLC_ZIP'],
        "CARD ZIPs":                local_dirs.get('CARD_ZIP_DIR', content_local / 'data' / 'card_zip'),
        "MS ZIPs":                  local_dirs.get('MS_ZIP_DIR', content_local / 'data' / 'ms_zip'),
        "Orbit files (POEORB)":     local_dirs['SAR_SLC_ORBIT_DIR'],
        "DEM tiles":                local_dirs['DEM_DIR'],
    }

    for name, path in local_locations.items():
        _print_dir_status(name, path)

    # =========================================================================
    # SNAP WORKDIR
    # =========================================================================

    print(f"\nSNAP work directories:")
    print("-" * 70)

    snap_locations = {
        "SNAP workdir":          snap_dirs['SNAP_WORKDIR'],
        "Temp download dir":     snap_dirs['TEMP_DOWNLOAD_DIR'],
        "Crossbattle stash":     snap_dirs['CROSSBATTLE_STASH_DIR'],
        "SNAP graph XMLs":       snap_dirs['SNAP_GRAPH_DIR'],
    }

    for name, path in snap_locations.items():
        _print_dir_status(name, path)

    if gpt_path.exists():
        print(f"  [   OK   ] {'SNAP GPT binary':35s} {gpt_path}")
    else:
        print(f"  [ MISSING ] {'SNAP GPT binary':35s} {gpt_path}")

    # =========================================================================
    # TRACKING FILES
    # =========================================================================

    print(f"\nTracking files:")
    print("-" * 70)

    tracking_display = {
        "InSAR tracker":        tracker_files['INSAR_TRACKER_FILE'],
        "MS download tracker":  tracker_files['MS_TRACKING_FILE'],
        "CARD download tracker": tracker_files['CARD_TRACKER_FILE'],
        "Scene discovery":      tracker_files['PROGRESS_FILE'],
        "SAR cities pkl":       tracker_files['CITIES_PKL_FILE'],
        "MS cities pkl":        tracker_files['MS_CITIES_PKL_FILE'],
    }

    for name, fpath in tracking_display.items():
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  [   OK   ] {name:30s} ({size_kb:.1f} KB)")
        else:
            print(f"  [ MISSING ] {name:30s}")

    # =========================================================================
    # CRITICAL DATA CHECK
    # =========================================================================

    print(f"\nCritical data check:")
    print("-" * 70)

    critical_checks = {
        "City boundaries (.geojson)":    (reference_dirs['CITIES_DIR'],      ["*.geojson"]),
        "SAR COH products (.tif)":       (satellite_dirs['SAR_COH_DIR'],     ["*.tif"]),
        "SAR CARD products (.tif)":      (satellite_dirs['SAR_CARD_DIR'],    ["*.tif"]),
        "Multispectral scenes (.tif)":   (satellite_dirs['MS_DIR'],          ["*.tif", "*.jp2"]),
        "SAR Metadata (.json)":          (satellite_dirs['SAR_METADATA_DIR'],["*.json"]),
        "Landuse classification (.tif)": (satellite_dirs['LANDUSE_DIR'],     ["*.tif"]),
        "OSM 2022 data":                 (reference_dirs['OSM_2022'],        ["*.pbf", "*.shp", "*.geojson"]),
        "OSM 2025 data":                 (reference_dirs['OSM_2025'],        ["*.pbf", "*.shp", "*.geojson"]),
        "Admin boundaries (.shp)":       (reference_dirs['UKR_BOUNDARIES'],  ["*.shp"]),
        "UNOSAT raw shapefiles":         (reference_dirs['UNOSAT_RAW_DIR'],  ["*.shp", "*.geojson", "*.zip"]),
        "Orbit files (.EOF)":            (local_dirs['SAR_SLC_ORBIT_DIR'],   ["*.EOF", "*.EOF.zip"]),
    }

    all_ok = True
    critical_missing = []

    for check_name, (path, patterns) in critical_checks.items():
        path = Path(path)
        if not path.exists():
            print(f"  X  {check_name:40s}: MISSING")
            all_ok = False
            critical_missing.append(check_name)
            continue

        found = []
        for pattern in patterns:
            found.extend(path.rglob(pattern))

        if found:
            print(f"  OK {check_name:40s}: {len(found)} files")
        else:
            print(f"  -- {check_name:40s}: empty")

    # =========================================================================
    # PER-CITY SATELLITE COVERAGE (FLAT STRUCTURE)
    # =========================================================================

    print(f"\nPer-city satellite coverage (flat dirs):")
    print("-" * 70)

    sar_coh_path = satellite_dirs['SAR_COH_DIR']
    sar_card_path = satellite_dirs['SAR_CARD_DIR']
    ms_path = satellite_dirs['MS_DIR']

    # collect all city dirs across modalities
    all_city_names = set()
    for base_dir in [sar_coh_path, sar_card_path, ms_path]:
        if base_dir.exists():
            for d in base_dir.iterdir():
                if d.is_dir() and d.name not in ('metadata', 'temp', 'desktop.ini', 'composites', 'rgb'):
                    all_city_names.add(d.name)

    if all_city_names:
        print(f"  {'City':25s} {'COH':>6s} {'CARD':>6s} {'MS':>6s}")
        print(f"  {'-'*50}")

        for city in sorted(all_city_names):
            # COH: flat TIFs (period derived at runtime from dates, not filename)
            coh_dir = sar_coh_path / city
            coh_files = list(coh_dir.glob('*.tif')) if coh_dir.exists() else []

            # CARD: flat TIFs
            card_dir = sar_card_path / city
            card_files = list(card_dir.glob('*.tif')) if card_dir.exists() else []
            card_count = len([f for f in card_files])

            # MS: flat TIFs
            ms_dir_city = ms_path / city
            ms_files = list(ms_dir_city.glob('*.tif')) if ms_dir_city.exists() else []
            ms_count = len(ms_files)

            if any([coh_files, card_count, ms_count]):
                print(f"  {city:25s} {len(coh_files):6d} {card_count:6d} {ms_count:6d}")
    else:
        print("  No city directories found")

    # =========================================================================
    # PRESTAGE ZIP INVENTORY
    # =========================================================================

    print(f"\nPrestage ZIP inventory:")
    print("-" * 70)

    raw_slc_path = local_dirs['RAW_SLC_ZIP']
    if raw_slc_path.exists():
        zips = list(raw_slc_path.glob('*.zip'))
        if zips:
            total_gb = sum(z.stat().st_size for z in zips) / (1024**3)
            print(f"  {len(zips)} ZIP files, {total_gb:.1f} GB total")
        else:
            print(f"  No ZIPs in {raw_slc_path}")
    else:
        print(f"  RAW_SLC_ZIP not mounted: {raw_slc_path}")

    # =========================================================================
    # DISK SPACE
    # =========================================================================

    print(f"\nDisk space:")
    print("-" * 70)

    if get_disk_space_fn:
        disk = get_disk_space_fn('/content/drive')
        if disk:
            print(f"  Drive:  {disk['used_gb']:.1f} / {disk['total_gb']:.1f} GB ({disk['free_gb']:.1f} GB free)")

        disk_local = get_disk_space_fn('/')
        if disk_local:
            print(f"  Local:  {disk_local['used_gb']:.1f} / {disk_local['total_gb']:.1f} GB ({disk_local['free_gb']:.1f} GB free)")

        if content_local.exists():
            disk_hdd = get_disk_space_fn(str(content_local))
            if disk_hdd:
                print(f"  HDD:    {disk_hdd['used_gb']:.1f} / {disk_hdd['total_gb']:.1f} GB ({disk_hdd['free_gb']:.1f} GB free)")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(f"\n{'='*70}")
    if not critical_missing:
        print("ALL CRITICAL DATA PRESENT")
    else:
        print(f"WARNING: {len(critical_missing)} data source(s) need attention:")
        for item in critical_missing:
            print(f"  - {item}")
    print(f"{'='*70}")

    VERIFICATION_STATUS = {
        'all_ok': all_ok,
        'critical_missing': critical_missing,
        'timestamp': datetime.now().isoformat()
    }

    return VERIFICATION_STATUS
