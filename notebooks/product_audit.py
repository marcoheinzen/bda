# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
product_audit.py
Derived product audit: scans COH/CARD/MS/Landuse products, cross-checks trackers,
detects superseded COH pairs, produces readiness summary for NB05.

Notebook usage:
    from product_audit import run as run_product_audit
    NB04_AUDIT = run_product_audit(
        sar_coh_dir=SAR_COH_DIR,
        sar_card_dir=SAR_CARD_DIR,
        ms_dir=MS_DIR,
        landuse_dir=LANDUSE_DIR,
        insar_tracker_file=INSAR_TRACKER_FILE,
        card_tracker_file=CARD_TRACKER_FILE,
        outputs_dir=OUTPUTS_DIR,
        cities_to_process=CITIES_TO_PROCESS,
    )
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from product_scan import (
    scan_coh_products, scan_card_products, scan_ms_products,
    summarize_by_city_period, count_by_type
)
from product_pre_audit import detect_superseded_coh


def run(sar_coh_dir, sar_card_dir, ms_dir, landuse_dir,
        insar_tracker_file, card_tracker_file, outputs_dir,
        cities_to_process=None):
    """
    Args:
        sar_coh_dir:         Path to SAR_COH_DIR
        sar_card_dir:        Path to SAR_CARD_DIR
        ms_dir:              Path to MS_DIR
        landuse_dir:         Path to LANDUSE_DIR
        insar_tracker_file:  Path to INSAR_TRACKER_FILE
        card_tracker_file:   Path to CARD_TRACKER_FILE
        outputs_dir:         Path to OUTPUTS_DIR
        cities_to_process:   list or None

    Returns:
        NB04_AUDIT dict
    """
    sar_coh_dir = Path(sar_coh_dir)
    sar_card_dir = Path(sar_card_dir)
    ms_dir = Path(ms_dir)
    landuse_dir = Path(landuse_dir)
    insar_tracker_file = Path(insar_tracker_file)
    card_tracker_file = Path(card_tracker_file)
    outputs_dir = Path(outputs_dir)

    print("=" * 80)
    print("CELL AUDIT: SCAN ALL NB03A-D DERIVED PRODUCTS")
    print("=" * 80)
    print(f"  SAR_COH_DIR: {sar_coh_dir}")
    print(f"  SAR_CARD_DIR: {sar_card_dir}")
    print(f"  MS_DIR: {ms_dir}")
    print(f"  LANDUSE_DIR: {landuse_dir}")
    if cities_to_process:
        print(f"  CITIES_TO_PROCESS: {len(cities_to_process)} cities")
    else:
        print(f"  CITIES_TO_PROCESS: None (scan all)")

    cities_filter = cities_to_process or None

    # =========================================================================
    # 1. SLC COHERENCE PRODUCTS (NB03A)
    # =========================================================================
    print(f"\n{'='*80}")
    print("1. COHERENCE PRODUCTS (NB03A)")
    print(f"{'='*80}")

    coh_raw = scan_coh_products(sar_coh_dir, cities_filter)
    coh_summary = summarize_by_city_period(coh_raw)
    coh_types = count_by_type(coh_raw)
    coh_total = sum(len(files) for files in coh_raw.values())
    coh_total_mb = sum(f['size_mb'] for files in coh_raw.values() for f in files)

    print(f"\n  Cities with COH products: {len(coh_raw)}")
    print(f"  Total TIFs: {coh_total:,}")
    print(f"  Total size: {coh_total_mb:,.0f} MB ({coh_total_mb/1024:.1f} GB)")
    print(f"  Types: {coh_types}")

    if coh_summary:
        print(f"\n  {'City':25s} {'Pre':>5s} {'Post':>5s} {'Cross':>5s} {'BL':>5s}")
        print(f"  {'-'*50}")
        for city in sorted(coh_summary):
            cs = coh_summary[city]
            pre = cs.get('prebattle', {}).get('n_files', 0)
            post = cs.get('postbattle', {}).get('n_files', 0)
            cross = cs.get('crossbattle', {}).get('n_files', 0)
            bl = cs.get('coherence_baseline', {}).get('n_files', 0)
            print(f"  {city:25s} {pre:>5d} {post:>5d} {cross:>5d} {bl:>5d}")

    # Superseded COH pairs
    coh_superseded = detect_superseded_coh(coh_raw)
    n_superseded = sum(len(v) for v in coh_superseded.values())
    if n_superseded > 0:
        print(f"\n  SUPERSEDED COH PAIRS: {n_superseded}")
        for city in sorted(coh_superseded):
            for d1, d3, pol, d2, fpath in coh_superseded[city]:
                mb = fpath.stat().st_size / (1024*1024) if fpath.exists() else 0
                print(f"    {city:25s} {pol} {d1}_{d3} -> {d1}_{d2}+{d2}_{d3}  ({mb:.1f} MB)")
    else:
        print(f"\n  Superseded COH pairs: 0 (clean)")

    # Cross-check with InSAR tracker
    if insar_tracker_file.exists():
        with open(insar_tracker_file) as f:
            insar_tracker = json.load(f)
        tracked_cities = set(insar_tracker.get('cities', {}).keys())
        disk_cities = set(coh_raw.keys())
        print(f"\n  Tracker cross-check:")
        print(f"    Tracker cities: {len(tracked_cities)}")
        print(f"    Disk cities:    {len(disk_cities)}")
        diff1 = disk_cities - tracked_cities
        diff2 = tracked_cities - disk_cities
        if diff1:
            print(f"    On disk not tracked: {diff1}")
        if diff2:
            print(f"    Tracked not on disk: {diff2}")
        if not diff1 and not diff2:
            print(f"    Match: OK")
    else:
        print(f"\n  InSAR tracker not found: {insar_tracker_file}")

    # =========================================================================
    # 2. CARD-BS EXTRACTED TIFS (NB03B)
    # =========================================================================
    print(f"\n{'='*80}")
    print("2. CARD-BS EXTRACTED TIFS (NB03B)")
    print(f"{'='*80}")

    card_raw = scan_card_products(sar_card_dir, cities_filter)
    card_total = sum(len(files) for files in card_raw.values())
    card_total_mb = sum(f['size_mb'] for files in card_raw.values() for f in files)

    print(f"\n  Cities with CARD products: {len(card_raw)}")
    print(f"  Total TIFs: {card_total:,}")
    print(f"  Total size: {card_total_mb:,.0f} MB ({card_total_mb/1024:.1f} GB)")

    if card_raw:
        vv_count = sum(1 for files in card_raw.values() for f in files if f.get('pol') == 'VV')
        vh_count = sum(1 for files in card_raw.values() for f in files if f.get('pol') == 'VH')
        print(f"  Polarization: VV={vv_count:,}, VH={vh_count:,}")

        print(f"\n  {'City':25s} {'TIFs':>6s} {'Dates':>6s} {'MB':>8s}")
        print(f"  {'-'*50}")
        for city in sorted(card_raw):
            files = card_raw[city]
            n_tifs = len([f for f in files if f['product_type'] == 'card_bs'])
            dates = set(f['date1'] for f in files if f.get('date1'))
            mb = sum(f['size_mb'] for f in files)
            print(f"  {city:25s} {n_tifs:>6d} {len(dates):>6d} {mb:>8.0f}")

    # Cross-check with CARD tracker
    if card_tracker_file.exists():
        with open(card_tracker_file) as f:
            card_tracker = json.load(f)
        extracted_count = sum(1 for v in card_tracker.values() if isinstance(v, dict) and v.get("extracted"))
        print(f"\n  CARD tracker: {extracted_count} entries marked extracted")

    # =========================================================================
    # 3. MULTISPECTRAL CLIPPED TIFS (NB03C)
    # =========================================================================
    print(f"\n{'='*80}")
    print("3. MULTISPECTRAL CLIPPED TIFS (NB03C)")
    print(f"{'='*80}")

    ms_raw = scan_ms_products(ms_dir, cities_filter)
    ms_total = sum(len(files) for files in ms_raw.values())
    ms_total_mb = sum(f['size_mb'] for files in ms_raw.values() for f in files)
    ms_types = count_by_type(ms_raw)

    ms_clipped = sum(1 for files in ms_raw.values() for f in files if f['product_type'] == 'clipped_band')
    ms_composites = sum(1 for files in ms_raw.values() for f in files if f['product_type'] == 'composite')

    print(f"\n  Cities with MS data: {len(ms_raw)}")
    print(f"  Total files: {ms_total:,}")
    print(f"  Clipped bands: {ms_clipped:,}")
    print(f"  Composites: {ms_composites:,}")
    print(f"  Total size: {ms_total_mb:,.0f} MB ({ms_total_mb/1024:.1f} GB)")
    print(f"  Types: {ms_types}")

    if ms_raw:
        print(f"\n  {'City':25s} {'Bands':>6s} {'Dates':>6s} {'Comp':>6s} {'MB':>8s}")
        print(f"  {'-'*55}")
        for city in sorted(ms_raw):
            files = ms_raw[city]
            n_bands = len([f for f in files if f['product_type'] == 'clipped_band'])
            dates = set(f['date1'] for f in files if f.get('date1') and f['product_type'] == 'clipped_band')
            n_comp = len([f for f in files if f['product_type'] == 'composite'])
            mb = sum(f['size_mb'] for f in files)
            print(f"  {city:25s} {n_bands:>6d} {len(dates):>6d} {n_comp:>6d} {mb:>8.0f}")

    # =========================================================================
    # 4. LANDUSE CLASSIFICATION (NB03D)
    # =========================================================================
    print(f"\n{'='*80}")
    print("4. LANDUSE CLASSIFICATION (NB03D)")
    print(f"{'='*80}")
    print(f"  LANDUSE_DIR: {landuse_dir}")

    landuse_inventory = {}
    if landuse_dir.exists():
        for city_dir in sorted(landuse_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            city = city_dir.name
            tifs = list(city_dir.rglob("*.tif"))
            csvs = list(city_dir.rglob("*.csv"))
            if tifs or csvs:
                landuse_inventory[city] = {
                    "tif_count": len(tifs),
                    "csv_count": len(csvs),
                    "total_mb": round(sum(t.stat().st_size for t in tifs) / (1024**2), 1),
                }

    print(f"  Cities with landuse: {len(landuse_inventory)}")
    if landuse_inventory:
        for city in sorted(landuse_inventory):
            cd = landuse_inventory[city]
            print(f"    {city:25s}: {cd['tif_count']} TIFs, {cd['csv_count']} CSVs ({cd['total_mb']:.1f} MB)")

    # =========================================================================
    # 5. AGGREGATE SUMMARY + READINESS
    # =========================================================================
    print(f"\n{'='*80}")
    print("5. AGGREGATE SUMMARY")
    print(f"{'='*80}")

    all_cities = set()
    all_cities.update(coh_raw.keys())
    all_cities.update(card_raw.keys())
    all_cities.update(ms_raw.keys())
    all_cities.update(landuse_inventory.keys())

    summary_rows = []
    for city in sorted(all_cities):
        coh_tifs = len(coh_raw.get(city, []))
        card_tifs = len([f for f in card_raw.get(city, []) if f['product_type'] == 'card_bs'])
        ms_clips = len([f for f in ms_raw.get(city, []) if f['product_type'] == 'clipped_band'])
        ms_comps = len([f for f in ms_raw.get(city, []) if f['product_type'] == 'composite'])
        lu_tifs = landuse_inventory.get(city, {}).get("tif_count", 0)

        has_coh = coh_tifs > 0
        has_card = card_tifs > 0
        has_ms = ms_clips > 0
        has_comp = ms_comps > 0
        has_lu = lu_tifs > 0

        if has_coh and has_card and has_ms and has_comp and has_lu:
            readiness = "READY"
        elif has_card and has_ms and has_comp:
            readiness = "PARTIAL (no COH)"
        elif has_card or has_ms:
            readiness = "MINIMAL"
        else:
            readiness = "EMPTY"

        summary_rows.append({
            "city": city,
            "coh_tifs": coh_tifs,
            "card_tifs": card_tifs,
            "ms_clips": ms_clips,
            "composites": ms_comps,
            "landuse": lu_tifs,
            "readiness": readiness,
        })

    summary_df = pd.DataFrame(summary_rows)

    print(f"\n  {'City':25s} {'COH':>5s} {'CARD':>5s} {'MS':>5s} {'Comp':>5s} {'LU':>5s} {'Status'}")
    print(f"  {'-'*75}")
    for _, row in summary_df.iterrows():
        print(f"  {row['city']:25s} {row['coh_tifs']:>5d} {row['card_tifs']:>5d} "
              f"{row['ms_clips']:>5d} {row['composites']:>5d} {row['landuse']:>5d} {row['readiness']}")

    readiness_counts = summary_df["readiness"].value_counts()
    print(f"\n  Readiness summary:")
    for status, count in readiness_counts.items():
        print(f"    {status}: {count}")

    # Save CSV
    summary_csv = outputs_dir / "nb04a_product_audit.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Saved: {summary_csv}")

    # =========================================================================
    # BUILD RESULT DICT
    # =========================================================================
    NB04_AUDIT = {
        "timestamp": datetime.now().isoformat(),
        "coh_raw": coh_raw,
        "card_raw": card_raw,
        "ms_raw": ms_raw,
        "coh_superseded": coh_superseded,
        "landuse_inventory": landuse_inventory,
        "summary": summary_rows,
        "summary_df": summary_df,
    }

    print(f"\n{'='*80}")
    print(f"PRODUCT AUDIT COMPLETE")
    print(f"  Cities: {len(all_cities)}")
    print(f"  COH: {sum(len(f) for f in coh_raw.values()):,} TIFs across {len(coh_raw)} cities")
    print(f"  CARD: {card_total:,} TIFs across {len(card_raw)} cities")
    print(f"  MS: {ms_clipped:,} clipped + {ms_composites:,} composites across {len(ms_raw)} cities")
    print(f"  Landuse: {len(landuse_inventory)} cities")
    print(f"  Superseded COH: {n_superseded}")
    print(f"  Ready for NB05: {len(summary_df[summary_df['readiness']=='READY'])}")
    print(f"{'='*80}")

    return NB04_AUDIT
