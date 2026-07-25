# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
dl_audit_products.py
Derived product audit: COH, CARD TIFs, MS clipped TIFs, composites, landuse.
Uses product_scan.py for flat-aware scanning.

Notebook usage:
    from dl_audit_products import run as run_audit_products
    PRODUCT_AUDIT = run_audit_products(
        sar_coh_dir=SAR_COH_DIR,
        sar_card_dir=SAR_CARD_DIR,
        ms_dir=MS_DIR,
        landuse_dir=LANDUSE_DIR,
        outputs_dir=OUTPUTS_DIR,
        insar_tracker_file=INSAR_TRACKER_FILE,
        card_tracker_file=CARD_TRACKER_FILE,
        cities_filter=CITIES_TO_PROCESS,
    )
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from product_scan import (
    scan_coh_products, scan_card_products, scan_ms_products,
    summarize_by_city_period, count_by_type
)


def run(sar_coh_dir, sar_card_dir, ms_dir, landuse_dir, outputs_dir,
        insar_tracker_file, card_tracker_file, cities_filter=None):

    sar_coh_dir = Path(sar_coh_dir)
    sar_card_dir = Path(sar_card_dir)
    ms_dir = Path(ms_dir)
    landuse_dir = Path(landuse_dir)
    outputs_dir = Path(outputs_dir)
    insar_tracker_file = Path(insar_tracker_file)
    card_tracker_file = Path(card_tracker_file)

    print("=" * 80)
    print("DL-AUDIT-PRODUCTS: DERIVED PRODUCT AUDIT")
    print("=" * 80)
    print(f"  SAR_COH_DIR: {sar_coh_dir}")
    print(f"  SAR_CARD_DIR: {sar_card_dir}")
    print(f"  MS_DIR: {ms_dir}")
    print(f"  LANDUSE_DIR: {landuse_dir}")
    if cities_filter:
        print(f"  Filter: {len(cities_filter)} cities")

    cf = set(cities_filter) if cities_filter else None

    # =========================================================================
    # 1. COHERENCE (NB03A)
    # =========================================================================
    print(f"\n{'='*80}")
    print("1. COHERENCE PRODUCTS (NB03A)")
    print(f"{'='*80}")

    coh_raw = scan_coh_products(sar_coh_dir, cf)
    coh_summary = summarize_by_city_period(coh_raw)
    coh_total = sum(len(f) for f in coh_raw.values())
    coh_mb = sum(f['size_mb'] for fs in coh_raw.values() for f in fs)

    print(f"\n  Cities: {len(coh_raw)}, TIFs: {coh_total:,}, Size: {coh_mb/1024:.1f} GB")
    print(f"  Types: {count_by_type(coh_raw)}")

    if coh_summary:
        print(f"\n  {'City':25s} {'Pre':>5s} {'Post':>5s} {'Cross':>5s} {'BL':>5s}")
        print(f"  {'-'*50}")
        for city in sorted(coh_summary):
            cs = coh_summary[city]
            print(f"  {city:25s} {cs.get('prebattle',{}).get('n_files',0):>5d} "
                  f"{cs.get('postbattle',{}).get('n_files',0):>5d} "
                  f"{cs.get('crossbattle',{}).get('n_files',0):>5d} "
                  f"{cs.get('coherence_baseline',{}).get('n_files',0):>5d}")

    # tracker cross-check
    if insar_tracker_file.exists():
        with open(insar_tracker_file) as f:
            it = json.load(f)
        tracked = set(it.get('cities', {}).keys())
        on_disk = set(coh_raw.keys())
        extra = on_disk - tracked
        missing = tracked - on_disk
        if extra:
            print(f"\n  On disk not tracked: {extra}")
        if missing:
            print(f"  Tracked not on disk: {missing}")
        if not extra and not missing:
            print(f"\n  Tracker cross-check: OK")

    # =========================================================================
    # 2. CARD TIFs (NB03B)
    # =========================================================================
    print(f"\n{'='*80}")
    print("2. CARD-BS EXTRACTED TIFS (NB03B)")
    print(f"{'='*80}")

    card_raw = scan_card_products(sar_card_dir, cf)
    card_total = sum(len(f) for f in card_raw.values())
    card_mb = sum(f['size_mb'] for fs in card_raw.values() for f in fs)

    print(f"\n  Cities: {len(card_raw)}, TIFs: {card_total:,}, Size: {card_mb/1024:.1f} GB")

    if card_raw:
        vv = sum(1 for fs in card_raw.values() for f in fs if f.get('pol') == 'VV')
        vh = sum(1 for fs in card_raw.values() for f in fs if f.get('pol') == 'VH')
        print(f"  Polarization: VV={vv:,}, VH={vh:,}")

        print(f"\n  {'City':25s} {'TIFs':>6s} {'Dates':>6s} {'MB':>8s}")
        print(f"  {'-'*50}")
        for city in sorted(card_raw):
            files = card_raw[city]
            bs_files = [f for f in files if f['product_type'] == 'card_bs']
            dates = set(f['date1'] for f in bs_files if f.get('date1'))
            mb = sum(f['size_mb'] for f in files)
            print(f"  {city:25s} {len(bs_files):>6d} {len(dates):>6d} {mb:>8.0f}")

    # =========================================================================
    # 3. MS CLIPPED TIFS (NB03C)
    # =========================================================================
    print(f"\n{'='*80}")
    print("3. MULTISPECTRAL CLIPPED TIFS (NB03C)")
    print(f"{'='*80}")

    ms_raw = scan_ms_products(ms_dir, cf)
    ms_total = sum(len(f) for f in ms_raw.values())
    ms_mb = sum(f['size_mb'] for fs in ms_raw.values() for f in fs)
    ms_types = count_by_type(ms_raw)

    clipped = sum(1 for fs in ms_raw.values() for f in fs if f['product_type'] == 'clipped_band')
    comps = sum(1 for fs in ms_raw.values() for f in fs if f['product_type'] == 'composite')

    print(f"\n  Cities: {len(ms_raw)}, Total: {ms_total:,}, Size: {ms_mb/1024:.1f} GB")
    print(f"  Clipped: {clipped:,}, Composites: {comps:,}")
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
    # 4. LANDUSE (NB03D)
    # =========================================================================
    print(f"\n{'='*80}")
    print("4. LANDUSE CLASSIFICATION (NB03D)")
    print(f"{'='*80}")

    landuse_inventory = {}
    if landuse_dir.exists():
        for city_dir in sorted(landuse_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            city = city_dir.name
            if cf and city not in cf:
                continue
            tifs = list(city_dir.rglob("*.tif"))
            csvs = list(city_dir.rglob("*.csv"))
            if tifs or csvs:
                landuse_inventory[city] = {
                    "tif_count": len(tifs),
                    "csv_count": len(csvs),
                    "total_mb": round(sum(t.stat().st_size for t in tifs) / (1024**2), 1),
                }

    print(f"  Cities: {len(landuse_inventory)}")
    for city in sorted(landuse_inventory):
        cd = landuse_inventory[city]
        print(f"    {city:25s}: {cd['tif_count']} TIFs, {cd['csv_count']} CSVs ({cd['total_mb']:.1f} MB)")

    # =========================================================================
    # 5. READINESS SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("5. READINESS SUMMARY")
    print(f"{'='*80}")

    all_cities = set()
    all_cities.update(coh_raw.keys())
    all_cities.update(card_raw.keys())
    all_cities.update(ms_raw.keys())
    all_cities.update(landuse_inventory.keys())

    summary_rows = []
    for city in sorted(all_cities):
        n_coh = len(coh_raw.get(city, []))
        n_card = len([f for f in card_raw.get(city, []) if f['product_type'] == 'card_bs'])
        n_ms = len([f for f in ms_raw.get(city, []) if f['product_type'] == 'clipped_band'])
        n_comp = len([f for f in ms_raw.get(city, []) if f['product_type'] == 'composite'])
        n_lu = landuse_inventory.get(city, {}).get('tif_count', 0)

        if n_coh and n_card and n_ms and n_comp and n_lu:
            status = "READY"
        elif n_card and n_ms and n_comp:
            status = "PARTIAL"
        elif n_card or n_ms:
            status = "MINIMAL"
        else:
            status = "EMPTY"

        summary_rows.append({
            'city': city, 'coh': n_coh, 'card': n_card,
            'ms': n_ms, 'comp': n_comp, 'lu': n_lu, 'status': status,
        })

    print(f"\n  {'City':25s} {'COH':>5s} {'CARD':>5s} {'MS':>5s} {'Comp':>5s} {'LU':>5s} {'Status'}")
    print(f"  {'-'*75}")
    for r in summary_rows:
        print(f"  {r['city']:25s} {r['coh']:>5d} {r['card']:>5d} {r['ms']:>5d} {r['comp']:>5d} {r['lu']:>5d} {r['status']}")

    df = pd.DataFrame(summary_rows)
    csv_path = outputs_dir / 'product_audit.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    rc = df['status'].value_counts()
    print(f"\n  Readiness:")
    for s, c in rc.items():
        print(f"    {s}: {c}")

    PRODUCT_AUDIT = {
        'coh_raw': coh_raw,
        'card_raw': card_raw,
        'ms_raw': ms_raw,
        'landuse_inventory': landuse_inventory,
        'summary': summary_rows,
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n{'='*80}")
    print(f"PRODUCT AUDIT COMPLETE ({len(all_cities)} cities)")
    print(f"  Ready: {len(df[df['status']=='READY'])}")
    print(f"{'='*80}")

    return PRODUCT_AUDIT
