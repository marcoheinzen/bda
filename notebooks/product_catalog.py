# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
product_catalog.py
Mirror NB03e (and NB03a-d) product audit+prune results into bda.sqlite.
Parallel to JSON trackers -- never touches or replaces them.

Tables:
    product_inventory  -- one row per TIF product across COH/CARD/MS/landuse
    product_audit_log  -- one row per audit run

Usage:
    from product_catalog import sync_product_audit_to_sqlite, sync_product_prune_to_sqlite

    sync_product_audit_to_sqlite(CATALOG_DB, NB04_AUDIT)
    sync_product_prune_to_sqlite(CATALOG_DB, PRUNE_RESULTS)
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS product_inventory (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    modality TEXT NOT NULL,
    product_type TEXT NOT NULL,
    filename TEXT NOT NULL,
    filepath TEXT,
    date1 TEXT,
    date2 TEXT,
    polarization TEXT,
    band TEXT,
    period TEXT,
    subdir TEXT,
    size_mb REAL,
    status TEXT DEFAULT 'active',
    prune_category TEXT,
    prune_detail TEXT,
    first_seen_at TEXT,
    last_seen_at TEXT,
    pruned_at TEXT,
    source_notebook TEXT,
    UNIQUE(city, filename)
);

CREATE INDEX IF NOT EXISTS idx_prod_city ON product_inventory(city);
CREATE INDEX IF NOT EXISTS idx_prod_modality ON product_inventory(modality);
CREATE INDEX IF NOT EXISTS idx_prod_status ON product_inventory(status);
CREATE INDEX IF NOT EXISTS idx_prod_type ON product_inventory(product_type);

CREATE TABLE IF NOT EXISTS product_audit_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    coh_count INTEGER,
    card_count INTEGER,
    ms_count INTEGER,
    landuse_count INTEGER,
    total_mb REAL,
    n_candidates INTEGER DEFAULT 0,
    n_pruned INTEGER DEFAULT 0,
    prune_dry_run INTEGER DEFAULT 1,
    prune_selector TEXT,
    notes TEXT
);
"""


def _connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    # WAL mode needs mmap -- fails on drvfs (WSL2 Windows mounts)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(DDL)
    return conn


def ensure_schema(db_path):
    """
    Create bda.sqlite and product_inventory / product_audit_log tables if not exist.
    Call unconditionally from NB03e setup cell so tables exist before any sync.
    Safe to call multiple times -- CREATE TABLE IF NOT EXISTS, never drops data.
    """
    conn = _connect(db_path)
    conn.close()
    print(f"  product_catalog: schema ready -> {db_path}")


def _extract_band(filename):
    m = re.search(r'_(B\d{2}|B8A|SCL|TCI|NDVI|NBR|NDWI|NDBI|SAVI)_', filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'_(B\d{2}|B8A|SCL|TCI|NDVI|NBR|NDWI|NDBI|SAVI)\.tif$', filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _notebook_for_modality(modality, product_type):
    if modality == 'COH':
        return 'NB03a'
    if modality == 'CARD':
        if product_type in ('temporal_stats',):
            return 'NB03d'
        return 'NB03b'
    if modality == 'MS':
        if product_type in ('composite', 'rgb', 'nbr', 'cloud_mask', 'visibility'):
            return 'NB03d'
        return 'NB03c'
    if modality == 'landuse':
        return 'NB03d'
    return None


def _upsert_file(cur, city, modality, f, now):
    filepath = str(f['file'])
    filename = Path(filepath).name
    cur.execute("""
        INSERT INTO product_inventory
            (city, modality, product_type, filename, filepath,
             date1, date2, polarization, band, period, subdir, size_mb,
             status, source_notebook, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
        ON CONFLICT(city, filename) DO UPDATE SET
            filepath = excluded.filepath,
            size_mb = excluded.size_mb,
            modality = excluded.modality,
            product_type = excluded.product_type,
            date1 = excluded.date1,
            date2 = excluded.date2,
            polarization = excluded.polarization,
            band = excluded.band,
            period = excluded.period,
            subdir = excluded.subdir,
            status = 'active',
            prune_category = NULL,
            prune_detail = NULL,
            last_seen_at = excluded.last_seen_at
    """, (
        city, modality, f.get('product_type', 'unknown'), filename, filepath,
        f.get('date1'), f.get('date2'), f.get('pol'),
        f.get('band') or _extract_band(filename),
        f.get('period'), f.get('subdir'), f.get('size_mb', 0),
        _notebook_for_modality(modality, f.get('product_type', '')),
        now, now,
    ))


def sync_product_audit_to_sqlite(db_path, nb04_audit):
    """
    Upsert all products from NB04_AUDIT into product_inventory.
    Marks products no longer on disk as status='missing'.
    """
    if nb04_audit is None:
        print("  product_catalog: NB04_AUDIT is None, skipping")
        return

    conn = _connect(db_path)
    now = datetime.now().isoformat()
    cur = conn.cursor()

    seen = set()
    counts = {'COH': 0, 'CARD': 0, 'MS': 0, 'landuse': 0}
    total_mb = 0.0

    # --- COH ---
    for city, files in nb04_audit.get('coh_raw', {}).items():
        for f in files:
            _upsert_file(cur, city, 'COH', f, now)
            seen.add((city, Path(str(f['file'])).name))
            counts['COH'] += 1
            total_mb += f.get('size_mb', 0)

    # --- CARD ---
    for city, files in nb04_audit.get('card_raw', {}).items():
        for f in files:
            _upsert_file(cur, city, 'CARD', f, now)
            seen.add((city, Path(str(f['file'])).name))
            counts['CARD'] += 1
            total_mb += f.get('size_mb', 0)

    # --- MS ---
    for city, files in nb04_audit.get('ms_raw', {}).items():
        for f in files:
            _upsert_file(cur, city, 'MS', f, now)
            seen.add((city, Path(str(f['file'])).name))
            counts['MS'] += 1
            total_mb += f.get('size_mb', 0)

    # --- Landuse (summary only, no per-file detail) ---
    for city, info in nb04_audit.get('landuse_inventory', {}).items():
        tif_count = info.get('tif_count', 0)
        lu_mb = info.get('total_mb', 0)
        cur.execute("""
            INSERT INTO product_inventory
                (city, modality, product_type, filename, filepath,
                 size_mb, status, source_notebook, first_seen_at, last_seen_at)
            VALUES (?, 'landuse', 'classification', ?, ?, ?, 'active', 'NB03d', ?, ?)
            ON CONFLICT(city, filename) DO UPDATE SET
                size_mb = excluded.size_mb,
                status = 'active',
                last_seen_at = excluded.last_seen_at
        """, (
            city,
            f"landuse_summary_{tif_count}tifs",
            str(Path('landuse') / city),
            lu_mb,
            now, now,
        ))
        seen.add((city, f"landuse_summary_{tif_count}tifs"))
        counts['landuse'] += tif_count
        total_mb += lu_mb

    # mark previously-active rows not seen this run as missing
    cur.execute("SELECT city, filename FROM product_inventory WHERE status = 'active'")
    active_rows = cur.fetchall()
    missing_count = 0
    for city, filename in active_rows:
        if (city, filename) not in seen:
            cur.execute("""
                UPDATE product_inventory SET status = 'missing', last_seen_at = ?
                WHERE city = ? AND filename = ?
            """, (now, city, filename))
            missing_count += 1

    # audit log
    cur.execute("""
        INSERT INTO product_audit_log
            (timestamp, coh_count, card_count, ms_count, landuse_count, total_mb)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (now, counts['COH'], counts['CARD'], counts['MS'], counts['landuse'], round(total_mb, 1)))

    conn.commit()

    # summary
    cur.execute("""
        SELECT modality, COUNT(*), COALESCE(SUM(size_mb), 0)
        FROM product_inventory WHERE status='active' GROUP BY modality
    """)
    rows = cur.fetchall()
    total_active = sum(r[1] for r in rows)
    total_active_mb = sum(r[2] or 0 for r in rows)
    print(f"\n  product_catalog: synced audit -> {db_path}")
    print(f"    Active: {total_active} products ({total_active_mb:.0f} MB, {total_active_mb/1024:.1f} GB)")
    for mod, cnt, mb in rows:
        print(f"      {mod:8s}: {cnt:6d} ({mb or 0:.0f} MB)")
    if missing_count:
        print(f"    Marked missing: {missing_count}")

    conn.close()
    return total_active


def sync_product_prune_to_sqlite(db_path, prune_results):
    """
    Update product_inventory with prune status from PRUNE_RESULTS.
    """
    if prune_results is None:
        print("  product_catalog: PRUNE_RESULTS is None, skipping")
        return
    if not prune_results.get('candidates'):
        print("  product_catalog: no prune candidates, skipping")
        return

    conn = _connect(db_path)
    now = datetime.now().isoformat()
    cur = conn.cursor()

    prune_was_dry = prune_results.get('dry_run', True)
    to_delete_paths = set(p for p, _, _, _ in prune_results.get('to_delete', []))

    updated = 0
    for path_str, category, size_mb, detail in prune_results.get('candidates', []):
        filepath = Path(path_str)
        filename = filepath.name
        city = filepath.parent.name
        # handle subdirs: if parent is a subdir, city is grandparent
        if city in ('temporal_stats', 'coherence_baseline', 'composites', 'rgb', 'nbr'):
            city = filepath.parent.parent.name

        if not prune_was_dry and path_str in to_delete_paths:
            new_status = 'pruned'
        elif path_str in to_delete_paths:
            new_status = 'surplus_to_delete'
        else:
            new_status = 'surplus_kept'

        cur.execute("""
            UPDATE product_inventory SET
                status = ?,
                prune_category = ?,
                prune_detail = ?,
                pruned_at = CASE WHEN ? = 'pruned' THEN ? ELSE pruned_at END
            WHERE city = ? AND filename = ?
        """, (new_status, category, detail, new_status, now, city, filename))
        if cur.rowcount > 0:
            updated += 1

    # update audit log (last row)
    n_candidates = len(prune_results.get('candidates', []))
    n_pruned = prune_results.get('deleted_count', 0)
    cur.execute("""
        UPDATE product_audit_log SET
            n_candidates = ?,
            n_pruned = ?,
            prune_dry_run = ?,
            prune_selector = ?
        WHERE log_id = (SELECT MAX(log_id) FROM product_audit_log)
    """, (
        n_candidates, n_pruned,
        1 if prune_was_dry else 0,
        str(prune_results.get('prune_selector', '')),
    ))

    conn.commit()

    # summary
    cur.execute("""
        SELECT status, COUNT(*), COALESCE(SUM(size_mb), 0)
        FROM product_inventory GROUP BY status ORDER BY status
    """)
    rows = cur.fetchall()
    print(f"\n  product_catalog: synced prune -> {db_path}")
    print(f"    dry_run={prune_was_dry}, updated={updated}")
    for status, cnt, mb in rows:
        print(f"      {status:22s}: {cnt:6d} ({mb:.0f} MB)")

    conn.close()


def register_products(db_path, city, modality, file_dicts, source_notebook=None):
    """
    Lightweight registration for use inside NB03a/b/c/d after processing a city.
    file_dicts: list of dicts with keys matching product_scan output:
        {'file': Path, 'date1': str, 'date2': str|None, 'pol': str|None,
         'period': str, 'product_type': str, 'size_mb': float, ...}
    """
    if not file_dicts:
        return 0
    conn = _connect(db_path)
    now = datetime.now().isoformat()
    cur = conn.cursor()
    count = 0
    for f in file_dicts:
        _upsert_file(cur, city, modality, f, now)
        if source_notebook:
            cur.execute("""
                UPDATE product_inventory SET source_notebook = ?
                WHERE city = ? AND filename = ?
            """, (source_notebook, city, Path(str(f['file'])).name))
        count += 1
    conn.commit()
    conn.close()
    print(f"  product_catalog: registered {count} {modality} products for {city}")
    return count


def register_tifs(db_path, city, modality, tif_paths, source_notebook=None):
    """
    Minimal registration from a list of Path objects (no pre-parsed dicts needed).
    Extracts date/pol/band from filename. For use in NB03a/b/c/d processing loops.
    """
    import re as _re
    file_dicts = []
    for p in tif_paths:
        p = Path(p)
        if not p.exists():
            continue
        fname = p.name
        size_mb = p.stat().st_size / (1024 * 1024)
        # extract dates
        dates = _re.findall(r'(\d{8})', fname)
        d1 = dates[0] if len(dates) >= 1 else None
        d2 = dates[1] if len(dates) >= 2 else None
        # extract pol
        pol_m = _re.search(r'_(VV|VH|vv|vh)_', fname)
        pol = pol_m.group(1).upper() if pol_m else None
        # extract band
        band = _extract_band(fname)
        # guess product_type
        if 'COH' in fname.upper() or 'coh' in fname:
            pt = 'coherence'
        elif 'CARD' in fname.upper():
            pt = 'card_bs'
        elif 'composite' in fname.lower():
            pt = 'composite'
        elif 'SCL' in fname.upper():
            pt = 'scl'
        elif band:
            pt = 'clipped_band'
        else:
            pt = 'derived'
        file_dicts.append({
            'file': p,
            'date1': d1,
            'date2': d2,
            'pol': pol,
            'band': band,
            'product_type': pt,
            'size_mb': round(size_mb, 2),
            'period': None,
            'subdir': None,
        })
    return register_products(db_path, city, modality, file_dicts, source_notebook)


def register_coh_products(db_path, sar_coh_dir, cities=None):
    """
    Scan SAR_COH_DIR and register all COH products for given cities.
    Uses product_scan.scan_coh_products for correct filename parsing.
    Drop-in cell for NB03a.
    """
    from product_scan import scan_coh_products
    coh_raw = scan_coh_products(Path(sar_coh_dir), cities)
    total = 0
    for city, files in coh_raw.items():
        total += register_products(db_path, city, 'COH', files, source_notebook='NB03a')
    print(f"  product_catalog: COH total = {total} products across {len(coh_raw)} cities")
    return total


def register_card_products(db_path, sar_card_dir, cities=None):
    """
    Scan SAR_CARD_DIR and register all CARD products for given cities.
    Uses product_scan.scan_card_products for correct filename parsing.
    Drop-in cell for NB03b.
    """
    from product_scan import scan_card_products
    card_raw = scan_card_products(Path(sar_card_dir), cities)
    total = 0
    for city, files in card_raw.items():
        total += register_products(db_path, city, 'CARD', files, source_notebook='NB03b')
    print(f"  product_catalog: CARD total = {total} products across {len(card_raw)} cities")
    return total


def register_ms_products(db_path, ms_dir, cities=None):
    """
    Scan MS_DIR and register all MS products for given cities.
    Uses product_scan.scan_ms_products for correct filename parsing.
    Drop-in cell for NB03c.
    """
    from product_scan import scan_ms_products
    ms_raw = scan_ms_products(Path(ms_dir), cities)
    total = 0
    for city, files in ms_raw.items():
        total += register_products(db_path, city, 'MS', files, source_notebook='NB03c')
    print(f"  product_catalog: MS total = {total} products across {len(ms_raw)} cities")
    return total


def register_temporal_products(db_path, temporal_root, cities=None):
    """
    Scan TEMPORAL_ROOT and register all temporal products (rolling, zscore, post_baseline).
    Structure: TEMPORAL_ROOT/{city}/{COH|CARD}/{rolling|zscore|post_baseline}/*.tif
    Drop-in cell for NB03d.
    """
    import re as _re
    temporal_root = Path(temporal_root)
    if not temporal_root.exists():
        print(f"  product_catalog: TEMPORAL_ROOT not found: {temporal_root}")
        return 0
    total = 0
    for city_dir in sorted(temporal_root.iterdir()):
        if not city_dir.is_dir() or city_dir.name == 'desktop.ini':
            continue
        city = city_dir.name
        if cities and city not in cities:
            continue
        for sensor_dir in city_dir.iterdir():
            if not sensor_dir.is_dir():
                continue
            sensor = sensor_dir.name.upper()
            if sensor not in ('COH', 'CARD'):
                continue
            for type_dir in sensor_dir.iterdir():
                if not type_dir.is_dir():
                    continue
                product_type = type_dir.name
                tifs = list(type_dir.glob('*.tif'))
                file_dicts = []
                for p in tifs:
                    fname = p.name
                    size_mb = p.stat().st_size / (1024 * 1024)
                    dates = _re.findall(r'(\d{8})', fname)
                    d1 = dates[0] if dates else None
                    pol_m = _re.search(r'_(VV|VH|vv|vh)_', fname)
                    pol = pol_m.group(1).upper() if pol_m else None
                    file_dicts.append({
                        'file': p,
                        'date1': d1,
                        'date2': None,
                        'pol': pol,
                        'band': None,
                        'product_type': product_type,
                        'size_mb': round(size_mb, 2),
                        'period': None,
                        'subdir': f"{sensor}/{product_type}",
                    })
                if file_dicts:
                    total += register_products(db_path, city, sensor, file_dicts, source_notebook='NB03d')
    print(f"  product_catalog: temporal total = {total} products")
    return total


def register_landuse_products(db_path, landuse_dir, cities=None):
    """
    Scan LANDUSE_DIR and register all landuse products.
    Structure: LANDUSE_DIR/{city}/{period}/{YYYYMMDD}/landuse_classification.tif
               LANDUSE_DIR/{city}/{period}/{YYYYMMDD}/indices/*.tif
    Drop-in cell for NB03d.
    """
    import re as _re
    landuse_dir = Path(landuse_dir)
    if not landuse_dir.exists():
        print(f"  product_catalog: LANDUSE_DIR not found: {landuse_dir}")
        return 0
    total = 0
    for city_dir in sorted(landuse_dir.iterdir()):
        if not city_dir.is_dir() or city_dir.name == 'desktop.ini':
            continue
        city = city_dir.name
        if cities and city not in cities:
            continue
        file_dicts = []
        for tif in city_dir.rglob('*.tif'):
            fname = tif.name
            size_mb = tif.stat().st_size / (1024 * 1024)
            rel = tif.relative_to(city_dir)
            parts = rel.parts
            period = parts[0] if len(parts) >= 1 else None
            date_str = parts[1] if len(parts) >= 2 else None
            if date_str and not _re.match(r'^\d{8}$', date_str):
                date_str = None
            subdir_path = str(rel.parent)
            if 'indices' in parts:
                pt = 'landuse_index'
                band = fname.replace('.tif', '').upper()
            elif fname == 'landuse_classification.tif':
                pt = 'landuse_classification'
                band = None
            else:
                pt = 'landuse_other'
                band = None
            file_dicts.append({
                'file': tif,
                'date1': date_str,
                'date2': None,
                'pol': None,
                'band': band,
                'product_type': pt,
                'size_mb': round(size_mb, 2),
                'period': period,
                'subdir': subdir_path,
            })
        if file_dicts:
            total += register_products(db_path, city, 'landuse', file_dicts, source_notebook='NB03d')
    print(f"  product_catalog: landuse total = {total} products")
    return total


def register_derived_products(db_path, sar_coh_dir, sar_card_dir, ms_dir,
                               temporal_root, landuse_dir, cities=None):
    """
    Convenience: register all NB03d products in one call.
    Calls the 3 source-dir scanners (COH/CARD/MS subdirs) + temporal + landuse.
    Drop-in cell for NB03d.
    """
    total = 0
    total += register_coh_products(db_path, sar_coh_dir, cities)
    total += register_card_products(db_path, sar_card_dir, cities)
    total += register_ms_products(db_path, ms_dir, cities)
    total += register_temporal_products(db_path, temporal_root, cities)
    total += register_landuse_products(db_path, landuse_dir, cities)
    print(f"  product_catalog: derived grand total = {total}")
    return total


def query_products(db_path, city=None, modality=None, status=None, product_type=None):
    """Quick query helper for downstream notebooks."""
    conn = _connect(db_path)
    sql = "SELECT * FROM product_inventory WHERE 1=1"
    params = []
    if city:
        sql += " AND city = ?"
        params.append(city)
    if modality:
        sql += " AND modality = ?"
        params.append(modality)
    if status:
        sql += " AND status = ?"
        params.append(status)
    if product_type:
        sql += " AND product_type = ?"
        params.append(product_type)
    sql += " ORDER BY city, modality, date1"
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def summary(db_path):
    """Print summary of product_inventory."""
    conn = _connect(db_path)
    cur = conn.cursor()

    print(f"\n  product_catalog summary: {db_path}")
    cur.execute("""
        SELECT modality, status, COUNT(*), COALESCE(SUM(size_mb), 0)
        FROM product_inventory GROUP BY modality, status ORDER BY modality, status
    """)
    rows = cur.fetchall()
    for mod, status, cnt, mb in rows:
        print(f"    {mod:8s} {status:22s}: {cnt:6d} ({mb:.0f} MB)")

    cur.execute("SELECT COUNT(*) FROM product_audit_log")
    n_logs = cur.fetchone()[0]
    if n_logs:
        cur.execute("""
            SELECT timestamp, coh_count, card_count, ms_count, landuse_count
            FROM product_audit_log ORDER BY log_id DESC LIMIT 1
        """)
        row = cur.fetchone()
        print(f"    Last audit: {row[0]} (COH={row[1]}, CARD={row[2]}, MS={row[3]}, LU={row[4]})")

    conn.close()
