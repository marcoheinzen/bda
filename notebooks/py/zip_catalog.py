# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
zip_catalog.py
Mirror NB02d audit+prune zip inventory into bda.sqlite.
Parallel to JSON trackers -- never touches or replaces them.

Usage:
    from zip_catalog import sync_audit_to_sqlite, sync_prune_to_sqlite

    sync_audit_to_sqlite(CATALOG_DB, NB02_AUDIT)
    sync_prune_to_sqlite(CATALOG_DB, PRUNE_RESULTS, dry_run=DRY_RUN_PRUNE)
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS zip_inventory (
    zip_id INTEGER PRIMARY KEY AUTOINCREMENT,
    modality TEXT NOT NULL,
    zip_stem TEXT NOT NULL,
    zip_filename TEXT NOT NULL,
    zip_path TEXT,
    size_gb REAL,
    satellite TEXT,
    date_str TEXT,
    date_raw TEXT,
    orbit_abs INTEGER,
    tile_id TEXT,
    status TEXT DEFAULT 'active',
    surplus_reason TEXT,
    cities_using TEXT,
    first_seen_at TEXT,
    last_seen_at TEXT,
    pruned_at TEXT,
    UNIQUE(modality, zip_stem)
);

CREATE INDEX IF NOT EXISTS idx_zip_modality ON zip_inventory(modality);
CREATE INDEX IF NOT EXISTS idx_zip_status ON zip_inventory(status);
CREATE INDEX IF NOT EXISTS idx_zip_date ON zip_inventory(date_str);

CREATE TABLE IF NOT EXISTS zip_audit_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    slc_count INTEGER,
    card_count INTEGER,
    ms_count INTEGER,
    slc_total_gb REAL,
    card_total_gb REAL,
    ms_total_gb REAL,
    slc_surplus INTEGER DEFAULT 0,
    card_surplus INTEGER DEFAULT 0,
    ms_surplus INTEGER DEFAULT 0,
    slc_pruned INTEGER DEFAULT 0,
    card_pruned INTEGER DEFAULT 0,
    ms_pruned INTEGER DEFAULT 0,
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


def _parse_ms_tile(name):
    m = re.search(r'_T(\d{2}[A-Z]{3})_', name)
    return m.group(1) if m else None


def _parse_ms_date(name):
    m = re.search(r'_(\d{8})T\d{6}_', name)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}", d
    return None, None


def sync_audit_to_sqlite(db_path, nb02_audit):
    """
    Upsert all zips from NB02_AUDIT into zip_inventory.
    Marks zips no longer on disk as status='missing'.
    """
    if nb02_audit is None:
        print("  zip_catalog: NB02_AUDIT is None, skipping")
        return

    conn = _connect(db_path)
    now = datetime.now().isoformat()
    cur = conn.cursor()

    # track which (modality, zip_stem) we see this run
    seen = set()

    # --- SLC ---
    slc_inv = nb02_audit.get('slc_inventory', {})
    for stem, info in slc_inv.items():
        seen.add(('SLC', stem))
        cur.execute("""
            INSERT INTO zip_inventory
                (modality, zip_stem, zip_filename, zip_path, size_gb,
                 satellite, date_str, date_raw, orbit_abs,
                 status, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(modality, zip_stem) DO UPDATE SET
                zip_path = excluded.zip_path,
                size_gb = excluded.size_gb,
                status = 'active',
                last_seen_at = excluded.last_seen_at
        """, (
            'SLC', stem, info.get('filename', ''), info.get('path', ''),
            info.get('size_gb'), info.get('satellite'),
            info.get('date'), info.get('date_raw'),
            info.get('orbit_abs'),
            now, now,
        ))

    # undersized SLC
    for info in nb02_audit.get('slc_undersized', []):
        stem = info.get('base_name', '')
        if not stem:
            continue
        seen.add(('SLC', stem))
        cur.execute("""
            INSERT INTO zip_inventory
                (modality, zip_stem, zip_filename, zip_path, size_gb,
                 satellite, date_str, date_raw, orbit_abs,
                 status, surplus_reason, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'undersized', 'undersized', ?, ?)
            ON CONFLICT(modality, zip_stem) DO UPDATE SET
                zip_path = excluded.zip_path,
                size_gb = excluded.size_gb,
                status = 'undersized',
                surplus_reason = 'undersized',
                last_seen_at = excluded.last_seen_at
        """, (
            'SLC', stem, info.get('filename', ''), info.get('path', ''),
            info.get('size_gb'), info.get('satellite'),
            info.get('date'), info.get('date_raw'),
            info.get('orbit_abs'),
            now, now,
        ))

    # --- CARD ---
    card_zips = nb02_audit.get('card_zips', {})
    for stem, info in card_zips.items():
        seen.add(('CARD', stem))
        cur.execute("""
            INSERT INTO zip_inventory
                (modality, zip_stem, zip_filename, zip_path, size_gb,
                 status, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(modality, zip_stem) DO UPDATE SET
                zip_path = excluded.zip_path,
                size_gb = excluded.size_gb,
                status = 'active',
                last_seen_at = excluded.last_seen_at
        """, (
            'CARD', stem, info.get('filename', ''), info.get('path', ''),
            info.get('size_gb'),
            now, now,
        ))

    # --- MS ---
    ms_zips = nb02_audit.get('ms_zips', {})
    for stem, info in ms_zips.items():
        tile = _parse_ms_tile(stem)
        date_str, date_raw = _parse_ms_date(stem)
        seen.add(('MS', stem))
        cur.execute("""
            INSERT INTO zip_inventory
                (modality, zip_stem, zip_filename, zip_path, size_gb,
                 tile_id, date_str, date_raw,
                 status, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(modality, zip_stem) DO UPDATE SET
                zip_path = excluded.zip_path,
                size_gb = excluded.size_gb,
                tile_id = excluded.tile_id,
                date_str = excluded.date_str,
                date_raw = excluded.date_raw,
                status = 'active',
                last_seen_at = excluded.last_seen_at
        """, (
            'MS', stem, info.get('filename', ''), info.get('path', ''),
            info.get('size_gb'),
            tile, date_str, date_raw,
            now, now,
        ))

    # mark rows not seen this run as missing (only if they were active)
    cur.execute("SELECT modality, zip_stem FROM zip_inventory WHERE status = 'active'")
    active_rows = cur.fetchall()
    missing_count = 0
    for modality, zip_stem in active_rows:
        if (modality, zip_stem) not in seen:
            cur.execute("""
                UPDATE zip_inventory SET status = 'missing', last_seen_at = ?
                WHERE modality = ? AND zip_stem = ?
            """, (now, modality, zip_stem))
            missing_count += 1

    # audit log entry
    cur.execute("""
        INSERT INTO zip_audit_log
            (timestamp, slc_count, card_count, ms_count,
             slc_total_gb, card_total_gb, ms_total_gb)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        now,
        len(slc_inv) + len(nb02_audit.get('slc_undersized', [])),
        len(card_zips),
        len(ms_zips),
        nb02_audit.get('slc_total_gb', 0),
        nb02_audit.get('card_zip_gb', 0),
        nb02_audit.get('ms_zip_gb', 0),
    ))

    conn.commit()

    # summary
    cur.execute("SELECT modality, COUNT(*), SUM(size_gb) FROM zip_inventory WHERE status='active' GROUP BY modality")
    rows = cur.fetchall()
    total_rows = sum(r[1] for r in rows)
    total_gb = sum(r[2] or 0 for r in rows)
    print(f"\n  zip_catalog: synced audit -> {db_path}")
    print(f"    Active: {total_rows} zips ({total_gb:.0f} GB)")
    for mod, cnt, gb in rows:
        print(f"      {mod:6s}: {cnt:5d} ({gb or 0:.1f} GB)")
    if missing_count:
        print(f"    Marked missing: {missing_count}")

    conn.close()
    return total_rows


def sync_prune_to_sqlite(db_path, prune_results, dry_run=True):
    """
    Update zip_inventory with surplus/prune status from PRUNE_RESULTS.
    If dry_run=True in prune: marks surplus only.
    If dry_run=False and prune actually deleted: marks pruned.
    """
    if prune_results is None:
        print("  zip_catalog: PRUNE_RESULTS is None, skipping")
        return
    if prune_results.get('aborted'):
        print(f"  zip_catalog: prune aborted ({prune_results.get('reason')}), skipping")
        return

    conn = _connect(db_path)
    now = datetime.now().isoformat()
    cur = conn.cursor()

    prune_was_dry = prune_results.get('dry_run', True)

    # --- SLC surplus ---
    slc_surplus = prune_results.get('slc_surplus', [])
    for stem in slc_surplus:
        if prune_was_dry:
            new_status = 'surplus'
        else:
            new_status = 'pruned'
        cur.execute("""
            UPDATE zip_inventory SET
                status = ?,
                surplus_reason = 'not_in_any_metadata',
                pruned_at = CASE WHEN ? = 'pruned' THEN ? ELSE pruned_at END
            WHERE modality = 'SLC' AND zip_stem = ?
        """, (new_status, new_status, now, stem))

    # --- MS surplus ---
    ms_classified = prune_results.get('ms_surplus_classified', {})
    ms_to_delete = set(prune_results.get('ms_to_delete', []))
    ms_kept = set(prune_results.get('ms_kept', []))

    for stem in prune_results.get('ms_surplus', []):
        info = ms_classified.get(stem)
        if info:
            tile, date_str, date_raw, reason, city_str = info
        else:
            reason = 'unknown'
            city_str = ''

        if not prune_was_dry and stem in ms_to_delete:
            new_status = 'pruned'
        elif stem in ms_to_delete:
            new_status = 'surplus_to_delete'
        else:
            new_status = 'surplus_kept'

        cur.execute("""
            UPDATE zip_inventory SET
                status = ?,
                surplus_reason = ?,
                cities_using = ?,
                pruned_at = CASE WHEN ? = 'pruned' THEN ? ELSE pruned_at END
            WHERE modality = 'MS' AND zip_stem = ?
        """, (new_status, reason, city_str if city_str != '-' else None,
              new_status, now, stem))

    # --- CARD surplus ---
    card_surplus = prune_results.get('card_surplus', [])
    for stem in card_surplus:
        if prune_was_dry:
            new_status = 'surplus'
        else:
            new_status = 'pruned'
        cur.execute("""
            UPDATE zip_inventory SET
                status = ?,
                surplus_reason = 'not_in_tracker',
                pruned_at = CASE WHEN ? = 'pruned' THEN ? ELSE pruned_at END
            WHERE modality = 'CARD' AND zip_stem = ?
        """, (new_status, new_status, now, stem))

    # update audit log (last row)
    slc_pruned = prune_results.get('slc_deleted', 0)
    ms_pruned = prune_results.get('ms_deleted', 0)
    card_pruned = prune_results.get('card_deleted', 0)

    cur.execute("""
        UPDATE zip_audit_log SET
            slc_surplus = ?,
            card_surplus = ?,
            ms_surplus = ?,
            slc_pruned = ?,
            card_pruned = ?,
            ms_pruned = ?,
            prune_dry_run = ?,
            prune_selector = ?
        WHERE log_id = (SELECT MAX(log_id) FROM zip_audit_log)
    """, (
        len(slc_surplus), len(card_surplus), len(prune_results.get('ms_surplus', [])),
        slc_pruned, card_pruned, ms_pruned,
        1 if prune_was_dry else 0,
        str(prune_results.get('prune_selector', '')),
    ))

    conn.commit()

    # summary
    cur.execute("""
        SELECT status, COUNT(*), COALESCE(SUM(size_gb), 0)
        FROM zip_inventory GROUP BY status ORDER BY status
    """)
    rows = cur.fetchall()
    print(f"\n  zip_catalog: synced prune -> {db_path}")
    print(f"    dry_run={prune_was_dry}")
    for status, cnt, gb in rows:
        print(f"      {status:22s}: {cnt:5d} ({gb:.1f} GB)")

    conn.close()


def query_zips(db_path, modality=None, status=None):
    """Quick query helper for downstream notebooks."""
    conn = _connect(db_path)
    sql = "SELECT * FROM zip_inventory WHERE 1=1"
    params = []
    if modality:
        sql += " AND modality = ?"
        params.append(modality)
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY modality, date_str"
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def summary(db_path):
    """Print summary of zip_inventory."""
    conn = _connect(db_path)
    cur = conn.cursor()

    print(f"\n  zip_catalog summary: {db_path}")
    cur.execute("""
        SELECT modality, status, COUNT(*), COALESCE(SUM(size_gb), 0)
        FROM zip_inventory GROUP BY modality, status ORDER BY modality, status
    """)
    rows = cur.fetchall()
    for mod, status, cnt, gb in rows:
        print(f"    {mod:6s} {status:22s}: {cnt:5d} ({gb:.1f} GB)")

    cur.execute("SELECT COUNT(*) FROM zip_audit_log")
    n_logs = cur.fetchone()[0]
    if n_logs:
        cur.execute("SELECT timestamp, slc_count, card_count, ms_count FROM zip_audit_log ORDER BY log_id DESC LIMIT 1")
        row = cur.fetchone()
        print(f"    Last audit: {row[0]} (SLC={row[1]}, CARD={row[2]}, MS={row[3]})")

    conn.close()
