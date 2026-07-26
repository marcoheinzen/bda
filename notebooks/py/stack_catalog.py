# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_catalog.py
SQLite catalog for BDA pipeline metadata.

Single file-based DB replacing 20+ JSON trackers. WAL mode for concurrent access.
Each notebook writes to its owned tables only (SPOC principle).

Table ownership:
    cities           NB01b creates, all read
    scenes           NB02a/b/c write (one sensor each), NB02d audits
    products         NB03a/b/c/d write, NB03e audits, NB05 reads
    processing_runs  NB03a/b/c/d write
    data_stack       NB05 ALIGN writes (tracks what's in data_stack)
    features         NB05 FEATURES writes (parquet column inventory)
    rename_lut       NB05 CATALOG writes (original -> convention filename mapping)

Concurrency: WAL journal + 10s busy_timeout. Safe for sequential notebook runs
and occasional parallel NB02+NB03 (different tables).

Usage:
    from stack_catalog import BDACatalog

    with BDACatalog(db_path) as cat:
        cat.upsert_city('Mariupol', tier=0, battle_start='2022-02-24', ...)
        cat.upsert_scene('Mariupol', 'CARD', 'Mariupol_CARD_VH_20220103.tif', ...)
        cities = cat.query_cities(tier=[0,1], has_unosat=True)
        scenes = cat.query_scenes(city='Mariupol', sensor='CARD')

    # Or without context manager:
    cat = BDACatalog(db_path)
    cat.upsert_city(...)
    cat.close()
"""

import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime


# =========================================================================
# SCHEMA
# =========================================================================

SCHEMA_SQL = """
-- Cities: one row per city, from cities_config.json + AOI.geojson
CREATE TABLE IF NOT EXISTS cities (
    city_name       TEXT PRIMARY KEY,
    tier            INTEGER,
    battle_start    TEXT,
    battle_stop     TEXT,
    conflict_ongoing INTEGER DEFAULT 0,
    has_unosat      INTEGER DEFAULT 0,
    lat             REAL,
    lon             REAL,
    utm_epsg        INTEGER,
    width_px        INTEGER,
    height_px       INTEGER,
    n_buildings     INTEGER,
    n_damaged       INTEGER,
    n_undamaged     INTEGER,
    orbit           INTEGER,
    orbit_card      INTEGER,
    updated_at      TEXT,
    updated_by      TEXT
);

-- Scenes: one row per downloaded zip/file (NB02 tracker equivalent)
CREATE TABLE IF NOT EXISTS scenes (
    scene_id        TEXT PRIMARY KEY,   -- e.g. 'Mariupol_CARD_VH_20220103'
    city_name       TEXT NOT NULL,
    sensor          TEXT NOT NULL,      -- CARD, SLC, MS, COH_CARD
    polarization    TEXT,               -- VV, VH, or NULL
    date_str        TEXT,               -- YYYYMMDD
    date2_str       TEXT,               -- second date for COH pairs
    zip_filename    TEXT,
    zip_path        TEXT,
    zip_size_bytes  INTEGER,
    download_status TEXT DEFAULT 'pending',  -- pending, downloading, success, failed
    download_date   TEXT,
    orbit_number    INTEGER,
    tile_id         TEXT,               -- S2 tile e.g. 36TYP
    source_api      TEXT,               -- copernicus, cdse, earthdata
    notes           TEXT,
    updated_at      TEXT,
    updated_by      TEXT,
    FOREIGN KEY (city_name) REFERENCES cities(city_name)
);

-- Products: one row per TIF in SATELLITE_DIR (NB03 output)
CREATE TABLE IF NOT EXISTS products (
    product_id      TEXT PRIMARY KEY,   -- e.g. 'Mariupol/temporal_stats/s1__vv__assessment__mean.tif'
    city_name       TEXT NOT NULL,
    sensor          TEXT NOT NULL,      -- CARD, COH, MS, landuse, composite, temporal
    product_type    TEXT NOT NULL,      -- flat, temporal_stats, coherence_baseline, composite, rolling, zscore, etc.
    filename        TEXT NOT NULL,      -- original filename (NB03 output)
    renamed         TEXT,               -- convention filename (after ALIGN rename)
    rel_path        TEXT,               -- relative path within SATELLITE_DIR/{city}/
    date_str        TEXT,
    polarization    TEXT,
    band            TEXT,
    period          TEXT,               -- baseline, assessment, prebattle_baseline, etc.
    statistic       TEXT,               -- mean, std, max, etc.
    file_size_bytes INTEGER,
    processing_status TEXT DEFAULT 'success',
    created_date    TEXT,
    updated_at      TEXT,
    updated_by      TEXT,
    FOREIGN KEY (city_name) REFERENCES cities(city_name)
);

-- Processing runs: log of NB03 processing steps
CREATE TABLE IF NOT EXISTS processing_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    notebook        TEXT NOT NULL,      -- NB03a, NB03b, NB03c, NB03d
    cell_id         TEXT,
    city_name       TEXT NOT NULL,
    status          TEXT NOT NULL,      -- success, failed, skipped
    started_at      TEXT,
    elapsed_s       REAL,
    n_inputs        INTEGER,
    n_outputs       INTEGER,
    n_errors        INTEGER,
    error_msg       TEXT,
    config_json     TEXT,               -- JSON blob of cell config
    updated_at      TEXT,
    FOREIGN KEY (city_name) REFERENCES cities(city_name)
);

-- Data stack: tracks what's in STACK_DIR after ALIGN (NB05)
CREATE TABLE IF NOT EXISTS data_stack (
    stack_id        TEXT PRIMARY KEY,   -- e.g. 'Mariupol/SAR_CARD/flat/s1__vv__20220103.tif'
    city_name       TEXT NOT NULL,
    group_name      TEXT NOT NULL,      -- SAR_CARD, SAR_SLC, multispectral, landuse, etc.
    subdir          TEXT NOT NULL,      -- flat, temporal_stats, composites/prebattle_baseline, etc.
    filename        TEXT NOT NULL,      -- convention filename in data_stack
    source_product_id TEXT,             -- FK to products.product_id
    width_px        INTEGER,
    height_px       INTEGER,
    aligned         INTEGER DEFAULT 1,
    file_size_bytes INTEGER,
    updated_at      TEXT,
    FOREIGN KEY (city_name) REFERENCES cities(city_name)
);

-- Features: parquet column inventory (NB05b REGISTER output)
CREATE TABLE IF NOT EXISTS features (
    feature_name    TEXT PRIMARY KEY,   -- column name in parquet
    parquet_name    TEXT,               -- which parquet: bda_product_prepost, bda_buildings, etc.
    sensor          TEXT,               -- s1, s2, cd, lulc, fire, qa, bldg, meta
    measurement     TEXT,               -- vv, vh, coh, b02, ndvi, landuse, etc.
    period          TEXT,               -- baseline, assessment, prebattle_baseline, etc.
    statistic       TEXT,               -- mean, std, max, mode, zscore
    zonal_stat      TEXT,               -- building-level: mean, std, max, mode
    group_name      TEXT,               -- for get_feature_groups compatibility
    column_role     TEXT DEFAULT 'feature',  -- feature, label, metadata, id
    is_ml_feature   INTEGER DEFAULT 1,  -- 0 for meta columns (legacy, derived from column_role)
    dtype           TEXT,               -- float64, int64, object
    description     TEXT,
    updated_at      TEXT
);

-- Rename LUT: maps NB03 original filenames to __ convention names (NB05 ALIGN)
CREATE TABLE IF NOT EXISTS rename_lut (
    lut_id          TEXT PRIMARY KEY,   -- '{city}_{sensor}_{product_type}_{original}'
    city_name       TEXT NOT NULL,
    sensor          TEXT NOT NULL,
    product_type    TEXT NOT NULL,
    original_filename TEXT NOT NULL,    -- NB03 output: Mariupol_card_assessment_mean_VV.tif
    renamed_filename TEXT NOT NULL,     -- convention: s1__vv__assessment__mean.tif
    src_rel_path    TEXT,               -- relative path within source dir
    updated_at      TEXT,
    FOREIGN KEY (city_name) REFERENCES cities(city_name)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_rename_lut_city ON rename_lut(city_name);
CREATE INDEX IF NOT EXISTS idx_rename_lut_original ON rename_lut(original_filename);
CREATE INDEX IF NOT EXISTS idx_rename_lut_renamed ON rename_lut(renamed_filename);
CREATE INDEX IF NOT EXISTS idx_scenes_city ON scenes(city_name);
CREATE INDEX IF NOT EXISTS idx_scenes_sensor ON scenes(sensor);
CREATE INDEX IF NOT EXISTS idx_scenes_city_sensor ON scenes(city_name, sensor);
CREATE INDEX IF NOT EXISTS idx_products_city ON products(city_name);
CREATE INDEX IF NOT EXISTS idx_products_city_sensor ON products(city_name, sensor);
CREATE INDEX IF NOT EXISTS idx_products_type ON products(product_type);
CREATE INDEX IF NOT EXISTS idx_data_stack_city ON data_stack(city_name);
CREATE INDEX IF NOT EXISTS idx_processing_runs_city ON processing_runs(city_name);
CREATE INDEX IF NOT EXISTS idx_processing_runs_nb ON processing_runs(notebook);
"""

# Migrations for existing DBs (ALTER TABLE is idempotent via try/except)
MIGRATIONS = [
    "ALTER TABLE features ADD COLUMN parquet_name TEXT",
    "ALTER TABLE features ADD COLUMN column_role TEXT DEFAULT 'feature'",
]


# =========================================================================
# FEATURE NAME PARSER
# =========================================================================

def parse_feature_name(col):
    """Parse __ convention column name into metadata dict.

    Examples:
        s1__vv__delta__mean_mean   -> sensor=s1, measurement=vv, period=delta, statistic=mean, zonal_stat=mean
        s2__landuse__prebattle_baseline_mode -> sensor=s2, measurement=landuse, period=prebattle_baseline, statistic=mode
        s2__b04__winter_baseline_std  -> sensor=s2, measurement=b04, period=winter_baseline, statistic=std
        s1__coh__baseline__mean_mean -> sensor=s1, measurement=coh, period=baseline, statistic=mean, zonal_stat=mean
        building_id -> sensor=meta (no __ separator)
    """
    parts = col.split('__')
    info = {'feature_name': col, 'is_ml_feature': 1}

    if len(parts) < 2:
        # no __ separator = metadata column
        info['sensor'] = 'meta'
        info['group_name'] = 'meta'
        info['is_ml_feature'] = 0
        return info

    info['sensor'] = parts[0]  # s1, s2, fire, lulc, etc.

    # handle composite namespace: s2__composite__landuse -> skip 'composite', measurement=landuse
    if len(parts) >= 3 and parts[1] == 'composite':
        parts = [parts[0]] + parts[2:]  # drop 'composite'

    if len(parts) >= 2:
        info['measurement'] = parts[1]  # vv, vh, coh, b04, ndvi, landuse, etc.

    if len(parts) >= 3:
        info['period'] = parts[2]  # delta, baseline, prebattle_baseline, etc.

    if len(parts) >= 4:
        raw_stat = parts[3]
    elif len(parts) == 3:
        # statistic is suffix of last part: e.g. prebattle_baseline_std -> period=prebattle_baseline, stat=std
        # or: delta__mean_mean -> period already set, stat in remaining
        raw_stat = None
    else:
        raw_stat = None

    # extract zonal_stat from compound stat like mean_mean, max_mean
    if raw_stat:
        stat_parts = raw_stat.rsplit('_', 1)
        if len(stat_parts) == 2 and stat_parts[1] in ('mean', 'std', 'max', 'min', 'median', 'mode', 'count'):
            info['statistic'] = stat_parts[0]
            info['zonal_stat'] = stat_parts[1]
        else:
            info['statistic'] = raw_stat
    else:
        # try to extract stat from the period field suffix
        last = parts[-1]
        for suffix in ('_mean', '_std', '_max', '_min', '_median', '_mode', '_count'):
            if last.endswith(suffix):
                info['statistic'] = suffix[1:]
                if 'period' in info and info['period'] == last:
                    info['period'] = last[:len(last)-len(suffix)]
                break

    # group_name for feature_groups compatibility
    m = info.get('measurement', '')
    s = info.get('sensor', '')
    if s == 's1' and m == 'coh':
        info['group_name'] = 'coh'
    elif s == 's1' and m in ('vv', 'vh'):
        info['group_name'] = 'card'
    elif s == 's2' and m.startswith('b'):
        info['group_name'] = 'ms_bands'
    elif s == 's2' and m == 'landuse':
        info['group_name'] = 'landuse'
    elif s == 's2':
        info['group_name'] = 'ms_indices'
    elif s in ('fire', 'burn'):
        info['group_name'] = 'fire'
    else:
        info['group_name'] = s

    return info

class BDACatalog:
    """SQLite catalog for BDA pipeline. WAL mode, thread-safe within process."""

    def __init__(self, db_path, readonly=False):
        self.db_path = Path(db_path)
        self.readonly = readonly
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if readonly:
            # plain connect (not URI mode which fails on WSL drvfs mounts)
            self.conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            self.conn.execute("PRAGMA query_only=ON;")
        else:
            self.conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            # WAL mode needs mmap -- fails on drvfs (WSL2 Windows mounts)
            try:
                self.conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                self.conn.execute("PRAGMA journal_mode=DELETE;")
            self.conn.execute("PRAGMA busy_timeout=10000;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.executescript(SCHEMA_SQL)
            for mig in MIGRATIONS:
                try:
                    self.conn.execute(mig)
                except sqlite3.OperationalError:
                    pass  # column already exists
            self.conn.commit()

        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _now(self):
        return datetime.now().isoformat()

    # =====================================================================
    # CITIES
    # =====================================================================

    def upsert_city(self, city_name, **kwargs):
        """Insert or update city. Caller provides any subset of columns."""
        kwargs['city_name'] = city_name
        kwargs['updated_at'] = self._now()
        cols = list(kwargs.keys())
        placeholders = ','.join(['?'] * len(cols))
        col_str = ','.join(cols)
        update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'city_name'])
        sql = f"""INSERT INTO cities ({col_str}) VALUES ({placeholders})
                  ON CONFLICT(city_name) DO UPDATE SET {update_parts}"""
        self.conn.execute(sql, [kwargs[c] for c in cols])
        self.conn.commit()

    def upsert_cities_from_config(self, config_path, updated_by='NB01b'):
        """Bulk import cities from cities_config.json.
        Handles both dict-of-dicts {"Mariupol": {...}} and list-of-dicts [{"name": "Mariupol", ...}].
        """
        with open(config_path) as f:
            cfg = json.load(f)

        # detect format: dict-of-dicts (keys=city names) vs list-of-dicts
        if isinstance(cfg, dict) and 'cities' not in cfg:
            # dict-of-dicts: {"Mariupol": {"tier": 0, ...}, ...}
            items = cfg.items()
        elif isinstance(cfg, dict) and 'cities' in cfg:
            items = cfg['cities'].items() if isinstance(cfg['cities'], dict) else [(c.get('city_name', c.get('name', '')), c) for c in cfg['cities']]
        elif isinstance(cfg, list):
            items = [(c.get('city_name', c.get('name', '')), c) for c in cfg]
        else:
            items = []

        n = 0
        for name, city in items:
            if not name:
                continue
            bs = city.get('battle_stop', '')
            self.upsert_city(
                name,
                tier=city.get('tier'),
                battle_start=city.get('battle_start'),
                battle_stop=bs,
                conflict_ongoing=1 if str(bs).lower() in ('', 'ongoing', 'none') else 0,
                has_unosat=1 if city.get('has_unosat', city.get('tier', 99) in (0, 1, 2)) else 0,
                lat=city.get('lat'),
                lon=city.get('lon'),
                updated_by=updated_by,
            )
            n += 1
        print(f"  Imported {n} cities from {config_path}")
        return n

    def upsert_orbits_from_plan(self, plan_path):
        """Update orbit and orbit_card from nb02a_scene_plan.json."""
        import json as _json
        plan_path = Path(plan_path)
        if not plan_path.exists():
            print(f"  Plan not found: {plan_path}")
            return 0
        with open(plan_path) as f:
            plan = _json.load(f)
        # ensure columns exist (for DBs created before orbit columns were added)
        try:
            self.conn.execute("ALTER TABLE cities ADD COLUMN orbit INTEGER")
        except Exception:
            pass
        try:
            self.conn.execute("ALTER TABLE cities ADD COLUMN orbit_card INTEGER")
        except Exception:
            pass
        n = 0
        for city, p in plan.items():
            orbit = p.get('orbit')
            orbit_card = p.get('orbit_card') or orbit
            if orbit is not None:
                self.conn.execute(
                    "UPDATE cities SET orbit = ?, orbit_card = ?, updated_at = datetime('now') WHERE city_name = ?",
                    [orbit, orbit_card, city]
                )
                n += 1
        self.conn.commit()
        print(f"  Orbits updated: {n} cities from {plan_path.name}")
        return n

    def query_cities(self, tier=None, has_unosat=None, conflict_ongoing=None):
        """Query cities with optional filters. Returns list of sqlite3.Row."""
        sql = "SELECT * FROM cities WHERE 1=1"
        params = []
        if tier is not None:
            if isinstance(tier, (list, tuple)):
                sql += f" AND tier IN ({','.join(['?']*len(tier))})"
                params.extend(tier)
            else:
                sql += " AND tier = ?"
                params.append(tier)
        if has_unosat is not None:
            sql += " AND has_unosat = ?"
            params.append(1 if has_unosat else 0)
        if conflict_ongoing is not None:
            sql += " AND conflict_ongoing = ?"
            params.append(1 if conflict_ongoing else 0)
        sql += " ORDER BY tier, city_name"
        return self.conn.execute(sql, params).fetchall()

    def get_city(self, city_name):
        """Get single city row or None."""
        row = self.conn.execute("SELECT * FROM cities WHERE city_name = ?", [city_name]).fetchone()
        return row

    # =====================================================================
    # SCENES
    # =====================================================================

    def upsert_scene(self, city_name, sensor, scene_id=None, **kwargs):
        """Insert or update a scene (downloaded zip/file)."""
        if scene_id is None:
            scene_id = f"{city_name}_{sensor}_{kwargs.get('date_str', 'unknown')}"
        kwargs['scene_id'] = scene_id
        kwargs['city_name'] = city_name
        kwargs['sensor'] = sensor
        kwargs['updated_at'] = self._now()
        cols = list(kwargs.keys())
        placeholders = ','.join(['?'] * len(cols))
        col_str = ','.join(cols)
        update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'scene_id'])
        sql = f"""INSERT INTO scenes ({col_str}) VALUES ({placeholders})
                  ON CONFLICT(scene_id) DO UPDATE SET {update_parts}"""
        self.conn.execute(sql, [kwargs[c] for c in cols])
        self.conn.commit()

    def upsert_scenes_batch(self, rows, commit=True):
        """Batch upsert scenes. rows = list of dicts with at least scene_id, city_name, sensor."""
        now = self._now()
        for row in rows:
            row.setdefault('updated_at', now)
            cols = list(row.keys())
            placeholders = ','.join(['?'] * len(cols))
            col_str = ','.join(cols)
            update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'scene_id'])
            sql = f"""INSERT INTO scenes ({col_str}) VALUES ({placeholders})
                      ON CONFLICT(scene_id) DO UPDATE SET {update_parts}"""
            self.conn.execute(sql, [row[c] for c in cols])
        if commit:
            self.conn.commit()

    def query_scenes(self, city=None, sensor=None, status=None):
        """Query scenes with filters."""
        sql = "SELECT * FROM scenes WHERE 1=1"
        params = []
        if city:
            sql += " AND city_name = ?"
            params.append(city)
        if sensor:
            sql += " AND sensor = ?"
            params.append(sensor)
        if status:
            sql += " AND download_status = ?"
            params.append(status)
        sql += " ORDER BY city_name, sensor, date_str"
        return self.conn.execute(sql, params).fetchall()

    def count_scenes(self, city=None, sensor=None):
        """Count scenes, optionally filtered."""
        sql = "SELECT COUNT(*) FROM scenes WHERE 1=1"
        params = []
        if city:
            sql += " AND city_name = ?"
            params.append(city)
        if sensor:
            sql += " AND sensor = ?"
            params.append(sensor)
        return self.conn.execute(sql, params).fetchone()[0]

    # =====================================================================
    # PRODUCTS
    # =====================================================================

    def upsert_product(self, city_name, sensor, product_type, filename, **kwargs):
        """Insert or update a product TIF."""
        rel_path = kwargs.pop('rel_path', f"{city_name}/{product_type}/{filename}")
        product_id = kwargs.pop('product_id', rel_path)
        kwargs['product_id'] = product_id
        kwargs['city_name'] = city_name
        kwargs['sensor'] = sensor
        kwargs['product_type'] = product_type
        kwargs['filename'] = filename
        kwargs['rel_path'] = rel_path
        kwargs['updated_at'] = self._now()
        cols = list(kwargs.keys())
        placeholders = ','.join(['?'] * len(cols))
        col_str = ','.join(cols)
        update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'product_id'])
        sql = f"""INSERT INTO products ({col_str}) VALUES ({placeholders})
                  ON CONFLICT(product_id) DO UPDATE SET {update_parts}"""
        self.conn.execute(sql, [kwargs[c] for c in cols])
        self.conn.commit()

    def upsert_products_batch(self, rows, commit=True):
        """Batch upsert products."""
        now = self._now()
        for row in rows:
            row.setdefault('updated_at', now)
            cols = list(row.keys())
            placeholders = ','.join(['?'] * len(cols))
            col_str = ','.join(cols)
            update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'product_id'])
            sql = f"""INSERT INTO products ({col_str}) VALUES ({placeholders})
                      ON CONFLICT(product_id) DO UPDATE SET {update_parts}"""
            self.conn.execute(sql, [row[c] for c in cols])
        if commit:
            self.conn.commit()

    def query_products(self, city=None, sensor=None, product_type=None):
        """Query products with filters."""
        sql = "SELECT * FROM products WHERE 1=1"
        params = []
        if city:
            sql += " AND city_name = ?"
            params.append(city)
        if sensor:
            sql += " AND sensor = ?"
            params.append(sensor)
        if product_type:
            sql += " AND product_type = ?"
            params.append(product_type)
        sql += " ORDER BY city_name, sensor, product_type, filename"
        return self.conn.execute(sql, params).fetchall()

    def count_products(self, city=None, sensor=None, product_type=None):
        """Count products."""
        sql = "SELECT COUNT(*) FROM products WHERE 1=1"
        params = []
        if city:
            sql += " AND city_name = ?"
            params.append(city)
        if sensor:
            sql += " AND sensor = ?"
            params.append(sensor)
        if product_type:
            sql += " AND product_type = ?"
            params.append(product_type)
        return self.conn.execute(sql, params).fetchone()[0]

    # =====================================================================
    # PROCESSING RUNS
    # =====================================================================

    def log_run(self, notebook, city_name, status, **kwargs):
        """Log a processing run."""
        kwargs['notebook'] = notebook
        kwargs['city_name'] = city_name
        kwargs['status'] = status
        kwargs['updated_at'] = self._now()
        if 'config_json' in kwargs and not isinstance(kwargs['config_json'], str):
            kwargs['config_json'] = json.dumps(kwargs['config_json'])
        cols = list(kwargs.keys())
        placeholders = ','.join(['?'] * len(cols))
        col_str = ','.join(cols)
        sql = f"INSERT INTO processing_runs ({col_str}) VALUES ({placeholders})"
        self.conn.execute(sql, [kwargs[c] for c in cols])
        self.conn.commit()

    def query_runs(self, notebook=None, city=None, status=None, last_n=None):
        """Query processing runs."""
        sql = "SELECT * FROM processing_runs WHERE 1=1"
        params = []
        if notebook:
            sql += " AND notebook = ?"
            params.append(notebook)
        if city:
            sql += " AND city_name = ?"
            params.append(city)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY run_id DESC"
        if last_n:
            sql += f" LIMIT {int(last_n)}"
        return self.conn.execute(sql, params).fetchall()

    # =====================================================================
    # DATA STACK
    # =====================================================================

    def upsert_stack_file(self, city_name, group_name, subdir, filename, **kwargs):
        """Track a file in data_stack."""
        stack_id = f"{city_name}/{subdir}/{filename}"
        kwargs['stack_id'] = stack_id
        kwargs['city_name'] = city_name
        kwargs['group_name'] = group_name
        kwargs['subdir'] = subdir
        kwargs['filename'] = filename
        kwargs['updated_at'] = self._now()
        cols = list(kwargs.keys())
        placeholders = ','.join(['?'] * len(cols))
        col_str = ','.join(cols)
        update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'stack_id'])
        sql = f"""INSERT INTO data_stack ({col_str}) VALUES ({placeholders})
                  ON CONFLICT(stack_id) DO UPDATE SET {update_parts}"""
        self.conn.execute(sql, [kwargs[c] for c in cols])
        self.conn.commit()

    def upsert_stack_batch(self, rows, commit=True):
        """Batch upsert data_stack entries."""
        now = self._now()
        for row in rows:
            row.setdefault('updated_at', now)
            stack_id = f"{row['city_name']}/{row['subdir']}/{row['filename']}"
            row['stack_id'] = stack_id
            cols = list(row.keys())
            placeholders = ','.join(['?'] * len(cols))
            col_str = ','.join(cols)
            update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'stack_id'])
            sql = f"""INSERT INTO data_stack ({col_str}) VALUES ({placeholders})
                      ON CONFLICT(stack_id) DO UPDATE SET {update_parts}"""
            self.conn.execute(sql, [row[c] for c in cols])
        if commit:
            self.conn.commit()

    def query_stack(self, city=None, group_name=None):
        """Query data_stack files."""
        sql = "SELECT * FROM data_stack WHERE 1=1"
        params = []
        if city:
            sql += " AND city_name = ?"
            params.append(city)
        if group_name:
            sql += " AND group_name = ?"
            params.append(group_name)
        sql += " ORDER BY city_name, group_name, subdir, filename"
        return self.conn.execute(sql, params).fetchall()

    # =====================================================================
    # RENAME LUT
    # =====================================================================

    def upsert_rename_lut_batch(self, rows, commit=True):
        """Batch insert rename mappings. rows = list of dicts with
        city_name, sensor, product_type, original_filename, renamed_filename."""
        now = self._now()
        for row in rows:
            lut_id = f"{row['city_name']}_{row['sensor']}_{row['product_type']}_{row['original_filename']}"
            row['lut_id'] = lut_id
            row.setdefault('updated_at', now)
            cols = list(row.keys())
            placeholders = ','.join(['?'] * len(cols))
            col_str = ','.join(cols)
            update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'lut_id'])
            sql = f"""INSERT INTO rename_lut ({col_str}) VALUES ({placeholders})
                      ON CONFLICT(lut_id) DO UPDATE SET {update_parts}"""
            self.conn.execute(sql, [row[c] for c in cols])
        if commit:
            self.conn.commit()

    def query_rename_lut(self, city=None, sensor=None, original=None, renamed=None):
        """Query rename LUT. Returns list of sqlite3.Row."""
        sql = "SELECT * FROM rename_lut WHERE 1=1"
        params = []
        if city:
            sql += " AND city_name = ?"
            params.append(city)
        if sensor:
            sql += " AND sensor = ?"
            params.append(sensor)
        if original:
            sql += " AND original_filename = ?"
            params.append(original)
        if renamed:
            sql += " AND renamed_filename = ?"
            params.append(renamed)
        sql += " ORDER BY city_name, sensor, original_filename"
        return self.conn.execute(sql, params).fetchall()

    def get_original_name(self, city_name, renamed_filename):
        """Reverse lookup: convention name -> original NB03 name."""
        row = self.conn.execute(
            "SELECT original_filename FROM rename_lut WHERE city_name = ? AND renamed_filename = ?",
            [city_name, renamed_filename]
        ).fetchone()
        return row['original_filename'] if row else None

    def get_renamed_name(self, city_name, original_filename):
        """Forward lookup: original NB03 name -> convention name."""
        row = self.conn.execute(
            "SELECT renamed_filename FROM rename_lut WHERE city_name = ? AND original_filename = ?",
            [city_name, original_filename]
        ).fetchone()
        return row['renamed_filename'] if row else None

    # =====================================================================
    # FEATURES
    # =====================================================================

    def upsert_feature(self, feature_name, **kwargs):
        """Register a parquet column."""
        kwargs['feature_name'] = feature_name
        kwargs['updated_at'] = self._now()
        cols = list(kwargs.keys())
        placeholders = ','.join(['?'] * len(cols))
        col_str = ','.join(cols)
        update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'feature_name'])
        sql = f"""INSERT INTO features ({col_str}) VALUES ({placeholders})
                  ON CONFLICT(feature_name) DO UPDATE SET {update_parts}"""
        self.conn.execute(sql, [kwargs[c] for c in cols])
        self.conn.commit()

    def upsert_features_batch(self, rows, commit=True):
        """Batch register parquet columns."""
        now = self._now()
        for row in rows:
            row.setdefault('updated_at', now)
            cols = list(row.keys())
            placeholders = ','.join(['?'] * len(cols))
            col_str = ','.join(cols)
            update_parts = ','.join([f"{c}=excluded.{c}" for c in cols if c != 'feature_name'])
            sql = f"""INSERT INTO features ({col_str}) VALUES ({placeholders})
                      ON CONFLICT(feature_name) DO UPDATE SET {update_parts}"""
            self.conn.execute(sql, [row[c] for c in cols])
        if commit:
            self.conn.commit()

    def query_features(self, group_name=None, is_ml=None, sensor=None):
        """Query registered features."""
        sql = "SELECT * FROM features WHERE 1=1"
        params = []
        if group_name:
            sql += " AND group_name = ?"
            params.append(group_name)
        if is_ml is not None:
            sql += " AND is_ml_feature = ?"
            params.append(1 if is_ml else 0)
        if sensor:
            sql += " AND sensor = ?"
            params.append(sensor)
        sql += " ORDER BY feature_name"
        return self.conn.execute(sql, params).fetchall()

    def register_parquet_columns(self, parquet_name, columns, role_overrides=None):
        """Auto-parse and register all columns from a parquet.
        Uses parse_feature_name() to decompose __ convention names.
        role_overrides: dict {col_name: role} where role is 'feature', 'label', 'metadata', or 'id'.
        """
        if role_overrides is None:
            role_overrides = {}
        rows = []
        for col in columns:
            info = parse_feature_name(col)
            info['parquet_name'] = parquet_name
            if col in role_overrides:
                role = role_overrides[col]
                info['column_role'] = role
                info['is_ml_feature'] = 1 if role == 'feature' else 0
            else:
                info['column_role'] = 'feature' if info.get('is_ml_feature', 1) else 'metadata'
            rows.append(info)
        self.upsert_features_batch(rows)
        return len(rows)

    def get_feature_cols(self, parquet_name=None):
        """Return sorted list of ML feature column names."""
        sql = "SELECT feature_name FROM features WHERE is_ml_feature = 1"
        params = []
        if parquet_name:
            sql += " AND parquet_name = ?"
            params.append(parquet_name)
        sql += " ORDER BY feature_name"
        return [r[0] for r in self.conn.execute(sql, params).fetchall()]

    def get_meta_cols(self, parquet_name=None):
        """Return sorted list of meta column names."""
        sql = "SELECT feature_name FROM features WHERE is_ml_feature = 0"
        params = []
        if parquet_name:
            sql += " AND parquet_name = ?"
            params.append(parquet_name)
        sql += " ORDER BY feature_name"
        return [r[0] for r in self.conn.execute(sql, params).fetchall()]

    def get_label_cols(self, parquet_name=None):
        """Return sorted list of label column names (target variables)."""
        sql = "SELECT feature_name FROM features WHERE column_role = 'label'"
        params = []
        if parquet_name:
            sql += " AND parquet_name = ?"
            params.append(parquet_name)
        sql += " ORDER BY feature_name"
        return [r[0] for r in self.conn.execute(sql, params).fetchall()]

    def get_id_cols(self, parquet_name=None):
        """Return sorted list of ID column names (join keys)."""
        sql = "SELECT feature_name FROM features WHERE column_role = 'id'"
        params = []
        if parquet_name:
            sql += " AND parquet_name = ?"
            params.append(parquet_name)
        sql += " ORDER BY feature_name"
        return [r[0] for r in self.conn.execute(sql, params).fetchall()]

    def get_feature_groups(self, parquet_name=None):
        """Return dict of {group_name: [col_names]} for ML features."""
        sql = "SELECT feature_name, group_name FROM features WHERE is_ml_feature = 1"
        params = []
        if parquet_name:
            sql += " AND parquet_name = ?"
            params.append(parquet_name)
        rows = self.conn.execute(sql, params).fetchall()
        groups = {}
        for r in rows:
            groups.setdefault(r['group_name'], []).append(r['feature_name'])
        return {k: sorted(v) for k, v in sorted(groups.items())}

    def get_features_by_measurement(self, measurement, parquet_name=None):
        """Return feature names matching a measurement (landuse, coh, vv, ndvi, etc.)."""
        sql = "SELECT feature_name FROM features WHERE measurement = ? AND is_ml_feature = 1"
        params = [measurement]
        if parquet_name:
            sql += " AND parquet_name = ?"
            params.append(parquet_name)
        sql += " ORDER BY feature_name"
        return [r[0] for r in self.conn.execute(sql, params).fetchall()]

    def get_features_by_group(self, group_name, parquet_name=None):
        """Return feature names in a group (coh, card, ms_bands, ms_indices, landuse, fire)."""
        sql = "SELECT feature_name FROM features WHERE group_name = ? AND is_ml_feature = 1"
        params = [group_name]
        if parquet_name:
            sql += " AND parquet_name = ?"
            params.append(parquet_name)
        sql += " ORDER BY feature_name"
        return [r[0] for r in self.conn.execute(sql, params).fetchall()]

    def get_battle_date(self, city_name):
        """Return battle_start string for a city, or None."""
        row = self.conn.execute(
            "SELECT battle_start FROM cities WHERE city_name = ?", [city_name]
        ).fetchone()
        return row['battle_start'] if row else None

    # =====================================================================
    # TRACKER JSON IMPORT (one-time migration)
    # =====================================================================

    def import_tracker_json(self, tracker_path, notebook, sensor):
        """Import an existing NB02/NB03 tracker JSON into scenes/processing_runs.
        Generic importer that handles common tracker formats.

        Args:
            tracker_path: Path to tracker JSON
            notebook:     source notebook name (NB02a, NB03a, etc.)
            sensor:       CARD, SLC, MS, COH
        """
        tracker_path = Path(tracker_path)
        if not tracker_path.exists():
            print(f"  Tracker not found: {tracker_path}")
            return 0

        with open(tracker_path) as f:
            tracker = json.load(f)

        n = 0
        # common formats: dict with city keys, or list of entries
        if isinstance(tracker, dict):
            for key, entry in tracker.items():
                if isinstance(entry, dict):
                    city = entry.get('city', entry.get('city_name', key))
                    status = entry.get('status', entry.get('download_status', 'unknown'))
                    date_str = entry.get('date', entry.get('date_str', ''))
                    filename = entry.get('filename', entry.get('file', key))

                    scene_id = f"{city}_{sensor}_{date_str}" if date_str else f"{city}_{sensor}_{key}"
                    self.upsert_scene(
                        city_name=city,
                        sensor=sensor,
                        scene_id=scene_id,
                        download_status=str(status),
                        date_str=str(date_str)[:8] if date_str else None,
                        zip_filename=str(filename),
                        updated_by=f"import_{notebook}",
                    )
                    n += 1
                elif isinstance(entry, str):
                    # simple key: status mapping
                    self.upsert_scene(
                        city_name=key,
                        sensor=sensor,
                        scene_id=f"{key}_{sensor}",
                        download_status=str(entry),
                        updated_by=f"import_{notebook}",
                    )
                    n += 1

        print(f"  Imported {n} entries from {tracker_path} -> scenes (sensor={sensor})")
        return n

    # =====================================================================
    # CONVENIENCE QUERIES
    # =====================================================================

    def summary(self):
        """Print summary of all tables."""
        print("=" * 70)
        print("BDA CATALOG SUMMARY")
        print("=" * 70)
        print(f"  DB: {self.db_path}")

        for table in ['cities', 'scenes', 'products', 'processing_runs', 'data_stack', 'features', 'rename_lut']:
            try:
                n = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                n = '(table missing)'
            print(f"  {table:<20s}: {n}")

        # city tier breakdown
        rows = self.conn.execute(
            "SELECT tier, COUNT(*), SUM(has_unosat) FROM cities GROUP BY tier ORDER BY tier"
        ).fetchall()
        if rows:
            print(f"\n  Tier breakdown:")
            for row in rows:
                print(f"    T{row[0]}: {row[1]} cities ({row[2]} with UNOSAT)")

        # scenes per sensor
        rows = self.conn.execute(
            "SELECT sensor, download_status, COUNT(*) FROM scenes GROUP BY sensor, download_status ORDER BY sensor"
        ).fetchall()
        if rows:
            print(f"\n  Scenes per sensor:")
            for row in rows:
                print(f"    {row[0]:<10s} {row[1]:<12s}: {row[2]}")

        # products per type
        rows = self.conn.execute(
            "SELECT product_type, COUNT(*) FROM products GROUP BY product_type ORDER BY product_type"
        ).fetchall()
        if rows:
            print(f"\n  Products per type:")
            for row in rows:
                print(f"    {row[0]:<25s}: {row[1]}")

        print("=" * 70)

    def cities_missing_sensor(self, sensor, tier=None):
        """Find cities that have no scenes for a given sensor."""
        sql = """
            SELECT c.city_name, c.tier FROM cities c
            WHERE c.city_name NOT IN (
                SELECT DISTINCT city_name FROM scenes WHERE sensor = ?
            )
        """
        params = [sensor]
        if tier is not None:
            if isinstance(tier, (list, tuple)):
                sql += f" AND c.tier IN ({','.join(['?']*len(tier))})"
                params.extend(tier)
            else:
                sql += " AND c.tier = ?"
                params.append(tier)
        sql += " ORDER BY c.tier, c.city_name"
        return self.conn.execute(sql, params).fetchall()

    def cities_product_coverage(self):
        """Per-city product type counts. Returns list of dicts."""
        sql = """
            SELECT city_name,
                   SUM(CASE WHEN sensor='CARD' AND product_type='flat' THEN 1 ELSE 0 END) as card_flat,
                   SUM(CASE WHEN sensor='CARD' AND product_type='temporal_stats' THEN 1 ELSE 0 END) as card_stats,
                   SUM(CASE WHEN sensor='COH' AND product_type='flat' THEN 1 ELSE 0 END) as coh_flat,
                   SUM(CASE WHEN sensor='COH' AND product_type='coherence_baseline' THEN 1 ELSE 0 END) as coh_bl,
                   SUM(CASE WHEN sensor='MS' AND product_type='flat' THEN 1 ELSE 0 END) as ms_flat,
                   SUM(CASE WHEN sensor='MS' AND product_type='composite' THEN 1 ELSE 0 END) as ms_comp
            FROM products
            GROUP BY city_name
            ORDER BY city_name
        """
        return self.conn.execute(sql).fetchall()

    def raw_sql(self, sql, params=None):
        """Execute raw SQL. For ad-hoc queries."""
        if params:
            return self.conn.execute(sql, params).fetchall()
        return self.conn.execute(sql).fetchall()
