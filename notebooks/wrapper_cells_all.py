# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

################################################################################
# WRAPPER CELLS FOR NB02a / NB02b / NB02c
# Replace existing DL-AUDIT and DL-RECONCILE cells, add DL-PRUNE-ZIPS at end.
################################################################################

# ==============================================================================
# CELL DL-SYNC (already exists in 2a/2b/2c, no change needed)
# ==============================================================================
# Already calls dl_sync.py which now imports product_scan.scan_card_tifs_for_sync()

# ==============================================================================
# CELL DL-AUDIT-ZIPS (replaces old DL-AUDIT in 2a/2b/2c)
# ==============================================================================
# @title CELL DL-AUDIT-ZIPS: ZIP INVENTORY + TRACKER STATUS
import sys, importlib
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import dl_audit_zips
importlib.reload(dl_audit_zips)
from dl_audit_zips import run as run_audit_zips

NB02_AUDIT = run_audit_zips(
    raw_slc_zip=RAW_SLC_ZIP,
    card_zip_dir=CARD_ZIP_DIR,
    ms_zip_dir=MS_ZIP_DIR,
    orbits_dir=ORBITS_DIR,
    outputs_dir=OUTPUTS_DIR,
    insar_tracker_file=INSAR_TRACKER_FILE,
    card_tracker_file=CARD_TRACKER_FILE,
    min_slc_size=MIN_SLC_SIZE,
)

# ==============================================================================
# CELL DL-RECONCILE (updated params: multispectral_root -> ms_zip_dir)
# ==============================================================================
# @title CELL DL-RECONCILE: TRACKER RECONCILER
FIX_MODE = True
BACKUP_BEFORE_FIX = True

import sys, importlib
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import dl_reconcile
importlib.reload(dl_reconcile)
from dl_reconcile import run as run_dl_reconcile

RECONCILE_RESULTS = run_dl_reconcile(
    raw_slc_zip=RAW_SLC_ZIP,
    sar_card_dir=SAR_CARD_DIR,
    ms_zip_dir=MS_ZIP_DIR,
    outputs_dir=OUTPUTS_DIR,
    insar_tracker_file=INSAR_TRACKER_FILE,
    card_tracker_file=CARD_TRACKER_FILE,
    nb02_audit=NB02_AUDIT,
    fix_mode=FIX_MODE,
    backup_before_fix=BACKUP_BEFORE_FIX,
)

# ==============================================================================
# CELL DL-PRUNE-ZIPS (NEW - add at end of 2a/2b/2c)
# ==============================================================================
# @title CELL DL-PRUNE-ZIPS: ZIP PRUNING
DRY_RUN_PRUNE = True

import sys, importlib
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import dl_prune_zips
importlib.reload(dl_prune_zips)
from dl_prune_zips import run as run_prune_zips

PRUNE_RESULTS = run_prune_zips(
    raw_slc_zip=RAW_SLC_ZIP,
    card_zip_dir=CARD_ZIP_DIR,
    data_root=DATA_ROOT,
    card_plan_file=CARD_PLAN_FILE,
    card_tracker_file=CARD_TRACKER_FILE,
    insar_tracker_file=INSAR_TRACKER_FILE,
    dl_sync=DL_SYNC if 'DL_SYNC' in dir() else None,
    nb02_audit=NB02_AUDIT if 'NB02_AUDIT' in dir() else None,
    dry_run=DRY_RUN_PRUNE,
    min_slc_size=MIN_SLC_SIZE,
)


################################################################################
# NB02d STANDALONE: SYNC + AUDIT + PRUNE (no downloads)
# Full notebook: 8 cells
################################################################################

# --- Cell 0: Markdown ---
# # NB02d: Download Sync, Audit & Prune (Standalone)
# Runs DL-SYNC, DL-AUDIT-ZIPS, DL-RECONCILE, DL-PRUNE-ZIPS without downloading.
# Standalone alternative to running full 2a/2b/2c pipelines.

# --- Cell 1: Config ---
# @title CELL 3: NB02d CONFIG
TIER_SELECTION = [0, 1]
CITY_SELECTION = None
DRY_RUN_PRUNE = True
FIX_MODE = True
BACKUP_BEFORE_FIX = True

# --- Cell 2: Setup ---
# @title CELL 4: LOAD GLOBAL SETUP
import platform, os
if platform.system() == 'Windows':
    _setup = r'F:\\PROJECTS\\masterthesis\\gdrive\\masterthesis\\notebooks\\global_setup.py'
elif os.path.exists('/content/drive_f'):
    _setup = '/content/drive_f/masterthesis/notebooks/global_setup.py'
else:
    _setup = '/mnt/f/PROJECTS/masterthesis/gdrive/masterthesis/notebooks/global_setup.py'
with open(_setup) as f:
    exec(f.read())

# --- Cell 3: Load scenes ---
# @title CELL 14A3: LOAD SCENES
import sys, importlib
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))
import scene_loader
importlib.reload(scene_loader)
from scene_loader import run as run_scene_loader
cities_df, sar_filtered = run_scene_loader(
    cities_dir=CITIES_DIR, data_root=DATA_ROOT,
    tier_selection=TIER_SELECTION, city_selection=CITY_SELECTION,
)

# --- Cell 4: Load MS scenes ---
# @title CELL 14A3-MS: LOAD MULTISPECTRAL SCENES
import ms_scene_loader
importlib.reload(ms_scene_loader)
from ms_scene_loader import run as run_ms_scene_loader
cities_df = run_ms_scene_loader(
    cities_df=cities_df, data_root=DATA_ROOT,
)

# --- Cell 5: Scene selection + filter ---
# @title CELL 14B1 + 14A-FILTER: SCENE SELECTION + TIER FILTER
import scene_selection
importlib.reload(scene_selection)
from scene_selection import select_city_scenes

import tier_filter
importlib.reload(tier_filter)
from tier_filter import run as run_tier_filter
cities_df_filtered = run_tier_filter(
    cities_df=cities_df,
    tier_selection=TIER_SELECTION,
    city_selection=CITY_SELECTION,
)

# --- Cell 6: DL-SYNC ---
# @title CELL DL-SYNC: PRE-DOWNLOAD SYNC
import dl_sync
importlib.reload(dl_sync)
from dl_sync import run as run_dl_sync
DL_SYNC = run_dl_sync(
    cities_df=cities_df,
    sar_filtered=cities_df_filtered,
    raw_slc_zip=RAW_SLC_ZIP,
    card_zip_dir=CARD_ZIP_DIR,
    ms_zip_dir=MS_ZIP_DIR,
    sar_card_dir=SAR_CARD_DIR,
    outputs_dir=OUTPUTS_DIR,
    card_tracker_file=CARD_TRACKER_FILE,
    slc_plan_file=SLC_PLAN_FILE,
    card_plan_file=CARD_PLAN_FILE,
    n_prebattle_baseline=N_PREBATTLE_BASELINE,
    short_conflict_threshold_days=SHORT_CONFLICT_THRESHOLD_DAYS,
    min_slc_size=MIN_SLC_SIZE,
    min_card_size=MIN_CARD_SIZE,
)

# --- Cell 7: DL-AUDIT-ZIPS ---
# @title CELL DL-AUDIT-ZIPS: ZIP INVENTORY + TRACKER STATUS
import dl_audit_zips
importlib.reload(dl_audit_zips)
from dl_audit_zips import run as run_audit_zips
NB02_AUDIT = run_audit_zips(
    raw_slc_zip=RAW_SLC_ZIP,
    card_zip_dir=CARD_ZIP_DIR,
    ms_zip_dir=MS_ZIP_DIR,
    orbits_dir=ORBITS_DIR,
    outputs_dir=OUTPUTS_DIR,
    insar_tracker_file=INSAR_TRACKER_FILE,
    card_tracker_file=CARD_TRACKER_FILE,
    min_slc_size=MIN_SLC_SIZE,
)

# --- Cell 8: DL-RECONCILE ---
# @title CELL DL-RECONCILE: TRACKER RECONCILER
import dl_reconcile
importlib.reload(dl_reconcile)
from dl_reconcile import run as run_dl_reconcile
RECONCILE_RESULTS = run_dl_reconcile(
    raw_slc_zip=RAW_SLC_ZIP,
    sar_card_dir=SAR_CARD_DIR,
    ms_zip_dir=MS_ZIP_DIR,
    outputs_dir=OUTPUTS_DIR,
    insar_tracker_file=INSAR_TRACKER_FILE,
    card_tracker_file=CARD_TRACKER_FILE,
    nb02_audit=NB02_AUDIT,
    fix_mode=FIX_MODE,
    backup_before_fix=BACKUP_BEFORE_FIX,
)

# --- Cell 9: DL-PRUNE-ZIPS ---
# @title CELL DL-PRUNE-ZIPS: ZIP PRUNING
import dl_prune_zips
importlib.reload(dl_prune_zips)
from dl_prune_zips import run as run_prune_zips
PRUNE_RESULTS = run_prune_zips(
    raw_slc_zip=RAW_SLC_ZIP,
    card_zip_dir=CARD_ZIP_DIR,
    data_root=DATA_ROOT,
    card_plan_file=CARD_PLAN_FILE,
    card_tracker_file=CARD_TRACKER_FILE,
    insar_tracker_file=INSAR_TRACKER_FILE,
    dl_sync=DL_SYNC,
    nb02_audit=NB02_AUDIT,
    dry_run=DRY_RUN_PRUNE,
    min_slc_size=MIN_SLC_SIZE,
)


################################################################################
# NB04b WRAPPER CELLS (product audit + prune)
# NB04a is now redundant - dl_audit_products.py replaces it
################################################################################

# --- Cell: DL-AUDIT-PRODUCTS (replaces NB04a entirely) ---
# @title CELL AUDIT-PRODUCTS: DERIVED PRODUCT AUDIT
import sys, importlib
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import dl_audit_products
importlib.reload(dl_audit_products)
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

# --- Cell: DL-PRUNE-PRODUCTS ---
# @title CELL PRUNE-PRODUCTS: PRODUCT TIF PRUNING
DRY_RUN_PRODUCTS = True

import dl_prune_products
importlib.reload(dl_prune_products)
from dl_prune_products import run as run_prune_products

PRODUCT_PRUNE = run_prune_products(
    sar_coh_dir=SAR_COH_DIR,
    sar_card_dir=SAR_CARD_DIR,
    ms_dir=MS_DIR,
    landuse_dir=LANDUSE_DIR,
    cities_dir=CITIES_DIR,
    outputs_dir=OUTPUTS_DIR,
    cities_filter=CITIES_TO_PROCESS,
    dry_run=DRY_RUN_PRODUCTS,
)
