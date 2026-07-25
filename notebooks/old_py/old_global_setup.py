# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

# @title CELL 4: GLOBAL SETUP
# =============================================================================
# Paste into every notebook. Per-notebook config cell (Cell 3) runs BEFORE this.
# Cell 3 sets: TIER_SELECTION, CITY_SELECTION, REQUIRE_UNOSAT, FORCE_RERUN,
#              DRY_RUN, OUTPUT_SUBDIR (NB09), and any NB-specific params.
# =============================================================================

CELL_ID = "cell_4_global_setup"
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['PROJ_DATA'] = '/home/alpineobotics/miniconda3/envs/bda/share/proj'
os.environ['PROJ_LIB'] = '/home/alpineobotics/miniconda3/envs/bda/share/proj'
import pyproj
pyproj.datadir.set_data_dir('/home/alpineobotics/miniconda3/envs/bda/share/proj')
import sys
import subprocess
import platform
from datetime import datetime

print("=" * 70)
print("BDA GLOBAL SETUP")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version.split()[0]}")
print("=" * 70)

# ============================================================================
# GLOBAL CONFIG (never changes across notebooks)
# ============================================================================
MIN_PRE_SCENES = 3
MIN_POST_SCENES = 3
N_PREBATTLE_BASELINE = 5
DATA_START_DATE = "2021-01-01"
DATA_END_DATE = None
CARD_SEARCH_FALLBACK_DAYS = 0
PRUNE_POST_CUTOFF_MONTH = 4
WAVELENGTH = 0.0555  # Sentinel-1 C-band (meters)

# ============================================================================
# SECTION 1: PATHS (platform-aware)
# Windows: G: = old data/GDrive, F: = NVMe scratch (data_stack, new gdrive)
# WSL:     fstab mounts to /content/*
# ============================================================================
print("\n[1/7] Directory Structure")
print("-" * 70)

from pathlib import Path

if platform.system() == 'Windows':
    GDRIVE_ROOT = Path(r'G:\GoogleDrive\masterthesis')
    DRIVE_F_ROOT = Path(r'F:\PROJECTS\masterthesis\gdrive\masterthesis')
    CONTENT_LOCAL = Path(r'G:\masterthesis_local')
    STACK_DIR = Path(r'F:\PROJECTS\masterthesis\data_stack')
elif os.path.exists('/content/drive'):
    GDRIVE_ROOT = Path('/content/drive/MyDrive/masterthesis')
    DRIVE_F_ROOT = Path('/content/drive_f/masterthesis')
    CONTENT_LOCAL = Path('/content/masterthesis_local')
    STACK_DIR = CONTENT_LOCAL / 'data_stack'
else:
    GDRIVE_ROOT = Path('/mnt/g/GoogleDrive/masterthesis')
    DRIVE_F_ROOT = Path('/mnt/f/PROJECTS/masterthesis/gdrive/masterthesis')
    CONTENT_LOCAL = Path('/mnt/g/masterthesis_local')
    STACK_DIR = Path('/mnt/f/PROJECTS/masterthesis/data_stack')

DRIVE_ROOT = DRIVE_F_ROOT   # switch to DRIVE_F_ROOT when G: GDrive decommissioned old: GDRIVE_ROOT
STACK_ROOT = STACK_DIR
DATA_ROOT = DRIVE_ROOT / 'data'
CITIES_DIR = DATA_ROOT / 'cities'
LOGS_DIR = DATA_ROOT / 'logs'
OUTPUTS_DIR = DATA_ROOT / 'outputs'
SENTINEL1_DIR = DATA_ROOT / 'sentinel1'
SATELLITE_DIR = DATA_ROOT / 'satellite'
SAR_SLC_DIR = SATELLITE_DIR / 'SAR_SLC'
SAR_SLC_BBOX_DIR = SATELLITE_DIR / 'SAR_SLC_bbox'
SAR_SLC_POLYGON_DIR = SATELLITE_DIR / 'SAR_SLC_polygon'
SAR_CARD_DIR = SATELLITE_DIR / 'SAR_CARD'
SAR_CARD_BBOX_DIR = SATELLITE_DIR / 'SAR_CARD_bbox'
SAR_CARD_POLYGON_DIR = SATELLITE_DIR / 'SAR_CARD_polygon'
SAR_METADATA_DIR = SAR_SLC_DIR / 'metadata'
MULTISPECTRAL_ROOT = SATELLITE_DIR / 'multispectral'
MS_METADATA_DIR = MULTISPECTRAL_ROOT / 'metadata'
MS_BASE_DIR = MULTISPECTRAL_ROOT
LANDUSE_DIR = SATELLITE_DIR / 'landuse_classification'
TEMPORAL_ROOT = SATELLITE_DIR / 'temporal_products'
TRANSITION_DIR = SATELLITE_DIR / 'transition_matrix'

TRACKER_FILE = OUTPUTS_DIR / 'insar_processing_tracker.json'

# Cities config (single source of truth for all city metadata)
CITIES_CONFIG_FILE = DATA_ROOT / 'cities_config.json'

UNOSAT_DIR = DATA_ROOT / 'unosat_damage_assessments'
UNOSAT_RAW_DIR = UNOSAT_DIR / 'raw'
UNOSAT_CITIES_DIR = UNOSAT_DIR / 'cities'
UNOSAT_COMPILED_DIR = UNOSAT_DIR / 'compiled'

OSM_ROOT = DATA_ROOT / 'osm' / 'ukraine'
OSM_2022 = OSM_ROOT / '20220101'
OSM_2025 = OSM_ROOT / '20251007'
OSM_OUTPUT_DIR = DATA_ROOT / 'osm_buildings'
OVERTURE_DIR = DATA_ROOT / 'overture_buildings'
OVERTURE_CLIPPED = DATA_ROOT / 'overture_buildings_clipped'
UKR_BOUNDARIES = DATA_ROOT / 'boundaries'
XBD_ROOT = DATA_ROOT / 'xBD'
CITY_BOUNDARIES_FILE = DATA_ROOT / 'bda_city_boundaries.geojson'

MODELS_ROOT = DRIVE_ROOT / 'models'
CHECKPOINTS_DIR = MODELS_ROOT / 'checkpoints'
PRETRAINED_DIR = MODELS_ROOT / 'pretrained'
TRAINED_DIR = MODELS_ROOT / 'trained'

RESULTS_ROOT = DRIVE_ROOT / 'results'
PREDICTIONS_DIR = RESULTS_ROOT / 'predictions'
VALIDATION_DIR = RESULTS_ROOT / 'validation'
VISUALIZATIONS_DIR = RESULTS_ROOT / 'visualizations'

NOTEBOOKS_ROOT = DRIVE_ROOT / 'notebooks'
SCRIPTS_ROOT = DRIVE_ROOT / 'scripts'
CONFIG_ROOT = DRIVE_ROOT / 'config'
BACKUPS_ROOT = DRIVE_ROOT / 'backups'
SNAP_GRAPH_DIR = DRIVE_ROOT / 'snap_graphs'

# Legacy paths
IMAGES_DIR = DATA_ROOT / 'images'
LABELS_DIR = DATA_ROOT / 'labels'
RAW_DATA_DIR = DATA_ROOT / 'raw'
PROCESSED_DATA_DIR = DATA_ROOT / 'processed'
CITIES_DIR_OLD = DATA_ROOT / 'cities_old'

# Local storage (zips, DEM, orbits on G:)
RAW_SLC_ZIP = CONTENT_LOCAL / 'data' / 'raw_slc_zip'
CARD_ZIP_DIR = CONTENT_LOCAL / 'data' / 'card_zip'
MS_ZIP_DIR = CONTENT_LOCAL / 'data' / 'ms_zip'
LOCAL_DEM_DIR = CONTENT_LOCAL / 'data' / 'dem'
LOCAL_ORBITS_DIR = CONTENT_LOCAL / 'data' / 'orbits'
DEM_DIR = LOCAL_DEM_DIR
ORBITS_DIR = LOCAL_ORBITS_DIR

XBD_ZIP_DIR = CONTENT_LOCAL / 'xBD_zip'
XBD_ORIGINAL_DIR = CONTENT_LOCAL / 'xBD_original'
XBD_SOURCE_DIR = XBD_ORIGINAL_DIR
XBD_OUTPUT_DIR = DATA_ROOT / 'xBD_64'

# Native ext4 fast storage
SNAP_WORK_DIR = Path('/content/snap_work')
SNAP_WORKDIR = SNAP_WORK_DIR / 'workdir'
SNAP_WORK_ROOT = SNAP_WORK_DIR
TEMP_DOWNLOAD_DIR = SNAP_WORK_DIR / 'temp'
CARD_TEMP_DIR = SNAP_WORK_DIR / 'card_temp'
HYP3_TEMP_DIR = SNAP_WORK_DIR / 'hyp3_temp'
CROSSBATTLE_STASH_DIR = SNAP_WORKDIR / 'crossbattle_stash'
CONTENT_TEMP = Path('/content/TEMP')


GLOBAL_DEM_PATH = DEM_DIR / 'srtm_all_cities.tif'
MS_TEMP_DIR = SNAP_WORK_DIR / 'ms_temp'

# Progress files (scene discovery, one per product)
SLC_PROGRESS_FILE = OUTPUTS_DIR / 'scene_discovery_progress.json'
PROGRESS_FILE = SLC_PROGRESS_FILE  # legacy alias
MS_PROGRESS_FILE = OUTPUTS_DIR / 'ms_scene_discovery_progress.json'

# Tracking files
INSAR_TRACKER_FILE = OUTPUTS_DIR / 'insar_processing_tracker.json'
MS_TRACKING_FILE = OUTPUTS_DIR / 'ms_download_tracking.json'
MS_DOWNLOAD_TRACKING_FILE = MS_TRACKING_FILE
CARD_TRACKER_FILE = OUTPUTS_DIR / 'card_download_tracker.json'
SLC_PLAN_FILE = OUTPUTS_DIR / 'slc_download_plan.json'
CARD_PLAN_FILE = OUTPUTS_DIR / 'card_download_plan.json'
MS_PLAN_FILE = OUTPUTS_DIR / 'ms_download_plan.json'
CITIES_PKL_FILE = SAR_METADATA_DIR / 'cities_dataframe.pkl'
MS_CITIES_PKL_FILE = MS_METADATA_DIR / 'ms_cities_dataframe.pkl'
BATTLE_DATES_FILE = DATA_ROOT / 'bda_battle_dates.json'

# Dietrich dirs
DIETRICH_LOCAL = CONTENT_LOCAL / 'data' / 'dietrich_2025'
DIETRICH_CARD_ZIP = DIETRICH_LOCAL / 'card_zip'
DIETRICH_MS_ZIP = DIETRICH_LOCAL / 'ms_zip'
DIETRICH_CARD_FULL = DIETRICH_LOCAL / 'card'
DIETRICH_DRIVE = DATA_ROOT / 'gdrive_dietrich2025'
DIETRICH_TRACKER_FILE = OUTPUTS_DIR / 'dietrich_card_tracker.json'
DIETRICH_SCENE_LIST_FILE = OUTPUTS_DIR / 'dietrich_scene_list.json'
EXISTING_CARD_DIR = SAR_CARD_DIR

# Dataset output root (shared across NB05, NB08, NB09, NB10)
DATASET_ROOT = CONTENT_LOCAL / 'nb08_dataset'
NB08_OUTPUT_ROOT = DATASET_ROOT  # legacy alias
DATASET_ROOT.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = DATASET_ROOT / 'bda_dataset_labeled.parquet'
META_PATH = DATASET_ROOT / 'bda_dataset_meta.json'
LUT_PATH = DATASET_ROOT / 'temporal_lut.json'
MODALITY_PATH = DATASET_ROOT / 'modality_availability.json'

# Per-notebook output dir (NB09 sets OUTPUT_SUBDIR in Cell 3)
if 'OUTPUT_SUBDIR' in dir():
    OUTPUT_ROOT = CONTENT_LOCAL / OUTPUT_SUBDIR
    for d in [OUTPUT_ROOT / 'logs', OUTPUT_ROOT / 'models', OUTPUT_ROOT / 'plots',
              OUTPUT_ROOT / 'results']:
        d.mkdir(parents=True, exist_ok=True)

# NB09e extra paths (CITY-dependent, updated by set_city_paths)
PATCH_INDEX_PATH = NB08_OUTPUT_ROOT / "bda_patch_index.parquet"
CHANNEL_CONFIG_PATH = NB08_OUTPUT_ROOT / "bda_patch_channels.json"
SS_INDEX_PATH = NB08_OUTPUT_ROOT / "bda_singlescene_index.parquet"
SS_BANDS = ['B02', 'B03', 'B04', 'B08']
SS_RGB_CH = [0, 1, 2]
XBD_IMG_DIR = GDRIVE_ROOT / "data" / "xBD_S2_64"
BDA_MASK_BASE_DIR = NB08_OUTPUT_ROOT / "bda_patches" / "masks"

# Manifest
MANIFEST_PATH = STACK_DIR / "data_stack_manifest.json"

# Create all directories
_dirs_to_create = [
    LOGS_DIR, OUTPUTS_DIR, SENTINEL1_DIR,
    SAR_SLC_DIR, SAR_SLC_BBOX_DIR, SAR_METADATA_DIR,
    SAR_CARD_DIR, SAR_CARD_BBOX_DIR,
    MULTISPECTRAL_ROOT, MS_METADATA_DIR,
    LANDUSE_DIR, TRANSITION_DIR,
    UNOSAT_DIR, UNOSAT_RAW_DIR, UNOSAT_CITIES_DIR, UNOSAT_COMPILED_DIR,
    OSM_OUTPUT_DIR, OVERTURE_DIR,OVERTURE_CLIPPED, UKR_BOUNDARIES, XBD_ROOT,
    MODELS_ROOT, CHECKPOINTS_DIR, PRETRAINED_DIR, TRAINED_DIR,
    RESULTS_ROOT, PREDICTIONS_DIR, MS_TEMP_DIR, VALIDATION_DIR, VISUALIZATIONS_DIR,
    NOTEBOOKS_ROOT, SCRIPTS_ROOT, CONFIG_ROOT, BACKUPS_ROOT,
    SNAP_WORK_DIR, SNAP_WORKDIR, TEMP_DOWNLOAD_DIR,
    CARD_TEMP_DIR, HYP3_TEMP_DIR, CROSSBATTLE_STASH_DIR,
    SNAP_GRAPH_DIR, BDA_MASK_BASE_DIR,
]
for d in _dirs_to_create:
    d.mkdir(exist_ok=True, parents=True)

for d in [RAW_SLC_ZIP, CARD_ZIP_DIR, MS_ZIP_DIR, LOCAL_DEM_DIR, LOCAL_ORBITS_DIR,
          DIETRICH_CARD_ZIP, DIETRICH_MS_ZIP, DIETRICH_DRIVE, DIETRICH_CARD_FULL]:
    d.mkdir(exist_ok=True, parents=True)

# Verify mounts
if not DATA_ROOT.exists():
    print("  ERROR: GDrive not mounted! Run: sudo mount -a")
else:
    print(f"  GDrive (G:):       {DRIVE_ROOT} OK")
if DRIVE_F_ROOT.exists():
    print(f"  GDrive (F:):       {DRIVE_F_ROOT} OK")
if CONTENT_LOCAL.exists():
    print(f"  Local data (G:):   {CONTENT_LOCAL / 'data'} {'OK' if (CONTENT_LOCAL / 'data').exists() else 'NOT MOUNTED'}")
    print(f"  Data stack (F:):   {STACK_DIR} {'OK' if STACK_DIR.exists() else 'NOT MOUNTED'}")

# ============================================================================
# SECTION 1B: CITY RESOLVER
# ============================================================================
import json as _json
import re as _re

# CITIES CONFIG (battle dates, coords, tiers)
with open(CITIES_CONFIG_FILE) as f:
    cities_config = _json.load(f)
    
# Canonical dicts derived from cities_config.json
# Keys: battle_start, battle_stop (same as cities_config.json and AOI.geojson)
BATTLE_DATES = {name: {'battle_start': info['battle_start'], 'battle_stop': info['battle_stop']}
                for name, info in cities_config.items()}
PRIORITY_CITIES_COORDS = [{'name': name, **info} for name, info in cities_config.items()]
TIER_0_CITIES = [c for c, info in cities_config.items() if info['tier'] == 0]

def _load_aoi_header(aoi_path):
    with open(aoi_path, 'r') as f:
        head = f.read(50000)
    props = {}
    m = _re.search(r'"battle_start"\s*:\s*"([^"]*)"', head)
    if m:
        props['battle_start'] = m.group(1)
    m = _re.search(r'"battle_stop"\s*:\s*"([^"]*)"', head)
    if m:
        props['battle_stop'] = m.group(1)
    m = _re.search(r'"tier"\s*:\s*(\d+)', head)
    if m:
        props['tier'] = int(m.group(1))
    return props

def resolve_cities(tier_selection=None, city_selection=None, require_unosat=False):
    """Resolve cities from AOI.geojson. Returns (city_list, battle_dates_dict)."""
    battle_dates = {}
    if city_selection:
        candidates = sorted(city_selection) if isinstance(city_selection, list) else [city_selection]
        for city in candidates:
            aoi_file = CITIES_DIR / city / 'AOI.geojson'
            if aoi_file.exists():
                try:
                    props = _load_aoi_header(aoi_file)
                    if props.get('battle_start'):
                        battle_dates[city] = {
                            'battle_start': props['battle_start'],
                            'battle_stop': props.get('battle_stop', ''),
                            'tier': props.get('tier'),
                        }
                except Exception:
                    pass
    else:
        candidates = []
        if not CITIES_DIR.exists():
            return candidates, battle_dates
        t_sel = tier_selection if tier_selection is not None else [0, 1, 2]
        for city_dir in sorted(CITIES_DIR.iterdir()):
            if not city_dir.is_dir() or city_dir.name == 'metadata':
                continue
            aoi_file = city_dir / 'AOI.geojson'
            if not aoi_file.exists():
                continue
            try:
                props = _load_aoi_header(aoi_file)
                tier = props.get('tier')
                if tier is not None and tier in t_sel:
                    candidates.append(city_dir.name)
                    if props.get('battle_start'):
                        battle_dates[city_dir.name] = {
                            'battle_start': props['battle_start'],
                            'battle_stop': props.get('battle_stop', ''),
                            'tier': tier,
                        }
            except Exception:
                pass

    if require_unosat and UNOSAT_CITIES_DIR.exists():
        filtered = []
        for city in candidates:
            unosat_file = UNOSAT_CITIES_DIR / city / 'unosat_damage.geojson'
            if unosat_file.exists():
                try:
                    with open(unosat_file) as f:
                        gj = _json.load(f)
                    if len(gj.get('features', [])) > 0:
                        filtered.append(city)
                except Exception:
                    pass
        candidates = filtered
    return candidates, battle_dates

# Resolve using per-notebook config (set before this cell runs)
_tier = TIER_SELECTION if 'TIER_SELECTION' in dir() else [0, 1, 2]
_city = CITY_SELECTION if 'CITY_SELECTION' in dir() else None
_unosat = REQUIRE_UNOSAT if 'REQUIRE_UNOSAT' in dir() else False
CITIES_TO_PROCESS, _resolved_dates = resolve_cities(_tier, _city, _unosat)
BATTLE_DATES.update(_resolved_dates)  # merge AOI.geojson dates into cities_config dates
print(f"\n  TIER_SELECTION: {_tier}")
print(f"  CITY_SELECTION: {_city or 'None (tier filter)'}")
print(f"  REQUIRE_UNOSAT: {_unosat}")
print(f"  CITIES_TO_PROCESS: {len(CITIES_TO_PROCESS)} cities")
if len(CITIES_TO_PROCESS) <= 12:
    for c in CITIES_TO_PROCESS:
        bd = BATTLE_DATES.get(c, {}).get('battle_start', 'N/A')
        print(f"    {c} (battle_start={bd})")

# ============================================================================
# SECTION 1C: set_city_paths — call to switch CITY across all dependent paths
# ============================================================================
CITY = CITIES_TO_PROCESS[0] if CITIES_TO_PROCESS else 'Mariupol'
STACK_DIR_SINGLE_CITY = STACK_DIR / CITY
DIETRICH_CITY_DIR = DIETRICH_DRIVE / CITY
DIETRICH_STATS_DIR = DIETRICH_CITY_DIR / 'temporal_stats'
DIETRICH_FEATURES_DIR = DIETRICH_CITY_DIR / 'features'
UNOSAT_PATH = UNOSAT_CITIES_DIR / CITY / 'unosat_damage.geojson'
OVERTURE_BUILDINGS_PATH = CITIES_DIR / CITY / f"{CITY}_buildings_overture.geojson"
OVERTURE_WITH_DAMAGE_PATH = CITIES_DIR / CITY / f"{CITY}_buildings_overture_with_damage.geojson"
LANDUSE_BASE = LANDUSE_DIR / CITY
PATCH_DIR = NB08_OUTPUT_ROOT / "bda_patches" / CITY
CITY_DIR_STACK = STACK_DIR / CITY
BUILDINGS_GEOJSON = CITY_DIR_STACK / "buildings_overture_with_damage.geojson"
SS_PRE_DIR = NB08_OUTPUT_ROOT / "bda_singlescene_pre" / CITY
SS_POST_DIR = NB08_OUTPUT_ROOT / "bda_singlescene_post" / CITY
SS_MASK_DIR = NB08_OUTPUT_ROOT / "bda_singlescene_masks" / CITY

for d in [DIETRICH_STATS_DIR, DIETRICH_FEATURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def set_city_paths(city_name):
    global CITY, STACK_DIR_SINGLE_CITY, DIETRICH_CITY_DIR, DIETRICH_STATS_DIR
    global DIETRICH_FEATURES_DIR, DIETRICH_SCENE_LIST_FILE
    global UNOSAT_PATH, OVERTURE_BUILDINGS_PATH, OVERTURE_WITH_DAMAGE_PATH
    global LANDUSE_BASE, PATCH_DIR, CITY_DIR_STACK, BUILDINGS_GEOJSON
    global SS_PRE_DIR, SS_POST_DIR, SS_MASK_DIR
    CITY = city_name
    STACK_DIR_SINGLE_CITY = STACK_DIR / city_name
    DIETRICH_CITY_DIR = DIETRICH_DRIVE / city_name
    DIETRICH_STATS_DIR = DIETRICH_CITY_DIR / 'temporal_stats'
    DIETRICH_FEATURES_DIR = DIETRICH_CITY_DIR / 'features'
    DIETRICH_SCENE_LIST_FILE = OUTPUTS_DIR / f'dietrich_scene_list_{city_name}.json'
    UNOSAT_PATH = UNOSAT_CITIES_DIR / city_name / 'unosat_damage.geojson'
    OVERTURE_BUILDINGS_PATH = CITIES_DIR / city_name / f"{city_name}_buildings_overture.geojson"
    OVERTURE_WITH_DAMAGE_PATH = CITIES_DIR / city_name / f"{city_name}_buildings_overture_with_damage.geojson"
    LANDUSE_BASE = LANDUSE_DIR / city_name
    PATCH_DIR = NB08_OUTPUT_ROOT / "bda_patches" / city_name
    CITY_DIR_STACK = STACK_DIR / city_name
    BUILDINGS_GEOJSON = CITY_DIR_STACK / "buildings_overture_with_damage.geojson"
    SS_PRE_DIR = NB08_OUTPUT_ROOT / "bda_singlescene_pre" / city_name
    SS_POST_DIR = NB08_OUTPUT_ROOT / "bda_singlescene_post" / city_name
    SS_MASK_DIR = NB08_OUTPUT_ROOT / "bda_singlescene_masks" / city_name
    for d in [DIETRICH_STATS_DIR, DIETRICH_FEATURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SECTION 2: CREDENTIALS
# ============================================================================
print("\n[2/7] Credentials")
print("-" * 70)

os.environ['COP_U'] = 'info@marconicolasheinzen.com'
os.environ['COP_PW'] = 'Copernicus12345678%'
os.environ['OPENTOPO_KEY'] = '4115d0672595f5d7a11cd047e6dc1ce1'
os.environ['EARTHDATA_USER'] = 'marcoheinzen'
os.environ['EARTHDATA_PASS'] = ')EL!6qwdRA#3!?_'
os.environ['NASA_TOKEN'] = """eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im1hcmNvaGVpbnplbiIsImV4cCI6MTc2OTcxMDM5NywiaWF0IjoxNzY0NTI2Mzk3LCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.nmnB6Xde5BaFgs1JNQQJLdygqP55tTArv74rwpsoJc-5eKt2ZVozXd_ECOaTCsYGMPfvj0bGvJgRoJqm3MbWN25COKGO3381O9o45sytSGFgS9wbTFWwgtds8uZa1j-BgXkSU1ySLJiy4F3zJTPMxTLB7aHbNVKPKidQv_Nw6VYgBELqSVDirFqDFwFemYAydb2obUtNZtvocA14mkFelSNbAIUTi-z9CMlNFrwLhrqj6Yt6uyOPD6v8AiaimSUUL1Nk5eL3TYtGnnvAgSYbboXivF9yZMdnMMT5kk-Th6HmOfzI90l9LqF-S_sa8PfDEfUKfR5emdACnt-e9oUikw"""

COPERNICUS_USERNAME = os.environ.get('COP_U', '')
COPERNICUS_PASSWORD = os.environ.get('COP_PW', '')
OPENTOPO_KEY = os.environ.get('OPENTOPO_KEY', '')
COP_U = os.environ['COP_U']
COP_PW = os.environ['COP_PW']
EARTHDATA_USER = os.environ['EARTHDATA_USER']
EARTHDATA_PASS = os.environ['EARTHDATA_PASS']
NASA_TOKEN = os.environ['NASA_TOKEN']

print(f"  Copernicus: {COPERNICUS_USERNAME[:3]}***")
print(f"  OpenTopography: {'OK' if OPENTOPO_KEY else 'MISSING'}")
print(f"  Earthdata: {EARTHDATA_USER}")

# ============================================================================
# SECTION 3: PYTHON PACKAGES
# ============================================================================
print("\n[3/7] Python Packages")
print("-" * 70)

import importlib
import pkg_resources

stats = {'already': [], 'installed': [], 'failed': []}

def install_pkg(package, import_name=None):
    base = package.split("[")[0]
    name = import_name or base.replace("-", "_")
    try:
        importlib.import_module(name)
        v = pkg_resources.get_distribution(base).version
        stats['already'].append(f"{base} ({v})")
        return True
    except:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            v = pkg_resources.get_distribution(base).version
            stats['installed'].append(f"{base} ({v})")
            print(f"    {base} ({v})")
            return True
        except Exception as e:
            stats['failed'].append(package)
            print(f"    FAILED: {package}")
            return False

install_pkg("numpy")
install_pkg("pandas")
install_pkg("matplotlib")
install_pkg("seaborn")
install_pkg("scipy")
install_pkg("torch")
install_pkg("torchvision")
install_pkg("segmentation-models-pytorch", "segmentation_models_pytorch")
install_pkg("geopandas")
install_pkg("shapely")
install_pkg("pyproj")
install_pkg("rasterio")
install_pkg("rioxarray")
install_pkg("xarray")
install_pkg("netCDF4")
install_pkg("affine")
install_pkg("requests")
install_pkg("tqdm")
install_pkg("folium")
install_pkg("Pillow", "PIL")
install_pkg("psutil")
install_pkg("pyvista")
install_pkg("panel")

print(f"\n  Already installed: {len(stats['already'])}")
print(f"  Newly installed:   {len(stats['installed'])}")
print(f"  Failed:            {len(stats['failed'])}")

# ============================================================================
# SECTION 4: GLOBAL IMPORTS + CONFIGURATION
# ============================================================================
print("\n[4/7] Global Imports & Configuration")
print("-" * 70)

import gc
import json
import time
import shutil
import zipfile
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray
import rasterio
from rasterio import features
from affine import Affine

from shapely.geometry import shape, box, Polygon, MultiPoint, mapping
from shapely.ops import unary_union

import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

print("  All imports loaded")

# ============================================================================
# SECTION 5: PROCESSING CONFIGURATION + SNAP
# ============================================================================
print("\n[5/7] Processing Config & SNAP")
print("-" * 70)

PROCESSING_CONFIG = {
    'scenes_per_period': 2,
    'max_retry_scenes': 8,
    'max_nodata_percent': 15,
    'subswaths': ['IW1', 'IW2', 'IW3'],
    'polarizations': ['VV', 'VH'],
    'resolution': 20,
    'chunk_size': 2048,
}

PREBATTLE_TOLERANCE_DAYS = 90
POSTBATTLE_TOLERANCE_DAYS = 90
MIN_TEMPORAL_BASELINE = 10
MAX_TEMPORAL_BASELINE = 24
MIN_COVERAGE_PCT = 100.0
MIN_VALID_DATA_PCT = 5.0

# MS config
MS_WINTER_MONTHS = [10, 11, 12, 1, 2]
MS_SAR_WINDOW_DAYS = 30
MS_SCENES_PER_PERIOD = 2
MS_BASELINE_N_SCENES = 5
MS_MIN_COVERAGE_PCT = 95.0
MS_CLOUD_STRICT = 3.0
MS_CLOUD_RELAXED = 5.0
MS_CLOUD_MAX = 15.0
MS_BASELINE_MAX_CLOUD = 15.0
MS_DL_MAX_CLOUD = 20
MS_DL_BIWEEKLY_MAX_CLOUD = 20
MS_DL_WINTER_MAX_CLOUD = 8
MS_DL_EXPAND_MAX_DAYS = 60
MS_SHOP_WINDOW_DAYS = 30
MS_SHOP_BIWEEKLY_WINDOW_DAYS = 45
MS_SHOP_WINTER_WINDOW_DAYS = 540
MS_SHOP_MAX_EXPAND_DAYS = 60

# ML params
RANDOM_STATE = 42
N_FOLDS = 5
TARGET_COL = 'damaged'
STAT_NAMES = ['min', 'max', 'mean', 'median', 'std', 'kurtosis', 'skewness']
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 600

# Reference baselines
DIETRICH_PAPER = {
    'auc': 0.813, 'f1': 0.749, 'precision': 0.671, 'recall': 0.846, 'accuracy': 0.803,
}
NB06_D3 = {
    'auc': 0.790, 'f1': 0.627, 'precision': 0.672, 'recall': 0.588,
}
NB07_BEST = {
    'auc': 0.903, 'classifier': 'Stacking_Meta', 'feature_group': 'all_multimodal',
}
NB06_D5_STACKING = {
    'auc': 0.9174, 'classifier': 'Stacking_Meta',
    'feature_group': 'all_multimodal_biweekly', 'note': 'Mariupol CV only',
}

# ML model param dicts
DIETRICH_RF_PARAMS = {
    'n_estimators': 50, 'min_samples_leaf': 3, 'max_leaf_nodes': 10000,
    'random_state': RANDOM_STATE, 'n_jobs': -1, 'class_weight': 'balanced',
}
RF_PARAMS = {
    'n_estimators': 100, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'class_weight': 'balanced',
}
AB_PARAMS = {
    'n_estimators': 100, 'learning_rate': 1.0, 'random_state': RANDOM_STATE,
}
GB_PARAMS = {
    'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': RANDOM_STATE,
}
XGB_PARAMS = {
    'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'eval_metric': 'logloss',
    'use_label_encoder': False, 'random_state': RANDOM_STATE, 'n_jobs': -1,
}
STACKING_RF_PARAMS = {
    'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 3,
    'class_weight': 'balanced', 'random_state': RANDOM_STATE, 'n_jobs': -1,
}
STACKING_GB_PARAMS = {
    'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': RANDOM_STATE,
}
STACKING_SVM_C = 10.0
STACKING_XGB_PARAMS = {
    'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
    'eval_metric': 'logloss', 'random_state': RANDOM_STATE, 'n_jobs': -1,
}
LGBM_PARAMS = {
    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'min_child_samples': 5,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': RANDOM_STATE,
    'n_jobs': -1, 'verbose': -1,
}

# SNAP
GPT_PATH = os.path.expanduser("~/esa-snap/bin/gpt")
GPT_FLAGS = [
    "-x", "-c", "40G", "-q", "4", "-J-Xmx64G", "-J-Xms2G", "-e",
    "-J-Dsnap.log.level=WARNING",
]

if not os.path.exists(GPT_PATH):
    print(f"  WARNING: GPT not found at {GPT_PATH}")
else:
    try:
        result = subprocess.run([GPT_PATH, "-h"], capture_output=True, text=True, timeout=30)
        first_line = result.stdout.strip().split('\n')[0] if result.stdout else 'OK'
        print(f"  GPT: {first_line}")
    except Exception as e:
        print(f"  GPT check failed: {e}")

print(f"  Temporal baseline: {MIN_TEMPORAL_BASELINE}-{MAX_TEMPORAL_BASELINE} days")
print(f"  Wavelength: {WAVELENGTH}")

# ============================================================================
# SECTION 6: UTILITY FUNCTIONS
# ============================================================================

def load_tracker():
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, 'r') as f:
            tracker = json.load(f)
            tracker.setdefault('processing_runs', [])
            tracker.setdefault('cities', {})
            return tracker
    return {'created': datetime.now().isoformat(), 'last_updated': datetime.now().isoformat(),
            'processing_runs': [], 'cities': {}}

def save_tracker(tracker):
    tracker['last_updated'] = datetime.now().isoformat()
    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

def load_dietrich_tracker():
    if DIETRICH_TRACKER_FILE.exists():
        with open(DIETRICH_TRACKER_FILE, 'r') as f:
            return json.load(f)
    return {'created': datetime.now().isoformat(), 'downloads': {}}

def save_dietrich_tracker(tracker):
    tracker['last_updated'] = datetime.now().isoformat()
    with open(DIETRICH_TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

class DualLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

_global_start = time.time()

def print_elapsed(label="", start=None):
    if start:
        print(f"  {label}: {time.time() - start:.1f}s")
    else:
        print(f"  Total: {time.time() - _global_start:.1f}s")

def get_disk_space(path='/content/drive'):
    try:
        stat = shutil.disk_usage(path)
        return {
            'total_gb': stat.total / (1024**3),
            'used_gb': stat.used / (1024**3),
            'free_gb': stat.free / (1024**3),
            'percent_used': (stat.used / stat.total) * 100
        }
    except:
        return None

# ============================================================================
# AOI.geojson LOADERS
# ============================================================================

_aoi_cache = {}

def _load_city_polygon_feature(city_name):
    if city_name in _aoi_cache:
        return _aoi_cache[city_name]
    aoi_file = CITIES_DIR / city_name / "AOI.geojson"
    if not aoi_file.exists():
        raise FileNotFoundError(f"AOI.geojson not found: {aoi_file}")
    with open(aoi_file) as f:
        gj = json.load(f)
    for feat in gj['features']:
        if feat.get('properties', {}).get('feature_type') == 'city_polygon':
            _aoi_cache[city_name] = feat
            return feat
    raise ValueError(f"No city_polygon feature in {aoi_file}")

def load_aoi(city_name):
    feat = _load_city_polygon_feature(city_name)
    props = dict(feat['properties'])
    props['geometry'] = shape(feat['geometry'])
    return pd.Series(props)

def load_aoi_gdf(city_name):
    feat = _load_city_polygon_feature(city_name)
    props = dict(feat['properties'])
    geom = shape(feat['geometry'])
    return gpd.GeoDataFrame([props], geometry=[geom], crs="EPSG:4326")

def load_aoi_bbox(city_name):
    aoi_file = CITIES_DIR / city_name / "AOI.geojson"
    if not aoi_file.exists():
        raise FileNotFoundError(f"AOI.geojson not found: {aoi_file}")
    with open(aoi_file) as f:
        gj = json.load(f)
    for feat in gj['features']:
        if feat.get('properties', {}).get('feature_type') == 'aoi_bbox':
            return shape(feat['geometry'])
    raise ValueError(f"No aoi_bbox feature in {aoi_file}")

def load_buildings(city_name):
    aoi_file = CITIES_DIR / city_name / "AOI.geojson"
    if not aoi_file.exists():
        raise FileNotFoundError(f"AOI.geojson not found: {aoi_file}")
    gdf = gpd.read_file(aoi_file)
    buildings = gdf[gdf['feature_type'] == 'building']
    if buildings.empty:
        bldg_file = CITIES_DIR / city_name / f"{city_name}_buildings_overture_with_damage.geojson"
        if bldg_file.exists():
            buildings = gpd.read_file(bldg_file)
        else:
            bldg_file = CITIES_DIR / city_name / f"{city_name}_buildings_overture.geojson"
            if bldg_file.exists():
                buildings = gpd.read_file(bldg_file)
    return buildings

def load_damage_aoi_geom(city_name):
    aoi_file = STACK_DIR / city_name / "aoi.geojson"
    if not aoi_file.exists():
        return None
    gdf = gpd.read_file(aoi_file)
    return gdf.geometry.unary_union

# ============================================================================
# SECTION 7: GPU CHECK
# ============================================================================
print("\n[6/7] GPU Status")
print("-" * 70)

import torch
if torch.cuda.is_available():
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"    CUDA version: {torch.version.cuda}")
    device = torch.device('cuda')
else:
    print("  Running on CPU")
    device = torch.device('cpu')

# ============================================================================
# SECTION 8: DISK SPACE
# ============================================================================
print("\n[7/7] Disk Space")
print("-" * 70)

for label, path in [("GDrive (G:)", str(GDRIVE_ROOT)),
                     ("GDrive (F:)", str(DRIVE_F_ROOT)),
                     ("Local data", str(CONTENT_LOCAL / 'data')),
                     ("Data stack", str(STACK_DIR)),
                     ("WSL ext4",   '/')]:
    disk = get_disk_space(path)
    if disk:
        print(f"  {label:15s} {disk['used_gb']:.1f}/{disk['total_gb']:.1f} GB ({disk['free_gb']:.1f} GB free)")
        
# Verify data_stack is on F: (not falling through to G:)
import subprocess
_mnt = subprocess.run(['findmnt', '-n', '-o', 'SOURCE', str(STACK_DIR)], 
                      capture_output=True, text=True).stdout.strip()
if _mnt and 'F:' not in _mnt and 'f:' not in _mnt.lower():
    print(f"\n  WARNING: STACK_DIR {STACK_DIR} is NOT on F: drive (mounted from: {_mnt})")
    print(f"  Run: sudo mount -a")
    raise RuntimeError("data_stack not mounted on F: — aborting to prevent writes to G:")
    
# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 70)
print("GLOBAL SETUP COMPLETE")
print("=" * 70)
print(f"  Torch device: {device}")
print(f"  Cities: {len(CITIES_TO_PROCESS)}, CITY={CITY}")
print(f"  Functions: load_aoi(), load_aoi_gdf(), load_aoi_bbox(), load_buildings()")
print(f"  Functions: set_city_paths(), resolve_cities(), load_tracker(), load_dietrich_tracker()")
print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
