# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
scene_download.py
Scene download functions for Copernicus and ASF.
Extracted from Cell 14B2.

Notebook usage:
    from scene_download import init as init_scene_download
    init_scene_download(
        temp_download_dir=TEMP_DOWNLOAD_DIR,
        raw_slc_zip=RAW_SLC_ZIP,
        data_root=DATA_ROOT,
        copernicus_username=COPERNICUS_USERNAME,
        copernicus_password=COPERNICUS_PASSWORD,
    )

    from scene_download import download_scene, download_scene_pair, download_scene_chain, ...
"""

import json
import os
import requests
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Module globals - set by init()
# ---------------------------------------------------------------------------
_TEMP_DOWNLOAD_DIR = None
_RAW_SLC_ZIP = None
_DATA_ROOT = None
_COPERNICUS_USERNAME = None
_COPERNICUS_PASSWORD = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def get_copernicus_token():
    """Get OAuth2 access token for Copernicus Data Space"""
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "username": _COPERNICUS_USERNAME,
        "password": _COPERNICUS_PASSWORD,
        "client_id": "cdse-public"
    }
    response = requests.post(token_url, data=data, timeout=60)
    response.raise_for_status()
    return response.json()["access_token"]


def lookup_copernicus_id(scene_name):
    """Query Copernicus to get product UUID from scene name"""
    # Clean scene name - remove .SAFE if present
    if scene_name.endswith('.SAFE'):
        scene_name = scene_name[:-5]

    catalog_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    # Build filter for exact name match
    filter_parts = [
        f"contains(Name,'{scene_name}')",
        "Collection/Name eq 'SENTINEL-1'"
    ]

    params = {
        "$filter": " and ".join(filter_parts),
        "$top": 1
    }

    try:
        response = requests.get(catalog_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get('value') and len(data['value']) > 0:
            product = data['value'][0]
            product_id = product.get('Id')
            print(f"    Found Copernicus ID: {product_id[:8]}...")
            return product_id
        else:
            print(f"    Scene not found in Copernicus catalog")
            return None
    except Exception as e:
        print(f"    Copernicus lookup failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Copernicus download
# ---------------------------------------------------------------------------
def download_scene_copernicus(scene_id, scene_name, output_dir, extract=True):
    """Download a scene from Copernicus Data Space.
    Checks RAW_SLC_ZIP (local HDD) first before downloading.

    Args:
        extract: If True (default), extract zip to .SAFE and return .SAFE path.
                 If False, just download/validate zip and return zip path.
                 Use extract=False in NB02 (download-only), True in NB03A.
    """
    print(f"  Downloading from Copernicus: {scene_name[:50]}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Handle scene_name that may already include .SAFE suffix
    if scene_name.endswith('.SAFE'):
        safe_dir = output_dir / scene_name
        zip_file = output_dir / f"{scene_name}.zip"
        base_name = scene_name[:-5]
    else:
        safe_dir = output_dir / f"{scene_name}.SAFE"
        zip_file = output_dir / f"{scene_name}.zip"
        base_name = scene_name

    if extract and safe_dir.exists():
        print(f"    Already extracted: {safe_dir.name}")
        return safe_dir

    # Check local HDD zip storage first (RAW_SLC_ZIP)
    local_zip = _RAW_SLC_ZIP / f"{base_name}.zip"
    if local_zip.exists():
        # Validate existing zip
        if not zipfile.is_zipfile(local_zip):
            corrupt_size = local_zip.stat().st_size / 1e9
            print(f"    Corrupt zip ({corrupt_size:.2f} GB) - deleting: {local_zip.name}")
            local_zip.unlink()
        else:
            print(f"    Found local zip: {local_zip.name} ({local_zip.stat().st_size/1e9:.2f} GB)")
            zip_file = local_zip  # Use local zip directly, skip download
            if not extract:
                return zip_file

    if not zip_file.exists() or not zipfile.is_zipfile(zip_file):
        # Download to RAW_SLC_ZIP
        download_target = _RAW_SLC_ZIP / f"{base_name}.zip"
        if download_target.exists() and not zipfile.is_zipfile(download_target):
            download_target.unlink()
        if not download_target.exists():
            print(f"    Downloading to: {download_target}")
            token = get_copernicus_token()
            download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({scene_id})/$value"
            headers = {"Authorization": f"Bearer {token}"}

            response = requests.get(download_url, headers=headers, stream=True, timeout=600)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(download_target, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r    Progress: {progress:.1f}% ({downloaded/1e9:.2f}/{total_size/1e9:.2f} GB)", end='')

            print(f"\n    Downloaded: {download_target.name} ({total_size/1e9:.2f} GB)")

            # Validate downloaded zip
            if not zipfile.is_zipfile(download_target):
                corrupt_size = download_target.stat().st_size / 1e9
                print(f"    Downloaded file is corrupt ({corrupt_size:.2f} GB) - deleting")
                download_target.unlink()
                raise ValueError(f"Downloaded corrupt zip for {scene_name}")

        zip_file = download_target

    if not extract:
        print(f"    Zip OK: {zip_file.name} ({zip_file.stat().st_size/1e9:.2f} GB)")
        return zip_file

    # Extract only when extract=True (NB03A usage)
    print(f"    Extracting...")
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(output_dir)

    if safe_dir.exists():
        print(f"    Extracted: {safe_dir.name}")
        return safe_dir

    for extracted in output_dir.glob("*.SAFE"):
        if base_name in extracted.name:
            print(f"    Extracted: {extracted.name}")
            return extracted

    raise ValueError(f"Extraction failed - no .SAFE directory found for {scene_name}")


# ---------------------------------------------------------------------------
# ASF download
# ---------------------------------------------------------------------------
def download_scene_asf(scene_name, output_dir, earthdata_user=None, earthdata_pass=None, extract=True):
    """
    Download a scene from ASF (Alaska Satellite Facility) / NASA Earthdata.

    Args:
        extract: If True (default), extract zip to .SAFE and return .SAFE path.
                 If False, just download/validate zip and return zip path.
    """
    try:
        import asf_search as asf
    except ImportError:
        print("    Installing asf_search...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "asf_search"])
        import asf_search as asf

    print(f"  Downloading from ASF: {scene_name[:50]}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Handle scene_name that may already include .SAFE suffix
    if scene_name.endswith('.SAFE'):
        safe_dir = output_dir / scene_name
        search_name = scene_name[:-5]
    else:
        safe_dir = output_dir / f"{scene_name}.SAFE"
        search_name = scene_name

    base_name = search_name

    if extract and safe_dir.exists():
        print(f"    Already exists: {safe_dir.name}")
        return safe_dir

    # Check if zip already on disk and valid
    zip_file = output_dir / f"{base_name}.zip"
    if zip_file.exists():
        if zipfile.is_zipfile(zip_file):
            print(f"    Zip exists and valid: {zip_file.name} ({zip_file.stat().st_size/1e9:.2f} GB)")
            if not extract:
                return zip_file
        else:
            corrupt_size = zip_file.stat().st_size / 1e9
            print(f"    Corrupt zip ({corrupt_size:.2f} GB) - deleting: {zip_file.name}")
            zip_file.unlink()

    if earthdata_user is None:
        earthdata_user = os.environ.get('EARTHDATA_USER', '')
        earthdata_pass = os.environ.get('EARTHDATA_PASS', '')
    if not earthdata_user or not earthdata_pass:
        raise ValueError("ASF download requires NASA Earthdata credentials (EARTHDATA_USER, EARTHDATA_PASS)")

    session = asf.ASFSession().auth_with_creds(earthdata_user, earthdata_pass)

    results = asf.granule_search([search_name])

    if not results:
        print(f"    Scene not found in ASF: {search_name}")
        return None

    result = results[0]
    print(f"    Found in ASF: {result.properties['sceneName']}")
    print(f"    Size: {result.properties.get('bytes', 0) / 1e9:.2f} GB")

    result.download(path=str(output_dir), session=session)

    # Validate downloaded zip
    zip_file = output_dir / f"{search_name}.zip"
    if zip_file.exists() and not zipfile.is_zipfile(zip_file):
        corrupt_size = zip_file.stat().st_size / 1e9
        print(f"    Downloaded corrupt zip ({corrupt_size:.2f} GB) - deleting")
        zip_file.unlink()
        raise ValueError(f"ASF downloaded corrupt zip for {scene_name}")

    if not extract:
        if zip_file.exists():
            print(f"    Zip OK: {zip_file.name} ({zip_file.stat().st_size/1e9:.2f} GB)")
            return zip_file
        raise ValueError(f"ASF zip not found after download for {scene_name}")

    # Extract only when extract=True (NB03A usage)
    if zip_file.exists():
        print(f"    Extracting...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(output_dir)

    if safe_dir.exists():
        print(f"    Extracted: {safe_dir.name}")
        return safe_dir

    for extracted in output_dir.glob("*.SAFE"):
        if base_name in extracted.name:
            print(f"    Extracted: {extracted.name}")
            return extracted

    raise ValueError(f"ASF download/extraction failed for {scene_name}")


# ---------------------------------------------------------------------------
# Combined download
# ---------------------------------------------------------------------------
def download_scene(scene_metadata, output_dir=None, prefer_copernicus=True):
    """
    Download a scene - ALWAYS tries Copernicus first, then ASF as fallback.

    Args:
        scene_metadata: Dict with 'id', 'name', 'source' keys
        output_dir: Where to download (default: TEMP_DOWNLOAD_DIR)
        prefer_copernicus: If True (default), always try Copernicus first

    Returns:
        Path to .SAFE directory
    """
    if output_dir is None:
        output_dir = _TEMP_DOWNLOAD_DIR

    output_dir = Path(output_dir)

    scene_id = scene_metadata.get('id')
    scene_name = scene_metadata.get('name')
    scene_source = scene_metadata.get('source', 'copernicus')

    # Handle scene_name that may already include .SAFE suffix
    if scene_name.endswith('.SAFE'):
        safe_dir = output_dir / scene_name
    else:
        safe_dir = output_dir / f"{scene_name}.SAFE"

    if safe_dir.exists():
        print(f"  Scene already exists: {safe_dir.name}")
        return safe_dir

    # Determine download order - Copernicus first by default
    if prefer_copernicus:
        sources_to_try = ['copernicus', 'asf']
    else:
        # Use discovery source first, then fallback
        if scene_source == 'asf':
            sources_to_try = ['asf', 'copernicus']
        else:
            sources_to_try = ['copernicus', 'asf']

    # If scene was discovered via ASF, we need to lookup Copernicus ID
    if prefer_copernicus and scene_source == 'asf':
        print(f"    Scene discovered via ASF, looking up Copernicus ID...")
        copernicus_id = lookup_copernicus_id(scene_name)
        if copernicus_id:
            scene_id = copernicus_id  # Use the looked-up UUID

    last_error = None

    for try_source in sources_to_try:
        try:
            if try_source == 'copernicus':
                if scene_id:
                    return download_scene_copernicus(scene_id, scene_name, output_dir)
                else:
                    print(f"    No Copernicus ID available, skipping Copernicus...")
                    continue
            elif try_source == 'asf':
                result = download_scene_asf(scene_name, output_dir)
                if result:
                    return result
                else:
                    raise ValueError(f"ASF returned None for {scene_name}")
        except Exception as e:
            last_error = e
            print(f"    {try_source.capitalize()} download failed: {e}")
            if try_source != sources_to_try[-1]:
                print(f"    Trying next source...")
            continue

    # All sources failed
    raise ValueError(f"All download sources failed for {scene_name}. Last error: {last_error}")


def download_scene_pair(scene1, scene2, output_dir=None, prefer_copernicus=True):
    """
    Download a pair of scenes.

    Returns:
        (path1, path2) tuple of .SAFE directories
    """
    if output_dir is None:
        output_dir = _TEMP_DOWNLOAD_DIR

    print(f"\n  Downloading scene pair to {output_dir}...")

    path1 = download_scene(scene1, output_dir, prefer_copernicus=prefer_copernicus)
    path2 = download_scene(scene2, output_dir, prefer_copernicus=prefer_copernicus)

    print(f"\n  Both scenes downloaded:")
    if path1:
        print(f"    Scene 1: {path1.name}")
    if path2:
        print(f"    Scene 2: {path2.name}")

    return path1, path2


def download_orbit_files(scene_paths, orbit_source_dir=None, orbit_dest_dir=None):
    """
    Copy and extract orbit files for downloaded scenes.
    """
    if orbit_source_dir is None:
        orbit_source_dir = _DATA_ROOT / 'satellite' / 'sentinel_1_orbits'

    if orbit_dest_dir is None:
        orbit_dest_dir = _TEMP_DOWNLOAD_DIR / 'orbitfiles'

    orbit_source_dir = Path(orbit_source_dir)
    orbit_dest_dir = Path(orbit_dest_dir)
    orbit_dest_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n  Processing orbit files...")

    for safe_path in scene_paths:
        safe_path = Path(safe_path)
        safe_name = safe_path.name

        parts = safe_name.split('_')
        satellite = parts[0]

        acq_time_str = None
        for part in parts:
            if len(part) == 15 and 'T' in part:
                acq_time_str = part
                break

        if not acq_time_str:
            print(f"    WARNING: Could not parse acquisition time from {safe_name}")
            continue

        acq_time = datetime.strptime(acq_time_str, '%Y%m%dT%H%M%S')

         # Search in specific subdirectory: satellite/year/month
        narrow_dir = orbit_source_dir / satellite / str(acq_time.year) / f'{acq_time.month:02d}'
        if narrow_dir.exists():
            orbit_files = list(narrow_dir.glob(f'{satellite}_OPER_AUX_POEORB_*.EOF.zip'))
        else:
            orbit_files = []

        matching_orbit = None
        for orbit_file in orbit_files:
            orbit_name = orbit_file.name
            try:
                validity_start_str = orbit_name.split('_V')[1].split('_')[0]
                validity_end_str = orbit_name.split('_V')[1].split('_')[1].split('.')[0]

                validity_start = datetime.strptime(validity_start_str, '%Y%m%dT%H%M%S')
                validity_end = datetime.strptime(validity_end_str, '%Y%m%dT%H%M%S')

                if validity_start <= acq_time <= validity_end:
                    matching_orbit = orbit_file
                    break
            except:
                continue

        if matching_orbit:
            eof_name = matching_orbit.name.replace('.zip', '')
            eof_path = orbit_dest_dir / eof_name

            if not eof_path.exists():
                zip_dest = orbit_dest_dir / matching_orbit.name
                shutil.copy2(matching_orbit, zip_dest)

                with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
                    zip_ref.extractall(orbit_dest_dir)

                zip_dest.unlink()
                print(f"    Extracted orbit for {safe_name[:30]}...")
            else:
                print(f"    Orbit exists for {safe_name[:30]}...")
        else:
            print(f"    WARNING: No matching orbit for {safe_name}")

    print(f"  Orbit files ready in {orbit_dest_dir}")
    return orbit_dest_dir


def cleanup_temp_scenes(scene_names=None, keep_safe=False):
    """
    Clean up downloaded scenes from temp directory.
    """
    print(f"\n  Cleaning up temp directory...")

    if scene_names is None:
        for item in _TEMP_DOWNLOAD_DIR.iterdir():
            try:
                if item.suffix == '.zip':
                    item.unlink()
                    print(f"    Deleted: {item.name}")
                elif item.is_dir() and item.name.endswith('.SAFE'):
                    if not keep_safe:
                        shutil.rmtree(item)
                        print(f"    Deleted: {item.name}")
            except Exception as e:
                print(f"    Warning: Could not delete {item.name}: {e}")
    else:
        for name in scene_names:
            # Handle names that may or may not have .SAFE suffix
            if name.endswith('.SAFE'):
                safe_dir = _TEMP_DOWNLOAD_DIR / name
                zip_file = _TEMP_DOWNLOAD_DIR / f"{name}.zip"
            else:
                safe_dir = _TEMP_DOWNLOAD_DIR / f"{name}.SAFE"
                zip_file = _TEMP_DOWNLOAD_DIR / f"{name}.zip"

            if zip_file.exists():
                zip_file.unlink()
                print(f"    Deleted: {zip_file.name}")

            if not keep_safe and safe_dir.exists():
                shutil.rmtree(safe_dir)
                print(f"    Deleted: {safe_dir.name}")

    print(f"  Cleanup complete")


def download_scene_chain(chain, output_dir=None, prefer_copernicus=True):
    """
    Download all scenes in a biweekly chain.

    Returns:
        List of .SAFE directory paths (None entries for failed downloads)
    """
    if output_dir is None:
        output_dir = _TEMP_DOWNLOAD_DIR

    output_dir = Path(output_dir)

    print(f"\n  Downloading biweekly chain: {len(chain)} scenes to {output_dir}...")

    paths = []
    for i, scene in enumerate(chain, 1):
        print(f"\n  [{i}/{len(chain)}] {scene.get('name', 'unknown')[:50]}...")
        try:
            path = download_scene(scene, output_dir, prefer_copernicus=prefer_copernicus)
            paths.append(path)
        except Exception as e:
            print(f"    FAILED: {e}")
            paths.append(None)

    success = sum(1 for p in paths if p is not None)
    print(f"\n  Chain download complete: {success}/{len(chain)} scenes")

    return paths


# ---------------------------------------------------------------------------
# init() - called from notebook cell to set globals
# ---------------------------------------------------------------------------
def init(temp_download_dir, raw_slc_zip, data_root,
         copernicus_username, copernicus_password):
    """
    Args:
        temp_download_dir:    Path - TEMP_DOWNLOAD_DIR
        raw_slc_zip:          Path - RAW_SLC_ZIP
        data_root:            Path - DATA_ROOT
        copernicus_username:  str  - COPERNICUS_USERNAME
        copernicus_password:  str  - COPERNICUS_PASSWORD
    """
    global _TEMP_DOWNLOAD_DIR, _RAW_SLC_ZIP, _DATA_ROOT
    global _COPERNICUS_USERNAME, _COPERNICUS_PASSWORD

    _TEMP_DOWNLOAD_DIR = Path(temp_download_dir)
    _RAW_SLC_ZIP = Path(raw_slc_zip)
    _DATA_ROOT = Path(data_root)
    _COPERNICUS_USERNAME = copernicus_username
    _COPERNICUS_PASSWORD = copernicus_password

    _TEMP_DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print("CELL 14B2: SCENE DOWNLOAD FUNCTIONS")
    print("=" * 80)

    print("\n  Functions defined:")
    print("    get_copernicus_token()")
    print("    lookup_copernicus_id(scene_name)")
    print("    download_scene_copernicus(scene_id, scene_name, output_dir)")
    print("    download_scene_asf(scene_name, output_dir, earthdata_user, earthdata_pass)")
    print("    download_scene(scene_metadata, output_dir, prefer_copernicus)")
    print("    download_scene_pair(scene1, scene2, output_dir, prefer_copernicus)")
    print("    download_orbit_files(scene_paths, orbit_source_dir, orbit_dest_dir)")
    print("    cleanup_temp_scenes(scene_names, keep_safe)")
    print("    download_scene_chain(chain, output_dir, prefer_copernicus)")

    print(f"\n  Config:")
    print(f"    TEMP_DOWNLOAD_DIR = {_TEMP_DOWNLOAD_DIR}")
    print(f"    RAW_SLC_ZIP = {_RAW_SLC_ZIP}")
    print(f"    Download preference: Copernicus FIRST, ASF fallback")

    print("\n" + "=" * 80)
    print("CELL 14B2 COMPLETE")
    print("=" * 80)
