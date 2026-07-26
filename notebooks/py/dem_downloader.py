# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
dem_downloader.py
Downloads global SRTM DEM covering all cities. Provides clip function.
Extracted from Cell 14A2-DEM.

Notebook usage:
    from dem_downloader import run as run_dem_download, clip_dem_to_bounds
    run_dem_download(
        cities_dir=CITIES_DIR,
        dem_dir=LOCAL_DEM_DIR,
        opentopo_key=OPENTOPO_KEY,
        force_rerun=FORCE_RERUN,
        dem_buffer=DEM_BUFFER,
    )
    # then use clip_dem_to_bounds(bbox, output_path) in processing cells
"""

import numpy as np
import requests
import rasterio
from pathlib import Path

from aoi_date_extend_loader import load_aoi


# ---------------------------------------------------------------------------
# Module global - set by run()
# ---------------------------------------------------------------------------
_GLOBAL_DEM_PATH = None


def clip_dem_to_bounds(bbox, output_path, buffer=0.1):
    """
    Clip the global DEM to a bounding box.

    Args:
        bbox: tuple (west, south, east, north) or shapely bounds
        output_path: Path for the clipped DEM
        buffer: degrees buffer around bbox (default 0.1)

    Returns:
        Path to clipped DEM
    """
    if _GLOBAL_DEM_PATH is None or not _GLOBAL_DEM_PATH.exists():
        raise ValueError("Global DEM not available. Run dem_downloader.run() first.")

    output_path = Path(output_path)

    west = bbox[0] - buffer
    south = bbox[1] - buffer
    east = bbox[2] + buffer
    north = bbox[3] + buffer

    with rasterio.open(_GLOBAL_DEM_PATH) as src:
        window = rasterio.windows.from_bounds(west, south, east, north, src.transform)
        window = window.round_offsets().round_lengths()
        transform_crop = src.window_transform(window)
        data = src.read(1, window=window)

        with rasterio.open(output_path, 'w', driver='GTiff',
                           height=data.shape[0], width=data.shape[1], count=1,
                           dtype=data.dtype, crs=src.crs, transform=transform_crop,
                           compress='lzw', nodata=src.nodata) as dst:
            dst.write(data, 1)

    print(f"      Clipped DEM: {data.shape[1]}x{data.shape[0]} -> {output_path.name}")
    return output_path


def run(cities_dir, dem_dir, opentopo_key, force_rerun=False, dem_buffer=0.15):
    """
    Args:
        cities_dir:    Path - CITIES_DIR
        dem_dir:       Path - LOCAL_DEM_DIR
        opentopo_key:  str  - OPENTOPO_KEY
        force_rerun:   bool
        dem_buffer:    float - degrees buffer around combined bbox (default 0.15)
    """
    global _GLOBAL_DEM_PATH

    cities_dir = Path(cities_dir)
    dem_dir = Path(dem_dir)
    dem_dir.mkdir(parents=True, exist_ok=True)
    _GLOBAL_DEM_PATH = dem_dir / 'srtm_all_cities.tif'

    print("\n" + "=" * 80)
    print("CELL 14A2-DEM: DOWNLOAD GLOBAL DEM (RUN ONCE)")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Compute combined bbox from all city boundaries
    # =========================================================================

    print("\n  Step 1: Computing combined bbox from all city boundaries...")

    all_bounds = []
    city_count = 0

    for city_dir in sorted(cities_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        aoi_file = city_dir / 'AOI.geojson'
        if not aoi_file.exists():
            continue
        try:
            aoi_row = load_aoi(city_dir.name, cities_dir)
            bounds = np.array(aoi_row.geometry.bounds)
            all_bounds.append(bounds)
            city_count += 1
        except Exception as e:
            print(f"    WARNING: Could not read {city_dir.name}: {e}")

    if not all_bounds:
        raise ValueError("No city boundaries found!")

    all_bounds = np.array(all_bounds)
    global_bbox = {
        'west': float(all_bounds[:, 0].min()) - dem_buffer,
        'south': float(all_bounds[:, 1].min()) - dem_buffer,
        'east': float(all_bounds[:, 2].max()) + dem_buffer,
        'north': float(all_bounds[:, 3].max()) + dem_buffer,
    }

    print(f"    Cities: {city_count}")
    print(f"    Combined bbox: [{global_bbox['west']:.2f}, {global_bbox['south']:.2f}, {global_bbox['east']:.2f}, {global_bbox['north']:.2f}]")
    approx_width = int((global_bbox['east'] - global_bbox['west']) / 0.000277778)
    approx_height = int((global_bbox['north'] - global_bbox['south']) / 0.000277778)
    print(f"    Approx DEM size: {approx_width} x {approx_height} pixels")

    # =========================================================================
    # STEP 2: Download DEM (tiled if bbox exceeds 450k km2 limit)
    # =========================================================================

    print("\n  Step 2: Downloading SRTM DEM...")

    if _GLOBAL_DEM_PATH.exists() and not force_rerun:
        with rasterio.open(_GLOBAL_DEM_PATH) as src:
            print(f"    DEM already exists: {_GLOBAL_DEM_PATH.name}")
            print(f"    Size: {src.width} x {src.height}")
            print(f"    Bounds: [{src.bounds.left:.2f}, {src.bounds.bottom:.2f}, {src.bounds.right:.2f}, {src.bounds.top:.2f}]")
    else:
        # OpenTopography limit: 450,000 km2 per request
        lon_range = global_bbox['east'] - global_bbox['west']
        lat_range = global_bbox['north'] - global_bbox['south']
        avg_lat = (global_bbox['north'] + global_bbox['south']) / 2
        km_per_deg_lon = 111.32 * np.cos(np.radians(avg_lat))
        km_per_deg_lat = 111.32
        total_area_km2 = (lon_range * km_per_deg_lon) * (lat_range * km_per_deg_lat)
        print(f"    Total area: {total_area_km2:.0f} km2")

        max_area = 430000
        n_strips = int(np.ceil(total_area_km2 / max_area))
        strip_width = lon_range / n_strips
        print(f"    Splitting into {n_strips} longitude strips ({strip_width:.1f} deg each)")

        tile_paths = []
        for i in range(n_strips):
            strip_west = global_bbox['west'] + i * strip_width
            strip_east = global_bbox['west'] + (i + 1) * strip_width
            tile_path = dem_dir / f'srtm_tile_{i}.tif'
            tile_paths.append(tile_path)

            if tile_path.exists() and not force_rerun:
                print(f"    Tile {i+1}/{n_strips} exists: [{strip_west:.2f}, {strip_east:.2f}]")
                continue

            print(f"    Downloading tile {i+1}/{n_strips}: [{strip_west:.2f}, {strip_east:.2f}]...")
            params = {
                'demtype': 'SRTMGL1',
                'south': global_bbox['south'],
                'north': global_bbox['north'],
                'west': strip_west,
                'east': strip_east,
                'outputFormat': 'GTiff',
                'API_Key': opentopo_key
            }
            response = requests.get("https://portal.opentopography.org/API/globaldem",
                                   params=params, stream=True, timeout=600)
            if response.status_code == 200:
                total_size = 0
                with open(tile_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        total_size += len(chunk)
                print(f"      Downloaded: {total_size / (1024*1024):.1f} MB")
            else:
                raise ValueError(f"Tile {i+1} download failed: {response.status_code} - {response.text[:200]}")

        # Merge tiles into single DEM
        print(f"    Merging {n_strips} tiles...")
        from rasterio.merge import merge
        datasets = [rasterio.open(p) for p in tile_paths]
        mosaic, mosaic_transform = merge(datasets)
        for ds in datasets:
            ds.close()

        with rasterio.open(_GLOBAL_DEM_PATH, 'w', driver='GTiff',
                           height=mosaic.shape[1], width=mosaic.shape[2], count=1,
                           dtype=mosaic.dtype, crs='EPSG:4326', transform=mosaic_transform,
                           compress='lzw', nodata=-32768) as dst:
            dst.write(mosaic)

        with rasterio.open(_GLOBAL_DEM_PATH) as src:
            print(f"    Merged DEM: {src.width} x {src.height}")
            print(f"    Bounds: [{src.bounds.left:.2f}, {src.bounds.bottom:.2f}, {src.bounds.right:.2f}, {src.bounds.top:.2f}]")

        # Clean up tiles
        for p in tile_paths:
            p.unlink(missing_ok=True)
        print(f"    Tiles cleaned up")

    # =========================================================================
    # STEP 3: clip function ready
    # =========================================================================

    print("\n  Step 3: clip_dem_to_bounds() available")
    print("    Usage: clip_dem_to_bounds((west, south, east, north), output_path)")

    print("\n" + "=" * 80)
    print(f"CELL 14A2-DEM COMPLETE - Global DEM: {_GLOBAL_DEM_PATH}")
    print("=" * 80)
