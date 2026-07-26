# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""nb05bv3_helpers.py — Point-first parquet helpers (NB05bV3).

bda -- Building Damage Assessment using satellite imagery
Copyright (C) 2024-2026 Marco Heinzen
SPDX-License-Identifier: AGPL-3.0-or-later

Purpose
-------
NB05bV3 produces per-point parquets where the sample unit is one UNOSAT damage
point (or one sampled negative) instead of one Overture building footprint. This
matches the Dietrich et al. 2025 setup (per-point pixel time series) and avoids
the 15m point-to-footprint label-collapse that NB01b imposes for V1/V2.

Functions in this module:
  load_unosat_points_for_city(city, raster_crs)
     -> GeoDataFrame with columns: unosat_id, date, damage, damage_binary,
        ep, lon, lat (raster CRS coords as 'x_utm', 'y_utm'), and 'geometry'

  load_builtup_mask(city, stack_root)
     -> 2-D bool array, shape (H, W) of reference grid; True = urban or bare

  build_negative_samples(positive_points, builtup_mask, transform, ratio,
                          min_dist_m, raster_crs, city, seed)
     -> GeoDataFrame of negative points, same columns as positive_points

  sample_raster_at_points(tif_path, points, ref_shape)
     -> 1-D float32 array of length len(points), NaN for out-of-bounds

  sample_raster_block(data_array, points)
     -> 1-D float32 array; pre-loaded raster + cached (row,col) indices

  precompute_point_indices(points, transform, ref_shape)
     -> adds 'row', 'col' int32 columns to points DataFrame

Design notes
------------
Direct array indexing is ~100x faster than rasterio.sample() for N points × M
rasters: read each raster once, compute (row, col) per point once, then index
data[rows, cols]. Out-of-bounds points get NaN.

Negative sampling: the exclusion buffer is computed in pixel space (≥3 pixels
at 10m resolution = 30m). All UNOSAT points (positives, negatives, excluded) are
buffered. Sampled negatives are drawn uniformly from {built-up pixels} \\
{buffered exclusion zone}.

CRS handling
------------
UNOSAT points are stored in EPSG:4326 (WGS84 lon/lat). Reference grid is per-city
UTM (e.g. Mariupol = EPSG:32637). Reprojection is via geopandas .to_crs().
The 'lon', 'lat' columns preserved are always WGS84 for traceability;
'x_utm', 'y_utm' columns are the projected coords in the city raster CRS.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pathlib import Path
from shapely.geometry import Point
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NEG_RATIO = 5
DEFAULT_NEG_MIN_DIST_M = 30.0  # 3 pixels at 10m resolution

# Landuse classes from extract_zonal_categorical / NB03e P5:
#   1=snow, 2=water, 3=dense_veg, 4=sparse_veg, 5=urban, 6=bare, 7=shadow, 8=cloud
LU_BUILTUP_CLASSES = {5, 6}  # urban + bare


# ---------------------------------------------------------------------------
# UNOSAT point loading
# ---------------------------------------------------------------------------

def load_unosat_points_for_city(city, raster_crs, unosat_root,
                                  drop_excluded=True):
    """Load UNOSAT damage points for one city, reprojected to raster CRS.

    Args:
      city: city name (with spaces, e.g. "Chasiv Yar"). Falls back to
            underscore-version on disk if the spaced version is missing.
      raster_crs: target CRS (e.g. "EPSG:32637") — the city's reference grid CRS.
      unosat_root: Path to .../unosat_damage_assessments/cities/
      drop_excluded: if True (default), drop rows with damage_binary == -1
                     (Impact Crater, exclusion class).

    Returns:
      GeoDataFrame with columns:
        unosat_id (int), date (str ISO), damage (int 1-7), damage_binary (int 0/1),
        damage_label (str), ep (int), city (str), point_source (str: 'unosat'),
        lon (float, WGS84), lat (float, WGS84),
        x_utm (float, raster CRS), y_utm (float, raster CRS),
        geometry (Point in raster CRS)
    """
    p = Path(unosat_root) / city / "unosat_damage.geojson"
    if not p.exists():
        # try underscore variant on disk (legacy duplicates)
        p = Path(unosat_root) / city.replace(' ', '_') / "unosat_damage.geojson"
    if not p.exists():
        raise FileNotFoundError(f"unosat_damage.geojson not found for {city} under {unosat_root}")

    gdf = gpd.read_file(p)

    # Source CRS: per metadata Mariupol came from EPSG:4326. Use whatever
    # geopandas reads from the file; if .crs is None, assume 4326.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Capture WGS84 lon/lat before reprojection
    if gdf.crs.to_epsg() == 4326:
        gdf['lon'] = gdf.geometry.x.astype(np.float64)
        gdf['lat'] = gdf.geometry.y.astype(np.float64)
    else:
        # source not 4326 — reproject a copy to 4326 just for lon/lat record
        wgs = gdf.to_crs("EPSG:4326")
        gdf['lon'] = wgs.geometry.x.astype(np.float64)
        gdf['lat'] = wgs.geometry.y.astype(np.float64)

    # Reproject to raster CRS
    gdf = gdf.to_crs(raster_crs)
    gdf['x_utm'] = gdf.geometry.x.astype(np.float64)
    gdf['y_utm'] = gdf.geometry.y.astype(np.float64)

    # Normalize columns
    gdf['unosat_id'] = gdf['unosat_id'].astype(np.int32)
    gdf['damage'] = gdf['damage'].astype(np.int8)
    gdf['damage_binary'] = gdf['damage_binary'].astype(np.int8)
    gdf['ep'] = gdf['ep'].astype(np.int8)
    if 'damage_label' in gdf.columns:
        gdf['damage_label'] = gdf['damage_label'].astype(str)
    gdf['city'] = city
    gdf['point_source'] = 'unosat'

    if drop_excluded:
        gdf = gdf[gdf['damage_binary'] != -1].reset_index(drop=True)

    # Keep only the columns we care about (plus geometry)
    keep = ['unosat_id', 'date', 'damage', 'damage_label', 'damage_binary',
            'ep', 'city', 'point_source', 'lon', 'lat', 'x_utm', 'y_utm', 'geometry']
    keep = [c for c in keep if c in gdf.columns]
    return gdf[keep].copy()


# ---------------------------------------------------------------------------
# Built-up mask from pre-battle landuse
# ---------------------------------------------------------------------------

def load_builtup_mask(city, stack_root, prebattle_subdir="landuse/prebattle",
                       builtup_classes=None):
    """Load pre-battle landuse and return a boolean built-up mask.

    Picks the LAST (most recent) pre-battle landuse date available for the city
    (closest to invasion). Falls back to first available if "last" can't be
    determined.

    Args:
      city: city name.
      stack_root: data_stack root path.
      prebattle_subdir: relative path under city dir.
      builtup_classes: set of class IDs to mark True (default: {5, 6}).

    Returns:
      (mask, transform, crs, ref_shape) where mask is bool ndarray of shape (H, W).
    """
    builtup_classes = builtup_classes or LU_BUILTUP_CLASSES
    pre_dir = Path(stack_root) / city / prebattle_subdir
    if not pre_dir.exists():
        raise FileNotFoundError(f"pre-battle landuse dir not found: {pre_dir}")

    date_dirs = sorted([d for d in pre_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not date_dirs:
        raise FileNotFoundError(f"no pre-battle landuse date dirs in {pre_dir}")
    # use the most recent pre-battle date (closest to invasion)
    chosen = date_dirs[-1]
    lu_path = chosen / "s2__landuse.tif"
    if not lu_path.exists():
        raise FileNotFoundError(f"s2__landuse.tif not found in {chosen}")

    with rasterio.open(lu_path) as src:
        lu = src.read(1)
        transform = src.transform
        crs = src.crs
        ref_shape = (src.height, src.width)

    mask = np.isin(lu, list(builtup_classes))
    return mask, transform, crs, ref_shape


# ---------------------------------------------------------------------------
# Point → pixel index conversion
# ---------------------------------------------------------------------------

def precompute_point_indices(points_df, transform, ref_shape):
    """Add 'row', 'col' columns to points_df mapping (x_utm, y_utm) → pixel.

    Out-of-bounds points get row=-1, col=-1 sentinel.

    Args:
      points_df: DataFrame with 'x_utm', 'y_utm' columns.
      transform: rasterio Affine transform.
      ref_shape: (H, W).

    Returns:
      DataFrame with new int32 'row', 'col' columns added.
    """
    out = points_df.copy()
    H, W = ref_shape
    rows, cols = rowcol(transform, out['x_utm'].values, out['y_utm'].values, op=int)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    # Bounds check
    oob = (rows < 0) | (rows >= H) | (cols < 0) | (cols >= W)
    rows[oob] = -1
    cols[oob] = -1
    out['row'] = rows
    out['col'] = cols
    return out


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Building centroid loading
# ---------------------------------------------------------------------------

def load_building_centroids(city, stack_root):
    """Compute per-building centroids from building_labels.tif.

    Vectorized via np.bincount; ~0.3s for 119k buildings.

    Args:
      city: city name.
      stack_root: data_stack root path.

    Returns:
      DataFrame with columns:
        building_id (int, 1-based, matches label values in building_labels.tif)
        x_utm, y_utm (float64, raster CRS)
        row, col (int32, pixel indices)
        n_pixels (int32, footprint size)
      Buildings with zero pixels (n_pixels == 0) are dropped.
    """
    labels_path = Path(stack_root) / city / "building_labels.tif"
    if not labels_path.exists():
        raise FileNotFoundError(f"building_labels.tif not found for {city}")

    with rasterio.open(labels_path) as src:
        labels = src.read(1)
        transform = src.transform

    H, W = labels.shape
    n_buildings = int(labels.max())
    if n_buildings == 0:
        return pd.DataFrame(columns=['building_id', 'x_utm', 'y_utm', 'row', 'col', 'n_pixels'])

    flat = labels.ravel()
    mask = flat > 0
    valid_labels = flat[mask]
    flat_rows = np.repeat(np.arange(H, dtype=np.float64), W)[mask]
    flat_cols = np.tile(np.arange(W, dtype=np.float64), H)[mask]

    sum_rows = np.bincount(valid_labels, weights=flat_rows, minlength=n_buildings + 1)[1:]
    sum_cols = np.bincount(valid_labels, weights=flat_cols, minlength=n_buildings + 1)[1:]
    counts = np.bincount(valid_labels, minlength=n_buildings + 1)[1:]

    with np.errstate(invalid='ignore'):
        mean_rows = sum_rows / counts
        mean_cols = sum_cols / counts

    xs = transform.c + mean_cols * transform.a + mean_rows * transform.b
    ys = transform.f + mean_cols * transform.d + mean_rows * transform.e

    df = pd.DataFrame({
        'building_id': np.arange(1, n_buildings + 1, dtype=np.int32),
        'x_utm': xs.astype(np.float64),
        'y_utm': ys.astype(np.float64),
        'row': mean_rows.astype(np.float64),
        'col': mean_cols.astype(np.float64),
        'n_pixels': counts.astype(np.int32),
    })
    # drop buildings with zero pixels
    df = df[df['n_pixels'] > 0].reset_index(drop=True)
    # cast row/col to int32 (centroid pixel)
    df['row'] = df['row'].round().astype(np.int32)
    df['col'] = df['col'].round().astype(np.int32)
    # bounds check
    valid = (df['row'] >= 0) & (df['row'] < H) & (df['col'] >= 0) & (df['col'] < W)
    df = df[valid].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Building-centroid negative sampling (preferred)
# ---------------------------------------------------------------------------

def build_negative_samples_from_buildings(positive_points, building_centroids,
                                            raster_crs, city,
                                            ratio=DEFAULT_NEG_RATIO,
                                            min_dist_m=DEFAULT_NEG_MIN_DIST_M,
                                            seed=None):
    """Sample N=ratio*n_positives building centroids, ≥min_dist_m from any positive.

    Negatives are real Overture buildings (not random pixels in a landuse mask).
    Each negative carries its building_id for cross-reference with V2 parquets.

    Args:
      positive_points: GeoDataFrame of UNOSAT positives (damage_binary==1).
      building_centroids: DataFrame from load_building_centroids().
      raster_crs: CRS of the raster grid (e.g. 'EPSG:32637').
      city: city name.
      ratio: number of negatives = ratio * len(positive_points).
      min_dist_m: exclusion radius around each UNOSAT positive (in metres).
      seed: numpy random seed for reproducibility.

    Returns:
      (gdf_neg, n_candidates) where gdf_neg is a GeoDataFrame with the same
      columns as positives, with point_source='building_centroid', damage=0,
      damage_binary=0, plus a 'building_id' column linking back to Overture.
    """
    rng = np.random.default_rng(seed)
    n_target = ratio * len(positive_points)

    if len(building_centroids) == 0 or len(positive_points) == 0:
        cols = ['unosat_id', 'date', 'damage', 'damage_label', 'damage_binary',
                'ep', 'city', 'point_source', 'lon', 'lat', 'x_utm', 'y_utm',
                'row', 'col', 'building_id', 'geometry']
        return gpd.GeoDataFrame({c: [] for c in cols}, crs=raster_crs), 0

    # KDTree distance from each centroid to nearest UNOSAT positive
    pos_xy = positive_points[['x_utm', 'y_utm']].values
    cen_xy = building_centroids[['x_utm', 'y_utm']].values
    tree = cKDTree(pos_xy)
    nearest_dist, _ = tree.query(cen_xy, k=1)

    # Filter
    keep_mask = nearest_dist >= min_dist_m
    candidates = building_centroids[keep_mask].reset_index(drop=True)
    n_candidates = len(candidates)
    if n_candidates < n_target:
        n_target = n_candidates
        if n_target == 0:
            print(f"  {city}: no building-centroid negatives available")

    # Sample without replacement
    if n_target > 0:
        sel = rng.choice(n_candidates, size=n_target, replace=False)
        sampled = candidates.iloc[sel].reset_index(drop=True)
    else:
        sampled = candidates.iloc[0:0].copy()

    # Build geometries (Point in raster CRS)
    geom = [Point(x, y) for x, y in zip(sampled['x_utm'].values, sampled['y_utm'].values)]
    gdf_neg = gpd.GeoDataFrame(geometry=geom, crs=raster_crs)
    if len(gdf_neg) > 0:
        wgs = gdf_neg.to_crs("EPSG:4326")
        lon = wgs.geometry.x.values
        lat = wgs.geometry.y.values
    else:
        lon = np.zeros(0, dtype=np.float64)
        lat = np.zeros(0, dtype=np.float64)

    out = gpd.GeoDataFrame({
        'unosat_id': -1 * (np.arange(len(sampled), dtype=np.int32) + 1),
        'date': pd.NaT,
        'damage': np.zeros(len(sampled), dtype=np.int8),
        'damage_label': np.array(['no_unosat_match'] * len(sampled), dtype=object),
        'damage_binary': np.zeros(len(sampled), dtype=np.int8),
        'ep': np.zeros(len(sampled), dtype=np.int8),
        'city': city,
        'point_source': 'building_centroid',
        'lon': lon,
        'lat': lat,
        'x_utm': sampled['x_utm'].values,
        'y_utm': sampled['y_utm'].values,
        'row': sampled['row'].values,
        'col': sampled['col'].values,
        'building_id': sampled['building_id'].values.astype(np.int32),
        'geometry': geom,
    }, crs=raster_crs)
    return out, n_candidates

def build_negative_samples_random_pixel(positive_points, builtup_mask, transform, raster_crs,
                            city, ratio=DEFAULT_NEG_RATIO,
                            min_dist_m=DEFAULT_NEG_MIN_DIST_M,
                            res_m=10.0, seed=None,
                            unosat_points_for_buffer=None):
    """Sample random negative points from built-up pixels, excluding a buffer
    around all UNOSAT points (including excluded class -1).

    Args:
      positive_points: GeoDataFrame of UNOSAT positives (damage_binary==1).
      builtup_mask: 2-D bool ndarray of built-up pixels (True = candidate).
      transform: rasterio Affine for the mask.
      raster_crs: CRS of the mask (e.g. 'EPSG:32637').
      city: city name.
      ratio: number of negatives = ratio * len(positive_points).
      min_dist_m: exclusion radius around all UNOSAT points (in metres).
      res_m: raster resolution in metres (10.0 for the BDA grid).
      seed: numpy random seed for reproducibility.
      unosat_points_for_buffer: optional GeoDataFrame of ALL UNOSAT points
        (including damage_binary==-1) to use for the exclusion buffer. If None,
        positive_points is used.

    Returns:
      GeoDataFrame of negative points with same columns as positive_points,
      damage_binary=0, damage=0, point_source='negative_sample',
      unosat_id=-1*(idx+1), date=NaN, ep=0.
    """
    rng = np.random.default_rng(seed)
    n_pos = len(positive_points)
    n_target = ratio * n_pos
    H, W = builtup_mask.shape

    # Buffer set: pixels within min_dist_m of any UNOSAT point are excluded.
    buffer_pts = unosat_points_for_buffer if unosat_points_for_buffer is not None else positive_points
    buffer_radius_px = int(np.ceil(min_dist_m / res_m))

    # Build an exclusion bool mask: dilate the UNOSAT-points pixel set by buffer_radius_px.
    excl = np.zeros_like(builtup_mask, dtype=bool)
    rows_buf, cols_buf = rowcol(transform, buffer_pts['x_utm'].values,
                                  buffer_pts['y_utm'].values, op=int)
    rows_buf = np.asarray(rows_buf, dtype=np.int32)
    cols_buf = np.asarray(cols_buf, dtype=np.int32)
    in_bounds = (rows_buf >= 0) & (rows_buf < H) & (cols_buf >= 0) & (cols_buf < W)
    rows_buf = rows_buf[in_bounds]
    cols_buf = cols_buf[in_bounds]
    excl[rows_buf, cols_buf] = True

    # Vectorized morphological dilation by buffer_radius_px (square kernel, fast)
    if buffer_radius_px > 0:
        # binary dilation via convolution-equivalent shifted-OR
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * buffer_radius_px + 1, 2 * buffer_radius_px + 1), dtype=bool)
        excl = binary_dilation(excl, structure=struct)

    # Candidate pool: built-up AND not in exclusion buffer
    candidate_mask = builtup_mask & ~excl
    n_candidates = int(candidate_mask.sum())
    if n_candidates < n_target:
        # not enough candidates — sample what's available
        n_target = n_candidates
        if n_target == 0:
            print(f"  {city}: no negative-sample candidates available")

    # Sample uniformly without replacement from candidate pixels
    cand_rows, cand_cols = np.where(candidate_mask)
    if n_target > 0:
        sel = rng.choice(len(cand_rows), size=n_target, replace=False)
        neg_rows = cand_rows[sel]
        neg_cols = cand_cols[sel]
    else:
        neg_rows = np.zeros(0, dtype=np.int32)
        neg_cols = np.zeros(0, dtype=np.int32)

    # Convert pixel centers to UTM coords (raster CRS)
    # transform * (col + 0.5, row + 0.5) gives the pixel center
    xs = transform.c + (neg_cols.astype(np.float64) + 0.5) * transform.a + (neg_rows.astype(np.float64) + 0.5) * transform.b
    ys = transform.f + (neg_cols.astype(np.float64) + 0.5) * transform.d + (neg_rows.astype(np.float64) + 0.5) * transform.e

    # Reproject to WGS84 for lon/lat record (consistent with positives)
    geom_utm = [Point(x, y) for x, y in zip(xs, ys)]
    gdf_neg = gpd.GeoDataFrame(geometry=geom_utm, crs=raster_crs)
    if len(gdf_neg) > 0:
        wgs = gdf_neg.to_crs("EPSG:4326")
        lon = wgs.geometry.x.values
        lat = wgs.geometry.y.values
    else:
        lon = np.zeros(0, dtype=np.float64)
        lat = np.zeros(0, dtype=np.float64)

    out = gpd.GeoDataFrame({
        'unosat_id': -1 * (np.arange(len(neg_rows), dtype=np.int32) + 1),  # negative IDs
        'date': pd.NaT,
        'damage': np.zeros(len(neg_rows), dtype=np.int8),
        'damage_label': np.array(['no_unosat'] * len(neg_rows), dtype=object),
        'damage_binary': np.zeros(len(neg_rows), dtype=np.int8),
        'ep': np.zeros(len(neg_rows), dtype=np.int8),
        'city': city,
        'point_source': 'negative_sample',
        'lon': lon,
        'lat': lat,
        'x_utm': xs,
        'y_utm': ys,
        'geometry': geom_utm,
    }, crs=raster_crs)

    return out, n_candidates


# ---------------------------------------------------------------------------
# Raster sampling
# ---------------------------------------------------------------------------

def sample_raster_at_points(tif_path, points_with_idx, ref_shape=None,
                              read_full=True):
    """Sample a raster at every point in points_with_idx.

    Args:
      tif_path: path to single-band TIF.
      points_with_idx: DataFrame with 'row', 'col' int32 columns
                        (precomputed via precompute_point_indices).
      ref_shape: optional (H, W) for shape mismatch handling.
      read_full: if True, read full raster once and index (fast for N>>1 points).
                  If False, use rasterio.sample (slow but lower memory).

    Returns:
      1-D ndarray of length len(points_with_idx). NaN for out-of-bounds points
      and for nodata pixels.
    """
    with rasterio.open(tif_path) as src:
        if read_full:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            # crop or pad if shape mismatch
            if ref_shape and data.shape != ref_shape:
                H, W = ref_shape
                padded = np.full(ref_shape, np.nan, dtype=np.float32)
                h = min(data.shape[0], H)
                w = min(data.shape[1], W)
                padded[:h, :w] = data[:h, :w]
                data = padded
            return sample_raster_block(data, points_with_idx)
        else:
            # rasterio.sample path (slower for N points)
            coords = list(zip(points_with_idx['x_utm'].values,
                              points_with_idx['y_utm'].values))
            samples = list(src.sample(coords))
            arr = np.array([s[0] for s in samples], dtype=np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            return arr


def sample_raster_block(data_array, points_with_idx):
    """Sample a pre-loaded raster at every point's (row, col).

    Out-of-bounds points (row=-1 or col=-1) get NaN.

    Args:
      data_array: 2-D float32 array (already nodata-replaced with NaN).
      points_with_idx: DataFrame with 'row', 'col' int32 columns.

    Returns:
      1-D ndarray of length len(points_with_idx).
    """
    rows = points_with_idx['row'].values
    cols = points_with_idx['col'].values
    H, W = data_array.shape
    out = np.full(len(rows), np.nan, dtype=np.float32)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    if valid.any():
        out[valid] = data_array[rows[valid], cols[valid]]
    return out


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner():
    print(f"  nb05bv3_helpers loaded:")
    print(f"    load_unosat_points_for_city, load_builtup_mask, load_building_centroids")
    print(f"    build_negative_samples_from_buildings  (PREFERRED — Overture centroids)")
    print(f"    build_negative_samples_random_pixel    (legacy — landuse-mask pixels)")
    print(f"    sample_raster_at_points, sample_raster_block")
    print(f"    precompute_point_indices")
    print(f"    defaults: ratio={DEFAULT_NEG_RATIO}, min_dist={DEFAULT_NEG_MIN_DIST_M}m")


if __name__ == '__main__':
    _print_banner()
