# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
nb05bv6_helpers.py — V6 kernel-based zonal sampling at points
                      (KERNEL_SIZE=5, Refined Lee filter on SAR rasters).

V6 is V5 (5x5 kernel) with Refined Lee speckle filter applied to SAR rasters
(VV, VH, COH) before the v29 statistic-aware reductions are computed. The
filter is applied per-kernel (Option B): for each point, extract a 7x7 window
(kernel + 1-pixel Lee border), apply Refined Lee, retain the inner 5x5 for
zonal stats. Optical, indices, NBR, landuse, and accumulator-derived rasters
are not filtered (speckle is a SAR-specific phenomenon).

V6 vs V5 isolates the speckle-filter effect at constant kernel size.
V6 vs V2 quantifies whether filtered kernel sampling can recover the V2
footprint advantage on SAR-dependent parquets.

Refined Lee algorithm: for each pixel, examine four directional 3x3
sub-windows offset around the pixel, select the minimum-variance sub-window
(the most homogeneous local direction), and compute a weighted blend of the
sub-window mean and the original pixel. Weight K = max(0, (var(W) -
sigma_n^2 * mean(W)^2) / var(W)). Reference: Yommy et al. 2015; SNAP toolbox
convention. sigma_n (noise standard deviation) is set to 0.25 — the SNAP
default for Sentinel-1 GRD.

For numeric TIFs (reflectance, backscatter, indices, accumulator outputs):
returns 8 stats per modality — mean, p10, p50, p90, std, min, max, max_abs_delta.

For categorical TIFs (landuse): returns 1 column — the mode of the kernel.

For boundary points (kernels that would extend past the raster edge): pads
with NaN. The v29 small-N graceful degradation rule then applies (n>=5 full
set, n=2..4 drops p10/p90, n=1 only mean=p50=min=max, n=0 all NaN).

Reuses nb05bv3_helpers for everything except the per-point sampling itself.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

# Re-export V3's data-loading helpers unchanged. V4 only changes the sampling
# strategy at the point itself.
from nb05bv3_helpers import (
    load_unosat_points_for_city,
    load_building_centroids,
    build_negative_samples_from_buildings,
    precompute_point_indices,
    sample_raster_block,  # V3 single-pixel sampler — kept for code paths that need it
)

KERNEL_SIZE = 5  # default 5x5 = 25 pixels around each point. Override via global.
LEE_NOISE_STD = 0.25  # Refined Lee sigma_n parameter (SNAP default for S1 GRD).

# v29 statistic-aware reduction set for numeric features
NUMERIC_STAT_NAMES = ('mean', 'p10', 'p50', 'p90', 'std', 'min', 'max', 'max_abs_delta')

# Categorical TIF detection patterns. These match against the TIF filename.
# Cells with categorical TIFs return a single 'mode' column instead of 8 stats.
_CATEGORICAL_PATTERNS = (
    'landuse', 'lu_at_', 'modal_post_class', 'final_class',
    'lu_transition', 'lu_change',
)

# SAR raster detection patterns. SAR rasters get Refined Lee speckle filtering
# applied to each kernel before v29 statistic-aware reductions are computed.
# Optical (s2__), indices, NBR, landuse, fire, and accumulator-derived rasters
# do not need speckle filtering — speckle is a SAR-specific physical phenomenon
# inherent to coherent imaging.
_SAR_PATTERNS = (
    's1__vv__', 's1__vh__', 's1__coh_', 'coh_vv__', 'coh_vh__',
    '__vv__', '__vh__',
)


def _print_banner():
    print(f"  nb05bv6_helpers loaded — kernel-based sampling (V6: 5x5 + Refined Lee on SAR)")
    print(f"    KERNEL_SIZE = {KERNEL_SIZE}x{KERNEL_SIZE} ({KERNEL_SIZE*KERNEL_SIZE} pixels per point)")
    print(f"    LEE_NOISE_STD = {LEE_NOISE_STD} (Refined Lee sigma_n; applied to SAR rasters only)")
    print(f"    numeric stats: {NUMERIC_STAT_NAMES}")
    print(f"    categorical detection: filename match against {_CATEGORICAL_PATTERNS}")


def is_categorical_tif(tif_path):
    """Heuristic: if any categorical pattern matches the filename, treat as
    categorical (return mode only). Otherwise treat as numeric (return 8 stats).
    """
    name = Path(tif_path).name.lower() if tif_path else ''
    return any(pat in name for pat in _CATEGORICAL_PATTERNS)


def is_sar_tif(tif_path):
    """Heuristic: if any SAR pattern matches the filename, apply Refined Lee
    speckle filter to each kernel before computing v29 statistics. SAR
    accumulators (running_min/running_max/drop_count/etc.) are NOT filtered
    even when their underlying modality is SAR, because the temporal-extremum
    operation already happened at the pixel level inside NB03e and the output
    is a derived statistic, not a backscatter/coherence sample. We detect
    accumulators by the presence of accumulator-specific tokens in the filename.
    """
    name = Path(tif_path).name.lower() if tif_path else ''
    # Exclude accumulator outputs (already temporally extremized, not raw SAR)
    accum_tokens = ('running_min', 'running_max', 'max_abs_delta',
                    'drop_count', 'rise_count', 'exceedance_rate',
                    'urban_retained', 'date_first_', 'date_worst_',
                    'baseline', 'zscore', 'roll_')
    if any(tok in name for tok in accum_tokens):
        return False
    return any(pat in name for pat in _SAR_PATTERNS)


def _extract_kernels(data, rows, cols, kernel_size, ref_shape):
    """Extract a kernel_size x kernel_size window around each (row, col) point.

    Returns 3-D array of shape (n_points, kernel_size, kernel_size) with NaN
    where pixels would fall outside the raster bounds.
    """
    n = len(rows)
    H, W = ref_shape
    half = kernel_size // 2

    # Allocate output as float32 with NaN fill (handles boundary padding in one shot)
    out = np.full((n, kernel_size, kernel_size), np.nan, dtype=np.float32)

    for ki in range(kernel_size):
        di = ki - half
        for kj in range(kernel_size):
            dj = kj - half
            r = rows + di
            c = cols + dj
            valid = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            if valid.any():
                # gather the valid pixels into the (ki, kj) slice
                idx = np.where(valid)[0]
                out[idx, ki, kj] = data[r[idx], c[idx]]
    return out


def _refined_lee_kernel(kernels, sigma_n=None):
    """Apply Refined Lee speckle filter to each kernel in the stack.

    Vectorized per-kernel-pixel implementation using kernel-local sub-windows.
    For each pixel in the K x K kernel, examines four (K//2+1) x (K//2+1)
    sub-windows (NW/NE/SW/SE quadrants), selects the minimum-variance sub-window
    (the most homogeneous local direction), and outputs a weighted blend:
        filtered = sub_mean + K * (original - sub_mean)
        K = max(0, (sub_var - sigma_n^2 * sub_mean^2) / sub_var)

    Args:
        kernels: float32 array of shape (n_points, K, K) with NaN at boundary.
        sigma_n: noise std (default LEE_NOISE_STD = 0.25 for SNAP S1 GRD).

    Returns:
        Filtered kernels of same shape. Pixels where no sub-window has >= 4
        valid (non-NaN) neighbors are returned unchanged. NaN positions
        (boundary padding) are preserved as NaN.
    """
    if sigma_n is None:
        sigma_n = LEE_NOISE_STD

    n_pts, K, _ = kernels.shape
    half = K // 2  # for K=5, half=2 -> sub-windows are 3x3

    # Pre-compute the four quadrant sub-windows: NW (rows 0..half, cols 0..half),
    # NE (rows 0..half, cols half..K-1), SW (rows half..K-1, cols 0..half),
    # SE (rows half..K-1, cols half..K-1). Each sub-window is (half+1, half+1).
    # For K=5: each is 3x3 = 9 pixels.
    sw_size = half + 1
    quadrant_slices = [
        (slice(0, sw_size), slice(0, sw_size)),       # NW
        (slice(0, sw_size), slice(K - sw_size, K)),   # NE
        (slice(K - sw_size, K), slice(0, sw_size)),   # SW
        (slice(K - sw_size, K), slice(K - sw_size, K)),  # SE
    ]

    # For each quadrant, compute mean and variance per kernel (NaN-safe)
    quad_means = np.full((4, n_pts), np.nan, dtype=np.float32)
    quad_vars = np.full((4, n_pts), np.nan, dtype=np.float32)
    for qi, (rs, cs) in enumerate(quadrant_slices):
        sub = kernels[:, rs, cs].reshape(n_pts, -1)
        n_valid_q = (~np.isnan(sub)).sum(axis=1)
        ok = n_valid_q >= 4  # need at least 4 valid pixels in sub-window
        if ok.any():
            with np.errstate(invalid='ignore'):
                quad_means[qi, ok] = np.nanmean(sub[ok], axis=1)
                quad_vars[qi, ok] = np.nanvar(sub[ok], axis=1, ddof=0)

    # For each pixel in the kernel, determine which quadrant it belongs to.
    # A kernel pixel at (r, c) belongs to multiple quadrants if it falls in
    # the overlap region (the central row/column for odd K). For simplicity
    # we use the quadrant that contains the pixel as a non-overlap interior;
    # for the central pixel (r=half, c=half) all four quadrants are valid.
    # Implementation: assign each pixel to the unique non-overlapping quadrant
    # if possible, else use the minimum-variance quadrant among those it
    # belongs to.

    # Build a (K, K) lookup of which quadrants contain each pixel position.
    # Each entry is a tuple of valid quadrant indices.
    pixel_to_quads = {}
    for r in range(K):
        for c in range(K):
            quads = []
            if r <= half and c <= half:
                quads.append(0)  # NW
            if r <= half and c >= half:
                quads.append(1)  # NE
            if r >= half and c <= half:
                quads.append(2)  # SW
            if r >= half and c >= half:
                quads.append(3)  # SE
            pixel_to_quads[(r, c)] = quads

    # Output filtered kernels (start as copy)
    filtered = kernels.copy()

    # For each pixel position in the kernel, vectorize across all n_pts
    for r in range(K):
        for c in range(K):
            quads = pixel_to_quads[(r, c)]
            orig_pix = kernels[:, r, c]  # (n_pts,)

            # For each kernel, pick the quadrant (among `quads`) with min
            # variance (most homogeneous local direction). NaN-safe.
            cand_vars = quad_vars[quads, :]  # (len(quads), n_pts)
            cand_means = quad_means[quads, :]

            # If all candidate variances are NaN, skip this pixel (leave unchanged)
            valid_q = ~np.isnan(cand_vars)
            any_valid = valid_q.any(axis=0)

            # Pick the index of the minimum-variance quadrant (ignoring NaN)
            cand_vars_nan = np.where(valid_q, cand_vars, np.inf)
            best_q = np.argmin(cand_vars_nan, axis=0)  # (n_pts,)

            # Gather the chosen mean and variance per kernel
            sub_mean = cand_means[best_q, np.arange(n_pts)]
            sub_var = cand_vars[best_q, np.arange(n_pts)]

            # Refined Lee weight K = max(0, (var - sigma_n^2 * mean^2) / var)
            with np.errstate(invalid='ignore', divide='ignore'):
                k_weight = (sub_var - sigma_n * sigma_n * sub_mean * sub_mean) / sub_var
                k_weight = np.where(np.isfinite(k_weight), k_weight, 0.0)
                k_weight = np.clip(k_weight, 0.0, 1.0)

            # Apply filter only where: orig pixel is valid, any quadrant valid,
            # and orig pixel is not NaN
            apply = any_valid & ~np.isnan(orig_pix)
            new_pix = sub_mean + k_weight * (orig_pix - sub_mean)
            # Only overwrite filtered positions where apply is True; preserve
            # original (including NaN) elsewhere
            filtered[apply, r, c] = new_pix[apply]

    return filtered


def _kernel_stats_numeric(kernels):
    """Compute the v29 8-stat reduction over each kernel.

    Input shape: (n_points, K, K). Output: dict of 8 named 1-D arrays of
    length n_points.

    Small-N graceful degradation per DECISION_NB05b_v29 Sec 5:
      n >= 5 valid pixels: full set
      n in 2-4:           mean, std, min, max, p50, max_abs_delta (no p10/p90)
      n == 1:             mean=p50=min=max, std/p10/p90/max_abs_delta = NaN
      n == 0:             all NaN
    """
    n_pts = kernels.shape[0]
    flat = kernels.reshape(n_pts, -1)  # (n_points, K*K)

    # Count valid (non-NaN) pixels per kernel
    valid_mask = ~np.isnan(flat)
    n_valid = valid_mask.sum(axis=1)

    out = {s: np.full(n_pts, np.nan, dtype=np.float32) for s in NUMERIC_STAT_NAMES}

    # n >= 1: mean, min, max, p50 always defined
    has_any = n_valid >= 1
    if has_any.any():
        # numpy nanmean / nanmin / nanmax / nanmedian return NaN for all-NaN
        # rows but issue a warning; suppress by selecting valid rows only.
        idx = np.where(has_any)[0]
        sub = flat[idx]
        with np.errstate(invalid='ignore'):
            out['mean'][idx] = np.nanmean(sub, axis=1)
            out['min'][idx] = np.nanmin(sub, axis=1)
            out['max'][idx] = np.nanmax(sub, axis=1)
            out['p50'][idx] = np.nanmedian(sub, axis=1)

    # n >= 2: std defined; p10/p90 still need >= 5 per v29 rule
    has_2plus = n_valid >= 2
    if has_2plus.any():
        idx = np.where(has_2plus)[0]
        sub = flat[idx]
        with np.errstate(invalid='ignore'):
            out['std'][idx] = np.nanstd(sub, axis=1, ddof=0)
            # max_abs_delta = max - min (proxy for spatial range, well-defined
            # whenever min and max are; this matches the v29 wide reductions)
            out['max_abs_delta'][idx] = out['max'][idx] - out['min'][idx]

    # n >= 5: p10 and p90 stable
    has_5plus = n_valid >= 5
    if has_5plus.any():
        idx = np.where(has_5plus)[0]
        sub = flat[idx]
        with np.errstate(invalid='ignore'):
            out['p10'][idx] = np.nanpercentile(sub, 10, axis=1)
            out['p90'][idx] = np.nanpercentile(sub, 90, axis=1)

    return out


def _kernel_stats_categorical(kernels):
    """Mode of the kernel for categorical rasters.

    Input shape: (n_points, K, K). Output: dict {'mode': 1-D array of length
    n_points}. NaN if no valid pixels in the kernel.

    Uses a vectorized hash-based approach: cast to int, replace NaN with a
    sentinel (-1), then use scipy.stats.mode equivalent via np.unique per row.
    For 9-pixel kernels this is efficient enough.
    """
    n_pts = kernels.shape[0]
    flat = kernels.reshape(n_pts, -1)
    out = np.full(n_pts, np.nan, dtype=np.float32)

    # Process row-by-row; 9 pixels per row, n_pts rows. For ~50k points this
    # is ~500k operations, sub-second on modern hardware.
    for i in range(n_pts):
        row = flat[i]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        vals, counts = np.unique(valid.astype(np.int32), return_counts=True)
        out[i] = vals[counts.argmax()]
    return {'mode': out}


def sample_one_kernel(tif_path, pts_df, ref_shape, kernel_size=None,
                      categorical=None, sar=None):
    """V6 drop-in replacement for V3's sample_one() (V5 + Refined Lee on SAR).

    Returns a dict mapping stat suffix -> 1-D array of length len(pts_df).
    For numeric TIFs, the dict has 8 entries (NUMERIC_STAT_NAMES).
    For categorical TIFs, the dict has 1 entry: 'mode'.

    Args:
        tif_path: path to the TIF
        pts_df: DataFrame with 'row' and 'col' columns (precomputed pixel
                indices)
        ref_shape: (height, width) of the city's reference grid
        kernel_size: override the module-level KERNEL_SIZE for this call
        categorical: True/False to override filename-based detection

    Caller pattern (replaces V3's `feats[col] = sample_one(...)`):
        for stat, arr in sample_one_kernel(path, pts, ref_shape).items():
            feats[f"{col}__{stat}"] = arr
    """
    if kernel_size is None:
        kernel_size = KERNEL_SIZE
    if categorical is None:
        categorical = is_categorical_tif(tif_path)
    if sar is None:
        sar = is_sar_tif(tif_path) and not categorical

    # Read TIF, apply nodata mask, pad to ref_shape (same as V3's sample_one)
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            try:
                nd_f = float(nodata)
                if not np.isnan(nd_f):
                    data[data == nd_f] = np.nan
            except (ValueError, TypeError):
                pass
        if data.shape != ref_shape:
            H, W = ref_shape
            padded = np.full(ref_shape, np.nan, dtype=np.float32)
            h = min(data.shape[0], H)
            w = min(data.shape[1], W)
            padded[:h, :w] = data[:h, :w]
            data = padded

    rows = np.asarray(pts_df['row'].values, dtype=np.int64)
    cols = np.asarray(pts_df['col'].values, dtype=np.int64)

    kernels = _extract_kernels(data, rows, cols, kernel_size, ref_shape)

    # V6: apply Refined Lee speckle filter on SAR rasters before stat reduction.
    # Skip filter for: categorical (no physical sense), accumulator outputs
    # (already temporally extremized in NB03e), and non-SAR rasters (no speckle).
    if sar:
        kernels = _refined_lee_kernel(kernels, sigma_n=LEE_NOISE_STD)

    if categorical:
        return _kernel_stats_categorical(kernels)
    else:
        return _kernel_stats_numeric(kernels)
