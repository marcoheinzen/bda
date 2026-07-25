# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
nb05bv7_helpers.py — V7 kernel-based zonal sampling at points (KERNEL_SIZE=7, no SAR filter).

V7 is V4 with one architectural change: KERNEL_SIZE = 7 (7x7 = 49 pixels per
point) instead of V4's 3x3 = 9. Otherwise identical to V4. No SAR filter.
V7 sits at the literature sweet spot for residential-scale change detection.

For numeric TIFs (reflectance, backscatter, indices, accumulator outputs):
returns 8 stats per modality — mean, p10, p50, p90, std, min, max, max_abs_delta.

For categorical TIFs (landuse): returns 1 column — the mode of the kernel.

For boundary points (kernels that would extend past the raster edge): pads
with NaN. The v29 small-N graceful degradation rule then applies: if fewer
than 5 valid pixels in the kernel, std/p10/p90/max_abs_delta become NaN; if
fewer than 2, all distributional stats become NaN; if 1 valid pixel, mean=
p50=min=max=that single value with std/p10/p90/max_abs_delta NaN; if 0 valid
pixels, all stats NaN.

This follows the exact same small-N rule documented in
DECISION_NB05b_v29_Zonal_Aggregation.md Sec 5.

Reuses nb05bv3_helpers for everything except the per-point sampling itself
(UNOSAT loading, building centroids, negative sampling, point indices,
sample_raster_block — V3's unchanged exports).
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

KERNEL_SIZE = 7  # default 7x7 = 49 pixels around each point. Override via global.

# v29 statistic-aware reduction set for numeric features
NUMERIC_STAT_NAMES = ('mean', 'p10', 'p50', 'p90', 'std', 'min', 'max', 'max_abs_delta')

# Categorical TIF detection patterns. These match against the TIF filename.
# Cells with categorical TIFs return a single 'mode' column instead of 8 stats.
_CATEGORICAL_PATTERNS = (
    'landuse', 'lu_at_', 'modal_post_class', 'final_class',
    'lu_transition', 'lu_change',
)


def _print_banner():
    print(f"  nb05bv7_helpers loaded — kernel-based sampling (V7: 7x7, no SAR filter)")
    print(f"    KERNEL_SIZE = {KERNEL_SIZE}x{KERNEL_SIZE} ({KERNEL_SIZE*KERNEL_SIZE} pixels per point)")
    print(f"    numeric stats: {NUMERIC_STAT_NAMES}")
    print(f"    categorical detection: filename match against {_CATEGORICAL_PATTERNS}")


def is_categorical_tif(tif_path):
    """Heuristic: if any categorical pattern matches the filename, treat as
    categorical (return mode only). Otherwise treat as numeric (return 8 stats).
    """
    name = Path(tif_path).name.lower() if tif_path else ''
    return any(pat in name for pat in _CATEGORICAL_PATTERNS)


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
                      categorical=None):
    """V4 drop-in replacement for V3's sample_one().

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

    if categorical:
        return _kernel_stats_categorical(kernels)
    else:
        return _kernel_stats_numeric(kernels)
