# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""Aggregation helpers for NB05b v2 (v29).

Three helpers that replace the universal `extract_zonal_stats_from_labels` call
across all 22 builder cells. Each helper takes the SAME interface signature:
    (label_array, data, prefix, n_buildings, ...)
and returns a dict {column_name: np.ndarray of length n_buildings}.

Implementation notes (v29.1, performance fix):
  - extract_zonal_aware uses pandas.groupby for per-building reductions.
    pandas.groupby is C-optimized for groupby aggregations and is the right
    tool here. The previous per-building Python loop with np.percentile calls
    was 25-30x slower.
  - extract_zonal_dates uses pandas.groupby AND keeps dates as int32
    throughout. Dates are integer-valued (YYYYMMDD or days-since-invasion),
    no float conversion, no precision loss. Sentinel -1 for "missing".
  - extract_zonal_categorical uses np.bincount + crosstab-equivalent
    arithmetic. Already vectorized; pandas crosstab is 9x slower for this
    workload because it builds a full sparse table.

Aggregation rules per Marco's master's thesis V2 design (NB03e v47 + NB05b v29):

  extract_zonal_aware:
    agg='raw'           per-scene raw bands, indices, NBR, rolling means
                        emits: mean, p10, p50, p90, std, min, max, max_abs_delta
    agg='composite_raw' composite bands (median-composited)
                        same as raw
    agg='delta'         delta products (composite_vs_scene, prepost CARD delta)
                        same as raw
    agg='accum_min'     accumulator running_min rasters
                        same as raw
    agg='accum_max'     accumulator running_max rasters
                        same as raw
    agg='count'         drop_count, rise_count, exceedance_count etc.
                        emits: mean, max, min
    agg='stat_of_stat'  block stats / rolling stats (7-stat tables)
                        emits: mean, p10, p90, mean_abs_delta, min, max

  extract_zonal_categorical:
    landuse class rasters (s2 SCL-derived, P5 outputs).
    Emits class fractions for every encountered class + mode_raw + mode_corrected
    + urban_or_bare_frac. Mode-correction rule:
      - if urban_frac + bare_frac >= 0.20: exclude veg, return majority of remaining
      - elif urban_frac + bare_frac < 0.20 AND veg_frac >= 0.80: return veg (honest)
      - else: return raw mode

  extract_zonal_dates:
    date rasters from accumulators (encoded YYYYMMDD or days-since-invasion).
    Emits min_nonzero, max, count_unique. Per V2 taxonomy these reductions are
    METADATA columns (battle-calendar leakage), NOT features.
    All output is int32. Sentinel -1 for "missing".

Small-N graceful degradation (universal across all helpers):
    n_pixels in footprint:
       >= 5: full reduction set
       2-4 : drop p10/p90, keep mean/std/min/max/p50/max_abs_delta
       1   : single-value duplication for mean/p50/min/max; std/p10/p90/delta -> NaN
       0   : all NaN
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')


DEFAULT_MIN_PIXELS = 3


# Landuse class mapping (matches Sentinel-2 SCL-derived classes used in NB03e P5)
LU_CLASS_NAMES = {
    1: 'snow',
    2: 'water',
    3: 'dense_veg',
    4: 'sparse_veg',
    5: 'urban',
    6: 'bare',
    7: 'shadow',
    8: 'cloud',
}

LU_VEG_CLASSES = {3, 4}
LU_URBAN_CLASSES = {5}
LU_BARE_CLASSES = {6}
LU_WATER_CLASSES = {2}
LU_INVALID_CLASSES = {0, 7, 8}
LU_TIEBREAK_PRIORITY = [6, 5, 2, 1]

VEG_OVERRIDE_THRESHOLD = 0.20
VEG_HONEST_THRESHOLD = 0.80


def _build_pixel_to_building_map(label_array, n_buildings):
    """Return (flat_labels, valid_mask) — same interface as before."""
    flat = label_array.ravel()
    valid = (flat > 0) & (flat <= n_buildings)
    return flat, valid


def extract_zonal_aware(label_array, data, prefix, n_buildings,
                        agg='raw', min_pixels=DEFAULT_MIN_PIXELS):
    """Per-building zonal reduction via pandas groupby.

    Args:
      label_array: 2-D int raster of building labels (0 = no building, 1..n = id).
      data: 2-D float raster of pixel values (NaN for invalid).
      prefix: column-name prefix, e.g. 's2__b11' or 's1__vv__roll7__running_min'.
      n_buildings: total number of buildings (max label ID).
      agg: aggregation rule from {'raw', 'composite_raw', 'delta', 'accum_min',
                                   'accum_max', 'count', 'stat_of_stat'}.
      min_pixels: small-N gate (currently informational; gating is per-stat).

    Returns:
      dict {column_name: np.ndarray of length n_buildings}.
      Keys are prefixed with `prefix + '__'`. Always includes
      `prefix + '__n_pixels_valid'`.
    """
    if data.shape != label_array.shape:
        raise ValueError(f"data shape {data.shape} != label_array shape {label_array.shape}")

    if agg == 'count':
        wanted = ['mean', 'min', 'max']
    elif agg == 'stat_of_stat':
        wanted = ['mean', 'p10', 'p90', 'min', 'max', 'mean_abs_delta']
    elif agg in ('raw', 'composite_raw', 'delta', 'accum_min', 'accum_max'):
        wanted = ['mean', 'p10', 'p50', 'p90', 'std', 'min', 'max', 'max_abs_delta']
    else:
        raise ValueError(f"unknown agg='{agg}'. Valid: raw, composite_raw, delta, "
                         f"accum_min, accum_max, count, stat_of_stat")

    # Initialize all output arrays
    out = {}
    for r in wanted:
        out[f"{prefix}__{r}"] = np.full(n_buildings, np.nan, dtype=np.float32)
    out[f"{prefix}__n_pixels_valid"] = np.zeros(n_buildings, dtype=np.float32)

    # Build flat (building, value) frame, drop invalid
    flat_lbl = label_array.ravel()
    flat_val = data.ravel()
    keep = (flat_lbl > 0) & (flat_lbl <= n_buildings) & np.isfinite(flat_val)
    if not keep.any():
        return out

    df = pd.DataFrame({
        'b': flat_lbl[keep].astype(np.int32),
        'v': flat_val[keep].astype(np.float32),
    })

    g = df.groupby('b', sort=True)
    pixel_count = g.size()
    idx = pixel_count.index.values - 1  # 1-indexed labels -> 0-indexed array slots

    out[f"{prefix}__n_pixels_valid"][idx] = pixel_count.values.astype(np.float32)

    has_5 = pixel_count.values >= 5
    has_2 = pixel_count.values >= 2

    # mean — always (n=1 returns the single value, which is correct)
    if 'mean' in wanted:
        out[f"{prefix}__mean"][idx] = g['v'].mean().values.astype(np.float32)

    # std (ddof=0 to match numpy convention) — gated at n>=2
    if 'std' in wanted:
        std_vals = g['v'].std(ddof=0).values.astype(np.float32)
        std_vals[~has_2] = np.nan
        out[f"{prefix}__std"][idx] = std_vals

    # min, max — n>=1 (n=1 returns the single value)
    if 'min' in wanted:
        out[f"{prefix}__min"][idx] = g['v'].min().values.astype(np.float32)
    if 'max' in wanted:
        out[f"{prefix}__max"][idx] = g['v'].max().values.astype(np.float32)

    # percentiles — get all wanted ones in one pandas call
    pct_q = [(0.10, 'p10'), (0.50, 'p50'), (0.90, 'p90')]
    needed_q = [(q, name) for q, name in pct_q if name in wanted]
    if needed_q:
        q_vals = [q for q, _ in needed_q]
        q_df = g['v'].quantile(q_vals).unstack()
        for q, name in needed_q:
            arr = q_df[q].values.astype(np.float32)
            if name == 'p50':
                # p50 valid for n>=2; for n=1 buildings, p50 = the single value
                arr[~has_2] = np.nan
                out[f"{prefix}__p50"][idx] = arr
                # n=1 special case: copy the single value (= mean for n=1)
                if 'mean' in wanted:
                    n1_mask = pixel_count.values == 1
                    if n1_mask.any():
                        out[f"{prefix}__p50"][idx[n1_mask]] = out[f"{prefix}__mean"][idx[n1_mask]]
            else:
                # p10, p90 require n>=5
                arr[~has_5] = np.nan
                out[f"{prefix}__{name}"][idx] = arr

    # max_abs_delta and mean_abs_delta — need median per building first
    if 'max_abs_delta' in wanted or 'mean_abs_delta' in wanted:
        median_per_b = g['v'].median()  # for n=1 returns the single value
        # broadcast median to per-pixel via map
        df['delta'] = (df['v'] - df['b'].map(median_per_b)).abs()
        g2 = df.groupby('b', sort=True)['delta']
        if 'max_abs_delta' in wanted:
            mad = g2.max().values.astype(np.float32)
            mad[~has_2] = np.nan
            out[f"{prefix}__max_abs_delta"][idx] = mad
        if 'mean_abs_delta' in wanted:
            mead = g2.mean().values.astype(np.float32)
            mead[~has_2] = np.nan
            out[f"{prefix}__mean_abs_delta"][idx] = mead

    return out


def extract_zonal_categorical(label_array, data, prefix, n_buildings,
                               min_pixels=DEFAULT_MIN_PIXELS,
                               class_names=None,
                               veg_classes=None,
                               urban_classes=None,
                               bare_classes=None,
                               water_classes=None,
                               invalid_classes=None,
                               veg_override_threshold=VEG_OVERRIDE_THRESHOLD,
                               veg_honest_threshold=VEG_HONEST_THRESHOLD,
                               tiebreak_priority=None):
    """Per-building reduction for categorical (landuse) rasters.

    Emits class fractions + mode_raw + mode_corrected + urban_or_bare_frac.
    Mode correction: drop veg if urban_or_bare >= 0.20; honest veg if
    veg >= 0.80; otherwise raw mode.

    Implementation: vectorized np.bincount over (building, class) pairs.
    Pandas crosstab was 9x slower on this workload.
    """
    if data.shape != label_array.shape:
        raise ValueError(f"data shape {data.shape} != label_array shape {label_array.shape}")

    class_names = class_names or LU_CLASS_NAMES
    veg_classes = veg_classes if veg_classes is not None else LU_VEG_CLASSES
    urban_classes = urban_classes if urban_classes is not None else LU_URBAN_CLASSES
    bare_classes = bare_classes if bare_classes is not None else LU_BARE_CLASSES
    water_classes = water_classes if water_classes is not None else LU_WATER_CLASSES
    invalid_classes = invalid_classes if invalid_classes is not None else LU_INVALID_CLASSES
    tiebreak_priority = tiebreak_priority if tiebreak_priority is not None else LU_TIEBREAK_PRIORITY

    flat_labels, valid_mask = _build_pixel_to_building_map(label_array, n_buildings)
    flat_classes = data.ravel().astype(np.int32)

    all_classes = sorted(set(class_names.keys()) - invalid_classes)

    out = {f"{prefix}__n_pixels_valid": np.zeros(n_buildings, dtype=np.float32)}
    for c in all_classes:
        cname = class_names.get(c, f'class{c}')
        out[f"{prefix}__{cname}_frac"] = np.full(n_buildings, np.nan, dtype=np.float32)
    out[f"{prefix}__urban_or_bare_frac"] = np.full(n_buildings, np.nan, dtype=np.float32)
    out[f"{prefix}__mode_raw"] = np.full(n_buildings, -1.0, dtype=np.float32)
    out[f"{prefix}__mode_corrected"] = np.full(n_buildings, -1.0, dtype=np.float32)

    invalid_mask = np.isin(flat_classes, list(invalid_classes))
    pixel_mask = valid_mask & ~invalid_mask & (flat_classes >= 0)
    if not pixel_mask.any():
        return out

    valid_lbl = flat_labels[pixel_mask].astype(np.int32)
    valid_cls = flat_classes[pixel_mask]

    max_cls = max(all_classes) if all_classes else 0
    flat_idx = (valid_lbl - 1) * (max_cls + 1) + valid_cls
    counts = np.bincount(flat_idx, minlength=n_buildings * (max_cls + 1))
    counts = counts[:n_buildings * (max_cls + 1)].reshape(n_buildings, max_cls + 1)

    bldg_total = counts.sum(axis=1)
    has_pixels = bldg_total >= min_pixels
    out[f"{prefix}__n_pixels_valid"] = bldg_total.astype(np.float32)

    for c in all_classes:
        if c <= max_cls:
            cname = class_names.get(c, f'class{c}')
            frac = np.full(n_buildings, np.nan, dtype=np.float32)
            np.divide(counts[:, c], bldg_total, out=frac, where=has_pixels)
            out[f"{prefix}__{cname}_frac"] = frac

    urban_count = np.zeros(n_buildings, dtype=np.float32)
    for c in urban_classes:
        if c <= max_cls:
            urban_count += counts[:, c]
    bare_count = np.zeros(n_buildings, dtype=np.float32)
    for c in bare_classes:
        if c <= max_cls:
            bare_count += counts[:, c]
    veg_count = np.zeros(n_buildings, dtype=np.float32)
    for c in veg_classes:
        if c <= max_cls:
            veg_count += counts[:, c]

    urban_bare_frac = np.full(n_buildings, np.nan, dtype=np.float32)
    np.divide(urban_count + bare_count, bldg_total, out=urban_bare_frac, where=has_pixels)
    out[f"{prefix}__urban_or_bare_frac"] = urban_bare_frac

    veg_frac = np.full(n_buildings, np.nan, dtype=np.float32)
    np.divide(veg_count, bldg_total, out=veg_frac, where=has_pixels)

    mode_raw = np.argmax(counts, axis=1).astype(np.float32)
    mode_raw[~has_pixels] = -1
    out[f"{prefix}__mode_raw"] = mode_raw

    mode_corr = mode_raw.copy()

    has_veg = veg_count > 0
    apply_rule1 = has_pixels & (urban_bare_frac >= veg_override_threshold) & has_veg
    if apply_rule1.any():
        counts_no_veg = counts.copy()
        for vc in veg_classes:
            if vc <= max_cls:
                counts_no_veg[:, vc] = 0
        argmax_no_veg = np.argmax(counts_no_veg, axis=1).astype(np.float32)
        for b_idx in np.where(apply_rule1)[0]:
            row = counts_no_veg[b_idx]
            max_count = row.max()
            if max_count > 0:
                tied = [c for c in range(max_cls + 1) if row[c] == max_count]
                if len(tied) > 1:
                    for pref_c in tiebreak_priority:
                        if pref_c in tied:
                            argmax_no_veg[b_idx] = pref_c
                            break
        mode_corr[apply_rule1] = argmax_no_veg[apply_rule1]

    apply_rule2 = (has_pixels &
                   (urban_bare_frac < veg_override_threshold) &
                   (veg_frac >= veg_honest_threshold))
    if apply_rule2.any():
        for b_idx in np.where(apply_rule2)[0]:
            best_veg_cls = None
            best_veg_count = 0
            for vc in veg_classes:
                if vc <= max_cls and counts[b_idx, vc] > best_veg_count:
                    best_veg_count = counts[b_idx, vc]
                    best_veg_cls = vc
            if best_veg_cls is not None:
                mode_corr[b_idx] = best_veg_cls

    out[f"{prefix}__mode_corrected"] = mode_corr
    return out


def extract_zonal_dates(label_array, data, prefix, n_buildings,
                         min_pixels=DEFAULT_MIN_PIXELS,
                         zero_means_no_event=True):
    """Per-building reduction for date rasters (event timing).

    Per V2 design these are METADATA, not features (they encode the battle
    calendar). NB05b builders adding these columns must register them in
    META_COLS_SET via metadata_filter.py.

    Date encoding: YYYYMMDD integer or days-since-invasion. Either way
    integer-valued. No floats. Sentinel -1 for "missing".

    Returns:
      {prefix}__min_nonzero    int32; -1 if no nonzero events in footprint
      {prefix}__max            int32; max date (incl 0 if all-zero); -1 if absent
      {prefix}__count_unique   int32; distinct nonzero dates; 0 if none
      {prefix}__n_pixels_valid int32; valid pixel count
    """
    if data.shape != label_array.shape:
        raise ValueError(f"data shape {data.shape} != label_array shape {label_array.shape}")

    flat_lbl = label_array.ravel()
    flat_val = data.ravel()
    in_bldg = (flat_lbl > 0) & (flat_lbl <= n_buildings)
    val_clean = np.where(np.isfinite(flat_val), flat_val, 0)

    df = pd.DataFrame({
        'b': flat_lbl[in_bldg].astype(np.int32),
        'v': val_clean[in_bldg].astype(np.int64),
    })

    out = {
        f"{prefix}__min_nonzero":    np.full(n_buildings, -1, dtype=np.int32),
        f"{prefix}__max":            np.full(n_buildings, -1, dtype=np.int32),
        f"{prefix}__count_unique":   np.zeros(n_buildings, dtype=np.int32),
        f"{prefix}__n_pixels_valid": np.zeros(n_buildings, dtype=np.int32),
    }

    if df.empty:
        return out

    g = df.groupby('b', sort=True)
    pixel_count = g.size()
    idx = pixel_count.index.values - 1
    out[f"{prefix}__n_pixels_valid"][idx] = pixel_count.values.astype(np.int32)
    has_pixels = pixel_count.values >= min_pixels

    # max — gated by has_pixels
    max_v = g['v'].max().values.astype(np.int32)
    max_v_gated = np.where(has_pixels, max_v, -1)
    out[f"{prefix}__max"][idx] = max_v_gated

    if zero_means_no_event:
        # Filter to nonzero rows BEFORE groupby — no python loop
        df_nz = df[df['v'] > 0]
        if not df_nz.empty:
            g_nz = df_nz.groupby('b', sort=True)
            min_nz = g_nz['v'].min()
            unique_count = g_nz['v'].nunique()
            # gate by total pixel count of these buildings (using reindex lookup)
            nz_b_ids = min_nz.index.values
            has_pixels_nz = pixel_count.reindex(nz_b_ids).values >= min_pixels
            slots = nz_b_ids - 1
            out[f"{prefix}__min_nonzero"][slots] = np.where(
                has_pixels_nz, min_nz.values.astype(np.int32), -1
            )
            out[f"{prefix}__count_unique"][slots] = np.where(
                has_pixels_nz, unique_count.values.astype(np.int32), 0
            )
    else:
        min_v = g['v'].min().values.astype(np.int32)
        unique_count = g['v'].nunique().values.astype(np.int32)
        out[f"{prefix}__min_nonzero"][idx] = np.where(has_pixels, min_v, -1)
        out[f"{prefix}__count_unique"][idx] = np.where(has_pixels, unique_count, 0)

    return out


def date_metadata_columns(prefix):
    """Return the list of column names produced by extract_zonal_dates() given
    a prefix. Use this to extend META_COLS_SET so the manifest correctly tags
    these as metadata, not features.
    """
    return [
        f"{prefix}__min_nonzero",
        f"{prefix}__max",
        f"{prefix}__count_unique",
        f"{prefix}__n_pixels_valid",
    ]


def _print_banner():
    print(f"  aggregation_helpers loaded (v29.1 pandas-fast):")
    print(f"    extract_zonal_aware (pandas groupby)")
    print(f"    extract_zonal_categorical (np.bincount)")
    print(f"    extract_zonal_dates (pandas groupby, int32 throughout)")
    print(f"    DEFAULT_MIN_PIXELS={DEFAULT_MIN_PIXELS}")
    print(f"    VEG_OVERRIDE_THRESHOLD={VEG_OVERRIDE_THRESHOLD}")
    print(f"    VEG_HONEST_THRESHOLD={VEG_HONEST_THRESHOLD}")


if __name__ == '__main__':
    _print_banner()
