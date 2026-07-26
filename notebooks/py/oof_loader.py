# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
oof_loader.py -- Discover, load, and normalize OOF prediction parquets.

Handles both V2 (building_id, polygon/zonal) and V3+ (point_id, point extraction)
schemas. Provides a unified interface with sample_id column and group labels.

bda -- Building Damage Assessment using satellite imagery
Copyright (C) 2024-2026 Marco Heinzen
SPDX-License-Identifier: AGPL-3.0-or-later
"""

from __future__ import annotations

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUP_A = "buildings_v2"     # V1-V2: building_id, polygon zonal stats
GROUP_B = "points_v3plus"    # V3-V7: point_id, pixel at centroid

SAMPLE_ID_COL = "sample_id"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_oof_parquets(search_roots: list[Path]) -> pd.DataFrame:
    """Glob all oof_*.parquet files, detect schema, build inventory DataFrame.

    Returns DataFrame with columns:
        file, notebook, model_id, experiment_id, variant_id, n_rows,
        n_damaged, TP, FN, FP, TN, recall, precision, path,
        id_col, group
    """
    all_paths = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            print(f"  WARNING: root does not exist: {root}")
            continue
        paths = []
        for _pat in ("oof_predictions/oof_*.parquet", "oof/oof_*.parquet"):
            paths.extend(root.rglob(_pat))
        paths.extend(root.glob("oof_*.parquet"))
        paths = sorted(set(paths))
        print(f"  {root}: {len(paths)} parquets")
        all_paths.extend(paths)

    print(f"\n  Total: {len(all_paths)} OOF parquet files discovered")

    if not all_paths:
        raise FileNotFoundError(
            "No OOF parquets found. Run the retrofitted NB07/NB08/NB09 notebooks first "
            "(they must write to OOF_DIR = RESULTS_ROOT/<nb_id>/oof_predictions/)."
        )

    inv_rows = []
    for p in all_paths:
        try:
            head = pd.read_parquet(p)
        except Exception as e:
            print(f"  WARNING: can't read {p.name}: {e}")
            continue

        # detect schema
        has_bid = "building_id" in head.columns
        has_pid = "point_id" in head.columns
        if has_pid:
            id_col = "point_id"
            group = GROUP_B
        elif has_bid:
            id_col = "building_id"
            group = GROUP_A
        else:
            print(f"  WARNING: {p.name} has neither building_id nor point_id, skipping")
            continue

        cm = head["cm_class"].value_counts()
        parts = p.parts
        nb_guess = next(
            (seg.lower() for seg in parts
             if seg.lower().startswith("nb0") or seg.lower().startswith("nb1")),
            "unknown",
        )

        inv_rows.append({
            "file": str(p.relative_to(p.parents[2]) if len(p.parents) >= 3 else p),
            "notebook": nb_guess,
            "model_id": head["model_id"].iloc[0],
            "experiment_id": head["experiment_id"].iloc[0],
            "variant_id": head["variant_id"].iloc[0] if "variant_id" in head.columns else "",
            "n_rows": len(head),
            "n_damaged": int((head["y_true"] == 1).sum()),
            "TP": int(cm.get("TP", 0)),
            "FN": int(cm.get("FN", 0)),
            "FP": int(cm.get("FP", 0)),
            "TN": int(cm.get("TN", 0)),
            "path": str(p),
            "id_col": id_col,
            "group": group,
        })

    inv_df = pd.DataFrame(inv_rows)
    inv_df["recall"] = inv_df["TP"] / (inv_df["TP"] + inv_df["FN"]).replace(0, np.nan)
    inv_df["precision"] = inv_df["TP"] / (inv_df["TP"] + inv_df["FP"]).replace(0, np.nan)

    print(f"\n  Breakdown by group:")
    for grp, g in inv_df.groupby("group"):
        print(f"    {grp:25s}: {len(g):4d} parquets  ({g['model_id'].nunique()} unique model_ids)")
    print(f"\n  Breakdown by notebook:")
    for nb, g in inv_df.groupby("notebook"):
        print(f"    {nb:35s}: {len(g):4d} parquets  ({g['model_id'].nunique()} unique model_ids)")

    return inv_df


# ---------------------------------------------------------------------------
# Loading + dedup + normalization
# ---------------------------------------------------------------------------

def load_oof_group(inv_df: pd.DataFrame, group: str) -> Optional[pd.DataFrame]:
    """Load all OOF parquets for one group (GROUP_A or GROUP_B).

    Deduplicates by model_id (keeps latest experiment_id).
    Normalizes the id column to SAMPLE_ID_COL.
    Returns None if group has no data.
    """
    grp_inv = inv_df[inv_df["group"] == group].copy()
    if len(grp_inv) == 0:
        print(f"  {group}: no parquets found")
        return None

    # dedup: per model_id, keep latest experiment_id
    grp_inv = grp_inv.sort_values(
        ["model_id", "experiment_id"], ascending=[True, False]
    )
    latest = grp_inv.drop_duplicates("model_id", keep="first")
    print(f"  {group}: {len(grp_inv)} parquets -> {len(latest)} after dedup")

    id_col = latest["id_col"].iloc[0]

    dfs = []
    for _, row in latest.iterrows():
        try:
            d = pd.read_parquet(row["path"])
            dfs.append(d)
        except Exception as e:
            print(f"    WARNING: can't load {row['file']}: {e}")

    if not dfs:
        return None

    oof = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # normalize id column
    if id_col in oof.columns:
        oof[SAMPLE_ID_COL] = oof[id_col]
    else:
        raise KeyError(f"Expected id column '{id_col}' not found in oof columns: {oof.columns.tolist()}")

    oof["group"] = group

    print(f"    Loaded {len(oof):,} rows, {oof['model_id'].nunique()} models, "
          f"{oof[SAMPLE_ID_COL].nunique():,} unique samples, "
          f"{oof['city'].nunique()} cities")

    return oof


def load_all_oof(inv_df: pd.DataFrame) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load and normalize both groups. Returns (oof_A, oof_B)."""
    print("=" * 70)
    print("LOAD OOF + DEDUP + NORMALIZE")
    print("=" * 70)
    oof_a = load_oof_group(inv_df, GROUP_A)
    oof_b = load_oof_group(inv_df, GROUP_B)
    return oof_a, oof_b


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata_both(dataset_root_v2: Path, dataset_root_v3: Path,
                       tiers: list[int], cities: list[str],
                       load_tier_parquets_fn) -> dict:
    """Load building and point metadata tables. Returns dict with keys:
       'buildings_df', 'points_df', 'coord_map_A', 'coord_map_B'

    coord_map_A: building_id -> (x, y) in UTM
    coord_map_B: point_id -> (x, y) in UTM
    """
    result = {"buildings_df": None, "points_df": None,
              "coord_map_A": {}, "coord_map_B": {}}

    # V2 buildings
    bldg_pattern = str(dataset_root_v2 / "bda_buildings_t{tier}.parquet")
    try:
        df_bldg = load_tier_parquets_fn(bldg_pattern, tiers)
        df_bldg = df_bldg[df_bldg["city"].isin(cities)].copy()
        df_bldg = df_bldg[df_bldg["damage_binary"] >= 0].copy()
        result["buildings_df"] = df_bldg
        # coord map: building_id -> (centroid_x, centroid_y)
        result["coord_map_A"] = dict(zip(
            df_bldg["building_id"],
            zip(df_bldg["centroid_x"], df_bldg["centroid_y"])
        ))
        print(f"  V2 buildings: {len(df_bldg):,}  "
              f"(damaged={int((df_bldg['damage_binary']==1).sum()):,})")
    except Exception as e:
        print(f"  WARNING: could not load V2 buildings: {e}")

    # V3+ points -- try v3 through v7, use first found
    for v in ["v3", "v4", "v5", "v6", "v7"]:
        pts_root = dataset_root_v3.parent / v
        pts_pattern = str(pts_root / "bda_points_t{tier}.parquet")
        try:
            df_pts = load_tier_parquets_fn(pts_pattern, tiers)
            df_pts = df_pts[df_pts["city"].isin(cities)].copy()
            df_pts = df_pts[df_pts["damage_binary"] >= 0].copy()
            result["points_df"] = df_pts
            # coord map: point_id -> (x_utm, y_utm)
            result["coord_map_B"] = dict(zip(
                df_pts["point_id"],
                zip(df_pts["x_utm"], df_pts["y_utm"])
            ))
            print(f"  {v} points: {len(df_pts):,}  "
                  f"(damaged={int((df_pts['damage_binary']==1).sum()):,})  "
                  f"[from {pts_root.name}]")
            break
        except Exception:
            continue

    if result["points_df"] is None:
        print("  WARNING: no V3-V7 points metadata found")

    return result


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def check_y_true_consistency(oof: pd.DataFrame, group_label: str):
    """Check that y_true is consistent per sample across models."""
    yt_check = oof.groupby(SAMPLE_ID_COL)["y_true"].nunique()
    n_bad = int((yt_check > 1).sum())
    if n_bad > 0:
        print(f"  WARNING [{group_label}]: {n_bad} samples have inconsistent y_true across models")
    else:
        print(f"  [{group_label}] y_true consistent across all models  OK")


# ---------------------------------------------------------------------------
# Model sanity (leakage / degeneracy detection)
# ---------------------------------------------------------------------------

def build_model_sanity(oof: pd.DataFrame, group_label: str,
                       leakage_markers: list[str]) -> pd.DataFrame:
    """Build per-model sanity table: AUC, CM counts, leakage/degeneracy flags."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for mid, mdf in oof.groupby("model_id"):
        yt = mdf["y_true"].values
        yp = mdf["y_proba"].values
        cm = mdf["cm_class"].value_counts()
        vid = mdf["variant_id"].iloc[0] if "variant_id" in mdf.columns else ""
        try:
            auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan
        except Exception:
            auc = np.nan
        is_leaky = any(m in str(vid) for m in leakage_markers) if leakage_markers else False
        is_degen = (cm.get("TP", 0) + cm.get("FP", 0) == 0) or \
                   (cm.get("TN", 0) + cm.get("FN", 0) == 0)
        rows.append({
            "model_id": mid,
            "variant_id": vid,
            "group": group_label,
            "auc": auc,
            "n_samples": len(mdf),
            "n_cities": mdf["city"].nunique(),
            "TP": int(cm.get("TP", 0)),
            "FN": int(cm.get("FN", 0)),
            "FP": int(cm.get("FP", 0)),
            "TN": int(cm.get("TN", 0)),
            "is_leaky": is_leaky,
            "is_degenerate": is_degen,
        })
    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print(f"  [{group_label}] {len(df)} models: "
          f"{int(df['is_leaky'].sum())} leaky, "
          f"{int(df['is_degenerate'].sum())} degenerate")
    return df


# ---------------------------------------------------------------------------
# Pivot helpers (for pairwise agreement)
# ---------------------------------------------------------------------------

def pivot_oof(oof: pd.DataFrame, value_col: str = "y_pred") -> pd.DataFrame:
    """Pivot to wide: rows=sample_id, cols=model_id, values=value_col."""
    return oof.pivot_table(
        index=SAMPLE_ID_COL, columns="model_id",
        values=value_col, aggfunc="first"
    )


# ---------------------------------------------------------------------------
# Spatial data helpers (for 11b)
# ---------------------------------------------------------------------------

def build_gdf_by_city(oof: pd.DataFrame, coord_map: dict) -> dict:
    """Build per-city DataFrames with sample_id + coordinates.
    Returns {city: DataFrame with sample_id, coord_x, coord_y}.
    """
    # get unique samples with city
    samples = oof[[SAMPLE_ID_COL, "city"]].drop_duplicates()
    samples["coord_x"] = samples[SAMPLE_ID_COL].map(lambda sid: coord_map.get(sid, (np.nan, np.nan))[0])
    samples["coord_y"] = samples[SAMPLE_ID_COL].map(lambda sid: coord_map.get(sid, (np.nan, np.nan))[1])
    # drop samples without coords
    before = len(samples)
    samples = samples.dropna(subset=["coord_x", "coord_y"])
    if len(samples) < before:
        print(f"    Dropped {before - len(samples)} samples without coordinates")

    gdf = {}
    for city, cdf in samples.groupby("city"):
        gdf[city] = cdf.copy()
    return gdf
