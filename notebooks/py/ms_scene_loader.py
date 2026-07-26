# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
ms_scene_loader.py
Loads discovered MS scene metadata into ms_cities_df DataFrame.
Extracted from Cell 14A3-MS.

Notebook usage:
    from ms_scene_loader import run as run_ms_scene_loader
    ms_cities_df = run_ms_scene_loader(
        cities_dir=CITIES_DIR,
        ms_metadata_dir=MS_METADATA_DIR,
        ms_cities_pkl_file=MS_CITIES_PKL_FILE,
        force_rerun=FORCE_RERUN,
    )
"""

import json
import pandas as pd
from pathlib import Path

from aoi_date_extend_loader import load_aoi_gdf


def build_ms_cities_dataframe(cities_dir, ms_metadata_dir):
    print("Building MS cities dataframe from JSON metadata files...")

    cities_dir = Path(cities_dir)
    ms_metadata_dir = Path(ms_metadata_dir)
    cities_data = []

    for metadata_file in ms_metadata_dir.glob("*_ms_scene_metadata.json"):

        city_name = metadata_file.stem.replace('_ms_scene_metadata', '')

        city_dir = cities_dir / city_name
        aoi_file = city_dir / "AOI.geojson"

        if not aoi_file.exists():
            print(f"  Missing AOI.geojson for {city_name}")
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        gdf = load_aoi_gdf(city_name, cities_dir)

        battle_start = metadata.get('battle_start')
        battle_end = metadata.get('battle_end')
        conflict_ongoing = metadata.get('conflict_ongoing', False)
        tier = metadata.get('tier', 99)
        sar_aligned = metadata.get('sar_aligned', False)

        pre_window = metadata.get('pre_window', {})
        post_window = metadata.get('post_window', {})
        battle_window = metadata.get('battle_window', {})
        baseline_window = metadata.get('baseline_window', {})
        post_baseline_window = metadata.get('post_baseline_window', {})

        pre_scenes = pre_window.get('scenes', [])
        post_scenes = post_window.get('scenes', [])
        battle_scenes = battle_window.get('scenes', [])
        baseline_scenes = baseline_window.get('scenes', [])
        post_baseline_scenes = post_baseline_window.get('scenes', [])
        pre_sar_dates = pre_window.get('sar_dates', [])
        post_sar_dates = post_window.get('sar_dates', [])
        pre_cloud_threshold = pre_window.get('cloud_threshold_used', 10.0)
        post_cloud_threshold = post_window.get('cloud_threshold_used', 10.0)
        battle_cloud_threshold = battle_window.get('cloud_threshold_used', 10.0)

        cities_data.append({
            'city': city_name,
            'tier': tier,
            'battle_start': battle_start,
            'battle_end': battle_end,
            'conflict_ongoing': conflict_ongoing,
            'boundary': gdf.geometry.iloc[0],
            'bounds': gdf.geometry.iloc[0].bounds,
            'sar_aligned': sar_aligned,
            'pre_sar_dates': pre_sar_dates,
            'post_sar_dates': post_sar_dates,
            'pre_scenes': pre_scenes,
            'post_scenes': post_scenes,
            'battle_scenes': battle_scenes,
            'baseline_scenes': baseline_scenes,
            'post_baseline_scenes': post_baseline_scenes,
            'pre_cloud_threshold': pre_cloud_threshold,
            'post_cloud_threshold': post_cloud_threshold,
            'battle_cloud_threshold': battle_cloud_threshold,
            'total_scenes': len(pre_scenes) + len(post_scenes) + len(battle_scenes) + len(baseline_scenes) + len(post_baseline_scenes)
        })

    if not cities_data:
        raise ValueError(f"No MS scene metadata found in {ms_metadata_dir}")

    ms_cities_df = pd.DataFrame(cities_data)
    ms_cities_df['battle_start'] = pd.to_datetime(ms_cities_df['battle_start'])
    ms_cities_df['battle_end'] = pd.to_datetime(ms_cities_df['battle_end'])

    ms_cities_df = ms_cities_df.sort_values(['tier', 'city']).reset_index(drop=True)

    print(f"  Built dataframe with {len(ms_cities_df)} cities")

    return ms_cities_df


def run(cities_dir, ms_metadata_dir, ms_cities_pkl_file, force_rerun=False):
    """
    Args:
        cities_dir:         Path - CITIES_DIR
        ms_metadata_dir:    Path - MS_METADATA_DIR
        ms_cities_pkl_file: Path - MS_CITIES_PKL_FILE
        force_rerun:        bool

    Returns:
        ms_cities_df: pd.DataFrame
    """
    cities_dir = Path(cities_dir)
    ms_metadata_dir = Path(ms_metadata_dir)
    ms_cities_pkl_file = Path(ms_cities_pkl_file)

    print("=" * 80)
    print("CELL 14A3-MS: LOAD MULTISPECTRAL SCENES FROM GOOGLE DRIVE")
    print("=" * 80)

    if force_rerun or not ms_cities_pkl_file.exists():

        if force_rerun:
            print(f"\n  FORCE_RERUN = True")
            print(f"  Rebuilding MS cities dataframe from source JSON files...")
        else:
            print(f"\n  Pickle file not found: {ms_cities_pkl_file}")
            print(f"  Building MS cities dataframe from source JSON files...")

        ms_cities_df = build_ms_cities_dataframe(cities_dir, ms_metadata_dir)

        print(f"\n  Saving MS cities dataframe to pickle...")
        ms_cities_df.to_pickle(ms_cities_pkl_file)
        print(f"  Saved to: {ms_cities_pkl_file}")

    else:

        print(f"\n  Loading MS cities dataframe from pickle...")
        print(f"  Source: {ms_cities_pkl_file}")

        ms_cities_df = pd.read_pickle(ms_cities_pkl_file)

        print(f"  Loaded {len(ms_cities_df)} cities from cache")

    print(f"\n{'='*80}")
    print(f"MS CITIES DATAFRAME LOADED")
    print(f"{'='*80}")
    print(f"Source: {ms_metadata_dir}")
    print(f"Cache: {ms_cities_pkl_file}")

    # Summary by tier
    print(f"\nTier summary:")
    tier_counts = ms_cities_df.groupby('tier').agg({
        'city': 'count',
        'total_scenes': 'sum'
    }).rename(columns={'city': 'cities'})
    for tier, row in tier_counts.iterrows():
        print(f"  Tier {tier}: {row['cities']} cities, {row['total_scenes']} total scenes")

    # Cities with missing scenes
    missing_pre = ms_cities_df[ms_cities_df['pre_scenes'].apply(len) == 0]
    missing_post = ms_cities_df[ms_cities_df['post_scenes'].apply(len) == 0]

    if len(missing_pre) > 0:
        print(f"\nCities missing pre-battle scenes ({len(missing_pre)}):")
        for _, row in missing_pre.iterrows():
            print(f"    [T{row['tier']}] {row['city']}")

    if len(missing_post) > 0:
        print(f"\nCities missing post-battle scenes ({len(missing_post)}):")
        for _, row in missing_post.iterrows():
            print(f"    [T{row['tier']}] {row['city']}")

    print(f"\nCities detail:")
    for idx, row in ms_cities_df.iterrows():
        battle_end_str = row['battle_end'].date() if pd.notna(row['battle_end']) else 'ongoing'
        sar_str = "SAR" if row['sar_aligned'] else "no SAR"
        battle_str = f", {len(row['battle_scenes'])} battle" if row['battle_scenes'] else ""
        bl_str = f", {len(row['baseline_scenes'])} preBL" if row['baseline_scenes'] else ""
        pbl_str = f", {len(row['post_baseline_scenes'])} postBL" if row['post_baseline_scenes'] else ""
        print(f"  {idx+1:2d}. [T{row['tier']}] {row['city']:<20} | {len(row['pre_scenes'])} pre, {len(row['post_scenes'])} post{battle_str}{bl_str}{pbl_str} | "
              f"{sar_str} | {row['battle_start'].date()} - {battle_end_str}")

    print(f"\nCell 14A3-MS: MS data loaded from Google Drive")
    print(f"Set FORCE_RERUN=True to rebuild from JSON sources")

    return ms_cities_df
