# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
scene_loader.py
Loads discovered SAR scene metadata into cities_df DataFrame.
Extracted from Cell 14A3.

Notebook usage:
    from scene_loader import run as run_scene_loader
    cities_df = run_scene_loader(
        cities_dir=CITIES_DIR,
        sar_metadata_dir=SAR_METADATA_DIR,
        cities_pkl_file=CITIES_PKL_FILE,
        force_rerun=FORCE_RERUN,
    )
"""

import json
import pandas as pd
from pathlib import Path

from aoi_date_extend_loader import load_aoi_gdf


def build_cities_dataframe(cities_dir, sar_metadata_dir):
    print("Building cities dataframe from JSON metadata files...")

    cities_dir = Path(cities_dir)
    sar_metadata_dir = Path(sar_metadata_dir)
    cities_data = []

    for metadata_file in sar_metadata_dir.glob("*_scene_metadata.json"):
        city_name = metadata_file.stem.replace('_scene_metadata', '')
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
        tier = metadata.get('tier', int(gdf.iloc[0].get('tier', 1)))
        product_types = metadata.get('product_types', ['SLC'])
        recommended_orbit = metadata.get('recommended_orbit')
        common_orbits = metadata.get('common_orbits', [])
        orbits_metadata = metadata.get('orbits', {})

        if recommended_orbit and str(recommended_orbit) in orbits_metadata:
            rec_orbit_data = orbits_metadata[str(recommended_orbit)]
            pre_scenes = rec_orbit_data.get('pre_scenes', [])
            post_scenes = rec_orbit_data.get('post_scenes', [])
            battle_scenes = rec_orbit_data.get('battle_scenes', [])
        else:
            pre_scenes = metadata.get('pre_scenes', [])
            post_scenes = metadata.get('post_scenes', [])
            battle_scenes = []

        cities_data.append({
            'city': city_name,
            'tier': int(tier),
            'product_types': product_types,
            'battle_start': battle_start,
            'battle_end': battle_end,
            'conflict_ongoing': conflict_ongoing,
            'boundary': gdf.geometry.iloc[0],
            'bounds': gdf.geometry.iloc[0].bounds,
            'recommended_orbit': recommended_orbit,
            'common_orbits': common_orbits,
            'pre_scenes': pre_scenes,
            'post_scenes': post_scenes,
            'battle_scenes': battle_scenes,
            'orbits': orbits_metadata
        })

    if not cities_data:
        raise ValueError(f"No scene metadata found in {sar_metadata_dir}")

    cities_df = pd.DataFrame(cities_data)
    cities_df['battle_start'] = pd.to_datetime(cities_df['battle_start'])
    cities_df['battle_end'] = pd.to_datetime(cities_df['battle_end'])

    # Sort by tier then city name
    cities_df = cities_df.sort_values(['tier', 'city']).reset_index(drop=True)

    print(f"  Built dataframe with {len(cities_df)} cities")
    return cities_df


def run(cities_dir, sar_metadata_dir, cities_pkl_file, force_rerun=False):
    """
    Args:
        cities_dir:       Path - CITIES_DIR
        sar_metadata_dir: Path - SAR_METADATA_DIR
        cities_pkl_file:  Path - CITIES_PKL_FILE
        force_rerun:      bool

    Returns:
        cities_df: pd.DataFrame
    """
    cities_dir = Path(cities_dir)
    sar_metadata_dir = Path(sar_metadata_dir)
    cities_pkl_file = Path(cities_pkl_file)

    print("=" * 80)
    print("CELL 14A3: LOAD SCENES FROM GOOGLE DRIVE")
    print("=" * 80)

    if force_rerun or not cities_pkl_file.exists():
        if force_rerun:
            print(f"\n  FORCE_RERUN = True")
            print(f"  Rebuilding cities dataframe from source JSON files...")
        else:
            print(f"\n  Pickle file not found: {cities_pkl_file}")
            print(f"  Building cities dataframe from source JSON files...")

        cities_df = build_cities_dataframe(cities_dir, sar_metadata_dir)

        print(f"\n  Saving cities dataframe to pickle...")
        cities_df.to_pickle(cities_pkl_file)
        print(f"  Saved to: {cities_pkl_file}")
    else:
        print(f"\n  Loading cities dataframe from pickle...")
        print(f"  Source: {cities_pkl_file}")
        cities_df = pd.read_pickle(cities_pkl_file)
        print(f"  Loaded {len(cities_df)} cities from cache")

    print(f"\n{'='*80}")
    print(f"CITIES DATAFRAME LOADED")
    print(f"{'='*80}")
    print(f"Source: {sar_metadata_dir}")
    print(f"Cache: {cities_pkl_file}")

    print(f"\nCities summary:")
    for idx, row in cities_df.iterrows():
        battle_end_str = row['battle_end'].date() if pd.notna(row['battle_end']) else 'ongoing'
        battle_str = f"{len(row.get('battle_scenes', []))} battle, " if row.get('battle_scenes') else ""
        print(f"  {idx+1:2d}. [T{row['tier']}] {row['city']:20s}: {len(row['pre_scenes'])} pre, {len(row['post_scenes'])} post, {battle_str}"
              f"Battle: {row['battle_start'].date()} - {battle_end_str} | "
              f"Orbit: {row['recommended_orbit']}")

    print(f"\nTier distribution:")
    for tier in sorted(cities_df['tier'].unique()):
        count = len(cities_df[cities_df['tier'] == tier])
        print(f"  Tier {tier}: {count} cities")

    print(f"\nCell 14A3: Data loaded from Google Drive")
    print(f"Set FORCE_RERUN=True to rebuild from JSON sources")

    return cities_df
