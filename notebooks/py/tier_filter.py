# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
tier_filter.py
Filters cities_df (SAR) and ms_cities_df (MS) by tier/city/scene count.
Extracted from Cell 14A-FILTER.

Notebook usage:
    from tier_filter import run as run_tier_filter
    cities_df_filtered, ms_cities_df_filtered, SAR_CITIES_TO_PROCESS, MS_CITIES_TO_PROCESS, CITIES_TO_PROCESS = run_tier_filter(
        cities_df=cities_df,
        ms_cities_df=ms_cities_df,
        tier_selection=TIER_SELECTION,
        city_selection=CITY_SELECTION,
        min_pre_scenes=MIN_PRE_SCENES,
        min_post_scenes=MIN_POST_SCENES,
        require_sar_aligned=REQUIRE_SAR_ALIGNED,
    )
"""

import pandas as pd
from datetime import datetime


def run(cities_df, ms_cities_df=None, tier_selection=None, city_selection=None,
        min_pre_scenes=1, min_post_scenes=1, require_sar_aligned=True):
    """
    Args:
        cities_df:           pd.DataFrame - SAR cities (from scene_loader)
        ms_cities_df:        pd.DataFrame - MS cities (from ms_scene_loader)
        tier_selection:       list or "ALL" - e.g. [0] or [0,1] or "ALL"
        city_selection:       list or None - e.g. ["Mariupol"] or None
        min_pre_scenes:       int
        min_post_scenes:      int
        require_sar_aligned:  bool

    Returns:
        (cities_df_filtered, ms_cities_df_filtered, SAR_CITIES_TO_PROCESS, MS_CITIES_TO_PROCESS, CITIES_TO_PROCESS)
    """

    # ============================================================================
    # VALIDATION
    # ============================================================================
    print("=" * 80)
    print("CELL 14A-FILTER: TIER SELECTION FILTER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    print(f"Source dataframes:")
    print(f"  cities_df (SAR):    {len(cities_df)} cities")
    if ms_cities_df is not None:
        print(f"  ms_cities_df (MS):  {len(ms_cities_df)} cities")
    else:
        print(f"  ms_cities_df (MS):  not provided (SAR-only mode)")

    # ============================================================================
    # FILTER CONFIGURATION
    # ============================================================================
    print(f"\nFILTER CONFIGURATION:")
    print(f"  TIER_SELECTION:     {tier_selection}")
    print(f"  CITY_SELECTION:     {city_selection}")
    print(f"  MIN_PRE_SCENES:     {min_pre_scenes}")
    print(f"  MIN_POST_SCENES:    {min_post_scenes}")
    print(f"  REQUIRE_SAR_ALIGNED: {require_sar_aligned}")

    # ============================================================================
    # FILTER SAR DATAFRAME
    # ============================================================================
    print(f"\n{'='*80}")
    print("FILTERING SAR DATAFRAME (cities_df)")
    print(f"{'='*80}")

    # Start with all cities
    sar_filtered = cities_df.copy()
    print(f"  Starting with {len(sar_filtered)} cities")

    # Tier filter
    if tier_selection != "ALL":
        sar_filtered = sar_filtered[sar_filtered['tier'].isin(tier_selection)]
        print(f"  After tier filter: {len(sar_filtered)} cities")

    # City name filter
    if city_selection is not None:
        sar_filtered = sar_filtered[sar_filtered['city'].isin(city_selection)]
        print(f"  After city filter: {len(sar_filtered)} cities")

    # Scene count filters
    sar_filtered = sar_filtered[sar_filtered['pre_scenes'].apply(len) >= min_pre_scenes]
    print(f"  After min pre-scenes filter: {len(sar_filtered)} cities")

    sar_filtered = sar_filtered[sar_filtered['post_scenes'].apply(len) >= min_post_scenes]
    print(f"  After min post-scenes filter: {len(sar_filtered)} cities")

    # Sort by tier, then city name
    sar_filtered = sar_filtered.sort_values(['tier', 'city']).reset_index(drop=True)

    # ============================================================================
    # FILTER MS DATAFRAME
    # ============================================================================
    if ms_cities_df is not None:
        print(f"\n{'='*80}")
        print("FILTERING MS DATAFRAME (ms_cities_df)")
        print(f"{'='*80}")

        # Start with all cities
        ms_filtered = ms_cities_df.copy()
        print(f"  Starting with {len(ms_filtered)} cities")

        # Tier filter
        if tier_selection != "ALL":
            ms_filtered = ms_filtered[ms_filtered['tier'].isin(tier_selection)]
            print(f"  After tier filter: {len(ms_filtered)} cities")

        # City name filter
        if city_selection is not None:
            ms_filtered = ms_filtered[ms_filtered['city'].isin(city_selection)]
            print(f"  After city filter: {len(ms_filtered)} cities")

        # Scene count filters
        ms_filtered = ms_filtered[ms_filtered['pre_scenes'].apply(len) >= min_pre_scenes]
        print(f"  After min pre-scenes filter: {len(ms_filtered)} cities")

        ms_filtered = ms_filtered[ms_filtered['post_scenes'].apply(len) >= min_post_scenes]
        print(f"  After min post-scenes filter: {len(ms_filtered)} cities")

        # SAR alignment filter
        if require_sar_aligned:
            ms_filtered = ms_filtered[ms_filtered['sar_aligned'] == True]
            print(f"  After SAR alignment filter: {len(ms_filtered)} cities")

        # Sort by tier, then city name
        ms_filtered = ms_filtered.sort_values(['tier', 'city']).reset_index(drop=True)
    else:
        ms_filtered = pd.DataFrame(columns=['city', 'tier'])

    # ============================================================================
    # CREATE OUTPUT VARIABLES
    # ============================================================================
    cities_df_filtered = sar_filtered
    ms_cities_df_filtered = ms_filtered

    # City name lists for downstream cells
    SAR_CITIES_TO_PROCESS = cities_df_filtered['city'].tolist()
    MS_CITIES_TO_PROCESS = ms_cities_df_filtered['city'].tolist()

    # Combined list (cities in both)
    CITIES_TO_PROCESS = list(set(SAR_CITIES_TO_PROCESS) & set(MS_CITIES_TO_PROCESS))
    CITIES_TO_PROCESS.sort()

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f"\n{'='*80}")
    print("FILTER RESULTS")
    print(f"{'='*80}")

    print(f"\nSAR cities to process: {len(SAR_CITIES_TO_PROCESS)}")
    print(f"MS cities to process:  {len(MS_CITIES_TO_PROCESS)}")
    print(f"Cities in BOTH:        {len(CITIES_TO_PROCESS)}")

    # Tier breakdown
    print(f"\nTier breakdown (filtered):")
    if len(cities_df_filtered) > 0:
        sar_tier_counts = cities_df_filtered.groupby('tier')['city'].count()
        for tier, count in sar_tier_counts.items():
            ms_count = len(ms_filtered[ms_filtered['tier'] == tier]) if len(ms_filtered) > 0 else 0
            print(f"  Tier {tier}: {count} SAR, {ms_count} MS")

    # Cities only in SAR
    sar_only = set(SAR_CITIES_TO_PROCESS) - set(MS_CITIES_TO_PROCESS)
    if sar_only:
        print(f"\nCities in SAR but not MS ({len(sar_only)}):")
        for city in sorted(sar_only):
            print(f"    - {city}")

    # Cities only in MS
    ms_only = set(MS_CITIES_TO_PROCESS) - set(SAR_CITIES_TO_PROCESS)
    if ms_only:
        print(f"\nCities in MS but not SAR ({len(ms_only)}):")
        for city in sorted(ms_only):
            print(f"    - {city}")

    # Detailed city list
    print(f"\n{'='*80}")
    if CITIES_TO_PROCESS:
        print("CITIES TO PROCESS (in both SAR and MS):")
    else:
        print("CITIES TO PROCESS (SAR only):")
    print(f"{'='*80}")

    cities_for_detail = CITIES_TO_PROCESS if CITIES_TO_PROCESS else SAR_CITIES_TO_PROCESS
    for idx, city in enumerate(cities_for_detail, 1):
        sar_row = cities_df_filtered[cities_df_filtered['city'] == city].iloc[0]

        tier = sar_row.get('tier', '?')
        battle_end = sar_row['battle_end']
        battle_end_str = battle_end.date() if pd.notna(battle_end) else 'ongoing'

        sar_pre = len(sar_row['pre_scenes'])
        sar_post = len(sar_row['post_scenes'])
        sar_battle = len(sar_row.get('battle_scenes', []))

        sar_str = f"{sar_pre}+{sar_post}"
        if sar_battle:
            sar_str += f"+{sar_battle}b"

        ms_str = ""
        if len(ms_cities_df_filtered) > 0 and city in ms_cities_df_filtered['city'].values:
            ms_row = ms_cities_df_filtered[ms_cities_df_filtered['city'] == city].iloc[0]
            ms_pre = len(ms_row['pre_scenes'])
            ms_post = len(ms_row['post_scenes'])
            ms_battle = len(ms_row.get('battle_scenes', []))
            ms_str = f" | MS: {ms_pre}+{ms_post}"
            if ms_battle:
                ms_str += f"+{ms_battle}b"

        print(f"  {idx:2d}. [T{tier}] {city:<20} | SAR: {sar_str}{ms_str} | {sar_row['battle_start'].date()} - {battle_end_str}")

    # ============================================================================
    # EXPORT INFO
    # ============================================================================
    print(f"\n{'='*80}")
    print("EXPORTED VARIABLES")
    print(f"{'='*80}")
    print(f"  cities_df_filtered     - SAR dataframe ({len(cities_df_filtered)} cities)")
    print(f"  ms_cities_df_filtered  - MS dataframe ({len(ms_cities_df_filtered)} cities)")
    print(f"  SAR_CITIES_TO_PROCESS  - List of {len(SAR_CITIES_TO_PROCESS)} SAR city names")
    print(f"  MS_CITIES_TO_PROCESS   - List of {len(MS_CITIES_TO_PROCESS)} MS city names")
    print(f"  CITIES_TO_PROCESS      - List of {len(CITIES_TO_PROCESS)} cities in BOTH")

    print(f"\nUse these in downstream cells:")
    print(f"  - Cell 14B: for city in SAR_CITIES_TO_PROCESS")
    print(f"  - Cell 15A: for city in MS_CITIES_TO_PROCESS")
    print(f"  - Combined: for city in CITIES_TO_PROCESS")

    return cities_df_filtered, ms_cities_df_filtered, SAR_CITIES_TO_PROCESS, MS_CITIES_TO_PROCESS, CITIES_TO_PROCESS
