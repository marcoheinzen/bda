# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
stack_align_rename.py
TIF filename rename from NB03 convention to sensor__measurement__period__stat convention.

Imported by stack_align.py, used as rename_fn in align group config.
Follows STAC/EO-ML conventions from RESEARCH_EO_ML_Dataset_Conventions.md.

Usage:
    from stack_align_rename import rename_tif_for_convention

    # In build_primary_groups / build_derived_groups:
    {
        "SAR_CARD": {
            "src_dir": ..., "dst_subdir": "SAR_CARD/flat",
            "recursive": False,
            "rename_fn": rename_tif_for_convention,
        },
    }
"""

import re


def rename_tif_for_convention(city_name, filename):
    """Rename TIF file from NB03 convention to __ (double-underscore) convention.

    Rules:
      - city name prefix stripped
      - sensor: s1 (Sentinel-1 CARD/COH), s2 (Sentinel-2 MS)
      - polarization lowercase: VV->vv, VH->vh
      - band lowercase: B02->b02, B8A->b8a
      - resolution suffix dropped (all 10m in data_stack after ALIGN)
      - __ separates hierarchy levels
      - _ within level for multi-word (e.g. cloud_mask, coh_vv)
      - dates kept as YYYYMMDD (no __ before date, just __ separator)

    Args:
        city_name: str, e.g. 'Mariupol'
        filename:  str, e.g. 'Mariupol_card_assessment_mean_VV.tif'

    Returns:
        str, renamed filename, e.g. 's1__vv__assessment__mean.tif'
        Falls back to city-prefix-stripped name if no pattern matches.
    """
    stem = filename
    ext = ''
    if filename.endswith('.tif'):
        stem = filename[:-4]
        ext = '.tif'
    elif filename.endswith('.json'):
        # dont rename metadata jsons
        if stem.startswith(f"{city_name}_"):
            return stem[len(city_name) + 1:] + '.json'
        return filename

    # strip city prefix
    stripped = stem
    if stripped.startswith(f"{city_name}_"):
        stripped = stripped[len(city_name) + 1:]

    # --- CARD temporal stats ---
    # card_assessment_mean_VV -> s1__vv__assessment__mean
    # card_baseline_kurtosis_VH -> s1__vh__baseline__kurtosis
    m = re.match(r'^card_(assessment|baseline)_(count|kurtosis|max|mean|median|min|skewness|std)_(VV|VH)$', stripped)
    if m:
        period, stat, pol = m.group(1), m.group(2), m.group(3).lower()
        return f"s1__{pol}__{period}__{stat}{ext}"

    # --- COH baseline stats ---
    # coh_baseline_mean -> s1__coh__baseline__mean
    m = re.match(r'^coh_baseline_(count|max|mean|median|min|std)$', stripped)
    if m:
        stat = m.group(1)
        return f"s1__coh__baseline__{stat}{ext}"

    # --- CARD flat scenes with orbit ---
    # CARD_VH_20220103 -> s1__vh__20220103 (orbit stripped for ML portability)
    # Also handles s1__vh__o029__20220103 -> s1__vh__20220103
    m = re.match(r'^CARD_(VV|VH)_(\d{8})$', stripped)
    if m:
        pol, date = m.group(1).lower(), m.group(2)
        return f"s1__{pol}__{date}{ext}"

    # new convention with orbit: s1__vv__o029__20220103 -> s1__vv__20220103
    m = re.match(r'^s1__(vv|vh)__o\d{3}__(\d{8})$', stripped)
    if m:
        pol, date = m.group(1), m.group(2)
        return f"s1__{pol}__{date}{ext}"

    # --- COH flat scenes with orbit (date pair, lowercase) ---
    # coh_VV_20211222_20220103 -> s1__coh_vv__20211222_20220103
    # s1__coh_vv__o029__20211222_20220103 -> s1__coh_vv__20211222_20220103
    m = re.match(r'^coh_(VV|VH)_(\d{8})_(\d{8})$', stripped)
    if m:
        pol, d1, d2 = m.group(1).lower(), m.group(2), m.group(3)
        return f"s1__coh_{pol}__{d1}_{d2}{ext}"

    # new convention COH with orbit: s1__coh_vv__o029__20220103_20220115 -> s1__coh_vv__20220103_20220115
    m = re.match(r'^s1__coh_(vv|vh)__o\d{3}__(\d{8})_(\d{8})$', stripped)
    if m:
        pol, d1, d2 = m.group(1), m.group(2), m.group(3)
        return f"s1__coh_{pol}__{d1}_{d2}{ext}"

    # --- COH with period tag, undashed dates (uppercase from NB03a old) ---
    # COH_VH_PRE_20220503_20220515 -> s1__coh_vh__20220503_20220515
    m = re.match(r'^COH_(VV|VH)_(PRE|CROSS|POST)_(\d{8})_(\d{8})$', stripped)
    if m:
        pol, d1, d2 = m.group(1).lower(), m.group(3), m.group(4)
        return f"s1__coh_{pol}__{d1}_{d2}{ext}"

    # --- COH with period tag, dashed dates (uppercase from NB03a v16) ---
    # COH_VH_PRE_2022-02-23_2022-03-07 -> s1__coh_vh__20220223_20220307
    m = re.match(r'^COH_(VV|VH)_(PRE|CROSS|POST)_(\d{4})-(\d{2})-(\d{2})_(\d{4})-(\d{2})-(\d{2})$', stripped)
    if m:
        pol = m.group(1).lower()
        d1 = m.group(3) + m.group(4) + m.group(5)
        d2 = m.group(6) + m.group(7) + m.group(8)
        return f"s1__coh_{pol}__{d1}_{d2}{ext}"

    # --- COH flat scenes (single date, from NB03a geocoded) ---
    # coh_VV_20220103 -> s1__coh_vv__20220103
    m = re.match(r'^coh_(VV|VH)_(\d{8})$', stripped)
    if m:
        pol, date = m.group(1).lower(), m.group(2)
        return f"s1__coh_{pol}__{date}{ext}"

    # --- MS flat scenes with band + resolution ---
    # S2_20220108_B02_10m -> s2__b02__20220108
    m = re.match(r'^S2_(\d{8})_(B\d{2}|B8A)_\d+m$', stripped)
    if m:
        date, band = m.group(1), m.group(2).lower()
        return f"s2__{band}__{date}{ext}"

    # --- MS cloud mask ---
    # S2_20220108_cloud_mask -> s2__cloud_mask__20220108
    m = re.match(r'^S2_(\d{8})_cloud_mask$', stripped)
    if m:
        date = m.group(1)
        return f"s2__cloud_mask__{date}{ext}"

    # --- MS SCL ---
    # S2_20220108_SCL_20m -> s2__scl__20220108
    m = re.match(r'^S2_(\d{8})_SCL_\d+m$', stripped)
    if m:
        date = m.group(1)
        return f"s2__scl__{date}{ext}"

    # --- MS visibility ---
    # S2_20220108_visibility -> s2__visibility__20220108
    m = re.match(r'^S2_(\d{8})_visibility$', stripped)
    if m:
        date = m.group(1)
        return f"s2__visibility__{date}{ext}"

    # --- CARD rolling ---
    # CARD_roll2_VH_20220103 -> s1__vh__roll2__20220103
    m = re.match(r'^CARD_roll(\d+)_(VV|VH)_(\d{8})$', stripped)
    if m:
        window, pol, date = m.group(1), m.group(2).lower(), m.group(3)
        return f"s1__{pol}__roll{window}__{date}{ext}"

    # --- COH rolling ---
    # coh_roll3_VV_20220316 -> s1__coh_vv__roll3__20220316
    m = re.match(r'^coh_roll(\d+)_(VV|VH)_(\d{8})$', stripped)
    if m:
        window, pol, date = m.group(1), m.group(2).lower(), m.group(3)
        return f"s1__coh_{pol}__roll{window}__{date}{ext}"

    # --- COH zscore ---
    # coh_zscore_VV_20220103 -> s1__coh_vv__zscore__20220103
    m = re.match(r'^coh_zscore_(VV|VH)_(\d{8})$', stripped)
    if m:
        pol, date = m.group(1).lower(), m.group(2)
        return f"s1__coh_{pol}__zscore__{date}{ext}"

    # --- COH post_baseline ---
    # coh_post_baseline_VV_20220316 -> s1__coh_vv__post_baseline__20220316
    m = re.match(r'^coh_post_baseline_(VV|VH)_(\d{8})$', stripped)
    if m:
        pol, date = m.group(1).lower(), m.group(2)
        return f"s1__coh_{pol}__post_baseline__{date}{ext}"

    # --- Composites (no city prefix) ---
    # composite_B02 -> s2__b02
    m = re.match(r'^composite_(B\d{2}|B8A)$', stripped)
    if m:
        band = m.group(1).lower()
        return f"s2__{band}{ext}"

    # --- dNBR / RBR change detection ---
    # dNBR_prebattle_vs_post -> cd__dnbr__prebattle_vs_post
    m = re.match(r'^(dNBR|RBR)_(.+)$', stripped)
    if m:
        metric, desc = m.group(1).lower(), m.group(2)
        return f"cd__{metric}__{desc}{ext}"

    # --- NBR scenes ---
    # NBR_20220108 -> s2__nbr__20220108
    m = re.match(r'^NBR_(\d{8})$', stripped)
    if m:
        date = m.group(1)
        return f"s2__nbr__{date}{ext}"

    # --- RGB ---
    # rgb_20220108 or S2_rgb_20220108 -> s2__rgb__20220108
    m = re.match(r'^(?:S2_)?rgb_(\d{8})$', stripped)
    if m:
        date = m.group(1)
        return f"s2__rgb__{date}{ext}"

    # --- Landuse classification ---
    # landuse_classification -> lulc__class
    if stripped == 'landuse_classification':
        return f"lulc__class{ext}"

    # --- Spectral indices (in landuse/*/indices/) ---
    # NDVI -> s2__ndvi, NBR -> s2__nbr, BSI -> s2__bsi, etc.
    known_indices = {'NDVI', 'NBR', 'NDWI', 'BSI', 'SAVI', 'EVI', 'NDBI', 'MNDWI', 'BAI', 'MIRBI'}
    if stripped.upper() in known_indices:
        return f"s2__{stripped.lower()}{ext}"

    # --- Fire products ---
    # active_fire -> fire__active, burn_scar -> fire__burn_scar
    if stripped == 'active_fire':
        return f"fire__active{ext}"
    if stripped == 'burn_scar':
        return f"fire__burn_scar{ext}"

    # --- cloud_frequency / obs_count (no city prefix, keep as-is with __ sep) ---
    if stripped == 'cloud_frequency':
        return f"qa__cloud_freq{ext}"
    if stripped == 'obs_count':
        return f"qa__obs_count{ext}"

    # --- Dietrich baseline/assessment files ---
    # These come from gdrive_dietrich2025/{city}/baseline/ or /assessment/
    # Format varies: VV_min.tif, VH_mean.tif, etc.
    m = re.match(r'^(VV|VH)_(count|kurtosis|max|mean|median|min|skewness|std)$', stripped)
    if m:
        pol, stat = m.group(1).lower(), m.group(2)
        # caller must set group hint via dst_subdir to distinguish baseline/assessment
        # return generic name, NB05 discover_rasters uses subdir context
        return f"s1__{pol}__{stat}{ext}"

    # --- FALLBACK: just return stripped (city prefix removed) ---
    return f"{stripped}{ext}"


# ---- Source product rename WITH orbit (for NB03d, NB03b) ----

def rename_source_tif_with_orbit(city_name, filename, date_orbit_map):
    """Rename source TIF to __ convention WITH orbit tag.

    Unlike rename_tif_for_convention (which strips orbits for ML/data_stack),
    this PRESERVES or ADDS orbit tags for source products in SAR_CARD_DIR/SAR_COH_DIR.

    Args:
        city_name:      str, e.g. 'Mariupol'
        filename:       str, e.g. 'Mariupol_CARD_VV_20220103.tif'
        date_orbit_map: dict, {YYYYMMDD: relative_orbit_int}, from zip scanning

    Returns:
        str, renamed filename with orbit, e.g. 's1__vv__o043__20220103.tif'
        Returns filename unchanged if already has orbit tag.
        Returns None if no matching pattern or no orbit found.
    """
    if re.search(r'__o\d{3}__', filename):
        return filename

    stem = filename[:-4] if filename.endswith('.tif') else filename
    ext = '.tif' if filename.endswith('.tif') else ''

    stripped = stem
    if stripped.startswith(f"{city_name}_"):
        stripped = stripped[len(city_name) + 1:]

    def _orbit_tag(date8):
        orb = date_orbit_map.get(date8)
        return f"__o{orb:03d}" if orb else None

    # CARD_VV_20220103
    m = re.match(r'^CARD_(VV|VH)_(\d{8})$', stripped)
    if m:
        pol, date8 = m.group(1).lower(), m.group(2)
        ot = _orbit_tag(date8)
        return f"s1__{pol}{ot}__{date8}{ext}" if ot else None

    # s1__vv__20220103
    m = re.match(r'^s1__(vv|vh)__(\d{8})$', stripped)
    if m:
        pol, date8 = m.group(1), m.group(2)
        ot = _orbit_tag(date8)
        return f"s1__{pol}{ot}__{date8}{ext}" if ot else None

    # COH_VV_PRE_20220103_20220115
    m = re.match(r'^COH_(VV|VH)_(PRE|CROSS|POST)_(\d{8})_(\d{8})$', stripped)
    if m:
        pol, d1, d2 = m.group(1).lower(), m.group(3), m.group(4)
        ot = _orbit_tag(d1) or _orbit_tag(d2)
        return f"s1__coh_{pol}{ot}__{d1}_{d2}{ext}" if ot else None

    # COH_VV_PRE_2022-02-23_2022-03-07
    m = re.match(r'^COH_(VV|VH)_(PRE|CROSS|POST)_(\d{4})-(\d{2})-(\d{2})_(\d{4})-(\d{2})-(\d{2})$', stripped)
    if m:
        pol = m.group(1).lower()
        d1 = m.group(3) + m.group(4) + m.group(5)
        d2 = m.group(6) + m.group(7) + m.group(8)
        ot = _orbit_tag(d1) or _orbit_tag(d2)
        return f"s1__coh_{pol}{ot}__{d1}_{d2}{ext}" if ot else None

    # coh_VV_20220103_20220115
    m = re.match(r'^coh_(VV|VH)_(\d{8})_(\d{8})$', stripped)
    if m:
        pol, d1, d2 = m.group(1).lower(), m.group(2), m.group(3)
        ot = _orbit_tag(d1) or _orbit_tag(d2)
        return f"s1__coh_{pol}{ot}__{d1}_{d2}{ext}" if ot else None

    # s1__coh_vv__20220103_20220115
    m = re.match(r'^s1__coh_(vv|vh)__(\d{8})_(\d{8})$', stripped)
    if m:
        pol, d1, d2 = m.group(1), m.group(2), m.group(3)
        ot = _orbit_tag(d1) or _orbit_tag(d2)
        return f"s1__coh_{pol}{ot}__{d1}_{d2}{ext}" if ot else None

    return None


# ---- Landuse flatten: normalize stem before appending __YYYYMMDD ----

LANDUSE_RENAME = {
    'landuse_classification': 'lulc__class',
    'ndvi': 's2__ndvi',
    'nbr': 's2__nbr',
    'bsi': 's2__bsi',
    'savi': 's2__savi',
    'mndwi': 's2__mndwi',
    'ndbi': 's2__ndbi',
}
# indices that keep their name without s2__ prefix
_LANDUSE_PASSTHROUGH = {'baei', 'ibi', 'ndsi', 'ui'}


def normalize_landuse_stem(stem):
    """Normalize a landuse TIF stem (no extension, no date) to __ convention.

    Used by ALIGN-LANDUSE cell in NB05a when flattening
    landuse/{period}/{date}/ structure to landuse/flat/.

    Args:
        stem: str, e.g. 'landuse_classification', 'ndvi', 'bsi', 'lulc__class'

    Returns:
        str, normalized stem, e.g. 'lulc__class', 's2__ndvi', 'baei'
    """
    low = stem.lower()
    if low in LANDUSE_RENAME:
        return LANDUSE_RENAME[low]
    if low in _LANDUSE_PASSTHROUGH:
        return low
    # already normalized (e.g. lulc__class, s2__ndvi, fire__active)
    if '__' in stem:
        return stem
    return stem


def _test():
    """Quick self-test for rename patterns."""
    cases = [
        ("Mariupol", "Mariupol_card_assessment_mean_VV.tif",   "s1__vv__assessment__mean.tif"),
        ("Mariupol", "Mariupol_card_baseline_kurtosis_VH.tif", "s1__vh__baseline__kurtosis.tif"),
        ("Mariupol", "Mariupol_coh_baseline_mean.tif",         "s1__coh__baseline__mean.tif"),
        ("Mariupol", "Mariupol_CARD_VH_20220103.tif",          "s1__vh__20220103.tif"),
        ("Mariupol", "Mariupol_S2_20220108_B02_10m.tif",       "s2__b02__20220108.tif"),
        ("Mariupol", "Mariupol_S2_20220108_B8A_20m.tif",       "s2__b8a__20220108.tif"),
        ("Mariupol", "Mariupol_S2_20220108_cloud_mask.tif",    "s2__cloud_mask__20220108.tif"),
        ("Mariupol", "Mariupol_S2_20220108_SCL_20m.tif",       "s2__scl__20220108.tif"),
        ("Mariupol", "Mariupol_S2_20220108_visibility.tif",    "s2__visibility__20220108.tif"),
        ("Mariupol", "Mariupol_CARD_roll2_VH_20220103.tif",    "s1__vh__roll2__20220103.tif"),
        ("Mariupol", "Mariupol_coh_roll3_VV_20220316.tif",     "s1__coh_vv__roll3__20220316.tif"),
        ("Mariupol", "Mariupol_coh_zscore_VV_20220103.tif",    "s1__coh_vv__zscore__20220103.tif"),
        ("Mariupol", "composite_B02.tif",                      "s2__b02.tif"),
        ("Mariupol", "dNBR_prebattle_vs_post.tif",             "cd__dnbr__prebattle_vs_post.tif"),
        ("Mariupol", "landuse_classification.tif",              "lulc__class.tif"),
        ("Mariupol", "NDVI.tif",                               "s2__ndvi.tif"),
        ("Mariupol", "active_fire.tif",                        "fire__active.tif"),
        ("Mariupol", "cloud_frequency.tif",                    "qa__cloud_freq.tif"),
        ("Sievierodonetsk", "Sievierodonetsk_card_assessment_mean_VV.tif", "s1__vv__assessment__mean.tif"),
        ("Sievierodonetsk", "Sievierodonetsk_CARD_VH_20220103.tif",       "s1__vh__20220103.tif"),
        # COH with period tag + undashed dates (old NB03a)
        ("Rubizhne", "Rubizhne_COH_VH_POST_20220515_20220527.tif",      "s1__coh_vh__20220515_20220527.tif"),
        ("Rubizhne", "Rubizhne_COH_VV_CROSS_20220503_20220515.tif",     "s1__coh_vv__20220503_20220515.tif"),
        # COH with period tag + dashed dates (NB03a v16)
        ("Rubizhne", "Rubizhne_COH_VH_PRE_2022-02-23_2022-03-07.tif",   "s1__coh_vh__20220223_20220307.tif"),
        ("Rubizhne", "Rubizhne_COH_VV_CROSS_2022-04-12_2022-05-06.tif", "s1__coh_vv__20220412_20220506.tif"),
        ("Rubizhne", "Rubizhne_COH_VH_POST_2023-06-11_2023-06-23.tif",  "s1__coh_vh__20230611_20230623.tif"),
        # orbit in filenames (NB03b/c v27+/v24+) - orbit stripped for ML portability
        ("Mariupol", "s1__vv__o043__20220103.tif",                       "s1__vv__20220103.tif"),
        ("Mariupol", "s1__vh__o043__20220315.tif",                       "s1__vh__20220315.tif"),
        ("Mariupol", "s1__coh_vv__o043__20220103_20220115.tif",          "s1__coh_vv__20220103_20220115.tif"),
        ("Mariupol", "s1__coh_vh__o043__20220223_20220307.tif",          "s1__coh_vh__20220223_20220307.tif"),
    ]
    passed = 0
    failed = 0
    for city, inp, expected in cases:
        result = rename_tif_for_convention(city, inp)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: rename_tif_for_convention('{city}', '{inp}')")
            print(f"    expected: {expected}")
            print(f"    got:      {result}")
            failed += 1
    print(f"\nRename self-test: {passed} passed, {failed} failed")

    # test normalize_landuse_stem
    lu_cases = [
        ('landuse_classification', 'lulc__class'),
        ('ndvi', 's2__ndvi'),
        ('nbr', 's2__nbr'),
        ('bsi', 's2__bsi'),
        ('savi', 's2__savi'),
        ('mndwi', 's2__mndwi'),
        ('ndbi', 's2__ndbi'),
        ('baei', 'baei'),
        ('ibi', 'ibi'),
        ('ndsi', 'ndsi'),
        ('ui', 'ui'),
        ('lulc__class', 'lulc__class'),
        ('s2__ndvi', 's2__ndvi'),
        ('fire__active', 'fire__active'),
    ]
    lu_passed = 0
    lu_failed = 0
    for inp, expected in lu_cases:
        result = normalize_landuse_stem(inp)
        if result == expected:
            lu_passed += 1
        else:
            print(f"  FAIL: normalize_landuse_stem('{inp}')")
            print(f"    expected: {expected}")
            print(f"    got:      {result}")
            lu_failed += 1
    print(f"Landuse stem self-test: {lu_passed} passed, {lu_failed} failed")

    return failed == 0 and lu_failed == 0


if __name__ == '__main__':
    _test()
