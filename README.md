# BDA - Building Damage Assessment in Ukrainian Conflict Cities

Code, results, and verification materials for the MSc thesis **"Building Damage Assessment with Multimodal Satellite Time Series and Machine Learning"** (UNIGIS, Paris Lodron University Salzburg, 2026) by Marco Heinzen.

The thesis evaluates how far open, medium-resolution satellite data can support building damage assessment (BDA) in conflict zones: Sentinel-1 SAR (backscatter and InSAR coherence) and Sentinel-2 multispectral time series over **20 Ukrainian cities**, **588,686 Overture building footprints**, and **7,326 UNOSAT-confirmed damage labels**, with classical ML and deep-learning classifiers evaluated under city-grouped spatial cross-validation (GroupKFold by city) and a city-level inferential protocol (bootstrap BCa confidence intervals, Friedman and Wilcoxon-Holm tests). The central methodological contribution is a decomposition of the cross-city generalization gap into sampling-unit, feature-scope, and distribution-shift components.

## Data (Zenodo)

The analysis-ready datasets (feature-engineered Parquet, DL patches, cross-validation fold assignments) are archived on Zenodo:

**DOI: [10.5281/zenodo.21327887](https://doi.org/10.5281/zenodo.21327887)**

With this deposit alone, the full modelling and evaluation chain (notebooks NB06-NB13) is reproducible without satellite-data downloads or API credentials. Raw satellite products (~800 GB) are not deposited and can be regenerated from open archives with notebooks NB01-NB03.

## Repository structure

| Folder | Content |
|---|---|
| `notebooks/` | 86 Jupyter notebooks (NB01-NB13 pipeline plus the NB11x tuning series), committed version by version with original save dates. Modules: `global_setup.py`, `cities_config.json`, helpers. |
| `results/` | Verification artifacts: experiment registry, 47 out-of-fold prediction parquets, Optuna tuning outputs, statistics (NB12), validation series (NB11a-f), interactive maps, per-notebook CSV records. See `results/README.md` and `results/MANIFEST.csv`. |
| `thesis/` | Claim-to-file verification package: the exact CSVs and figures referenced by the thesis manuscript, with a trace table in `thesis/README.md`. |
| `Theory/` | Background and methodology documents. |
| `old/` | Superseded early-stage material, kept for provenance. |

## Pipeline overview

`NB01` external sources (Overture, UNOSAT, xBD) - `NB02` satellite downloaders (Copernicus, ASF) - `NB03` products (MS clipping, SAR CARD, SLC/SNAP coherence, composites, landuse) - `NB04` product QA - `NB05` data stack and dataset builders (V1-V7 parquet variants) - `NB06` geospatial statistics - `NB07` Dietrich-pipeline replication - `NB08` temporal analysis and xBD diagnostics - `NB09` paper experiments (RF baseline, ensembles, stacking, U-Net, landuse plausibility) - `NB10` faithful method replications - `NB11x` hyperparameter tuning (Optuna, per experiment family) - `NB12` inferential statistics and thesis figures - `NB13` validation and results reporting.

## Replication

Summary of thesis Section III.7.1:

1. Clone this repository.
2. Download the data stack from Zenodo (DOI above) and unpack the ZIPs into a local stack directory (`dataset/` ZIPs into `<stack>/dataset/`, `stack_meta.zip` into `<stack>/`).
3. Edit the three roots (`DRIVE_F_ROOT`, `STACK_DIR`, `CONTENT_LOCAL`) in the matching environment block of `notebooks/global_setup.py`.
4. Create a Python 3.12 conda environment with JupyterLab and run NB06-NB13.

Credentials (`REDACTED_*` environment variables: Copernicus, NASA Earthdata, OpenTopography) are only needed to re-download raw satellite data with NB01-NB03; the stack-only replication requires none.

## Data attribution

Contains modified Copernicus Sentinel data (2020-2026). Damage reference labels derived from UNOSAT (United Nations Satellite Centre, UNITAR) assessments. Building footprints from the Overture Maps Foundation. Elevation data via OpenTopography. Comparator method and compiled UNOSAT points after Dietrich et al. 2025 ([doi:10.1038/s43247-025-02183-7](https://doi.org/10.1038/s43247-025-02183-7)).

## Related identifiers

- Dataset: https://doi.org/10.5281/zenodo.21327887 (this repository is supplemented by the Zenodo deposit)
- Comparator tool: https://github.com/prs-eth/ukraine-damage-mapping-tool/

Contact: https://www.linkedin.com/in/marcoheinzen/

## License

This project is licensed under the **GNU Affero General Public License v3.0
or later** - see the [LICENSE](LICENSE) file for the full text.

**SPDX-License-Identifier:** `AGPL-3.0-or-later`

### What this means

- You are free to use, modify, and redistribute this code.
- If you distribute modified versions or deploy them as a network service,
  you must release your modifications under the same AGPL-3.0-or-later terms.
- See [NOTICE](NOTICE) for attribution and acknowledgments.

## License and AI disclosure

Copyright (C) 2024-2026 Marco Heinzen - AGPL-3.0-or-later (see LICENSE and NOTICE).
Parts of this code were written or improved with the assistance of Claude (Anthropic); all other code, and the concept, research, architecture, design, execution, testing and validation throughout, are the author's work.
