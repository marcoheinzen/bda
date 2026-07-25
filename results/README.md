# Results Package

Verification artifacts for the thesis "Building Damage Assessment with Multimodal Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026" (Marco Heinzen, UNIGIS MSc, Paris Lodron University Salzburg). This folder lets a reviewer verify every reported number without running the notebooks. `MANIFEST.csv` lists every file with its original source path.

Where result folders contain timestamped run artifacts, only the newest run per artifact is included; the full run history remains in the local archive. Large binary intermediates (model pickles, OOF pickles, per-run parquet dumps of legacy notebooks) are excluded; the analysis-ready dataset itself is published separately on Zenodo (DOI in thesis Section III.7.1).

## Core verification chain (current pipeline)

| Folder | Written by | Content |
|---|---|---|
| registry/ | 11x_G_write_results_registry, 13d | Experiment registry: every experiment row (AUC, per-city metrics, profiles) behind the thesis tables. `NB11_V2_experiments_20260703_083441.csv` is the frozen headline source. |
| oof/ | 11x_01-14 tuning notebooks | 47 out-of-fold prediction parquets (y_true, y_proba, city, fold_id) - recompute any metric directly. |
| nb11_tuning/ | 11x_01-14 | Optuna best_params, studies, and summaries per experiment (model pickles excluded). |
| nb12/ | 12a, 12b, 12c, 12d | Mean-of-folds vs pooled AUC, per-city AUC matrix, bootstrap BCa CIs, Friedman + Wilcoxon-Holm, dispatch ensemble, and the thesis figures (figures/). |
| nb11a/ | 13a_OOF_Overlap_Analysis | OOF overlap and per-building/per-sample error-rate analyses. |
| nb11b/ | 13b_OOF_Spatial_Viz | Spatial visualization CSV/JSON summaries (the ~1,200 per-city map PNGs are excluded for size; see interactive_maps/). |
| nb11c/ | 13c_OOF_CV_Rigor | CV-rigor checks incl. cell_m7 leakage-delta bootstrap and cell_m8 label-noise summaries. |
| nb11d/ | 13d_Results_Report | Consolidated experiment and per-city metric reports. |
| nb11e/, nb11f/ | 13e, 13f | Tuning comparison and hyperparameter analysis. |
| interactive_maps/ | interactive map cells | Two standalone HTML maps (city- and oblast-level). Download and open locally; GitHub does not render files this size in the browser. |

## Exploratory / legacy series (CSV+JSON record only)

nb04 (product QA), nb06/nb06v3/nb06v4 (geospatial statistics V2-V4), nb07/nb07v3 (Dietrich pipeline), nb08/nb08b (temporal analysis, xBD RGB diagnostic), nb09a/b/c/e and their _v2/_v3 variants (paper experiments), nb10b_v3 (faithful replication). Parquet/pickle intermediates of these runs are excluded.

## Reproducing from scratch

See thesis Section III.7.1 and the repository root: clone the repo, download the data stack from Zenodo, set the three roots in `notebooks/global_setup.py`, and run NB06-NB13 in JupyterLab.
