# Thesis Verification Package

Exact files referenced by the thesis manuscript (Chapters I-V; manuscript = Chapter II, technical report = Chapter III), copied verbatim so each claim can be traced without navigating the full results tree. Duplicates with `/results` are intentional: this folder mirrors the `[src: ...]` trace tags in the manuscript.

## results/ - claim-to-file map

| Thesis claim / section | File(s) |
|---|---|
| Headline mean-of-folds vs pooled AUCs, D4 Strategy-A/B pair 0.428 -> 0.663 (II.5, III.4) | nb12/nb12a_meanfolds_vs_pooled.csv, nb12/nb12a_per_city_auc_matrix.csv |
| Bootstrap BCa confidence intervals (II.5.4, III.4) | nb12/nb12b_meanfolds_bca_ci.csv |
| Friedman omnibus + pairwise Wilcoxon-Holm (II.5.4) | nb12/nb12b_friedman_omnibus.csv, nb12/nb12b_pairwise_wilcoxon_holm__*.csv |
| Modality-dispatch ensemble vs members (II.6.2) | nb12/nb12c_dispatch_per_city.csv, nb12/nb12c_ensemble_vs_members.csv |
| Geographic-leakage delta = -0.0000, CI [-0.014, +0.013] (II.6.3.4) | nb11c_cell_m7/M7_leakage_delta_bootstrap_*.csv |
| Label-noise / PU-learning quantification (II.6.3.5, II.7.1) | nb11c_cell_m8/M8_label_noise_summary_*.csv |
| Frozen consolidated experiment tables (III.4) | nb11d/consolidated_experiments_20260703_084619.csv, nb11d/per_city_metrics_20260703_084619.csv |
| Frozen registry snapshots incl. xbd_on_xbd row AUC=0.9643 (xBD ceiling, II.5.5) | registry/NB11_V2_experiments_20260703_083441.csv, registry/NB09c_v2_experiments_20260614_180054.csv, registry/NB09d_experiments_20260621_171647.csv |
| MS pre/post scene count vs AUC (III.3) | nb07_cell_d1y/ms_prepost_vs_auc.csv |
| NB11/NB12/NB13 reconciliation checks (III.4) | reconciliation/reconciliation_summary.csv, .json |
| Appendix tables A2 (UNOSAT snapshot dates), A4 (metric ruler), A6 (effect sizes) | appendix_tables/A2_*.csv, A4_*.csv, A6_*.csv |

## figures/ - embedded manuscript figures

The ten `fig_*.png` files embedded in the manuscript (study area, mean-of-folds BCa CIs, screening enrichment, V2-vs-V3 modality, mean-of-folds vs pooled, label noise, per-city AUC spread, screening heatmap example, size confound) plus `make_review_figures.py`, the script that generates the review figures from the CSVs above.
