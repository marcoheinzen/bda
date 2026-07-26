# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
nb11_finalize.py

Shared OPTUNA-FINAL logic for all NB11 hyperparameter-tuning notebooks.
Refits the best Optuna params, recomputes pooled OOF + per-fold AUCs, computes
mean_folds_auc / std_folds_auc, saves OOF parquet + joblib model + best_params JSON,
and logs to the ResultRegistry.

OOF drift between Optuna search and refit is a WARNING (not a hard assert) because
LightGBM/XGBoost are not bit-deterministic under multi-threading. Threshold 5e-3.

Three entry points, one per notebook group:
    finalize_group_a(globals())       # NB11a/b/c/d  -- LightGBM-MIA, per parquet
    finalize_group_b_r5(globals())    # NB11e        -- multi-classifier, single feature set
    finalize_group_b_r3(globals())    # NB11f        -- RF-200, per (parquet, label)

Each reads runtime objects (result dicts, prep dicts, config constants, registry,
save_oof_from_result, classifier classes) from the notebook namespace passed in.
"""
import json as _json
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DRIFT_WARN_THRESHOLD = 5e-3


def _get(ns, name):
    if name not in ns:
        raise KeyError(f"nb11_finalize: '{name}' not found in notebook namespace; "
                       f"run the CONFIG, BASELINE and OPTUNA cells first.")
    return ns[name]


def _refit_oof(clf, X, y, groups, n_folds):
    """GroupKFold OOF refit. Returns (y_proba_oof, fold_id, valid, pooled_auc, fold_aucs)."""
    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    y_proba_oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    fold_aucs = []
    for fi, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf_c = clone(clf)
        clf_c.fit(X[tr], y[tr])
        proba = clf_c.predict_proba(X[te])[:, 1]
        y_proba_oof[te] = proba
        fold_id[te] = fi
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], proba))
    valid = ~np.isnan(y_proba_oof)
    pooled_auc = roc_auc_score(y[valid], y_proba_oof[valid])
    return y_proba_oof, fold_id, valid, pooled_auc, fold_aucs


def _check_drift(label, pooled_auc, search_auc):
    drift = pooled_auc - search_auc
    print(f"    Tuned OOF AUC (refit): {pooled_auc:.4f}  (study.best={search_auc:.4f})  "
          f"delta={drift:+.4f}")
    if abs(drift) >= DRIFT_WARN_THRESHOLD:
        print(f"    WARNING: large OOF drift {drift:+.4f} between search and refit "
              f"(tree-ensemble multi-thread nondeterminism); proceeding with refit value")
    return drift


def _persist(ns, base_clf, best_params, X, y, groups, building_ids, clean_cols,
             pooled_auc, fold_id, fold_aucs, valid, mean_folds, std_folds,
             experiment, model_id, study_tag, out_dir,
             reg_kwargs):
    """Save OOF parquet, joblib model, best_params JSON, registry log."""
    save_oof_from_result = _get(ns, 'save_oof_from_result')
    registry = _get(ns, 'registry')
    nb_name = _get(ns, 'NB11_NAME')
    n_folds = _get(ns, 'N_FOLDS')
    patience = _get(ns, 'OPTUNA_PATIENCE')

    tuned_res = {
        'y_true':       y[valid].astype(int),
        'y_proba':      None,  # set by caller via reg_kwargs; we reuse below
        'groups':       groups[valid],
        'building_id':  np.asarray(building_ids)[valid],
        'fold_id':      fold_id[valid],
        'experiment':   experiment,
        'auc_groupkfold': pooled_auc,
        'fold_aucs':    fold_aucs,
    }
    # y_proba is passed in reg_kwargs['y_proba_oof'] (full array); slice with valid
    y_proba_oof = reg_kwargs.pop('y_proba_oof')
    tuned_res['y_proba'] = y_proba_oof[valid]

    save_oof_from_result(tuned_res, model_id=model_id, oof_dir=out_dir / "oof",
                         variant_id=study_tag, is_final=True)

    best_clf_full = clone(base_clf).fit(X, y)
    model_path = out_dir / "models" / f"{study_tag}__best.joblib"
    joblib.dump({'model': best_clf_full, 'feature_cols': clean_cols,
                 'params': best_params, 'tuned_auc': pooled_auc,
                 'mean_folds_auc': mean_folds, 'std_folds_auc': std_folds,
                 'baseline_auc': reg_kwargs['baseline_auc'],
                 'study_tag': study_tag}, model_path)
    print(f"    Saved model: {model_path}")

    bp_path = out_dir / "best_params" / f"{study_tag}__best.json"
    bp_payload = {
        'study_tag': study_tag, 'notebook': nb_name,
        'best_params': best_params,
        'tuned_auc': pooled_auc, 'baseline_auc': reg_kwargs['baseline_auc'],
        'delta_auc': pooled_auc - reg_kwargs['baseline_auc'],
        'mean_folds_auc': mean_folds, 'std_folds_auc': std_folds,
        'fold_aucs': fold_aucs,
        'n_trials_done': reg_kwargs['n_trials_done'],
        'n_trials_requested': reg_kwargs['n_trials_requested'],
        'patience': patience,
        'n_features': len(clean_cols), 'n_buildings': int(valid.sum()),
        'n_cities': int(len(np.unique(groups))),
    }
    bp_payload.update(reg_kwargs.get('bp_extra', {}))
    with open(bp_path, 'w') as f:
        _json.dump(bp_payload, f, indent=2, default=str)
    print(f"    Saved best_params: {bp_path}")

    registry.log_experiment(
        cell_id=_get(ns, 'NB11_CELL_ID'),
        experiment_name=experiment,
        parquet_name=reg_kwargs['parquet_name'],
        feature_set_name=reg_kwargs['feature_set_name'],
        classifier_name=reg_kwargs['classifier_name'],
        classifier_params=best_params,
        feature_cols=clean_cols,
        cv_method='GroupKFold', n_folds=n_folds,
        imputation=reg_kwargs['imputation'],
        y_true=tuned_res['y_true'], y_proba=tuned_res['y_proba'],
        groups=tuned_res['groups'],
        note=(f"NB11 Optuna tuned {reg_kwargs['classifier_name']} on "
              f"{reg_kwargs['note_subject']}; "
              f"budget={reg_kwargs['n_trials_requested']}, "
              f"trials_done={reg_kwargs['n_trials_done']}, patience={patience}; "
              f"pooled_auc={pooled_auc:.4f} mean_folds_auc={mean_folds:.4f} "
              f"std_folds_auc={std_folds:.4f} fold_aucs={fold_aucs}"),
        tags=reg_kwargs['tags'],
    )


# =====================================================================
# Group A -- LightGBM-MIA, per parquet (NB11a/b/c/d)
# =====================================================================
def finalize_group_a(ns):
    results = _get(ns, 'NB11_OPTUNA_RESULTS')
    prep_all = _get(ns, 'NB11_PREP')
    fixed_lgbm = _get(ns, 'FIXED_LGBM')
    LGBMClassifier = _get(ns, 'LGBMClassifier')
    out_dir = _get(ns, 'NB11_OUT_DIR')
    n_folds = _get(ns, 'N_FOLDS')
    nb_name = _get(ns, 'NB11_NAME')

    for pq_id, info in results.items():
        manifest_key = info['manifest_key']
        prep = prep_all[pq_id]
        X, y, groups = prep['X'], prep['y'], prep['groups']
        building_ids, clean_cols = prep['building_ids'], prep['clean_cols']

        print(f"\n  === refit [{pq_id}] {manifest_key} ===")
        best_params_full = {**fixed_lgbm, **info['best_params']}
        base_clf = LGBMClassifier(**best_params_full)

        y_proba_oof, fold_id, valid, pooled_auc, fold_aucs = _refit_oof(
            base_clf, X, y, groups, n_folds)
        mean_folds = float(np.mean(fold_aucs))
        std_folds = float(np.std(fold_aucs))
        _check_drift(pq_id, pooled_auc, info['tuned_auc'])
        print(f"    mean(folds)={mean_folds:.4f}  std(folds)={std_folds:.4f}")

        info['fold_aucs'] = fold_aucs
        info['tuned_auc_refit'] = pooled_auc
        info['mean_folds_auc'] = mean_folds
        info['std_folds_auc'] = std_folds

        _persist(
            ns, base_clf, best_params_full, X, y, groups, building_ids, clean_cols,
            pooled_auc, fold_id, fold_aucs, valid, mean_folds, std_folds,
            experiment=f'C_{pq_id}_{manifest_key}__OPTUNA',
            model_id=f"{nb_name}__{manifest_key}__LightGBM-MIA-Optuna",
            study_tag=info['study_tag'], out_dir=out_dir,
            reg_kwargs={
                'y_proba_oof': y_proba_oof,
                'baseline_auc': info['baseline_auc'],
                'n_trials_done': info['n_trials_done'],
                'n_trials_requested': info['n_trials_requested'],
                'parquet_name': f'bda_{manifest_key}_v2',
                'feature_set_name': 'manifest_default_minus_leakage',
                'classifier_name': 'LightGBM-MIA-Optuna',
                'imputation': 'native_nan',
                'note_subject': manifest_key,
                'tags': ['nb11_optuna', 'lightgbm_mia', pq_id, manifest_key],
                'bp_extra': {'manifest_key': manifest_key, 'parquet_id': pq_id,
                             'classifier': 'LightGBM-MIA',
                             'best_params_full': best_params_full},
            })


# =====================================================================
# Group B R5 -- multi-classifier, single feature set (NB11e)
# =====================================================================
def finalize_group_b_r5(ns):
    results = _get(ns, 'NB11_OPTUNA_RESULTS')
    X = _get(ns, 'X_NB11'); y = _get(ns, 'y_NB11'); groups = _get(ns, 'groups_NB11')
    building_ids = _get(ns, 'building_ids_NB11'); clean_cols = _get(ns, 'clean_cols_NB11')
    out_dir = _get(ns, 'NB11_OUT_DIR')
    n_folds = _get(ns, 'N_FOLDS')
    nb_name = _get(ns, 'NB11_NAME')
    parquet_id = _get(ns, 'NB11_PARQUET_ID')
    parquet_name = _get(ns, 'NB11_PARQUET_NAME')
    manifest_key = _get(ns, 'NB11_MANIFEST_KEY')
    feature_set = _get(ns, 'NB11_FEATURE_SET')

    for clf_name, info in results.items():
        print(f"\n  === refit {clf_name} ===")
        base_clf = info['make_clf'](info['best_params'])

        y_proba_oof, fold_id, valid, pooled_auc, fold_aucs = _refit_oof(
            base_clf, X, y, groups, n_folds)
        mean_folds = float(np.mean(fold_aucs))
        std_folds = float(np.std(fold_aucs))
        _check_drift(clf_name, pooled_auc, info['tuned_auc'])
        print(f"    mean(folds)={mean_folds:.4f}  std(folds)={std_folds:.4f}")

        info['fold_aucs'] = fold_aucs
        info['tuned_auc_refit'] = pooled_auc
        info['mean_folds_auc'] = mean_folds
        info['std_folds_auc'] = std_folds

        _persist(
            ns, base_clf, info['best_params'], X, y, groups, building_ids, clean_cols,
            pooled_auc, fold_id, fold_aucs, valid, mean_folds, std_folds,
            experiment=f"R5_{clf_name}_{feature_set}__OPTUNA",
            model_id=f"{nb_name}__{manifest_key}__{clf_name}-Optuna",
            study_tag=info['study_tag'], out_dir=out_dir,
            reg_kwargs={
                'y_proba_oof': y_proba_oof,
                'baseline_auc': info['baseline_auc'],
                'n_trials_done': info['n_trials_done'],
                'n_trials_requested': info['n_trials_requested'],
                'parquet_name': parquet_name,
                'feature_set_name': feature_set,
                'classifier_name': f'{clf_name}-Optuna',
                'imputation': 'median',
                'note_subject': f'{manifest_key}/{feature_set}',
                'tags': ['nb11_optuna', clf_name, parquet_id, manifest_key, feature_set],
                'bp_extra': {'manifest_key': manifest_key, 'parquet_id': parquet_id,
                             'feature_set': feature_set, 'classifier': clf_name},
            })


# =====================================================================
# Group B R3 -- RF-200, per (parquet, label) (NB11f)
# =====================================================================
def finalize_group_b_r3(ns):
    results = _get(ns, 'NB11_OPTUNA_RESULTS')
    prep_all = _get(ns, 'NB11_PREP')
    out_dir = _get(ns, 'NB11_OUT_DIR')
    n_folds = _get(ns, 'N_FOLDS')
    nb_name = _get(ns, 'NB11_NAME')

    for pq_id, info in results.items():
        manifest_key = info['manifest_key']
        label = info['label']
        prep = prep_all[pq_id]
        X, y, groups = prep['X'], prep['y'], prep['groups']
        building_ids, clean_cols = prep['building_ids'], prep['clean_cols']

        print(f"\n  === refit [{pq_id}] {label} ({manifest_key}) ===")
        base_clf = info['make_clf'](info['best_params'])

        y_proba_oof, fold_id, valid, pooled_auc, fold_aucs = _refit_oof(
            base_clf, X, y, groups, n_folds)
        mean_folds = float(np.mean(fold_aucs))
        std_folds = float(np.std(fold_aucs))
        _check_drift(label, pooled_auc, info['tuned_auc'])
        print(f"    mean(folds)={mean_folds:.4f}  std(folds)={std_folds:.4f}")

        info['fold_aucs'] = fold_aucs
        info['tuned_auc_refit'] = pooled_auc
        info['mean_folds_auc'] = mean_folds
        info['std_folds_auc'] = std_folds

        _persist(
            ns, base_clf, info['best_params'], X, y, groups, building_ids, clean_cols,
            pooled_auc, fold_id, fold_aucs, valid, mean_folds, std_folds,
            experiment=f"R3_{label}__OPTUNA",
            model_id=f"{nb_name}__{manifest_key}__{label}__RF-200-Optuna",
            study_tag=info['study_tag'], out_dir=out_dir,
            reg_kwargs={
                'y_proba_oof': y_proba_oof,
                'baseline_auc': info['baseline_auc'],
                'n_trials_done': info['n_trials_done'],
                'n_trials_requested': info['n_trials_requested'],
                'parquet_name': f'bda_{manifest_key}_v2',
                'feature_set_name': label,
                'classifier_name': 'RF-200-Optuna',
                'imputation': 'median',
                'note_subject': f'{manifest_key}/{label}',
                'tags': ['nb11_optuna', 'rf_200', 'sensor_ablation',
                         pq_id, manifest_key, label],
                'bp_extra': {'manifest_key': manifest_key, 'parquet_id': pq_id,
                             'label': label, 'classifier': 'RF-200'},
            })
