# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

"""
bda_results.py  --  BDA Result Registry
========================================
Lives in NOTEBOOKS_DIR alongside global_setup.py.
Every NB06-NB11 notebook imports this and calls registry.log_experiment()
after each experiment cell. NB13 calls registry.load_all() to consolidate.

Usage in any notebook:
    from bda_results import ResultRegistry
    registry = ResultRegistry(RESULTS_ROOT)

    # after an experiment:
    registry.log_experiment(
        notebook='NB09a', cell_id='cell_r0',
        experiment_name='R0_dietrich28_groupkfold',
        parquet_name='bda_product_prepost',
        parquet_fmt=str(PARQUET_PREPOST_TIER_FMT),
        tier_selection=[0,1,2],
        cities=list(np.unique(groups)),
        classifier_name='RF-200',
        classifier_params={'n_estimators': 200, 'min_samples_leaf': 3},
        feature_set_name='dietrich28',
        feature_cols=clean,
        cv_method='GroupKFold', n_folds=4,
        imputation='median',
        y_true=y_v, y_proba=p_v, groups=groups_v,
        note='Dietrich 2025 replication with 28 CARD features',
    )

    # at end of notebook:
    registry.save()
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import OrderedDict


# ---------------------------------------------------------------------------
# metric computation from y_true / y_proba
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_proba, threshold=0.5):
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        accuracy_score, balanced_accuracy_score, matthews_corrcoef,
        average_precision_score, log_loss, brier_score_loss,
        confusion_matrix,
    )
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_proba).astype(float)

    n_pos = int(y_t.sum())
    n_neg = int(len(y_t) - n_pos)

    m = OrderedDict()
    m['n_samples'] = len(y_t)
    m['n_positive'] = n_pos
    m['n_negative'] = n_neg
    m['prevalence'] = n_pos / len(y_t) if len(y_t) > 0 else 0.0
    m['threshold'] = threshold

    try:
        m['auc'] = float(roc_auc_score(y_t, y_p))
    except Exception:
        m['auc'] = None
    try:
        m['average_precision'] = float(average_precision_score(y_t, y_p))
    except Exception:
        m['average_precision'] = None
    try:
        m['log_loss'] = float(log_loss(y_t, y_p))
    except Exception:
        m['log_loss'] = None
    try:
        m['brier_score'] = float(brier_score_loss(y_t, y_p))
    except Exception:
        m['brier_score'] = None

    m['accuracy'] = float(accuracy_score(y_t, y_pred))
    m['balanced_accuracy'] = float(balanced_accuracy_score(y_t, y_pred))
    m['f1'] = float(f1_score(y_t, y_pred, zero_division=0))
    m['precision'] = float(precision_score(y_t, y_pred, zero_division=0))
    m['recall'] = float(recall_score(y_t, y_pred, zero_division=0))
    m['mcc'] = float(matthews_corrcoef(y_t, y_pred))

    try:
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        m['tp'] = int(tp)
        m['fp'] = int(fp)
        m['tn'] = int(tn)
        m['fn'] = int(fn)
        m['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        m['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    except Exception:
        m['tp'] = m['fp'] = m['tn'] = m['fn'] = None
        m['specificity'] = m['fpr'] = None

    # precision at high recall thresholds
    for recall_target in [0.80, 0.90, 0.95]:
        try:
            from sklearn.metrics import precision_recall_curve
            prec_arr, rec_arr, thr_arr = precision_recall_curve(y_t, y_p)
            mask = rec_arr >= recall_target
            if mask.any():
                m[f'precision_at_recall_{int(recall_target*100)}'] = float(prec_arr[mask][-1])
                m[f'threshold_at_recall_{int(recall_target*100)}'] = float(thr_arr[mask.nonzero()[0][-1]]) if len(thr_arr) > mask.nonzero()[0][-1] else None
            else:
                m[f'precision_at_recall_{int(recall_target*100)}'] = None
                m[f'threshold_at_recall_{int(recall_target*100)}'] = None
        except Exception:
            m[f'precision_at_recall_{int(recall_target*100)}'] = None
            m[f'threshold_at_recall_{int(recall_target*100)}'] = None

    return m


def compute_per_city_metrics(y_true, y_proba, groups, threshold=0.5):
    results = {}
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_proba)
    g = np.asarray(groups)
    for city in sorted(np.unique(g)):
        mask = g == city
        if mask.sum() < 10:
            continue
        y_t_c = y_t[mask]
        y_p_c = y_p[mask]
        if len(np.unique(y_t_c)) < 2:
            results[city] = {'n': int(mask.sum()), 'auc': None, 'note': 'single class'}
            continue
        m = compute_metrics(y_t_c, y_p_c, threshold)
        m['city'] = city
        results[city] = m
    return results


def compute_fold_metrics(fold_results):
    if not fold_results:
        return {}
    aucs = [f['auc'] for f in fold_results if f.get('auc') is not None]
    f1s = [f['f1'] for f in fold_results if f.get('f1') is not None]
    m = {}
    if aucs:
        m['auc_mean'] = float(np.mean(aucs))
        m['auc_std'] = float(np.std(aucs))
        m['auc_min'] = float(np.min(aucs))
        m['auc_max'] = float(np.max(aucs))
        m['fold_aucs'] = [float(a) for a in aucs]
    if f1s:
        m['f1_mean'] = float(np.mean(f1s))
        m['f1_std'] = float(np.std(f1s))
        m['fold_f1s'] = [float(f) for f in f1s]
    m['n_folds_computed'] = len(fold_results)
    return m


# ---------------------------------------------------------------------------
# feature metadata
# ---------------------------------------------------------------------------
def describe_features(df, feature_cols):
    from scipy import stats as sp_stats
    rows = []
    for col in feature_cols:
        vals = df[col].dropna().values
        if len(vals) < 5:
            rows.append({'feature': col, 'count': len(vals), 'nan_pct': df[col].isna().mean() * 100})
            continue
        rows.append({
            'feature': col,
            'count': len(vals),
            'nan_pct': float(df[col].isna().mean() * 100),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)),
            'median': float(np.median(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'skewness': float(sp_stats.skew(vals)),
            'kurtosis': float(sp_stats.kurtosis(vals)),
            'q25': float(np.percentile(vals, 25)),
            'q75': float(np.percentile(vals, 75)),
            'iqr': float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            'range': float(np.max(vals) - np.min(vals)),
            'cv': float(np.std(vals, ddof=1) / np.mean(vals)) if np.mean(vals) != 0 else None,
        })
    return rows


# ---------------------------------------------------------------------------
# ResultRegistry
# ---------------------------------------------------------------------------
class ResultRegistry:
    """Accumulates experiment results during a notebook run, saves to JSON."""

    SCHEMA_VERSION = 1

    def __init__(self, results_root, notebook=None):
        self.results_root = Path(results_root)
        self.registry_dir = self.results_root / 'registry'
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.notebook = notebook
        self.experiments = []
        self.dataset_profiles = []
        self._run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"  ResultRegistry: {self.registry_dir} (run_id={self._run_id})")

    def log_experiment(self,
                       # identity
                       notebook=None,
                       cell_id='',
                       experiment_name='',
                       # data source
                       parquet_name='',
                       parquet_fmt='',
                       parquet_path='',
                       tier_selection=None,
                       cities=None,
                       n_buildings=None,
                       n_damaged=None,
                       n_undamaged=None,
                       # features
                       feature_set_name='',
                       feature_cols=None,
                       feature_groups_used=None,
                       # method
                       classifier_name='',
                       classifier_params=None,
                       cv_method='GroupKFold',
                       n_folds=None,
                       imputation='median',
                       scaling='none',
                       class_weight='balanced',
                       # predictions (raw arrays for metric computation)
                       y_true=None,
                       y_proba=None,
                       groups=None,
                       threshold=0.5,
                       # precomputed metrics (if y_true/y_proba not available)
                       metrics=None,
                       fold_metrics=None,
                       per_city_metrics=None,
                       # feature stats
                       feature_stats=None,
                       # oof
                       oof_path=None,
                       # misc
                       note='',
                       is_reference=False,
                       reference_paper='',
                       tags=None,
                       extra=None,
                       ):
        nb = notebook or self.notebook or 'unknown'
        ts = datetime.now().isoformat()

        # compute metrics from predictions if provided
        if metrics is None and y_true is not None and y_proba is not None:
            y_t = np.asarray(y_true)
            y_p = np.asarray(y_proba)
            valid = ~(np.isnan(y_t) | np.isnan(y_p))
            y_t = y_t[valid]
            y_p = y_p[valid]
            metrics = compute_metrics(y_t, y_p, threshold)
            if groups is not None:
                g = np.asarray(groups)[valid]
                per_city_metrics = compute_per_city_metrics(y_t, y_p, g, threshold)

        # auto-detect counts
        if y_true is not None:
            y_arr = np.asarray(y_true)
            valid_y = y_arr[~np.isnan(y_arr)].astype(int)
            if n_buildings is None:
                n_buildings = len(valid_y)
            if n_damaged is None:
                n_damaged = int(valid_y.sum())
            if n_undamaged is None:
                n_undamaged = int(len(valid_y) - valid_y.sum())

        if cities is None and groups is not None:
            cities = sorted(list(set(np.asarray(groups).tolist())))

        record = OrderedDict()

        # identity
        record['schema_version'] = self.SCHEMA_VERSION
        record['run_id'] = self._run_id
        record['timestamp'] = ts
        record['notebook'] = nb
        record['cell_id'] = cell_id
        record['experiment_name'] = experiment_name

        # data source
        record['parquet_name'] = parquet_name
        record['parquet_fmt'] = parquet_fmt
        record['parquet_path'] = str(parquet_path) if parquet_path else ''
        record['tier_selection'] = tier_selection if tier_selection is not None else []
        record['cities'] = cities if cities is not None else []
        record['n_cities'] = len(cities) if cities else 0
        record['n_buildings'] = n_buildings
        record['n_damaged'] = n_damaged
        record['n_undamaged'] = n_undamaged
        record['damage_rate'] = float(n_damaged / n_buildings) if n_buildings and n_damaged is not None else None

        # features
        record['feature_set_name'] = feature_set_name
        record['feature_cols'] = feature_cols if feature_cols is not None else []
        record['n_features'] = len(feature_cols) if feature_cols else 0
        record['feature_groups_used'] = feature_groups_used if feature_groups_used is not None else []

        # method
        record['classifier_name'] = classifier_name
        record['classifier_params'] = _safe_dict(classifier_params) if classifier_params else {}
        record['cv_method'] = cv_method
        record['n_folds'] = n_folds
        record['imputation'] = imputation
        record['scaling'] = scaling
        record['class_weight'] = class_weight

        # metrics
        record['metrics'] = _safe_dict(metrics) if metrics else {}
        record['fold_metrics'] = _safe_dict(fold_metrics) if fold_metrics else {}
        record['per_city_metrics'] = {k: _safe_dict(v) for k, v in per_city_metrics.items()} if per_city_metrics else {}

        # feature stats (optional, can be large)
        record['feature_stats'] = feature_stats if feature_stats else []

        # oof
        record['oof_path'] = str(oof_path) if oof_path else ''

        # misc
        record['note'] = note
        record['is_reference'] = is_reference
        record['reference_paper'] = reference_paper
        record['tags'] = tags if tags else []
        record['extra'] = _safe_dict(extra) if extra else {}

        self.experiments.append(record)

        # print summary
        auc = record['metrics'].get('auc', '?')
        auc_s = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        f1 = record['metrics'].get('f1', '?')
        f1_s = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        print(f"  REG: {experiment_name:45s} AUC={auc_s} F1={f1_s} "
              f"n={n_buildings} feat={record['n_features']} "
              f"cities={record['n_cities']} [{nb}/{cell_id}]")

        return record

    def log_reference(self, experiment_name, auc, f1=None, precision=None,
                      recall=None, n_features=None, note='', paper='', **kwargs):
        metrics = {'auc': auc}
        if f1 is not None: metrics['f1'] = f1
        if precision is not None: metrics['precision'] = precision
        if recall is not None: metrics['recall'] = recall
        feature_cols = list(range(n_features)) if n_features else None
        return self.log_experiment(
            notebook='reference',
            experiment_name=experiment_name,
            metrics=metrics,
            feature_cols=feature_cols,
            note=note,
            is_reference=True,
            reference_paper=paper,
            **kwargs,
        )

    def log_dataset_profile(self, parquet_name, parquet_fmt, tier_selection,
                            df, feature_cols, meta_cols=None):
        from scipy import stats as sp_stats
        profile = OrderedDict()
        profile['parquet_name'] = parquet_name
        profile['parquet_fmt'] = parquet_fmt
        profile['tier_selection'] = tier_selection
        profile['n_rows'] = len(df)
        profile['n_cols'] = len(df.columns)
        profile['n_features'] = len(feature_cols)
        profile['n_meta'] = len(meta_cols) if meta_cols else 0
        profile['columns'] = list(df.columns)
        profile['feature_cols'] = list(feature_cols)
        profile['meta_cols'] = list(meta_cols) if meta_cols else []
        profile['dtypes'] = {c: str(df[c].dtype) for c in df.columns}

        if 'city' in df.columns:
            profile['cities'] = sorted(df['city'].unique().tolist())
            profile['n_cities'] = df['city'].nunique()
            profile['buildings_per_city'] = df.groupby('city').size().to_dict()
        if 'damage_binary' in df.columns:
            profile['n_damaged'] = int((df['damage_binary'] == 1).sum())
            profile['n_undamaged'] = int((df['damage_binary'] == 0).sum())
            profile['n_unlabeled'] = int((df['damage_binary'] < 0).sum())
            profile['damage_rate'] = float(profile['n_damaged'] / (profile['n_damaged'] + profile['n_undamaged'])) if (profile['n_damaged'] + profile['n_undamaged']) > 0 else None
            if 'city' in df.columns:
                profile['damage_rate_per_city'] = {}
                for city in profile['cities']:
                    c_df = df[df['city'] == city]
                    d = (c_df['damage_binary'] == 1).sum()
                    u = (c_df['damage_binary'] == 0).sum()
                    profile['damage_rate_per_city'][city] = float(d / (d + u)) if (d + u) > 0 else None

        # nan structure
        nan_rates = df[feature_cols].isna().mean()
        profile['nan_rate_overall'] = float(nan_rates.mean())
        profile['nan_rate_per_feature'] = {c: float(nan_rates[c]) for c in feature_cols}
        profile['n_all_nan_features'] = int((nan_rates == 1.0).sum())
        profile['n_zero_nan_features'] = int((nan_rates == 0.0).sum())
        if 'city' in df.columns:
            profile['nan_rate_per_city'] = {}
            for city in profile['cities']:
                c_df = df[df['city'] == city]
                profile['nan_rate_per_city'][city] = float(c_df[feature_cols].isna().mean().mean())

        # per-feature descriptive stats
        profile['feature_stats'] = describe_features(df, feature_cols)

        profile['timestamp'] = datetime.now().isoformat()
        self.dataset_profiles.append(profile)
        print(f"  PROFILE: {parquet_name}: {len(df)} rows, {len(feature_cols)} features, "
              f"NaN={profile['nan_rate_overall']*100:.1f}%")
        return profile

    def log_statistics(self,
                       notebook=None,
                       cell_id='',
                       analysis_name='',
                       parquet_name='',
                       tier_selection=None,
                       cities=None,
                       n_buildings=None,
                       n_features_tested=None,
                       summary_metrics=None,
                       top_features=None,
                       results_records=None,
                       note='',
                       tags=None,
                       extra=None,
                       ):
        """Log NB06/NB08-style statistical analysis results (not ML predictions).
        Use log_experiment() for ML experiments with y_true/y_proba.

        Parameters
        ----------
        summary_metrics : dict
            Aggregate stats, e.g. {'mean_best_auc': 0.72, 'n_significant': 42}.
        top_features : list of dict
            Best features per some criterion, e.g.
            [{'feature': 'x', 'auc': 0.78, 'p': 1e-5}, ...].
        results_records : list of dict
            Full per-feature or per-city results (optional, can be large).
        """
        nb = notebook or self.notebook or 'unknown'
        ts = datetime.now().isoformat()

        record = OrderedDict()
        record['schema_version'] = self.SCHEMA_VERSION
        record['record_type'] = 'statistics'
        record['run_id'] = self._run_id
        record['timestamp'] = ts
        record['notebook'] = nb
        record['cell_id'] = cell_id
        record['analysis_name'] = analysis_name
        record['experiment_name'] = analysis_name
        record['parquet_name'] = parquet_name
        record['tier_selection'] = tier_selection if tier_selection is not None else []
        record['cities'] = cities if cities is not None else []
        record['n_cities'] = len(cities) if cities else 0
        record['n_buildings'] = n_buildings
        record['n_features_tested'] = n_features_tested
        record['summary_metrics'] = _safe_dict(summary_metrics) if summary_metrics else {}
        record['top_features'] = top_features if top_features else []
        record['results_records'] = results_records if results_records else []
        record['note'] = note
        record['tags'] = tags if tags else []
        record['extra'] = _safe_dict(extra) if extra else {}

        self.experiments.append(record)

        n_feat = n_features_tested or 0
        top_metric = ''
        if summary_metrics:
            for k in ['mean_best_auc', 'mean_jm', 'mean_auc']:
                if k in summary_metrics:
                    top_metric = f" {k}={summary_metrics[k]:.4f}"
                    break
        print(f"  STAT: {analysis_name:45s}{top_metric} "
              f"feat={n_feat} cities={record['n_cities']} [{nb}/{cell_id}]")

        return record

    def save(self, notebook=None):
        nb = notebook or self.notebook or 'unknown'
        ts = self._run_id

        # save experiments
        if self.experiments:
            path = self.registry_dir / f'{nb}_experiments_{ts}.json'
            with open(path, 'w') as f:
                json.dump(self.experiments, f, indent=2, default=_json_default)
            print(f"  SAVED: {path.name} ({len(self.experiments)} experiments)")

            # also save flat CSV for quick inspection
            flat_rows = []
            for exp in self.experiments:
                row = OrderedDict()
                row['experiment_name'] = exp.get('experiment_name', exp.get('analysis_name', ''))
                row['record_type'] = exp.get('record_type', 'experiment')
                row['notebook'] = exp.get('notebook', '')
                row['cell_id'] = exp.get('cell_id', '')
                row['parquet_name'] = exp.get('parquet_name', '')
                row['tier_selection'] = str(exp.get('tier_selection', []))
                row['n_cities'] = exp.get('n_cities', 0)
                row['cities'] = ','.join(exp.get('cities', []))
                row['n_buildings'] = exp.get('n_buildings')
                row['n_damaged'] = exp.get('n_damaged')
                row['n_undamaged'] = exp.get('n_undamaged')
                row['damage_rate'] = exp.get('damage_rate')
                row['classifier_name'] = exp.get('classifier_name', '')
                row['feature_set_name'] = exp.get('feature_set_name', '')
                row['n_features'] = exp.get('n_features', exp.get('n_features_tested'))
                row['cv_method'] = exp.get('cv_method', '')
                row['n_folds'] = exp.get('n_folds')
                row['imputation'] = exp.get('imputation', '')
                row['scaling'] = exp.get('scaling', '')
                row['class_weight'] = exp.get('class_weight', '')
                metrics = exp.get('metrics', exp.get('summary_metrics', {}))
                for metric_key in ['auc', 'f1', 'precision', 'recall', 'accuracy',
                                   'balanced_accuracy', 'mcc', 'average_precision',
                                   'log_loss', 'brier_score', 'specificity',
                                   'tp', 'fp', 'tn', 'fn',
                                   'precision_at_recall_80', 'precision_at_recall_90',
                                   'threshold_at_recall_90',
                                   'n_samples', 'prevalence',
                                   'mean_best_auc', 'mean_jm', 'mean_auc']:
                    row[metric_key] = metrics.get(metric_key) if isinstance(metrics, dict) else None
                fold = exp.get('fold_metrics', {})
                row['auc_mean'] = fold.get('auc_mean') if isinstance(fold, dict) else None
                row['auc_std'] = fold.get('auc_std') if isinstance(fold, dict) else None
                row['note'] = exp.get('note', '')
                row['is_reference'] = exp.get('is_reference', False)
                row['reference_paper'] = exp.get('reference_paper', '')
                row['timestamp'] = exp.get('timestamp', '')
                row['run_id'] = exp.get('run_id', '')
                flat_rows.append(row)
            csv_path = self.registry_dir / f'{nb}_experiments_{ts}.csv'
            pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
            print(f"  SAVED: {csv_path.name}")

        # save dataset profiles
        if self.dataset_profiles:
            path = self.registry_dir / f'{nb}_profiles_{ts}.json'
            with open(path, 'w') as f:
                json.dump(self.dataset_profiles, f, indent=2, default=_json_default)
            print(f"  SAVED: {path.name} ({len(self.dataset_profiles)} profiles)")

        return self.registry_dir

    # --- loading (used by NB13) ---

    @staticmethod
    def load_all(results_root):
        registry_dir = Path(results_root) / 'registry'
        if not registry_dir.exists():
            print(f"  Registry directory not found: {registry_dir}")
            return [], []

        all_experiments = []
        all_profiles = []

        # load experiment JSONs (take latest per notebook)
        exp_files = sorted(registry_dir.glob('*_experiments_*.json'),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        seen_notebooks = set()
        for f in exp_files:
            # extract notebook name from filename: NB09a_experiments_YYYYMMDD_HHMMSS.json
            parts = f.stem.split('_experiments_')
            nb = parts[0] if parts else f.stem
            if nb in seen_notebooks:
                continue  # skip older runs of same notebook
            seen_notebooks.add(nb)
            with open(f) as fh:
                experiments = json.load(fh)
            all_experiments.extend(experiments)
            print(f"  Loaded: {f.name} ({len(experiments)} experiments, notebook={nb})")

        # load profile JSONs
        prof_files = sorted(registry_dir.glob('*_profiles_*.json'),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        seen_prof_nbs = set()
        for f in prof_files:
            parts = f.stem.split('_profiles_')
            nb = parts[0] if parts else f.stem
            if nb in seen_prof_nbs:
                continue
            seen_prof_nbs.add(nb)
            with open(f) as fh:
                profiles = json.load(fh)
            all_profiles.extend(profiles)
            print(f"  Loaded: {f.name} ({len(profiles)} profiles)")

        print(f"  Total: {len(all_experiments)} experiments, {len(all_profiles)} profiles "
              f"from {len(seen_notebooks)} notebooks")
        return all_experiments, all_profiles

    @staticmethod
    def load_all_runs(results_root):
        registry_dir = Path(results_root) / 'registry'
        if not registry_dir.exists():
            return [], []
        all_experiments = []
        all_profiles = []
        for f in sorted(registry_dir.glob('*_experiments_*.json')):
            with open(f) as fh:
                all_experiments.extend(json.load(fh))
        for f in sorted(registry_dir.glob('*_profiles_*.json')):
            with open(f) as fh:
                all_profiles.extend(json.load(fh))
        return all_experiments, all_profiles

    @staticmethod
    def experiments_to_dataframe(experiments):
        if not experiments:
            return pd.DataFrame()
        rows = []
        for exp in experiments:
            row = OrderedDict()
            row['record_type'] = exp.get('record_type', 'experiment')
            for key in ['experiment_name', 'notebook', 'cell_id',
                        'parquet_name', 'parquet_fmt',
                        'n_cities', 'n_buildings', 'n_damaged', 'n_undamaged', 'damage_rate',
                        'classifier_name', 'feature_set_name',
                        'cv_method', 'n_folds', 'imputation', 'scaling', 'class_weight',
                        'note', 'is_reference', 'reference_paper',
                        'timestamp', 'run_id']:
                row[key] = exp.get(key)
            row['n_features'] = exp.get('n_features', exp.get('n_features_tested'))
            row['tier_selection'] = str(exp.get('tier_selection', []))
            row['cities'] = ','.join(exp.get('cities', []))
            row['tags'] = ','.join(exp.get('tags', []))
            metrics = exp.get('metrics', exp.get('summary_metrics', {}))
            for mk in ['auc', 'f1', 'precision', 'recall', 'accuracy',
                        'balanced_accuracy', 'mcc', 'average_precision',
                        'log_loss', 'brier_score', 'specificity', 'fpr',
                        'tp', 'fp', 'tn', 'fn',
                        'n_samples', 'prevalence',
                        'precision_at_recall_80', 'precision_at_recall_90', 'precision_at_recall_95',
                        'threshold_at_recall_90',
                        'mean_best_auc', 'mean_jm', 'mean_auc']:
                row[mk] = metrics.get(mk) if isinstance(metrics, dict) else None
            fold = exp.get('fold_metrics', {})
            row['auc_mean'] = fold.get('auc_mean')
            row['auc_std'] = fold.get('auc_std')
            row['auc_min'] = fold.get('auc_min')
            row['auc_max'] = fold.get('auc_max')
            row['f1_mean'] = fold.get('f1_mean')
            row['f1_std'] = fold.get('f1_std')
            rows.append(row)
        df = pd.DataFrame(rows)
        if 'auc' in df.columns:
            df = df.sort_values('auc', ascending=False, na_position='last').reset_index(drop=True)
        return df

    @staticmethod
    def profiles_to_dataframe(profiles):
        if not profiles:
            return pd.DataFrame()
        rows = []
        for p in profiles:
            rows.append({
                'parquet_name': p.get('parquet_name'),
                'tier_selection': str(p.get('tier_selection', [])),
                'n_rows': p.get('n_rows'),
                'n_features': p.get('n_features'),
                'n_cities': p.get('n_cities'),
                'n_damaged': p.get('n_damaged'),
                'n_undamaged': p.get('n_undamaged'),
                'damage_rate': p.get('damage_rate'),
                'nan_rate_overall': p.get('nan_rate_overall'),
                'n_all_nan_features': p.get('n_all_nan_features'),
                'n_zero_nan_features': p.get('n_zero_nan_features'),
                'timestamp': p.get('timestamp'),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def feature_stats_to_dataframe(profiles):
        if not profiles:
            return pd.DataFrame()
        rows = []
        for p in profiles:
            pq = p.get('parquet_name', '?')
            for fs in p.get('feature_stats', []):
                row = dict(fs)
                row['parquet_name'] = pq
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def per_city_to_dataframe(experiments):
        if not experiments:
            return pd.DataFrame()
        rows = []
        for exp in experiments:
            pcm = exp.get('per_city_metrics', {})
            for city, m in pcm.items():
                row = {
                    'experiment_name': exp.get('experiment_name'),
                    'notebook': exp.get('notebook'),
                    'city': city,
                    'parquet_name': exp.get('parquet_name'),
                    'classifier_name': exp.get('classifier_name'),
                    'feature_set_name': exp.get('feature_set_name'),
                }
                if isinstance(m, dict):
                    for k, v in m.items():
                        if k != 'city':
                            row[k] = v
                rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_dict(d):
    if d is None:
        return {}
    if isinstance(d, dict):
        return {str(k): _make_serializable(v) for k, v in d.items()}
    return {}


def _make_serializable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_make_serializable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _make_serializable(val) for k, val in v.items()}
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if pd.isna(v) if isinstance(v, float) else False:
        return None
    return v


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)
