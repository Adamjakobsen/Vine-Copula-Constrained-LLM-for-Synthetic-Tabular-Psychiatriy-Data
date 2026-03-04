import argparse
import json
import math
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.spatial.distance import cdist, pdist, jensenshannon

from get_severity import calculate_severities
from evaluate_synthetic_severity_utility import (
    ALL_TIER_LABELS,
    DEFAULT_CTGAN_CANDIDATES,
    DEFAULT_LLM_RUNS_CANDIDATES,
    DEFAULT_REAL_CANDIDATES,
    DEFAULT_TVAE_CANDIDATES,
    DISORDER_SPECS,
    LLM_VARIANT_PREFIX,
    METRIC_ORDER,
    PROTOCOLS,
    align_to_real_columns,
    class_distribution,
    coerce_numeric,
    confidence_interval_95,
    discover_llm_variants,
    first_existing_path,
    format_ci,
    get_predictor_columns,
    get_target_columns,
    get_target_vector,
    infer_real_demo_columns,
    load_csv,
    load_synthetic_dataset,
    metric_dict,
    parse_named_paths,
    pick_cv_strategy,
    safe_fit_predict,
    tune_logistic,
)


DEFAULT_OUTDIR = './results/severity_fidelity_utility'
MISSING_TOKEN = '__MISSING__'
MAIN_METHOD_ORDER = ['CTGAN', 'TVAE', 'LLM-Simple']
ABLATION_METHOD_ORDER = ['LLM-Full-Agentic', 'LLM-Validated', 'LLM-Demographics-Only']
FIDELITY_METRICS = [
    ('fidelity_jsd', 'JSD'),
    ('fidelity_mae_v', 'MAE$_V$'),
    ('fidelity_ed', 'ED'),
]
UTILITY_TABLE_METRICS = [
    ('Acc', 'Accuracy'),
    ('F1', 'Macro-F1'),
]

DISORDER_DISPLAY = {
    'depression': 'Depression',
    'separation_anxiety': 'Separation Anxiety',
    'specific_phobia': 'Specific Phobia',
    'social_anxiety': 'Social Anxiety',
    'panic': 'Panic',
    'agoraphobia': 'Agoraphobia',
    'generalized_anxiety': 'Generalized Anxiety',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate item-level fidelity of synthetic datasets against the real aligned item table '
            'and downstream severity-tier utility under TRTR/TSTR/TAUG protocols.'
        )
    )
    parser.add_argument('--real', type=str, default='', help='CSV with the real item-level dataset. If omitted, common project paths are tried automatically.')
    parser.add_argument('--ctgan', type=str, default='', help='Optional CTGAN CSV. If omitted, ./results/sdv_baselines/ctgan.csv is used when present.')
    parser.add_argument('--tvae', type=str, default='', help='Optional TVAE CSV. If omitted, ./results/sdv_baselines/tvae.csv is used when present.')
    parser.add_argument('--synthetic', action='append', default=[], help='Additional synthetic datasets in NAME=PATH format.')
    parser.add_argument('--llm-runs-dir', type=str, default='', help='Optional directory containing LLM run folders with workflow_*/responses.csv files. If omitted, common project paths are tried automatically.')
    parser.add_argument('--llm-response', action='append', default=[], help='Additional LLM response CSVs in NAME=PATH format. Demographics are reconstructed from nearby prompt bundles.')
    parser.add_argument('--llm-profile-mode', choices=['copula_profile', 'demographics_only', 'all'], default='all', help='Optional filter when discovering LLM runs. Default keeps all profile modes.')
    parser.add_argument('--outdir', type=str, default='', help='Output directory. Defaults to ./results/severity_fidelity_utility.')
    parser.add_argument(
        '--disorders',
        nargs='*',
        default=list(DISORDER_SPECS.keys()),
        choices=list(DISORDER_SPECS.keys()),
        help='Disorder severity targets to evaluate. Default: all.',
    )
    parser.add_argument('--outer-folds', type=int, default=5, help='Requested folds for outer CV.')
    parser.add_argument('--outer-repeats', type=int, default=3, help='Repeats for outer CV.')
    parser.add_argument('--inner-folds', type=int, default=3, help='Requested folds for inner tuning CV.')
    parser.add_argument('--age-bin-count', type=int, default=5, help='Requested number of quantile bins for the age predictor in fidelity evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()




def disorder_display_name(disorder: str) -> str:
    return DISORDER_DISPLAY.get(disorder, str(disorder).replace('_', ' ').title())


def canonicalize_dataset_name(name: str) -> str:
    raw = str(name).strip()
    low = raw.lower()
    if low == 'ctgan' or 'ctgan' == low:
        return 'CTGAN'
    if low == 'tvae' or 'tvae' == low:
        return 'TVAE'
    if raw in {'LLM-Simple', 'LLM-Full-Agentic', 'LLM-Validated', 'LLM-Demographics-Only'}:
        return raw
    if raw.startswith(f'{LLM_VARIANT_PREFIX}::'):
        tail = raw.split('::', 1)[1]
        parts = [p for p in tail.split('/') if p]
        profile = parts[-2] if len(parts) >= 2 else ''
        workflow = parts[-1] if len(parts) >= 1 else ''
        if profile == 'copula_profile' and workflow == 'simple':
            return 'LLM-Simple'
        if profile == 'copula_profile' and workflow == 'full_agentic':
            return 'LLM-Full-Agentic'
        if profile == 'copula_profile' and workflow == 'validated':
            return 'LLM-Validated'
        if profile == 'demographics_only' and workflow == 'simple':
            return 'LLM-Demographics-Only'
    return raw


def deduplicate_dataset_specs(specs: Sequence[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    kept: List[Tuple[str, str]] = []
    dropped: List[Dict[str, str]] = []
    seen_labels = set()
    seen_exact = set()
    for raw_name, path in specs:
        canonical = canonicalize_dataset_name(raw_name)
        resolved = str(Path(path).resolve())
        exact_key = (canonical, resolved)
        if exact_key in seen_exact:
            continue
        if canonical in seen_labels:
            dropped.append({'raw_name': raw_name, 'canonical_name': canonical, 'path': resolved, 'reason': 'duplicate_canonical_name'})
            continue
        seen_exact.add(exact_key)
        seen_labels.add(canonical)
        kept.append((canonical, path))
    return kept, dropped


def as_cat_str(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    out = pd.Series(index=series.index, dtype='string')
    if numeric.notna().any():
        vals = np.full(len(series), MISSING_TOKEN, dtype=object)
        mask = numeric.notna().to_numpy()
        arr = numeric.to_numpy(dtype=float, na_value=np.nan)
        rounded = np.round(arr[mask], 8)
        int_like = np.isclose(rounded, np.round(rounded), atol=1e-8)
        canon = []
        for v, is_int in zip(rounded, int_like):
            canon.append(str(int(round(v))) if is_int else f"{float(v):.8f}".rstrip('0').rstrip('.'))
        vals[mask] = canon
        out[:] = vals
        return out.astype(str)
    out = series.astype('string').fillna(MISSING_TOKEN)
    return out.astype(str)


def marginal_jsd_mean(x_real: pd.DataFrame, x_synth: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    per_col: Dict[str, float] = {}
    for column in x_real.columns:
        real_series = as_cat_str(x_real[column])
        synth_series = as_cat_str(x_synth[column])
        support = pd.Index(real_series.unique()).union(pd.Index(synth_series.unique()))
        real_counts = real_series.value_counts().reindex(support, fill_value=0).to_numpy(dtype=float)
        synth_counts = synth_series.value_counts().reindex(support, fill_value=0).to_numpy(dtype=float)
        p = real_counts / max(real_counts.sum(), 1.0)
        q = synth_counts / max(synth_counts.sum(), 1.0)
        per_col[column] = float(jensenshannon(p, q, base=2.0) ** 2)
    values = [v for v in per_col.values() if np.isfinite(v)]
    return (float(np.mean(values)) if values else np.nan), per_col


def _encode_joint_categorical_matrix(
    x_real: pd.DataFrame,
    x_synth: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    if list(x_real.columns) != list(x_synth.columns):
        raise ValueError('Real and synthetic fidelity tables must have identical column order.')
    real_cols: List[np.ndarray] = []
    synth_cols: List[np.ndarray] = []
    for column in x_real.columns:
        real_series = as_cat_str(x_real[column])
        synth_series = as_cat_str(x_synth[column])
        categories = pd.Index(real_series.unique()).union(pd.Index(synth_series.unique()))
        real_codes = pd.Categorical(real_series, categories=categories).codes.astype(np.int32, copy=False)
        synth_codes = pd.Categorical(synth_series, categories=categories).codes.astype(np.int32, copy=False)
        real_cols.append(real_codes)
        synth_cols.append(synth_codes)
    if not real_cols or not synth_cols:
        return np.empty((len(x_real), 0), dtype=np.int32), np.empty((len(x_synth), 0), dtype=np.int32)
    return np.column_stack(real_cols), np.column_stack(synth_cols)


def energy_distance_hamming(x_real: pd.DataFrame, x_synth: pd.DataFrame) -> float:
    real_arr, synth_arr = _encode_joint_categorical_matrix(x_real, x_synth)
    if real_arr.size == 0 or synth_arr.size == 0:
        return 0.0
    cross = cdist(real_arr, synth_arr, metric='hamming')
    within_real = pdist(real_arr, metric='hamming') if len(real_arr) > 1 else np.array([], dtype=float)
    within_synth = pdist(synth_arr, metric='hamming') if len(synth_arr) > 1 else np.array([], dtype=float)
    a = float(cross.mean()) if cross.size else 0.0
    b = float(within_real.mean()) if within_real.size else 0.0
    c = float(within_synth.mean()) if within_synth.size else 0.0
    val = max(2.0 * a - b - c, 0.0)
    return float(math.sqrt(val))


def _is_id_like_column(name: str) -> bool:
    low = str(name).strip().lower()
    return low in {'id', 'row_id', 'record_id'} or low.endswith('_id')


def build_item_fidelity_tables(
    real_aligned: pd.DataFrame,
    synth_aligned: pd.DataFrame,
    age_bin_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    keep_cols = [c for c in real_aligned.columns if c in synth_aligned.columns ]
    if not keep_cols:
        raise ValueError('No shared non-ID columns available for item-level fidelity.')
    real_items = real_aligned[keep_cols].copy().reset_index(drop=True)
    synth_items = synth_aligned[keep_cols].copy().reset_index(drop=True)
    real_items, synth_items, age_bin_meta = bin_age_for_fidelity(real_items, synth_items, requested_bins=age_bin_count)
    return real_items, synth_items, {'age_binning': age_bin_meta, 'n_columns_item_level': int(len(keep_cols))}


def compute_fidelity_metrics(task_real: pd.DataFrame, task_synth: pd.DataFrame) -> Dict[str, float]:
    jsd_mean, _ = marginal_jsd_mean(task_real, task_synth)
    mae_v = cramers_v_mae(task_real, task_synth)
    ed = energy_distance_hamming(task_real, task_synth)
    return {
        'fidelity_jsd': float(jsd_mean),
        'fidelity_mae_v': float(mae_v),
        'fidelity_ed': float(ed),
    }


def bias_corrected_cramers_v(table: pd.DataFrame | np.ndarray) -> float:
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    r, k = arr.shape
    n = float(arr.sum())
    if n <= 1 or r <= 1 or k <= 1:
        return 0.0
    try:
        chi2 = float(chi2_contingency(arr, correction=False)[0])
    except Exception:
        return 0.0
    phi2 = chi2 / n
    correction = ((k - 1.0) * (r - 1.0)) / max(n - 1.0, 1.0)
    phi2_corr = max(0.0, phi2 - correction)
    r_corr = r - (((r - 1.0) ** 2) / max(n - 1.0, 1.0))
    k_corr = k - (((k - 1.0) ** 2) / max(n - 1.0, 1.0))
    denom = min(k_corr - 1.0, r_corr - 1.0)
    if denom <= 0:
        return 0.0
    return float(math.sqrt(phi2_corr / denom))


def cramers_v_mae(x_real: pd.DataFrame, x_synth: pd.DataFrame) -> float:
    cols = list(x_real.columns)
    if len(cols) <= 1:
        return 0.0
    diffs: List[float] = []
    real_cache = {c: as_cat_str(x_real[c]) for c in cols}
    synth_cache = {c: as_cat_str(x_synth[c]) for c in cols}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            real_table = pd.crosstab(real_cache[c1], real_cache[c2], dropna=False)
            synth_table = pd.crosstab(synth_cache[c1], synth_cache[c2], dropna=False)
            v_real = bias_corrected_cramers_v(real_table)
            v_synth = bias_corrected_cramers_v(synth_table)
            diffs.append(abs(v_real - v_synth))
    return float(np.mean(diffs)) if diffs else 0.0


def _u_within_mismatch_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    n = float(counts.sum())
    if n <= 1:
        return 0.0
    same_prob = float(np.sum(counts * (counts - 1.0)) / (n * (n - 1.0)))
    return 1.0 - same_prob


def _between_mismatch_from_counts(counts_real: np.ndarray, counts_synth: np.ndarray) -> float:
    counts_real = counts_real.astype(float)
    counts_synth = counts_synth.astype(float)
    n = float(counts_real.sum())
    m = float(counts_synth.sum())
    if n == 0 or m == 0:
        return 0.0
    return 1.0 - float(np.dot(counts_real / n, counts_synth / m))


def energy_distance_squared_hamming(x_real: pd.DataFrame, x_synth: pd.DataFrame) -> float:
    total = 0.0
    for column in x_real.columns:
        real_series = as_cat_str(x_real[column])
        synth_series = as_cat_str(x_synth[column])
        support = pd.Index(real_series.unique()).union(pd.Index(synth_series.unique()))
        real_counts = real_series.value_counts().reindex(support, fill_value=0).to_numpy(dtype=float)
        synth_counts = synth_series.value_counts().reindex(support, fill_value=0).to_numpy(dtype=float)
        between = _between_mismatch_from_counts(real_counts, synth_counts)
        within_real = _u_within_mismatch_from_counts(real_counts)
        within_synth = _u_within_mismatch_from_counts(synth_counts)
        total += 2.0 * between - within_real - within_synth
    return float(max(total, 0.0))


def bin_age_for_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame, requested_bins: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    real_out = real_df.copy()
    synth_out = synth_df.copy()
    metadata: Dict[str, object] = {'applied': False, 'age_col': '', 'n_bins_requested': int(requested_bins), 'n_bins_actual': 0, 'labels': []}

    try:
        age_col, _ = infer_real_demo_columns(real_df)
    except Exception:
        return real_out, synth_out, metadata

    if age_col not in real_out.columns or age_col not in synth_out.columns:
        return real_out, synth_out, metadata

    real_age = pd.to_numeric(real_out[age_col], errors='coerce')
    synth_age = pd.to_numeric(synth_out[age_col], errors='coerce')
    observed = real_age.dropna()
    if observed.empty:
        real_out[age_col] = pd.Series(np.where(real_age.isna(), pd.NA, 'bin_1'), index=real_out.index, dtype='string')
        synth_out[age_col] = pd.Series(np.where(synth_age.isna(), pd.NA, 'bin_1'), index=synth_out.index, dtype='string')
        metadata.update({'applied': True, 'age_col': age_col, 'n_bins_actual': 1, 'labels': ['bin_1']})
        return real_out, synth_out, metadata

    quantiles = np.linspace(0.0, 1.0, max(int(requested_bins), 1) + 1)
    edges = np.quantile(observed.to_numpy(dtype=float), quantiles)
    edges = np.unique(edges)
    if len(edges) <= 1:
        real_out[age_col] = pd.Series(np.where(real_age.isna(), pd.NA, 'bin_1'), index=real_out.index, dtype='string')
        synth_out[age_col] = pd.Series(np.where(synth_age.isna(), pd.NA, 'bin_1'), index=synth_out.index, dtype='string')
        metadata.update({'applied': True, 'age_col': age_col, 'n_bins_actual': 1, 'labels': ['bin_1']})
        return real_out, synth_out, metadata

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = [f'bin_{i + 1}' for i in range(len(edges) - 1)]
    real_out[age_col] = pd.cut(real_age, bins=edges, labels=labels, include_lowest=True, ordered=True).astype('string')
    synth_out[age_col] = pd.cut(synth_age, bins=edges, labels=labels, include_lowest=True, ordered=True).astype('string')
    metadata.update({'applied': True, 'age_col': age_col, 'n_bins_actual': len(labels), 'labels': labels})
    return real_out, synth_out, metadata


def evaluate_dataset(
    dataset_name: str,
    dataset_path: str,
    real_raw: pd.DataFrame,
    disorders: Sequence[str],
    outer_folds: int,
    outer_repeats: int,
    inner_folds: int,
    age_bin_count: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, List[Dict[str, object]]]:
    synth_source_df, load_mode = load_synthetic_dataset(dataset_path, real_df=real_raw)
    real_aligned, synth_aligned = align_to_real_columns(real_raw, synth_source_df)
    real_numeric = coerce_numeric(real_aligned)
    synth_numeric = coerce_numeric(synth_aligned)

    item_real, item_synth, fidelity_meta = build_item_fidelity_tables(real_aligned, synth_aligned, age_bin_count=age_bin_count)
    fidelity_metrics = compute_fidelity_metrics(item_real, item_synth)

    real_severity = calculate_severities(real_aligned)
    synth_severity = calculate_severities(synth_aligned)

    summary_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    fidelity_metadata_rows: List[Dict[str, object]] = []

    for disorder in disorders:
        target_cols = get_target_columns(real_numeric, disorder)
        predictor_cols = get_predictor_columns(real_numeric, target_cols)
        X_real = real_numeric[predictor_cols].copy().reset_index(drop=True)
        y_real = get_target_vector(real_severity, disorder).reset_index(drop=True)
        X_synth_full = synth_numeric[predictor_cols].copy().reset_index(drop=True)
        y_synth_full = get_target_vector(synth_severity, disorder).reset_index(drop=True)

        if y_real.isna().any():
            missing_real = int(y_real.isna().sum())
            raise ValueError(f'{dataset_name}/{disorder}: real data has {missing_real} missing severity targets.')

        synth_keep = y_synth_full.notna().to_numpy()
        X_synth = X_synth_full.loc[synth_keep].reset_index(drop=True)
        y_synth = y_synth_full.loc[synth_keep].astype(str).reset_index(drop=True)
        missing_synth = int((~synth_keep).sum())
        if len(X_synth) == 0:
            raise ValueError(f'{dataset_name}/{disorder}: synthetic data has no scorable severity targets.')

        real_counts = class_distribution(y_real.astype(str))
        synth_counts = class_distribution(y_synth.astype(str))
        class_rows.append({'dataset': dataset_name, 'disorder': disorder, 'source': 'real', **real_counts, 'missing_target': 0})
        class_rows.append({'dataset': dataset_name, 'disorder': disorder, 'source': 'synthetic', **synth_counts, 'missing_target': missing_synth})

        if not fidelity_metadata_rows:
            fidelity_metadata_rows.append(
                {
                    'dataset': dataset_name,
                    'dataset_path': dataset_path,
                    'dataset_load_mode': load_mode,
                    'disorder': 'all_items',
                    'n_rows_real_task': int(len(item_real)),
                    'n_rows_synth_task': int(len(item_synth)),
                    'n_columns_task': int(item_real.shape[1]),
                    'age_binning_json': json.dumps(fidelity_meta['age_binning']),
                }
            )

        outer_splitter, outer_strategy, outer_folds_actual, rarest_class = pick_cv_strategy(
            y=y_real,
            requested_folds=outer_folds,
            repeats=outer_repeats,
            seed=seed,
            stage=f'outer_evaluation/{dataset_name}/{disorder}',
        )
        outer_gen = outer_splitter.split(X_real, y_real if 'stratified' in outer_strategy else None)
        rng = np.random.default_rng(seed)

        protocol_metric_store = {protocol: {metric_name: [] for metric_name, _ in METRIC_ORDER} for protocol in PROTOCOLS}

        for fold_id, (train_idx, test_idx) in enumerate(outer_gen, start=1):
            X_real_train = X_real.iloc[train_idx].reset_index(drop=True)
            y_real_train = y_real.iloc[train_idx].astype(str).reset_index(drop=True)
            X_real_test = X_real.iloc[test_idx].reset_index(drop=True)
            y_real_test = y_real.iloc[test_idx].astype(str).reset_index(drop=True)

            estimator, best_label, best_inner_f1, inner_strategy, inner_folds_actual, inner_min_count, candidate_rows = tune_logistic(
                X=X_real_train,
                y=y_real_train,
                seed=seed,
                requested_folds=inner_folds,
            )
            for row in candidate_rows:
                fold_rows.append(
                    {
                        'dataset': dataset_name,
                        'dataset_path': dataset_path,
                        'dataset_load_mode': load_mode,
                        'disorder': disorder,
                        'fold_id': fold_id,
                        'protocol': 'TUNING_CANDIDATE',
                        'outer_strategy': outer_strategy,
                        'outer_folds_actual': outer_folds_actual,
                        'outer_repeats_actual': outer_repeats,
                        'outer_rarest_class_count': rarest_class,
                        'inner_strategy': row['inner_strategy'],
                        'inner_folds_actual': row['inner_folds_actual'],
                        'inner_rarest_class_count': row['inner_rarest_class_count'],
                        'best_candidate': row['candidate_label'],
                        'selected_for_fold': row['candidate_label'] == best_label,
                        'candidate_cv_f1_macro_mean': row['cv_f1_macro_mean'],
                        'candidate_cv_f1_macro_values_json': row['cv_f1_macro_values_json'],
                        'n_real_train': int(len(train_idx)),
                        'n_real_test': int(len(test_idx)),
                        'n_synth_train': int(len(train_idx)),
                    }
                )

            if len(X_synth) >= len(train_idx):
                synth_idx = rng.choice(len(X_synth), size=len(train_idx), replace=False)
            else:
                synth_idx = rng.choice(len(X_synth), size=len(train_idx), replace=True)
            X_synth_train = X_synth.iloc[synth_idx].reset_index(drop=True)
            y_synth_train = y_synth.iloc[synth_idx].reset_index(drop=True)

            x_aug = pd.concat([X_real_train, X_synth_train], axis=0, ignore_index=True)
            y_aug = pd.concat([y_real_train, y_synth_train], axis=0, ignore_index=True)

            predictions = {
                'TRTR': safe_fit_predict(estimator, X_real_train, y_real_train, X_real_test),
                'TSTR': safe_fit_predict(estimator, X_synth_train, y_synth_train, X_real_test),
                'TAUG': safe_fit_predict(estimator, x_aug, y_aug, X_real_test),
            }

            for protocol in PROTOCOLS:
                metrics = metric_dict(y_real_test.to_numpy(), predictions[protocol])
                for metric_name, _ in METRIC_ORDER:
                    protocol_metric_store[protocol][metric_name].append(metrics[metric_name])
                fold_rows.append(
                    {
                        'dataset': dataset_name,
                        'dataset_path': dataset_path,
                        'dataset_load_mode': load_mode,
                        'disorder': disorder,
                        'fold_id': fold_id,
                        'protocol': protocol,
                        'outer_strategy': outer_strategy,
                        'outer_folds_actual': outer_folds_actual,
                        'outer_repeats_actual': outer_repeats,
                        'outer_rarest_class_count': rarest_class,
                        'inner_strategy': inner_strategy,
                        'inner_folds_actual': inner_folds_actual,
                        'inner_rarest_class_count': inner_min_count,
                        'best_candidate': best_label,
                        'selected_for_fold': True,
                        'candidate_cv_f1_macro_mean': best_inner_f1,
                        'candidate_cv_f1_macro_values_json': '',
                        'n_real_train': int(len(train_idx)),
                        'n_real_test': int(len(test_idx)),
                        'n_synth_train': int(len(synth_idx)),
                        **metrics,
                    }
                )
                for local_row_id, true_label, pred_label in zip(test_idx, y_real_test.to_numpy(), predictions[protocol]):
                    pred_rows.append(
                        {
                            'dataset': dataset_name,
                            'dataset_path': dataset_path,
                            'dataset_load_mode': load_mode,
                            'disorder': disorder,
                            'fold_id': fold_id,
                            'protocol': protocol,
                            'row_id': int(local_row_id),
                            'y_true': str(true_label),
                            'y_pred': str(pred_label),
                        }
                    )

        summary_row: Dict[str, object] = {
            'dataset': dataset_name,
            'dataset_path': dataset_path,
            'dataset_load_mode': load_mode,
            'disorder': disorder,
            'n_real': int(len(X_real)),
            'n_synth': int(len(X_synth)),
            'n_predictors': int(X_real.shape[1]),
            'outer_strategy': outer_strategy,
            'outer_folds_actual': outer_folds_actual,
            'outer_repeats_actual': outer_repeats,
            'outer_rarest_class_count': rarest_class,
            'synth_missing_targets': missing_synth,
            **{f'real_{k}': v for k, v in real_counts.items()},
            **{f'synth_{k}': v for k, v in synth_counts.items()},
            **fidelity_metrics,
        }
        for protocol in PROTOCOLS:
            protocol_fold_df = pd.DataFrame(protocol_metric_store[protocol])
            for metric_name, metric_short in METRIC_ORDER:
                mean, ci95 = confidence_interval_95(protocol_fold_df[metric_name].tolist())
                summary_row[f'{protocol}_{metric_short}_mean'] = mean
                summary_row[f'{protocol}_{metric_short}_ci95'] = ci95
                summary_row[f'{protocol}_{metric_short}_fold_values'] = protocol_metric_store[protocol][metric_name]

        for _, metric_short in METRIC_ORDER:
            trtr = np.asarray(summary_row[f'TRTR_{metric_short}_fold_values'], dtype=float)
            tstr = np.asarray(summary_row[f'TSTR_{metric_short}_fold_values'], dtype=float)
            taug = np.asarray(summary_row[f'TAUG_{metric_short}_fold_values'], dtype=float)
            gain_tstr_mean, gain_tstr_ci = confidence_interval_95((tstr - trtr).tolist())
            gain_taug_mean, gain_taug_ci = confidence_interval_95((taug - trtr).tolist())
            summary_row[f'Gain_TSTR_{metric_short}_mean'] = gain_tstr_mean
            summary_row[f'Gain_TSTR_{metric_short}_ci95'] = gain_tstr_ci
            summary_row[f'Gain_TAUG_{metric_short}_mean'] = gain_taug_mean
            summary_row[f'Gain_TAUG_{metric_short}_ci95'] = gain_taug_ci

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    class_df = pd.DataFrame(class_rows)
    return summary_df, fold_df, pred_df, class_df, load_mode, fidelity_metadata_rows


def make_macro_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset, block in summary_df.groupby('dataset', sort=True):
        row: Dict[str, object] = {
            'dataset': dataset,
            'dataset_path': block['dataset_path'].iloc[0],
            'dataset_load_mode': block['dataset_load_mode'].iloc[0],
            'n_disorders': int(len(block)),
            'fidelity_jsd_mean': float(block['fidelity_jsd'].mean()),
            'fidelity_mae_v_mean': float(block['fidelity_mae_v'].mean()),
            'fidelity_ed_mean': float(block['fidelity_ed'].mean()),
        }
        for prefix in ['TRTR', 'TSTR', 'TAUG', 'Gain_TSTR', 'Gain_TAUG']:
            for _, metric_short in METRIC_ORDER:
                values = block[f'{prefix}_{metric_short}_mean'].astype(float).tolist()
                mean, ci95 = confidence_interval_95(values)
                row[f'{prefix}_{metric_short}_mean'] = mean
                row[f'{prefix}_{metric_short}_ci95'] = ci95
        rows.append(row)
    return pd.DataFrame(rows).sort_values('dataset').reset_index(drop=True)


def compute_fidelity_ablation_deltas(fidelity_df: pd.DataFrame, baseline: str = 'LLM-Simple') -> pd.DataFrame:
    fidelity_df = fidelity_df.drop_duplicates(subset=['dataset']).copy()
    if baseline not in set(fidelity_df['dataset']):
        return pd.DataFrame(columns=['scope', 'method', 'delta_jsd', 'delta_mae_v', 'delta_ed'])
    baseline_row = fidelity_df.loc[
        fidelity_df['dataset'] == baseline,
        ['fidelity_jsd', 'fidelity_mae_v', 'fidelity_ed'],
    ].iloc[0]
    rows: List[Dict[str, object]] = []
    for method in ABLATION_METHOD_ORDER:
        block = fidelity_df.loc[fidelity_df['dataset'] == method, ['fidelity_jsd', 'fidelity_mae_v', 'fidelity_ed']].copy()
        if block.empty:
            continue
        row = block.iloc[0]
        rows.append(
            {
                'scope': 'all_items',
                'method': method,
                'delta_jsd': float(baseline_row['fidelity_jsd'] - row['fidelity_jsd']),
                'delta_mae_v': float(baseline_row['fidelity_mae_v'] - row['fidelity_mae_v']),
                'delta_ed': float(baseline_row['fidelity_ed'] - row['fidelity_ed']),
            }
        )
    return pd.DataFrame(rows)


def compute_utility_ablation_deltas(fold_df: pd.DataFrame, baseline: str = 'LLM-Simple') -> pd.DataFrame:
    usable = fold_df.loc[fold_df['protocol'].isin(PROTOCOLS)].copy()
    if baseline not in set(usable['dataset']):
        return pd.DataFrame(
            columns=[
                'disorder', 'method',
                'Delta_TSTR_Acc_mean', 'Delta_TSTR_Acc_ci95', 'Delta_TAUG_Acc_mean', 'Delta_TAUG_Acc_ci95',
                'Delta_TSTR_F1_mean', 'Delta_TSTR_F1_ci95', 'Delta_TAUG_F1_mean', 'Delta_TAUG_F1_ci95',
            ]
        )
    rows: List[Dict[str, object]] = []
    baseline_block = usable.loc[usable['dataset'] == baseline].copy()

    for method in ABLATION_METHOD_ORDER:
        method_block = usable.loc[usable['dataset'] == method].copy()
        if method_block.empty:
            continue
        row: Dict[str, object] = {'method': method}
        available_disorders = sorted(set(baseline_block['disorder']).intersection(set(method_block['disorder'])))
        for disorder in available_disorders:
            record: Dict[str, object] = {'disorder': disorder, 'method': method}
            for protocol, short in [('TSTR', 'TSTR'), ('TAUG', 'TAUG')]:
                base = baseline_block.loc[(baseline_block['disorder'] == disorder) & (baseline_block['protocol'] == protocol), ['fold_id', 'accuracy', 'f1_macro']].copy()
                comp = method_block.loc[(method_block['disorder'] == disorder) & (method_block['protocol'] == protocol), ['fold_id', 'accuracy', 'f1_macro']].copy()
                merged = base.merge(comp, on='fold_id', how='inner', suffixes=('_base', '_method'))
                if merged.empty:
                    record[f'Delta_{short}_Acc_mean'] = np.nan
                    record[f'Delta_{short}_Acc_ci95'] = np.nan
                    record[f'Delta_{short}_F1_mean'] = np.nan
                    record[f'Delta_{short}_F1_ci95'] = np.nan
                    continue
                acc_mean, acc_ci = confidence_interval_95((merged['accuracy_method'] - merged['accuracy_base']).tolist())
                f1_mean, f1_ci = confidence_interval_95((merged['f1_macro_method'] - merged['f1_macro_base']).tolist())
                record[f'Delta_{short}_Acc_mean'] = acc_mean
                record[f'Delta_{short}_Acc_ci95'] = acc_ci
                record[f'Delta_{short}_F1_mean'] = f1_mean
                record[f'Delta_{short}_F1_ci95'] = f1_ci
            rows.append(record)
    return pd.DataFrame(rows)





def render_markdown(summary_df: pd.DataFrame, macro_df: pd.DataFrame, fidelity_delta_df: pd.DataFrame, utility_delta_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append('# Severity fidelity + utility benchmark')
    lines.append('')
    lines.append('')
    lines.append('## Macro-average across disorders')
    lines.append('')
    lines.append('| Dataset | JSD | MAE_V | ED | TSTR Acc | ΔAcc TSTR | TAUG Acc | ΔAcc TAUG | TSTR F1 | ΔF1 TSTR | TAUG F1 | ΔF1 TAUG |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for _, row in macro_df.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['fidelity_jsd_mean']:.3f} | {row['fidelity_mae_v_mean']:.3f} | {row['fidelity_ed_mean']:.3f} | "
            f"{format_ci(row['TSTR_Acc_mean'], row['TSTR_Acc_ci95'])} | {format_ci(row['Gain_TSTR_Acc_mean'], row['Gain_TSTR_Acc_ci95'], signed=True)} | "
            f"{format_ci(row['TAUG_Acc_mean'], row['TAUG_Acc_ci95'])} | {format_ci(row['Gain_TAUG_Acc_mean'], row['Gain_TAUG_Acc_ci95'], signed=True)} | "
            f"{format_ci(row['TSTR_F1_mean'], row['TSTR_F1_ci95'])} | {format_ci(row['Gain_TSTR_F1_mean'], row['Gain_TSTR_F1_ci95'], signed=True)} | "
            f"{format_ci(row['TAUG_F1_mean'], row['TAUG_F1_ci95'])} | {format_ci(row['Gain_TAUG_F1_mean'], row['Gain_TAUG_F1_ci95'], signed=True)} |"
        )
    lines.append('')
    lines.append('## Fidelity deltas versus LLM-Simple')
    lines.append('')
    if fidelity_delta_df.empty:
        lines.append('No LLM ablation variants were available.')
    else:
        lines.append('| Scope | Method | ΔJSD | ΔMAE_V | ΔED |')
        lines.append('|---|---|---:|---:|---:|')
        for _, row in fidelity_delta_df.sort_values(['method']).iterrows():
            scope = row['scope'] if 'scope' in row.index else 'all_items'
            lines.append(
                f"| {scope} | {row['method']} | {row['delta_jsd']:+.3f} | {row['delta_mae_v']:+.3f} | {row['delta_ed']:+.3f} |"
            )
    lines.append('')
    lines.append('## Utility deltas versus LLM-Simple')
    lines.append('')
    if utility_delta_df.empty:
        lines.append('No LLM ablation variants were available.')
    else:
        lines.append('| Disorder | Method | ΔTSTR Acc | ΔTAUG Acc | ΔTSTR F1 | ΔTAUG F1 |')
        lines.append('|---|---|---:|---:|---:|---:|')
        for _, row in utility_delta_df.sort_values(['disorder', 'method']).iterrows():
            lines.append(
                f"| {row['disorder']} | {row['method']} | {format_ci(row['Delta_TSTR_Acc_mean'], row['Delta_TSTR_Acc_ci95'], signed=True)} | "
                f"{format_ci(row['Delta_TAUG_Acc_mean'], row['Delta_TAUG_Acc_ci95'], signed=True)} | "
                f"{format_ci(row['Delta_TSTR_F1_mean'], row['Delta_TSTR_F1_ci95'], signed=True)} | "
                f"{format_ci(row['Delta_TAUG_F1_mean'], row['Delta_TAUG_F1_ci95'], signed=True)} |"
            )
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()

    args.real = args.real or first_existing_path(DEFAULT_REAL_CANDIDATES)
    args.ctgan = args.ctgan or first_existing_path(DEFAULT_CTGAN_CANDIDATES)
    args.tvae = args.tvae or first_existing_path(DEFAULT_TVAE_CANDIDATES)
    args.llm_runs_dir = args.llm_runs_dir or first_existing_path(DEFAULT_LLM_RUNS_CANDIDATES)
    args.outdir = args.outdir or DEFAULT_OUTDIR

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.real):
        raise FileNotFoundError(
            f'Real dataset not found at {args.real}. Checked defaults: {DEFAULT_REAL_CANDIDATES}'
        )
    real_raw = load_csv(args.real)

    dataset_specs: List[Tuple[str, str]] = []
    if os.path.exists(args.ctgan):
        dataset_specs.append(('CTGAN', args.ctgan))
    if os.path.exists(args.tvae):
        dataset_specs.append(('TVAE', args.tvae))
    dataset_specs.extend(parse_named_paths(args.synthetic))
    dataset_specs.extend(parse_named_paths(args.llm_response))
    if args.llm_runs_dir:
        dataset_specs.extend(discover_llm_variants(Path(args.llm_runs_dir), profile_mode_filter=args.llm_profile_mode))

    if not dataset_specs:
        raise FileNotFoundError(
            'No synthetic dataset files were found. Provide --ctgan/--tvae, --synthetic NAME=PATH, '\
            '--llm-response NAME=PATH, and/or --llm-runs-dir. '
            f'Defaults checked: ctgan={DEFAULT_CTGAN_CANDIDATES}, tvae={DEFAULT_TVAE_CANDIDATES}, llm_runs_dir={DEFAULT_LLM_RUNS_CANDIDATES}.'
        )

    dataset_specs, dropped_specs = deduplicate_dataset_specs(dataset_specs)

    summary_parts: List[pd.DataFrame] = []
    fold_parts: List[pd.DataFrame] = []
    pred_parts: List[pd.DataFrame] = []
    class_parts: List[pd.DataFrame] = []
    dataset_manifest_rows: List[Dict[str, object]] = []
    fidelity_meta_rows: List[Dict[str, object]] = []

    for dataset_name, dataset_path in dataset_specs:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)
        print(f'Evaluating severity fidelity + utility for {dataset_name} from {dataset_path} ...')
        summary_df, fold_df, pred_df, class_df, load_mode, fidelity_rows = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            real_raw=real_raw,
            disorders=args.disorders,
            outer_folds=args.outer_folds,
            outer_repeats=args.outer_repeats,
            inner_folds=args.inner_folds,
            age_bin_count=args.age_bin_count,
            seed=args.seed,
        )
        dataset_manifest_rows.append(
            {
                'name': dataset_name,
                'path': str(Path(dataset_path).resolve()),
                'load_mode': load_mode,
                'is_llm': str(dataset_name).startswith('LLM'),
            }
        )
        summary_parts.append(summary_df)
        fold_parts.append(fold_df)
        pred_parts.append(pred_df)
        class_parts.append(class_df)
        fidelity_meta_rows.extend(fidelity_rows)

    summary_df = pd.concat(summary_parts, axis=0, ignore_index=True).sort_values(['dataset', 'disorder']).reset_index(drop=True)
    fold_df = pd.concat(fold_parts, axis=0, ignore_index=True)
    pred_df = pd.concat(pred_parts, axis=0, ignore_index=True)
    class_df = pd.concat(class_parts, axis=0, ignore_index=True)
    macro_df = make_macro_summary(summary_df)
    fidelity_summary_df = summary_df[['dataset', 'dataset_path', 'dataset_load_mode', 'fidelity_jsd', 'fidelity_mae_v', 'fidelity_ed']].drop_duplicates(subset=['dataset']).sort_values('dataset').reset_index(drop=True)
    fidelity_delta_df = compute_fidelity_ablation_deltas(fidelity_summary_df)
    utility_delta_df = compute_utility_ablation_deltas(fold_df)

    utility_csv_df = summary_df.copy()
    for prefix in ['TRTR', 'TSTR', 'TAUG']:
        for _, metric_short in METRIC_ORDER:
            col = f'{prefix}_{metric_short}_fold_values'
            utility_csv_df[col] = utility_csv_df[col].apply(json.dumps)

    fidelity_csv = outdir / 'fidelity_summary.csv'
    utility_csv = outdir / 'utility_summary.csv'
    combined_csv = outdir / 'combined_summary.csv'
    macro_csv = outdir / 'macro_summary.csv'
    folds_csv = outdir / 'fold_metrics.csv'
    preds_csv = outdir / 'predictions.csv'
    class_csv = outdir / 'class_distributions.csv'
    datasets_csv = outdir / 'datasets.csv'
    dropped_csv = outdir / 'dropped_duplicate_datasets.csv'
    fidelity_meta_csv = outdir / 'fidelity_task_metadata.csv'
    fidelity_delta_csv = outdir / 'fidelity_ablation_deltas.csv'
    utility_delta_csv = outdir / 'utility_ablation_deltas.csv'
    report_md = outdir / 'report.md'


    fidelity_summary_df.to_csv(fidelity_csv, index=False)
    utility_csv_df.to_csv(utility_csv, index=False)
    utility_csv_df.to_csv(combined_csv, index=False)
    macro_df.to_csv(macro_csv, index=False)
    fold_df.sort_values(['dataset', 'disorder', 'fold_id', 'protocol']).to_csv(folds_csv, index=False)
    pred_df.sort_values(['dataset', 'disorder', 'protocol', 'fold_id', 'row_id']).to_csv(preds_csv, index=False)
    class_df.sort_values(['dataset', 'disorder', 'source']).to_csv(class_csv, index=False)
    pd.DataFrame(dataset_manifest_rows).sort_values(['name', 'path']).to_csv(datasets_csv, index=False)
    pd.DataFrame(dropped_specs).to_csv(dropped_csv, index=False)
    pd.DataFrame(fidelity_meta_rows).sort_values(['dataset', 'disorder']).to_csv(fidelity_meta_csv, index=False)
    fidelity_delta_df.sort_values(['method']).to_csv(fidelity_delta_csv, index=False)
    utility_delta_df.sort_values(['disorder', 'method']).to_csv(utility_delta_csv, index=False)

    report_md.write_text(render_markdown(summary_df, macro_df, fidelity_delta_df, utility_delta_df), encoding='utf-8')


    mapped_demo_columns = list(infer_real_demo_columns(real_raw))




    print(f'Saved fidelity summary to {fidelity_csv}')
    print(f'Saved utility summary to {utility_csv}')
    print(f'Saved macro summary to {macro_csv}')
    print(f'Saved fold metrics to {folds_csv}')
    print(f'Saved predictions to {preds_csv}')
    print(f'Saved class distributions to {class_csv}')
    print(f'Saved fidelity task metadata to {fidelity_meta_csv}')



if __name__ == '__main__':
    main()
