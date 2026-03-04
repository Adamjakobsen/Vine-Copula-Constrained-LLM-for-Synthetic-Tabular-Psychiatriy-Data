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
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from get_severity import calculate_severities


DISORDER_SPECS: Dict[str, Dict[str, object]] = {
    'depression': {
        'prefix': 'depression_it',
        'expected_items': 9,
        'tier_col': 'depression_tier_code',
    },
    'separation_anxiety': {
        'prefix': 'separation_anxiety_it',
        'expected_items': 10,
        'tier_col': 'separation_anxiety_tier_code',
    },
    'specific_phobia': {
        'prefix': 'specific_phobia_it',
        'expected_items': 10,
        'tier_col': 'specific_phobia_tier_code',
    },
    'social_anxiety': {
        'prefix': 'social_anxiety_it',
        'expected_items': 10,
        'tier_col': 'social_anxiety_tier_code',
    },
    'panic': {
        'prefix': 'panic_it',
        'expected_items': 10,
        'tier_col': 'panic_tier_code',
    },
    'agoraphobia': {
        'prefix': 'agoraphobia_it',
        'expected_items': 10,
        'tier_col': 'agoraphobia_tier_code',
    },
    'generalized_anxiety': {
        'prefix': 'generalized_anxiety_it',
        'expected_items': 10,
        'tier_col': 'generalized_anxiety_tier_code',
    },
}

ALL_TIER_LABELS = [f'tier_{i}' for i in range(5)]
METRIC_ORDER = [
    ('accuracy', 'Acc'),
    ('precision_macro', 'Prec'),
    ('recall_macro', 'Rec'),
    ('f1_macro', 'F1'),
]
PROTOCOLS = ['TRTR', 'TSTR', 'TAUG']

DEFAULT_REAL_CANDIDATES = [
    './data_processed.csv',
    './data/data_processed.csv',
]
DEFAULT_CTGAN_CANDIDATES = [
    './results/sdv_baselines/ctgan.csv',
]
DEFAULT_TVAE_CANDIDATES = [
    './results/sdv_baselines/tvae.csv',
]
DEFAULT_LLM_RUNS_CANDIDATES = [
    './data/llm_generation_runs',
    './llm_generation_runs',
]
DEFAULT_OUTDIR = './results/synthetic_severity_utility'


def first_existing_path(candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate synthetic-data severity-tier utility using logistic regression only. '
            'For each disorder, severity tiers are computed from item responses and predicted '
            'from the remaining observed variables under TRTR, TSTR, and TAUG. '
            'LLM-generated response files can be discovered automatically from run directories; '
            'their age/sex predictors are reconstructed from prompt bundles.'
        )
    )
    parser.add_argument('--real', type=str, default='', help='CSV with the real item-level dataset. If omitted, common project paths are tried automatically.')
    parser.add_argument('--ctgan', type=str, default='', help='Optional CTGAN CSV. If omitted, ./results/sdv_baselines/ctgan.csv is used when present.')
    parser.add_argument('--tvae', type=str, default='', help='Optional TVAE CSV. If omitted, ./results/sdv_baselines/tvae.csv is used when present.')
    parser.add_argument('--synthetic', action='append', default=[], help='Additional synthetic datasets in NAME=PATH format.')
    parser.add_argument('--llm-runs-dir', type=str, default='', help='Optional directory containing LLM run folders with workflow_*/responses.csv files. If omitted, common project paths are tried automatically.')
    parser.add_argument('--llm-response', action='append', default=[], help='Additional LLM response CSVs in NAME=PATH format. Demographics are reconstructed from nearby prompt bundles.')
    parser.add_argument('--llm-profile-mode', choices=['copula_profile', 'demographics_only', 'all'], default='copula_profile', help='Optional filter when discovering LLM runs. Default focuses on the copula-constrained LLM contribution.')
    parser.add_argument('--outdir', type=str, default='', help='Output directory. Defaults to ./results/synthetic_severity_utility.')
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


def parse_named_paths(entries: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for entry in entries:
        if '=' not in entry:
            raise ValueError(f"Expected NAME=PATH format, got: {entry}")
        name, path = entry.split('=', 1)
        parsed.append((name.strip(), path.strip()))
    return parsed


def confidence_interval_95(values: Iterable[float]) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    mean = float(np.mean(vals)) if len(vals) else np.nan
    if len(vals) <= 1:
        return mean, 0.0
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    return mean, 1.96 * se


def format_ci(mean: float, ci: float, signed: bool = False) -> str:
    low = mean - ci
    high = mean + ci
    if signed:
        return f'{mean:+.3f} [{low:+.3f}, {high:+.3f}]'
    return f'{mean:.3f} [{low:.3f}, {high:.3f}]'


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path).reset_index(drop=True)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_first_col(columns: Sequence[str], prefix_lower: str) -> str | None:
    cmap = {str(c).lower(): str(c) for c in columns}
    return next((cmap[k] for k in cmap if k.startswith(prefix_lower)), None)


def infer_real_demo_columns(real_df: pd.DataFrame) -> Tuple[str, str]:
    age_col = find_first_col(real_df.columns, 'w1_age')
    sex_col = find_first_col(real_df.columns, 'w1_sex')
    if age_col is None or sex_col is None:
        raise ValueError('Could not locate real age/sex columns. Expected columns starting with W1_age and W1_sex.')
    return age_col, sex_col


def align_to_real_columns(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    real_cols = list(real_df.columns)
    synth_map = {c.lower(): c for c in synth_df.columns}
    missing = [c for c in real_cols if c.lower() not in synth_map]
    if missing:
        raise ValueError(f'Synthetic dataset is missing required columns: {missing}')
    synth_aligned = synth_df[[synth_map[c.lower()] for c in real_cols]].copy()
    synth_aligned.columns = real_cols
    return real_df.copy(), synth_aligned


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def normalize_sex_value(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {'1', '1.0', 'male', 'm'}:
        return 1
    if s in {'2', '2.0', 'female', 'f'}:
        return 2
    try:
        return int(round(float(s)))
    except Exception:
        return np.nan


LLM_VARIANT_PREFIX = 'LLM'


def get_run_dir_from_responses_path(path: Path) -> Path:
    if path.parent.name.startswith('workflow_'):
        return path.parent.parent
    return path.parent


def load_prompt_manifest(run_dir: Path) -> Dict[str, object]:
    manifest_path = run_dir / 'prompt_manifest.json'
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def infer_profile_mode_from_manifest(run_dir: Path) -> str:
    manifest = load_prompt_manifest(run_dir)
    profile_mode = manifest.get('profile_mode')
    if isinstance(profile_mode, str) and profile_mode:
        return profile_mode
    return 'unknown_profile'


def discover_llm_variants(runs_dir: Path, profile_mode_filter: str = 'all') -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []
    if not runs_dir.exists():
        return variants

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        profile_mode = infer_profile_mode_from_manifest(run_dir)
        if profile_mode_filter != 'all' and profile_mode != profile_mode_filter:
            continue

        workflow_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('workflow_'))
        for workflow_dir in workflow_dirs:
            responses = workflow_dir / 'responses.csv'
            if responses.exists():
                workflow_name = workflow_dir.name.replace('workflow_', '')
                label = f'{LLM_VARIANT_PREFIX}::{run_dir.name}/{profile_mode}/{workflow_name}'
                variants.append((label, str(responses)))

        direct_responses = run_dir / 'responses.csv'
        if direct_responses.exists():
            label = f'{LLM_VARIANT_PREFIX}::{run_dir.name}/{profile_mode}'
            variants.append((label, str(direct_responses)))

    seen = set()
    deduped: List[Tuple[str, str]] = []
    for label, path in variants:
        key = (label, str(Path(path).resolve()))
        if key not in seen:
            seen.add(key)
            deduped.append((label, path))
    return deduped


def find_prompt_bundle_for_responses(path: Path) -> Path | None:
    candidates = []
    if path.parent.name.startswith('workflow_'):
        candidates.append(path.parent.parent / 'prompt_bundle.jsonl')
    candidates.append(path.parent / 'prompt_bundle.jsonl')
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def reconstruct_llm_demographics_df(responses_path: Path, real_df: pd.DataFrame) -> pd.DataFrame | None:
    prompt_bundle = find_prompt_bundle_for_responses(responses_path)
    if prompt_bundle is None:
        return None

    responses_df = load_csv(str(responses_path))
    id_col = next((c for c in responses_df.columns if str(c).lower() in {'id', 'patient_id'}), None)
    if id_col is None:
        raise ValueError(f'LLM responses at {responses_path} do not contain an ID column needed to merge prompt demographics.')

    prompts = read_jsonl(prompt_bundle)
    age_col, sex_col = infer_real_demo_columns(real_df)
    meta_rows: List[Dict[str, object]] = []
    for rec in prompts:
        profile_metadata = rec.get('profile_metadata', {}) or {}
        meta_rows.append(
            {
                'ID': rec.get('patient_id'),
                age_col: pd.to_numeric(profile_metadata.get('AGE'), errors='coerce'),
                sex_col: normalize_sex_value(profile_metadata.get('SEX')),
            }
        )
    demo_df = pd.DataFrame(meta_rows)
    if demo_df.empty:
        raise ValueError(f'Prompt bundle at {prompt_bundle} did not contain any patient metadata rows.')

    merged = responses_df.merge(demo_df, left_on=id_col, right_on='ID', how='left', suffixes=('', '_prompt'))
    if 'ID_prompt' in merged.columns:
        merged = merged.drop(columns=['ID_prompt'])

    missing_age = int(merged[age_col].isna().sum()) if age_col in merged.columns else len(merged)
    missing_sex = int(merged[sex_col].isna().sum()) if sex_col in merged.columns else len(merged)
    if missing_age or missing_sex:
        raise ValueError(
            f'Could not reconstruct all demographics for {responses_path}: '
            f'missing_age={missing_age}, missing_sex={missing_sex}. '
            f'Check ID alignment with {prompt_bundle}.'
        )
    return merged


def load_synthetic_dataset(path: str, real_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = load_csv(path)
    required = {c.lower() for c in real_df.columns}
    present = {c.lower() for c in df.columns}
    missing = required - present
    if not missing:
        return df, 'direct_csv'

    reconstructed = reconstruct_llm_demographics_df(Path(path), real_df)
    if reconstructed is not None:
        return reconstructed, 'llm_responses_plus_prompt_demographics'

    raise ValueError(
        f'{path} is missing required columns {sorted(missing)} and no usable prompt bundle was found to reconstruct LLM demographics.'
    )


def get_target_columns(df: pd.DataFrame, disorder: str) -> List[str]:
    prefix = str(DISORDER_SPECS[disorder]['prefix']).lower()
    target_cols = [c for c in df.columns if prefix in c.lower()]
    expected = int(DISORDER_SPECS[disorder]['expected_items'])
    if len(target_cols) != expected:
        raise ValueError(
            f'{disorder}: expected {expected} item columns using prefix {prefix!r}, found {len(target_cols)}: {target_cols}'
        )
    return target_cols


def get_predictor_columns(df: pd.DataFrame, target_cols: Sequence[str]) -> List[str]:
    excluded_ids = {'ID', 'id', 'Id', 'source_row_id'}
    return [c for c in df.columns if c not in target_cols and c not in excluded_ids]


def get_target_vector(scored_df: pd.DataFrame, disorder: str) -> pd.Series:
    tier_col = str(DISORDER_SPECS[disorder]['tier_col'])
    y = scored_df[tier_col].copy()
    return y.astype('string')


def class_distribution(y: pd.Series) -> Dict[str, int]:
    counts = y.value_counts(dropna=False).to_dict()
    return {label: int(counts.get(label, 0)) for label in ALL_TIER_LABELS}


def pick_cv_strategy(y: pd.Series, requested_folds: int, repeats: int, seed: int, stage: str):
    observed = y.dropna().astype(str)
    counts = observed.value_counts()
    if counts.empty:
        raise ValueError(f'{stage}: no non-missing labels available.')
    min_count = int(counts.min())
    if min_count >= 2:
        actual_folds = min(requested_folds, min_count)
        if actual_folds < 2:
            actual_folds = 2
        if repeats > 1:
            splitter = RepeatedStratifiedKFold(n_splits=actual_folds, n_repeats=repeats, random_state=seed)
            strategy = f'repeated_stratified_kfold({actual_folds}x{repeats})'
        else:
            splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
            strategy = f'stratified_kfold({actual_folds})'
    else:
        actual_folds = min(requested_folds, len(observed))
        if actual_folds < 2:
            raise ValueError(f'{stage}: need at least 2 samples for CV, found {len(observed)}.')
        if repeats > 1:
            splitter = RepeatedKFold(n_splits=actual_folds, n_repeats=repeats, random_state=seed)
            strategy = f'repeated_kfold_fallback({actual_folds}x{repeats}; rarest_class={min_count})'
        else:
            splitter = KFold(n_splits=actual_folds, shuffle=True, random_state=seed)
            strategy = f'kfold_fallback({actual_folds}; rarest_class={min_count})'
    return splitter, strategy, actual_folds, min_count


def logistic_candidates(seed: int) -> List[Tuple[str, Pipeline]]:
    candidates: List[Tuple[str, Pipeline]] = []
    for c in [0.1, 1.0, 10.0]:
        for class_weight in [None, 'balanced']:
            label = f'C={c},class_weight={class_weight}'
            estimator = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                (
                    'model',
                    LogisticRegression(
                        C=c,
                        class_weight=class_weight,
                        max_iter=500,
                        solver='lbfgs',
                        random_state=seed,
                    ),
                ),
            ])
            candidates.append((label, estimator))
    return candidates


def safe_fit_predict(estimator: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    nonmissing_mask = y_train.notna().to_numpy()
    x_train = x_train.iloc[nonmissing_mask].copy()
    y_train = y_train.iloc[nonmissing_mask].astype(str).copy()
    unique = np.unique(y_train)
    if len(unique) <= 1:
        fill = unique[0] if len(unique) == 1 else ALL_TIER_LABELS[0]
        return np.full(len(x_test), fill, dtype=object)
    est = clone(estimator)
    est.fit(x_train, y_train)
    return est.predict(x_test)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', labels=ALL_TIER_LABELS, zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', labels=ALL_TIER_LABELS, zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', labels=ALL_TIER_LABELS, zero_division=0)),
    }


def tune_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    requested_folds: int,
) -> Tuple[Pipeline, str, float, str, int, int, List[Dict[str, object]]]:
    splitter, strategy, actual_folds, min_count = pick_cv_strategy(
        y=y,
        requested_folds=requested_folds,
        repeats=1,
        seed=seed,
        stage='inner_tuning',
    )
    best_estimator: Pipeline | None = None
    best_label: str | None = None
    best_score: float | None = None
    candidate_rows: List[Dict[str, object]] = []
    split_gen = splitter.split(X, y if 'stratified' in strategy else None)

    for label, estimator in logistic_candidates(seed=seed):
        scores: List[float] = []
        for train_idx, valid_idx in split_gen:
            pred = safe_fit_predict(estimator, X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx])
            score = f1_score(
                y.iloc[valid_idx].astype(str),
                pred,
                average='macro',
                labels=ALL_TIER_LABELS,
                zero_division=0,
            )
            scores.append(float(score))
        split_gen = splitter.split(X, y if 'stratified' in strategy else None)
        mean_score = float(np.mean(scores))
        candidate_rows.append(
            {
                'candidate_label': label,
                'cv_f1_macro_mean': mean_score,
                'cv_f1_macro_values_json': json.dumps([float(x) for x in scores]),
                'inner_strategy': strategy,
                'inner_folds_actual': actual_folds,
                'inner_rarest_class_count': min_count,
            }
        )
        if best_score is None or mean_score > best_score:
            best_estimator = clone(estimator)
            best_label = label
            best_score = mean_score

    assert best_estimator is not None and best_label is not None and best_score is not None
    return best_estimator, best_label, best_score, strategy, actual_folds, min_count, candidate_rows


def summarize_protocol_metrics(fold_df: pd.DataFrame, summary_row: Dict[str, object], prefix: str) -> None:
    for metric_name, metric_short in METRIC_ORDER:
        mean, ci95 = confidence_interval_95(fold_df[metric_name].tolist())
        summary_row[f'{prefix}_{metric_short}_mean'] = mean
        summary_row[f'{prefix}_{metric_short}_ci95'] = ci95


def add_gain_metrics(summary_row: Dict[str, object]) -> None:
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


def render_markdown(summary_df: pd.DataFrame, macro_df: pd.DataFrame, class_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append('# Synthetic severity-tier utility benchmark')
    lines.append('')
    lines.append(
        'Utility is evaluated for each disorder severity target using logistic regression under paired outer CV. '
        'Within each outer training split, logistic hyperparameters are selected on the real training data only, '
        'then the chosen configuration is used for TRTR, TSTR, and TAUG.'
    )
    lines.append('')
    lines.append('## Macro-average across disorders')
    lines.append('')
    lines.append('| Dataset | Acc TRTR | Acc TSTR | ΔAcc TSTR | Acc TAUG | ΔAcc TAUG | F1 TRTR | F1 TSTR | ΔF1 TSTR | F1 TAUG | ΔF1 TAUG |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for _, row in macro_df.iterrows():
        lines.append(
            f"| {row['dataset']} | {format_ci(row['TRTR_Acc_mean'], row['TRTR_Acc_ci95'])} | "
            f"{format_ci(row['TSTR_Acc_mean'], row['TSTR_Acc_ci95'])} | "
            f"{format_ci(row['Gain_TSTR_Acc_mean'], row['Gain_TSTR_Acc_ci95'], signed=True)} | "
            f"{format_ci(row['TAUG_Acc_mean'], row['TAUG_Acc_ci95'])} | "
            f"{format_ci(row['Gain_TAUG_Acc_mean'], row['Gain_TAUG_Acc_ci95'], signed=True)} | "
            f"{format_ci(row['TRTR_F1_mean'], row['TRTR_F1_ci95'])} | "
            f"{format_ci(row['TSTR_F1_mean'], row['TSTR_F1_ci95'])} | "
            f"{format_ci(row['Gain_TSTR_F1_mean'], row['Gain_TSTR_F1_ci95'], signed=True)} | "
            f"{format_ci(row['TAUG_F1_mean'], row['TAUG_F1_ci95'])} | "
            f"{format_ci(row['Gain_TAUG_F1_mean'], row['Gain_TAUG_F1_ci95'], signed=True)} |"
        )
    lines.append('')
    lines.append('## Class distributions')
    lines.append('')
    lines.append('| Dataset | Disorder | Source | tier_0 | tier_1 | tier_2 | tier_3 | tier_4 | missing_target |')
    lines.append('|---|---|---|---:|---:|---:|---:|---:|---:|')
    for _, row in class_df.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['disorder']} | {row['source']} | {row['tier_0']} | {row['tier_1']} | {row['tier_2']} | {row['tier_3']} | {row['tier_4']} | {row['missing_target']} |"
        )
    lines.append('')
    for dataset in summary_df['dataset'].unique():
        lines.append(f'## {dataset}')
        lines.append('')
        lines.append('| Disorder | TRTR Acc | TSTR Acc | ΔAcc TSTR | TAUG Acc | ΔAcc TAUG | TRTR F1 | TSTR F1 | ΔF1 TSTR | TAUG F1 | ΔF1 TAUG |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        block = summary_df.loc[summary_df['dataset'] == dataset].copy()
        for _, row in block.iterrows():
            lines.append(
                f"| {row['disorder']} | {format_ci(row['TRTR_Acc_mean'], row['TRTR_Acc_ci95'])} | "
                f"{format_ci(row['TSTR_Acc_mean'], row['TSTR_Acc_ci95'])} | "
                f"{format_ci(row['Gain_TSTR_Acc_mean'], row['Gain_TSTR_Acc_ci95'], signed=True)} | "
                f"{format_ci(row['TAUG_Acc_mean'], row['TAUG_Acc_ci95'])} | "
                f"{format_ci(row['Gain_TAUG_Acc_mean'], row['Gain_TAUG_Acc_ci95'], signed=True)} | "
                f"{format_ci(row['TRTR_F1_mean'], row['TRTR_F1_ci95'])} | "
                f"{format_ci(row['TSTR_F1_mean'], row['TSTR_F1_ci95'])} | "
                f"{format_ci(row['Gain_TSTR_F1_mean'], row['Gain_TSTR_F1_ci95'], signed=True)} | "
                f"{format_ci(row['TAUG_F1_mean'], row['TAUG_F1_ci95'])} | "
                f"{format_ci(row['Gain_TAUG_F1_mean'], row['Gain_TAUG_F1_ci95'], signed=True)} |"
            )
        lines.append('')
    return '\n'.join(lines)


def evaluate_dataset(
    dataset_name: str,
    dataset_path: str,
    real_raw: pd.DataFrame,
    disorders: Sequence[str],
    outer_folds: int,
    outer_repeats: int,
    inner_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    synth_source_df, load_mode = load_synthetic_dataset(dataset_path, real_raw)
    real_aligned, synth_aligned = align_to_real_columns(real_raw, synth_source_df)
    real_numeric = coerce_numeric(real_aligned)
    synth_numeric = coerce_numeric(synth_aligned)

    real_severity = calculate_severities(real_aligned)
    synth_severity = calculate_severities(synth_aligned)

    summary_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []

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
        }
        for protocol in PROTOCOLS:
            protocol_fold_df = pd.DataFrame(protocol_metric_store[protocol])
            summarize_protocol_metrics(protocol_fold_df, summary_row, protocol)
            for metric_name, metric_short in METRIC_ORDER:
                summary_row[f'{protocol}_{metric_short}_fold_values'] = protocol_metric_store[protocol][metric_name]
        add_gain_metrics(summary_row)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    class_df = pd.DataFrame(class_rows)
    return summary_df, fold_df, pred_df, class_df, load_mode


def make_macro_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset, block in summary_df.groupby('dataset', sort=True):
        row: Dict[str, object] = {
            'dataset': dataset,
            'dataset_path': block['dataset_path'].iloc[0],
            'dataset_load_mode': block['dataset_load_mode'].iloc[0],
            'n_disorders': int(len(block)),
        }
        for prefix in ['TRTR', 'TSTR', 'TAUG', 'Gain_TSTR', 'Gain_TAUG']:
            for _, metric_short in METRIC_ORDER:
                values = block[f'{prefix}_{metric_short}_mean'].astype(float).tolist()
                mean, ci95 = confidence_interval_95(values)
                row[f'{prefix}_{metric_short}_mean'] = mean
                row[f'{prefix}_{metric_short}_ci95'] = ci95
        rows.append(row)
    return pd.DataFrame(rows).sort_values('dataset').reset_index(drop=True)


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
            'No synthetic dataset files were found. Provide --ctgan/--tvae, --synthetic NAME=PATH, '
            '--llm-response NAME=PATH, and/or --llm-runs-dir. '
            f'Defaults checked: ctgan={DEFAULT_CTGAN_CANDIDATES}, tvae={DEFAULT_TVAE_CANDIDATES}, llm_runs_dir={DEFAULT_LLM_RUNS_CANDIDATES}.'
        )

    seen = set()
    deduped_specs: List[Tuple[str, str]] = []
    for name, path in dataset_specs:
        key = (name, str(Path(path).resolve()))
        if key not in seen:
            seen.add(key)
            deduped_specs.append((name, path))
    dataset_specs = deduped_specs

    summary_parts: List[pd.DataFrame] = []
    fold_parts: List[pd.DataFrame] = []
    pred_parts: List[pd.DataFrame] = []
    class_parts: List[pd.DataFrame] = []
    dataset_manifest_rows: List[Dict[str, object]] = []

    for dataset_name, dataset_path in dataset_specs:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)
        print(f'Evaluating synthetic severity utility for {dataset_name} from {dataset_path} ...')
        summary_df, fold_df, pred_df, class_df, load_mode = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            real_raw=real_raw,
            disorders=args.disorders,
            outer_folds=args.outer_folds,
            outer_repeats=args.outer_repeats,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )
        dataset_manifest_rows.append(
            {
                'name': dataset_name,
                'path': str(Path(dataset_path).resolve()),
                'load_mode': load_mode,
                'is_llm': str(dataset_name).startswith(f'{LLM_VARIANT_PREFIX}::'),
            }
        )
        summary_parts.append(summary_df)
        fold_parts.append(fold_df)
        pred_parts.append(pred_df)
        class_parts.append(class_df)

    summary_df = pd.concat(summary_parts, axis=0, ignore_index=True)
    fold_df = pd.concat(fold_parts, axis=0, ignore_index=True)
    pred_df = pd.concat(pred_parts, axis=0, ignore_index=True)
    class_df = pd.concat(class_parts, axis=0, ignore_index=True)
    macro_df = make_macro_summary(summary_df)

    summary_for_csv = summary_df.copy()
    for prefix in ['TRTR', 'TSTR', 'TAUG']:
        for _, metric_short in METRIC_ORDER:
            col = f'{prefix}_{metric_short}_fold_values'
            summary_for_csv[col] = summary_for_csv[col].apply(json.dumps)

    summary_csv = outdir / 'summary.csv'
    macro_csv = outdir / 'macro_summary.csv'
    folds_csv = outdir / 'fold_metrics.csv'
    preds_csv = outdir / 'predictions.csv'
    class_csv = outdir / 'class_distributions.csv'
    datasets_csv = outdir / 'datasets.csv'
    report_md = outdir / 'report.md'
    manifest_json = outdir / 'manifest.json'

    summary_for_csv.sort_values(['dataset', 'disorder']).to_csv(summary_csv, index=False)
    macro_df.to_csv(macro_csv, index=False)
    fold_df.sort_values(['dataset', 'disorder', 'fold_id', 'protocol']).to_csv(folds_csv, index=False)
    pred_df.sort_values(['dataset', 'disorder', 'protocol', 'fold_id', 'row_id']).to_csv(preds_csv, index=False)
    class_df.sort_values(['dataset', 'disorder', 'source']).to_csv(class_csv, index=False)
    pd.DataFrame(dataset_manifest_rows).sort_values(['name', 'path']).to_csv(datasets_csv, index=False)
    report_md.write_text(render_markdown(summary_df.sort_values(['dataset', 'disorder']), macro_df, class_df.sort_values(['dataset', 'disorder', 'source'])), encoding='utf-8')

    manifest = {
        'real_data_path': str(Path(args.real).resolve()),
        'datasets': dataset_manifest_rows,
        'output_directory': str(outdir.resolve()),
        'disorders': list(args.disorders),
        'model_family': 'logistic_regression_only',
        'inner_tuning_grid': {
            'C': [0.1, 1.0, 10.0],
            'class_weight': [None, 'balanced'],
        },
        'target_definition': 'severity_tier_classification',
        'target_scoring_source': 'get_severity.calculate_severities',
        'exclude_target_block_from_predictors': True,
        'llm_runs_dir': str(Path(args.llm_runs_dir).resolve()) if args.llm_runs_dir else '',
        'llm_profile_mode_filter': args.llm_profile_mode,
        'llm_demographic_reconstruction': {
            'enabled': True,
            'source': 'prompt_bundle.jsonl next to each LLM responses.csv when needed',
            'mapped_real_columns': list(infer_real_demo_columns(real_raw)),
            'sex_mapping': {'male': 1, 'female': 2},
        },
        'outer_folds_requested': args.outer_folds,
        'outer_repeats_requested': args.outer_repeats,
        'inner_folds_requested': args.inner_folds,
        'seed': args.seed,
        'summary_csv': str(summary_csv.resolve()),
        'macro_summary_csv': str(macro_csv.resolve()),
        'fold_metrics_csv': str(folds_csv.resolve()),
        'predictions_csv': str(preds_csv.resolve()),
        'class_distributions_csv': str(class_csv.resolve()),
        'datasets_csv': str(datasets_csv.resolve()),
        'report_md': str(report_md.resolve()),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'Saved summary to {summary_csv}')
    print(f'Saved macro summary to {macro_csv}')
    print(f'Saved fold metrics to {folds_csv}')
    print(f'Saved predictions to {preds_csv}')
    print(f'Saved class distributions to {class_csv}')
    print(f'Saved dataset inventory to {datasets_csv}')
    print(f'Saved report to {report_md}')
    print(f'Saved manifest to {manifest_json}')


if __name__ == '__main__':
    main()
