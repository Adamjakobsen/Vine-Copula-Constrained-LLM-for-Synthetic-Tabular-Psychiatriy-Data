import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvinecopulib as pv
import seaborn as sns
from scipy import stats as st
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

"""
Profile generation + latent-space validation.

Current version:
- Fit separate vines by sex x age-stratum on disorder score variables
- Sample exact real stratum sizes
- Assign exact real ages within each stratum by one-to-one optimal matching
- Preserve sex and age-group marginals exactly by construction
- Validate with:
    * score/full correlation diagnostics
    * tier prevalence
    * tier pairwise NMI
    * elevated-tier conditional probabilities
    * conditional score means by sex and age-group
    * latent-space propensity AUC (mean, CI, symmetric AUC)

Outputs are organized under:
./data/profile_generation_validation/
    plots/
    tables/
    summary.md
"""

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
INPUT_FILE = './data/patient_severities.csv'
OUTPUT_JSON = './data/synthetic_patient_profiles_vine_continuous.json'

VALID_DIR = './data/profile_generation_validation'
PLOTS_DIR = os.path.join(VALID_DIR, 'plots')
TABLES_DIR = os.path.join(VALID_DIR, 'tables')
SUMMARY_MD = os.path.join(VALID_DIR, 'summary.md')

BATCH_METRICS_CSV = os.path.join(TABLES_DIR, 'profile_generation_batch_metrics.csv')
STRATA_OVERVIEW_CSV = os.path.join(TABLES_DIR, 'strata_overview.csv')
TIER_PREVALENCE_CSV = os.path.join(TABLES_DIR, 'tier_prevalence_comparison.csv')
TIER_NMI_CSV = os.path.join(TABLES_DIR, 'tier_pairwise_nmi_comparison.csv')
TIER_CONDITIONAL_PROBS_CSV = os.path.join(TABLES_DIR, 'tier_conditional_probabilities.csv')
COND_MEANS_SEX_CSV = os.path.join(TABLES_DIR, 'conditional_score_means_by_sex.csv')
COND_MEANS_AGE_CSV = os.path.join(TABLES_DIR, 'conditional_score_means_by_age_group.csv')
PROPENSITY_CSV = os.path.join(TABLES_DIR, 'latent_space_propensity.csv')

CORR_PLOT = os.path.join(PLOTS_DIR, 'correlation_comparison_vine_full_profile.png')
DEMO_MARGINALS_PLOT = os.path.join(PLOTS_DIR, 'demographic_alignment_vine_conditional.png')
COND_MEANS_PLOT = os.path.join(PLOTS_DIR, 'conditional_score_means_differences.png')
TIER_NMI_PLOT = os.path.join(PLOTS_DIR, 'tier_nmi_comparison.png')

RANDOM_SEED = 42
N_CANDIDATE_BATCHES = 20

# Stratification
STRATA_AGE_Q = 3          # used for generation strata
DIAG_AGE_Q = 5            # used for diagnostic plots/tables
MIN_STRATUM_FOR_VINE = 25 # if smaller, fallback to sex-level vine

# Correlation relative error plot
REL_ERROR_MIN_ABS_RHO = 0.10

# Tier conditional checks
ELEVATED_TIER_LEVEL = 2

# Priority relationships for selection objective
PRIORITY_SCORE_PAIRS = [
    ('depression_score', 'generalized_anxiety_score'),
    ('panic_score', 'agoraphobia_score'),
    ('separation_anxiety_score', 'agoraphobia_score'),
    ('separation_anxiety_score', 'generalized_anxiety_score'),
]

PRIORITY_TIER_PAIRS = [
    ('panic_score', 'agoraphobia_score'),
    ('separation_anxiety_score', 'agoraphobia_score'),
    ('separation_anxiety_score', 'generalized_anxiety_score'),
]

# ---------------------------------------------------------------------
# Standardized tier mappings
# ---------------------------------------------------------------------
DEPRESSION_LABELS = {
    'tier_0': 'none',
    'tier_1': 'mild',
    'tier_2': 'moderate',
    'tier_3': 'moderately_severe',
    'tier_4': 'severe',
}

ANXIETY_LABELS = {
    'tier_0': 'none',
    'tier_1': 'mild',
    'tier_2': 'moderate',
    'tier_3': 'severe',
    'tier_4': 'extreme',
}


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def ensure_dirs() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs('./data', exist_ok=True)


def _find_first_col(cols: List[str], prefix_lower: str) -> str | None:
    col_map = {c.lower(): c for c in cols}
    return next((col_map[c] for c in col_map if c.startswith(prefix_lower)), None)


def _find_id_col(cols: List[str]) -> str | None:
    for candidate in ['ID', 'id', 'Id', 'source_row_id']:
        if candidate in cols:
            return candidate
    return None


def safe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_string(index=index) + "\n```"


def get_pseudo_obs(series: pd.Series) -> np.ndarray:
    return series.rank(method='average').to_numpy() / (len(series) + 1)


def inverse_ecdf(sim_u: np.ndarray, original_series: pd.Series) -> np.ndarray:
    sorted_data = np.sort(original_series.dropna().values)
    idx = np.clip(np.floor(sim_u * len(sorted_data)).astype(int), 0, len(sorted_data) - 1)
    return sorted_data[idx]


def depression_tier_code(total: float) -> str:
    if total <= 4:
        return 'tier_0'
    if total <= 9:
        return 'tier_1'
    if total <= 14:
        return 'tier_2'
    if total <= 19:
        return 'tier_3'
    return 'tier_4'


def anxiety_tier_code(avg: float) -> str:
    if avg < 0.5:
        return 'tier_0'
    if avg < 1.5:
        return 'tier_1'
    if avg < 2.5:
        return 'tier_2'
    if avg < 3.5:
        return 'tier_3'
    return 'tier_4'


def tier_code_and_label(score_col: str, value: float) -> Tuple[str, str]:
    if score_col.lower() == 'depression_score':
        code = depression_tier_code(value)
        return code, DEPRESSION_LABELS[code]
    code = anxiety_tier_code(value)
    return code, ANXIETY_LABELS[code]


def disorder_json_key(score_col: str) -> str:
    if score_col.lower() == 'depression_score':
        return 'DEPRESSION'
    return score_col[:-6].upper().replace('_', ' ')


def sex_to_label(x) -> str:
    s = str(x).strip().lower()
    if s in {'1', 'male', 'm'}:
        return 'Male'
    if s in {'2', 'female', 'f'}:
        return 'Female'
    return str(x)


def get_real_age_bins(age_series: pd.Series, q: int = 5) -> np.ndarray:
    age_num = pd.to_numeric(age_series, errors='coerce').dropna()
    n_unique = age_num.nunique()
    q_eff = max(2, min(q, n_unique))
    _, bins = pd.qcut(age_num, q=q_eff, retbins=True, duplicates='drop')
    bins = np.array(bins, dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def apply_age_bins(age_series: pd.Series, bins: np.ndarray) -> pd.Series:
    age_num = pd.to_numeric(age_series, errors='coerce')
    return pd.cut(age_num, bins=bins, include_lowest=True).astype(str)


def standardize_against_real(real_scores: pd.DataFrame, other_scores: pd.DataFrame) -> np.ndarray:
    mu = real_scores.mean(axis=0)
    sd = real_scores.std(axis=0).replace(0, 1.0)
    return ((other_scores - mu) / sd).to_numpy()


# ---------------------------------------------------------------------
# Tier utilities
# ---------------------------------------------------------------------
def tier_codes_from_scores(df_scores: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_scores.index)
    for col in df_scores.columns:
        if col.lower() == 'depression_score':
            out[col.replace('_score', '_tier_code')] = df_scores[col].apply(depression_tier_code)
        else:
            out[col.replace('_score', '_tier_code')] = df_scores[col].apply(anxiety_tier_code)
    return out


def tier_levels_from_codes(df_tiers: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_tiers.index)
    for c in df_tiers.columns:
        out[c] = df_tiers[c].str.replace('tier_', '', regex=False).astype(int)
    return out


# ---------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------
def corr_summary(real_corr: pd.DataFrame, synth_corr: pd.DataFrame) -> Dict[str, float]:
    diff = real_corr.values - synth_corr.values
    abs_diff = np.abs(diff)
    upper = np.triu_indices_from(abs_diff, k=1)

    mae = float(np.mean(abs_diff[upper]))
    maxae = float(np.max(abs_diff[upper]))

    strong_mask = np.abs(real_corr.values) >= REL_ERROR_MIN_ABS_RHO
    strong_upper = strong_mask & np.triu(np.ones_like(strong_mask, dtype=bool), k=1)

    if np.any(strong_upper):
        rel_abs = abs_diff[strong_upper] / np.abs(real_corr.values[strong_upper])
        mare = float(np.mean(rel_abs))
        maxre = float(np.max(rel_abs))
    else:
        mare = np.nan
        maxre = np.nan

    return {
        'mae': mae,
        'maxae': maxae,
        'mare_strong': mare,
        'maxre_strong': maxre,
    }


def pair_error(real_corr: pd.DataFrame, synth_corr: pd.DataFrame, a: str, b: str) -> float:
    return float(abs(real_corr.loc[a, b] - synth_corr.loc[a, b]))


# ---------------------------------------------------------------------
# Tier prevalence
# ---------------------------------------------------------------------
def tier_prevalence_table(real_scores: pd.DataFrame, synth_scores: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    rows = []

    for col in score_cols:
        if col.lower() == 'depression_score':
            real_tiers = real_scores[col].apply(depression_tier_code)
            synth_tiers = synth_scores[col].apply(depression_tier_code)
            tier_order = list(DEPRESSION_LABELS.keys())
            label_map = DEPRESSION_LABELS
        else:
            real_tiers = real_scores[col].apply(anxiety_tier_code)
            synth_tiers = synth_scores[col].apply(anxiety_tier_code)
            tier_order = list(ANXIETY_LABELS.keys())
            label_map = ANXIETY_LABELS

        real_p = real_tiers.value_counts(normalize=True).reindex(tier_order, fill_value=0.0)
        synth_p = synth_tiers.value_counts(normalize=True).reindex(tier_order, fill_value=0.0)

        for tier in tier_order:
            rows.append({
                'score_variable': col,
                'tier_code': tier,
                'severity_label': label_map[tier],
                'real_prevalence': real_p[tier],
                'synthetic_prevalence': synth_p[tier],
                'absolute_error': abs(real_p[tier] - synth_p[tier]),
            })

    return pd.DataFrame(rows)


def tier_prevalence_mae(real_scores: pd.DataFrame, synth_scores: pd.DataFrame, score_cols: List[str]) -> float:
    return float(tier_prevalence_table(real_scores, synth_scores, score_cols)['absolute_error'].mean())


# ---------------------------------------------------------------------
# Tier NMI and conditional probabilities
# ---------------------------------------------------------------------
def pairwise_tier_nmi_table(real_scores: pd.DataFrame, synth_scores: pd.DataFrame, score_cols: List[str]):
    real_tiers = tier_codes_from_scores(real_scores)
    synth_tiers = tier_codes_from_scores(synth_scores)

    names = score_cols
    real_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    synth_mat = pd.DataFrame(index=names, columns=names, dtype=float)

    rows = []
    upper_errors = []

    for i, c1 in enumerate(names):
        t1r = real_tiers[c1.replace('_score', '_tier_code')]
        t1s = synth_tiers[c1.replace('_score', '_tier_code')]
        for j, c2 in enumerate(names):
            t2r = real_tiers[c2.replace('_score', '_tier_code')]
            t2s = synth_tiers[c2.replace('_score', '_tier_code')]

            nmi_r = normalized_mutual_info_score(t1r, t2r, average_method='arithmetic')
            nmi_s = normalized_mutual_info_score(t1s, t2s, average_method='arithmetic')

            real_mat.loc[c1, c2] = nmi_r
            synth_mat.loc[c1, c2] = nmi_s

            if i < j:
                err = abs(nmi_r - nmi_s)
                upper_errors.append(err)
                rows.append({
                    'var1': c1,
                    'var2': c2,
                    'real_nmi': nmi_r,
                    'synthetic_nmi': nmi_s,
                    'absolute_error': err,
                })

    diff_mat = real_mat - synth_mat
    rows_df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        'tier_pairwise_nmi_mae': float(np.mean(upper_errors)),
        'tier_pairwise_nmi_maxae': float(np.max(upper_errors)),
    }])

    return real_mat, synth_mat, diff_mat, rows_df, summary


def conditional_elevated_prob_table(real_scores: pd.DataFrame, synth_scores: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    real_levels = tier_levels_from_codes(tier_codes_from_scores(real_scores))
    synth_levels = tier_levels_from_codes(tier_codes_from_scores(synth_scores))

    rows = []
    level_cols = [c.replace('_score', '_tier_code') for c in score_cols]

    for a in level_cols:
        for b in level_cols:
            if a == b:
                continue

            real_mask = real_levels[a] >= ELEVATED_TIER_LEVEL
            synth_mask = synth_levels[a] >= ELEVATED_TIER_LEVEL

            real_prob = float((real_levels.loc[real_mask, b] >= ELEVATED_TIER_LEVEL).mean()) if real_mask.any() else np.nan
            synth_prob = float((synth_levels.loc[synth_mask, b] >= ELEVATED_TIER_LEVEL).mean()) if synth_mask.any() else np.nan

            rows.append({
                'given_var': a.replace('_tier_code', '_score'),
                'target_var': b.replace('_tier_code', '_score'),
                'threshold': f'>= tier_{ELEVATED_TIER_LEVEL}',
                'real_prob': real_prob,
                'synthetic_prob': synth_prob,
                'absolute_error': abs(real_prob - synth_prob) if pd.notna(real_prob) and pd.notna(synth_prob) else np.nan,
            })

    return pd.DataFrame(rows).sort_values('absolute_error', ascending=False)


# ---------------------------------------------------------------------
# Conditional demographic diagnostics
# ---------------------------------------------------------------------
def conditional_score_means_tables(real_full: pd.DataFrame, synth_full: pd.DataFrame, score_cols: List[str], age_col: str, sex_col: str):
    sex_levels = sorted(pd.Index(real_full[sex_col].dropna().unique()).union(pd.Index(synth_full[sex_col].dropna().unique())))
    sex_rows = []

    for sex in sex_levels:
        for score in score_cols:
            r = float(real_full.loc[real_full[sex_col] == sex, score].mean())
            s = float(synth_full.loc[synth_full[sex_col] == sex, score].mean())
            sex_rows.append({
                'group_variable': sex_col,
                'group_value': sex,
                'score_variable': score,
                'real_mean': r,
                'synthetic_mean': s,
                'absolute_error': abs(r - s),
            })

    sex_df = pd.DataFrame(sex_rows)

    age_bins = get_real_age_bins(real_full[age_col], q=DIAG_AGE_Q)
    real_age_group = apply_age_bins(real_full[age_col], age_bins)
    synth_age_group = apply_age_bins(synth_full[age_col], age_bins)
    age_levels = list(pd.Index(real_age_group.unique()))

    age_rows = []
    for grp in age_levels:
        for score in score_cols:
            r = float(real_full.loc[real_age_group == grp, score].mean())
            s = float(synth_full.loc[synth_age_group == grp, score].mean())
            age_rows.append({
                'group_variable': 'age_group',
                'group_value': grp,
                'score_variable': score,
                'real_mean': r,
                'synthetic_mean': s,
                'absolute_error': abs(r - s),
            })

    age_df = pd.DataFrame(age_rows)
    overall_mae = float(pd.concat([sex_df[['absolute_error']], age_df[['absolute_error']]], axis=0)['absolute_error'].mean())
    return sex_df, age_df, overall_mae


def depression_conditional_penalty(sex_df: pd.DataFrame, age_df: pd.DataFrame) -> float:
    sex_dep = sex_df.loc[sex_df['score_variable'] == 'depression_score', 'absolute_error']
    age_dep = age_df.loc[age_df['score_variable'] == 'depression_score', 'absolute_error']
    vals = pd.concat([sex_dep, age_dep], axis=0)
    return float(vals.mean())


def depression_age_penalty(age_df: pd.DataFrame) -> float:
    vals = age_df.loc[age_df['score_variable'] == 'depression_score', 'absolute_error']
    return float(vals.mean())


# ---------------------------------------------------------------------
# Propensity AUC
# ---------------------------------------------------------------------
def latent_space_propensity(real_full: pd.DataFrame, synth_full: pd.DataFrame, score_cols: List[str], age_col: str, sex_col: str) -> pd.DataFrame:
    real = real_full.copy()
    synth = synth_full.copy()

    real[age_col] = real[age_col].astype(str)
    real[sex_col] = real[sex_col].astype(str)
    synth[age_col] = synth[age_col].astype(str)
    synth[sex_col] = synth[sex_col].astype(str)

    X = pd.concat([real, synth], axis=0).reset_index(drop=True)
    y = np.array([0] * len(real) + [1] * len(synth))

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

    pre = ColumnTransformer(
        transformers=[
            ('cat', ohe, [age_col, sex_col]),
            ('num', StandardScaler(), score_cols),
        ],
        remainder='drop',
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        max_iter=5000,
        solver='saga',
        penalty='l2',
        class_weight='balanced',
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    aucs = []

    for tr, te in skf.split(X, y):
        Xt = pre.fit_transform(X.iloc[tr], y[tr])
        Xv = pre.transform(X.iloc[te])
        clf.fit(Xt, y[tr])
        p = clf.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(y[te], p))

    auc_mean = float(np.mean(aucs))
    auc_ci = float(st.t.ppf(0.975, len(aucs) - 1) * st.sem(aucs)) if len(aucs) > 1 else 0.0
    auc_symmetric = float(max(auc_mean, 1.0 - auc_mean))

    return pd.DataFrame([{
        'latent_propensity_auc_mean': auc_mean,
        'latent_propensity_auc_ci95': auc_ci,
        'latent_propensity_auc_symmetric': auc_symmetric,
        'target': 0.5,
    }])


# ---------------------------------------------------------------------
# Vine fitting / simulation
# ---------------------------------------------------------------------
def fit_score_vine(real_scores: pd.DataFrame) -> pv.Vinecop:
    n, d = real_scores.shape
    U = np.zeros((n, d), dtype=float)

    for j, col in enumerate(real_scores.columns):
        U[:, j] = get_pseudo_obs(real_scores[col])

    controls = pv.FitControlsVinecop(
        trunc_lvl=d - 1,
        selection_criterion='aic',
        tree_criterion='tau',
        show_trace=False,
    )

    return pv.Vinecop.from_data(data=U, var_types=['c'] * d, controls=controls)


def simulate_score_batch(copula: pv.Vinecop, real_scores: pd.DataFrame, n_out: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    u_sim = copula.simulate(n=n_out)

    synth_scores = pd.DataFrame(index=np.arange(n_out), columns=real_scores.columns, dtype=float)
    for j, col in enumerate(real_scores.columns):
        synth_scores[col] = inverse_ecdf(u_sim[:, j], real_scores[col]).astype(float)

    return synth_scores


# ---------------------------------------------------------------------
# Stratified generation helpers
# ---------------------------------------------------------------------
def assign_exact_ages_within_stratum(real_scores: pd.DataFrame, synth_scores: pd.DataFrame, real_age: pd.Series) -> pd.Series:
    """
    Exact one-to-one assignment of real ages within a stratum.
    Since stratum already fixes sex and age band, score distance is enough.
    """
    X_real = standardize_against_real(real_scores, real_scores)
    X_syn = standardize_against_real(real_scores, synth_scores)

    cost = cdist(X_syn, X_real, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost)

    donor_order = np.empty(len(row_ind), dtype=int)
    donor_order[row_ind] = col_ind

    return real_age.iloc[donor_order].reset_index(drop=True)


def generate_stratified_candidate(
    real_full: pd.DataFrame,
    score_cols: List[str],
    age_col: str,
    sex_col: str,
    batch_seed: int,
    strata_bins: np.ndarray,
):
    real_full = real_full.copy()
    real_full['_age_stratum'] = apply_age_bins(real_full[age_col], strata_bins)

    parts = []
    overview_rows = []

    sex_levels = list(real_full[sex_col].dropna().unique())

    # Pre-fit sex-level vines for fallback
    sex_level_vines = {}
    sex_level_scores = {}
    for sex in sex_levels:
        sub = real_full.loc[real_full[sex_col] == sex].copy()
        sex_level_scores[sex] = sub[score_cols].copy().reset_index(drop=True)
        sex_level_vines[sex] = fit_score_vine(sex_level_scores[sex])

    age_strata_levels = list(pd.Index(real_full['_age_stratum'].unique()))

    for i, sex in enumerate(sex_levels):
        for j, age_stratum in enumerate(age_strata_levels):
            sub = real_full.loc[(real_full[sex_col] == sex) & (real_full['_age_stratum'] == age_stratum)].copy().reset_index(drop=True)
            n_sub = len(sub)
            if n_sub == 0:
                continue

            real_sub_scores = sub[score_cols].copy()
            real_sub_age = sub[age_col].copy()

            if n_sub >= MIN_STRATUM_FOR_VINE:
                model_used = 'sex_age_stratum_vine'
                copula = fit_score_vine(real_sub_scores)
                synth_scores = simulate_score_batch(
                    copula=copula,
                    real_scores=real_sub_scores,
                    n_out=n_sub,
                    seed=batch_seed + (i + 1) * 1000 + (j + 1) * 10,
                )
            else:
                model_used = 'sex_level_vine_fallback'
                copula = sex_level_vines[sex]
                synth_scores = simulate_score_batch(
                    copula=copula,
                    real_scores=sex_level_scores[sex],
                    n_out=n_sub,
                    seed=batch_seed + (i + 1) * 1000 + (j + 1) * 10,
                )

            synth_age = assign_exact_ages_within_stratum(
                real_scores=real_sub_scores if model_used == 'sex_age_stratum_vine' else sex_level_scores[sex],
                synth_scores=synth_scores,
                real_age=real_sub_age,
            )

            synth_sub = synth_scores.copy()
            synth_sub[age_col] = synth_age.values
            synth_sub[sex_col] = sex
            synth_sub['_age_stratum'] = age_stratum
            synth_sub = synth_sub[[age_col, sex_col, '_age_stratum'] + score_cols]
            parts.append(synth_sub)

            overview_rows.append({
                'sex_value': sex,
                'age_stratum': age_stratum,
                'n_real': n_sub,
                'model_used': model_used,
            })

    synth_full = pd.concat(parts, axis=0).reset_index(drop=True)

    rng = np.random.default_rng(batch_seed)
    order = rng.permutation(len(synth_full))
    synth_full = synth_full.iloc[order].reset_index(drop=True)

    overview_df = pd.DataFrame(overview_rows)
    return synth_full, overview_df


# ---------------------------------------------------------------------
# Priority penalties
# ---------------------------------------------------------------------
def priority_score_pair_penalty(real_scores: pd.DataFrame, synth_scores: pd.DataFrame) -> float:
    real_corr = real_scores.corr(method='spearman')
    synth_corr = synth_scores.corr(method='spearman')
    errs = [pair_error(real_corr, synth_corr, a, b) for a, b in PRIORITY_SCORE_PAIRS]
    return float(np.mean(errs))


def priority_tier_pair_penalty(real_scores: pd.DataFrame, synth_scores: pd.DataFrame) -> float:
    real_tiers = tier_codes_from_scores(real_scores)
    synth_tiers = tier_codes_from_scores(synth_scores)

    errs = []
    for a, b in PRIORITY_TIER_PAIRS:
        ar = real_tiers[a.replace('_score', '_tier_code')]
        br = real_tiers[b.replace('_score', '_tier_code')]
        as_ = synth_tiers[a.replace('_score', '_tier_code')]
        bs_ = synth_tiers[b.replace('_score', '_tier_code')]

        nmi_r = normalized_mutual_info_score(ar, br, average_method='arithmetic')
        nmi_s = normalized_mutual_info_score(as_, bs_, average_method='arithmetic')
        errs.append(abs(nmi_r - nmi_s))

    return float(np.mean(errs))


# ---------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------
def evaluate_candidate(real_full: pd.DataFrame, synth_full: pd.DataFrame, score_cols: List[str], age_col: str, sex_col: str) -> Dict[str, float]:
    real_score_corr = real_full[score_cols].corr(method='spearman')
    synth_score_corr = synth_full[score_cols].corr(method='spearman')
    score_metrics = corr_summary(real_score_corr, synth_score_corr)

    corr_cols = [age_col, sex_col] + score_cols
    real_full_corr = real_full[corr_cols].corr(method='spearman')
    synth_full_corr = synth_full[corr_cols].corr(method='spearman')
    full_metrics = corr_summary(real_full_corr, synth_full_corr)

    tier_prev_mae = tier_prevalence_mae(real_full[score_cols], synth_full[score_cols], score_cols)

    _, _, _, _, tier_nmi_summary = pairwise_tier_nmi_table(real_full[score_cols], synth_full[score_cols], score_cols)
    tier_nmi_mae = float(tier_nmi_summary.iloc[0]['tier_pairwise_nmi_mae'])

    sex_df, age_df, cond_demo_mae = conditional_score_means_tables(real_full, synth_full, score_cols, age_col, sex_col)
    dep_demo_pen = depression_conditional_penalty(sex_df, age_df)
    dep_age_pen = depression_age_penalty(age_df)

    priority_score_pen = priority_score_pair_penalty(real_full[score_cols], synth_full[score_cols])
    priority_tier_pen = priority_tier_pair_penalty(real_full[score_cols], synth_full[score_cols])

    objective = (
        1.00 * score_metrics['mae'] +
        0.75 * tier_prev_mae +
        0.60 * tier_nmi_mae +
        0.45 * cond_demo_mae +
        0.90 * dep_demo_pen +
        1.00 * dep_age_pen +
        0.60 * priority_score_pen +
        0.60 * priority_tier_pen +
        0.25 * full_metrics['mae'] +
        0.20 * score_metrics['maxae']
    )

    return {
        'objective': objective,
        'score_corr_mae': score_metrics['mae'],
        'score_corr_maxae': score_metrics['maxae'],
        'score_corr_mare_strong': score_metrics['mare_strong'],
        'full_corr_mae': full_metrics['mae'],
        'full_corr_maxae': full_metrics['maxae'],
        'full_corr_mare_strong': full_metrics['mare_strong'],
        'tier_prevalence_mae': tier_prev_mae,
        'tier_pairwise_nmi_mae': tier_nmi_mae,
        'conditional_demo_score_mae': cond_demo_mae,
        'depression_conditional_penalty': dep_demo_pen,
        'depression_age_penalty': dep_age_pen,
        'priority_score_pair_penalty': priority_score_pen,
        'priority_tier_pair_penalty': priority_tier_pen,
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_full_correlation_comparison(real_full: pd.DataFrame, synth_full: pd.DataFrame, corr_cols: List[str], out_path: str) -> None:
    real_corr = real_full[corr_cols].corr(method='spearman')
    synth_corr = synth_full[corr_cols].corr(method='spearman')

    diff = real_corr - synth_corr
    abs_diff = np.abs(diff)

    rel = abs_diff.copy()
    denom = np.abs(real_corr)
    strong_mask = denom >= REL_ERROR_MIN_ABS_RHO
    rel.loc[:, :] = np.nan
    rel[strong_mask] = (abs_diff[strong_mask] / denom[strong_mask]) * 100.0

    plt.figure(figsize=(24, 5))

    plt.subplot(1, 4, 1)
    sns.heatmap(real_corr, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Real Correlation (Age + Sex + Scores)')
    plt.xticks(rotation=90)

    plt.subplot(1, 4, 2)
    sns.heatmap(synth_corr, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Synthetic Correlation')
    plt.xticks(rotation=90)

    plt.subplot(1, 4, 3)
    sns.heatmap(
        diff,
        cmap='RdBu_r',
        vmin=-0.2,
        vmax=0.2,
        square=True,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7},
    )
    plt.title('Signed Error (Real - Synthetic)')
    plt.xticks(rotation=90)

    plt.subplot(1, 4, 4)
    sns.heatmap(
        rel,
        cmap='magma',
        vmin=0,
        vmax=100,
        square=True,
        annot=True,
        fmt='.0f',
        annot_kws={'size': 7},
        mask=rel.isna(),
    )
    plt.title(f'Absolute Relative Error (%)\nShown only if |real rho| ≥ {REL_ERROR_MIN_ABS_RHO:.2f}')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_demographic_alignment(real_full: pd.DataFrame, synth_full: pd.DataFrame, age_col: str, sex_col: str, out_path: str) -> Tuple[float, float]:
    real_sex = real_full[sex_col]
    synth_sex = synth_full[sex_col]

    sex_levels = sorted(pd.Index(real_sex.dropna().unique()).union(pd.Index(synth_sex.dropna().unique())))
    real_sex_p = real_sex.value_counts(normalize=True).reindex(sex_levels, fill_value=0.0)
    synth_sex_p = synth_sex.value_counts(normalize=True).reindex(sex_levels, fill_value=0.0)

    age_bins = get_real_age_bins(real_full[age_col], q=DIAG_AGE_Q)
    real_age_group = apply_age_bins(real_full[age_col], age_bins)
    synth_age_group = apply_age_bins(synth_full[age_col], age_bins)

    age_levels = list(pd.Index(real_age_group.unique()))
    real_age_p = real_age_group.value_counts(normalize=True).reindex(age_levels, fill_value=0.0)
    synth_age_p = synth_age_group.value_counts(normalize=True).reindex(age_levels, fill_value=0.0)

    sex_mae = float(np.mean(np.abs(real_sex_p.values - synth_sex_p.values)))
    age_mae = float(np.mean(np.abs(real_age_p.values - synth_age_p.values)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    sex_df = pd.DataFrame({'Real': real_sex_p, 'Synthetic': synth_sex_p})
    sex_df.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('Sex Distribution')
    axes[0].set_ylabel('Proportion')
    axes[0].set_xlabel(sex_col)

    age_df = pd.DataFrame({'Real': real_age_p, 'Synthetic': synth_age_p})
    age_df.plot(kind='bar', ax=axes[1], rot=45)
    axes[1].set_title('Age-Group Distribution (shared real-derived bins)')
    axes[1].set_ylabel('Proportion')
    axes[1].set_xlabel(age_col)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return sex_mae, age_mae


def plot_conditional_score_means(sex_df: pd.DataFrame, age_df: pd.DataFrame, out_path: str) -> None:
    sex_pivot = sex_df.pivot(index='group_value', columns='score_variable', values='synthetic_mean') - \
                sex_df.pivot(index='group_value', columns='score_variable', values='real_mean')

    age_pivot = age_df.pivot(index='group_value', columns='score_variable', values='synthetic_mean') - \
                age_df.pivot(index='group_value', columns='score_variable', values='real_mean')

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(sex_pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
    plt.title('Conditional Mean Difference by Sex\n(Synthetic - Real)')
    plt.xlabel('Score variable')
    plt.ylabel('Sex')

    plt.subplot(1, 2, 2)
    sns.heatmap(age_pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
    plt.title('Conditional Mean Difference by Age Group\n(Synthetic - Real)')
    plt.xlabel('Score variable')
    plt.ylabel('Age group')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_tier_nmi(real_mat: pd.DataFrame, synth_mat: pd.DataFrame, diff_mat: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(real_mat, cmap='viridis', vmin=0, vmax=1, square=True)
    plt.title('Real Pairwise Tier NMI')
    plt.xticks(rotation=90)

    plt.subplot(1, 3, 2)
    sns.heatmap(synth_mat, cmap='viridis', vmin=0, vmax=1, square=True)
    plt.title('Synthetic Pairwise Tier NMI')
    plt.xticks(rotation=90)

    plt.subplot(1, 3, 3)
    sns.heatmap(diff_mat, cmap='RdBu_r', center=0, square=True, annot=True, fmt='.2f')
    plt.title('Tier NMI Difference (Real - Synthetic)')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------
def write_summary_md(
    best_metrics: Dict[str, float],
    batch_df: pd.DataFrame,
    strata_df: pd.DataFrame,
    tier_prev_df: pd.DataFrame,
    tier_nmi_summary: pd.DataFrame,
    cond_probs_df: pd.DataFrame,
    sex_df: pd.DataFrame,
    age_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    sex_mae: float,
    age_mae: float,
) -> None:
    top_batches = batch_df.sort_values('objective').head(5).copy()

    top_tier_errors = (
        tier_prev_df.sort_values('absolute_error', ascending=False)
        .head(10)
        [['score_variable', 'tier_code', 'severity_label', 'real_prevalence', 'synthetic_prevalence', 'absolute_error']]
    )

    top_conditional_probs = (
        cond_probs_df.sort_values('absolute_error', ascending=False)
        .head(10)
        [['given_var', 'target_var', 'real_prob', 'synthetic_prob', 'absolute_error']]
    )

    top_sex_errors = (
        sex_df.sort_values('absolute_error', ascending=False)
        .head(10)
        [['group_value', 'score_variable', 'real_mean', 'synthetic_mean', 'absolute_error']]
    )

    top_age_errors = (
        age_df.sort_values('absolute_error', ascending=False)
        .head(10)
        [['group_value', 'score_variable', 'real_mean', 'synthetic_mean', 'absolute_error']]
    )

    with open(SUMMARY_MD, 'w') as f:
        f.write("# Profile generation validation summary\n\n")

        f.write("## Selected batch metrics\n\n")
        f.write(safe_to_markdown(pd.DataFrame([best_metrics]), index=False))
        f.write("\n\n")

        f.write("## Batch selection overview\n\n")
        f.write(safe_to_markdown(top_batches, index=False))
        f.write("\n\n")

        f.write("## Strata overview\n\n")
        f.write(safe_to_markdown(strata_df, index=False))
        f.write("\n\n")

        f.write("## Latent-space propensity\n\n")
        f.write("The symmetric AUC is `max(AUC, 1 - AUC)` and should be as close to **0.5** as possible.\n\n")
        f.write(safe_to_markdown(propensity_df, index=False))
        f.write("\n\n")

        f.write("## Key validation figures\n\n")
        f.write(f"![Correlation comparison](plots/{os.path.basename(CORR_PLOT)})\n\n")
        f.write(f"![Demographic alignment](plots/{os.path.basename(DEMO_MARGINALS_PLOT)})\n\n")
        f.write(f"![Conditional score means](plots/{os.path.basename(COND_MEANS_PLOT)})\n\n")
        f.write(f"![Tier NMI comparison](plots/{os.path.basename(TIER_NMI_PLOT)})\n\n")

        f.write("## Demographic marginal errors\n\n")
        f.write(f"- Sex distribution MAE: **{sex_mae:.4f}**\n")
        f.write(f"- Age-group distribution MAE: **{age_mae:.4f}**\n\n")

        f.write("## Tier prevalence errors\n\n")
        f.write(safe_to_markdown(top_tier_errors, index=False))
        f.write("\n\n")

        f.write("## Tier co-occurrence checks\n\n")
        f.write(safe_to_markdown(tier_nmi_summary, index=False))
        f.write("\n\n")
        f.write(safe_to_markdown(top_conditional_probs, index=False))
        f.write("\n\n")

        f.write("## Conditional demographic score means\n\n")
        f.write("Top sex-conditioned errors:\n\n")
        f.write(safe_to_markdown(top_sex_errors, index=False))
        f.write("\n\n")
        f.write("Top age-group-conditioned errors:\n\n")
        f.write(safe_to_markdown(top_age_errors, index=False))
        f.write("\n\n")

    print(f"Saved markdown summary to {SUMMARY_MD}")


# ---------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------
def generate_profiles(df: pd.DataFrame) -> List[dict]:
    cols = df.columns.tolist()

    id_col = _find_id_col(cols)
    age_col = _find_first_col(cols, 'w1_age')
    sex_col = _find_first_col(cols, 'w1_sex')

    if age_col is None or sex_col is None:
        raise ValueError("Could not locate W1_age* and W1_sex* columns in patient_severities.csv")

    score_cols = [c for c in cols if c.lower().endswith('_score')]
    if not score_cols:
        raise ValueError("No *_score columns found in patient_severities.csv")

    keep_cols = ([id_col] if id_col else []) + [age_col, sex_col] + score_cols
    df_clean = df[keep_cols].dropna().copy()
    real_full = df_clean[[age_col, sex_col] + score_cols].copy().reset_index(drop=True)

    strata_bins = get_real_age_bins(real_full[age_col], q=STRATA_AGE_Q)

    print(f"Dataset has {len(df_clean)} real patients.")
    print(f"Fitting separate vines by sex x age-stratum on {len(score_cols)} continuous score variables.")
    print("Assigning exact ages within each stratum via one-to-one optimal matching.")
    print(f"Generating {N_CANDIDATE_BATCHES} candidate synthetic batches and keeping the best.")

    batch_rows = []
    best_objective = None
    best_metrics = None
    best_synth_full = None
    best_strata_df = None

    for batch_id in range(N_CANDIDATE_BATCHES):
        batch_seed = RANDOM_SEED + batch_id + 1

        synth_full, strata_df = generate_stratified_candidate(
            real_full=real_full,
            score_cols=score_cols,
            age_col=age_col,
            sex_col=sex_col,
            batch_seed=batch_seed,
            strata_bins=strata_bins,
        )

        metrics = evaluate_candidate(
            real_full=real_full,
            synth_full=synth_full,
            score_cols=score_cols,
            age_col=age_col,
            sex_col=sex_col,
        )
        metrics['batch_id'] = batch_id + 1
        batch_rows.append(metrics)

        print(
            f"Batch {batch_id + 1:02d}: "
            f"objective={metrics['objective']:.4f}, "
            f"score_mae={metrics['score_corr_mae']:.4f}, "
            f"full_mae={metrics['full_corr_mae']:.4f}, "
            f"tier_prev_mae={metrics['tier_prevalence_mae']:.4f}, "
            f"tier_nmi_mae={metrics['tier_pairwise_nmi_mae']:.4f}, "
            f"cond_demo_mae={metrics['conditional_demo_score_mae']:.4f}, "
            f"dep_demo_pen={metrics['depression_conditional_penalty']:.4f}, "
            f"dep_age_pen={metrics['depression_age_penalty']:.4f}"
        )

        if best_objective is None or metrics['objective'] < best_objective:
            best_objective = metrics['objective']
            best_metrics = metrics
            best_synth_full = synth_full.copy()
            best_strata_df = strata_df.copy()

    batch_df = pd.DataFrame(batch_rows).sort_values('objective')
    batch_df.to_csv(BATCH_METRICS_CSV, index=False)
    print(f"\nSaved batch metrics to {BATCH_METRICS_CSV}")

    best_strata_df.to_csv(STRATA_OVERVIEW_CSV, index=False)
    print(f"Saved strata overview to {STRATA_OVERVIEW_CSV}")

    print("\nSelected best batch:")
    for k in [
        'batch_id', 'objective', 'score_corr_mae', 'score_corr_maxae', 'score_corr_mare_strong',
        'full_corr_mae', 'full_corr_maxae', 'full_corr_mare_strong',
        'tier_prevalence_mae', 'tier_pairwise_nmi_mae',
        'conditional_demo_score_mae', 'depression_conditional_penalty',
        'depression_age_penalty',
        'priority_score_pair_penalty', 'priority_tier_pair_penalty'
    ]:
        v = best_metrics[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    best_synth_scores = best_synth_full[score_cols].copy()
    real_scores = real_full[score_cols].copy()

    tier_prev_df = tier_prevalence_table(real_scores, best_synth_scores, score_cols)
    tier_prev_df.to_csv(TIER_PREVALENCE_CSV, index=False)
    print(f"Saved tier prevalence comparison to {TIER_PREVALENCE_CSV}")

    real_tier_nmi, synth_tier_nmi, diff_tier_nmi, tier_nmi_rows, tier_nmi_summary = pairwise_tier_nmi_table(
        real_scores, best_synth_scores, score_cols
    )
    tier_nmi_rows.to_csv(TIER_NMI_CSV, index=False)
    print(f"Saved tier pairwise NMI comparison to {TIER_NMI_CSV}")

    cond_probs_df = conditional_elevated_prob_table(real_scores, best_synth_scores, score_cols)
    cond_probs_df.to_csv(TIER_CONDITIONAL_PROBS_CSV, index=False)
    print(f"Saved tier conditional probabilities to {TIER_CONDITIONAL_PROBS_CSV}")

    sex_df, age_df, _ = conditional_score_means_tables(
        real_full, best_synth_full, score_cols, age_col, sex_col
    )
    sex_df.to_csv(COND_MEANS_SEX_CSV, index=False)
    age_df.to_csv(COND_MEANS_AGE_CSV, index=False)
    print(f"Saved conditional score means by sex to {COND_MEANS_SEX_CSV}")
    print(f"Saved conditional score means by age group to {COND_MEANS_AGE_CSV}")

    propensity_df = latent_space_propensity(real_full, best_synth_full, score_cols, age_col, sex_col)
    propensity_df.to_csv(PROPENSITY_CSV, index=False)
    print(f"Saved latent-space propensity to {PROPENSITY_CSV}")

    plot_full_correlation_comparison(
        real_full=real_full,
        synth_full=best_synth_full,
        corr_cols=[age_col, sex_col] + score_cols,
        out_path=CORR_PLOT,
    )
    print(f"Saved correlation comparison plot to {CORR_PLOT}")

    sex_mae, age_mae = plot_demographic_alignment(
        real_full=real_full,
        synth_full=best_synth_full,
        age_col=age_col,
        sex_col=sex_col,
        out_path=DEMO_MARGINALS_PLOT,
    )
    print(f"Sex distribution MAE        : {sex_mae:.4f}")
    print(f"Age-group distribution MAE  : {age_mae:.4f}")
    print(f"Saved demographic alignment plot to {DEMO_MARGINALS_PLOT}")

    plot_conditional_score_means(sex_df, age_df, COND_MEANS_PLOT)
    print(f"Saved conditional score means plot to {COND_MEANS_PLOT}")

    plot_tier_nmi(real_tier_nmi, synth_tier_nmi, diff_tier_nmi, TIER_NMI_PLOT)
    print(f"Saved tier NMI plot to {TIER_NMI_PLOT}")

    write_summary_md(
        best_metrics=best_metrics,
        batch_df=batch_df,
        strata_df=best_strata_df,
        tier_prev_df=tier_prev_df,
        tier_nmi_summary=tier_nmi_summary,
        cond_probs_df=cond_probs_df,
        sex_df=sex_df,
        age_df=age_df,
        propensity_df=propensity_df,
        sex_mae=sex_mae,
        age_mae=age_mae,
    )

    age_bins_diag = get_real_age_bins(real_full[age_col], q=DIAG_AGE_Q)
    synth_age_groups = apply_age_bins(best_synth_full[age_col], age_bins_diag)

    synthetic_patients: List[dict] = []

    for i in range(len(best_synth_full)):
        patient_json: Dict[str, object] = {'patient_id': f"syn_vine_{i+1:04d}"}

        age_val = best_synth_full.iloc[i][age_col]
        sex_val = best_synth_full.iloc[i][sex_col]

        patient_json['AGE'] = int(round(float(age_val))) if pd.notna(age_val) else None
        patient_json['SEX'] = sex_to_label(sex_val)
        patient_json['AGE_GROUP'] = str(synth_age_groups.iloc[i])

        for col in score_cols:
            score_val = float(best_synth_scores.iloc[i][col])
            key = disorder_json_key(col)
            tier_code, severity_label = tier_code_and_label(col, score_val)

            patient_json[key] = severity_label
            patient_json[f'{key}_TIER_CODE'] = tier_code
            patient_json[f'{key}_SCORE_TARGET'] = round(score_val, 3)

        synthetic_patients.append(patient_json)

    return synthetic_patients


def main() -> None:
    ensure_dirs()

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run get_severity.py first.")

    df = pd.read_csv(INPUT_FILE)
    profiles = generate_profiles(df)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(profiles, f, indent=4)

    print(f"\nSuccessfully saved {len(profiles)} JSON profiles to {OUTPUT_JSON}!")


if __name__ == '__main__':
    main()