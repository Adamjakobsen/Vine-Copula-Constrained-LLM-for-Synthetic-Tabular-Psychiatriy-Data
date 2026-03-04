import os
import numpy as np
import pandas as pd

"""
Compute latent disorder severities from item-level questionnaire data.

Internal standardized severity codes:
    tier_0, tier_1, tier_2, tier_3, tier_4

Display labels:
    Depression (PHQ-9):
        tier_0 -> none
        tier_1 -> mild
        tier_2 -> moderate
        tier_3 -> moderately_severe
        tier_4 -> severe

    Anxiety disorders:
        tier_0 -> none
        tier_1 -> mild
        tier_2 -> moderate
        tier_3 -> severe
        tier_4 -> extreme

Scoring:
    Depression:
        - 9 items, each 0–3
        - score = sum of items (0–27)
        - cutoffs: 0–4 / 5–9 / 10–14 / 15–19 / 20–27

    Anxiety disorders:
        - 10 items, each 0–4
        - score = average total score (total / 10), scale 0–4
        - cutoffs by midpoint around anchored severity integers:
            [0.0, 0.5)   -> tier_0
            [0.5, 1.5)   -> tier_1
            [1.5, 2.5)   -> tier_2
            [2.5, 3.5)   -> tier_3
            [3.5, 4.0]   -> tier_4

Missing-data handling:
    - if 1–2 items missing: prorate and round raw total
    - if >=3 items missing: do not score (NaN)
"""

# --- Configuration ---
INPUT_FILE = './data/data_processed.csv'
OUTPUT_FILE = './data/patient_severities.csv'

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


def _find_first_col(df: pd.DataFrame, prefix_lower: str) -> str | None:
    col_map = {c.lower(): c for c in df.columns}
    return next((col_map[c] for c in col_map if c.startswith(prefix_lower)), None)


def _canonicalize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Rename age/sex columns to canonical names used downstream."""
    age_col = _find_first_col(df, 'w1_age')
    sex_col = _find_first_col(df, 'w1_sex')

    rename = {}
    if age_col and age_col != 'W1_age_r':
        rename[age_col] = 'W1_age_r'
    if sex_col and sex_col != 'W1_sex_r':
        rename[sex_col] = 'W1_sex_r'

    return df.rename(columns=rename) if rename else df


def _get_id_col(df: pd.DataFrame) -> str:
    """Use existing ID column if present; otherwise create a stable source row id."""
    for candidate in ['ID', 'id', 'Id']:
        if candidate in df.columns:
            return candidate

    df['source_row_id'] = [f"src_{i+1:05d}" for i in range(len(df))]
    return 'source_row_id'


def _prorated_total(row: pd.Series, item_cols: list[str], n_items: int) -> float:
    """Prorate raw total if 1–2 items missing; return NaN if >=3 missing."""
    vals = pd.to_numeric(row[item_cols], errors='coerce')
    answered = vals.dropna()
    missing = n_items - len(answered)

    if missing >= 3:
        return np.nan

    partial_sum = float(answered.sum())

    if missing in (1, 2):
        prorated = partial_sum * n_items / len(answered)
        return float(np.round(prorated))

    return partial_sum


def depression_tier_code(total: float):
    if pd.isna(total):
        return pd.NA
    if total <= 4:
        return 'tier_0'
    if total <= 9:
        return 'tier_1'
    if total <= 14:
        return 'tier_2'
    if total <= 19:
        return 'tier_3'
    return 'tier_4'


def anxiety_tier_code(avg: float):
    if pd.isna(avg):
        return pd.NA
    if avg < 0.5:
        return 'tier_0'
    if avg < 1.5:
        return 'tier_1'
    if avg < 2.5:
        return 'tier_2'
    if avg < 3.5:
        return 'tier_3'
    return 'tier_4'


def calculate_severities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _canonicalize_demographics(df)
    id_col = _get_id_col(df)

    # --- Item columns ---
    dep_cols = [c for c in df.columns if 'depression_it' in c.lower()]
    sep_cols = [c for c in df.columns if 'separation_anxiety_it' in c.lower()]
    spec_cols = [c for c in df.columns if 'specific_phobia_it' in c.lower()]
    soc_cols = [c for c in df.columns if 'social_anxiety_it' in c.lower()]
    panic_cols = [c for c in df.columns if 'panic_it' in c.lower()]
    ag_cols = [c for c in df.columns if 'agoraphobia_it' in c.lower()]
    gad_cols = [c for c in df.columns if 'generalized_anxiety_it' in c.lower()]

    expected = {
        'depression': (dep_cols, 9),
        'separation_anxiety': (sep_cols, 10),
        'specific_phobia': (spec_cols, 10),
        'social_anxiety': (soc_cols, 10),
        'panic': (panic_cols, 10),
        'agoraphobia': (ag_cols, 10),
        'generalized_anxiety': (gad_cols, 10),
    }

    for name, (cols, n_expected) in expected.items():
        if len(cols) != n_expected:
            raise ValueError(
                f"{name}: expected {n_expected} item columns, found {len(cols)}.\nColumns: {cols}"
            )

    # --- Depression ---
    df['depression_score'] = df.apply(lambda r: _prorated_total(r, dep_cols, 9), axis=1)
    df['depression_tier_code'] = df['depression_score'].apply(depression_tier_code)
    df['depression_severity_label'] = df['depression_tier_code'].map(DEPRESSION_LABELS)
    df['depression_tier_level'] = (
        df['depression_tier_code']
        .str.replace('tier_', '', regex=False)
        .astype('Int64')
    )

    # --- Anxiety disorders ---
    for base, cols in [
        ('separation_anxiety', sep_cols),
        ('specific_phobia', spec_cols),
        ('social_anxiety', soc_cols),
        ('panic', panic_cols),
        ('agoraphobia', ag_cols),
        ('generalized_anxiety', gad_cols),
    ]:
        total = df.apply(lambda r: _prorated_total(r, cols, 10), axis=1)
        df[f'{base}_score'] = total / 10.0
        df[f'{base}_tier_code'] = df[f'{base}_score'].apply(anxiety_tier_code)
        df[f'{base}_severity_label'] = df[f'{base}_tier_code'].map(ANXIETY_LABELS)
        df[f'{base}_tier_level'] = (
            df[f'{base}_tier_code']
            .str.replace('tier_', '', regex=False)
            .astype('Int64')
        )

    demo_cols = [c for c in ['W1_age_r', 'W1_sex_r'] if c in df.columns]

    out_cols = [id_col] + demo_cols + [
        'depression_score', 'depression_tier_code', 'depression_severity_label', 'depression_tier_level',
        'separation_anxiety_score', 'separation_anxiety_tier_code', 'separation_anxiety_severity_label', 'separation_anxiety_tier_level',
        'specific_phobia_score', 'specific_phobia_tier_code', 'specific_phobia_severity_label', 'specific_phobia_tier_level',
        'social_anxiety_score', 'social_anxiety_tier_code', 'social_anxiety_severity_label', 'social_anxiety_tier_level',
        'panic_score', 'panic_tier_code', 'panic_severity_label', 'panic_tier_level',
        'agoraphobia_score', 'agoraphobia_tier_code', 'agoraphobia_severity_label', 'agoraphobia_tier_level',
        'generalized_anxiety_score', 'generalized_anxiety_tier_code', 'generalized_anxiety_severity_label', 'generalized_anxiety_tier_level',
    ]

    return df[out_cols]


def main() -> None:
    os.makedirs('./data', exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    processed = calculate_severities(df)
    processed.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved patient severities to {OUTPUT_FILE} (n={len(processed)})")


if __name__ == '__main__':
    main()