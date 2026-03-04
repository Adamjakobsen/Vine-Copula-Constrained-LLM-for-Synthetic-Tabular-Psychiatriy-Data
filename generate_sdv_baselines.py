import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyreadstat

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer


TABLE_NAME = 'table'
DEFAULT_REAL_CSV_CANDIDATES = [
    './data_processed.csv',
    './data/data_processed.csv',
]
DEFAULT_REAL_SAV_CANDIDATES = [
    './data/data_set_final_osf.sav',
]
LEGACY_LEAKY_COLS = {'ID', 'ALLWAVES', 'TIEMPOIN', 'TIEMPOINTS'}
MISSING_TOKEN = '__MISSING__'


def first_existing_path(candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return candidates[0] if candidates else ''


def find_first_col(columns: Sequence[str], prefix_lower: str) -> Optional[str]:
    cmap = {str(c).lower(): str(c) for c in columns}
    return next((cmap[k] for k in cmap if k.startswith(prefix_lower)), None)


def is_item_column(column: str) -> bool:
    low = str(column).lower()
    return low.endswith('_it') or '_it_' in low


def canonicalize_categorical(series: pd.Series) -> pd.Series:
    s = series.copy()
    numeric = pd.to_numeric(s, errors='coerce')
    non_missing = s.notna()
    numeric_non_missing = numeric[non_missing]

    if non_missing.any() and numeric_non_missing.notna().all():
        vals = numeric_non_missing.to_numpy(dtype=float)
        if len(vals) and np.all(np.isclose(vals, np.round(vals), atol=1e-8)):
            s = pd.Series(pd.array(np.round(numeric), dtype='Int64'), index=s.index)
        else:
            s = pd.Series(np.round(numeric.to_numpy(dtype=float), 8), index=s.index)

    out = s.astype('string')
    out = out.fillna(MISSING_TOKEN)
    return out.astype(str)


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def load_real_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_to_drop = [c for c in df.columns if str(c).upper() in LEGACY_LEAKY_COLS]
    df = df.drop(columns=cols_to_drop, errors='ignore').reset_index(drop=True)
    return df


def load_real_from_sav(path: str) -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(path)
    cols = list(df.columns)

    w1_sex_end = next(i for i, col in enumerate(cols) if col.upper().startswith('W1_SEX'))
    w1_survey_start = next(i for i, col in enumerate(cols) if col.upper().startswith('W1_DEPRE'))
    w2_survey_start = next(i for i, col in enumerate(cols) if col.upper().startswith('W2_DEPRE'))

    feature_cols = cols[2:w1_sex_end + 1] + cols[w1_survey_start:w2_survey_start]
    x_real = df[feature_cols].copy()

    cols_to_drop = [c for c in x_real.columns if c.upper() in LEGACY_LEAKY_COLS]
    x_real = x_real.drop(columns=cols_to_drop, errors='ignore').reset_index(drop=True)
    return x_real


def load_real_table(real_path: str, real_sav_path: str) -> Tuple[pd.DataFrame, str]:
    if real_path:
        path = Path(real_path)
        if path.suffix.lower() == '.sav':
            return load_real_from_sav(str(path)), str(path)
        return load_real_from_csv(str(path)), str(path)

    csv_candidate = first_existing_path(DEFAULT_REAL_CSV_CANDIDATES)
    if csv_candidate and Path(csv_candidate).exists():
        return load_real_from_csv(csv_candidate), csv_candidate

    sav_candidate = real_sav_path or first_existing_path(DEFAULT_REAL_SAV_CANDIDATES)
    if sav_candidate and Path(sav_candidate).exists():
        return load_real_from_sav(sav_candidate), sav_candidate

    raise FileNotFoundError(
        'Could not locate a real training table. Provide --real pointing to the same CSV used by the evaluator '
        '(preferred), or provide --real-sav for the legacy SAV extraction path.'
    )


def infer_column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    age_col = find_first_col(df.columns, 'w1_age')
    sex_col = find_first_col(df.columns, 'w1_sex')

    categorical_cols: List[str] = []
    numerical_cols: List[str] = []
    other_cols: List[str] = []

    for column in df.columns:
        if column == age_col:
            numerical_cols.append(column)
        elif column == sex_col or is_item_column(column):
            categorical_cols.append(column)
        else:
            nunique = df[column].nunique(dropna=True)
            if nunique <= 20:
                categorical_cols.append(column)
            else:
                other_cols.append(column)
                numerical_cols.append(column)

    return {
        'age_col': [age_col] if age_col else [],
        'sex_col': [sex_col] if sex_col else [],
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'other_cols': other_cols,
    }


def prepare_training_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    roles = infer_column_roles(df)
    out = df.copy()

    # Keep age numerical; represent symptom items and sex as categorical with canonicalized discrete labels.
    for column in roles['categorical_cols']:
        out[column] = canonicalize_categorical(out[column])
    for column in roles['numerical_cols']:
        out[column] = coerce_numeric(out[column])

    # Drop rows that are entirely missing over modelled columns, but otherwise retain missingness.
    all_model_cols = list(dict.fromkeys(roles['categorical_cols'] + roles['numerical_cols']))
    if all_model_cols:
        out = out.loc[~out[all_model_cols].isna().all(axis=1)].reset_index(drop=True)

    meta = {
        'n_rows': int(len(out)),
        'n_columns': int(out.shape[1]),
        'categorical_cols': list(roles['categorical_cols']),
        'numerical_cols': list(roles['numerical_cols']),
        'age_col': roles['age_col'][0] if roles['age_col'] else '',
        'sex_col': roles['sex_col'][0] if roles['sex_col'] else '',
    }
    return out, meta


def make_mixed_metadata(data: pd.DataFrame, categorical_cols: Sequence[str], numerical_cols: Sequence[str]) -> Metadata:
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name=TABLE_NAME,
        infer_sdtypes=True,
        infer_keys=None,
    )
    if categorical_cols:
        metadata.update_columns(
            column_names=list(categorical_cols),
            sdtype='categorical',
            table_name=TABLE_NAME,
        )
    if numerical_cols:
        metadata.update_columns(
            column_names=list(numerical_cols),
            sdtype='numerical',
            table_name=TABLE_NAME,
        )
    metadata.validate()
    metadata.validate_table(data=data, table_name=TABLE_NAME)
    return metadata


def postprocess_sample(sampled: pd.DataFrame, real_reference: pd.DataFrame, categorical_cols: Sequence[str], numerical_cols: Sequence[str]) -> pd.DataFrame:
    out = sampled.copy()

    for column in categorical_cols:
        if column not in out.columns:
            continue
        out[column] = canonicalize_categorical(out[column])
        real_non_missing = canonicalize_categorical(real_reference[column])
        support = [v for v in pd.unique(real_non_missing) if v != MISSING_TOKEN]
        if support:
            invalid = ~out[column].isin(support + [MISSING_TOKEN])
            if invalid.any():
                out.loc[invalid, column] = MISSING_TOKEN

    for column in numerical_cols:
        if column not in out.columns:
            continue
        real_num = pd.to_numeric(real_reference[column], errors='coerce')
        out[column] = pd.to_numeric(out[column], errors='coerce')
        valid_real = real_num.dropna()
        if valid_real.empty:
            continue
        out[column] = out[column].clip(lower=float(valid_real.min()), upper=float(valid_real.max()))
        if np.all(np.isclose(valid_real.to_numpy(dtype=float), np.round(valid_real.to_numpy(dtype=float)), atol=1e-8)):
            out[column] = np.round(out[column]).astype('Int64')

    return out[real_reference.columns].copy()


def fit_and_sample(
    synthesizer_cls,
    metadata: Metadata,
    data: pd.DataFrame,
    n_rows: int,
    epochs: int,
    seed: int,
    enable_gpu: bool,
) -> pd.DataFrame:
    synthesizer = synthesizer_cls(
        metadata,
        epochs=epochs,
        enforce_min_max_values=True,
        enforce_rounding=True,
        enable_gpu=enable_gpu,
        verbose=False,
    )
    np.random.seed(seed)
    synthesizer.fit(data)
    sampled = synthesizer.sample(num_rows=n_rows)
    return sampled


def save_manifest(path: Path, manifest: Dict[str, object]) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Train CTGAN and TVAE baselines on the same real table used by downstream evaluation. '
            'Age is kept numerical; symptom items and sex are treated as categorical. '
            'This avoids the previous all-categorical representation that was especially unfavourable to TVAE.'
        )
    )
    parser.add_argument('--real', type=str, default='', help='Preferred: CSV used by evaluation (e.g., ./data_processed.csv). A SAV path also works.')
    parser.add_argument('--real-sav', type=str, default='', help='Optional fallback SAV path for the legacy extraction route.')
    parser.add_argument('--outdir', type=str, default='./results/sdv_baselines')
    parser.add_argument('--ctgan-epochs', type=int, default=500)
    parser.add_argument('--tvae-epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-rows', type=int, default=None, help='Default: same number of rows as the real dataset.')
    parser.add_argument('--disable-gpu', action='store_true')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    real_data_raw, real_source_path = load_real_table(args.real, args.real_sav)
    real_data, prep_meta = prepare_training_table(real_data_raw)
    n_rows = args.num_rows or len(real_data)

    metadata = make_mixed_metadata(
        data=real_data,
        categorical_cols=prep_meta['categorical_cols'],
        numerical_cols=prep_meta['numerical_cols'],
    )
    metadata_path = outdir / 'metadata.json'
    metadata.save_to_json(str(metadata_path), mode='overwrite')

    outputs = {
        'ctgan': (CTGANSynthesizer, args.ctgan_epochs, args.seed),
        'tvae': (TVAESynthesizer, args.tvae_epochs, args.seed + 1),
    }

    saved_files: Dict[str, str] = {}
    training_manifest: Dict[str, object] = {
        'real_data_path': real_source_path,
        'output_directory': str(outdir),
        'num_rows_sampled': int(n_rows),
        'n_real_rows_used': int(len(real_data)),
        'n_columns': int(real_data.shape[1]),
        'metadata_json': str(metadata_path),
        'preprocessing': {
            'age_kept_numerical': bool(prep_meta['age_col']),
            'categorical_cols': prep_meta['categorical_cols'],
            'numerical_cols': prep_meta['numerical_cols'],
            'categorical_values_canonicalized': True,
            'enforce_min_max_values': True,
            'enforce_rounding': True,
        },
        'files': {},
    }

    for name, (synthesizer_cls, epochs, seed) in outputs.items():
        print(f'Training {name.upper()} for {epochs} epochs on {len(real_data)} rows from {real_source_path} ...')
        sampled = fit_and_sample(
            synthesizer_cls=synthesizer_cls,
            metadata=metadata,
            data=real_data,
            n_rows=n_rows,
            epochs=epochs,
            seed=seed,
            enable_gpu=not args.disable_gpu,
        )
        sampled = postprocess_sample(
            sampled=sampled,
            real_reference=real_data,
            categorical_cols=prep_meta['categorical_cols'],
            numerical_cols=prep_meta['numerical_cols'],
        )
        csv_path = outdir / f'{name}.csv'
        sampled.to_csv(csv_path, index=False)
        saved_files[name] = str(csv_path)
        training_manifest['files'][name] = {
            'path': str(csv_path),
            'epochs': int(epochs),
            'seed': int(seed),
        }
        print(f'Saved {name.upper()} sample to {csv_path}')

    save_manifest(outdir / 'manifest.json', training_manifest)
    print(f'Saved manifest to {outdir / "manifest.json"}')


if __name__ == '__main__':
    main()
