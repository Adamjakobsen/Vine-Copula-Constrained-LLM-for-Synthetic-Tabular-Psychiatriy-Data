import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_PROFILES_FILE = './data/synthetic_patient_profiles_vine_continuous.json'
DEFAULT_REAL_DEMOGRAPHICS_FILE = './data/patient_severities.csv'
DEFAULT_KG_FILE = './documents/knowledge_graph.json'
DEFAULT_RUNS_DIR = './data/llm_generation_runs'

DISORDER_KEYS = [
    'DEPRESSION',
    'SEPARATION ANXIETY',
    'SPECIFIC PHOBIA',
    'SOCIAL ANXIETY',
    'PANIC',
    'AGORAPHOBIA',
    'GENERALIZED ANXIETY',
]


@dataclass
class TargetSpec:
    disorder_key: str
    tier_code: str
    severity_label: str
    score_target: float
    allowed_min: float
    allowed_max: float
    target_tolerance: float
    measurement: str


def _norm(x: str) -> str:
    return str(x).strip().lower().replace('-', '_').replace(' ', '_')


def _load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _find_first_col(df: pd.DataFrame, prefix_lower: str) -> str | None:
    cmap = {c.lower(): c for c in df.columns}
    return next((cmap[k] for k in cmap if k.startswith(prefix_lower)), None)


def allowed_range_from_tier(disorder_key: str, tier_code: str) -> Tuple[float, float, float, str]:
    tier = _norm(tier_code)
    depression_ranges = {
        'tier_0': (0, 4, 0.0, 'sum'),
        'tier_1': (5, 9, 0.0, 'sum'),
        'tier_2': (10, 14, 0.0, 'sum'),
        'tier_3': (15, 19, 0.0, 'sum'),
        'tier_4': (20, 27, 0.0, 'sum'),
    }
    anxiety_ranges = {
        'tier_0': (0.0, 0.499, 0.05, 'average'),
        'tier_1': (0.5, 1.499, 0.05, 'average'),
        'tier_2': (1.5, 2.499, 0.05, 'average'),
        'tier_3': (2.5, 3.499, 0.05, 'average'),
        'tier_4': (3.5, 4.0, 0.05, 'average'),
    }
    if _norm(disorder_key) == 'depression':
        return depression_ranges[tier]
    return anxiety_ranges[tier]


def extract_target_specs(profile: dict) -> Dict[str, TargetSpec]:
    specs: Dict[str, TargetSpec] = {}
    for disorder_key in DISORDER_KEYS:
        tier_key = f'{disorder_key}_TIER_CODE'
        score_key = f'{disorder_key}_SCORE_TARGET'
        if tier_key not in profile or score_key not in profile:
            continue

        tier_code = str(profile[tier_key])
        severity_label = str(profile.get(disorder_key, 'unspecified'))
        score_target = float(profile[score_key])
        lo, hi, tol, measurement = allowed_range_from_tier(disorder_key, tier_code)
        specs[disorder_key] = TargetSpec(
            disorder_key=disorder_key,
            tier_code=tier_code,
            severity_label=severity_label,
            score_target=score_target,
            allowed_min=lo,
            allowed_max=hi,
            target_tolerance=tol,
            measurement=measurement,
        )
    return specs


def build_anchor_text(kg: dict, disorder_key: str, severity_label: str) -> str:
    if disorder_key not in kg:
        return ''
    if _norm(severity_label) in {'none', 'tier_0'}:
        return ''

    kg_data = kg[disorder_key]
    anchors: List[str] = []
    for _, crits in kg_data.get('dsm5_criteria', {}).items():
        for crit in crits:
            questions = crit.get('clinical_probing_questions', [])
            if questions:
                anchors.append(questions[0])
    anchors = anchors[:5]
    if not anchors:
        return ''

    lines = ['Clinical anchors likely to be relevant at this severity:']
    lines.extend(f'- {a}' for a in anchors)
    return '\n'.join(lines)


def build_copula_profile_prompt(profile: dict, kg: dict) -> Tuple[str, Dict[str, dict]]:
    age = profile.get('AGE', 'an adult')
    sex = profile.get('SEX', 'person')
    patient_id = profile.get('patient_id', 'unknown')
    specs = extract_target_specs(profile)

    lines = [
        f'Patient ID: {patient_id}',
        f'Demographics: {age}-year-old {sex}.',
        '',
        'You are answering as this person. Use the assigned symptom profile faithfully.',
        'Use the exact target scores to guide intensity within each disorder block, not only the broad severity label.',
        'Aim to match the target closely while keeping the overall response pattern clinically coherent and varied.',
        '',
        '=== CLINICAL PROFILE ===',
    ]

    prompt_metadata: Dict[str, dict] = {}
    for disorder_key in DISORDER_KEYS:
        if disorder_key not in specs:
            continue
        spec = specs[disorder_key]
        full_name = kg.get(disorder_key, {}).get('full_name', disorder_key.title())
        anchor_text = build_anchor_text(kg, disorder_key, spec.severity_label)
        lines.extend([
            '',
            f'Condition: {full_name}',
            f'Assigned severity label: {spec.severity_label}',
            f'Canonical tier code: {spec.tier_code}',
            f'Target {spec.measurement}: {spec.score_target:.3f}',
            f'Allowed range: [{spec.allowed_min}, {spec.allowed_max}]',
            f'Target tolerance: ±{spec.target_tolerance}',
        ])
        if anchor_text:
            lines.append(anchor_text)

        prompt_metadata[disorder_key] = asdict(spec)

    return '\n'.join(lines), prompt_metadata


def build_demographics_only_prompt(age: int | float | str, sex: str, patient_id: str) -> Tuple[str, Dict[str, dict]]:
    lines = [
        f'Patient ID: {patient_id}',
        f'Demographics: {age}-year-old {sex}.',
        '',
        'No structured psychopathology profile is provided for this condition.',
        'Answer as a psychologically plausible person from this demographic context.',
        'Do not force all symptom blocks to the same intensity; vary responses naturally across domains.',
        'Keep the overall pattern coherent rather than random.',
    ]
    return '\n'.join(lines), {}


def sample_demographics(real_demographics_file: str, n: int, seed: int) -> List[dict]:
    df = pd.read_csv(real_demographics_file)
    age_col = _find_first_col(df, 'w1_age')
    sex_col = _find_first_col(df, 'w1_sex')
    if age_col is None or sex_col is None:
        raise ValueError('Could not locate age/sex columns in the real demographics file.')

    sample = df[[age_col, sex_col]].dropna().sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    records = []
    for i, row in sample.iterrows():
        raw_sex = row[sex_col]
        sex = 'Female' if str(raw_sex).strip() in {'2', '2.0', 'Female', 'female', 'F', 'f'} else 'Male'
        records.append({
            'patient_id': f'demog_{i+1:04d}',
            'AGE': int(round(float(row[age_col]))),
            'SEX': sex,
        })
    return records


def build_prompt_records(profile_mode: str, profiles_file: str, demographics_file: str, kg_file: str, n_patients: int | None, seed: int) -> Tuple[List[dict], dict]:
    kg = _load_json(kg_file)
    profile_mode = _norm(profile_mode)

    if profile_mode == 'copula_profile':
        profiles = _load_json(profiles_file)
        records = []
        for profile in profiles:
            prompt_text, prompt_meta = build_copula_profile_prompt(profile, kg)
            records.append({
                'patient_id': profile['patient_id'],
                'profile_mode': 'copula_profile',
                'dynamic_profile': prompt_text,
                'profile_metadata': {
                    'AGE': profile.get('AGE'),
                    'SEX': profile.get('SEX'),
                    'AGE_GROUP': profile.get('AGE_GROUP'),
                    'targets': prompt_meta,
                },
            })
        manifest = {
            'profile_mode': 'copula_profile',
            'n_patients': len(records),
            'profiles_file': profiles_file,
            'kg_file': kg_file,
        }
        return records, manifest

    if profile_mode == 'demographics_only':
        if n_patients is None:
            if os.path.exists(profiles_file):
                n_patients = len(_load_json(profiles_file))
            else:
                raise ValueError('For demographics_only mode, provide --n-patients if no profiles file exists.')

        sampled = sample_demographics(demographics_file, n=n_patients, seed=seed)
        records = []
        for row in sampled:
            prompt_text, prompt_meta = build_demographics_only_prompt(
                age=row['AGE'], sex=row['SEX'], patient_id=row['patient_id']
            )
            records.append({
                'patient_id': row['patient_id'],
                'profile_mode': 'demographics_only',
                'dynamic_profile': prompt_text,
                'profile_metadata': {
                    'AGE': row['AGE'],
                    'SEX': row['SEX'],
                    'AGE_GROUP': None,
                    'targets': prompt_meta,
                },
            })
        manifest = {
            'profile_mode': 'demographics_only',
            'n_patients': len(records),
            'real_demographics_file': demographics_file,
            'sampling': 'empirical joint age-sex sampling with replacement',
            'seed': seed,
        }
        return records, manifest

    raise ValueError(f'Unknown profile mode: {profile_mode}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build prompt bundles for LLM questionnaire generation.')
    parser.add_argument('--profile-mode', required=True, choices=['copula_profile', 'demographics_only'])
    parser.add_argument('--run-name', required=True, help='Name of the output run directory under ./data/llm_generation_runs/')
    parser.add_argument('--profiles-file', default=DEFAULT_PROFILES_FILE)
    parser.add_argument('--real-demographics-file', default=DEFAULT_REAL_DEMOGRAPHICS_FILE)
    parser.add_argument('--kg-file', default=DEFAULT_KG_FILE)
    parser.add_argument('--runs-dir', default=DEFAULT_RUNS_DIR)
    parser.add_argument('--n-patients', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_records, manifest = build_prompt_records(
        profile_mode=args.profile_mode,
        profiles_file=args.profiles_file,
        demographics_file=args.real_demographics_file,
        kg_file=args.kg_file,
        n_patients=args.n_patients,
        seed=args.seed,
    )

    prompts_path = run_dir / 'prompt_bundle.jsonl'
    manifest_path = run_dir / 'prompt_manifest.json'

    _write_jsonl(str(prompts_path), prompt_records)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'Wrote {len(prompt_records)} prompt records to {prompts_path}')
    print(f'Wrote prompt manifest to {manifest_path}')
    if prompt_records:
        print('\n--- Sample dynamic profile ---')
        print(prompt_records[0]['dynamic_profile'])


if __name__ == '__main__':
    main()
