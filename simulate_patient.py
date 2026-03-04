import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv()

DEFAULT_RUNS_DIR = './data/llm_generation_runs'
DEFAULT_QUESTIONNAIRE_FILE = './documents/questionnaire.json'
DEFAULT_ACTOR_MODEL = 'gemini-2.5-flash'
DEFAULT_CRITIC_MODEL = 'gemini-2.5-flash'

DISORDER_TO_ITEM_PREFIX = {
    'DEPRESSION': 'depression_it',
    'SEPARATION ANXIETY': 'separation_anxiety_it',
    'SPECIFIC PHOBIA': 'specific_phobia_it',
    'SOCIAL ANXIETY': 'social_anxiety_it',
    'PANIC': 'panic_it',
    'AGORAPHOBIA': 'agoraphobia_it',
    'GENERALIZED ANXIETY': 'generalized_anxiety_it',
}


@dataclass
class ValidationResult:
    hard_pass: bool
    errors: List[str]
    block_stats: Dict[str, Dict[str, float]]
    target_deviation_mean: float | None


client = genai.Client()


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


def _norm(x: str) -> str:
    return str(x).strip().lower().replace('-', '_').replace(' ', '_')


def parse_loose_json(text: str) -> dict:
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.startswith('json'):
            text = text[4:].strip()
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError('No JSON object found in model output.')
    return json.loads(text[start:end + 1])


def load_questionnaire(path: str) -> tuple[dict, List[str], Dict[str, int], Dict[str, str]]:
    questionnaire = json.load(open(path, 'r', encoding='utf-8'))
    item_keys: List[str] = []
    item_max: Dict[str, int] = {}
    item_label: Dict[str, str] = {}

    for disorder_name, block in questionnaire.items():
        max_value = 3 if 'depression' in disorder_name.lower() else 4
        for item in block['items']:
            key = item['key']
            item_keys.append(key)
            item_max[key] = max_value
            item_label[key] = item['label']

    return questionnaire, item_keys, item_max, item_label


def format_questionnaire_prefix(questionnaire: dict, workflow_mode: str) -> str:
    lines = [
        'You are completing a structured psychological questionnaire as the described person.',
        'Return only valid JSON.',
        '',
        'Required JSON schema:',
        '{',
        '  "scores": {"ITEM_KEY": INTEGER, ...}',
        '}',
        '',
        'Rules:',
        '- Include every questionnaire item exactly once under "scores".',
        '- Use integers only.',
        '- Depression items use 0-3.',
        '- Anxiety items use 0-4.',
    ]

    if workflow_mode in {'validated', 'full_agentic'}:
        lines.extend([
            '- When a clinical profile provides exact targets, try to match those targets as closely as possible.',
            '- Maintain coherence across items.',
        ])

    lines.extend(['', '=== QUESTIONNAIRE ITEMS ==='])
    for disorder_name, block in questionnaire.items():
        lines.append(f'[{disorder_name}]')
        for item in block['items']:
            lines.append(f"{item['key']}: {item['label']}")
        lines.append('')

    return '\n'.join(lines)


def build_actor_prompt(static_prefix: str, prompt_record: dict, repair_feedback: str | None = None, current_scores: dict | None = None) -> str:
    lines = [static_prefix, '', '=== PATIENT CONTEXT ===', prompt_record['dynamic_profile']]

    if repair_feedback:
        lines.extend([
            '',
            '=== REVISION REQUEST ===',
            'Revise the current JSON by changing as few item scores as possible while resolving the issues below.',
            repair_feedback,
        ])
    if current_scores:
        lines.extend([
            '',
            'Current candidate scores:',
            json.dumps({'scores': current_scores}, ensure_ascii=False),
        ])
    return '\n'.join(lines)


def build_critic_prompt(prompt_record: dict, item_labels: Dict[str, str], scores: dict, block_stats: dict) -> str:
    compact_labels = {k: item_labels[k] for k in scores if k in item_labels}
    payload = {
        'profile_mode': prompt_record['profile_mode'],
        'dynamic_profile': prompt_record['dynamic_profile'],
        'profile_metadata': prompt_record['profile_metadata'],
        'scores': scores,
        'item_labels': compact_labels,
        'block_stats': block_stats,
    }
    return (
        'You are a clinical consistency critic for synthetic questionnaire responses. '\
        
        'Assess whether the item pattern is semantically coherent and aligned with the provided profile. '\
        'Return JSON with this schema:\n'
        '{\n'
        '  "pass": true/false,\n'
        '  "major_issues": ["..."],\n'
        '  "minor_issues": ["..."],\n'
        '  "repair_guidance": ["..."]\n'
        '}\n\n'
        'Flag only clinically meaningful contradictions or clear profile mismatches. '\
        
        f'INPUT:\n{json.dumps(payload, ensure_ascii=False)}'
    )


def block_item_keys(item_keys: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in DISORDER_TO_ITEM_PREFIX}
    for disorder_key, prefix in DISORDER_TO_ITEM_PREFIX.items():
        out[disorder_key] = [k for k in item_keys if prefix in k.lower()]
    return out


def normalize_scores(raw_scores: dict, item_keys: List[str]) -> dict:
    out = {}
    for key in item_keys:
        if key not in raw_scores:
            continue
        value = raw_scores[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, str)):
            try:
                out[key] = int(float(value))
            except Exception:
                continue
    return out


def compute_block_stats(scores: dict, profile_targets: Dict[str, dict], grouped_keys: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for disorder_key, keys in grouped_keys.items():
        vals = [scores[k] for k in keys if k in scores]
        if not vals:
            continue
        spec = profile_targets.get(disorder_key)
        measurement = spec.get('measurement') if spec else ('sum' if disorder_key == 'DEPRESSION' else 'average')
        observed = float(sum(vals)) if measurement == 'sum' else float(sum(vals) / len(vals))
        stats[disorder_key] = {
            'measurement': measurement,
            'observed': observed,
            'n_items_present': len(vals),
        }
        if spec:
            stats[disorder_key]['target'] = float(spec['score_target'])
            stats[disorder_key]['allowed_min'] = float(spec['allowed_min'])
            stats[disorder_key]['allowed_max'] = float(spec['allowed_max'])
            stats[disorder_key]['target_tolerance'] = float(spec['target_tolerance'])
    return stats


def validate_scores(scores: dict, item_keys: List[str], item_max: Dict[str, int], profile_mode: str, profile_targets: Dict[str, dict], grouped_keys: Dict[str, List[str]]) -> ValidationResult:
    errors: List[str] = []

    missing = [k for k in item_keys if k not in scores]
    extra = [k for k in scores if k not in item_keys]
    if missing:
        errors.append(f'Missing item keys: {missing}')
    if extra:
        errors.append(f'Unexpected item keys: {extra}')

    for key in item_keys:
        if key not in scores:
            continue
        value = scores[key]
        if not isinstance(value, int):
            errors.append(f'{key} is not an integer: {value!r}')
            continue
        if not (0 <= value <= item_max[key]):
            errors.append(f'{key}={value} is outside allowed range 0-{item_max[key]}.')

    block_stats = compute_block_stats(scores, profile_targets, grouped_keys)
    deviations: List[float] = []

    if profile_mode == 'copula_profile':
        for disorder_key, spec in profile_targets.items():
            keys = grouped_keys.get(disorder_key, [])
            vals = [scores[k] for k in keys if k in scores]
            if len(vals) != len(keys):
                errors.append(f'{disorder_key}: expected {len(keys)} items, found {len(vals)}.')
                continue

            measurement = spec['measurement']
            observed = float(sum(vals)) if measurement == 'sum' else float(sum(vals) / len(vals))
            lo = float(spec['allowed_min'])
            hi = float(spec['allowed_max'])
            target = float(spec['score_target'])
            tol = float(spec['target_tolerance'])

            if not (lo <= observed <= hi):
                errors.append(f'{disorder_key}: observed {observed:.3f} is outside allowed range [{lo}, {hi}].')

            deviation = abs(observed - target)
            deviations.append(deviation)
            if deviation > tol + 1e-9:
                errors.append(f'{disorder_key}: observed {observed:.3f} differs from exact target {target:.3f} by {deviation:.3f} (> {tol:.3f}).')

    target_deviation_mean = statistics.mean(deviations) if deviations else None
    return ValidationResult(hard_pass=(len(errors) == 0), errors=errors, block_stats=block_stats, target_deviation_mean=target_deviation_mean)


def call_model_json(model_id: str, prompt: str) -> dict:
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type='application/json'),
    )
    text = response.text or '{}'
    return parse_loose_json(text)


def critic_review(model_id: str, prompt_record: dict, item_labels: Dict[str, str], scores: dict, block_stats: dict) -> dict:
    prompt = build_critic_prompt(prompt_record, item_labels, scores, block_stats)
    try:
        data = call_model_json(model_id, prompt)
    except Exception as exc:
        return {
            'pass': True,
            'major_issues': [],
            'minor_issues': [f'Critic failed: {exc}'],
            'repair_guidance': [],
        }

    return {
        'pass': bool(data.get('pass', False)),
        'major_issues': list(data.get('major_issues', []) or []),
        'minor_issues': list(data.get('minor_issues', []) or []),
        'repair_guidance': list(data.get('repair_guidance', []) or []),
    }


def make_repair_feedback(validation: ValidationResult | None, critic_result: dict | None) -> str:
    lines: List[str] = []
    if validation and validation.errors:
        lines.append('Deterministic validation issues:')
        lines.extend(f'- {e}' for e in validation.errors)
    if critic_result:
        majors = critic_result.get('major_issues', []) or []
        minors = critic_result.get('minor_issues', []) or []
        guidance = critic_result.get('repair_guidance', []) or []
        if majors:
            lines.append('Semantic major issues:')
            lines.extend(f'- {x}' for x in majors)
        if minors:
            lines.append('Semantic minor issues:')
            lines.extend(f'- {x}' for x in minors[:5])
        if guidance:
            lines.append('Repair guidance:')
            lines.extend(f'- {x}' for x in guidance)
    return '\n'.join(lines)


def accept_candidate(workflow_mode: str, validation: ValidationResult, critic_result: dict | None) -> bool:
    if workflow_mode == 'simple':
        return True
    if workflow_mode == 'validated':
        return validation.hard_pass
    if workflow_mode == 'full_agentic':
        return validation.hard_pass and bool(critic_result and critic_result.get('pass', False))
    raise ValueError(f'Unknown workflow mode: {workflow_mode}')


def generate_one_patient(
    prompt_record: dict,
    static_prefix: str,
    item_keys: List[str],
    item_max: Dict[str, int],
    item_labels: Dict[str, str],
    grouped_keys: Dict[str, List[str]],
    workflow_mode: str,
    actor_model: str,
    critic_model: str,
    max_retries: int,
) -> tuple[dict, dict]:
    profile_targets = prompt_record['profile_metadata'].get('targets', {})
    patient_id = prompt_record['patient_id']

    current_scores: dict | None = None
    repair_feedback: str | None = None

    first_validation: ValidationResult | None = None
    first_critic: dict | None = None

    final_validation: ValidationResult | None = None
    final_critic: dict | None = None

    accepted = False
    parse_failures = 0

    for attempt in range(1, max_retries + 2):
        actor_prompt = build_actor_prompt(
            static_prefix=static_prefix,
            prompt_record=prompt_record,
            repair_feedback=repair_feedback,
            current_scores=current_scores,
        )

        try:
            actor_data = call_model_json(actor_model, actor_prompt)
            scores = normalize_scores(actor_data.get('scores', {}), item_keys)
        except Exception as exc:
            parse_failures += 1
            current_scores = current_scores or {}
            repair_feedback = f'Model output could not be parsed as valid JSON: {exc}. Return JSON only.'
            if workflow_mode == 'simple':
                break
            continue

        validation = validate_scores(
            scores=scores,
            item_keys=item_keys,
            item_max=item_max,
            profile_mode=prompt_record['profile_mode'],
            profile_targets=profile_targets,
            grouped_keys=grouped_keys,
        )

        critic_result = None
        if workflow_mode == 'full_agentic' and validation.hard_pass:
            critic_result = critic_review(
                model_id=critic_model,
                prompt_record=prompt_record,
                item_labels=item_labels,
                scores=scores,
                block_stats=validation.block_stats,
            )

        if attempt == 1:
            first_validation = validation
            first_critic = critic_result

        current_scores = scores
        final_validation = validation
        final_critic = critic_result

        if accept_candidate(workflow_mode, validation, critic_result):
            accepted = True
            break

        if attempt >= max_retries + 1:
            break

        repair_feedback = make_repair_feedback(validation, critic_result)
        if not repair_feedback:
            repair_feedback = 'The previous JSON did not satisfy the requested constraints. Revise it carefully.'

    row = {'ID': patient_id}
    for key in item_keys:
        row[key] = current_scores.get(key) if current_scores else None

    log = {
        'patient_id': patient_id,
        'profile_mode': prompt_record['profile_mode'],
        'workflow_mode': workflow_mode,
        'accepted': accepted,
        'attempts_used': attempt,
        'parse_failures': parse_failures,
        'first_hard_pass': first_validation.hard_pass if first_validation else False,
        'final_hard_pass': final_validation.hard_pass if final_validation else False,
        'first_target_deviation_mean': first_validation.target_deviation_mean if first_validation else None,
        'final_target_deviation_mean': final_validation.target_deviation_mean if final_validation else None,
        'first_error_count': len(first_validation.errors) if first_validation else None,
        'final_error_count': len(final_validation.errors) if final_validation else None,
        'first_errors': first_validation.errors if first_validation else [],
        'final_errors': final_validation.errors if final_validation else [],
        'first_critic_pass': None if first_critic is None else bool(first_critic.get('pass', False)),
        'final_critic_pass': None if final_critic is None else bool(final_critic.get('pass', False)),
        'first_critic_major_issues': [] if first_critic is None else list(first_critic.get('major_issues', [])),
        'final_critic_major_issues': [] if final_critic is None else list(final_critic.get('major_issues', [])),
        'first_critic_minor_issues': [] if first_critic is None else list(first_critic.get('minor_issues', [])),
        'final_critic_minor_issues': [] if final_critic is None else list(final_critic.get('minor_issues', [])),
    }
    return row, log


def build_summary(log_df: pd.DataFrame, manifest: dict, output_md: str) -> None:
    n = len(log_df)
    accepted_rate = float(log_df['accepted'].mean()) if n else 0.0
    first_hard_pass_rate = float(log_df['first_hard_pass'].mean()) if n else 0.0
    final_hard_pass_rate = float(log_df['final_hard_pass'].mean()) if n else 0.0
    mean_attempts = float(log_df['attempts_used'].mean()) if n else 0.0

    target_dev_cols = log_df['final_target_deviation_mean'].dropna().tolist()
    mean_target_dev = float(sum(target_dev_cols) / len(target_dev_cols)) if target_dev_cols else None

    critic_mask = log_df['final_critic_pass'].notna()
    if critic_mask.any():
        critic_pass_rate = float(log_df.loc[critic_mask, 'final_critic_pass'].mean())
        major_issue_rate = float((log_df.loc[critic_mask, 'final_critic_major_issues'].apply(len) > 0).mean())
    else:
        critic_pass_rate = None
        major_issue_rate = None

    worst_examples = log_df.sort_values(['accepted', 'final_error_count', 'attempts_used'], ascending=[True, False, False]).head(10)

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('# LLM generation run summary\n\n')
        f.write('## Run manifest\n\n')
        f.write('```json\n')
        f.write(json.dumps(manifest, indent=2))
        f.write('\n```\n\n')
        f.write('## Aggregate metrics\n\n')
        f.write(f'- Number of patients: **{n}**\n')
        f.write(f'- Accepted final responses: **{accepted_rate:.3f}**\n')
        f.write(f'- First-attempt hard-pass rate: **{first_hard_pass_rate:.3f}**\n')
        f.write(f'- Final hard-pass rate: **{final_hard_pass_rate:.3f}**\n')
        f.write(f'- Mean attempts used: **{mean_attempts:.3f}**\n')
        if mean_target_dev is not None:
            f.write(f'- Mean final exact-target deviation: **{mean_target_dev:.3f}**\n')
        if critic_pass_rate is not None:
            f.write(f'- Final critic-pass rate: **{critic_pass_rate:.3f}**\n')
        if major_issue_rate is not None:
            f.write(f'- Final major-issue rate: **{major_issue_rate:.3f}**\n')
        f.write('\n## Hardest cases\n\n')
        try:
            f.write(worst_examples[['patient_id', 'accepted', 'attempts_used', 'final_hard_pass', 'final_error_count']].to_markdown(index=False))
        except Exception:
            f.write('```\n' + worst_examples[['patient_id', 'accepted', 'attempts_used', 'final_hard_pass', 'final_error_count']].to_string(index=False) + '\n```')
        f.write('\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run LLM questionnaire generation for one prompt bundle.')
    parser.add_argument('--run-name', required=True, help='Name of the run directory under ./data/llm_generation_runs/')
    parser.add_argument('--workflow-mode', required=True, choices=['simple', 'validated', 'full_agentic'])
    parser.add_argument('--runs-dir', default=DEFAULT_RUNS_DIR)
    parser.add_argument('--questionnaire-file', default=DEFAULT_QUESTIONNAIRE_FILE)
    parser.add_argument('--actor-model', default=DEFAULT_ACTOR_MODEL)
    parser.add_argument('--critic-model', default=DEFAULT_CRITIC_MODEL)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--sleep-seconds', type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.runs_dir) / args.run_name
    prompts_path = run_dir / 'prompt_bundle.jsonl'
    manifest_path = run_dir / 'prompt_manifest.json'
    if not prompts_path.exists():
        raise FileNotFoundError(f'Missing prompt bundle: {prompts_path}')
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing prompt manifest: {manifest_path}')

    workflow_dir = run_dir / f'workflow_{args.workflow_mode}'
    workflow_dir.mkdir(parents=True, exist_ok=True)

    responses_csv = workflow_dir / 'responses.csv'
    log_jsonl = workflow_dir / 'generation_log.jsonl'
    summary_md = workflow_dir / 'summary.md'
    run_manifest_path = workflow_dir / 'run_manifest.json'

    prompts = _read_jsonl(str(prompts_path))
    manifest = json.load(open(manifest_path, 'r', encoding='utf-8'))
    questionnaire, item_keys, item_max, item_labels = load_questionnaire(args.questionnaire_file)
    grouped_keys = block_item_keys(item_keys)
    static_prefix = format_questionnaire_prefix(questionnaire, args.workflow_mode)

    run_manifest = {
        'prompt_manifest': manifest,
        'workflow_mode': args.workflow_mode,
        'actor_model': args.actor_model,
        'critic_model': args.critic_model,
        'max_retries': args.max_retries,
        'questionnaire_file': args.questionnaire_file,
    }
    with open(run_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(run_manifest, f, indent=2)

    rows: List[dict] = []
    logs: List[dict] = []

    print(f'Starting LLM generation for {len(prompts)} patients.')
    print(f'Workflow mode: {args.workflow_mode}')
    print(f'Output directory: {workflow_dir}')

    for prompt_record in tqdm(prompts, desc='Generating patients', unit='patient'):
        row, log = generate_one_patient(
            prompt_record=prompt_record,
            static_prefix=static_prefix,
            item_keys=item_keys,
            item_max=item_max,
            item_labels=item_labels,
            grouped_keys=grouped_keys,
            workflow_mode=args.workflow_mode,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            max_retries=args.max_retries,
        )
        rows.append(row)
        logs.append(log)
        time.sleep(args.sleep_seconds)

    df = pd.DataFrame(rows)
    ordered_cols = ['ID'] + item_keys
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = None
    df = df[ordered_cols]
    df.to_csv(responses_csv, index=False)
    _write_jsonl(str(log_jsonl), logs)

    log_df = pd.DataFrame(logs)
    build_summary(log_df, run_manifest, str(summary_md))

    print(f'Wrote responses to {responses_csv}')
    print(f'Wrote generation log to {log_jsonl}')
    print(f'Wrote run summary to {summary_md}')


if __name__ == '__main__':
    main()
