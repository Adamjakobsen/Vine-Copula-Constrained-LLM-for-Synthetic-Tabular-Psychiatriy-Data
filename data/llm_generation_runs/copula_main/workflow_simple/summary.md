# LLM generation run summary

## Run manifest

```json
{
  "prompt_manifest": {
    "profile_mode": "copula_profile",
    "n_patients": 562,
    "profiles_file": "./data/synthetic_patient_profiles_vine_continuous.json",
    "kg_file": "./documents/knowledge_graph.json"
  },
  "workflow_mode": "simple",
  "actor_model": "gemini-2.5-flash",
  "critic_model": "gemini-2.5-flash",
  "max_retries": 2,
  "questionnaire_file": "./documents/questionnaire.json"
}
```

## Aggregate metrics

- Number of patients: **562**
- Accepted final responses: **1.000**
- First-attempt hard-pass rate: **0.991**
- Final hard-pass rate: **0.991**
- Mean attempts used: **1.000**
- Mean final exact-target deviation: **0.001**

## Hardest cases

| patient_id    | accepted   |   attempts_used | final_hard_pass   |   final_error_count |
|:--------------|:-----------|----------------:|:------------------|--------------------:|
| syn_vine_0156 | True       |               1 | False             |                   2 |
| syn_vine_0019 | True       |               1 | False             |                   1 |
| syn_vine_0039 | True       |               1 | False             |                   1 |
| syn_vine_0166 | True       |               1 | False             |                   1 |
| syn_vine_0490 | True       |               1 | False             |                   1 |
| syn_vine_0001 | True       |               1 | True              |                   0 |
| syn_vine_0002 | True       |               1 | True              |                   0 |
| syn_vine_0003 | True       |               1 | True              |                   0 |
| syn_vine_0004 | True       |               1 | True              |                   0 |
| syn_vine_0005 | True       |               1 | True              |                   0 |
