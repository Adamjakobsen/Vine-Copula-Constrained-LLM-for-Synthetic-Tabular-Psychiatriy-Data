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
  "workflow_mode": "validated",
  "actor_model": "gemini-2.5-flash",
  "critic_model": "gemini-2.5-flash",
  "max_retries": 2,
  "questionnaire_file": "./documents/questionnaire.json"
}
```

## Aggregate metrics

- Number of patients: **562**
- Accepted final responses: **1.000**
- First-attempt hard-pass rate: **0.996**
- Final hard-pass rate: **1.000**
- Mean attempts used: **1.004**
- Mean final exact-target deviation: **0.000**

## Hardest cases

| patient_id    | accepted   |   attempts_used | final_hard_pass   |   final_error_count |
|:--------------|:-----------|----------------:|:------------------|--------------------:|
| syn_vine_0213 | True       |               2 | True              |                   0 |
| syn_vine_0482 | True       |               2 | True              |                   0 |
| syn_vine_0001 | True       |               1 | True              |                   0 |
| syn_vine_0002 | True       |               1 | True              |                   0 |
| syn_vine_0003 | True       |               1 | True              |                   0 |
| syn_vine_0004 | True       |               1 | True              |                   0 |
| syn_vine_0005 | True       |               1 | True              |                   0 |
| syn_vine_0006 | True       |               1 | True              |                   0 |
| syn_vine_0007 | True       |               1 | True              |                   0 |
| syn_vine_0008 | True       |               1 | True              |                   0 |
