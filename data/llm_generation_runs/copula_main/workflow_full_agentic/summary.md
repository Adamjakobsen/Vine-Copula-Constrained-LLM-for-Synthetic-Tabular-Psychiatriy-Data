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
  "workflow_mode": "full_agentic",
  "actor_model": "gemini-2.5-flash",
  "critic_model": "gemini-2.5-flash",
  "max_retries": 2,
  "questionnaire_file": "./documents/questionnaire.json"
}
```

## Aggregate metrics

- Number of patients: **562**
- Accepted final responses: **0.957**
- First-attempt hard-pass rate: **0.995**
- Final hard-pass rate: **0.986**
- Mean attempts used: **1.315**
- Mean final exact-target deviation: **0.001**
- Final critic-pass rate: **0.971**
- Final major-issue rate: **0.025**

## Hardest cases

| patient_id    | accepted   |   attempts_used | final_hard_pass   |   final_error_count |
|:--------------|:-----------|----------------:|:------------------|--------------------:|
| syn_vine_0096 | False      |               3 | False             |                   3 |
| syn_vine_0464 | False      |               3 | False             |                   2 |
| syn_vine_0496 | False      |               3 | False             |                   2 |
| syn_vine_0153 | False      |               3 | False             |                   1 |
| syn_vine_0212 | False      |               3 | False             |                   1 |
| syn_vine_0238 | False      |               3 | False             |                   1 |
| syn_vine_0263 | False      |               3 | False             |                   1 |
| syn_vine_0476 | False      |               3 | False             |                   1 |
| syn_vine_0038 | False      |               3 | True              |                   0 |
| syn_vine_0063 | False      |               3 | True              |                   0 |
