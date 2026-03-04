# LLM generation run summary

## Run manifest

```json
{
  "prompt_manifest": {
    "profile_mode": "demographics_only",
    "n_patients": 562,
    "real_demographics_file": "./data/patient_severities.csv",
    "sampling": "empirical joint age-sex sampling with replacement",
    "seed": 42
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
- First-attempt hard-pass rate: **1.000**
- Final hard-pass rate: **1.000**
- Mean attempts used: **1.000**

## Hardest cases

| patient_id   | accepted   |   attempts_used | final_hard_pass   |   final_error_count |
|:-------------|:-----------|----------------:|:------------------|--------------------:|
| demog_0001   | True       |               1 | True              |                   0 |
| demog_0002   | True       |               1 | True              |                   0 |
| demog_0003   | True       |               1 | True              |                   0 |
| demog_0004   | True       |               1 | True              |                   0 |
| demog_0005   | True       |               1 | True              |                   0 |
| demog_0006   | True       |               1 | True              |                   0 |
| demog_0007   | True       |               1 | True              |                   0 |
| demog_0008   | True       |               1 | True              |                   0 |
| demog_0009   | True       |               1 | True              |                   0 |
| demog_0010   | True       |               1 | True              |                   0 |
