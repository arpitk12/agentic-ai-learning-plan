# Project 28 — A/B Testing + Model Management

> **Stack**: LiteLLM · SciPy · MLflow · SQLite · FastAPI  
> **Phase 7 — Advanced Production** | Priority: P1 🟠

---

## What You'll Build

A production A/B testing system for LLM agents: deterministic user assignment, shadow mode, metric tracking, statistical significance testing, and MLflow model registry integration.

---

## Milestones

### Milestone 1 — Hash-Based Assignment
Implement deterministic variant assignment using `MD5(experiment_id:user_id) % 100`. Verify 10% traffic goes to treatment across 1000 users (within ±2%).

### Milestone 2 — Experiment Tracker
Build a SQLite-backed metric tracker. Schema: one row per call, tracking variant, model, success, latency, cost, quality score. Query methods: get_stats() for control vs treatment summary.

### Milestone 3 — Shadow Mode
Run control + treatment models simultaneously in parallel (`asyncio.gather`). User always receives control response. Log shadow comparison (match/mismatch rate, latency delta).

### Milestone 4 — A/B Router
Route live traffic: 10% to treatment. Track every call. Include a quality evaluator that scores responses 0-1 (exact match for classifiers, LLM-as-judge for open-ended).

### Milestone 5 — Statistical Significance
Implement chi-square test for proportion difference. Add Bayesian A/B test using Beta distribution. Print full experiment report with winner declaration (require p < 0.05 AND n ≥ 100).

### Milestone 6 — MLflow Model Registry
Log experiment results to MLflow: parameters (model name, prompt version), metrics (accuracy, latency P50/P95, cost). Register winning model. Implement canary rollout: promote from Staging → 10% → 100%.

### Milestone 7 — FastAPI Dashboard
Expose experiment endpoints:
- `GET /experiments/{id}/report` — current stats + significance test
- `POST /experiments/{id}/promote` — promote treatment to control
- `GET /experiments` — list all active experiments

---

## Setup

```bash
pip install litellm scipy mlflow fastapi uvicorn pydantic python-dotenv
mlflow ui   # access at http://localhost:5000
```

---

## Expected Output

```
=== A/B Experiment: compliance-model-v2 ===

Assignment distribution (1000 users):
  Control:   906 (90.6%) | Treatment: 94 (9.4%) ✓ within ±2%

After 200 calls:
  | Metric      | Control      | Treatment    |
  |-------------|--------------|--------------|
  | Samples     | 181          | 19           |
  | Success %   | 78.5%        | 91.0%        |
  | Avg latency | 820ms        | 180ms        |
  | Avg cost    | $0.0006      | $0.0003      |
  | p-value     | 0.031 < 0.05 | → significant|
  | Winner      | TREATMENT ✅  |              |

P(treatment better): 97.3% — promote treatment to control? [y/N]
```
