# Project 43 — Continuous Evaluation Scheduler

> **Stack**: LiteLLM · APScheduler · SQLite · FastAPI · aiohttp  
> **Guide**: [§14 LLMOps](../../guide/14_llmops.md) — §§5, 6  
> **Time**: ~6 hours | **Difficulty**: ⭐⭐⭐

## What You'll Build

A **production continuous eval system** that runs on a schedule, evaluates your agent against a golden dataset, detects regressions, and fires alerts — automatically, without any human trigger.

```
APScheduler (cron: every 6 hours)
    │
    ▼
GoldenDataset.load()          ← 50 fixed test cases with assertions
    │
    ▼
EvalRunner.run(agent_fn)      ← calls agent on each test case
    │
    ├── LLMJudge(query, response)   → score 0–10
    ├── AssertionChecker(response)  → pass/fail per assertion
    └── LatencyTimer                → ms per call
    │
    ▼
RegressionDetector.check(scores)  ← compares to baseline + 7-day trend
    │
    ├── No regression → store results to DB, update trend
    └── Regression!  → AlertChannel.send()
                           │
                     ┌─────┴─────┐
                     ▼           ▼
                 Slack         Email
                 webhook       (smtplib)

GET /eval/history     ← last 30 eval runs + scores
GET /eval/latest      ← most recent eval result
GET /eval/trend       ← score trend chart data
GET /eval/golden      ← view golden dataset
POST /eval/run        ← trigger manual eval run
POST /eval/baseline   ← set current score as baseline
```

## Expected Output

```
=== Continuous Eval Scheduler ===

[2026-06-09 06:00:01] Scheduled eval started (run #12)
  Dataset: compliance_golden_v1.jsonl (50 cases)
  Model: gpt-4o-mini | Prompt: compliance-agent v2

  Running 50 test cases...
  ████████████████████████████████████████ 50/50

  Results:
    Pass rate:    92.0% (46/50) ✅ (baseline: 91.0%)
    Quality avg:   8.3/10       ✅ (baseline: 8.1/10)
    Latency P50:   820ms
    Latency P95:  1840ms
    Total cost:   $0.18

  Failed cases:
    - compliance-023: missing citation (expected "Article 17")
    - compliance-041: answer too long (max 200 words)
    - complexity-007: said "I don't know" for an answerable question
    - edge-003: hallucinated Article number "Article 92" (doesn't exist)

  Regression check: PASS (no regression vs baseline)
  Stored run #12 to DB

[2026-06-09 12:00:01] Scheduled eval started (run #13)
  ...
  Pass rate: 79.0% ← DROP from 92.0%

  ⚠️ REGRESSION DETECTED
  Reason: Pass rate dropped 13% (92.0% → 79.0%)
  Likely cause: Model API update (check OpenAI release notes)
  Action: Slack alert sent to #llm-alerts
          Email sent to team-lead@company.com
```

## Milestones

### Milestone 1 — Golden Dataset
Define `GoldenDataset` with 20 test cases covering:
- 10 regulatory queries (GDPR articles) — assertions: must cite article, answer ≤ 200 words
- 5 edge cases (ambiguous, multi-article) — assertions: must not say "I don't know"
- 3 adversarial inputs (off-topic, injection) — assertions: must decline politely
- 2 complex multi-step queries — assertions: structured answer required

Each case: `id`, `input`, `category`, `priority`, `assertions` dict:
`{must_contain: [], must_not_contain: [], max_words: int, llm_judge_criterion: str}`

### Milestone 2 — LLM Judge
Implement `LLMJudge.score(query, response, criterion)` → float (0–10).
Prompt: "Rate the following response for {criterion}. Scale 1–10. Only output a number.\nQ: {query}\nA: {response}"
Includes retry logic (up to 3 attempts) and fallback to 5.0 if parse fails.

### Milestone 3 — EvalRunner
Implement `EvalRunner.run(agent_fn, dataset, judge)` → `EvalResult`.
For each test case:
- Time the agent call
- Check `must_contain` / `must_not_contain` / `max_words` assertions
- Score with LLM judge
Returns `EvalResult` with: `run_id`, `timestamp`, `pass_rate`, `quality_avg`,
`latency_p50_ms`, `latency_p95_ms`, `total_cost_usd`, `failed_cases` list,
per-category breakdown.

### Milestone 4 — RegressionDetector
Implement `RegressionDetector.check(result)` → `(is_regression, reason)`.
Detection rules (configurable):
- Absolute drop: pass_rate or quality drops > `abs_threshold` (default 0.05 / 0.5 pts)
- Trend: 3-run linear regression slope < -0.01/run (gradual degradation)
- New failure modes: 3+ new case IDs failing that were passing last run

### Milestone 5 — Results Store
Implement `ResultsStore` (SQLite): stores every eval run.
Schema: `run_id`, `timestamp`, `pass_rate`, `quality_avg`, `latency_p50`, `latency_p95`,
`total_cost`, `failed_cases` (JSON), `category_breakdown` (JSON), `is_regression`.
Methods: `save`, `get_latest`, `get_history(n=30)`, `get_baseline`, `set_baseline`.

### Milestone 6 — Alert Channels
Implement `AlertChannel` with two backends:
- `SlackChannel`: POST to SLACK_WEBHOOK_URL with formatted message
- `EmailChannel`: smtplib SMTP (reads SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL from env)
- `CompositeChannel`: sends to all configured channels

Alert message must include: run_id, pass_rate drop, failed cases list, suggested actions.

### Milestone 7 — EvalScheduler
Implement `EvalScheduler` using APScheduler's `BackgroundScheduler`.
`schedule(agent_fn, dataset, interval_hours)` → starts background cron job.
On each trigger: run eval → check regression → store → alert if regression.
`get_next_run_time()` → datetime.
`trigger_now()` → runs eval immediately in background thread.

### Milestone 8 — FastAPI API
Endpoints:
- `GET  /eval/latest` — most recent eval result
- `GET  /eval/history?n=30` — last N eval results
- `GET  /eval/trend` — `{timestamps, pass_rates, quality_avgs}` for chart
- `GET  /eval/golden` — golden dataset cases
- `POST /eval/run` — trigger manual eval run (async, returns run_id)
- `GET  /eval/run/{run_id}` — get specific run result
- `POST /eval/baseline` — set current latest result as baseline
- `GET  /eval/status` — scheduler status (next run, last run, is_running)

## Stack

```
pip install litellm apscheduler fastapi uvicorn pydantic python-dotenv aiohttp
```

## Project Structure

```
project43_continuous_eval/
├── README.md
└── starter/
    └── starter.py   ← 12 TODOs
```

## Key Concepts

| Concept | Implementation | Guide |
|---|---|---|
| Golden dataset design | `GoldenDataset` | §14.5.3 |
| LLM-as-judge in prod | `LLMJudge` | §14.5 |
| Regression detection | `RegressionDetector` | §14.5.4 |
| Scheduled background jobs | `APScheduler` | §14.5.2 |
| Alert routing | `CompositeChannel` | §14.6.5 |
| Trend analysis | Linear regression on scores | §14.6.1 |
