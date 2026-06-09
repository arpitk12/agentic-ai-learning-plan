# Project 42 — Prompt Version Control System

> **Stack**: LiteLLM · SQLite · Pydantic · FastAPI · Python-dotenv  
> **Guide**: [§14 LLMOps](../../guide/14_llmops.md) — §§3, 7  
> **Time**: ~6 hours | **Difficulty**: ⭐⭐⭐

## What You'll Build

A **production-grade prompt registry** — version control for prompts, the same way Git versions code. Every prompt change is a deployment, and you need the ability to compare versions, A/B test, canary-promote, and roll back in under 60 seconds.

```
                  PromptRegistry (SQLite)
                  ┌─────────────────────┐
                  │ name: "compliance"  │
                  │ v1 → archived       │
                  │ v2 → archived       │
  Developer ────► │ v3 → staging        │ ────► Eval gate
                  │ v4 → production 90% │
                  │ v4 → canary     10% │ ← new version being tested
                  └─────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          ABRouter    CanaryRouter  MetricsCollector
          (50/50      (10%→50%→     (per-version:
           hash-       100% auto)    latency, score,
           based)                    cost, calls)
              │            │            │
              └────────────┴────────────┘
                           │
                    MonitoredRouter
                    routes each query to
                    the right prompt version
```

## Expected Output

```
=== Prompt Registry Demo ===

Created v1: hash=a3b4c5d | stage=development
Created v2: hash=f8e2b1a | stage=development
Promoted v2 → staging
Eval gate: accuracy=0.91 ✅ (threshold=0.85)
Promoted v2 → production

--- Diff v1 → v2 ---
- You are a compliance assistant. Answer questions about GDPR.
+ You are a senior compliance attorney with 20 years of experience.
+ Always cite the specific article number when answering.
+ Keep answers under 150 words unless complexity requires more.

A/B Test: 1000 queries routed
  v1 (50%): avg quality=7.8, avg latency=920ms, cost=$3.12
  v2 (50%): avg quality=8.6, avg latency=1050ms, cost=$3.84
  Winner: v2 (quality +0.8 pts, cost +23%)

Canary deploy v3: 10% → metrics OK → 50% → OK → 100% ✅

Rollback v3 → v2: completed in 0.003s ✅

GET /prompts/compliance
  {"name": "compliance", "version": "v2", "stage": "production",
   "metrics": {"accuracy": 0.91, "quality": 8.6, "calls": 5832}}
```

## Milestones

### Milestone 1 — PromptVersion Model
Define `PromptVersion` Pydantic model with all fields:
`name`, `version`, `stage`, `text`, `model`, `temperature`, `max_tokens`,
`created_at`, `author`, `commit_sha`, `hash` (SHA-256 of text), `changelog`,
`metrics` (dict: accuracy, quality_avg, latency_p95_ms, cost_per_call_usd, total_calls).

### Milestone 2 — PromptRegistry
Implement `PromptRegistry` with SQLite backend:
- `create(name, text, model, temperature, author, changelog)` → version string
- `get(name, version="latest")` → PromptVersion
- `get_by_stage(name, stage)` → PromptVersion
- `list_versions(name)` → list[PromptVersion] ordered by version desc
- `promote(name, version, to_stage)` → demotes previous holder of that stage
- `rollback(name)` → promotes most recent archived version to production
- `update_metrics(name, version, metrics)` → merge into existing metrics dict
- `diff(name, v1, v2)` → unified diff string

### Milestone 3 — ABRouter
Implement `ABRouter` for deterministic 50/50 traffic split.
`route(user_id, name, v_a, v_b) → PromptVersion`
Uses `hashlib.md5(user_id.encode()).hexdigest()` to assign buckets deterministically.
Same user always gets the same variant. Tracks call counts per variant.

### Milestone 4 — CanaryRouter
Implement `CanaryRouter` for gradual rollout.
`route(user_id, name, canary_version, canary_pct)` → PromptVersion
`auto_promote(name, canary_version, check_fn, thresholds)` → bool
  - Starts at 10% → waits for `min_calls` (default 100) → runs `check_fn(metrics)`
  - If pass: promote to 50% → wait → 100% → done
  - If fail at any stage: rollback immediately

### Milestone 5 — MetricsCollector
Track per-version performance metrics with `MetricsCollector`.
`record(name, version, latency_ms, cost_usd, quality_score=None, error=False)`
`get_report(name)` → dict mapping version → aggregated metrics
`compare(name, v1, v2)` → side-by-side comparison dict with winner per metric

### Milestone 6 — MonitoredRouter
Combine registry + routing + metrics into one interface.
`MonitoredRouter.query(user_query, user_id, agent_fn)` → dict
  - Gets correct prompt from registry (respects A/B or canary if active)
  - Calls `agent_fn(prompt_text, user_query)` to get response
  - Records metrics via MetricsCollector
  - Updates version metrics in registry every 100 calls

### Milestone 7 — FastAPI API
Build these endpoints:
- `GET  /prompts/{name}` — current production prompt
- `GET  /prompts/{name}/versions` — all versions with metrics
- `POST /prompts/{name}` — create new version
- `POST /prompts/{name}/promote` — body: `{version, to_stage}`
- `POST /prompts/{name}/rollback` — rollback to previous production
- `GET  /prompts/{name}/diff?v1=v1&v2=v2` — unified diff
- `GET  /prompts/{name}/compare?v1=v1&v2=v2` — metric comparison
- `POST /prompts/{name}/canary` — body: `{version, pct}`
- `GET  /prompts/{name}/metrics` — metrics report for all versions

## Stack

```
pip install litellm fastapi uvicorn pydantic python-dotenv
```

## Project Structure

```
project42_prompt_registry/
├── README.md
└── starter/
    └── starter.py   ← 12 TODOs
```

## Key Concepts

| Concept | Implementation | Guide |
|---|---|---|
| Prompt as versioned artifact | SQLite registry | §14.3 |
| Stage promotion | `promote()` method | §14.7 |
| Hash-based deterministic routing | `ABRouter` | §14.7.3 |
| Gradual rollout | `CanaryRouter` | §14.7.3 |
| Per-version metrics | `MetricsCollector` | §14.4 |
| Rollback in < 1 second | `rollback()` method | §14.7 |
