# Project 41 — LLMOps Monitoring Hub

> **Stack**: LiteLLM · Langfuse · Prometheus · SQLite · FastAPI · SciPy  
> **Guide**: [§14 LLMOps](../../guide/14_llmops.md) — §§6, 8, 10  
> **Time**: ~8 hours | **Difficulty**: ⭐⭐⭐⭐

## What You'll Build

A **production LLMOps monitoring hub** that wraps any LLM agent with:

- **Full trace capture** — every call logged with tokens, latency, cost, user ID
- **Embedding drift detection** — KS test on input distribution vs reference
- **Quality drift monitoring** — rolling LLM-judge score tracking
- **SLO tracking** — Prometheus metrics for latency/error/quality/cost
- **Alert manager** — Slack/webhook/email alerts on SLO breaches
- **FastAPI dashboard** — live metrics, drift status, alert history

```
User ──► FastAPI /query ──► MonitoredAgent ──► litellm ──► LLM API
              │                   │                  │
              │                   ▼                  │
              │             TraceLogger          TokenCost
              │             (SQLite store)       Calculator
              │                   │
              │       ┌───────────┴───────────┐
              │       ▼                       ▼
              │  EmbeddingDrift         QualityDrift
              │  Detector               Monitor
              │  (KS test)              (LLM judge, 5%)
              │       │                       │
              │       └───────────┬───────────┘
              │                   ▼
              │              SLOTracker
              │              (Prometheus)
              │                   │
              ▼                   ▼
        GET /metrics       AlertManager
        GET /dashboard     (Slack/webhook)
        GET /drift
        GET /alerts
```

## Expected Output

```
=== LLMOps Monitoring Hub ===

[12:00:01] Query: "What does Article 28 of GDPR require?"
  → Model: gpt-4o-mini | Latency: 834ms | Tokens: 312+48 | Cost: $0.0032
  → Quality sample: 8.2/10 | Trace: trace-a3b4c5

[12:00:05] Drift check:
  Embedding: OK (KS=0.041, p=0.82)
  Quality:   OK (avg=8.1/10, baseline=8.3/10, drop=0.2 pts)
  Topics:    OK (similarity=0.91)

[12:01:00] SLO status:
  P50 latency:  820ms  ✅ (SLO < 1500ms)
  P95 latency: 1840ms  ✅ (SLO < 5000ms)
  Error rate:   0.0%   ✅ (SLO < 0.5%)
  Quality:      8.1/10 ✅ (SLO ≥ 7.5)
  Cost/call:   $0.003  ✅ (SLO < $0.015)

[12:02:30] ⚠️  ALERT: Quality drift detected!
  Score dropped 0.8 pts (8.3 → 7.5/10) over last 50 samples
  Action: Slack alert sent to #llm-alerts
```

## Milestones

### Milestone 1 — Trace Logger
Build `TraceLogger`: captures every LLM call to SQLite.
Fields: `trace_id`, `timestamp`, `model`, `prompt_name`, `prompt_version`, `user_id`,
`tokens_in`, `tokens_out`, `latency_ms`, `cost_usd`, `error`, `quality_score`.
Query helpers: `get_recent(n)`, `get_by_user(user_id)`, `get_stats(hours)`.

### Milestone 2 — Cost Calculator
Build `CostCalculator`: maps model names → per-token prices. Returns `(cost_usd, tokens_in, tokens_out)`
for any litellm `ModelResponse`. Include gpt-4o-mini, gpt-4o, gpt-4.1, claude-haiku-4,
claude-sonnet-4, claude-opus-4.

### Milestone 3 — Embedding Drift Detector
Implement `EmbeddingDriftDetector` using the KS test on PCA projections (see Guide §14.6.3).
First 500 queries build the reference. After that, check every 100 queries.
Output: `{drift: bool, ks_stat: float, p_value: float, reason: str}`.

### Milestone 4 — Quality Drift Monitor
Implement `QualityDriftMonitor`: samples 5% of traffic with an LLM judge
(`gpt-4o-mini` grading response quality 0–10). Detects:
- Sudden drop > 0.5 pts below baseline
- Gradual 7-day trend declining > 0.3 pts/day
Rolling window of last 100 scored samples.

### Milestone 5 — SLO Tracker
Implement `SLOTracker` emitting Prometheus metrics:
- `llm_request_duration_seconds` (Histogram, buckets: 0.5, 1, 2, 5, 10)
- `llm_request_errors_total` (Counter, labels: error_type)
- `llm_quality_score` (Gauge, rolling avg)
- `llm_cost_per_call_usd` (Histogram)
- `llm_tokens_total` (Counter, labels: direction=in|out)
Expose `/metrics` endpoint for Prometheus scrape.

### Milestone 6 — Alert Manager
Build `AlertManager`: rule-based alert engine.
Rules (all configurable):
- P95 latency > 5s for last 100 calls → WARNING
- Error rate > 1% for last 1h → CRITICAL
- Quality score drop > 0.5 pts → WARNING
- Embedding drift detected → WARNING
- Daily cost > $10 → WARNING
Send to: Slack webhook (JSON POST) and/or email (smtplib).
Dedup: same alert doesn't fire twice within 30 minutes.

### Milestone 7 — Monitored Agent
Wrap all components into `MonitoredAgent.query(user_query, user_id)`.
The wrapper: logs trace, checks per-call cost limit, runs drift check every 100 calls,
updates SLO metrics, triggers quality sampling. Returns response + trace metadata.

### Milestone 8 — FastAPI Dashboard
Add endpoints:
- `GET /metrics` — Prometheus format
- `GET /dashboard` — JSON summary (last 24h stats, SLO status, drift status)
- `GET /drift` — current drift status for all detectors
- `GET /alerts` — last 50 alerts with timestamps
- `GET /traces` — last 100 traces (paginated)
- `POST /query` — the monitored agent endpoint

### Milestone 9 — End-to-End Demo
Send 200 simulated queries (50 from reference distribution, 150 with shifted topics).
Verify:
- Drift detector triggers at ~100-150 queries
- Alert manager fires Slack notification
- Prometheus metrics are scraped and visible
- Dashboard JSON shows accurate stats

## Stack

```
pip install litellm langfuse prometheus-client fastapi uvicorn \
            scipy scikit-learn numpy pydantic python-dotenv aiohttp
```

## Project Structure

```
project41_llmops_monitoring/
├── README.md
└── starter/
    └── starter.py   ← 14 TODOs
```

## Key Concepts

| Concept | Where | Guide |
|---|---|---|
| KS test for drift | `EmbeddingDriftDetector` | §14.6.3 |
| Prometheus metrics | `SLOTracker` | §14.8.4 |
| LLM-judge sampling | `QualityDriftMonitor` | §14.5 |
| Alert dedup | `AlertManager` | §14.8 |
| Cost attribution | `CostCalculator` | §14.12 |
| Trace storage | `TraceLogger` | §14.10 |
