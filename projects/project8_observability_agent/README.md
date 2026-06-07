# Project 8 — Fully Observed Agent

## What You Build

Instrument a ReAct research agent with the full production observability stack:
**structured JSON logs + Prometheus metrics + OpenTelemetry traces + cost tracking**.
Every LLM call, every tool execution, and every agent run is tracked and measurable.

## Production Skills Practised

| Skill | Guide Section |
|-------|--------------|
| structlog JSON structured logging | §9.2 |
| Prometheus Counter / Histogram / Gauge | §2.17, §9.3 |
| OpenTelemetry spans (traces) | §2.18, §9.3 |
| Per-run cost tracking with calc_cost() | §7, §6.5 |
| run_id threading through all log entries | §9.2 |
| /metrics endpoint for Prometheus scraping | §9.3 |

## Architecture

```
User Query
    │
    ▼
run_observed_agent(query)
    │ binds run_id to structlog context
    │ starts OTel root span "react_agent"
    │
    ├──► instrumented_chat()
    │       ├── starts OTel child span "llm_call"
    │       ├── records Prometheus latency histogram
    │       ├── increments llm_calls_total counter
    │       ├── increments llm_cost_usd_total counter
    │       └── emits JSON log: {run_id, latency_ms, tokens, status}
    │
    ├──► instrumented_tool_call()
    │       ├── starts OTel child span "tool_{name}"
    │       ├── increments tool_calls_total counter
    │       └── emits JSON log: {run_id, tool, result_len, status}
    │
    └── agent_steps_per_run histogram observed on completion
```

## Setup

```bash
pip install litellm python-dotenv pydantic structlog prometheus-client \
            opentelemetry-api opentelemetry-sdk
```

Optional (Jaeger for trace visualization):
```bash
docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one
```

## Usage

```bash
python starter.py "What are the main design patterns in multi-agent AI systems?"
python solution.py "Explain the CAP theorem with examples"

# View metrics in Prometheus text format:
curl http://localhost:9090/metrics  # (metrics server started by solution.py)
```

## What To Implement (5 TODOs)

1. **`setup_structlog()`** — configure structlog with JSON output + timestamp
2. **`setup_prometheus_metrics()`** — define Counter, Histogram, Gauge; start metrics server
3. **`instrumented_chat(messages, run_id, **kwargs)`** — wrap chat() with metrics + OTel span
4. **`instrumented_tool_call(tool_name, args, run_id)`** — wrap tool with OTel span + metrics
5. **`run_observed_agent(query)`** — full ReAct loop with run_id, root span, step histogram

## Key Insight

Without observability you are flying blind. These three tools answer different questions:

| Question | Tool |
|----------|------|
| "What happened step by step?" | structlog JSON logs |
| "How often is it failing? How slow is it?" | Prometheus metrics |
| "Why did THIS request take 45 seconds?" | OTel traces |

All three are necessary in production.
