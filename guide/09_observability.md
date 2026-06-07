[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §8 Security](guide/08_security.md) | [§10 Deployment →](guide/10_deployment.md)

---

## 9. Observability Stack — See Everything, Miss Nothing

### 9.1 The Three Pillars of Observability

**Logs**: What happened. Structured events with timestamps and context.
**Metrics**: How much / how fast / how often. Numerical time-series data.
**Traces**: Why it's slow. End-to-end request paths across services.

All three are necessary. Logs tell you "the agent failed." Metrics tell you "it's failing 15% of the time." Traces tell you "it fails because the vector DB query takes 45 seconds."

### 9.2 Structured Logging with structlog

```python
# logging_config.py
import structlog
import logging
import sys
import json

def configure_logging(service_name: str = "agent-api", log_level: str = "INFO"):
    """Configure structured JSON logging for production."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,          # includes bound context
            structlog.processors.add_log_level,               # adds "level" field
            structlog.processors.TimeStamper(fmt="iso"),      # ISO timestamp
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),              # outputs as JSON
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    )

logger = structlog.get_logger()

# Usage patterns
def log_agent_start(run_id: str, user_id: str, query: str):
    logger.info(
        "agent_run_started",
        run_id=run_id,
        user_id=user_id,
        query_length=len(query),
        query_preview=query[:100],
    )

def log_tool_call(run_id: str, tool_name: str, args: dict, result_length: int, duration_ms: float):
    logger.info(
        "tool_called",
        run_id=run_id,
        tool=tool_name,
        result_length=result_length,
        duration_ms=round(duration_ms, 2),
    )

def log_llm_call(run_id: str, model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float, duration_ms: float):
    logger.info(
        "llm_call",
        run_id=run_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(cost_usd, 6),
        duration_ms=round(duration_ms, 2),
    )

def log_agent_complete(run_id: str, steps: int, total_cost: float, duration_ms: float, success: bool):
    logger.info(
        "agent_run_complete",
        run_id=run_id,
        steps=steps,
        total_cost_usd=round(total_cost, 6),
        duration_ms=round(duration_ms, 2),
        success=success,
    )

def log_error(run_id: str, error_type: str, error_msg: str, **kwargs):
    logger.error(
        "agent_error",
        run_id=run_id,
        error_type=error_type,
        error_message=error_msg[:500],
        **kwargs,
    )
```

### 9.3 Complete Metrics Setup — prometheus_client

```python
# metrics.py — define all metrics once, import everywhere
from prometheus_client import Counter, Histogram, Gauge, start_http_server, Summary
import time

# ── HTTP Metrics ──────────────────────────────────────────────
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status_code"]
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["path"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120]
)

# ── LLM Metrics ───────────────────────────────────────────────
llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["model", "status"]  # status: success, error, rate_limited
)
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total tokens consumed",
    ["model", "token_type"]  # token_type: prompt, completion
)
llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Total LLM cost in USD",
    ["model"]
)
llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "LLM API call duration",
    ["model"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60]
)

# ── Agent Metrics ─────────────────────────────────────────────
agent_runs_total = Counter(
    "agent_runs_total",
    "Total agent run attempts",
    ["status"]  # status: success, failure, timeout, cost_limit
)
agent_steps_per_run = Histogram(
    "agent_steps_per_run",
    "Number of steps (LLM calls) per agent run",
    buckets=[1, 2, 3, 5, 8, 13, 21, 34]
)
agent_runs_active = Gauge(
    "agent_runs_active",
    "Currently executing agent runs"
)
agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Total agent run duration",
    buckets=[1, 5, 15, 30, 60, 120, 300]
)

# ── Tool Metrics ──────────────────────────────────────────────
tool_calls_total = Counter(
    "tool_calls_total",
    "Tool invocations",
    ["tool_name", "status"]
)
tool_call_duration_seconds = Histogram(
    "tool_call_duration_seconds",
    "Tool execution time",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 30]
)

# ── RAG Metrics ───────────────────────────────────────────────
rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Vector search latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]
)
rag_chunk_scores = Histogram(
    "rag_chunk_similarity_scores",
    "Similarity scores of retrieved chunks",
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
)

# ── Safety Metrics ────────────────────────────────────────────
guardrail_blocks_total = Counter(
    "guardrail_blocks_total",
    "Requests blocked by safety guardrails",
    ["guardrail_type"]  # input_injection, output_pii, output_api_key, cost_limit
)

# ── Instrumentation Context Manager ───────────────────────────
class AgentRunContext:
    """Context manager to instrument a complete agent run."""
    
    def __init__(self, run_id: str, user_id: str):
        self.run_id = run_id
        self.user_id = user_id
        self.start_time = None
        self.steps = 0
        self.total_cost = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        agent_runs_active.inc()
        return self
    
    def record_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, cost: float, duration: float):
        self.steps += 1
        self.total_cost += cost
        llm_calls_total.labels(model=model, status="success").inc()
        llm_tokens_total.labels(model=model, token_type="prompt").inc(prompt_tokens)
        llm_tokens_total.labels(model=model, token_type="completion").inc(completion_tokens)
        llm_cost_usd_total.labels(model=model).inc(cost)
        llm_call_duration_seconds.labels(model=model).observe(duration)
    
    def record_tool_call(self, tool_name: str, status: str, duration: float):
        tool_calls_total.labels(tool_name=tool_name, status=status).inc()
        tool_call_duration_seconds.labels(tool_name=tool_name).observe(duration)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        agent_runs_active.dec()
        status = "failure" if exc_type else "success"
        agent_runs_total.labels(status=status).inc()
        agent_steps_per_run.observe(self.steps)
        agent_duration_seconds.observe(duration)
        return False  # don't suppress exceptions
```

### 9.4 Alerting Rules (Prometheus Alertmanager)

```yaml
# alerts/agent_alerts.yml
groups:
  - name: agent_api_alerts
    interval: 30s
    rules:

      # P1: Service is down
      - alert: AgentAPIDown
        expr: up{job="agent-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Agent API is down"
          description: "The Agent API has been unreachable for 1 minute."

      # P1: High error rate
      - alert: HighErrorRate
        expr: |
          rate(http_requests_total{status_code=~"5.."}[5m]) /
          rate(http_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 5% for 2 minutes"
          description: "Current error rate: {{ $value | humanizePercentage }}"

      # P2: Cost spike
      - alert: LLMCostSpike
        expr: increase(llm_cost_usd_total[1h]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM cost exceeded $10 in the last hour"
          description: "Cost this hour: ${{ $value }}"

      # P2: High latency
      - alert: HighP95Latency
        expr: |
          histogram_quantile(0.95, 
            rate(http_request_duration_seconds_bucket[10m])
          ) > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 response latency above 60 seconds"

      # P2: Agents stuck
      - alert: AgentsStuck
        expr: agent_runs_active > 50
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "50+ concurrent agents running for 10+ minutes"

      # P3: Safety guardrails firing frequently
      - alert: FrequentGuardrailBlocks
        expr: rate(guardrail_blocks_total[10m]) > 1
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "More than 1 guardrail block per minute — possible attack"
```

### 9.5 Log Analysis Queries

```bash
# Find all failed agent runs in the last hour
cat agent.log | jq -c '. | select(.event == "agent_run_complete" and .success == false)' | tail -50

# Average cost per agent run today
cat agent.log | jq -r '. | select(.event == "agent_run_complete") | .total_cost_usd' \
    | awk '{sum+=$1; n++} END {printf "Avg: $%.6f (%d runs)\n", sum/n, n}'

# Top 10 most expensive agent runs
cat agent.log | jq -c '. | select(.event == "agent_run_complete")' \
    | sort -t '"total_cost_usd":' -k2 -nr | head -10

# Most called tools (last 24h)
cat agent.log | jq -r '. | select(.event == "tool_called") | .tool' \
    | sort | uniq -c | sort -rn | head -10

# P95 latency per endpoint
cat agent.log | jq -r '. | select(.event == "request_complete") | "\(.path) \(.duration_ms)"' \
    | awk '{data[$1][NR]=$2} END {for(p in data){n=asort(data[p]); print p, data[p][int(n*0.95)]"ms P95"}}'
```

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §8 Security](guide/08_security.md) | [§10 Deployment →](guide/10_deployment.md)
