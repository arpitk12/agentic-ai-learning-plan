# Week 12 — Deploy, Monitor & Optimize

## Topics
1. Docker + Kubernetes deployment for agent APIs
2. Cost optimization: model routing (small model first, escalate)
3. Production monitoring: alerting on cost, latency, errors
4. Incident playbook for agent failures

## Key Concepts

### Dockerfile for Agent API
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Model Routing Strategy
Route based on task complexity to minimize cost:
```python
def pick_model(query: str) -> str:
    complexity = estimate_complexity(query)  # 1-3
    if complexity == 1:
        return "claude-haiku-4-5-20251001"   # $0.00025/1K in
    elif complexity == 2:
        return "claude-sonnet-4-6"  # $0.003/1K in
    else:
        return "claude-opus-4-6"    # $0.015/1K in

def estimate_complexity(query: str) -> int:
    # Simple heuristics:
    # - Short factual questions → 1
    # - Multi-step reasoning → 2
    # - Complex code/analysis → 3
    word_count = len(query.split())
    has_code = any(kw in query.lower() for kw in ["code", "implement", "debug", "architect"])
    if word_count < 20 and not has_code:
        return 1
    elif word_count < 100:
        return 2
    return 3
```

### Monitoring Dashboard (Grafana)
Key panels to build:
- Requests/minute (last 1h)
- P50/P95/P99 latency
- Cost/hour (by model)
- Error rate (4xx/5xx)
- Token usage by model
- Active jobs in queue

### Incident Playbook
```
SEVERITY 1 — Agent returning errors > 10% of requests
  1. Check Anthropic status page: https://status.anthropic.com
  2. Check logs for specific error codes
  3. If API error: enable fallback model
  4. If bug: rollback to last green deployment
  5. Notify users via status page

SEVERITY 2 — Cost spike (> 2x normal)
  1. Check which model is being called
  2. Check if query routing is working correctly
  3. Check for runaway agent loops (max_steps not enforced)
  4. Enable emergency budget cap if needed
```

## Exercises
- `ex1_dockerfile.py` — Dockerize the agent API
- `ex2_model_router.py` — intelligent model selection
- `ex3_monitoring.py` — Prometheus metrics + Grafana dashboard
- `ex4_load_test.py` — locust load test your API

## Checklist
- [ ] Agent API running in Docker
- [ ] Model router reduces cost by 30%+ on test traffic
- [ ] Grafana dashboard showing key metrics
- [ ] Load test: 50 concurrent users, < 5s P95 latency
- [ ] Incident playbook documented and tested
