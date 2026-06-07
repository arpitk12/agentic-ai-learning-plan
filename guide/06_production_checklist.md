[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §5 Vector Search](guide/05_vector_search.md) | [§7 Cost Optimization →](guide/07_cost_optimization.md)

---

## 6. Production Checklist — Complete Pre-Launch Verification

### 6.1 API Layer ✅

- [ ] **Authentication on every endpoint** — JWT or API key in `Authorization` or `X-API-Key` header. No endpoint is publicly accessible.
- [ ] **Rate limiting per user and per IP** — Use Redis-backed rate limiter (not in-memory, which breaks with multiple workers). Default: 10 req/min per user.
- [ ] **Request size limits** — `max_length=10000` on query fields. Prevents prompt injection with massive payloads.
- [ ] **Input validation with Pydantic** — All request bodies use Pydantic models with field validators. No raw dict access.
- [ ] **Global error handler** — Catches unhandled exceptions, logs full traceback internally, returns `{"error": "Internal server error"}` to client. Never expose stack traces.
- [ ] **`/health` endpoint returns 200** — Checks LLM availability, DB connection, Redis connection. Returns unhealthy if any dependency is down.
- [ ] **`/metrics` endpoint for Prometheus** — Returns Prometheus text format. All key metrics exposed.
- [ ] **CORS configured** — Allow only your frontend domains. Never use `allow_origins=["*"]` in production.
- [ ] **TLS termination** — HTTPS only. Terminate at load balancer or nginx. Redirect HTTP → HTTPS.
- [ ] **Request ID header** — Generate and log a unique `X-Request-ID` for every request. Enables distributed tracing.

### 6.2 Agent Safety ✅

- [ ] **Input guardrails** — Check for prompt injection patterns (`ignore previous instructions`, role-playing directives). Reject or sanitize before sending to LLM.
- [ ] **Output guardrails** — Scan LLM outputs for: raw API keys, private IP addresses, system prompt leakage. Block if found.
- [ ] **PII detection before logging** — Never log user-submitted content without PII scan. Use regex or a dedicated library (`presidio-analyzer`) to detect SSN, credit cards, emails.
- [ ] **Tool whitelist enforced** — Agent can only call tools explicitly listed in `ALLOWED_TOOLS`. No dynamic tool loading from user input.
- [ ] **Max steps limit on all agents** — Default `max_steps=15`. No agent runs indefinitely. Return partial result with explanation after limit.
- [ ] **Cost limit per user per day** — Track cumulative cost in Redis. Reject new requests when user exceeds daily limit. Send warning at 80%.
- [ ] **Timeout on all tool calls** — Every tool call wrapped in `asyncio.wait_for(timeout=30)`. Agent continues (not crashes) if tool times out.

### 6.3 Reliability ✅

- [ ] **Celery queue for requests > 30s** — Any agent that might take longer than 30s MUST go through Celery. Synchronous HTTP endpoints are for fast queries only.
- [ ] **Redis connection pooling** — `redis.ConnectionPool(max_connections=50)`. Do not create new Redis connection per request.
- [ ] **Database connection pooling** — Use SQLAlchemy with `pool_size=5, max_overflow=10` or `asyncpg` connection pool.
- [ ] **Retry with exponential backoff** — All LLM API calls wrapped in retry logic. Use `tenacity`:
  ```python
  from tenacity import retry, stop_after_attempt, wait_exponential
  @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
  def call_llm(...): ...
  ```
- [ ] **Circuit breaker for external APIs** — If web search fails 5 times in 60 seconds, open circuit (skip search, tell agent it's unavailable) for 120 seconds.
- [ ] **Graceful shutdown** — Handle SIGTERM: finish current requests, reject new ones, flush logs. `uvicorn --graceful-timeout 30`

### 6.4 Data Layer ✅

- [ ] **Vector DB indices created** — Don't rely on sequential scans. Create HNSW index before going live.
- [ ] **Payload indices for filtered queries** — If you filter by `user_id`, `category`, `date`, create payload indices in Qdrant or pgvector.
- [ ] **Database backups scheduled** — Daily backup of PostgreSQL + Qdrant. Tested restore procedure. Recovery time < 1 hour.
- [ ] **Migrations versioned** — Use Alembic for PostgreSQL schema migrations. Never ALTER TABLE in production without a migration script.
- [ ] **Separate embedding model from query model** — Document what embedding model was used to build the index. Store this in your DB metadata. Changing models requires re-ingestion.

### 6.5 Observability ✅

- [ ] **Structured JSON logging** — Every log line includes: `timestamp`, `run_id`, `user_id`, `level`, `event`, `duration_ms`.
- [ ] **Cost tracked per user, per model, per endpoint** — Granular cost data in your DB. Required for billing and optimization.
- [ ] **Prometheus metrics scraped** — `/metrics` endpoint registered in Prometheus config. All key counters and histograms defined.
- [ ] **Grafana dashboards configured** — At minimum: request rate, error rate, P95 latency, LLM cost/hour, active agents.
- [ ] **Error alerting** — PagerDuty or Slack notification when: error rate > 5% for 2min, cost > $5/hour, P95 latency > 60s.
- [ ] **Agent run audit log** — Every agent run logged: user_id, query (sanitized), steps, tools called, cost, duration, outcome.

### 6.6 Deployment ✅

- [ ] **Multi-stage Dockerfile** — Final image < 500MB. Use `python:3.12-slim` not `python:3.12`. No dev dependencies in production image.
- [ ] **Non-root user in container** — `USER appuser`. Never run as root.
- [ ] **Secrets via env vars only** — No secrets in code, Docker image, or git history. Use `.env` for local, K8s secrets for production.
- [ ] **Health probes in K8s** — Both `livenessProbe` and `readinessProbe` configured. Liveness restarts stuck pod. Readiness removes unhealthy pod from load balancer.
- [ ] **Resource limits set** — `resources.limits.cpu: "2", resources.limits.memory: "2Gi"`. Prevents one pod from starving others.
- [ ] **CI/CD pipeline** — Every push: lint → test → build image → push → deploy to staging → run smoke tests → gate on production.
- [ ] **Rollback procedure tested** — `kubectl rollout undo deployment/agent-api` works and restores service in < 2 minutes.

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §5 Vector Search](guide/05_vector_search.md) | [§7 Cost Optimization →](guide/07_cost_optimization.md)
