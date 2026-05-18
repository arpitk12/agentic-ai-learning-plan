# Project 4 — Agent-as-a-Service API

## Brief
Wrap the Phase 3 code review agent into a full production FastAPI service.
This is your first "real" deployment-grade agentic system.

## Requirements
- [ ] FastAPI service with async endpoints
- [ ] SSE endpoint for real-time progress streaming
- [ ] Celery background worker for long-running jobs
- [ ] Per-user cost tracking (stored in Redis)
- [ ] Input sanitization (no prompt injection)
- [ ] OpenTelemetry traces visible in Jaeger
- [ ] `/health` and `/metrics` (Prometheus) endpoints
- [ ] Dockerized (docker-compose with API + worker + Redis + Jaeger)

## Setup
```bash
pip install fastapi uvicorn celery redis anthropic opentelemetry-sdk \
            opentelemetry-exporter-jaeger prometheus-client pydantic python-dotenv
docker-compose up -d   # starts Redis + Jaeger
python -m uvicorn main:app --reload
celery -A worker worker --loglevel=info
```

## Endpoints
```
POST /jobs              → submit review job, returns {job_id}
GET  /jobs/{id}         → poll status: pending|running|done|failed
GET  /jobs/{id}/stream  → SSE: real-time token stream
GET  /health            → {status: "ok", queue_depth: N}
GET  /metrics           → Prometheus metrics
```

## Architecture
```
Client → POST /jobs → Redis Queue → Celery Worker → Agent
                   ↓
            Job ID returned
                   ↓
Client → GET /jobs/{id}/stream → SSE (tokens from Redis pub/sub)
```

## Docker Compose
```yaml
services:
  api:     image: agent-api, ports: 8000
  worker:  image: agent-api, command: celery worker
  redis:   image: redis:7
  jaeger:  image: jaegertracing/all-in-one, ports: 16686, 14268
```

## Hints
- Use Redis pub/sub for streaming tokens from worker to API
- OpenTelemetry: instrument every LLM call as a span
- Cost tracking: `HINCRBY user:{user_id}:tokens input_tokens N`
- Rate limit: `INCR rate:{user_id}:{minute}` + EXPIRE 60
