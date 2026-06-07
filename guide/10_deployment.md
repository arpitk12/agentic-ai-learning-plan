[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §9 Observability](guide/09_observability.md) | [§11 Exercises Index →](guide/11_exercises_index.md)

---

## 10. Deployment Playbook

### 10.1 Dockerizing the Agent

```dockerfile
# Dockerfile — multi-stage for small, secure images
FROM python:3.12-slim AS builder

# System dependencies for common ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ───────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
COPY . .

# Security: create and use non-root user
RUN useradd -m -u 1001 -s /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Metadata
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use exec form to handle SIGTERM correctly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--graceful-timeout", "30", "--timeout", "120"]
```

```yaml
# docker-compose.yml — local development
version: "3.9"
services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://agent:agent@postgres:5432/agentdb
      - MODEL=gemini/gemini-2.0-flash
      - API_KEY=${API_KEY}
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    volumes:
      - ./chroma_db:/app/chroma_db  # persist vector DB

  celery-worker:
    build: .
    command: celery -A celery_app worker --loglevel=info --concurrency=4
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://agent:agent@postgres:5432/agentdb
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: agent
      POSTGRES_DB: agentdb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

### 10.2 Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  labels:
    app: agent-api
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1         # add 1 new pod before removing old
      maxUnavailable: 0   # zero-downtime rolling update
  template:
    metadata:
      labels:
        app: agent-api
    spec:
      containers:
        - name: agent-api
          image: ghcr.io/yourorg/agent-api:v1.0.0
          ports:
            - containerPort: 8000
          
          # Resource limits — CRITICAL: prevents OOM kill cascade
          resources:
            requests:
              cpu: "500m"       # 0.5 CPU cores
              memory: "512Mi"
            limits:
              cpu: "2000m"      # 2 CPU cores max
              memory: "2Gi"     # 2GB RAM max
          
          # Liveness: restart pod if health check fails repeatedly
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 90   # wait for model loading
            periodSeconds: 30
            failureThreshold: 3
          
          # Readiness: remove from load balancer if not ready
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 2
          
          # Load secrets from K8s secrets (never hardcode)
          envFrom:
            - secretRef:
                name: agent-api-secrets
          env:
            - name: MODEL
              value: "gemini/gemini-2.0-flash"
            - name: REDIS_URL
              value: "redis://redis-service:6379/0"
          
          # Mount persistent storage for vector DB
          volumeMounts:
            - name: chroma-storage
              mountPath: /app/chroma_db
      
      volumes:
        - name: chroma-storage
          persistentVolumeClaim:
            claimName: chroma-pvc
      
      # Spread pods across nodes for HA
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values: [agent-api]
                topologyKey: kubernetes.io/hostname

---
# Horizontal Pod Autoscaler — scale based on CPU and requests
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: agent_runs_active   # custom Prometheus metric
        target:
          type: AverageValue
          averageValue: "10"
```

### 10.3 GitHub Actions CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # ── Quality Gate ──────────────────────────────────────────
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Lint with ruff
        run: ruff check . --output-format=github
      
      - name: Type check with mypy
        run: mypy . --ignore-missing-imports
      
      - name: Run tests
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY_TEST }}
          MODEL: "gemini/gemini-2.0-flash"
        run: pytest tests/ -v --tb=short --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  # ── Build & Push Image ────────────────────────────────────
  build:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Docker meta (tags)
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=sha,prefix=sha-
            type=raw,value=latest
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ── Deploy to Production ──────────────────────────────────
  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/agent-api \
            agent-api=ghcr.io/${{ github.repository }}:sha-${{ github.sha }}
          kubectl rollout status deployment/agent-api --timeout=300s
      
      - name: Smoke test
        run: |
          curl -f -X GET https://api.yourdomain.com/health
          curl -f -X POST https://api.yourdomain.com/agent/run \
            -H "X-API-Key: ${{ secrets.SMOKE_TEST_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d '{"query": "What is 2+2?"}'
      
      - name: Rollback on failure
        if: failure()
        run: kubectl rollout undo deployment/agent-api
```

### 10.4 Scaling Decision Guide

| Metric | Threshold | Immediate Action | Longer-term Fix |
|--------|-----------|-----------------|----------------|
| CPU > 70% for 5min | Sustained | HPA adds replicas automatically | Profile: is it LLM calls or CPU code? |
| P95 latency > 60s | Any | Add Celery workers | Investigate bottleneck with traces |
| Celery queue depth > 100 | Instant | Add Celery workers | Consider priority queues |
| LLM error rate > 5% | 2min | Alert, check API key limits | Add fallback provider via LiteLLM |
| LLM cost > $20/hr | Immediate | Check for cost attack | Review model routing thresholds |
| Redis memory > 80% | Daily | Set TTLs on cache keys | Add Redis cluster or eviction policy |
| Vector DB latency > 500ms | 5min | Check index exists | Consider Qdrant vs ChromaDB migration |
| Active agents > 100 | 5min | Check for hung agents | Review timeout settings |

---

## Quick Reference: Common Agent Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Missing `assistant_message()` | "Tool not found" error after tool call | Always call `messages.append(assistant_message(response))` |
| Not processing all tool calls | Agent loops, ignores some calls | Process ALL tool_calls before next LLM call |
| Context overflow | 400 error, token limit exceeded | Implement sliding window or summarization |
| Infinite loop | Agent never stops | Add `max_steps` limit, check `stop_reason` |
| JSON parse failure | 500 error after LLM returns invalid JSON | Add retry with error feedback in prompt |
| Tool timeout | Agent stuck | Add `timeout=30` to all tool calls |
| Cost runaway | Bills spike overnight | Per-user daily budget in CostTracker |
| Prompt injection | Agent takes unexpected actions | Add injection detection guardrail |
| Missing tool result | LLM confused after tool call | Check `tool_result_message(id, result)` is appended |
| Wrong model string | LiteLLM exception | Check `llm.py` `MODEL` env var |

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §9 Observability](guide/09_observability.md) | [§11 Exercises Index →](guide/11_exercises_index.md)
