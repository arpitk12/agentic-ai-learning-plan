# Week 12 — Production Deployment: Docker, Kubernetes, Monitoring & Model Routing

## What This Week Is About
The final week: getting your agent to production and keeping it running. Docker containers, Kubernetes orchestration, Prometheus + Grafana monitoring, load testing with Locust, and intelligent model routing that balances cost vs capability.

---

## 1. Docker — Packaging Your Agent

**What it is**: A platform for packaging applications with all their dependencies into portable containers. Your agent runs identically on your laptop, CI, and production servers.

**Why Docker for agents:**
- Reproducible environments — no "works on my machine"
- Easy scaling — spin up 10 identical containers behind a load balancer
- Isolation — agent can't interfere with other services
- Rollback — deploy new version, roll back in seconds if broken

### Multi-Stage Dockerfile

```dockerfile
# Dockerfile — multi-stage build for minimal production image

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools (not needed in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Minimal production image
FROM python:3.12-slim AS production

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user (security best practice)
RUN useradd --system --no-create-home --shell /bin/false agent
USER agent

# Copy application code
COPY --chown=agent:agent . .

# Environment defaults (override with docker run -e or Kubernetes secrets)
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```bash
# Build and run
docker build -t my-agent:latest .
docker run -d \
    -p 8000:8000 \
    --env-file .env \
    --name agent \
    my-agent:latest

# Check health
docker logs agent
curl http://localhost:8000/health

# Push to registry
docker tag my-agent:latest ghcr.io/myorg/my-agent:latest
docker push ghcr.io/myorg/my-agent:latest
```

---

## 2. Docker Compose — Local Full-Stack

```yaml
# docker-compose.yml — run the full agent stack locally

version: "3.9"

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL=gemini/gemini-2.0-flash
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://agent:secret@postgres:5432/agentdb
    env_file: .env
    depends_on:
      redis: {condition: service_healthy}
      postgres: {condition: service_healthy}
    restart: unless-stopped

  celery-worker:
    build: .
    command: celery -A celery_app worker --loglevel=info --concurrency=4
    environment:
      - REDIS_URL=redis://redis:6379/0
    env_file: .env
    depends_on: [redis, postgres]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      retries: 3
    volumes:
      - redis-data:/data

  postgres:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: agentdb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent -d agentdb"]
      interval: 10s
      retries: 5
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./migrations/init.sql:/docker-entrypoint-initdb.d/init.sql

  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    depends_on: [prometheus]
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  redis-data:
  postgres-data:
  grafana-data:
```

---

## 3. Prometheus + Grafana Monitoring

**Prometheus**: Collects metrics from your agent API at `/metrics`. Stores time-series data.
**Grafana**: Visualizes Prometheus data in dashboards with alerts.

### Exposing Metrics from FastAPI

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

# Define metrics
REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total agent API requests",
    ["endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "agent_request_duration_seconds",
    "Agent request duration",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)
AGENT_COST = Counter(
    "agent_cost_usd_total",
    "Total LLM cost in USD",
    ["model"]
)
ACTIVE_RUNS = Gauge(
    "agent_active_runs",
    "Currently running agent tasks"
)
TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Tool calls made by agents",
    ["tool_name", "success"]
)

# Expose metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Middleware to track request metrics
@app.middleware("http")
async def track_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
    
    return response
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "agent-api"
    static_configs:
      - targets: ["agent-api:8000"]
    metrics_path: /metrics
```

### Key Grafana Dashboards to Build

1. **Request Rate**: `rate(agent_requests_total[5m])` — requests per second
2. **Error Rate**: `rate(agent_requests_total{status="500"}[5m])` 
3. **P50/P95/P99 Latency**: `histogram_quantile(0.95, agent_request_duration_seconds)`
4. **Daily Cost**: `increase(agent_cost_usd_total[24h])`
5. **Active Runs**: `agent_active_runs`
6. **Tool Usage**: `rate(agent_tool_calls_total[1h])`

---

## 4. Model Routing — Cost vs Capability Optimization

**The Problem**: GPT-4o costs 50x more than GPT-4o-mini. For simple queries (greetings, factual lookups), a cheap model is sufficient. Only route complex reasoning to expensive models.

```python
from llm import chat, get_text, MODEL

class ModelRouter:
    """Route queries to appropriate models based on complexity."""
    
    MODELS = {
        "fast": "gemini/gemini-2.0-flash",        # Cheap, fast — simple queries
        "standard": "openai/gpt-4o-mini",         # Balanced — most queries
        "powerful": "openai/gpt-4o",              # Expensive — complex reasoning
        "code": "anthropic/claude-3-5-sonnet",    # Best for code
    }
    
    def classify_query(self, query: str) -> str:
        """Classify query complexity using a cheap model."""
        classification = get_text(chat(
            messages=[{
                "role": "user",
                "content": f"""Classify this query complexity:
                
Query: "{query}"

Reply with ONLY one word:
- "simple" — greeting, single fact, yes/no question
- "standard" — explanation, comparison, writing task
- "complex" — multi-step reasoning, analysis, research
- "code" — programming, debugging, code generation"""
            }],
            system="Classify with one word only.",
            max_tokens=5
        ))
        
        category = classification.strip().lower()
        if category not in ["simple", "standard", "complex", "code"]:
            category = "standard"  # safe default
        return category
    
    def route(self, query: str) -> str:
        """Return the appropriate model name for this query."""
        category = self.classify_query(query)
        model = self.MODELS.get(category, self.MODELS["standard"])
        print(f"Routing '{query[:50]}...' → {model} (category: {category})")
        return model
    
    def run_with_routing(self, query: str, messages: list) -> str:
        """Run query with automatically selected model."""
        model = self.route(query)
        import litellm
        response = litellm.completion(model=model, messages=messages)
        return response.choices[0].message.content

router = ModelRouter()

# A/B testing: compare router vs always using powerful model
def benchmark_router(test_queries: list) -> dict:
    router_costs = []
    fixed_costs = []
    
    for query in test_queries:
        # Router cost
        model = router.route(query)
        # (calculate cost based on model pricing)
        
        # Fixed model cost (gpt-4o always)
        # (calculate gpt-4o cost)
    
    return {
        "router_avg_cost": sum(router_costs) / len(router_costs),
        "fixed_avg_cost": sum(fixed_costs) / len(fixed_costs),
        "savings_pct": (1 - sum(router_costs) / sum(fixed_costs)) * 100
    }
```

---

## 5. Load Testing with Locust

**Locust**: Python load testing framework. Simulates many concurrent users hitting your agent API.

```python
# locustfile.py
from locust import HttpUser, task, between
import random, json

SAMPLE_QUERIES = [
    "What is the capital of Japan?",
    "Explain quantum entanglement in simple terms",
    "Write a Python function to sort a list",
    "What are the key differences between REST and GraphQL?",
    "Summarize the main causes of World War I",
]

class AgentUser(HttpUser):
    wait_time = between(1, 5)  # wait 1-5 seconds between requests
    
    @task(3)  # weight 3 — most common
    def run_simple_query(self):
        query = random.choice(SAMPLE_QUERIES[:2])  # simple queries
        with self.client.post(
            "/agent/run",
            json={"query": query, "session_id": "load-test"},
            headers={"X-API-Key": "test-key"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "result" not in data:
                    response.failure("Missing 'result' in response")
            elif response.status_code == 429:
                response.success()  # rate limiting is expected behavior
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)  # weight 1 — less common
    def run_complex_query(self):
        query = random.choice(SAMPLE_QUERIES[2:])
        self.client.post(
            "/agent/run",
            json={"query": query, "session_id": "load-test"},
            headers={"X-API-Key": "test-key"},
            timeout=60  # complex queries take longer
        )

# Run: locust -f locustfile.py --host http://localhost:8000
# Web UI at: http://localhost:8089
```

---

## 6. Kubernetes Deployment (Production Scale)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  labels:
    app: agent-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  template:
    spec:
      containers:
      - name: agent-api
        image: ghcr.io/myorg/my-agent:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        env:
        - name: MODEL
          value: "gemini/gemini-2.0-flash"
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: gemini-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10

---
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
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 7. Production Incident Playbook

When your agent goes wrong in production:

```
SEVERITY 1 — Complete outage (API returning 5xx for >5 min)
1. Check /health endpoint — is the API up?
2. Check Grafana error rate dashboard
3. Check Redis/Postgres connectivity
4. Rollback to previous Docker image: kubectl rollout undo deployment/agent-api
5. Page on-call: PagerDuty/OpsGenie

SEVERITY 2 — High error rate or cost spike
1. Check structlog for error patterns (grep logs)
2. Check Prometheus: agent_cost_usd_total spike?
3. Check Celery queue depth (Flower UI)
4. Temporarily raise rate limits if legitimate traffic spike
5. If cost runaway: set MODEL to cheapest option immediately

SEVERITY 3 — Quality degradation (bad answers)
1. Run golden dataset evaluation: python eval/run_golden.py
2. Check if model was changed recently
3. Review RAGAS scores for recent runs
4. Check prompt injection in recent logs
```

---

## Tools & Libraries Used This Week

| Tool | Purpose | Install |
|------|---------|---------|
| **Docker** | Container packaging | Docker Desktop |
| **Docker Compose** | Multi-container local stack | Included with Docker |
| **Prometheus** | Metrics collection | Docker |
| **Grafana** | Metrics dashboards | Docker |
| **prometheus-client** | Expose Python metrics | `pip install prometheus-client` |
| **Locust** | Load testing | `pip install locust` |
| **Kubernetes** | Production orchestration | minikube (local) |

---

## Tools Deep Dive — Week 12

### Docker — Why Containers Changed Everything

**The classic problem**: "It works on my machine." Production runs Python 3.10 + specific versions of dependencies. Your laptop runs Python 3.12 + different versions. Different behavior, impossible to debug.

**Docker's solution**: Package your application AND its entire environment (Python version, all dependencies, OS libraries) into a single file (image) that runs identically everywhere.

```
Without Docker:                    With Docker:
Developer laptop: Python 3.12     Container: Python 3.12 (fixed)
CI server: Python 3.10            Container: Python 3.12 (fixed)
Production: Python 3.11           Container: Python 3.12 (fixed)
→ "Works on my machine" problem   → Identical everywhere
```

**The layer system** — why Docker builds are fast after the first time:
```dockerfile
# Each line creates a new layer
FROM python:3.12-slim            # Layer 1: base OS + Python (cached)
COPY requirements.txt .           # Layer 2: just requirements.txt
RUN pip install -r requirements.txt  # Layer 3: installed packages (CACHED if requirements unchanged)
COPY . .                          # Layer 4: your code (this changes often)
CMD ["uvicorn", "app:app"]        # Layer 5: startup command

# If you change app.py but not requirements.txt:
# Layers 1-3 are cached → rebuild takes 3 seconds, not 3 minutes
```

**Multi-stage builds** — why your production image should be tiny:
```dockerfile
# Stage 1: Build (has build tools, compilers, etc.)
FROM python:3.12 AS builder
RUN pip install --user -r requirements.txt   # installs to /root/.local

# Stage 2: Runtime (NO build tools, minimal OS)
FROM python:3.12-slim AS runtime
COPY --from=builder /root/.local /root/.local  # copy only installed packages
COPY . .

# Result: 
# Single-stage image: ~1.5GB (includes build tools)
# Multi-stage image: ~300MB (only runtime code)
# Smaller = faster pull, smaller attack surface, lower storage cost
```

---

### Kubernetes — When You Need It vs When You Don't

**You DON'T need K8s when**:
- Single server, < 10 req/sec
- One instance of your agent is enough
- You're not at production scale yet

**You DO need K8s when**:
- Need to run 3+ replicas for high availability
- Need automatic scaling (10x traffic spike → auto-add pods)
- Need rolling deployments with zero downtime
- Managing multiple services (api + worker + redis + postgres)

**Key K8s concepts for agents**:

```yaml
# Deployment: manages pods (running containers)
# - Ensures N replicas always running
# - Handles rolling updates (new version deployed without downtime)
# - Restarts crashed pods automatically

# Service: stable network endpoint for pods
# - Pods have random IPs that change when they restart
# - Service gives them a stable DNS name (e.g., "agent-api-service:8000")
# - Load balances traffic across all healthy pods

# HPA (HorizontalPodAutoscaler): automatic scaling
# - Monitors CPU, memory, or custom metrics
# - Automatically adds/removes pods based on load

# ConfigMap: non-secret configuration
# - Environment variables that aren't sensitive
# - Configuration files mounted into containers

# Secret: sensitive configuration
# - API keys, passwords, connection strings
# - Stored encrypted in etcd (K8s database)
# - Mounted as environment variables or files
```

```bash
# Essential kubectl commands for agent deployment

# Deploy/update
kubectl apply -f k8s/                    # apply all YAML files in directory
kubectl set image deployment/agent-api agent-api=image:v2  # rolling update

# Monitor
kubectl get pods -l app=agent-api        # list running pods
kubectl logs -f pod/agent-api-abc123    # stream logs from pod
kubectl describe pod agent-api-abc123   # debug failing pod

# Scale
kubectl scale deployment agent-api --replicas=5   # manual scale
kubectl get hpa                          # check autoscaler status

# Troubleshoot
kubectl exec -it agent-api-abc123 -- bash  # shell into running pod
kubectl port-forward pod/agent-api-abc123 8000:8000  # test pod directly
kubectl rollout undo deployment/agent-api  # rollback to previous version
```

---

### Prometheus — Pull-Based Metrics Collection

**Why pull-based (Prometheus scrapes you) vs push-based (you send to a server)**:

Pull-based advantages:
- Prometheus controls the scrape interval (15s by default)
- If your service dies, Prometheus knows (scrape fails = alert)
- Easier to scale — Prometheus just adds more scrape targets
- No client configuration needed to change metrics server

```python
# The complete prometheus_client pattern for agents
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, REGISTRY
)

# Histogram buckets — design these for your use case
# For agent latency (seconds): most complete in 1-60s
LATENCY_BUCKETS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]

agent_duration = Histogram(
    "agent_duration_seconds",
    "Agent run time",
    buckets=LATENCY_BUCKETS
)

# How to use in code:
with agent_duration.time():  # context manager times the block
    result = run_agent(query)

# Or manually:
start = time.time()
result = run_agent(query)
agent_duration.observe(time.time() - start)

# Expose metrics at /metrics for Prometheus to scrape
# (usually done at startup)
start_http_server(9090)  # separate port from your API

# Or with FastAPI:
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

### Locust — Load Testing Your Agent API

**Why load test**:
- Find your breaking point BEFORE production traffic does
- Identify which component fails first (API, LLM rate limit, DB)
- Measure real P95/P99 latency under realistic concurrent load

```python
# locustfile.py — simulate realistic agent usage
from locust import HttpUser, task, between, events
import json, random

class AgentUser(HttpUser):
    """Simulates a real user sending queries to the agent API."""
    
    # Wait 1-10 seconds between tasks (realistic user think time)
    wait_time = between(1, 10)
    
    # Test queries — use real examples from your use case
    queries = [
        "What is machine learning?",
        "Explain how neural networks work",
        "What are the best practices for Python error handling?",
        "Summarize the key concepts of RAG",
        "How does vector similarity search work?",
    ]
    
    def on_start(self):
        """Called once when a simulated user starts."""
        self.api_key = "test-api-key"
        self.session_id = f"load_test_{random.randint(1, 10000)}"
    
    @task(weight=70)  # 70% of requests are sync agent calls
    def call_agent_sync(self):
        query = random.choice(self.queries)
        with self.client.post(
            "/agent/run",
            json={"query": query, "session_id": self.session_id},
            headers={"X-API-Key": self.api_key},
            catch_response=True,
            timeout=120,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.success()  # rate limit is expected behavior
            else:
                response.failure(f"Got status {response.status_code}")
    
    @task(weight=20)  # 20% check health
    def check_health(self):
        self.client.get("/health")
    
    @task(weight=10)  # 10% stream responses
    def stream_agent(self):
        # SSE streaming load test
        with self.client.post(
            "/agent/stream",
            json={"query": random.choice(self.queries)},
            headers={"X-API-Key": self.api_key},
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                # Consume the stream
                for line in response.iter_lines():
                    pass
                response.success()

# Run: locust -f locustfile.py --host=http://localhost:8000
# Then visit http://localhost:8089 for web UI
```

**What to measure**:
- RPS (requests/second) at different concurrency levels
- P95 latency — should stay < 30s for sync, < 5s for SSE first token
- Error rate — should stay < 0.1%
- The "knee of the curve" — where latency starts growing fast (your capacity limit)

---

### Model Router — The Biggest Cost Lever

```python
# Decision tree for routing — order matters for efficiency
def classify_query_complexity(query: str) -> str:
    """Fast, deterministic classification with no LLM call needed."""
    query_lower = query.lower().strip()
    word_count = len(query_lower.split())
    
    # Simple: clearly trivial questions
    simple_patterns = [
        r"^(hi|hello|hey|thanks|thank you|bye|goodbye)\b",
        r"^what is \d+\s*[+\-*/]\s*\d+",  # basic arithmetic
        r"^(yes|no|ok|okay)\b",
        r"^translate .{1,30} to \w+$",     # short translation
    ]
    if word_count < 5 and any(re.match(p, query_lower) for p in simple_patterns):
        return "simple"
    
    # Complex: clearly requires reasoning
    complex_patterns = [
        r"\banalyze\b.*\bcompare\b",
        r"\bimplement\b.*\b(function|class|algorithm)\b",
        r"\bdebug\b",
        r"\boptimize\b.*\bperformance\b",
        r"\bdesign\b.*\b(architecture|system|database)\b",
        r"\bwrite\b.*\b(report|essay|article)\b",
        r"pros and cons",
        r"step.by.step",
    ]
    if any(re.search(p, query_lower) for p in complex_patterns):
        return "complex"
    
    # Standard: everything else
    return "standard"

MODELS = {
    "simple": "gemini/gemini-2.0-flash",        # $0.10/1M
    "standard": "openai/gpt-4o-mini",            # $0.30/1M  
    "complex": "anthropic/claude-3-5-sonnet",    # $6.00/1M
}

def routed_chat(messages: list, system: str = None) -> str:
    query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    complexity = classify_query_complexity(query)
    model = MODELS[complexity]
    
    # Track routing decisions for analysis
    logger.info("model_routed", complexity=complexity, model=model, query_preview=query[:50])
    
    return get_text(chat(messages, system=system, model=model))
```

---

## Common Pitfalls — Week 12

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Single-stage Dockerfile | 1.5GB+ image, slow pulls | Multi-stage: build + runtime |
| Running as root in container | Security vulnerability | `USER appuser` with non-root uid |
| K8s pods with no resource limits | One pod starves others (OOM kill cascade) | Always set `resources.requests` and `resources.limits` |
| No liveness probe in K8s | Stuck pods never get restarted | Add `livenessProbe` with httpGet to `/health` |
| Prometheus metrics without labels | Can't filter by model or endpoint | Add labels from the start: `Counter("...", "...", ["model", "endpoint"])` |
| Load testing with too few virtual users | Doesn't expose concurrency bugs | Start at 1 user, ramp to 100+ to find breaking point |
| Model router with complex LLM call | Routing costs as much as the query | Use regex/rules for routing — no LLM call needed |
- `ex2_docker_compose.yml` — full stack: agent + celery + redis + postgres + grafana
- `ex3_model_router.py` — route queries to 3 model tiers based on complexity
- `ex4_locust_test.py` — load test your agent, find the breaking point

## Checklist
- [ ] Multi-stage Dockerfile builds a <500MB image with non-root user
- [ ] `docker compose up` starts full stack in one command
- [ ] Prometheus metrics endpoint at `/metrics` with request count, duration, cost
- [ ] Grafana dashboard shows request rate, P95 latency, and daily cost
- [ ] Model router reduces average cost by >30% vs always using the most powerful model
- [ ] Locust load test identifies maximum RPS before p95 latency exceeds 30s
