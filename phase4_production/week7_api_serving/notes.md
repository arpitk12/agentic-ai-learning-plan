# Week 7 — Production API Serving: FastAPI, Celery & Redis

## What This Week Is About
A working agent on your laptop is not production. Production means: HTTP endpoints, authenticated API access, request queuing for long-running tasks, streaming responses over the web, and resilience to failures. This week covers FastAPI for serving, Celery + Redis for background task queues, and SSE for streaming.

---

## 1. FastAPI — The Production API Framework for Python Agents

**What it is**: A modern Python web framework for building APIs. Built on Starlette (async HTTP) and Pydantic (data validation). The fastest Python web framework, competing with Node.js in benchmarks.

**Why FastAPI for agents:**
- Native `async`/`await` support — handles concurrent agent requests without blocking
- Automatic OpenAPI docs generation (`/docs` endpoint)
- Pydantic request/response validation built-in
- Native SSE and WebSocket support for streaming agent responses

**Install**: `pip install fastapi uvicorn[standard]`

### Basic Agent API

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio, litellm, os

app = FastAPI(title="Agent API", version="1.0.0")

# Request/Response models (Pydantic validates automatically)
class AgentRequest(BaseModel):
    query: str
    session_id: str = "default"
    max_steps: int = 10

class AgentResponse(BaseModel):
    result: str
    session_id: str
    steps_taken: int
    cost_usd: float

# Simple API key auth
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    api_key: str = Depends(verify_api_key)
) -> AgentResponse:
    """Run the agent synchronously (blocks until complete)."""
    result, steps, cost = await execute_agent(request.query, request.max_steps)
    return AgentResponse(
        result=result,
        session_id=request.session_id,
        steps_taken=steps,
        cost_usd=cost
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model": os.getenv("MODEL", "unset")}
```

### Running with Uvicorn

```bash
# Development
uvicorn app:app --reload --port 8000

# Production
uvicorn app:app --workers 4 --port 8000 --host 0.0.0.0

# With Gunicorn (more robust process management)
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 2. Server-Sent Events (SSE) — Streaming Agent Responses

**What it is**: A web standard for one-way streaming from server to client. Perfect for streaming LLM tokens to users.

**Why SSE over WebSockets**: SSE is simpler (HTTP only, auto-reconnect) and sufficient for LLM streaming. WebSockets are bidirectional — use for real-time chat interfaces.

```python
from fastapi.responses import StreamingResponse
import json, asyncio

@app.post("/agent/stream")
async def stream_agent(request: AgentRequest) -> StreamingResponse:
    """Stream agent response token by token."""
    
    async def generate():
        messages = [{"role": "user", "content": request.query}]
        
        # Stream tokens as SSE events
        async for chunk in await litellm.acompletion(
            model=os.getenv("MODEL"),
            messages=messages,
            stream=True
        ):
            delta = chunk.choices[0].delta
            if delta.content:
                # SSE format: "data: {json}\n\n"
                event = {"type": "token", "content": delta.content}
                yield f"data: {json.dumps(event)}\n\n"
        
        # Send completion event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering
        }
    )

# Client-side JavaScript:
# const source = new EventSource('/agent/stream');
# source.onmessage = (e) => {
#     const data = JSON.parse(e.data);
#     if (data.type === 'token') output.textContent += data.content;
# };
```

---

## 3. Background Tasks with Celery + Redis

**The Problem**: Agent tasks can take 30-120 seconds. HTTP requests time out after 30s by default. You need a **task queue**.

**Celery**: A Python distributed task queue. You submit a task, Celery runs it in a background worker process, and you poll for the result.

**Redis**: In-memory data store. Celery uses it as both a **broker** (task queue) and **result backend** (stores task results).

**Install**: `pip install celery redis`

**Architecture**:
```
Client → POST /agent/submit → FastAPI → Redis (broker) → Celery Worker → runs agent
Client → GET /agent/result/{task_id} → FastAPI → Redis (backend) → return result
```

### Celery Setup

```python
# celery_app.py
from celery import Celery
import os

celery_app = Celery(
    "agent_tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=["tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,  # results expire after 1 hour
    worker_max_tasks_per_child=100,  # restart worker after 100 tasks (memory hygiene)
)
```

```python
# tasks.py
from celery_app import celery_app
from llm import chat, get_text
import time

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_agent_task(self, query: str, session_id: str) -> dict:
    """Long-running agent task executed by Celery worker."""
    try:
        self.update_state(state="STARTED", meta={"progress": 0, "status": "Initializing..."})
        
        # Run the agent (can take minutes)
        result = react_agent(query)
        
        return {
            "result": result,
            "session_id": session_id,
            "completed_at": time.time()
        }
    except Exception as exc:
        # Retry on failure
        raise self.retry(exc=exc)
```

### FastAPI + Celery Integration

```python
# api.py
from fastapi import FastAPI
from celery.result import AsyncResult
from tasks import run_agent_task

app = FastAPI()

@app.post("/agent/submit")
async def submit_task(request: AgentRequest) -> dict:
    """Submit task to Celery queue. Returns task_id immediately."""
    task = run_agent_task.delay(request.query, request.session_id)
    return {"task_id": task.id, "status": "queued"}

@app.get("/agent/result/{task_id}")
async def get_result(task_id: str) -> dict:
    """Poll for task result."""
    result = AsyncResult(task_id)
    
    if result.state == "PENDING":
        return {"status": "queued", "task_id": task_id}
    elif result.state == "STARTED":
        return {"status": "running", "progress": result.info.get("progress", 0)}
    elif result.state == "SUCCESS":
        return {"status": "done", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "error", "error": str(result.result)}
```

### Running Celery Workers

```bash
# Start Redis (Docker)
docker run -d -p 6379:6379 redis:alpine

# Start Celery worker
celery -A celery_app worker --loglevel=info --concurrency=4

# Monitor with Flower (web UI for Celery)
pip install flower
celery -A celery_app flower --port=5555
```

---

## 4. Rate Limiting & Throttling

Protect your API from abuse and cost overruns:

```python
from fastapi import Request
from collections import defaultdict
import time

# Simple in-memory rate limiter (use Redis for production/multi-process)
request_counts = defaultdict(list)

async def rate_limit(request: Request, max_requests: int = 10, window: int = 60):
    """Limit each IP to max_requests per window seconds."""
    client_ip = request.client.host
    now = time.time()
    
    # Clean old requests
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < window]
    
    if len(request_counts[client_ip]) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_requests} requests per {window}s.",
            headers={"Retry-After": str(window)}
        )
    
    request_counts[client_ip].append(now)

# Use as dependency
@app.post("/agent/run")
async def run_agent(request: AgentRequest, _=Depends(rate_limit)):
    ...
```

---

## 5. Request Validation & Error Handling

```python
from fastapi import Request
from fastapi.responses import JSONResponse

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the full error internally
    import traceback
    print(f"Unhandled error: {traceback.format_exc()}")
    
    # Return safe error to client (don't leak internals)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again."}
    )

# Pydantic validation (automatic)
class AgentRequest(BaseModel):
    query: str
    max_steps: int = Field(default=10, ge=1, le=50)  # 1-50 steps
    
    @validator("query")
    def query_not_empty(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters")
        if len(v) > 10000:
            raise ValueError("Query too long (max 10,000 chars)")
        return v.strip()
```

---

## 6. Production Configuration

```python
from pydantic_settings import BaseSettings  # pip install pydantic-settings

class Settings(BaseSettings):
    model: str = "gemini/gemini-2.0-flash"
    api_key: str
    redis_url: str = "redis://localhost:6379/0"
    max_concurrent_agents: int = 10
    cost_limit_usd_per_day: float = 10.0
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Tools & Libraries Used This Week — Deep Dive

### FastAPI — Why It's the Standard for Python AI APIs

**What makes FastAPI special vs Flask/Django?**

1. **True async**: FastAPI is built on Starlette (async HTTP) + ASGI. Under the hood, every endpoint can run concurrently with `await`. Flask (by default) is sync — one request blocks others.

2. **Automatic validation**: Pydantic models are first-class citizens. Define a `BaseModel` for your request body → FastAPI automatically validates, gives 422 errors with field-level details, and generates OpenAPI docs.

3. **Auto-generated docs**: Visit `/docs` for Swagger UI, `/redoc` for ReDoc — both fully interactive, always up-to-date, requiring zero extra work.

4. **Type-safe dependency injection**: `Depends()` lets you inject auth, DB connections, rate limiters cleanly into any endpoint.

```python
# FastAPI's dependency injection — the underappreciated feature
from fastapi import FastAPI, Depends, HTTPException, Header, Request
import time

app = FastAPI()

# Dependency 1: Authentication
async def verify_api_key(x_api_key: str = Header(...)) -> dict:
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(401, "Invalid API key")
    return {"user_id": "authenticated_user"}

# Dependency 2: Rate limiting
rate_state = {}  # In production: use Redis
async def rate_limit(request: Request) -> None:
    ip = request.client.host
    now = time.time()
    rate_state[ip] = [t for t in rate_state.get(ip, []) if now - t < 60]
    if len(rate_state[ip]) >= 10:
        raise HTTPException(429, "Too many requests", headers={"Retry-After": "60"})
    rate_state[ip].append(now)

# Dependency 3: Database connection
async def get_db():
    db = await create_db_connection()
    try:
        yield db  # provide the connection
    finally:
        await db.close()  # cleanup after request

# Combine all dependencies
@app.post("/agent/run")
async def run_agent(
    req: AgentRequest,
    user: dict = Depends(verify_api_key),      # must be authenticated
    _: None = Depends(rate_limit),             # must be within rate limit
    db = Depends(get_db),                      # gets a DB connection
) -> AgentResponse:
    # If we reach here: authenticated, not rate-limited, DB connected
    ...
```

---

### Uvicorn — The ASGI Server

**What it is**: An ASGI (Asynchronous Server Gateway Interface) web server implemented in Python. Uvicorn runs your FastAPI app and handles HTTP connections.

**Why ASGI over WSGI (used by Flask/Django)**: WSGI is synchronous — one thread handles one request. ASGI is async — one thread handles thousands of concurrent connections. Essential for agent APIs where requests take 5-30+ seconds.

```bash
# Development — single worker, auto-reload on code change
uvicorn app:app --reload --port 8000 --log-level debug

# Production — multiple workers for CPU parallelism
uvicorn app:app --workers 4 --port 8000 --host 0.0.0.0

# Recommended production: Gunicorn manages workers, Uvicorn handles each request
gunicorn app:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \         # 5 min timeout (for long agent runs via sync endpoint)
    --graceful-timeout 30   # wait 30s for requests to finish on shutdown
```

**Workers vs async**: Uvicorn's workers use multiprocessing for CPU-bound parallelism. Within each worker, asyncio handles I/O concurrency. So 4 workers × 100 concurrent async requests = 400 concurrent operations with only 4 CPU cores.

---

### Server-Sent Events (SSE) — The Streaming Protocol

**SSE vs WebSockets — when to use which**:

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server → Client only | Bidirectional |
| Protocol | HTTP (standard) | WebSocket (upgrade) |
| Browser support | Excellent, built-in `EventSource` | Good, needs `WebSocket` API |
| Auto-reconnect | ✅ Built in | ❌ Manual |
| Proxy support | Works through any HTTP proxy | May be blocked |
| **Use for LLM streaming** | ✅ Ideal | Overkill |
| **Use for real-time chat** | ❌ Wrong direction | ✅ |

**How SSE works at the byte level**:
```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Transfer-Encoding: chunked
[stream begins]
data: {"type": "token", "content": "Hello"}\n\n    ← each SSE event ends with \n\n
data: {"type": "token", "content": " world"}\n\n
data: {"type": "done"}\n\n
[connection closes]
```

The `\n\n` (double newline) is the SSE event delimiter — it tells the browser "event complete, fire onmessage."

```javascript
// Client-side SSE consumption
const source = new EventSource('/agent/stream', {
    headers: { 'X-API-Key': apiKey }  // headers need the fetch polyfill for EventSource
});

let fullText = '';
source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'token') {
        fullText += data.content;
        document.getElementById('output').textContent = fullText;
    } else if (data.type === 'done') {
        source.close();  // clean up
    } else if (data.type === 'error') {
        console.error(data.error);
        source.close();
    }
};

source.onerror = () => {
    console.error('SSE connection error');
    source.close();
};
```

---

### Celery — Task Queue Architecture

**The fundamental problem**: HTTP is request-response. A response must arrive within the timeout (30-60 seconds by default). Agent tasks that take 2-10 minutes CANNOT return via synchronous HTTP.

**Celery's solution**: Decouple task submission from task execution.

```
Client                    API Server                Redis           Celery Worker
  │                           │                       │                  │
  │ POST /agent/submit ────→  │                       │                  │
  │                           │ LPUSH task to Redis ─→│                  │
  │ ←── {task_id: "abc123"}   │                       │                  │
  │                           │                       │ BRPOP task ─────→│
  │                           │                       │                  │ (runs agent)
  │ GET /status/abc123 ─────→ │                       │                  │
  │                           │ GET result from Redis ─→                  │
  │ ←── {status: "running"}   │                       │                  │
  │                           │                       │←── save result ──│
  │ GET /status/abc123 ─────→ │                       │                  │
  │                           │ GET result from Redis ─→                  │
  │ ←── {status: "done", result: "..."}               │                  │
```

**Redis serves two roles**:
1. **Broker**: Where Celery stores the queue of pending tasks (LPUSH/BRPOP)
2. **Result backend**: Where Celery stores completed task results (GET/SET)

These can be separate Redis databases (`/0` for broker, `/1` for results) to avoid key collisions.

```python
# Celery task design — key patterns

@celery_app.task(
    bind=True,              # self = task instance (for self.update_state, self.retry)
    max_retries=3,          # retry up to 3 times on failure
    default_retry_delay=60, # wait 60s between retries
    soft_time_limit=300,    # raise SoftTimeLimitExceeded after 5 min (caught, cleanup)
    time_limit=360,         # hard kill after 6 min (uncaught)
    rate_limit="10/m",      # max 10 executions per minute across ALL workers
)
def run_agent_task(self, query: str, user_id: str) -> dict:
    try:
        self.update_state(state="STARTED", meta={"step": "planning", "progress": 10})
        plan = create_plan(query)
        
        self.update_state(state="STARTED", meta={"step": "executing", "progress": 50})
        result = execute_plan(plan)
        
        return {"result": result, "cost": get_total_cost()}
    
    except SoftTimeLimitExceeded:
        # Graceful cleanup on soft limit
        cleanup_partial_state(self.request.id)
        raise
    
    except ExternalAPIError as exc:
        # Retry on transient errors
        raise self.retry(exc=exc, countdown=30)  # retry in 30s
```

**Flower — Celery monitoring UI**:
```bash
# Run Flower alongside your worker
celery -A celery_app flower --port=5555 --basic_auth=admin:secret

# Visit http://localhost:5555 to see:
# - Active workers and their tasks
# - Task history with success/failure rates
# - Queue depths (how many tasks waiting)
# - Worker resource usage
```

---

### Redis — More Than Just a Cache

In production agent systems, Redis plays multiple roles:

```python
import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Role 1: Celery message broker (automatic — Celery handles this)
# Role 2: Celery result backend (automatic)

# Role 3: Application cache
r.setex("cache:query:sha256hash", 3600, result)  # 1 hour TTL
cached = r.get("cache:query:sha256hash")

# Role 4: Rate limiting (sliding window)
r.zadd(f"rate:{user_id}", {str(time.time()): time.time()})
r.zremrangebyscore(f"rate:{user_id}", 0, time.time() - 60)
request_count = r.zcard(f"rate:{user_id}")

# Role 5: Session storage (for multi-turn conversations)
import json
r.setex(f"session:{session_id}", 86400, json.dumps(messages))
messages = json.loads(r.get(f"session:{session_id}") or "[]")

# Role 6: Pub/Sub for streaming results
# Worker publishes tokens, API subscribes and streams to client
r.publish(f"stream:{task_id}", json.dumps({"token": "Hello"}))
pubsub = r.pubsub()
pubsub.subscribe(f"stream:{task_id}")
for message in pubsub.listen():
    if message["type"] == "message":
        yield message["data"]
```

---

## Common Pitfalls — Week 7

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Sync agent in async FastAPI endpoint | Event loop blocks, no concurrency | Use `await asyncio.to_thread(sync_agent, query)` or make agent async |
| SSE without `Cache-Control: no-cache` | Browser caches stream, stops updating | Always set `Cache-Control: no-cache` and `X-Accel-Buffering: no` |
| Celery `task_acks_late=False` | Lost tasks on worker crash | Set `task_acks_late=True` — only ack after completion |
| Redis without connection pooling | New connection per request, connection exhaustion | Use `redis.ConnectionPool(max_connections=20)` |
| No `time_limit` on Celery tasks | Worker stuck on infinite loop | Always set `time_limit` (hard) and `soft_time_limit` |
| Pydantic v1 `validator` in v2 project | Silent failures or wrong behavior | Use `@field_validator` in Pydantic v2 |
| Returning stack traces to client | Security leak, user sees internals | Global exception handler returns `{"error": "Internal error"}` only |
- `ex2_sse_streaming.py` — SSE streaming to a simple HTML client
- `ex3_celery_worker.py` — Celery task for long-running agent + polling endpoint
- `ex4_rate_limiter.py` — Redis-backed rate limiting per API key

## Checklist
- [ ] FastAPI endpoint that runs agent and returns result (POST /agent/run)
- [ ] SSE streaming endpoint with proper media_type and headers
- [ ] Celery + Redis queue: submit task → get task_id → poll for result
- [ ] Rate limiting: 10 requests/minute per IP or API key
- [ ] Global error handler that logs full trace, returns safe message to client
- [ ] `/health` endpoint with model name and Redis connectivity check
