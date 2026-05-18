# Week 7 — Serving Agents as APIs

## Topics
1. FastAPI async endpoints, SSE streaming responses
2. Background task queues: Celery + Redis, ARQ
3. Webhook callbacks for long-running agent tasks
4. Rate limiting, retry with exponential backoff

## Key Concepts

### Agent API Architecture
```
Client → POST /agent/run → {job_id}
Client → GET /agent/status/{job_id} → {status, result}
         OR
Client → GET /agent/stream/{job_id} → SSE stream of tokens
```

### SSE Streaming
```python
from fastapi.responses import StreamingResponse
import asyncio

async def stream_agent(query: str):
    async def event_stream():
        async for token in agent.astream(query):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### Background Queue Pattern
```
POST /run → enqueue job → return job_id
Worker picks up job → runs agent → stores result in Redis
GET /status/{id} → poll Redis → return result when ready
```

### Rate Limiting
Use slowapi or a Redis-based token bucket:
- Per-user: 10 requests/minute
- Global: 100 requests/minute
- Per-model: track token usage, enforce budget

## Exercises
- `ex1_fastapi_agent.py` — expose agent as FastAPI endpoint
- `ex2_sse_streaming.py` — stream tokens via SSE
- `ex3_celery_worker.py` — background job queue
- `ex4_rate_limiter.py` — per-user rate limiting

## Checklist
- [ ] Agent exposed as FastAPI POST endpoint
- [ ] SSE streaming implemented
- [ ] Celery background worker running
- [ ] Rate limiting enforced per user
