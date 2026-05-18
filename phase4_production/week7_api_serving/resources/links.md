# Week 7 Resources — Serving Agents as APIs

## Official Docs
- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn: https://www.uvicorn.org/
- Server-Sent Events (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- Celery: https://docs.celeryq.dev/
- ARQ (async Redis queue): https://arq-docs.helpmanual.io/

## Key Patterns
- **SSE Streaming**: `StreamingResponse` with `text/event-stream` mime type
- **Background jobs**: POST returns job_id immediately, GET polls status
- **Webhooks**: agent POSTs result to callback URL when done
- **Rate limiting**: `slowapi` for FastAPI per-route limits

## Install
```
pip install fastapi uvicorn anthropic python-dotenv sse-starlette slowapi
pip install celery redis  # for background jobs
```

## Test Commands
```bash
# Start server
uvicorn ex1_fastapi_agent:app --reload --port 8000

# Test streaming
curl -N "http://localhost:8000/agent/stream?query=what+is+42+times+7"

# Test blocking endpoint
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query": "what is 100 factorial last 3 digits"}'
```
