"""
Project 4 Starter — Agent-as-a-Service API

Build a production-grade FastAPI service that wraps the code-review agent:
  1. POST /jobs         — accepts a PR URL, queues the review, returns job_id immediately
  2. GET  /jobs/{id}    — poll job status and metadata
  3. GET  /jobs/{id}/stream — stream LLM output token-by-token as Server-Sent Events (SSE)
  4. GET  /health       — liveness check + active job count
  5. GET  /users/{id}/cost — per-user cumulative LLM spend

Key concepts to practise:
  - Background coroutines with asyncio.create_task()
  - Streaming responses with AsyncGenerator + StreamingResponse
  - SSE protocol: each event is `data: <payload>\\n\\n`
  - Input sanitization against prompt-injection attacks
  - Per-user cost tracking with COST_LEDGER

Run:
    pip install fastapi uvicorn litellm
    uvicorn starter:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/jobs \\
         -H "Content-Type: application/json" \\
         -d '{"pr_url": "https://github.com/owner/repo/pull/1", "user_id": "alice"}'

    curl -N http://localhost:8000/jobs/<job_id>/stream
    curl    http://localhost:8000/jobs/<job_id>
    curl    http://localhost:8000/users/alice/cost

What you need to implement (TODOs 1-5):
  1. sanitize_input()          — reject injection patterns, cap length to 2 000 chars
  2. submit_job() POST /jobs   — create job dict, fire background task, return JobResponse
  3. get_job()   GET /jobs/{id} — look up JOBS dict, raise 404 if missing
  4. stream_job() GET /jobs/{id}/stream — SSE generator that tails job["events"]
  5. run_agent_with_stream()   — background coroutine: LiteLLM stream → emit tokens
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import asyncio
import json
import uuid
import time
from enum import Enum
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import calc_cost, MODEL

load_dotenv()

app = FastAPI(title="Agent API", version="1.0.0")


# ── In-Memory State (replace with Redis in production) ────────────────────────

JOBS: dict[str, dict]         = {}   # job_id → job dict
COST_LEDGER: dict[str, float] = {}   # user_id → cumulative USD spend


# ── Data Models ────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class JobRequest(BaseModel):
    pr_url:     str
    user_id:    str   = "anonymous"
    budget_usd: float = 0.10


class JobResponse(BaseModel):
    job_id:     str
    status:     JobStatus
    stream_url: str


# ── Input Sanitization ─────────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "you are now",
    "disregard your",
    "forget your instructions",
    "system prompt",
]


def sanitize_input(text: str) -> str:
    """
    Reject prompt-injection attempts and cap input length.

    TODO 1:
      a. Lowercase text and check whether any string in INJECTION_PATTERNS
         is a substring of it.
      b. If a pattern is found, raise:
             HTTPException(status_code=400,
                           detail=f"Potential prompt injection detected: '{pattern}'")
      c. Return text[:2000] to cap the input length.

    Example:
        lower = text.lower()
        for pattern in INJECTION_PATTERNS:
            if pattern in lower:
                raise HTTPException(status_code=400, detail=...)
        return text[:2000]
    """
    # TODO 1: implement injection guard + length cap
    raise NotImplementedError("sanitize_input() not implemented yet")


# ── Cost Tracking ──────────────────────────────────────────────────────────────

def track_cost(user_id: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for this LLM call and accumulate it in COST_LEDGER."""
    cost = calc_cost(MODEL, input_tokens, output_tokens)
    COST_LEDGER[user_id] = COST_LEDGER.get(user_id, 0) + cost
    return cost


# ── Background Agent ───────────────────────────────────────────────────────────

async def run_agent_with_stream(job_id: str, pr_url: str, user_id: str, budget: float):
    """
    Background coroutine: runs the AI review and appends SSE events to JOBS[job_id]["events"].

    The SSE stream_job() endpoint tails this list in real time.

    Event JSON shape:  {"type": "<event_type>", "content": "<text>", "job_id": "<id>"}

    Event types used:
      "status"  — progress messages ("Starting...", "Fetching PR...", etc.)
      "token"   — individual streaming LLM token deltas
      "cost"    — final cost string  e.g. "$0.002341"
      "warning" — budget exceeded warning
      "done"    — full final text (marks completion)
      "error"   — exception message (marks failure)

    TODO 5 — implement this coroutine:
      a. Set JOBS[job_id]["status"] = JobStatus.RUNNING.
      b. Get a reference to the events list:  events = JOBS[job_id]["events"]
      c. Define a local helper:
             def emit(event_type, content):
                 events.append(json.dumps({"type": event_type,
                                           "content": content,
                                           "job_id": job_id}))
      d. emit("status", "Starting code review...")
         emit("status", f"Fetching PR: {pr_url}")
      e. Build a prompt asking for a structured review (security / performance /
         style / testing) of the given PR URL.
      f. Call LiteLLM with streaming:
             from litellm import acompletion
             response = await acompletion(model=MODEL, messages=[...],
                                         max_tokens=1500, stream=True)
         Iterate with `async for chunk in response:` and for each chunk:
           - extract delta text: chunk.choices[0].delta.content or ""
           - if non-empty: emit("token", delta) and accumulate full_text
           - capture token counts from chunk.usage if present
      g. After the loop: cost = track_cost(user_id, input_tokens, output_tokens)
         If cost > budget: emit("warning", f"Budget exceeded: ${cost:.4f} > ${budget:.4f}")
         emit("cost", f"${cost:.6f}")
         emit("done", full_text)
         Update JOBS[job_id] with status=DONE, result=full_text, cost_usd=cost.
      h. Wrap a–g in try/except Exception as e:
             emit("error", str(e))
             JOBS[job_id]["status"] = JobStatus.FAILED
    """
    JOBS[job_id]["status"] = JobStatus.RUNNING
    events: list[str] = JOBS[job_id]["events"]

    def emit(event_type: str, content: str):
        events.append(json.dumps({"type": event_type, "content": content, "job_id": job_id}))

    try:
        emit("status", "Starting code review...")
        emit("status", f"Fetching PR: {pr_url}")
        # TODO 5: call LiteLLM with stream=True and emit tokens here
        emit("done", "Not implemented yet")
        JOBS[job_id]["status"] = JobStatus.DONE
    except Exception as e:
        emit("error", str(e))
        JOBS[job_id]["status"] = JobStatus.FAILED


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.post("/jobs", response_model=JobResponse)
async def submit_job(request: JobRequest):
    """
    Accept a new review job and start it in the background.

    TODO 2:
      a. Sanitize the URL: pr_url = sanitize_input(request.pr_url)
         (sanitize_input raises HTTPException on injection — just let it propagate)
      b. Generate a unique ID: job_id = str(uuid.uuid4())
      c. Store the job in JOBS[job_id] as a dict with these keys:
             "id"         → job_id
             "pr_url"     → pr_url
             "user_id"    → request.user_id
             "status"     → JobStatus.PENDING
             "events"     → []           ← the SSE event list
             "result"     → None
             "cost_usd"   → 0.0
             "created_at" → time.time()
      d. Fire the background coroutine (do NOT await it):
             asyncio.create_task(run_agent_with_stream(
                 job_id, pr_url, request.user_id, request.budget_usd))
      e. Return JobResponse(job_id=job_id, status=JobStatus.PENDING,
                            stream_url=f"/jobs/{job_id}/stream")
    """
    # TODO 2: implement job creation + background task launch
    raise NotImplementedError("submit_job() not implemented yet")


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Return job status and metadata (excludes the raw events list).

    TODO 3:
      a. Look up: job = JOBS.get(job_id)
      b. If job is None: raise HTTPException(status_code=404, detail="Job not found")
      c. Return the job dict without the "events" key:
             return {k: v for k, v in job.items() if k != "events"}
    """
    # TODO 3: implement job status lookup
    raise NotImplementedError("get_job() not implemented yet")


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """
    Stream job events as Server-Sent Events (SSE).

    SSE format — each event is a string like:
        data: {"type": "token", "content": "Hello", "job_id": "..."}\\n\\n

    TODO 4:
      a. Validate: job = JOBS.get(job_id); if not job → raise HTTPException(404)
      b. Define an async generator:
             async def event_stream() -> AsyncGenerator[str, None]:
                 seen = 0
                 while True:
                     events = job["events"]
                     while seen < len(events):
                         yield f"data: {events[seen]}\\n\\n"
                         seen += 1
                     if job["status"] in {JobStatus.DONE, JobStatus.FAILED}:
                         break
                     await asyncio.sleep(0.1)
      c. Return StreamingResponse(event_stream(),
                                  media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})
    """
    # TODO 4: implement SSE streaming endpoint
    raise NotImplementedError("stream_job() not implemented yet")


# ── Utility Endpoints (already complete) ──────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check — returns active job count."""
    return {
        "status": "ok",
        "active_jobs": sum(1 for j in JOBS.values() if j["status"] == JobStatus.RUNNING),
    }


@app.get("/users/{user_id}/cost")
async def user_cost(user_id: str):
    """Return cumulative LLM cost for a user."""
    return {"user_id": user_id, "total_cost_usd": round(COST_LEDGER.get(user_id, 0), 6)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
