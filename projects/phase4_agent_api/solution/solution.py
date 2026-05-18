"""
SOLUTION — Project 4: Agent-as-a-Service API
Production-grade FastAPI service wrapping the code review agent.
Uses SSE for streaming, Redis for job state, per-user cost tracking.

Run:
    docker-compose up -d        # start Redis
    uvicorn solution:app --reload --port 8000
    celery -A solution worker --loglevel=info

Test:
    curl -X POST http://localhost:8000/jobs -H "Content-Type: application/json" \
         -d '{"pr_url": "https://github.com/owner/repo/pull/1", "user_id": "alice"}'
"""
import asyncio
import json
import uuid
import time
from enum import Enum
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

client = AsyncAnthropic()
app = FastAPI(title="Agent API", version="1.0.0")

# In-memory storage (replace with Redis in production)
JOBS: dict[str, dict] = {}
COST_LEDGER: dict[str, float] = {}  # user_id → total_usd

COST_PER_1K = {
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "claude-opus-4-5":           {"input": 0.015,   "output": 0.075},
}


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobRequest(BaseModel):
    pr_url: str
    user_id: str = "anonymous"
    budget_usd: float = 0.10


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
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
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            raise HTTPException(status_code=400, detail=f"Potential prompt injection detected: '{pattern}'")
    return text[:2000]  # cap length


# ── Cost Tracking ──────────────────────────────────────────────────────────────

def track_cost(user_id: str, model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K.get(model, COST_PER_1K["claude-haiku-4-5-20251001"])
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000
    COST_LEDGER[user_id] = COST_LEDGER.get(user_id, 0) + cost
    return cost


# ── Agent Core ─────────────────────────────────────────────────────────────────

async def run_agent_with_stream(job_id: str, pr_url: str, user_id: str, budget: float):
    JOBS[job_id]["status"] = JobStatus.RUNNING
    events: list[str] = JOBS[job_id]["events"]
    total_cost = 0.0

    def emit(event_type: str, content: str):
        event = json.dumps({"type": event_type, "content": content, "job_id": job_id})
        events.append(event)

    try:
        emit("status", "Starting code review...")
        emit("status", f"Fetching PR: {pr_url}")

        # Simplified single-agent review (swap for full multi-agent in production)
        prompt = (
            f"Review this PR URL and provide a structured security, performance, style, "
            f"and testing analysis: {pr_url}\n\n"
            "Format your response with clear sections for each category."
        )

        emit("status", "Running AI review...")
        full_text = ""

        async with client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                full_text += text
                emit("token", text)

            final = await stream.get_final_message()
            cost = track_cost(user_id, "claude-opus-4-5",
                              final.usage.input_tokens, final.usage.output_tokens)
            total_cost += cost

        if total_cost > budget:
            emit("warning", f"Budget exceeded: ${total_cost:.4f} > ${budget:.4f}")

        emit("cost", f"${total_cost:.6f}")
        emit("done", full_text)
        JOBS[job_id].update({"status": JobStatus.DONE, "result": full_text, "cost_usd": total_cost})

    except Exception as e:
        emit("error", str(e))
        JOBS[job_id]["status"] = JobStatus.FAILED


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.post("/jobs", response_model=JobResponse)
async def submit_job(request: JobRequest):
    pr_url = sanitize_input(request.pr_url)
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "id": job_id,
        "pr_url": pr_url,
        "user_id": request.user_id,
        "status": JobStatus.PENDING,
        "events": [],
        "result": None,
        "cost_usd": 0.0,
        "created_at": time.time(),
    }
    # Fire and forget
    asyncio.create_task(run_agent_with_stream(job_id, pr_url, request.user_id, request.budget_usd))
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        stream_url=f"/jobs/{job_id}/stream",
    )


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items() if k != "events"}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream() -> AsyncGenerator[str, None]:
        seen = 0
        while True:
            events = job["events"]
            while seen < len(events):
                yield f"data: {events[seen]}\n\n"
                seen += 1
            if job["status"] in {JobStatus.DONE, JobStatus.FAILED}:
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": sum(1 for j in JOBS.values() if j["status"] == JobStatus.RUNNING)}


@app.get("/users/{user_id}/cost")
async def user_cost(user_id: str):
    return {"user_id": user_id, "total_cost_usd": round(COST_LEDGER.get(user_id, 0), 6)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
