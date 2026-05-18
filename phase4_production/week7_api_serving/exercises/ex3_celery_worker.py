"""
Exercise 3: Background Job Queue with Celery + Redis
Goal: Move agent execution to a Celery worker. Poll for status.

pip install celery redis anthropic fastapi uvicorn python-dotenv

Start Redis: docker run -p 6379:6379 redis
Start worker: celery -A ex3_celery_worker worker --loglevel=info
Start API: python ex3_celery_worker.py
"""
import json
import uuid
from celery import Celery
from fastapi import FastAPI
from pydantic import BaseModel
from llm import chat, get_text, MODEL
import redis

# --- Celery Setup ---
celery_app = Celery(
    "agent_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# --- Redis for status ---
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# --- FastAPI ---
api = FastAPI(title="Async Agent API")


class JobRequest(BaseModel):
    query: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | failed
    result: str | None = None


# --- Celery Task ---
@celery_app.task(bind=True)
def run_agent_task(self, job_id: str, query: str):
    """
    TODO: Implement the agent logic inside this Celery task.
    1. Set job status to "running" in Redis
    2. Run the ReAct agent loop
    3. Set status to "done" with result in Redis
    4. On exception: set status to "failed"
    """
    r.hset(f"job:{job_id}", mapping={"status": "running", "result": ""})

    try:
        # TODO: run agent, get final answer
        answer = "Not implemented"

        r.hset(f"job:{job_id}", mapping={"status": "done", "result": answer})
        r.expire(f"job:{job_id}", 3600)  # expire after 1 hour
    except Exception as e:
        r.hset(f"job:{job_id}", mapping={"status": "failed", "result": str(e)})


# --- API Endpoints ---
@api.post("/jobs", response_model=JobStatus)
async def submit_job(request: JobRequest):
    """Submit a new agent job. Returns job_id immediately."""
    job_id = str(uuid.uuid4())
    r.hset(f"job:{job_id}", mapping={"status": "pending", "result": ""})

    # TODO: Dispatch Celery task with job_id and query
    # run_agent_task.delay(job_id, request.query)

    return JobStatus(job_id=job_id, status="pending")


@api.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Poll job status and result."""
    data = r.hgetall(f"job:{job_id}")
    if not data:
        return JobStatus(job_id=job_id, status="not_found")
    return JobStatus(
        job_id=job_id,
        status=data.get("status", "unknown"),
        result=data.get("result") or None
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8001)
