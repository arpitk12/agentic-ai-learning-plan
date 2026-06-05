"""
SOLUTION — Exercise 3: Background Job Queue with Celery + Redis
"""
import json
import uuid
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from celery import Celery
from fastapi import FastAPI
from pydantic import BaseModel
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message
import redis

celery_app = Celery(
    "agent_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

api = FastAPI(title="Async Agent API")

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]


def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            return str(eval(args["expression"], {"__builtins__": {}}))
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown: {name}"


class JobRequest(BaseModel):
    query: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: str | None = None


@celery_app.task(bind=True)
def run_agent_task(self, job_id: str, query: str):
    r.hset(f"job:{job_id}", mapping={"status": "running", "result": ""})
    try:
        messages = [{"role": "user", "content": query}]
        answer = ""
        for _ in range(8):
            response = chat(messages, tools=TOOLS, max_tokens=512)
            messages.append(assistant_message(response))
            if stop_reason(response) == "end_turn":
                answer = get_text(response)
                break
            for tc in get_tool_calls(response):
                result = run_tool(tc["name"], tc["arguments"])
                messages.append(tool_result_message(tc["id"], result))

        r.hset(f"job:{job_id}", mapping={"status": "done", "result": answer})
        r.expire(f"job:{job_id}", 3600)
    except Exception as e:
        r.hset(f"job:{job_id}", mapping={"status": "failed", "result": str(e)})


@api.post("/jobs", response_model=JobStatus)
async def submit_job(request: JobRequest):
    job_id = str(uuid.uuid4())
    r.hset(f"job:{job_id}", mapping={"status": "pending", "result": ""})
    run_agent_task.delay(job_id, request.query)
    return JobStatus(job_id=job_id, status="pending")


@api.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    data = r.hgetall(f"job:{job_id}")
    if not data:
        return JobStatus(job_id=job_id, status="not_found")
    return JobStatus(
        job_id=job_id,
        status=data.get("status", "unknown"),
        result=data.get("result") or None,
    )


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8001)
