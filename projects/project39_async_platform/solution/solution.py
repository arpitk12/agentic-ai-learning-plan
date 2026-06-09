"""
Project 39 — Full Async Agent Platform (SOLUTION)
==================================================
Full implementation of all TODOs from starter.py.
Run with:
  uvicorn solution:app --reload          (FastAPI server)
  celery -A solution worker -l info      (Celery worker)
  python solution.py                     (integration test client)
"""
from __future__ import annotations
import asyncio, json, os, uuid, time, hashlib
from datetime import datetime
from typing import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, Session, SQLModel, create_engine, select
import redis.asyncio as aioredis
import redis as sync_redis
from celery import Celery
import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL        = os.getenv("MODEL", "openai/gpt-4o-mini")
REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = "sqlite:///agent_platform.db"

# ── Celery ─────────────────────────────────────────────────────────────────────
celery_app = Celery("agent_platform", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(task_serializer="json", result_serializer="json")

# ── Redis (sync — for worker) ──────────────────────────────────────────────────
redis_sync = sync_redis.from_url(REDIS_URL, decode_responses=True)

# ── Database ───────────────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Database Models
# ══════════════════════════════════════════════════════════════════════════════

class AgentRun(SQLModel, table=True):
    id:            str   = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    status:        str   = SQLField(default="pending")
    query:         str   = SQLField(default="")
    model:         str   = SQLField(default=MODEL)
    input_tokens:  int   = SQLField(default=0)
    output_tokens: int   = SQLField(default=0)
    cost_usd:      float = SQLField(default=0.0)
    steps:         int   = SQLField(default=0)
    error:         str   = SQLField(default="")
    answer:        str   = SQLField(default="")
    started_at:    str   = SQLField(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at:      str   = SQLField(default="")


def create_db():
    SQLModel.metadata.create_all(engine)


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Step Pub/Sub
# ══════════════════════════════════════════════════════════════════════════════

def channel(run_id: str) -> str:
    return f"agent:{run_id}:steps"


class StepPublisher:
    def publish(self, run_id: str, event_type: str, content: str):
        redis_sync.publish(channel(run_id), json.dumps({"type": event_type, "content": content}))


class StepSubscriber:
    async def listen(self, run_id: str, timeout: float = 120.0) -> AsyncIterator[dict]:
        r = await aioredis.from_url(REDIS_URL, decode_responses=True)
        pubsub = r.pubsub()
        await pubsub.subscribe(channel(run_id))
        deadline = time.time() + timeout
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                yield data
                if data["type"] in ("done", "error"):
                    break
            if time.time() > deadline:
                break
        await pubsub.unsubscribe(channel(run_id))
        await r.aclose()


publisher  = StepPublisher()
subscriber = StepSubscriber()


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class CheckpointData:
    run_id:   str
    step:     int
    messages: list[dict]
    cost:     float


class AgentCheckpoint:
    TTL = 3600

    def save(self, data: CheckpointData):
        redis_sync.setex(
            f"checkpoint:{data.run_id}",
            self.TTL,
            json.dumps(data.__dict__),
        )

    def load(self, run_id: str) -> CheckpointData | None:
        raw = redis_sync.get(f"checkpoint:{run_id}")
        if not raw:
            return None
        d = json.loads(raw)
        return CheckpointData(**d)

    def clear(self, run_id: str):
        redis_sync.delete(f"checkpoint:{run_id}")


checkpoint_store = AgentCheckpoint()


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Mock Tool
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }
]


def web_search(query: str) -> str:
    return (
        f"[Search results for '{query}']: GDPR Article 28 requires written DPAs. "
        "Controllers must ensure processors provide sufficient guarantees. "
        "Non-compliance: fines up to €10M or 2% of global turnover."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Celery Task (Agent Loop)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from billiard.exceptions import SoftTimeLimitExceeded  # type: ignore
except ImportError:
    SoftTimeLimitExceeded = Exception


def _get_db():
    return Session(engine)


def _update_run(run_id: str, **kwargs):
    with _get_db() as db:
        run = db.get(AgentRun, run_id)
        if run:
            for k, v in kwargs.items():
                setattr(run, k, v)
            db.add(run)
            db.commit()


@celery_app.task(
    bind=True,
    max_retries=3,
    soft_time_limit=120,
    time_limit=150,
    acks_late=True,
    name="run_agent",
)
def run_agent_task(self, run_id: str, query: str):
    """ReAct agent loop running in Celery worker."""
    create_db()  # ensure tables exist in worker process

    try:
        # Update status → running
        _update_run(run_id, status="running")
        publisher.publish(run_id, "status", "running")

        # Load checkpoint if it exists (resume)
        cp = checkpoint_store.load(run_id)
        if cp:
            messages = cp.messages
            step     = cp.step
            total_cost = cp.cost
            publisher.publish(run_id, "info", f"Resuming from checkpoint at step {step}")
        else:
            messages = [
                {"role": "system", "content": "You are a helpful compliance assistant. Use web_search when needed. Answer concisely."},
                {"role": "user",   "content": query},
            ]
            step       = 0
            total_cost = 0.0

        answer = ""
        total_in = total_out = 0

        for _ in range(10):  # max steps
            step += 1
            publisher.publish(run_id, "thinking", f"Step {step}: calling LLM…")

            resp = litellm.completion(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=512,
                temperature=0.0,
            )

            # Accumulate tokens / cost
            usage = resp.usage or {}
            total_in  += getattr(usage, "prompt_tokens", 0)
            total_out += getattr(usage, "completion_tokens", 0)
            total_cost += getattr(resp, "_hidden_params", {}).get("response_cost", 0.0) or 0.0

            msg = resp.choices[0].message

            # Tool call branch
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments or "{}")
                    publisher.publish(run_id, "tool_call", f"{fn_name}({fn_args})")

                    if fn_name == "web_search":
                        result = web_search(fn_args.get("query", ""))
                    else:
                        result = f"[Unknown tool: {fn_name}]"

                    publisher.publish(run_id, "tool_result", result[:200])

                    messages.append(msg.model_dump(exclude_none=True))
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
            else:
                # Final answer
                answer = msg.content or ""
                publisher.publish(run_id, "answer", answer[:200])
                break

            # Checkpoint after each step
            checkpoint_store.save(CheckpointData(run_id, step, messages, total_cost))
            _update_run(run_id, steps=step, input_tokens=total_in, output_tokens=total_out, cost_usd=total_cost)

        # Success
        checkpoint_store.clear(run_id)
        _update_run(
            run_id,
            status="done",
            answer=answer,
            steps=step,
            input_tokens=total_in,
            output_tokens=total_out,
            cost_usd=total_cost,
            ended_at=datetime.utcnow().isoformat(),
        )
        publisher.publish(run_id, "done", answer[:200])

    except SoftTimeLimitExceeded:
        _update_run(run_id, status="failed", error="timeout", ended_at=datetime.utcnow().isoformat())
        publisher.publish(run_id, "error", "Task timed out")

    except Exception as exc:
        err_str = str(exc)
        retry_count = self.request.retries
        if retry_count < self.max_retries:
            _update_run(run_id, status="running", error=f"Retrying ({retry_count+1}): {err_str}")
            raise self.retry(exc=exc, countdown=2 ** retry_count)
        else:
            _update_run(run_id, status="failed", error=err_str, ended_at=datetime.utcnow().isoformat())
            publisher.publish(run_id, "error", err_str[:200])


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Idempotency
# ══════════════════════════════════════════════════════════════════════════════

def idempotency_key_hash(key: str, path: str, body: bytes) -> str:
    return hashlib.sha256(f"{key}:{path}:{body}".encode()).hexdigest()


async def check_idempotency(key: str, path: str, body: bytes) -> dict | None:
    h = idempotency_key_hash(key, path, body)
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    raw = await r.get(f"idem:{h}")
    await r.aclose()
    return json.loads(raw) if raw else None


async def store_idempotency(key: str, path: str, body: bytes, response: dict, ttl: int = 86400):
    h = idempotency_key_hash(key, path, body)
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    await r.setex(f"idem:{h}", ttl, json.dumps(response))
    await r.aclose()


# ══════════════════════════════════════════════════════════════════════════════
# 7 — FastAPI App
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db()
    yield


app = FastAPI(title="Agent Platform", lifespan=lifespan)


class RunRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    model: str = Field(default=MODEL)


class RunResponse(BaseModel):
    run_id:  str
    status:  str
    message: str


@app.post("/agent/run", status_code=202, response_model=RunResponse)
async def create_run(
    request: Request,
    body: RunRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
):
    raw_body = await request.body()

    # Idempotency check
    if idempotency_key:
        cached = await check_idempotency(idempotency_key, str(request.url.path), raw_body)
        if cached:
            return JSONResponse(content=cached, status_code=202)

    # Create run in DB
    run = AgentRun(query=body.query, model=body.model)
    with Session(engine) as db:
        db.add(run)
        db.commit()
        db.refresh(run)

    # Enqueue Celery task
    run_agent_task.delay(run.id, body.query)

    response_data = RunResponse(
        run_id=run.id,
        status="pending",
        message="Agent run queued",
    ).model_dump()

    # Cache for idempotency
    if idempotency_key:
        await store_idempotency(idempotency_key, str(request.url.path), raw_body, response_data)

    return JSONResponse(content=response_data, status_code=202)


@app.get("/agent/run/{run_id}/status")
async def get_status(run_id: str):
    with Session(engine) as db:
        run = db.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id":     run.id,
        "status":     run.status,
        "steps":      run.steps,
        "cost_usd":   run.cost_usd,
        "started_at": run.started_at,
        "ended_at":   run.ended_at,
    }


@app.get("/agent/run/{run_id}/stream")
async def stream_run(run_id: str):
    from sse_starlette.sse import EventSourceResponse  # type: ignore

    async def event_generator():
        async for event in subscriber.listen(run_id):
            yield {"data": json.dumps(event)}
            if event["type"] in ("done", "error"):
                break

    return EventSourceResponse(event_generator())


@app.get("/agent/run/{run_id}/result")
async def get_result(run_id: str):
    with Session(engine) as db:
        run = db.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in ("done", "failed"):
        raise HTTPException(status_code=404, detail=f"Run not complete (status={run.status})")
    return {
        "run_id":        run.id,
        "status":        run.status,
        "answer":        run.answer,
        "error":         run.error,
        "cost_usd":      run.cost_usd,
        "steps":         run.steps,
        "input_tokens":  run.input_tokens,
        "output_tokens": run.output_tokens,
        "latency_ms":    (
            (datetime.fromisoformat(run.ended_at) - datetime.fromisoformat(run.started_at)).total_seconds() * 1000
            if run.ended_at else None
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test Client
# ══════════════════════════════════════════════════════════════════════════════

async def test_client():
    import httpx
    BASE      = "http://localhost:8000"
    QUERY     = "What are the requirements of GDPR Article 28 for Data Processing Agreements?"
    IDEM_KEY  = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=180) as client:
        # Submit
        print("POST /agent/run")
        resp = await client.post(
            f"{BASE}/agent/run",
            json={"query": QUERY},
            headers={"Idempotency-Key": IDEM_KEY},
        )
        assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
        data   = resp.json()
        run_id = data["run_id"]
        print(f"  run_id: {run_id}")

        # Poll status
        print("\nPolling status…")
        for i in range(30):
            await asyncio.sleep(1)
            s      = await client.get(f"{BASE}/agent/run/{run_id}/status")
            status = s.json()["status"]
            print(f"  [{i+1:2d}s] status: {status}")
            if status in ("done", "failed"):
                break

        # Stream
        print("\nStreaming events:")
        async with client.stream("GET", f"{BASE}/agent/run/{run_id}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    print(f"  [{payload['type']}] {str(payload.get('content', ''))[:80]}")
                    if payload["type"] in ("done", "error"):
                        break

        # Result
        print("\nResult:")
        for _ in range(5):
            r = await client.get(f"{BASE}/agent/run/{run_id}/result")
            if r.status_code == 200:
                result = r.json()
                print(f"  answer: {str(result.get('answer', ''))[:100]}…")
                print(f"  cost:   ${result.get('cost_usd', 0):.5f}")
                print(f"  steps:  {result.get('steps', 0)}")
                break
            await asyncio.sleep(2)

        # Idempotency test
        print("\nIdempotency check:")
        resp2 = await client.post(
            f"{BASE}/agent/run",
            json={"query": QUERY},
            headers={"Idempotency-Key": IDEM_KEY},
        )
        same = resp2.json()["run_id"] == run_id
        print(f"  Same run_id: {same} {'✅' if same else '❌'}")


if __name__ == "__main__":
    asyncio.run(test_client())
