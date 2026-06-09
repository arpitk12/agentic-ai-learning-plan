"""
Project 39 — Full Async Agent Platform (starter)
=================================================
Build the complete async agent execution platform:
  POST /agent/run → 202 + run_id
  GET  /agent/run/{id}/status → polling endpoint
  GET  /agent/run/{id}/stream → SSE step-by-step events
  GET  /agent/run/{id}/result → final answer
  Celery worker → actual agent execution
  Redis pub/sub → step events
  Checkpointing → crash recovery
  Idempotency → safe client retries

Companion: guide/13_system_design.md §9 — Async, Queues & Concurrency

Fill in every # TODO block. Run with uvicorn + celery (see README).
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

MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = "sqlite:///agent_platform.db"

# ── Celery ─────────────────────────────────────────────────────────────────────
celery_app = Celery("agent_platform", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(task_serializer="json", result_serializer="json")

# ── Redis clients ──────────────────────────────────────────────────────────────
redis_sync   = sync_redis.from_url(REDIS_URL, decode_responses=True)
# aioredis client created per-request (thread-safe)

# ── Database ───────────────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Database Models
# ══════════════════════════════════════════════════════════════════════════════

class AgentRun(SQLModel, table=True):
    # TODO 1: Define fields:
    # id: str (UUID, primary key, default_factory=lambda: str(uuid.uuid4()))
    # status: str (default "pending") — "pending"|"running"|"done"|"failed"|"cancelled"
    # query: str
    # model: str (default MODEL)
    # input_tokens: int (default 0)
    # output_tokens: int (default 0)
    # cost_usd: float (default 0.0)
    # steps: int (default 0)
    # error: str (default "")
    # answer: str (default "")
    # started_at: str (default_factory=lambda: datetime.utcnow().isoformat())
    # ended_at: str (default "")
    id: str = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    status: str = SQLField(default="pending")
    query: str = SQLField(default="")
    # TODO 1: add remaining fields
    raise NotImplementedError  # remove this line and implement above


def create_db():
    SQLModel.metadata.create_all(engine)


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Step Pub/Sub (Redis)
# ══════════════════════════════════════════════════════════════════════════════

def channel(run_id: str) -> str:
    return f"agent:{run_id}:steps"


class StepPublisher:
    """Publishes step events to Redis pub/sub. Used from Celery worker (sync)."""

    def publish(self, run_id: str, event_type: str, content: str):
        # TODO 2: redis_sync.publish(channel(run_id), json.dumps({"type": event_type, "content": content}))
        raise NotImplementedError


class StepSubscriber:
    """Async generator that yields step events from Redis pub/sub. Used from FastAPI."""

    async def listen(self, run_id: str, timeout: float = 120.0) -> AsyncIterator[dict]:
        # TODO 3:
        # r = await aioredis.from_url(REDIS_URL, decode_responses=True)
        # pubsub = r.pubsub()
        # await pubsub.subscribe(channel(run_id))
        # deadline = time.time() + timeout
        # async for message in pubsub.listen():
        #     if message["type"] == "message":
        #         data = json.loads(message["data"])
        #         yield data
        #         if data["type"] in ("done", "error"):
        #             break
        #     if time.time() > deadline:
        #         break
        raise NotImplementedError


publisher   = StepPublisher()
subscriber  = StepSubscriber()


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Agent Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

@dataclass  # type: ignore — define as a regular dataclass
class CheckpointData:
    run_id: str
    step: int
    messages: list[dict]
    cost: float


class AgentCheckpoint:
    TTL = 3600  # 1 hour

    def save(self, data: CheckpointData):
        # TODO 4: redis_sync.setex(f"checkpoint:{data.run_id}", self.TTL, json.dumps(data.__dict__))
        raise NotImplementedError

    def load(self, run_id: str) -> CheckpointData | None:
        # TODO 5: get from Redis; return None if missing; deserialize CheckpointData
        raise NotImplementedError

    def clear(self, run_id: str):
        # TODO 6: redis_sync.delete(f"checkpoint:{run_id}")
        raise NotImplementedError


checkpoint_store = AgentCheckpoint()


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Mock Tool
# ══════════════════════════════════════════════════════════════════════════════

def web_search(query: str) -> str:
    """Simulated web search."""
    return (
        f"[Search results for '{query}']: GDPR Article 28 requires written DPAs. "
        "Controllers must ensure processors provide sufficient guarantees. "
        "Non-compliance: fines up to €10M or 2% of global turnover."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Celery Task (Agent Loop)
# ══════════════════════════════════════════════════════════════════════════════

from billiard.exceptions import SoftTimeLimitExceeded  # type: ignore

@celery_app.task(
    bind=True,
    max_retries=3,
    soft_time_limit=120,
    time_limit=150,
    acks_late=True,
    name="run_agent",
)
def run_agent_task(self, run_id: str, query: str):
    """
    Main agent execution. Runs in Celery worker.
    Publishes step events to Redis pub/sub.
    Checkpoints after every step.
    Updates DB on completion or failure.
    """
    # TODO 7: Update DB status → "running"
    # TODO 8: Load checkpoint (resume if exists)
    # TODO 9: ReAct loop (max 10 steps):
    #   - LLM call (sync: litellm.completion)
    #   - publish step event (thinking / tool_call / tool_result)
    #   - if tool call → execute web_search → append result
    #   - if final answer → break
    #   - checkpoint after each step
    #   - Update DB (steps count, cost)
    # TODO 10: On success → clear checkpoint, update DB (status=done, answer, cost, ended_at), publish done
    # TODO 11: On SoftTimeLimitExceeded → update DB (status=failed, error="timeout"), publish error
    # TODO 12: On other exception → update DB (status=failed, error=str(e)), retry or publish error
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Idempotency
# ══════════════════════════════════════════════════════════════════════════════

def idempotency_key_hash(key: str, path: str, body: bytes) -> str:
    return hashlib.sha256(f"{key}:{path}:{body}".encode()).hexdigest()


async def check_idempotency(key: str, path: str, body: bytes) -> dict | None:
    """Return cached response dict if this request was already processed, else None."""
    # TODO 13: hash key → redis GET → json.loads if found, None otherwise
    raise NotImplementedError


async def store_idempotency(key: str, path: str, body: bytes, response: dict, ttl: int = 86400):
    """Cache the response for 24h."""
    # TODO 14: hash key → redis SETEX(hash, ttl, json.dumps(response))
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 7 — FastAPI App
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db()
    yield
    # Cleanup if needed

app = FastAPI(title="Agent Platform", lifespan=lifespan)


class RunRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    model: str = Field(default=MODEL)


class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


@app.post("/agent/run", status_code=202, response_model=RunResponse)
async def create_run(
    request: Request,
    body: RunRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
):
    """
    Submit an agent run. Returns 202 immediately with run_id.
    Idempotency-Key header enables safe retries.
    """
    # TODO 15: If idempotency_key → check cache → return cached response if found
    # TODO 16: Create AgentRun in DB (status=pending)
    # TODO 17: Enqueue run_agent_task.delay(run_id, query)
    # TODO 18: Build RunResponse(run_id, status="pending", message="queued")
    # TODO 19: If idempotency_key → store response in cache
    # TODO 20: Return 202 with RunResponse
    raise NotImplementedError


@app.get("/agent/run/{run_id}/status")
async def get_status(run_id: str):
    """Poll for current run status."""
    # TODO 21: Fetch AgentRun from DB; 404 if not found
    # TODO 22: Return {run_id, status, steps, cost_usd, started_at, ended_at}
    raise NotImplementedError


@app.get("/agent/run/{run_id}/stream")
async def stream_run(run_id: str):
    """
    SSE stream of step events. Client connects and receives events as they happen.
    Format: "data: {json}\n\n"
    """
    from sse_starlette.sse import EventSourceResponse  # type: ignore

    async def event_generator():
        # TODO 23: Use StepSubscriber.listen(run_id) to yield SSE events
        # Each event: {"data": json.dumps(step_event)}
        # Yield done or error as final event
        raise NotImplementedError

    return EventSourceResponse(event_generator())


@app.get("/agent/run/{run_id}/result")
async def get_result(run_id: str):
    """Return the final result. 404 if not done yet."""
    # TODO 24: Fetch AgentRun from DB; raise 404 if not done
    # Return full result: answer, cost_usd, steps, input_tokens, output_tokens, latency
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Test Client (run standalone)
# ══════════════════════════════════════════════════════════════════════════════

async def test_client():
    """Integration test: submit → poll → stream → result → idempotency check."""
    import httpx

    BASE = "http://localhost:8000"
    QUERY = "What are the requirements of GDPR Article 28 for Data Processing Agreements?"
    IDEM_KEY = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=180) as client:
        # Step 1: Submit
        print("POST /agent/run")
        resp = await client.post(
            f"{BASE}/agent/run",
            json={"query": QUERY},
            headers={"Idempotency-Key": IDEM_KEY},
        )
        assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
        data = resp.json()
        run_id = data["run_id"]
        print(f"  run_id: {run_id}")

        # Step 2: Poll status
        print("\nPolling status...")
        for i in range(30):
            await asyncio.sleep(1)
            s = await client.get(f"{BASE}/agent/run/{run_id}/status")
            status = s.json()["status"]
            print(f"  [{i+1}s] status: {status}")
            if status in ("running", "done", "failed"):
                break

        # Step 3: Stream events (non-blocking)
        print("\nStreaming events:")
        async with client.stream("GET", f"{BASE}/agent/run/{run_id}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    print(f"  [{payload['type']}] {str(payload.get('content',''))[:80]}")
                    if payload["type"] in ("done", "error"):
                        break

        # Step 4: Get result
        print("\nResult:")
        for _ in range(10):
            r = await client.get(f"{BASE}/agent/run/{run_id}/result")
            if r.status_code == 200:
                result = r.json()
                print(f"  answer:     {str(result.get('answer',''))[:100]}...")
                print(f"  cost:       ${result.get('cost_usd', 0):.5f}")
                print(f"  steps:      {result.get('steps', 0)}")
                break
            await asyncio.sleep(2)

        # Step 5: Idempotency test
        print("\nIdempotency test:")
        resp2 = await client.post(
            f"{BASE}/agent/run",
            json={"query": QUERY},
            headers={"Idempotency-Key": IDEM_KEY},
        )
        data2 = resp2.json()
        same = data2["run_id"] == run_id
        print(f"  Same run_id: {same} {'✅' if same else '❌'}")


if __name__ == "__main__":
    asyncio.run(test_client())
