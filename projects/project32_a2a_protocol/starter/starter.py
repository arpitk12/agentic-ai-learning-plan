"""Project 32 — A2A Protocol: Starter File
pip install fastapi uvicorn httpx pyjwt litellm pydantic python-dotenv
"""
from __future__ import annotations
import os, json, asyncio
from uuid import uuid4
import httpx, litellm
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# ── A2A Types ─────────────────────────────────────────────────────────────────

class AgentCard(BaseModel):
    name: str; url: str; version: str = "1.0.0"
    capabilities: dict = {"streaming": False, "pushNotifications": False}
    authentication: dict = {"schemes": ["bearer"]}
    skills: list[dict] = []

class A2ATask(BaseModel):
    id: str = ""; message: dict = {}; skill: str = ""

class A2AResult(BaseModel):
    id: str; status: dict; result: dict | None = None

# TODO 1: Agent Card endpoint — /.well-known/agent.json
# Returns the agent's capabilities and skills declaration
AGENT_CARD = AgentCard(
    name="TODO-1: agent name",
    url="http://localhost:8001",
    skills=[
        # TODO 1: Add at least one skill dict:
        # {"id": "skill_id", "name": "...", "description": "...",
        #  "inputModes": ["text/plain"], "outputModes": ["application/json"]}
    ],
)

# TODO 2: A2A Task receiver endpoint — POST /a2a/tasks/send
# Accepts a task, runs the agent, returns structured result
async def run_agent_skill(skill: str, text_input: str) -> dict:
    """TODO 2: Execute the requested skill and return result dict."""
    raise NotImplementedError

# TODO 3: A2A Client — discover and call another agent
class A2AClient:
    def __init__(self, base_url: str, token: str = ""):
        self.base_url = base_url
        self.token = token

    async def get_card(self) -> AgentCard:
        """TODO 3: GET {base_url}/.well-known/agent.json and return AgentCard."""
        raise NotImplementedError

    async def send_task(self, skill: str, text: str) -> A2AResult:
        """TODO 3: POST task to /a2a/tasks/send with JWT bearer auth. Return A2AResult."""
        raise NotImplementedError

# TODO 4: JWT auth helpers
SECRET = os.environ.get("A2A_JWT_SECRET", "dev-secret-change-in-prod")

def create_service_token(caller_id: str) -> str:
    """TODO 4: Create JWT with {"sub": caller_id, "exp": now+300}. Use SECRET."""
    # import jwt
    raise NotImplementedError

def verify_token(authorization: str) -> str:
    """TODO 4: Verify Bearer JWT. Return subject. Raise HTTPException 401 if invalid."""
    raise NotImplementedError

# TODO 5: Agent registry (simple dict-based service discovery)
AGENT_REGISTRY: dict[str, str] = {}  # name → URL

def register_agent(name: str, url: str):
    """TODO 5: Register agent in registry. Called on startup."""
    AGENT_REGISTRY[name] = url

async def discover_agent(name: str) -> A2AClient | None:
    """TODO 5: Look up agent by name, health check, return A2AClient or None."""
    raise NotImplementedError

# TODO 6: Streaming A2A (SSE)
async def send_streaming_task(base_url: str, skill: str, text: str):
    """TODO 6: POST to /a2a/tasks/send-streaming, yield SSE events as they arrive."""
    raise NotImplementedError

# TODO 7: Multi-agent chain (compliance → legal → regulatory)
async def orchestrated_review(document: str) -> dict:
    """
    TODO 7: Chain 3 agents via A2A:
    1. Send doc to compliance-agent (port 8001)
    2. Compliance delegates to legal-agent (port 8002) via A2A
    3. Legal delegates to regulatory-agent (port 8003) via A2A
    Return combined result with attribution.
    """
    raise NotImplementedError

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(title="A2A Compliance Agent")

@app.get("/.well-known/agent.json")
async def agent_card_endpoint():
    return AGENT_CARD

@app.post("/a2a/tasks/send", response_model=A2AResult)
async def receive_task(task: A2ATask, authorization: str = Header(...)):
    caller = verify_token(authorization)
    text = task.message.get("parts", [{}])[0].get("text", "")
    result = await run_agent_skill(task.skill, text)
    return A2AResult(id=task.id, status={"state": "completed"}, result=result)

async def main():
    print("=== Project 32: A2A Protocol ===\n")
    print("Agent card:", AGENT_CARD.model_dump())
    token = create_service_token("test-caller")
    print(f"JWT token: {token[:40]}...")
    subject = verify_token(f"Bearer {token}")
    print(f"Verified subject: {subject}")
    print("\nTo run full multi-agent test: start 3 FastAPI servers on ports 8001-8003")
    print("then run: asyncio.run(orchestrated_review('your contract text here'))")

if __name__ == "__main__":
    asyncio.run(main())
