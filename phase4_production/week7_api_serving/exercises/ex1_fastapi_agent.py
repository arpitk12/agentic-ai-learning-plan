"""
Exercise 1: Expose Agent as FastAPI Endpoint with SSE Streaming
Goal: Wrap a simple ReAct agent in a FastAPI service with streaming.

pip install fastapi uvicorn litellm python-dotenv sse-starlette
"""
import asyncio
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from llm import achat, stream_chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, MODEL
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Agent API")


class RunRequest(BaseModel):
    query: str
    max_steps: int = 5


class RunResponse(BaseModel):
    answer: str
    steps: int


# --- Simple tool for demo ---
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
]


async def run_tool(name: str, inputs: dict) -> str:
    if name == "calculator":
        import math
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        return str(eval(inputs["expression"], {"__builtins__": {}}, allowed))
    return f"Unknown tool: {name}"


# --- Streaming agent generator ---
async def agent_stream(query: str):
    """
    TODO: Implement an async ReAct loop that yields SSE-formatted events.
    Define your own event types (e.g. token, tool, done, error) and format.
    """
    messages = [{"role": "user", "content": query}]

    # TODO: implement the streaming agent loop

    yield f"data: {json.dumps({'type': 'done', 'content': 'Not implemented'})}\n\n"


# --- Endpoints ---
@app.post("/agent/run", response_model=RunResponse)
async def run_agent(request: RunRequest):
    """
    TODO: Non-streaming endpoint — run agent, return full result.
    Collect all tokens, return as RunResponse.
    """
    return RunResponse(answer="Not implemented", steps=0)


@app.get("/agent/stream")
async def stream_agent(query: str):
    """Streaming endpoint — returns SSE stream of agent tokens."""
    return StreamingResponse(
        agent_stream(query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Test: curl "http://localhost:8000/agent/stream?query=what+is+2**10"
