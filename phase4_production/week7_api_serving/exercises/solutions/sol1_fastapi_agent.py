"""
SOLUTION — Exercise 1: FastAPI Agent with SSE Streaming
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
import math
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import acompletion
from llm import MODEL, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()
app = FastAPI(title="Agent API")


class RunRequest(BaseModel):
    query: str
    max_steps: int = 5


class RunResponse(BaseModel):
    answer: str
    steps: int


TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }
]

_MATH_ALLOWED = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}


async def run_tool(name: str, inputs: dict) -> str:
    if name == "calculator":
        return str(eval(inputs["expression"], {"__builtins__": {}}, _MATH_ALLOWED))  # noqa: S307
    return f"Unknown tool: {name}"


def _sse(event_type: str, content: str) -> str:
    return f"data: {json.dumps({'type': event_type, 'content': content})}\n\n"


async def agent_stream(query: str, max_steps: int = 5):
    messages = [{"role": "user", "content": query}]
    steps = 0

    try:
        while steps < max_steps:
            steps += 1
            full_text = ""

            # Stream tokens via LiteLLM
            stream = await acompletion(
                model=MODEL,
                messages=messages,
                max_tokens=1024,
                tools=TOOLS,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_text += delta
                if delta:
                    yield _sse("token", delta)

            # Get final non-streamed response for tool calls
            response = await acompletion(
                model=MODEL, messages=messages, max_tokens=1024, tools=TOOLS
            )
            messages.append(assistant_message(response))

            if stop_reason(response) == "end_turn":
                yield _sse("done", full_text)
                return

            if stop_reason(response) == "tool_use":
                for tc in get_tool_calls(response):
                    yield _sse("tool", f"{tc['name']}({tc['arguments']})")
                    result = await run_tool(tc["name"], tc["arguments"])
                    yield _sse("result", result)
                    messages.append(tool_result_message(tc["id"], result))

        yield _sse("error", "max_steps reached")
    except Exception as e:
        yield _sse("error", str(e))


@app.post("/agent/run", response_model=RunResponse)
async def run_agent(request: RunRequest):
    tokens = []
    steps = 0
    async for event_str in agent_stream(request.query, request.max_steps):
        data = json.loads(event_str.removeprefix("data: ").strip())
        if data["type"] == "token":
            tokens.append(data["content"])
        elif data["type"] == "tool":
            steps += 1
    return RunResponse(answer="".join(tokens), steps=steps)


@app.get("/agent/stream")
async def stream_agent(query: str, max_steps: int = 5):
    return StreamingResponse(
        agent_stream(query, max_steps),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    # Test: curl "http://localhost:8000/agent/stream?query=what+is+sqrt(144)"
