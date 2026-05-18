"""
SOLUTION — Exercise 1: FastAPI Agent with SSE Streaming
"""
import json
import math
import asyncio
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()
client = AsyncAnthropic()
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
            # Stream tokens
            full_text = ""
            tool_uses = []

            async with client.messages.stream(
                model="claude-opus-4-5",
                max_tokens=1024,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    full_text += text
                    yield _sse("token", text)

                final = await stream.get_final_message()

            messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason == "end_turn":
                yield _sse("done", full_text)
                return

            if final.stop_reason == "tool_use":
                tool_results = []
                for block in final.content:
                    if block.type == "tool_use":
                        yield _sse("tool", f"{block.name}({json.dumps(block.input)})")
                        result = await run_tool(block.name, block.input)
                        yield _sse("result", result)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})

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
