"""
Exercise 2: SSE Streaming — Stream Agent Tokens to Browser
Goal: Build a FastAPI endpoint that streams agent tokens via Server-Sent Events.

Install: pip install fastapi uvicorn

Run: uvicorn ex2_sse_streaming:app --reload --port 8001
Test: curl "http://localhost:8001/stream?query=what+is+2+plus+2"
  OR open: http://localhost:8001/demo  (HTML page with live streaming)

Tasks:
  1. Complete token_stream() — async generator that yields SSE-formatted strings.
     Each token: "data: {token}\n\n"
     On done:    "data: [DONE]\n\n"
     On error:   "data: ERROR:{message}\n\n"
  2. Complete the /stream endpoint — return StreamingResponse with correct headers.
  3. Complete the /chat endpoint — non-streaming, returns full text.
  4. Add a simple HTML demo page at /demo that connects to /stream via EventSource.
  5. (Bonus) Add ?model= query param to switch models at runtime.

Expected curl output:
  data: The
  data:  answer
  data:  is
  data:  4
  data: [DONE]
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dotenv import load_dotenv
load_dotenv()

try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse, HTMLResponse
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("Run: pip install fastapi uvicorn")

from llm import stream_chat, chat, get_text

app = FastAPI(title="SSE Streaming Agent")

# ── SSE Generator ──────────────────────────────────────────────────────────────

async def token_stream(query: str):
    """
    Async generator yielding SSE-formatted chunks.
    TODO:
      1. Use stream_chat() from llm.py (it's a sync generator — wrap with asyncio)
         OR iterate with litellm.acompletion(stream=True).
      2. Yield each chunk as: f"data: {chunk}\n\n"
      3. After the loop, yield "data: [DONE]\n\n"
      4. Wrap in try/except — yield "data: ERROR:{e}\n\n" on failure.
    """
    # Hint: stream_chat is sync — run in thread executor or switch to async version:
    # from litellm import acompletion
    # response = await acompletion(model=MODEL, messages=[...], stream=True)
    # async for chunk in response:
    #     delta = chunk.choices[0].delta.content or ""
    #     if delta: yield f"data: {delta}\n\n"
    raise NotImplementedError


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/stream")
async def stream_endpoint(query: str = "Tell me a fun fact about Python."):
    """
    TODO: return StreamingResponse(
        token_stream(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    """
    raise NotImplementedError


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Non-streaming: return full response as JSON."""
    # TODO: call chat() and return {"answer": get_text(response)}
    raise NotImplementedError


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Simple HTML page demonstrating EventSource streaming."""
    return HTMLResponse("""<!DOCTYPE html>
<html>
<head><title>SSE Agent Demo</title>
<style>body{font-family:monospace;padding:20px}#output{white-space:pre-wrap;border:1px solid #ccc;padding:10px;min-height:100px}</style>
</head>
<body>
<h2>SSE Streaming Agent</h2>
<input id="query" value="Explain asyncio in 3 sentences." style="width:400px">
<button onclick="stream()">Stream</button>
<div id="output"></div>
<script>
function stream() {
  const q = document.getElementById('query').value;
  const out = document.getElementById('output');
  out.textContent = '';
  const es = new EventSource('/stream?query=' + encodeURIComponent(q));
  es.onmessage = e => {
    if (e.data === '[DONE]') { es.close(); return; }
    if (e.data.startsWith('ERROR:')) { out.textContent += '\\n' + e.data; es.close(); return; }
    out.textContent += e.data;
  };
  es.onerror = () => es.close();
}
</script>
</body></html>""")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
