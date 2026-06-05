"""
SOLUTION — Exercise 2: SSE Streaming — Stream Agent Tokens to Browser
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dotenv import load_dotenv
load_dotenv()

try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse, HTMLResponse
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("Run: pip install fastapi uvicorn")

import litellm
from llm import chat, get_text, MODEL

app = FastAPI(title="SSE Streaming Agent")


async def token_stream(query: str):
    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": query}],
            max_tokens=512,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: ERROR:{e}\n\n"


@app.get("/stream")
async def stream_endpoint(query: str = "Tell me a fun fact about Python."):
    return StreamingResponse(
        token_stream(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = chat([{"role": "user", "content": request.query}], max_tokens=512)
    return {"answer": get_text(response)}


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
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
