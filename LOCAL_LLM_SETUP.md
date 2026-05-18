# Local LLM Setup Guide

## Why Local First?

| | Local (Ollama) | Cloud (Anthropic/OpenAI) |
|---|---|---|
| Cost | Free | Pay per token |
| Speed | Depends on your GPU/CPU | Fast, consistent |
| Privacy | 100% local | Data sent to provider |
| Tool calling quality | Good (Qwen2.5) / OK (Llama3) | Excellent |
| Structured output | Needs careful prompting | Very reliable |
| Production realism | 90% — same patterns apply | 100% |

**Verdict**: Start local, switch to cloud for Projects 4-6 where reliability matters more.

---

## Step 1 — Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com/download
```

---

## Step 2 — Pull a Model

For this course, pull in this order of preference:

```bash
# Best for tool calling (recommended for Weeks 2-6)
ollama pull qwen2.5:7b           # 4.7GB

# Good all-rounder, lighter
ollama pull llama3.2             # 2.0GB

# Fast, good for eval exercises (Week 10)
ollama pull mistral              # 4.1GB
```

---

## Step 3 — Start Ollama

```bash
ollama serve
# Runs on http://localhost:11434 — leave this terminal open
```

---

## Step 4 — Configure .env

```bash
cp .env.example .env
# Default is already: MODEL=ollama/llama3.2
# Change to: MODEL=ollama/qwen2.5:7b for better tool calling
```

---

## Step 5 — Install Python deps

```bash
pip install litellm python-dotenv pydantic
```

---

## Step 6 — Smoke test

```bash
python llm.py
# Should print: Sync: 'hello'
```

---

## Using `llm.py` in Exercises

All exercises import from `llm.py` instead of the Anthropic SDK directly.
This is the **only** file you change when switching to cloud.

```python
# At the top of every exercise:
from llm import chat, stream_chat, achat, get_text, get_tool_calls, stop_reason, MODEL

# Sync call
response = chat(messages, system="You are helpful.")
text = get_text(response)

# Streaming
for chunk in stream_chat(messages):
    print(chunk, end="", flush=True)

# Async (Week 6 fan-out exercises)
response = await achat(messages)
```

---

## Switching to Cloud (When Ready)

1. Get your API key: https://console.anthropic.com or https://platform.openai.com
2. Edit `.env`:
   ```
   MODEL=claude-opus-4-5
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Run the same exercise — **zero code changes needed**.

---

## Known Differences: Local vs Cloud

| Behaviour | Local (Llama/Qwen) | Cloud (Claude/GPT) |
|---|---|---|
| JSON format adherence | Needs `format: json` param or explicit schema in prompt | Reliable with system prompt alone |
| Tool calling | Works, but may need retries | Very reliable |
| Multi-step reasoning | Good up to ~4 steps | Excellent up to 10+ steps |
| Context window | 8K-128K depending on model | 200K (Claude) |
| Parallel tool calls | Limited support | Full support |

**For Weeks 2-3 tool-calling exercises**: use `qwen2.5:7b` — it's the best local model for this.

---

## Recommended Model by Phase

| Phase | Recommended Local Model | Cloud Equivalent |
|---|---|---|
| Week 1 — LLM APIs | `llama3.2` | any |
| Week 2-3 — Tool Use, LangGraph | `qwen2.5:7b` | `claude-haiku-4-5-20251001` |
| Week 4 — RAG | `llama3.2` or `mistral` | any |
| Week 5-6 — Multi-Agent | `qwen2.5:7b` | `claude-opus-4-5` |
| Week 7-8 — Production API | `qwen2.5:7b` | `claude-haiku-4-5-20251001` |
| Week 9-10 — Evals | `mistral` (fast) | `claude-haiku-4-5-20251001` |
| Projects 4-6 | 🔄 Switch to cloud | `claude-opus-4-5` |
