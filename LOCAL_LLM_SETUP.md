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

## Recommended Model by Week & Exercise

| Week | Topic | Recommended Local | Cloud Equiv. | Why | Exercises |
|---|---|---|---|---|---|
| **1** | LLM APIs, chat | `llama3.2` | any | Basic inference | ex1-ex4 |
| **2** | Tool use, ReAct | `qwen2.5:7b` | `gpt-4o-mini` | Best tool calling | ex1-ex3 |
| **3** | LangGraph, frameworks | `qwen2.5:7b` | `claude-opus-4-5` | Reasoning + state mgmt | ex1-ex5 |
| **4** | RAG, retrieval | `llama3.2`/`mistral` | `gpt-4o-mini` | Fast synthesis | ex1-ex6 |
| **5** | Multi-agent orchestration | `qwen2.5:7b` | `claude-opus-4-5` | Complex planning | ex1-ex3 |
| **6** | Parallelism, fan-out | `mistral` | `cerebras/llama3.1-70b` | Speed for async | ex1-ex2 |
| **7** | FastAPI, production | `qwen2.5:7b` | `gpt-4o-mini` | Cost tracking | ex1-ex5 |
| **8** | Observability, logging | 🔄 Cloud required | `gpt-4o-mini` | Cost calc, security | ex1-ex4 |
| **9** | Planning, reflexion | `qwen2.5:7b` | `claude-opus-4-5` | Reasoning chains | ex1-ex4 |
| **10** | Evaluation, LLM judge | `mistral` | `gpt-4o-mini` | Fast scoring | ex1-ex11 |
| **11** | MCP, routing | 🔄 Cloud needed | `gpt-4o-mini` | Multi-model ops | ex1-ex5 |
| **12** | Deployment, K8s | — | `gpt-4o-mini` | Docker, prod serving | ex1-ex5 |

### By Project Group

| Group | Project Range | Focus | Recommended Model |
|---|---|---|---|
| **1 (Core)** | P1–P6 | Foundations to capstone | Start local, switch to cloud |
| **2 (Raw libs)** | P7–P17 | Security, observability, batching | 🔄 Cloud for P8, P15 |
| **3 (Frameworks)** | P18–P22 | CrewAI, AutoGen, LangGraph showcase | `qwen2.5:7b` → cloud |
| **4 (Enterprise)** | P23 | Document processing pipeline | 🔄 Cloud required |
| **5 (Phase 7)** | P24–P35 | Fine-tuning, memory, multimodal | 🔄 Cloud required |
| **6 (Capstone)** | P36 | Enterprise multimodal agent | 🔄 Cloud required |
| **7 (System Design)** | P37–P40 | Context engine, scaling, loadtest | Local OK for testing |
| **8 (LLMOps)** | P41–P43 | Monitoring, prompt versioning, eval | 🔄 Cloud required |

---

## Quick Decision Tree

**Q: I want to start today with zero cost?**
- A: Use `groq/llama-3.3-70b-versatile` (free, no credit card). Get API key at https://console.groq.com

**Q: I prefer running fully offline?**
- A: `ollama pull qwen2.5:7b` then `ollama serve`. Works for Weeks 1–7 + Projects 1–22

**Q: Which single model works best across all exercises?**
- A: `qwen2.5:7b` (local) or `gpt-4o-mini` (cloud). Good tool calling, decent reasoning, low cost

**Q: I'm doing Week 10 (evaluation) — what's fastest?**
- A: `mistral` (local, 4.1GB) or `groq/qwen-qwq-32b` (cloud, free, very fast)

**Q: I'm hitting Week 8+ (observability, production) — can I stay local?**
- A: No — you need `gpt-4o-mini` (cloud) to track actual costs and test security hardening

**Q: Should I use Claude for anything?**
- A: Yes! Weeks 3, 5, 9 benefit from `claude-opus-4-5` (reasoning chains, multi-step planning)
  - But `qwen2.5:7b` is a good free alternative for these weeks

---

## Model Sizing Reference

| Model | Size | Speed | Tool Calling | Reasoning | Best For |
|---|---|---|---|---|---|
| `llama3.2` | 2.0GB | ⚡⚡⚡ | ⭐⭐ | ⭐⭐ | Week 1, basic tasks |
| `mistral` | 4.1GB | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | Week 6 (fast), Week 10 (eval) |
| `qwen2.5:7b` | 4.7GB | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | **Recommended for Weeks 2–7** |
| `gpt-4o-mini` | cloud | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Production, cost-optimal |
| `claude-opus-4-5` | cloud | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Complex reasoning (Weeks 3–5, 9) |
