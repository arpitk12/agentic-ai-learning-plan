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

## Detailed Model Specifications & Hardware Requirements

### Model Selection by Task

#### For **Tool Calling** (Weeks 2–3)
Best choice: **`qwen2.5:7b`**
- 7 billion parameters
- 4.7 GB download (quantized)
- Fastest tool calling among open-source models
- Minimum: **8 GB total system RAM**

```bash
ollama pull qwen2.5:7b
```

#### For **General Chat** (Week 1)
Best choice: **`llama3.2`**
- 1B parameters (lightweight) or 8B (balanced)
- 2.0 GB (1B) / 4.7 GB (8B)
- Lightweight, good for testing
- Minimum: **4 GB RAM** (1B) or **8 GB RAM** (8B)

```bash
ollama pull llama3.2         # Auto selects best version
```

#### For **Reasoning** (Weeks 3, 5, 9)
Best choice: **`qwen2.5:7b`** or **`qwen-qwq:32b`** (if you have >32GB RAM)
- Superior multi-step reasoning
- Better chain-of-thought
- Minimum: **8 GB RAM** (7B) or **40 GB RAM** (32B)

```bash
ollama pull qwen2.5:7b              # 7B version
# OR for more reasoning power:
ollama pull qwen-qwq:32b            # 32B version (requires powerful machine)
```

#### For **Coding** (Projects with code generation)
Best choice: **`mistral`**
- 7 billion parameters
- 4.1 GB download
- Strong code generation (70%+ on HumanEval)
- Fast inference for iteration loops
- Minimum: **8 GB RAM**

```bash
ollama pull mistral
```

#### For **Speed/Evaluation** (Week 10)
Best choice: **`mistral`**
- Fastest inference (~100 tokens/sec on CPU)
- Adequate quality for judge scoring
- Minimum: **8 GB RAM**

#### For **RAG & Synthesis** (Week 4)
Best choice: **`llama3.2:8b`** or **`mistral`**
- Good retrieval-augmented answers
- Fast context window processing
- Minimum: **8 GB RAM**

---

### Model Comparison Table

| Model | Params | Download | RAM | Speed | Tool Calling | Reasoning | Coding | Best For |
|---|---|---|---|---|---|---|---|---|
| **llama3.2:1b** | 1B | 1.1 GB | 4 GB | ⚡⚡⚡⚡ | ⭐ | ⭐ | ⭐ | Testing, basic chat |
| **llama3.2:8b** | 8B | 4.7 GB | 8 GB | ⚡⚡⚡ | ⭐⭐ | ⭐⭐ | ⭐⭐ | Balanced, general use |
| **mistral:7b** | 7B | 4.1 GB | 8 GB | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Fast, coding |
| **qwen2.5:7b** | 7B | 4.7 GB | 8 GB | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **Tool calling** ⭐ |
| **qwen-qwq:32b** | 32B | 20 GB | 40 GB | ⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Deep reasoning |
| **neural-chat:7b** | 7B | 4.1 GB | 8 GB | ⚡⚡⚡ | ⭐⭐ | ⭐⭐ | ⭐⭐ | Chat, lightweight |
| **llama2:13b** | 13B | 7.4 GB | 12 GB | ⚡⚡ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | More reasoning |

---

### Apple Silicon Hardware Matrix

#### **Mac with M1 / M2 chip**
```
GPU cores:  7–10 cores
Unified RAM: 8 GB (base model)

✅ Can run:
  - llama3.2:1b      (comfortable)
  - llama3.2:8b      (acceptable, ~10–15 tokens/sec)
  - mistral:7b       (acceptable, ~10–15 tokens/sec)
  - qwen2.5:7b       (acceptable, ~10–15 tokens/sec)

❌ Cannot run smoothly:
  - qwen-qwq:32b     (would need 32GB+ RAM)
  - llama2:13b       (would be very slow)

Recommendation: Use **qwen2.5:7b** for best tool calling quality.
               Use **mistral** if speed is more important.
```

---

#### **Mac with M1 Pro / M1 Max / M2 Pro / M2 Max**
```
GPU cores:  14–19 cores
Unified RAM: 16 GB (common) or 32 GB (Pro)

✅ Can run smoothly:
  - All 7B models (qwen2.5, mistral, llama3.2:8b)
    Speed: ~20–40 tokens/sec
  - llama2:13b       (~15–25 tokens/sec, 16GB works but tight)

⚠️  Can run with patience:
  - qwen-qwq:32b     (only on M2 Max with 32GB)
    Speed: ~5–8 tokens/sec

Recommendation: **qwen2.5:7b** for tool calling
               **mistral** for coding
               **llama3.2:8b** for balanced approach
```

---

#### **Mac with M3 / M3 Pro / M3 Max / M4**
```
GPU cores:  8–20 cores (M3 Max)
Unified RAM: 24 GB (typical)

✅ Can run very well:
  - All 7B models (qwen2.5, mistral, llama3.2:8b)
    Speed: ~25–50 tokens/sec
  - llama2:13b       (~20–35 tokens/sec)

✅ Can run:
  - qwen-qwq:32b     (~10–15 tokens/sec on Max with 24GB)

Recommendation: **qwen2.5:7b** is the best all-rounder
               **qwen-qwq:32b** if you do heavy reasoning (Weeks 5, 9)
               **mistral** for coding tasks
```

---

#### **Mac with M4 Pro / M4 Ultra**
```
GPU cores:  12–20 cores
Unified RAM: 32+ GB (recommended)

✅ Can run with optimal speed:
  - All 7B models    Speed: ~40–80 tokens/sec
  - llama2:13b       Speed: ~30–50 tokens/sec
  - qwen-qwq:32b     Speed: ~15–25 tokens/sec (on M4 Ultra with 32GB+)

Recommendation: Use **qwen2.5:7b** for best balance
               Use **qwen-qwq:32b** for complex reasoning tasks
               Runs entire curriculum locally without compromise
```

---

#### **Mac with M5 Pro**
```
GPU cores:  20 cores
Unified RAM: 36 GB (recommended)

✅ Can run with excellent speed:
  - All 7B models    Speed: ~40–55 tokens/sec
  - llama2:13b       Speed: ~23–32 tokens/sec
  - qwen-qwq:32b     Speed: ~15–20 tokens/sec (very smooth)

✅ Can also run:
  - Larger models    Speed: Good performance even on larger variants

Recommendation: **qwen2.5:7b** for best overall performance
               **qwen-qwq:32b** for reasoning-heavy tasks (Weeks 5, 9)
               **mistral** for fast iteration on coding
               All models run smoothly; best value for the performance tier
```

---

### Detailed Hardware Specs by Mac Model

| Mac Model | Year | Cores | GPU | Max RAM | 7B Model Speed | 13B Model Speed | Notes |
|---|---|---|---|---|---|---|---|
| MacBook Air M1 | 2020 | 8 core | 7 GPU | 16 GB | ~10–15 tok/s | ❌ Too slow | Entry-level, struggles |
| MacBook Pro M1 | 2021 | 8–10 core | 8–10 GPU | 16 GB | ~15–20 tok/s | ~8 tok/s (tight) | Base model OK |
| MacBook Pro M1 Pro | 2021 | 10–12 core | 14–16 GPU | 32 GB | ~20–30 tok/s | ~12–15 tok/s | Good for tools |
| MacBook Pro M1 Max | 2021 | 10 core | 16–32 GPU | 64 GB | ~30–40 tok/s | ~18–25 tok/s | Excellent |
| MacBook Air M2 | 2022 | 8 core | 8–10 GPU | 24 GB | ~12–18 tok/s | ❌ Very slow | Improved, still tight |
| MacBook Pro M2 | 2023 | 8–12 core | 10–19 GPU | 24 GB | ~18–28 tok/s | ~10–15 tok/s | Good value |
| MacBook Pro M2 Pro | 2023 | 12 core | 16–19 GPU | 32 GB | ~28–40 tok/s | ~16–22 tok/s | Strong performer |
| MacBook Pro M2 Max | 2023 | 12 core | 16–19 GPU | 96 GB | ~35–50 tok/s | ~22–30 tok/s | Overkill for this |
| MacBook Air M3 | 2024 | 8 core | 8–10 GPU | 24 GB | ~15–20 tok/s | ~9–12 tok/s | Better chip |
| MacBook Pro M3 | 2024 | 8–12 core | 8–10 GPU | 24 GB | ~20–30 tok/s | ~12–18 tok/s | Solid baseline |
| MacBook Pro M3 Pro | 2024 | 12–18 core | 14–18 GPU | 36 GB | ~30–45 tok/s | ~18–25 tok/s | **Recommended** |
| MacBook Pro M3 Max | 2024 | 12–18 core | 20–40 GPU | 128 GB | ~40–60 tok/s | ~25–35 tok/s | Overkill |
| MacBook Pro M4 | 2025 | 10–12 core | 10 GPU | 24 GB | ~25–35 tok/s | ~14–20 tok/s | Next-gen |
| MacBook Pro M4 Pro | 2025 | 12–14 core | 20 GPU | 36 GB | ~35–50 tok/s | ~20–30 tok/s | **Future pick** |
| MacBook Pro M5 Pro | 2026 | 12–14 core | 20 GPU | 36 GB | ~40–55 tok/s | ~23–32 tok/s | Latest, top-tier |
| Mac Studio M2 Max | 2023 | 12 core | 19 GPU | 128 GB | ~45–60 tok/s | ~28–40 tok/s | Desktop powerhouse |

---

### RAM Requirements by Model (Simplified)

```
MODEL                MINIMUM RAM    RECOMMENDED RAM    COMFORTABLE RAM
llama3.2:1b         4 GB           8 GB               16 GB
llama3.2:8b         8 GB           12 GB              16+ GB
mistral:7b          8 GB           12 GB              16+ GB
qwen2.5:7b          8 GB           12 GB              16+ GB
llama2:13b          12 GB          16 GB              24+ GB
qwen-qwq:32b        24 GB          32 GB              48+ GB
```

---

### Running Multiple Models Simultaneously

If you have enough RAM, you can keep multiple models loaded:

```bash
# Terminal 1: Keep qwen2.5 running for tool calling
ollama serve --model qwen2.5:7b

# Terminal 2: Quick inference with mistral (separate server on different port)
ollama serve --model mistral --addr 127.0.0.1:11435

# In your code:
# For tool calling
response = chat(messages, model="ollama/qwen2.5:7b", base_url="http://localhost:11434/v1")
# For fast coding
response = chat(messages, model="ollama/mistral", base_url="http://localhost:11435/v1")
```

Requires: **16 GB RAM** minimum for two 7B models

---

### CPU vs GPU Inference

**On Apple Silicon Macs**: GPU is integrated and automatic
- Ollama automatically uses the GPU (Unified Memory)
- All disk bandwidth is fast SSD-based
- No separate GPU VRAM to worry about

**Inference speed depends on**:
1. Total unified RAM size (more = faster)
2. Chip generation (M1 → M2 → M3 → M4 progressively faster)
3. GPU cores (more cores = parallel processing)
4. Model size (larger = slower, but better quality)

---

### Estimated Performance by Model

**Test case**: "Respond to this query in 100 words" on MacBook Pro M3 Pro with 18GB RAM

| Model | Tokens/sec | Latency (first token) | Typical response time |
|---|---|---|---|
| llama3.2:1b | 60+ | <100ms | 2–3 seconds |
| mistral:7b | 25–30 | 200–300ms | 4–6 seconds |
| qwen2.5:7b | 20–25 | 300–400ms | 5–7 seconds |
| llama2:13b | 12–15 | 500–700ms | 8–12 seconds |
| qwen-qwq:32b | 8–10 | 800–1000ms | 12–18 seconds |

---

### Recommended Setup by Mac Model & Budget

#### **Budget: <$1000 (MacBook Air M3 with 16GB RAM)**
```
RAM: 16 GB
Recommended model: qwen2.5:7b
Backup: mistral:7b (faster when speed matters)
Speed: Good enough (~15–20 tok/s)
Use for: Weeks 1–7 + some Week 9
Limitation: Tight on RAM, Week 8+ better on cloud
Example: MacBook Air M3 + 16GB unified memory
```

#### **Value: $1200–1800 (MacBook Air M4 with 24GB RAM or Pro M3 with 18GB)**
```
RAM: 18–24 GB
Recommended: qwen2.5:7b (primary, best tool calling)
Secondary: mistral:7b (coding tasks, fast iteration)
Speed: Very good (~20–30 tok/s for 7B)
Use for: Full curriculum Weeks 1–9 locally
Can add: qwen-qwq:32b if you have 24GB+ (for Week 5, 9 heavy reasoning)
Example: MacBook Air M4 + 24GB or Pro M3 + 18GB
```

#### **Mid-range: $1800–2500 (MacBook Pro M3 Pro / M4 with 24GB+ RAM)**
```
RAM: 24–36 GB
Primary: qwen2.5:7b (general purpose, tool calling)
Secondary: mistral:7b (coding tasks, ~25–35 tok/s)
Tertiary: qwen-qwq:32b (heavy reasoning, Week 5 + 9)
Speed: Excellent (~25–45 tok/s for 7B models)
Use for: Full curriculum Weeks 1–10 entirely locally
Can run: 2–3 models simultaneously if needed
Example: MacBook Pro M3 Pro 12-core + 24GB or M4 + 24GB
```

#### **Latest: $2500–3500 (MacBook Pro M5 Pro with 36GB RAM)**
```
RAM: 36 GB (recommended)
Primary: qwen2.5:7b (best balance of quality + speed)
Secondary: qwen-qwq:32b (complex reasoning, Weeks 5 & 9)
Tertiary: mistral:7b (ultra-fast coding iterations, ~40–55 tok/s)
Speed: Optimal (~40–55 tok/s for 7B, ~15–20 tok/s for 32B)
Use for: Entire curriculum locally, zero cloud dependency
Can run: 2–3 models with comfort, or evaluate all models in parallel
Example: MacBook Pro M5 Pro + 36GB (latest generation)
✨ Best value for serious local development
```

#### **Premium: >$3500 (MacBook Pro M4/M5 Max with 36GB+ RAM)**
```
RAM: 36–96 GB
Primary: qwen2.5:7b (production-ready)
Secondary: qwen-qwq:32b (best reasoning available locally)
Tertiary: mistral:7b + Neural Chat (various use cases)
Speed: Overkill for this curriculum (~40–80 tok/s)
Use for: Enterprise deployments, multiple projects simultaneously
Can run: All models + cloud APIs in parallel
Example: MacBook Pro M5 Max + 48–96GB (future-proofed)
```

---

### Quick Recommendation (TL;DR)

| Goal | Budget | Mac Model | RAM | Primary Model |
|---|---|---|---|---|
| **Try it out** | <$1K | Air M3 | 16GB | mistral (faster) |
| **Full course** | $1.5K | Pro M3 or Air M4 | 24GB | qwen2.5:7b |
| **Best value** | $2.5K | Pro M5 Pro | 36GB | qwen2.5:7b + qwen-qwq:32b |
| **Enterprise** | >$3.5K | Pro M5 Max | 48GB+ | All models available |

---

### Installation for Mac

```bash
# 1. Install Ollama
brew install ollama

# 2. Pull primary model
ollama pull qwen2.5:7b

# 3. Pull optional backup models
ollama pull mistral
ollama pull llama3.2

# 4. Start Ollama daemon
ollama serve

# 5. In another terminal, verify it works
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"Hello"}'
```

---

## Model Sizing Reference

| Model | Size | Speed | Tool Calling | Reasoning | Coding | Best For |
|---|---|---|---|---|---|---|
| `llama3.2` | 2.0GB | ⚡⚡⚡ | ⭐⭐ | ⭐⭐ | ⭐⭐ | Week 1, basic tasks |
| `mistral` | 4.1GB | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Speed, coding |
| `qwen2.5:7b` | 4.7GB | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **Recommended** ⭐ |
| `qwen-qwq:32b` | 20GB | ⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Complex reasoning |
| `gpt-4o-mini` | cloud | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Production, cost-optimal |
| `claude-opus-4-5` | cloud | ⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Complex reasoning (Weeks 3–5, 9) |
