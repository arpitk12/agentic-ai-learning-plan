# Free Cloud LLM Options

All providers below work with `llm.py` out of the box — no code changes,
just update your `.env` file and install the matching SDK if needed.

---

## Quick Switch

```bash
# 1. Edit .env — change one line
MODEL=groq/llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...

# 2. Run any exercise exactly as before
python ex1_basic_tools.py
```

That's it. `llm.py` passes the model string to LiteLLM which routes automatically.

---

## Provider Details

### 1. Groq ⭐ Recommended
**Best for:** Weeks 2–5 (tool calling), fast iteration, all exercises

| | |
|---|---|
| Sign up | [console.groq.com](https://console.groq.com) — no credit card |
| Free tier | 6,000–14,400 tokens/min depending on model |
| Tool calling | ✅ Excellent |

**`.env` setup:**
```env
MODEL=groq/llama-3.3-70b-versatile
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

**Good model options:**
```env
# Fast + smart general use
MODEL=groq/llama-3.3-70b-versatile

# Best reasoning / self-reflection (Week 9)
MODEL=groq/qwen-qwq-32b

# Fastest, cheapest for evals (Week 10)
MODEL=groq/llama3-8b-8192
```

**Install:**
```bash
pip install litellm  # already included — no extra SDK needed
```

---

### 2. Google Gemini
**Best for:** Week 4 (RAG — huge context window), highest free quota

| | |
|---|---|
| Sign up | [aistudio.google.com](https://aistudio.google.com) — no credit card |
| Free tier | 1,500 req/day, 1M tokens/min (Flash) |
| Tool calling | ✅ Good |

**`.env` setup:**
```env
MODEL=gemini/gemini-2.0-flash
GEMINI_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Good model options:**
```env
# Best free model — fast + capable
MODEL=gemini/gemini-2.0-flash

# More capable, still free
MODEL=gemini/gemini-1.5-pro
```

**Install:**
```bash
pip install litellm google-generativeai
```

---

### 3. Cerebras
**Best for:** Speed benchmarking (Week 6 — parallelism), extremely fast

| | |
|---|---|
| Sign up | [cloud.cerebras.ai](https://cloud.cerebras.ai) — no credit card |
| Free tier | 60 req/min, 1M tokens/day |
| Tool calling | ✅ Supported |

**`.env` setup:**
```env
MODEL=cerebras/llama3.1-70b
CEREBRAS_API_KEY=csk_xxxxxxxxxxxxxxxxxxxx
```

**Install:**
```bash
pip install litellm cerebras-cloud-sdk
```

---

### 4. OpenRouter (Free Models)
**Best for:** Trying many different models without separate accounts

| | |
|---|---|
| Sign up | [openrouter.ai](https://openrouter.ai) — no credit card for free models |
| Free tier | Rate-limited free models always available |
| Tool calling | Varies by model |

**`.env` setup:**
```env
MODEL=openrouter/mistralai/mistral-7b-instruct:free
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
```

**Free models to try:**
```env
MODEL=openrouter/mistralai/mistral-7b-instruct:free
MODEL=openrouter/meta-llama/llama-3.2-3b-instruct:free
MODEL=openrouter/google/gemma-3-12b-it:free
```

**Install:**
```bash
pip install litellm  # no extra SDK needed
```

---

## Week-by-Week Recommendations

| Week | Topic | Recommended Free Model |
|---|---|---|
| 1 | LLM API basics | `groq/llama-3.3-70b-versatile` or `gemini/gemini-2.0-flash` |
| 2 | Tool use | `groq/llama-3.3-70b-versatile` (best tool calling) |
| 3 | LangGraph | `groq/llama-3.3-70b-versatile` |
| 4 | RAG | `gemini/gemini-2.0-flash` (large context) |
| 5 | Orchestrator | `groq/llama-3.3-70b-versatile` |
| 6 | Parallelism | `cerebras/llama3.1-70b` (fastest) |
| 7 | FastAPI | `groq/llama-3.3-70b-versatile` |
| 8 | Observability | Any (cost tracking works for all) |
| 9 | Planning / reflection | `groq/qwen-qwq-32b` (best reasoning) |
| 10 | Evaluation / LLM Judge | `groq/llama-3.3-70b-versatile` as judge |

---

## LangGraph Exercises (Week 3) — Special Case

Week 3 uses `ChatLiteLLM` instead of the raw `chat()` function.
Pass the same model string there:

```python
from langchain_community.chat_models import ChatLiteLLM
import os

llm = ChatLiteLLM(model=os.getenv("MODEL", "groq/llama-3.3-70b-versatile")).bind_tools(TOOLS)
```

**Install for Week 3:**
```bash
pip install langchain-community langgraph
```

---

## Switching Models Mid-Exercise

You can override `MODEL` per-run without editing `.env`:

```bash
# Run with Groq
MODEL=groq/llama-3.3-70b-versatile python ex1_basic_tools.py

# Run with Gemini
MODEL=gemini/gemini-2.0-flash python ex1_basic_tools.py

# Run with local Ollama
MODEL=ollama/llama3.2 python ex1_basic_tools.py
```

---

## Troubleshooting

**`AuthenticationError`** — API key not set or wrong env var name. Check the table above.

**`RateLimitError`** — Hit free tier limit. Wait a minute or switch to another provider.

**Tool calls not working** — Some free models ignore tool schemas. Switch to:
- `groq/llama-3.3-70b-versatile` (most reliable tool calling on free tier)
- `gemini/gemini-2.0-flash` (second best)

**JSON output malformed (Week 2, structured output)** — Use a larger model:
- `groq/llama-3.3-70b-versatile` instead of 8b variants

**`litellm.exceptions.BadRequestError`** — Model doesn't support a parameter.
Try removing `tools=` argument or switching models.
