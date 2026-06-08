# Agentic AI — Production Learning Plan

**12 Weeks · Python 3.11+ · Project-First**

A structured, hands-on plan to go from LLM API basics to production-ready agentic AI systems. Every week has notes, exercises (no hints — figure it out), reference solutions, and curated resources. Each phase ends with a real project.

Works with **local LLMs (Ollama)**, **free cloud (Groq, Gemini)**, or any paid provider — swap with one `.env` change. See [`FREE_CLOUD_LLM.md`](FREE_CLOUD_LLM.md) and [`LOCAL_LLM_SETUP.md`](LOCAL_LLM_SETUP.md).

---

## 🗂 Structure

```
agentic_ai_learning_plan/
├── phase1_foundations/          Weeks 1–2   · LLM APIs, Tool Use, ReAct
├── phase2_memory_rag/           Weeks 3–4   · LangGraph, RAG, Hybrid Search
├── phase3_multi_agent/          Weeks 5–6   · Orchestrators, Async Parallelism
├── phase4_production/           Weeks 7–8   · FastAPI, SSE, Observability, Guardrails
├── phase5_advanced/             Weeks 9–10  · Planning, Self-Reflection, Eval Pipelines
├── phase6_capstone/             Weeks 11–12 · MCP, Model Routing, Full-Stack Deploy
├── projects/                    6 projects — one per phase, solutions kept separate
├── llm.py                       Unified LLM wrapper (local ↔ cloud, zero code change)
├── .env.example                 Copy to .env and fill in your keys
├── FREE_CLOUD_LLM.md            Groq, Gemini, Cerebras, OpenRouter setup
├── LOCAL_LLM_SETUP.md           Ollama setup and model recommendations
└── SECURITY.md                  Keeping API keys safe
```

---

## 🗺 Phase Overview

| Phase | Weeks | Core Skills | Project |
|---|---|---|---|
| 1 — Foundations | 1–2 | LLM APIs, tool calling, ReAct loop | Research Assistant CLI |
| 2 — Memory & RAG | 3–4 | LangGraph, ChromaDB, hybrid search | Personal Knowledge Agent |
| 3 — Multi-Agent | 5–6 | Orchestrator pattern, asyncio fan-out | Parallel Code Reviewer |
| 4 — Production | 7–8 | FastAPI SSE, cost tracking, guardrails | Agent-as-a-Service API |
| 5 — Advanced | 9–10 | Self-reflection, Reflexion, RAGAS, DeepEval, LangSmith, safety & perf eval | Self-Improving Coding Agent |
| 6 — Capstone | 11–12 | MCP, model routing, Docker deploy | Full-Stack Content Pipeline |
| **Bonus** | — | Security, observability, batch pipelines, LangGraph HITL, eval pipeline, customer support, data analyst, document intelligence, WhatsApp MCP agent, **production RAG** | Projects 7–16 |

---

## 📁 Each Week Contains

```
week_N/
├── notes.md           — concepts, patterns, code snippets, checklist
├── exercises/
│   ├── ex1_*.py       — starter code with TODOs (no hints — solve it yourself)
│   ├── ex2_*.py
│   └── solutions/
│       ├── sol1_*.py  — full working implementations
│       └── sol2_*.py
└── resources/
    └── links.md       — papers, docs, courses, install commands
```

---

## 🏗 Projects

```
projects/
├── phase1_research_assistant/    Web search + ReAct → structured JSON report
├── phase2_knowledge_agent/       RAG + LangGraph + SQLite memory → chat over docs
├── phase3_code_review/           4 async agents → scored PR review report
├── phase4_agent_api/             FastAPI + SSE + cost tracking → production API
├── phase5_coding_agent/          Reflexion loop → solves failing pytest suites
├── phase6_capstone/              Async multi-agent content pipeline (full-stack)
├── project7_security_agent/      5-layer security: injection · PII scan · Pydantic validation · HITL · output scan
├── project8_observability_agent/ structlog + Prometheus + OpenTelemetry + cost-per-run dashboard
├── project9_batch_pipeline/      Fan-Out / Fan-In + Map-Reduce for 500+ items with asyncio.Semaphore
├── project10_langgraph_agent/    LangGraph StateGraph · conditional routing · MemorySaver checkpointing · HITL
├── project11_eval_pipeline/      End-to-end eval: golden dataset · safety suite · tool quality · RAG faithfulness
│                                 · latency benchmark · multi-turn · JSON + HTML report · CI gate
├── project12_customer_support/   Multi-tier triage → CRM tool use → specialist agents → escalation → SLA · session report
├── project13_data_analyst/       NL question → plan → LLM code gen → subprocess exec → self-correct (3×) → narrative report
├── project14_doc_intelligence/   Batch classify → LLM+Pydantic extract → validate → SHA-256 dedup → anomaly detect → HTML report
├── project15_whatsapp_agent/     MCP server (FastMCP + RAG) + multi-agent routing + WhatsApp (Twilio) + Telegram + session history
├── project16_production_rag/     Production RAG: offline embed pipeline · ChromaDB · BM25+vector hybrid · LLM rerank
                                  · multi-agent orchestrator · FastAPI (lifespan, middleware, rate limit) · MCP server
                                  · LLM-judge eval gate · Docker + CI/CD (GitHub Actions)
├── project17_enterprise_rag/     Enterprise RAG at 10M docs · zero hallucination · Qdrant · Kafka · GPU embeddings
├── project18_langchain_agent/    LangChain LCEL · ReAct agent · custom tools · ConversationBufferMemory · astream_events
├── project19_langgraph_workflow/ LangGraph StateGraph · interrupt() HITL · conditional edges · SqliteSaver · FastAPI SSE
├── project20_crewai_pipeline/    CrewAI 4-agent team (Researcher→Writer→Editor→SEO) · Pydantic output · hierarchical crew
├── project21_llamaindex_agent/   LlamaIndex IngestionPipeline · SubQuestionQueryEngine · RouterQueryEngine · ReActAgent
└── project22_autogen_team/       AutoGen 5-agent GroupChat · custom speaker selection · code execution · nested chat
```

Each project has:
- `README.md` — requirements, architecture diagram, milestones
- `starter/` — scaffold with numbered TODOs
- `solution/` — full working implementation with config + `.env.example`

> 📖 **Not sure which framework to use?** See [`framework_selection_guide.md`](framework_selection_guide.md)

---

## 🧭 Framework Selection Guide

| You need… | Use |
|---|---|
| Composable chains, tools, basic RAG | **LangChain** (Project 18) |
| Stateful workflow, branching, human approval | **LangGraph** (Project 19) |
| Role-based specialist agents (research/content) | **CrewAI** (Project 20) |
| Document-heavy RAG with complex queries | **LlamaIndex** (Project 21) |
| Agents that write and execute code | **AutoGen** (Project 22) |
| High-throughput production API | **Raw Libraries** (Projects 16-17) |

→ Full decision flowchart, comparison table, anti-patterns, and real-world scenarios: **[`framework_selection_guide.md`](framework_selection_guide.md)**

---

## 🏗 Old Projects structure (Phase 1–6)
- `README.md` — requirements, architecture, milestones, expected output
- `starter.py` — scaffold with TODO sections (6–8 per project)
- `solution/solution.py` — full working implementation

---

## ⚡ Quick Start

### Option A — Free cloud (Groq, recommended)

```bash
# 1. Clone and set up env
cp .env.example .env
# Edit .env: set MODEL=groq/llama-3.3-70b-versatile and GROQ_API_KEY=...
# Get a free key at https://console.groq.com (no credit card)

# 2. Install dependencies
pip install litellm python-dotenv

# 3. Run your first exercise
python phase1_foundations/week1_llm_api/exercises/ex1_multiturn_chatbot.py
```

### Option B — Local (Ollama, fully offline)

```bash
brew install ollama
ollama pull llama3.2
ollama serve

cp .env.example .env   # default MODEL=ollama/llama3.2 already set
pip install litellm python-dotenv
python phase1_foundations/week1_llm_api/exercises/ex1_multiturn_chatbot.py
```

See [`FREE_CLOUD_LLM.md`](FREE_CLOUD_LLM.md) for Gemini, Cerebras, and OpenRouter options.

---

## 📦 Full Install Reference

```bash
# Core (all phases)
pip install litellm python-dotenv pydantic httpx

# Phase 2–3 (LangGraph + RAG)
pip install langgraph langchain-community langchain-core
pip install chromadb sentence-transformers rank-bm25 pypdf

# Phase 3 (web search)
pip install tavily-python

# Phase 4 (API serving + observability)
pip install fastapi uvicorn structlog prometheus-client opentelemetry-sdk opentelemetry-exporter-otlp

# Phase 5 (evaluation frameworks)
pip install ragas datasets          # RAGAS: RAG quality metrics
pip install deepeval                # DeepEval: 14+ LLM metrics + pytest integration
pip install langsmith               # LangSmith: datasets, evaluators, experiment versioning
pip install pytest pytest-asyncio   # behavioural testing

# Project 15 (WhatsApp agent — MCP + RAG + messaging)
pip install mcp fastapi uvicorn httpx   # MCP SDK + API server
# Optional: real WhatsApp integration
pip install twilio

# Week 3 LangGraph exercises specifically
pip install langchain-community  # provides ChatLiteLLM
```

---

## 📐 Recommended Pace

1. **Read `notes.md`** — understand the concepts (30 min)
2. **Do exercises** — fill in TODOs yourself, no peeking at solutions (2–3 hrs)
3. **Read solutions** — compare your approach, note differences (30 min)
4. **Build the project** — apply everything end-to-end (4–6 hrs)
5. **Check `resources/links.md`** — go deeper on what interested you

> ⚠️ **Don't skip Phase 4 (production).** Most tutorials do. Real systems live or die there.

---

## 🛑 Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| No `max_steps` guard in tool loops | Always add one — agents can loop forever |
| Logging raw user input | Redact PII before any log write |
| Always using the biggest model | Route by complexity — save 80%+ on cost |
| No eval suite | LLM-as-judge + golden dataset catches regressions before deployment |
| Interpolating user input into system prompts | Wrap in `<user_input>` tags, instruct model to ignore instructions inside |
| Not handling parallel tool calls | Loop over ALL content blocks, not just the first |
| Evaluating only on happy-path cases | Include adversarial, edge-case, and safety test cases |
| Using exact-match eval for open-ended tasks | Use LLM-as-judge or DeepEval metrics instead |
| Skipping multi-turn evaluation | Test full conversation sessions, not just individual turns |

---

## 🔒 Security

- **Never commit `.env`** — it is gitignored; only `.env.example` (no real keys) is committed
- **Never hardcode API keys** in `.py` files — always use `os.getenv()`
- **Never paste `.env` contents into any LLM chat** (Copilot, ChatGPT, etc.)

See [`SECURITY.md`](SECURITY.md) for the full guide including key rotation and pre-push checklists.
