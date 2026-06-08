# Agentic AI — Production Learning Plan

**12 Weeks · Python 3.11+ · Project-First**

A structured, hands-on plan to go from LLM API basics to production-ready agentic AI systems. Every week has notes, exercises (no hints — figure it out), reference solutions, and curated resources. Each phase ends with a real project.

Works with **local LLMs (Ollama)**, **free cloud (Groq, Gemini)**, or any paid provider — swap with one `.env` change. See [`FREE_CLOUD_LLM.md`](FREE_CLOUD_LLM.md) and [`LOCAL_LLM_SETUP.md`](LOCAL_LLM_SETUP.md).

---

## 🗂 Repository Structure

```
agentic_ai_learning_plan/
├── phase1_foundations/           Weeks 1–2   · LLM APIs, Tool Use, ReAct
├── phase2_memory_rag/            Weeks 3–4   · LangGraph, RAG, Hybrid Search
├── phase3_multi_agent/           Weeks 5–6   · Orchestrators, Async Parallelism
├── phase4_production/            Weeks 7–8   · FastAPI, SSE, Observability, Guardrails
├── phase5_advanced/              Weeks 9–10  · Planning, Self-Reflection, Eval Pipelines
├── phase6_capstone/              Weeks 11–12 · MCP, Model Routing, Full-Stack Deploy
├── projects/                     22 projects — see full list below
├── guide/                        Additional topic guides
├── llm.py                        Unified LLM wrapper (local ↔ cloud, zero code change)
├── .env.example                  Copy to .env and fill in your keys
├── framework_selection_guide.md  How to choose LangChain / LangGraph / CrewAI / LlamaIndex / AutoGen
├── resources_master.md           Curated papers, courses, and reference links
├── PRODUCTION_AGENT_GUIDE.md     Production deployment patterns and checklists
├── FREE_CLOUD_LLM.md             Groq, Gemini, Cerebras, OpenRouter setup
├── LOCAL_LLM_SETUP.md            Ollama setup and model recommendations
└── SECURITY.md                   Keeping API keys safe
```

---

## 🗺 Phase Overview

| Phase | Weeks | Core Skills | Capstone Project |
|---|---|---|---|
| 1 — Foundations | 1–2 | LLM APIs, tool calling, ReAct loop | Research Assistant CLI |
| 2 — Memory & RAG | 3–4 | LangGraph, ChromaDB, hybrid search | Personal Knowledge Agent |
| 3 — Multi-Agent | 5–6 | Orchestrator pattern, asyncio fan-out | Parallel Code Reviewer |
| 4 — Production | 7–8 | FastAPI SSE, cost tracking, guardrails | Agent-as-a-Service API |
| 5 — Advanced | 9–10 | Self-reflection, Reflexion, RAGAS, DeepEval | Self-Improving Coding Agent |
| 6 — Capstone | 11–12 | MCP, model routing, Docker deploy | Full-Stack Content Pipeline |
| **Bonus** | — | Security · observability · batch · HITL · eval · customer support · data analyst · doc intelligence · WhatsApp MCP · production RAG · enterprise RAG · **LangChain · LangGraph · CrewAI · LlamaIndex · AutoGen** | Projects 7–22 |

---

## 📁 Each Week Contains

```
week_N/
├── notes.md           — concepts, patterns, code snippets, checklist
├── exercises/
│   ├── ex1_*.py       — starter code with TODOs (solve it yourself first)
│   └── ex2_*.py
└── resources/
    └── links.md       — papers, docs, courses, install commands
```

---

## 🏗 All 22 Projects

### Phase 1–6 (Core Curriculum)

Each has: `README.md` · `starter.py` (TODOs) · `solution/solution.py`

```
projects/
├── phase1_research_assistant/    Web search + ReAct → structured JSON report
├── phase2_knowledge_agent/       RAG + LangGraph + SQLite memory → chat over docs
├── phase3_code_review/           4 async agents → scored PR review report
├── phase4_agent_api/             FastAPI + SSE + cost tracking → production API
├── phase5_coding_agent/          Reflexion loop → solves failing pytest suites
└── phase6_capstone/              Async multi-agent content pipeline (full-stack)
```

### Projects 7–17 (Bonus — Raw Libraries, No Framework)

Each has: `README.md` · `starter/` (numbered TODOs) · `solution/` (full implementation)

```
├── project7_security_agent/              5-layer security: prompt injection · PII scan · Pydantic
│                                         validation · human-in-the-loop · output scan
├── project8_observability_agent/         structlog + Prometheus + OpenTelemetry +
│                                         cost-per-run dashboard
├── project9_batch_pipeline/              Fan-Out/Fan-In + Map-Reduce for 500+ items with
│                                         asyncio.Semaphore
├── project10_langgraph_agent/            LangGraph StateGraph · conditional routing ·
│                                         MemorySaver checkpointing · HITL interrupt/resume
├── project11_eval_pipeline/              Golden dataset · safety suite · tool quality ·
│                                         RAG faithfulness · latency benchmark · CI gate
├── project12_customer_support_agent/     Multi-tier triage → CRM tool use → specialist agents
│                                         → escalation → SLA enforcement · session report
├── project13_data_analyst_agent/         NL question → plan → LLM code gen → subprocess exec
│                                         → self-correct (3×) → narrative report
├── project14_document_intelligence_agent/ Batch classify → LLM+Pydantic extract → validate
│                                         → SHA-256 dedup → anomaly detect → HTML report
├── project15_whatsapp_agent/             MCP server (FastMCP + RAG) + multi-agent routing +
│                                         WhatsApp (Twilio) + Telegram + session history
├── project16_production_rag/             Offline embed pipeline · ChromaDB · BM25+vector
│                                         hybrid · LLM rerank · FastAPI (lifespan, rate limit)
│                                         · MCP server · LLM-judge eval gate · Docker + CI/CD
└── project17_enterprise_rag/             10M-doc RAG · zero hallucination · Qdrant · Kafka ·
                                          GPU embeddings · confidence scoring
```

### Projects 18–22 (Framework Showcase)

Each has: `README.md` · `GUIDE.md` (step-by-step phases) · `starter/src/` (numbered TODOs) · `solution/src/` (complete) · `.env.example`

```
├── project18_langchain_agent/    LangChain LCEL chains · ReAct AgentExecutor · custom @tools ·
│                                 ConversationBufferWindowMemory · astream_events · FAISS ·
│                                 LangSmith tracing
├── project19_langgraph_workflow/ LangGraph StateGraph · 6 nodes · interrupt() HITL ·
│                                 conditional edges · SqliteSaver checkpointer · FastAPI SSE
├── project20_crewai_pipeline/    CrewAI: Researcher→Writer→Editor→SEO Analyst ·
│                                 Pydantic task output · sequential + hierarchical process
├── project21_llamaindex_agent/   LlamaIndex IngestionPipeline + cache · VectorStoreIndex ·
│                                 SubQuestionQueryEngine · RouterQueryEngine · ReActAgent ·
│                                 SentenceTransformerRerank
└── project22_autogen_team/       AutoGen 5-agent GroupChat (PM→Architect→Developer→Tester
                                  →Reviewer) · LocalCommandLineCodeExecutor · custom speaker
                                  selection · two-agent nested chat with carryover
```

---

## 🧭 Framework Selection Guide

Not sure which framework to use for your project?

| You need… | Best fit |
|---|---|
| Composable chains, tools, basic RAG | **LangChain** (Project 18) |
| Stateful workflow, branching, human approval | **LangGraph** (Projects 10, 19) |
| Role-based specialist agents (research / content) | **CrewAI** (Project 20) |
| Document-heavy RAG with complex queries | **LlamaIndex** (Project 21) |
| Agents that write and execute code | **AutoGen** (Project 22) |
| High-throughput production API | **Raw Libraries** (Projects 16–17) |

→ Full decision flowchart, comparison table, code examples, anti-patterns: **[`framework_selection_guide.md`](framework_selection_guide.md)**

> 💰 **Reduce LLM costs by 70–90%**: See [`token_optimization_guide.md`](token_optimization_guide.md) — token counting, prompt compression, model routing, semantic caching, context management, and 5 exercises.

---

## ⚡ Quick Start

### Option A — Free cloud (Groq, recommended)

```bash
# 1. Clone and set up env
git clone https://github.com/arpitk12/agentic-ai-learning-plan.git
cd agentic-ai-learning-plan
cp .env.example .env
# Edit .env: set MODEL=groq/llama-3.3-70b-versatile and GROQ_API_KEY=...
# Get a free key at https://console.groq.com (no credit card)

# 2. Install core dependencies
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

# Phase 2–3 (RAG + hybrid search)
pip install langgraph langchain-community langchain-core
pip install chromadb sentence-transformers rank-bm25 pypdf

# Phase 3 (web search)
pip install tavily-python

# Phase 4 (API serving + observability)
pip install fastapi uvicorn structlog prometheus-client \
            opentelemetry-sdk opentelemetry-exporter-otlp

# Phase 5 (evaluation frameworks)
pip install ragas datasets          # RAGAS: RAG quality metrics
pip install deepeval                # DeepEval: 14+ LLM metrics + pytest integration
pip install langsmith               # LangSmith: datasets, evaluators, experiment versioning
pip install pytest pytest-asyncio

# Project 15 (WhatsApp agent — MCP + RAG + messaging)
pip install mcp fastapi uvicorn httpx twilio

# Projects 16–17 (Production / Enterprise RAG)
pip install qdrant-client kafka-python celery redis

# Project 18 (LangChain)
pip install langchain langchain-community langchain-huggingface \
            langchain-litellm faiss-cpu tavily-python langsmith

# Project 19 (LangGraph)
pip install langgraph langgraph-checkpoint-sqlite langchain langchain-litellm fastapi

# Project 20 (CrewAI)
pip install crewai crewai-tools tavily-python litellm

# Project 21 (LlamaIndex)
pip install llama-index llama-index-embeddings-huggingface \
            llama-index-vector-stores-faiss llama-index-llms-litellm faiss-cpu

# Project 22 (AutoGen)
pip install pyautogen docker litellm
```

---

## 📐 Recommended Pace

1. **Read `notes.md`** — understand the concepts (30 min)
2. **Do the exercises** — fill in TODOs yourself, no peeking at solutions (2–3 hrs)
3. **Read solutions** — compare your approach, note the differences (30 min)
4. **Build the project** — apply everything end-to-end (4–6 hrs)
5. **Check `resources/links.md`** — go deeper on what interested you

> ⚠️ **Don't skip Phase 4 (Production).** Most tutorials do. Real systems live or die there.

---

## 🛑 Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| No `max_steps` guard in tool loops | Always add one — agents can loop forever |
| Logging raw user input | Redact PII before any log write |
| Always using the biggest model | Route by complexity — save 80%+ on cost |
| No eval suite | LLM-as-judge + golden dataset catches regressions before deploy |
| Interpolating user input into system prompts | Wrap in `<user_input>` tags, instruct model to treat as data |
| Not handling parallel tool calls | Loop over ALL content blocks, not just the first |
| Only testing happy-path cases | Include adversarial, edge-case, and safety test cases |
| Using exact-match eval for open-ended tasks | Use LLM-as-judge or DeepEval metrics instead |
| Skipping multi-turn evaluation | Test full conversation sessions, not just individual turns |
| Adding a framework to a 10-line script | Raw `openai` SDK is simpler — frameworks have overhead |

---

## 🔒 Security

- **Never commit `.env`** — it is gitignored; only `.env.example` (no real keys) is committed
- See [`SECURITY.md`](SECURITY.md) for full guidance on secrets management, PII handling, and prompt injection defence
- **Never hardcode API keys** in `.py` files — always use `os.getenv()`
- **Never paste `.env` contents into any LLM chat** (Copilot, ChatGPT, etc.)

See [`SECURITY.md`](SECURITY.md) for the full guide including key rotation and pre-push checklists.
