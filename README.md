# Agentic AI — Production Learning Plan
## 12 Weeks · Python Prerequisite · Project-First

````markdown
# Agentic AI — Production Learning Plan
## 12 Weeks · Python Prerequisite · Project-First

```
agentic_ai_learning_plan/
├── phase1_foundations/          Weeks 1–2  · LLM APIs, Tool Use, ReAct
├── phase2_memory_rag/           Weeks 3–4  · LangGraph, RAG, Hybrid Search
├── phase3_multi_agent/          Weeks 5–6  · Orchestrators, Async Parallelism
├── phase4_production/           Weeks 7–8  · FastAPI, SSE, Observability, Guardrails
├── phase5_advanced/             Weeks 9–10 · Planning, Self-Reflection, Eval Pipelines
├── phase6_capstone/             Weeks 11–12· MCP, Model Routing, Full-Stack Deploy
└── projects/                    6 projects — one per phase, solutions kept separate
```

---

## 🗺 Phase Overview

| Phase | Weeks | Core Skills | Project |
|---|---|---|---|
| 1 — Foundations | 1-2 | Anthropic SDK, tool calling, ReAct loop | Research Assistant CLI |
| 2 — Memory & RAG | 3-4 | LangGraph, ChromaDB, hybrid search | Personal Knowledge Agent |
| 3 — Multi-Agent | 5-6 | Orchestrator pattern, asyncio fan-out | Parallel Code Reviewer |
| 4 — Production | 7-8 | FastAPI SSE, cost tracking, guardrails | Agent-as-a-Service API |
| 5 — Advanced | 9-10 | Self-reflection, Reflexion, LLM-as-judge | Self-Improving Coding Agent |
| 6 — Capstone | 11-12 | MCP, model routing, Docker deploy | Full-Stack Content Pipeline |

---

## 📁 Each Week Contains

```
week_N/
├── notes.md           — concepts, patterns, code snippets, checklist
├── exercises/
│   ├── ex1_*.py       — starter code (TODOs for you to fill in)
│   ├── ex2_*.py
│   └── solutions/
│       ├── sol1_*.py  — full working implementations
│       └── sol2_*.py
└── resources/
    └── links.md       — papers, docs, courses, install commands
```

---

## 🏗 Projects (one per phase)

```
projects/
├── phase1_research_assistant/    Web search + ReAct → structured JSON report
├── phase2_knowledge_agent/       RAG + LangGraph + SQLite memory → chat over docs
├── phase3_code_review/           4 async agents → scored PR review report
├── phase4_agent_api/             FastAPI + SSE + cost tracking → production API
├── phase5_coding_agent/          Reflexion loop → solves failing pytest suites
└── phase6_capstone/              Async multi-agent content pipeline (full-stack)
```

Each project:
- `README.md` — requirements, architecture diagram, setup, eval criteria
- `starter.py` — scaffold with TODOs (where it exists)
- `solution/solution.py` — full working implementation

---

## ⚡ Quick Start

```bash
pip install anthropic openai langchain langgraph langchain-anthropic \
            langchain-core chromadb sentence-transformers rank-bm25 \
            fastapi uvicorn pydantic structlog python-dotenv httpx \
            pypdf tavily-python
```

Create `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...      # free tier at tavily.com
GITHUB_TOKEN=ghp_...         # for project 3 (GitHub PR fetcher)
```

---

## 📐 Recommended Pace

1. **Read `notes.md`** — understand the concepts (30 min)
2. **Do exercises** — fill in TODOs yourself before checking solutions (2–3 hrs)
3. **Read solutions** — compare your approach (30 min)
4. **Build project** — apply everything, deadline is end of the week (4–6 hrs)
5. **Check resources/links.md** — go deeper on what interested you

> ⚠️ Don't skip Phase 4 (production). Most tutorials do. Real systems live or die there.

---

## 🛑 Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| No `max_steps` guard in tool loops | Always add one — agents can loop forever |
| Logging raw user input | Redact PII before any log write |
| Always using the biggest model | Route by complexity — save 80%+ on cost |
| No eval suite | LLM-as-judge on 20 golden cases catches regressions |
| Interpolating user input into system prompts | Wrap in `<user_input>` tags, instruct model to ignore instructions inside |
| Not handling parallel tool calls | Loop over ALL content blocks, not just the first |

---

## 📦 Full Install Reference

```bash
# Phase 1-2
pip install anthropic python-dotenv pydantic httpx

# Phase 2-3
pip install langgraph langchain-anthropic langchain-core langsmith

# Phase 2 RAG
pip install chromadb sentence-transformers rank-bm25 pypdf

# Phase 3
pip install tavily-python  # web search

# Phase 4
pip install fastapi uvicorn structlog opentelemetry-sdk

# Phase 5
pip install pytest deepeval

# Phase 6
pip install litellm  # model routing + unified cost tracking
```
````

## Quick Start
Each phase folder contains:
- `week_N/notes.md` — Topics, concepts, exercises
- `week_N/exercises/` — Starter code to practice with
- `week_N/resources/links.md` — Curated reading & videos

Each project folder contains:
- `README.md` — Project brief, requirements, hints
- `starter.py` — Scaffold to get you going
- `solution/` — Full working implementation

## Prerequisites
- Python 3.11+
- `pip install anthropic openai python-dotenv pydantic`
- Set `ANTHROPIC_API_KEY` in a `.env` file

## Recommended Pace
Complete all exercises before moving to the project. Don't skip Phase 4 (production) — most tutorials do, and it's where real systems live or die.
