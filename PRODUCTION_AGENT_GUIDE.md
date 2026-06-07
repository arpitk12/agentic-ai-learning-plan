# Production Agent Guide — End-to-End Reference

A comprehensive reference for building, deploying, and operating production-grade AI agents.

> **This guide has been split into focused section files for easier reading.**
> Jump directly to the topic you need using the table below.

---

## 📚 Table of Contents

| # | Section | Size | What's inside |
|---|---------|------|---------------|
| 1 | [**The Complete Agentic Stack**](guide/01_agentic_stack.md) | ~130 lines | Layer diagram, mental model, 5-minute orientation |
| 2 | [**Framework & Tool Selection Guide**](guide/02_framework_selection.md) | ~1,150 lines | LiteLLM, LangGraph, MCP, every tool with code examples |
| 3 | [**RAG Architecture Deep Dive**](guide/03_rag_architecture.md) | ~580 lines | Chunking, embeddings, reranking, hybrid search, HyDE |
| 4 | [**Multi-Agent Design Patterns**](guide/04_multi_agent.md) | ~540 lines | Orchestrator, fan-out, critic loop, Reflexion, LATS |
| 5 | [**Vector Search Reference**](guide/05_vector_search.md) | ~220 lines | ChromaDB vs Qdrant vs Pinecone, index config, HNSW |
| 6 | [**Production Checklist**](guide/06_production_checklist.md) | ~65 lines | Pre-launch verification — copy this to your PR |
| 7 | [**Cost Optimization Strategies**](guide/07_cost_optimization.md) | ~230 lines | Model routing, caching, token budgets, batching |
| 8 | [**Security Hardening**](guide/08_security.md) | ~185 lines | Prompt injection, PII, secrets, OWASP LLM Top 10 |
| 9 | [**Observability Stack**](guide/09_observability.md) | ~320 lines | structlog, Prometheus, OpenTelemetry, cost tracking |
| 10 | [**Deployment Playbook**](guide/10_deployment.md) | ~355 lines | Docker, FastAPI, Celery, Railway, Modal, CI/CD |
| 11 | [**Exercises Index**](guide/11_exercises_index.md) | ~230 lines | All exercises mapped by topic + all 15 projects |
| 12 | [**Agent Evaluation & Quality Assurance**](guide/12_evaluation.md) | ~555 lines | RAGAS, DeepEval, LangSmith, safety eval, CI gate |

---

## ⚡ Quick-Start by Goal

| I want to… | Start here |
|------------|-----------|
| Understand the full agent architecture | [§1 Agentic Stack](guide/01_agentic_stack.md) |
| Pick the right LLM / framework / tool | [§2 Framework Selection](guide/02_framework_selection.md) |
| Build a RAG pipeline | [§3 RAG Architecture](guide/03_rag_architecture.md) |
| Design a multi-agent system | [§4 Multi-Agent Patterns](guide/04_multi_agent.md) |
| Choose a vector database | [§5 Vector Search](guide/05_vector_search.md) |
| Ship to production today | [§6 Production Checklist](guide/06_production_checklist.md) |
| Cut my LLM bill | [§7 Cost Optimization](guide/07_cost_optimization.md) |
| Harden against prompt injection | [§8 Security](guide/08_security.md) |
| Add tracing and metrics | [§9 Observability](guide/09_observability.md) |
| Deploy with Docker / Railway | [§10 Deployment](guide/10_deployment.md) |
| Find the right exercise | [§11 Exercises Index](guide/11_exercises_index.md) |
| Evaluate agent quality | [§12 Evaluation & QA](guide/12_evaluation.md) |

---

## 🗂 Projects (15 Total)

| Project | What you build |
|---------|---------------|
| [phase1 — Research Assistant](projects/phase1_research_assistant/) | Web search + ReAct → JSON report |
| [phase2 — Knowledge Agent](projects/phase2_knowledge_agent/) | RAG + LangGraph + SQLite memory |
| [phase3 — Code Review](projects/phase3_code_review/) | 4 async agents → scored PR review |
| [phase4 — Agent API](projects/phase4_agent_api/) | FastAPI + SSE + Celery + cost tracking |
| [phase5 — Coding Agent](projects/phase5_coding_agent/) | Reflexion loop → solves failing tests |
| [phase6 — Capstone](projects/phase6_capstone/) | Async multi-agent content pipeline |
| [project7 — Security Agent](projects/project7_security_agent/) | 5-layer security: injection · PII · HITL |
| [project8 — Observability Agent](projects/project8_observability_agent/) | structlog + Prometheus + OTel dashboard |
| [project9 — Batch Pipeline](projects/project9_batch_pipeline/) | Fan-out/Fan-in for 500+ items |
| [project10 — LangGraph Agent](projects/project10_langgraph_agent/) | StateGraph + HITL + MemorySaver |
| [project11 — Eval Pipeline](projects/project11_eval_pipeline/) | End-to-end eval + HTML report + CI gate |
| [project12 — Customer Support](projects/project12_customer_support_agent/) | Multi-tier triage → CRM tools → SLA |
| [project13 — Data Analyst](projects/project13_data_analyst_agent/) | Code gen → subprocess → self-correct |
| [project14 — Doc Intelligence](projects/project14_document_intelligence_agent/) | Classify → extract → dedup → report |
| [project15 — WhatsApp Agent](projects/project15_whatsapp_agent/) | MCP + RAG + WhatsApp/Telegram |
| [project16 — Production RAG](projects/project16_production_rag/) | Modular RAG · hybrid search · Docker · CI/CD · LLM-judge eval gate |

---

## 🔑 Key Concepts at a Glance

| Concept | Guide Section |
|---------|--------------|
| ReAct loop | [§2.2](guide/02_framework_selection.md) |
| Pydantic structured output | [§2.1](guide/02_framework_selection.md) |
| LangGraph StateGraph + HITL | [§2.4](guide/02_framework_selection.md) |
| MCP client + server | [§2.16](guide/02_framework_selection.md) |
| Chunking strategies | [§3.2](guide/03_rag_architecture.md) |
| Hybrid search (BM25 + vector) | [§3.5](guide/03_rag_architecture.md) |
| Orchestrator pattern | [§4.2](guide/04_multi_agent.md) |
| Self-reflection / Reflexion | [§4.5](guide/04_multi_agent.md) |
| Model routing by complexity | [§7.1](guide/07_cost_optimization.md) |
| Prompt injection defence | [§8.2](guide/08_security.md) |
| OpenTelemetry tracing | [§9.3](guide/09_observability.md) |
| LLM-as-judge evaluation | [§12.3](guide/12_evaluation.md) |
| Safety adversarial testing | [§12.5](guide/12_evaluation.md) |
| CI/CD quality gate | [§12.8](guide/12_evaluation.md) |

---

## 📦 Install Reference

```bash
# Core
pip install litellm python-dotenv pydantic httpx

# RAG + Memory
pip install langgraph langchain-community langchain-core
pip install chromadb sentence-transformers rank-bm25 pypdf

# Production API
pip install fastapi uvicorn celery redis structlog
pip install prometheus-client opentelemetry-sdk opentelemetry-exporter-otlp

# Evaluation
pip install ragas datasets deepeval langsmith pytest pytest-asyncio

# WhatsApp Agent (project 15)
pip install mcp fastapi uvicorn httpx twilio
```

---

*Last updated: June 2026 · 4,700+ lines across 12 section files · Built with LiteLLM + `llm.py` on `gemini/gemini-2.0-flash`*
