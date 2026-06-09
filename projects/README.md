# Projects — Classification Index

> 40 projects across 9 learning tracks. Use this as your map.  
> Each group has a clear **what you learn** summary and a **suggested order**.

---

## Quick Reference — All 40 by Group

| # | Project | Group |
|---|---|---|
| P1 | phase1_research_assistant | 1 — Core Curriculum |
| P2 | phase2_knowledge_agent | 1 — Core Curriculum |
| P3 | phase3_code_review | 1 — Core Curriculum |
| P4 | phase4_agent_api | 1 — Core Curriculum |
| P5 | phase5_coding_agent | 1 — Core Curriculum |
| P6 | phase6_capstone | 1 — Core Curriculum |
| 7  | project7_security_agent | 2 — Safety & Guardrails |
| 29 | project29_guardrails | 2 — Safety & Guardrails |
| 8  | project8_observability_agent | 3 — Observability & Cost |
| 11 | project11_eval_pipeline | 4 — Evaluation |
| 28 | project28_ab_testing | 4 — Evaluation |
| 9  | project9_batch_pipeline | 5 — Data Processing |
| 13 | project13_data_analyst_agent | 5 — Data Processing |
| 14 | project14_document_intelligence_agent | 5 — Data Processing |
| 16 | project16_production_rag | 6 — RAG & Retrieval |
| 17 | project17_enterprise_rag | 6 — RAG & Retrieval |
| 30 | project30_graph_rag | 6 — RAG & Retrieval |
| 10 | project10_langgraph_agent | 7 — Agent Frameworks |
| 18 | project18_langchain_agent | 7 — Agent Frameworks |
| 19 | project19_langgraph_workflow | 7 — Agent Frameworks |
| 20 | project20_crewai_pipeline | 7 — Agent Frameworks |
| 21 | project21_llamaindex_agent | 7 — Agent Frameworks |
| 22 | project22_autogen_team | 7 — Agent Frameworks |
| 12 | project12_customer_support_agent | 8 — Real-World Applications |
| 15 | project15_whatsapp_agent | 8 — Real-World Applications |
| 23 | project23_enterprise_architect | 8 — Real-World Applications |
| 24 | project24_finetune_agent | 9 — Advanced LLM Techniques |
| 25 | project25_memory_agent | 9 — Advanced LLM Techniques |
| 26 | project26_multimodal_agent | 9 — Advanced LLM Techniques |
| 27 | project27_dspy_optimizer | 9 — Advanced LLM Techniques |
| 31 | project31_sandboxed_tools | 9 — Advanced LLM Techniques |
| 35 | project35_reasoning | 9 — Advanced LLM Techniques |
| 32 | project32_a2a_protocol | 10 — Production Infrastructure |
| 33 | project33_multitenant | 10 — Production Infrastructure |
| 34 | project34_resilience | 10 — Production Infrastructure |
| 36 | project36_enterprise_multimodal | 10 — Production Infrastructure |
| 37 | project37_topology_benchmark | 11 — System Design Applied |
| 38 | project38_context_engine | 11 — System Design Applied |
| 39 | project39_async_platform | 11 — System Design Applied |
| 40 | project40_scale_loadtest | 11 — System Design Applied |

---

## Group 1 — Core Curriculum

> **What you learn**: The complete agentic AI foundation — from first LLM call to a deployed multi-agent system. Do these first, in order.

| Project | What you build | Core skill |
|---|---|---|
| [phase1_research_assistant](phase1_research_assistant/README.md) | Web search + ReAct → structured JSON report | ReAct loop, tool calling, streaming |
| [phase2_knowledge_agent](phase2_knowledge_agent/README.md) | RAG + LangGraph + SQLite memory → chat over docs | RAG, vector DB, multi-turn state |
| [phase3_code_review](phase3_code_review/README.md) | 4 async agents → scored PR review report | Async agents, fan-out, asyncio |
| [phase4_agent_api](phase4_agent_api/README.md) | FastAPI + SSE + cost tracking → production API | API serving, SSE, token cost tracking |
| [phase5_coding_agent](phase5_coding_agent/README.md) | Reflexion loop → solves failing pytest suites | Self-reflection, self-correction |
| [phase6_capstone](phase6_capstone/README.md) | Async multi-agent content pipeline (full-stack) | End-to-end system, all Phase 1–5 skills |

**Suggested order**: P1 → P2 → P3 → P4 → P5 → P6  
**Time estimate**: ~30 hours total (5 hrs/project)

---

## Group 2 — Safety & Guardrails

> **What you learn**: How to make agents safe — detect prompt injection, scrub PII, enforce content policies, add human approval gates, and apply production-grade guardrail libraries.

| Project | What you build | Core skill |
|---|---|---|
| [project7_security_agent](project7_security_agent/README.md) | 5-layer security stack: injection detection · PII scan · Pydantic validation · HITL approval · output scan | Defense-in-depth for agents |
| [project29_guardrails](project29_guardrails/README.md) | 4-layer safety pipeline: regex injection · PII anonymizer · Llama Guard (14 hazard categories) · NeMo Colang rails | Production guardrail libraries |

**Key concepts**: OWASP LLM Top 10, prompt injection taxonomy, PII detection (presidio), Llama Guard, NeMo Guardrails, human-in-the-loop interrupt/resume  
**Prerequisite**: Group 1 complete

---

## Group 3 — Observability & Cost

> **What you learn**: How to see what your agent is doing, measure cost precisely, set budget limits, trace every LLM call end-to-end, and alert on regressions.

| Project | What you build | Core skill |
|---|---|---|
| [project8_observability_agent](project8_observability_agent/README.md) | structlog JSON logs · Prometheus metrics · OpenTelemetry traces · cost-per-run dashboard | Production observability stack |

**Key concepts**: Structured logging, metrics (p50/p95/p99), distributed traces, cost attribution per run, alerting thresholds  
**Prerequisite**: Group 1 complete  
**Also see**: `phase4_production/week8_observability/exercises/` — 5 exercises on logging, cost tracking, guardrails, OpenTelemetry, token optimization

---

## Group 4 — Evaluation

> **What you learn**: How to rigorously measure agent quality — beyond "it seems to work" — using golden datasets, RAG-specific metrics, LLM-as-judge, statistical A/B tests, and automated CI gates.

| Project | What you build | Core skill |
|---|---|---|
| [project11_eval_pipeline](project11_eval_pipeline/README.md) | Golden dataset · safety suite · tool quality · RAG faithfulness · latency benchmark · CI gate | Systematic agent evaluation |
| [project28_ab_testing](project28_ab_testing/README.md) | Hash-based traffic split · shadow mode · chi-square significance · Bayesian Beta test · MLflow registry | Statistical model comparison |

**Key concepts**: RAGAS (faithfulness, relevancy, precision, recall), DeepEval metrics, LLM-as-judge, golden datasets, null hypothesis testing, shadow deployment  
**Prerequisite**: Group 1 + Group 6 (RAG) for project 11

---

## Group 5 — Data Processing

> **What you learn**: Using agents to process large volumes of data — batch jobs, parallel pipelines, NL-to-SQL, document extraction, and self-correcting code execution.

| Project | What you build | Core skill |
|---|---|---|
| [project9_batch_pipeline](project9_batch_pipeline/README.md) | Fan-Out/Fan-In + Map-Reduce for 500+ items with asyncio.Semaphore | High-throughput async batch processing |
| [project13_data_analyst_agent](project13_data_analyst_agent/README.md) | NL question → plan → LLM code gen → subprocess exec → self-correct (3×) → narrative report | NL-to-SQL, code execution, self-correction |
| [project14_document_intelligence_agent](project14_document_intelligence_agent/README.md) | Batch classify → extract (Pydantic) → validate → SHA-256 dedup → anomaly detect → HTML report | Document ETL pipeline |

**Key concepts**: asyncio.Semaphore rate limiting, Map-Reduce for LLMs, sandboxed code execution, structured data extraction with Pydantic  
**Prerequisite**: Group 1 (project 9 can be done after P3)

---

## Group 6 — RAG & Retrieval

> **What you learn**: Retrieval-Augmented Generation at every level — from basic ChromaDB + BM25 hybrid search to 10M-document Kafka-ingested enterprise RAG to multi-hop knowledge graph retrieval.

| Project | What you build | Core skill |
|---|---|---|
| [project16_production_rag](project16_production_rag/README.md) | Offline embed pipeline · BM25+vector hybrid · LLM rerank · FastAPI + rate limit · MCP server · LLM-judge eval · Docker | Production RAG system |
| [project17_enterprise_rag](project17_enterprise_rag/README.md) | 10M-doc RAG · Qdrant · Kafka streaming ingest · GPU embeddings · confidence scoring | Enterprise-scale RAG |
| [project30_graph_rag](project30_graph_rag/README.md) | spaCy NER → Neo4j knowledge graph → LLM Cypher generation → hybrid graph+vector retrieval | Graph RAG for multi-hop questions |

**Key concepts**: BM25 + vector hybrid (RRF fusion), cross-encoder reranking, parent-child chunking, Qdrant collections, Kafka consumer groups, Neo4j Cypher, multi-hop reasoning  
**Prerequisite**: Group 1 (phase2 covers RAG basics)

---

## Group 7 — Agent Frameworks

> **What you learn**: How the major agent frameworks work under the hood — their strengths, weaknesses, and when to use each. Build the same conceptual agent in each framework; compare the code.

| Project | Framework | What you build | When to use it |
|---|---|---|---|
| [project10_langgraph_agent](project10_langgraph_agent/README.md) | LangGraph | StateGraph · conditional edges · HITL interrupt/resume · SqliteSaver checkpoints | Stateful workflows with branching |
| [project18_langchain_agent](project18_langchain_agent/README.md) | LangChain | LCEL chains · ReAct AgentExecutor · custom tools · FAISS · LangSmith | Composable chains, basic RAG |
| [project19_langgraph_workflow](project19_langgraph_workflow/README.md) | LangGraph + FastAPI | 6-node graph · interrupt() HITL · FastAPI SSE streaming | Production LangGraph API |
| [project20_crewai_pipeline](project20_crewai_pipeline/README.md) | CrewAI | Researcher→Writer→Editor→SEO pipeline · Pydantic task output | Role-based specialist agents |
| [project21_llamaindex_agent](project21_llamaindex_agent/README.md) | LlamaIndex | IngestionPipeline + VectorStoreIndex · SubQuestion + Router engine · ReActAgent + Rerank | Document-heavy RAG with complex queries |
| [project22_autogen_team](project22_autogen_team/README.md) | AutoGen | 5-agent GroupChat (PM→Arch→Dev→Tester→Reviewer) · code executor | Agents that write and run code |

**Suggested order**: 10 → 18 → 19 → 20 → 21 → 22  
**Also see**: [`framework_selection_guide.md`](../framework_selection_guide.md) — decision flowchart

---

## Group 8 — Real-World Applications

> **What you learn**: Building agents for specific real-world domains — customer support, messaging platforms, and enterprise compliance. These combine many earlier skills into a focused vertical.

| Project | What you build | Core skill |
|---|---|---|
| [project12_customer_support_agent](project12_customer_support_agent/README.md) | Multi-tier triage → CRM tools → specialist handoff → escalation → SLA enforcement | Customer support automation |
| [project15_whatsapp_agent](project15_whatsapp_agent/README.md) | MCP server (FastMCP) + RAG + multi-agent routing + WhatsApp (Twilio) + Telegram + session history | Messaging platform agent with MCP |
| [project23_enterprise_architect](project23_enterprise_architect/README.md) | PydanticAI typed contracts · LangGraph 7-node compliance workflow · MCP servers · Langfuse · AWS AgentCore · hash-chained audit trail | Enterprise compliance automation |

**Prerequisite**: Group 1, Group 6 (RAG), Group 7 (frameworks) recommended before project23

---

## Group 9 — Advanced LLM Techniques

> **What you learn**: Techniques that go beyond standard prompting — fine-tuning your own model, multi-modal input, prompt optimization with DSPy, sandboxed code execution, and advanced reasoning strategies.

| Project | What you build | Core skill |
|---|---|---|
| [project24_finetune_agent](project24_finetune_agent/README.md) | QLoRA fine-tune llama-3.2-3B on synthetic compliance data · DPO alignment · vLLM serving · 95% inference cost reduction | Model fine-tuning + serving |
| [project25_memory_agent](project25_memory_agent/README.md) | Mem0 all 4 memory types (episodic · semantic · procedural · user profile) · memory consolidation · multi-user isolation | Long-term agent memory |
| [project26_multimodal_agent](project26_multimodal_agent/README.md) | PDF layout extraction · GPT-4V chart analysis · Whisper audio · multi-modal ChromaDB RAG · cross-modality QA | Multi-modal RAG |
| [project27_dspy_optimizer](project27_dspy_optimizer/README.md) | DSPy Signatures + ChainOfThought · BootstrapFewShot · MIPROv2 instruction optimization · +18% accuracy | Automatic prompt optimization |
| [project31_sandboxed_tools](project31_sandboxed_tools/README.md) | E2B cloud sandbox · Docker-in-Docker · reversibility classifier · tool execution audit log | Safe code execution |
| [project35_reasoning](project35_reasoning/README.md) | Tree of Thought BFS (depth=3, breadth=3) · self-consistency majority vote · complexity router · MCTS planning | Advanced reasoning strategies |

**Prerequisite**: Group 1 complete; project24 requires GPU access (or Colab)

---

## Group 10 — Production Infrastructure

> **What you learn**: The infrastructure that makes agents production-ready at scale — multi-tenancy, resilience patterns, agent-to-agent protocols, and a full enterprise-grade capstone combining everything.

| Project | What you build | Core skill |
|---|---|---|
| [project32_a2a_protocol](project32_a2a_protocol/README.md) | Google A2A agent cards · cross-framework delegation · JWT service auth · streaming tasks · 3-agent chain | Agent interoperability standard |
| [project33_multitenant](project33_multitenant/README.md) | Per-tenant LangGraph namespace isolation · Redis token bucket rate limiting · ChromaDB namespace isolation · RBAC capability tiers · Langfuse cost tracking per tenant | Multi-tenant SaaS architecture |
| [project34_resilience](project34_resilience/README.md) | Circuit breaker (CLOSED/OPEN/HALF_OPEN) · 4-model fallback chain · Saga with compensation · DLQ · idempotency store | Failure resilience patterns |
| [project36_enterprise_multimodal](project36_enterprise_multimodal/README.md) | 55-file enterprise capstone: PDF+Vision+Audio ingestion · Graph RAG (Neo4j) · 4-layer guardrails · Mem0 memory · hybrid retrieval · circuit breaker · FastAPI observability · Docker | Everything combined |

**Prerequisite**: Groups 1–9 recommended before project36 (it's the capstone)

---

## Group 11 — System Design Applied

> **What you learn**: How to *think* about agent systems at an architectural level — measure trade-offs between topologies, enforce context budgets, build the async execution platform, and load-test + capacity-plan for production.

> **Companion**: [`guide/13_system_design.md`](../guide/13_system_design.md) — read the linked section before each project.

| Project | Guide section | What you build | Core skill |
|---|---|---|---|
| [project37_topology_benchmark](project37_topology_benchmark/README.md) | §2 Topology Patterns | Benchmark 5 topologies on same task → decision matrix (quality/cost/latency) | Measure trade-offs empirically |
| [project38_context_engine](project38_context_engine/README.md) | §5 Context Architecture | Budget allocator: 8K window → 5 sources, 4 memory levels, savings report | Context as a constrained resource |
| [project39_async_platform](project39_async_platform/README.md) | §9 Async & Queues | Full platform: FastAPI 202+SSE+Celery+Redis pub/sub+checkpointing+idempotency | Production async execution |
| [project40_scale_loadtest](project40_scale_loadtest/README.md) | §10 Scalability | Locust load test → fix bottlenecks → capacity plan for your SLA | Finding and fixing breaking points |

**Prerequisite**: Group 1 + Group 10 (do project39 before project40)

---

## Recommended Learning Paths

### Path A — "I want to build a production agent as fast as possible" (6–8 weeks)

```
Group 1 (Core) → Group 6 (RAG) → Group 3 (Observability) → Group 2 (Safety) → Group 10 (Infra)
```

### Path B — "I want to understand all frameworks" (4–6 weeks, alongside Group 1)

```
Group 1 (Core) → Group 7 (Frameworks) → Group 8 (Real-World Apps)
```

### Path C — "I want to optimize for cost and scale" (3–4 weeks)

```
Group 3 (Observability) → Group 11 (System Design) → Group 5 (Data Processing)
```
> Also: `phase4_production/week8_observability/resources/token_optimization_guide.md`

### Path D — "I'm preparing for a system design interview on AI agents" (2–3 weeks)

```
guide/13_system_design.md (read all 17 sections) → Group 11 (all 4 projects) → Group 10 (infra)
```

### Path E — "I want to go deep on advanced techniques" (6–8 weeks)

```
Group 1 → Group 6 → Group 9 (Advanced Techniques) → Group 4 (Evaluation)
```

### Path F — "I want the full curriculum in order"

```
Group 1 → Group 2 → Group 3 → Group 4 → Group 5 → Group 6
         → Group 7 → Group 8 → Group 9 → Group 10 → Group 11
```

---

## Skills Map — What Builds on What

```
Group 1 (Core Curriculum)
    │
    ├──► Group 2 (Safety)           — add guardrails to Phase 1–6 agents
    ├──► Group 3 (Observability)    — instrument Phase 4 agent API
    ├──► Group 4 (Evaluation)       — evaluate Phase 5/6 agents
    ├──► Group 5 (Data Processing)  — scale Phase 3 batch processing
    │
    ├──► Group 6 (RAG)              — deepen Phase 2 RAG knowledge
    │         │
    │         └──► Group 8 (Real-World Apps) — use RAG in domain agents
    │
    ├──► Group 7 (Frameworks)       — re-implement Phase 1–6 in each framework
    │         │
    │         └──► Group 8 (Real-World Apps) — pick a framework, build a vertical
    │
    ├──► Group 9 (Advanced Techniques) — push quality and capability further
    │
    └──► Group 10 (Infra)           ← needs Groups 1–9 as foundation
              │
              └──► Group 11 (System Design) ← applies Group 10 concepts empirically
```

---

## Skills You Master (by end of all 40)

| Domain | Skills |
|---|---|
| **Core agent patterns** | ReAct, Reflexion, Plan-Execute, Debate, Fan-Out, Tree of Thought, MCTS |
| **RAG** | Chunking, embedding, hybrid BM25+vector, reranking, Graph RAG, multimodal RAG |
| **Frameworks** | LangChain, LangGraph, CrewAI, LlamaIndex, AutoGen, MCP, A2A |
| **APIs & serving** | FastAPI, SSE streaming, Celery workers, Redis pub/sub, 202+polling pattern |
| **Observability** | structlog, Prometheus, OpenTelemetry, cost tracking, LLM-as-judge eval |
| **Safety** | Prompt injection defense, PII scrubbing, Llama Guard, NeMo Guardrails, HITL |
| **Scale & reliability** | Circuit breaker, Saga, DLQ, idempotency, multi-tenancy, load testing, HPA |
| **Advanced techniques** | QLoRA fine-tuning, DPO, DSPy prompt optimization, multi-modal, sandboxed code exec |
| **System design** | Context budgeting, topology trade-offs, async platforms, capacity planning |
| **Cost optimization** | Token counting, RAG budgeting, model routing, prompt caching, savings measurement |
