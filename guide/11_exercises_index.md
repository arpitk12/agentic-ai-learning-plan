[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §10 Deployment](guide/10_deployment.md) | [§12 Evaluation & QA →](guide/12_evaluation.md)

---

## 11. Exercises Index — Topics Mapped to Practice

Every section of this guide has one or more hands-on exercises. Use this index to find
the exercise for any topic you want to practice, or to check which guide section a given
exercise is teaching.

---

### Phase 1 — LLM Foundations

**Week 1: LLM API & Structured Output** (`phase1_foundations/week1_llm_api/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_multiturn_chatbot.py` | §1 (Agentic Stack), §2.2 (LiteLLM) | Multi-turn message history, conversation memory, llm.py API |
| `ex2_structured_output.py` | §2.8 (Pydantic) | JSON schema prompting, Pydantic validation, retry on parse failure |
| `ex3_streaming.py` | §2.9 (FastAPI streaming) | SSE token streaming, litellm stream=True, async generators |
| `ex4_prompt_comparison.py` | §2.2 (LiteLLM), §7.1 (model routing) | Prompt engineering, few-shot vs zero-shot, temperature effects |

**Week 2: Tool Use & ReAct** (`phase1_foundations/week2_tool_use/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_basic_tools.py` | §1 (Tool Layer), §2.2 (LiteLLM tools) | Tool schema definition, normalize_tools(), get_tool_calls() |
| `ex2_react_loop.py` | §1 (ReAct loop), §4.2 (Orchestrator) | Full ReAct cycle: reason → act → observe → repeat |
| `ex3_error_handling.py` | §6.2 (Agent Safety), §6.3 (Reliability) | Tool timeout, retry with exponential backoff, partial results |

---

### Phase 2 — Memory & RAG

**Week 3: Agent Frameworks** (`phase2_memory_rag/week3_frameworks/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_langgraph_react.py` | §2.4 (LangGraph) | StateGraph, TypedDict state, add_edge, compile(), invoke() |
| `ex2_persistence.py` | §2.4 (LangGraph checkpointing) | MemorySaver, thread_id, resuming interrupted graphs |
| `ex3_langsmith_tracing.py` | §2.7 (LangSmith) | Automatic tracing, @traceable decorator, LangSmith UI navigation |
| `ex4_crewai_pipeline.py` ⭐ | §2.5 (CrewAI), §4.8 (Pipeline Pattern) | Role-based agents, Task dependencies, Process.sequential |
| `ex5_langchain_lcel.py` ⭐ | §2.3 (LangChain) | LCEL pipe syntax, document loaders, RecursiveCharacterTextSplitter, retrieval chain |

**Week 4: RAG Pipelines** (`phase2_memory_rag/week4_rag/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_rag_basic.py` | §3 (RAG Architecture), §2.11 (ChromaDB) | PDF loading, chunking, ChromaDB upsert, similarity search, grounded answers |
| `ex2_chunking_compare.py` | §3.4 (Chunking Strategies) | Fixed vs recursive vs token vs semantic chunking, quality comparison |
| `ex3_persistent_memory.py` | §3.6 (Ingestion Pipeline), §2.14 (sentence-transformers) | Persistent ChromaDB, conversation memory types, episodic/semantic memory |
| `ex4_hybrid_search.py` | §2.16 (rank-bm25), §3.7 (Retrieval) | BM25 keyword search, vector search, Reciprocal Rank Fusion (RRF) |
| `ex5_advanced_retrieval.py` ⭐ | §3.7 (HyDE, Multi-Query, Reranking) | HyDE hypothetical embeddings, multi-query generation, cross-encoder reranking |
| `ex6_vector_dbs.py` ⭐ | §2.12 (Qdrant), §2.13 (FAISS), §5 (Vector Search) | FAISS IndexFlatIP/HNSW, ChromaDB native filtering, Qdrant payload indexes |

---

### Phase 3 — Multi-Agent Systems

**Week 5: Orchestration Patterns** (`phase3_multi_agent/week5_orchestrator/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_orchestrator.py` | §4.2 (Orchestrator-Worker) | Planner LLM, task decomposition, WORKERS dict, synthesis |
| `ex2_human_approval.py` | §4.7 (HITL Pattern), §6.2 (Agent Safety) | Risk classification, interrupt_before, human approval loop |
| `ex3_debate_pattern.py` | §4.3 (Debate/Adversarial) | Pro/con agents, structured debate rounds, judge synthesis |

**Week 6: Parallel Processing** (`phase3_multi_agent/week6_parallelism/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_async_fan_out.py` | §4.4 (Fan-Out/Fan-In) | asyncio.gather(), Semaphore concurrency limiting, error handling |
| `ex2_map_reduce.py` | §4.5 (Map-Reduce) | Parallel map phase, hierarchical reduce, token-budget-aware batching |
| `ex3_rate_limiter.py` | §6.3 (Reliability), §7 (Cost) | Token bucket, sliding window rate limiting, 429 backoff |

---

### Phase 4 — Production APIs

**Week 7: API Serving** (`phase4_production/week7_api_serving/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_fastapi_agent.py` | §2.9 (FastAPI) | FastAPI routes, Pydantic request/response models, Depends() auth |
| `ex2_sse_streaming.py` | §2.9 (FastAPI SSE) | StreamingResponse, server-sent events, real-time token delivery |
| `ex3_celery_worker.py` | §2.10 (Celery + Redis) | Task queue, update_state(), AsyncResult polling, task retry |
| `ex4_rate_limiter.py` | §6.1 (API Layer), §6.2 (Safety) | Redis-backed rate limiting, per-user limits, 429 responses |
| `ex5_semantic_cache.py` ⭐ | §7.2 (Semantic Caching) | Cosine similarity cache lookup, TTL eviction, hit rate tracking |

**Week 8: Observability** (`phase4_production/week8_observability/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_structured_logging.py` | §9.2 (structlog) | JSON structured logs, context binding, log levels, run_id threading |
| `ex2_cost_tracker.py` | §7 (Cost Optimization), §6.5 (Observability) | Per-model cost tracking, calc_cost(), daily budget enforcement |
| `ex3_guardrails.py` | §8 (Security Hardening) | Injection detection, PII scanning, output filtering, tool validation |
| `ex4_otel_tracing.py` | §9.3 (OpenTelemetry) | Trace spans, tool/LLM instrumentation, Jaeger export |

---

### Phase 5 — Advanced Patterns

**Week 9: Planning Strategies** (`phase5_advanced/week9_planning/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_plan_execute.py` | §1 (Planning Module), §4.2 (Orchestrator) | Planner → task list → executor → synthesizer pipeline |
| `ex2_self_reflection.py` | §4.6 (Reflexion) | Generate → evaluate → reflect → retry cycle |
| `ex3_reflexion.py` | §4.6 (Reflexion) | Verbal reflection accumulation across attempts |
| `ex4_tree_of_thought.py` | §1 (Planning Module) | Beam search over thought branches, branch scoring, pruning |

**Week 10: Evaluation** (`phase5_advanced/week10_evaluation/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_golden_dataset.py` | §3.8, §12.3.1 | Build ground-truth Q&A dataset, baseline measurement |
| `ex2_llm_judge.py` | §12.3.2 | LLM-as-judge prompt design, rubric scoring, calibration |
| `ex3_ragas_eval.py` | §12.3.3 | RAGAS 4 core metrics: faithfulness, answer_relevancy, context_precision, context_recall |
| `ex4_pytest_agent.py` | §12.4 | pytest fixtures, mock LLM responses, deterministic agent tests |
| `ex5_safety_adversarial.py` ⭐ | §12.2.5, §8 | Harmful refusal rate, injection blocking, PII leak, over-refusal |
| `ex6_tool_quality_eval.py` ⭐ | §12.2.3 | Tool selection accuracy, Pydantic arg validation, unnecessary call rate |
| `ex7_performance_benchmark.py` ⭐ | §12.2.6 | Latency P50/P95/P99, cost/run, token budget compliance |
| `ex8_conversation_eval.py` ⭐ | §12.6 Challenge 4 | Multi-turn continuity, coreference, contradiction detection |
| `ex9_ragas_advanced.py` ⭐ | §12.3.3, §3.8 | Custom LLM wrapper for RAGAS, chunk-size comparison (128/256/512), per-metric recommendations |
| `ex10_deepeval.py` ⭐ | §12.3, §12.9 | DeepEval LLMTestCase, AnswerRelevancy/Faithfulness/Hallucination/Bias, custom BaseMetric, pytest |
| `ex11_langsmith_eval.py` ⭐ | §2.7, §12.3 | LangSmith datasets, custom evaluator functions, evaluate(), agent v1 vs v2 comparison |

---

### Phase 6 — Capstone & Deployment

**Week 11: System Integration** (`phase6_capstone/week11_integration/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_mcp_server.py` | §2 (Tool Layer — MCP) | MCP server definition, tool registration, JSON-RPC protocol |
| `ex2_db_schema.py` | §1 (Data Layer), §6.4 (Data Layer checklist) | PostgreSQL schema design, asyncpg pool, agent run audit log |
| `ex3_ci_pipeline.yml` | §10.3 (CI/CD Pipeline) | GitHub Actions lint → test → build → deploy → smoke test |

**Week 12: Deployment** (`phase6_capstone/week12_deployment/exercises/`)

| Exercise | Guide Sections | What You Practice |
|----------|---------------|-------------------|
| `ex1_dockerfile.py` | §2.19 (Docker), §10.1 (Dockerizing) | Multi-stage build, non-root user, HEALTHCHECK, layer caching |
| `ex2_model_router.py` | §7.1 (Model Routing) | Rule-based + LLM-based complexity classification, MODEL_TIERS dict |
| `ex3_monitoring.py` | §9.3 (Prometheus metrics), §9.4 (Alerting) | prometheus_client counters/histograms, /metrics endpoint, alerting YAML |
| `ex4_load_test.py` | §10.4 (Scaling), Locust | Locust HttpUser, task weights, ramp rate, throughput ceiling |
| `ex5_k8s_deploy.py` ⭐ | §10.2 (Kubernetes) | Deployment, Service, HPA, Secret, ConfigMap, PVC manifest generation |

---

### Projects

| Project | Covers | Guide Sections |
|---------|--------|----------------|
| `phase1_research_assistant/` | LLM API + tools + ReAct | §1, §2.2, §2.15 (Tavily) |
| `phase2_knowledge_agent/` | RAG + ChromaDB + memory | §3, §2.11, §2.14 |
| `phase3_code_review/` | Multi-agent debate + critic | §4.3, §2.5, §2.4 |
| `phase4_agent_api/` | FastAPI + Celery + observability | §2.9, §2.10, §9 |
| `phase5_coding_agent/` | Planning + Reflexion + tools | §4.5, §4.6, §1 |
| `phase6_capstone/` | Full production stack | All sections |
| `project7_security_agent/` ⭐ | Prompt injection + PII detection + HITL for risky tools | §8, §6.1, §6.2 |
| `project8_observability_agent/` ⭐ | structlog + Prometheus metrics + OTel traces + cost tracking | §9, §2.17, §2.18 |
| `project9_batch_pipeline/` ⭐ | Fan-Out/Fan-In + Map-Reduce for 500+ items | §4.4, §4.5 |
| `project10_langgraph_agent/` ⭐ | LangGraph StateGraph + HITL approval + MemorySaver checkpointing | §2.4, §4.7 |
| `project11_eval_pipeline/` ⭐ | End-to-end eval: golden dataset + safety suite + tool quality + RAG faithfulness + perf benchmark + multi-turn + HTML report + CI gate | §12 |
| `project12_customer_support/` ⭐ | Multi-tier intent triage → specialist sub-agents → CRM tool use → escalation → SLA tracking → PII guard → session report | §2, §6, §7 |
| `project13_data_analyst/` ⭐ | NL question → plan decomposition → LLM code generation → safe subprocess execution → self-correction (max 3×) → narrative Markdown report | §4.6, §5 |
| `project14_doc_intelligence/` ⭐ | Batch classify → LLM+Pydantic extraction → two-pass validation → SHA-256 dedup → anomaly detection → JSON+HTML report | §2.1, §6, §7 |
| `project15_whatsapp_agent/` ⭐ | **MCP server** (FastMCP + TF-IDF RAG) + **multi-agent routing** (intent → 4 specialists via MCP tools) + **WhatsApp** (Twilio TwiML) + **Telegram** Bot API + per-user session history | §2, §3, §4, §7 |

---

### Quick Navigation by Guide Topic

| If you want to practice… | Go to exercise |
|--------------------------|----------------|
| LiteLLM unified API | `week1/ex1`, `week1/ex4` |
| Pydantic structured output from LLM | `week1/ex2` |
| Token streaming / SSE | `week1/ex3`, `week7/ex2` |
| Tool calling + ReAct loop | `week2/ex1`, `week2/ex2` |
| LangGraph stateful agents | `week3/ex1`, `week3/ex2` |
| LangGraph checkpointing | `week3/ex2` |
| LangSmith tracing | `week3/ex3` |
| CrewAI role-based pipeline | `week3/ex4` ⭐ |
| LangChain LCEL + document loaders | `week3/ex5` ⭐ |
| Basic RAG (ChromaDB + PDF) | `week4/ex1` |
| Chunking strategies comparison | `week4/ex2` |
| Hybrid BM25 + vector search | `week4/ex4` |
| HyDE + multi-query + reranking | `week4/ex5` ⭐ |
| FAISS / ChromaDB / Qdrant comparison | `week4/ex6` ⭐ |
| Orchestrator-Worker pattern | `week5/ex1` |
| Human-in-the-loop approval | `week5/ex2` |
| Debate / adversarial review | `week5/ex3` |
| Parallel fan-out with asyncio | `week6/ex1` |
| Map-reduce for large datasets | `week6/ex2` |
| FastAPI production agent API | `week7/ex1` |
| Celery background tasks | `week7/ex3` |
| Semantic cache | `week7/ex5` ⭐ |
| Structured JSON logging | `week8/ex1` |
| LLM cost tracking + budgets | `week8/ex2` |
| Prompt injection + PII guardrails | `week8/ex3` |
| OpenTelemetry distributed tracing | `week8/ex4` |
| Plan-Execute agent | `week9/ex1` |
| Reflexion self-correction | `week9/ex2`, `week9/ex3` |
| Tree of Thought | `week9/ex4` |
| RAGAS RAG evaluation | `week10/ex3` |
| LLM-as-judge | `week10/ex2` |
| MCP server | `week11/ex1` |
| Docker multi-stage build | `week12/ex1` |
| Model routing for cost saving | `week12/ex2` |
| Prometheus metrics + alerting | `week12/ex3` |
| Locust load testing | `week12/ex4` |
| Kubernetes deployment | `week12/ex5` ⭐ |
| **Production Security** (5 layers: injection, PII, tool validation, HITL, output scan) | `project7_security_agent/` ⭐ |
| **Full Observability** (structlog + Prometheus + OTel + cost per run) | `project8_observability_agent/` ⭐ |
| **Batch Fan-Out + Map-Reduce** (500+ items, Semaphore, hierarchical reduce) | `project9_batch_pipeline/` ⭐ |
| **LangGraph HITL + Checkpointing** (StateGraph, conditional routing, MemorySaver) | `project10_langgraph_agent/` ⭐ |
| **Safety adversarial testing** (harmful, injection, PII-leak, over-refusal) | `week10/ex5` ⭐ |
| **Tool selection accuracy + arg validity eval** | `week10/ex6` ⭐ |
| **Latency P50/P95/P99 + cost/run benchmarking** | `week10/ex7` ⭐ |
| **Multi-turn conversation eval** (context continuity, contradiction detection) | `week10/ex8` ⭐ |
| **RAGAS advanced** (custom LLM config, chunk-size comparison, recommendations) | `week10/ex9` ⭐ |
| **DeepEval** (LLMTestCase, 4 metrics, custom BaseMetric, pytest integration) | `week10/ex10` ⭐ |
| **LangSmith evaluation** (datasets, custom evaluators, version comparison) | `week10/ex11` ⭐ |
| **End-to-end eval pipeline + HTML report + CI gate** | `project11_eval_pipeline/` ⭐ |
| **Multi-tier customer support agent (triage → CRM tools → SLA)** | `project12_customer_support/` ⭐ |
| **Data analyst agent (code gen → subprocess exec → self-correct)** | `project13_data_analyst/` ⭐ |
| **Document intelligence pipeline (classify → extract → dedup → report)** | `project14_doc_intelligence/` ⭐ |
| **WhatsApp/Telegram agent with MCP server + RAG** | `project15_whatsapp_agent/` ⭐ |
| **Agent evaluation concepts, failure modes, quality dimensions** | [§12 Agent Evaluation](#12-agent-evaluation--quality-assurance) ⭐ |
| **Golden dataset eval + LLM-as-judge** | `week10/ex2`, `week10/ex3`, [§12.3](#123-evaluation-methods) ⭐ |
| **CI/CD quality gate pipeline** | [§12.8](#128-wiring-evaluation-into-cicd) ⭐ |

> ⭐ = newly added (not in original repo)

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §10 Deployment](guide/10_deployment.md) | [§12 Evaluation & QA →](guide/12_evaluation.md)
