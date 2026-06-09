# Master Resource List — Agentic AI

## 📚 Books
- "Designing Machine Learning Systems" — Chip Huyen (production mindset)
- "Building LLM Applications" — various authors on O'Reilly

## 🎓 Short Courses (DeepLearning.AI — all free)
- AI Agents in LangGraph
- Building Agentic RAG with LlamaIndex
- Multi AI Agent Systems with crewAI
- Evaluating and Debugging Generative AI
- Functions, Tools and Agents with LangChain

## 📄 Must-Read Papers
- ReAct (2022): https://arxiv.org/abs/2210.03629
- Reflexion (2023): https://arxiv.org/abs/2303.11366
- Tree of Thought (2023): https://arxiv.org/abs/2305.10601
- AutoGen (2023): https://arxiv.org/abs/2308.00352
- LATS (2023): https://arxiv.org/abs/2310.04406
- RAGAS (2023): https://arxiv.org/abs/2309.15217
- LLM-as-a-Judge / MT-Bench (2023): https://arxiv.org/abs/2306.05685
- AgentBench (2023): https://arxiv.org/abs/2308.03688
- Can LLMs Replace Human Evaluators? (2024): https://arxiv.org/abs/2404.03622
- Anthropic's "Building Effective Agents": https://www.anthropic.com/research/building-effective-agents

## 🛠 Key Libraries
| Library | Use |
|---|---|
| anthropic | Anthropic SDK |
| litellm | Unified LLM API (OpenAI, Anthropic, Gemini, …) |
| langchain / langgraph | Agent framework |
| chromadb / qdrant | Vector database |
| sentence-transformers | Local embeddings |
| fastapi + uvicorn | API serving |
| celery + redis | Background jobs |
| structlog | Structured logging |
| opentelemetry | Distributed tracing |
| ragas | RAG evaluation (faithfulness, relevancy, precision, recall) |
| datasets | HuggingFace datasets — eval data management |
| deepeval | LLM evaluation with 14+ typed metrics and pytest integration |
| langsmith | LangSmith tracing **and** evaluation datasets + experiments |
| arize-phoenix | Open-source LLM observability + eval dashboard |
| pydantic | Data validation |
| httpx | Async HTTP |
| tavily-python | Web search tool |

## � System Design for AI Agents (in this repo)

**Full guide**: [`guide/13_system_design.md`](guide/13_system_design.md) — 17 sections:

| Section | What you'll learn |
|---|---|
| Agent Topology Patterns | Single / Orchestrator-Worker / Pipeline / Fan-Out / Debate / Hierarchical / Event-driven |
| Stateless vs Stateful | Trade-offs, hybrid pattern, external state stores |
| The Agent Loop | max_steps, per-step timeout, total timeout, cost cap, HITL interrupt |
| Context Architecture | The four memory levels, context budget allocation, assembly order |
| Tool Layer Design | Least privilege, idempotency, timeout, SSRF prevention, schema design |
| Memory & Storage | Three-tier pattern (Redis/Postgres/S3/Qdrant), schema for agent runs |
| Multi-Agent Communication | Shared memory / message passing / event bus / A2A protocol |
| Async & Queues | Sync vs async decision matrix, Celery task design, parallel tool execution |
| Scalability | Horizontal scaling, auto-scaling by queue depth, multi-tenancy models, rate limiting |
| Reliability | Circuit breaker, fallback cascade, checkpointing, idempotency keys |
| Security | Threat model, input validation, output sanitisation, tool permission matrix |
| Cost as a Design Constraint | Budget hierarchy, cost attribution, cost-quality frontier |
| Deployment Architectures | Single process → monolith → microservices → serverless → Kubernetes |
| Reference Architectures | Perplexity-style search, Cursor-style coding, enterprise doc intelligence |
| Interview Framework | 6-step framework for system-design interviews on AI agent systems |
| Decision Cheat Sheet | One-row-per-decision table covering all major architecture choices |

**Related guides in this repo**:
- [`guide/01_agentic_stack.md`](guide/01_agentic_stack.md) — full layer-by-layer stack
- [`guide/04_multi_agent.md`](guide/04_multi_agent.md) — complete pattern implementations with code
- [`guide/06_production_checklist.md`](guide/06_production_checklist.md) — pre-deploy checklist
- [`phase4_production/week8_observability/resources/token_optimization_guide.md`](phase4_production/week8_observability/resources/token_optimization_guide.md) — token cost optimization (16 sections)

---

## 🔄 LLMOps (Guide 14)

> See [`guide/14_llmops.md`](guide/14_llmops.md) for the complete reference (~1,200 lines, 16 sections).

| LLMOps Topic | What it covers |
|---|---|
| LLMOps vs MLOps | Why LLM apps need a specialised ops practice |
| Prompt Registry & Versioning | SQLite registry, Git-based, LangSmith, PromptLayer |
| Experiment Tracking | MLflow, W&B, LangSmith — what to log per run |
| Continuous Evaluation | APScheduler + golden dataset + regression detection |
| Drift Detection | Input drift (KS/TF-IDF), embedding drift (KS on PCA), quality drift (LLM-judge rolling) |
| Model Registry & Promotion | Staging → Shadow → Canary → Production → Rollback |
| SLOs for LLM Systems | Latency P50/P95/P99, error rate, quality score, cost/call thresholds |
| LLMOps Platforms | Langfuse · Arize Phoenix · Helicone · W&B Weave · LangSmith · MLflow comparison |
| Production Monitoring | Prometheus metrics, Grafana layout, OpenTelemetry + litellm |
| CI/CD for Agents | Full GitHub Actions workflow with eval gate, Docker, rolling deploy, smoke test |
| Cost Governance | Budget hierarchy, per-call cap, daily hard stop, model tiering |
| Reference Architecture | Full stack: Langfuse + Prometheus + Grafana + PagerDuty + prompt registry |

**Related projects**:
- [`project41_llmops_monitoring`](projects/project41_llmops_monitoring/) — drift detection, SLOs, alerts
- [`project42_prompt_registry`](projects/project42_prompt_registry/) — prompt versioning, A/B, canary
- [`project43_continuous_eval`](projects/project43_continuous_eval/) — scheduled evals, regression detection

---

## �📰 Newsletters & Blogs
- **Complete Guide to Production-Ready AI Agents** (start here): https://medium.com/@devkapiltech/a-complete-guide-to-building-production-ready-ai-agents-from-your-first-afternoon-project-to-d5c2f3597565
- The Batch (DeepLearning.AI): https://www.deeplearning.ai/the-batch/
- Latent Space: https://www.latent.space/
- Ahead of AI (Sebastian Raschka): https://magazine.sebastianraschka.com/
- Simon Willison's Weblog: https://simonwillison.net/

## 🎥 YouTube Channels
- Andrej Karpathy
- Yannic Kilcher
- AI Explained
- Matt Wolfe

## 🔧 Tools & Platforms
- LangSmith (tracing + eval datasets): https://smith.langchain.com/
- Weights & Biases (experiment tracking): https://wandb.ai/
- Helicone (LLM observability): https://www.helicone.ai/
- Phoenix / Arize (open-source LLM eval dashboard): https://phoenix.arize.com/
- BrainTrust (evals + tracing): https://www.braintrustdata.com/
- Tavily (web search API): https://tavily.com/
- Modal (serverless GPU/CPU): https://modal.com/
- Railway (deploy): https://railway.app/

## 🔐 Security Resources
- OWASP LLM Top 10: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Prompt injection examples: https://github.com/greshake/llm-security
- Lakera AI (guardrails): https://www.lakera.ai/

## 📊 Benchmarks (know what your agent is measured against)
- SWE-bench (software engineering): https://www.swebench.com/
- HumanEval (code generation): https://github.com/openai/human-eval
- GAIA (general AI assistants): https://huggingface.co/datasets/gaia-benchmark/GAIA
- AgentBench (agent evaluation): https://arxiv.org/abs/2308.03688
- BigCodeBench (realistic code tasks): https://bigcode-bench.github.io/
- TruthfulQA (hallucination detection): https://github.com/sylinrl/TruthfulQA
