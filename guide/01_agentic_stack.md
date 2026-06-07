[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [§2 Framework Selection →](guide/02_framework_selection.md)

---

## 1. The Complete Agentic Stack

Understanding the full stack is crucial before building any production agent. Every layer has a specific job, and failures in one layer cascade upward. This section walks through the complete architecture from user request to LLM response and back.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                            │
│         CLI (terminal) │ Web App (React/Next.js) │ Slack/Teams Bot        │
│                    REST API Client │ Mobile App                           │
├──────────────────────────────────────────────────────────────────────────┤
│                            API GATEWAY LAYER                              │
│   FastAPI + Uvicorn │ Auth (JWT/API Key) │ Rate Limiting │ Load Balancer  │
│          Request Validation (Pydantic) │ CORS │ TLS termination           │
├──────────────────────────────────────────────────────────────────────────┤
│                         ASYNC TASK QUEUE LAYER                            │
│        Celery Workers │ Redis Broker │ Task Scheduling │ Result Backend    │
│         For: long-running agents, batch processing, retries               │
├──────────────────────────────────────────────────────────────────────────┤
│                           AGENT CORE LAYER                                │
│  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  Planning Module │  │  Execution Loop │  │    Memory & Context     │ │
│  │  Plan-Execute    │  │  ReAct / LATS   │  │  RAG + SQLite + Redis   │ │
│  │  Tree of Thought │  │  Tool Dispatch  │  │  Sliding Window         │ │
│  │  Reflexion       │  │  Error Recovery │  │  Episodic / Semantic    │ │
│  └──────────────────┘  └─────────────────┘  └─────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│                        MULTI-AGENT LAYER (optional)                       │
│   Orchestrator │ Specialist Workers │ Debate Agents │ Review Agents        │
│           CrewAI / LangGraph / Custom orchestration                        │
├──────────────────────────────────────────────────────────────────────────┤
│                         LLM PROVIDER LAYER                                │
│   LiteLLM (unified API) → Gemini │ GPT-4o │ Claude │ Llama │ Mistral      │
│             Token counting │ Cost calculation │ Retry logic               │
├──────────────────────────────────────────────────────────────────────────┤
│                            TOOL LAYER                                     │
│  Web Search (Tavily) │ Code Execution (E2B) │ Database Queries            │
│  File Read/Write │ External APIs │ Calculator │ Email │ Calendar           │
│                     All sandboxed and validated                            │
├──────────────────────────────────────────────────────────────────────────┤
│                          DATA & MEMORY LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │   Vector Database │  │   Relational DB  │  │         Redis          │ │
│  │  ChromaDB / Qdrant│  │   PostgreSQL     │  │  Cache │ Sessions      │ │
│  │  FAISS / Weaviate │  │  Agent runs      │  │  Rate limiting         │ │
│  │  RAG embeddings   │  │  Users, costs    │  │  Celery broker         │ │
│  └──────────────────┘  └──────────────────┘  └────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────────┤
│                        OBSERVABILITY LAYER                                │
│  structlog (JSON logs) │ Prometheus (metrics) │ Grafana (dashboards)      │
│  OpenTelemetry (traces) │ Alertmanager │ PagerDuty/Slack alerts           │
├──────────────────────────────────────────────────────────────────────────┤
│                         DEPLOYMENT LAYER                                  │
│  Docker (containers) │ Kubernetes (orchestration) │ Helm (packaging)      │
│  GitHub Actions (CI/CD) │ Container Registry │ Secrets Manager            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities — Deep Dive

| Layer | What It Does | Key Tools | Failure Mode |
|-------|-------------|-----------|-------------|
| **User Interface** | Formats user input, renders output | React, CLI, Slack SDK | Bad UX, no streaming |
| **API Gateway** | HTTP interface, auth, rate limiting | FastAPI, uvicorn | 401 errors, DDoS |
| **Task Queue** | Background agent execution | Celery, Redis | Tasks stuck, queue backup |
| **Agent Core** | Reasoning, planning, tool orchestration | llm.py, LangGraph | Infinite loops, bad reasoning |
| **Multi-Agent** | Specialist coordination | CrewAI, LangGraph | Context loss, cost explosion |
| **LLM Provider** | Actual model inference | LiteLLM | Rate limits, high cost, hallucination |
| **Tool Layer** | Real-world actions | Custom functions | Security holes, timeout |
| **Data** | Persistence, retrieval, caching | PostgreSQL, Chroma, Redis | Stale data, slow retrieval |
| **Observability** | Logging, metrics, tracing | structlog, Prometheus | Blind to failures |
| **Deployment** | Packaging and orchestration | Docker, K8s | Downtime, scaling failures |

### Request Lifecycle — What Happens in 30 Seconds

```
1. User sends: POST /agent/run {"query": "Analyze Q3 sales data and find anomalies"}

2. API Gateway (50ms):
   - Authenticate API key
   - Validate request (Pydantic: query not empty, length < 10K)
   - Check rate limit: 10 req/min for this user
   - Route to Celery if expected duration > 30s

3. Task Queue (1ms):
   - Serialize task to Redis
   - Return task_id to user immediately

4. Celery Worker picks up task:
   
5. Agent Core - PLANNING (2s):
   - Planner LLM creates execution plan: [load_data, analyze, identify_anomalies, report]
   - Plan saved to Redis for tracking

6. Agent Core - EXECUTION LOOP (20s):
   Step 1: load_csv_tool("sales_q3.csv") → 5000 rows loaded
   Step 2: analyze_tool("find statistical anomalies") → calls Python code
   Step 3: LLM analyzes code output → identifies 3 anomalies
   Step 4: report_tool("generate PDF") → PDF created

7. Memory Layer (async):
   - Save this run to PostgreSQL (user_id, task_id, cost, duration)
   - Update user's usage counters in Redis

8. Observability (every step):
   - structlog writes JSON log entry for each action
   - Prometheus increments: llm_calls_total, tool_calls_total
   - Cost tracked: $0.023 for this run

9. User polls GET /agent/result/{task_id}:
   - Result fetched from Redis result backend
   - PDF URL returned with summary text

Total: ~25 seconds, $0.023, 4 LLM calls, 3 tool calls
```

### Technology Selection Rationale

Why these specific tools were chosen for this stack:

| Decision | Why This Tool | Alternatives Considered |
|----------|--------------|------------------------|
| LiteLLM | Switch providers without code changes | Direct OpenAI SDK (vendor lock-in) |
| FastAPI | Async, auto-docs, Pydantic native | Flask (sync), Django (heavy) |
| Celery | Battle-tested, rich monitoring | RQ (simpler), Dramatiq (modern) |
| Redis | Fast, supports multiple roles (queue + cache + sessions) | RabbitMQ (queue only), Memcached |
| ChromaDB → Qdrant | Chroma for dev, Qdrant for prod scale | Pinecone (expensive), pgvector (limited) |
| structlog | JSON output, context binding | Python logging (unstructured) |
| Prometheus + Grafana | Industry standard, pull-based | DataDog (expensive), New Relic |
| Docker + K8s | Portable, scalable, industry standard | Bare metal (inflexible) |

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [§2 Framework Selection →](guide/02_framework_selection.md)
