[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §12 Evaluation](guide/12_evaluation.md)

---

# 13. System Design for AI Agents

> **How to think about, architect, and scale production agentic systems** — from a single-function agent to a distributed multi-tenant AI platform.

This guide is the "architect's handbook" for the repo. Use it:
- When making architectural decisions for a new agent system
- As prep material for system-design interview rounds focused on AI/LLM systems
- As a reference when the exercises in Phase 3–7 leave you wondering *why* a pattern exists

---

## Table of Contents

1. [The Core Design Question](#1-the-core-design-question)
2. [Agent Topology Patterns](#2-agent-topology-patterns)
3. [Stateless vs Stateful Agents](#3-stateless-vs-stateful-agents)
4. [The Agent Loop — Internal Design](#4-the-agent-loop--internal-design)
5. [Context Architecture](#5-context-architecture)
6. [Tool Layer Design](#6-tool-layer-design)
7. [Memory & Storage Architecture](#7-memory--storage-architecture)
8. [Multi-Agent Communication](#8-multi-agent-communication)
9. [Async, Queues & Concurrency](#9-async-queues--concurrency)
10. [Scalability Patterns](#10-scalability-patterns)
11. [Reliability Engineering for Agents](#11-reliability-engineering-for-agents)
12. [Security Architecture](#12-security-architecture)
13. [Cost as a First-Class Design Constraint](#13-cost-as-a-first-class-design-constraint)
14. [Deployment Architectures](#14-deployment-architectures)
15. [Real-World Reference Architectures](#15-real-world-reference-architectures)
16. [System Design Interview Framework](#16-system-design-interview-framework)
17. [Decision Cheat Sheet](#17-decision-cheat-sheet)

---

## 1. The Core Design Question

Before writing a single line of code, answer these five questions. Every architectural decision flows from them.

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. LATENCY   Is the user waiting? (sync) or is this a background   │
│               job? (async)                                          │
│                                                                     │
│  2. STATEFULNESS  Does the agent need to remember across turns?     │
│                   Across sessions? Across users?                    │
│                                                                     │
│  3. AUTONOMY  How many steps can it take unsupervised?              │
│               When must a human approve?                            │
│                                                                     │
│  4. SCALE     10 req/day? 10K/day? 10M/day?                        │
│               1 user? 100 tenants? Millions of users?               │
│                                                                     │
│  5. RISK      What is the worst case if the agent is wrong?         │
│               Read-only? Write? Delete? Financial transaction?       │
└─────────────────────────────────────────────────────────────────────┘
```

The answers determine your topology, persistence strategy, auth model, and how aggressively you can automate.

---

## 2. Agent Topology Patterns

### 2.1 Single Agent (ReAct)

The simplest architecture. One agent, one loop, one context window.

```
User ──► Agent ──► LLM ──► Tools ──► LLM ──► User
            ↑_______________________________↓
                    (loop until done)
```

**Use when**:
- Task fits in one context window
- Single specialization needed
- ≤8–10 steps to completion
- Latency budget is tight (each extra LLM hop adds 500–2000ms)

**Examples**: Research assistant, Q&A bot, code explainer, data extractor

**Failure modes**: Infinite loops, hallucinated tool calls, context overflow on long tasks

---

### 2.2 Orchestrator-Worker

A planner agent decomposes the task; specialist workers execute subtasks independently.

```
User ──► Orchestrator (planner LLM)
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
 Researcher  Coder    Writer     (parallel workers)
    │         │          │
    └─────────┴──────────┘
              │
         Synthesizer ──► User
```

**Use when**:
- Task has clearly distinct specializations
- Subtasks can run independently (or in parallel)
- Quality of individual subtask output matters

**Cost model**: N+2 LLM calls (1 plan + N workers + 1 synthesis). Budget accordingly.

**Examples**: Report generation, code review + fix + test, SEO content pipeline

---

### 2.3 Pipeline (Sequential)

Each agent's output is the next agent's input. Assembly-line processing.

```
Extractor ──► Enricher ──► Validator ──► Formatter ──► Output
```

**Use when**:
- Each step has a strict dependency on the previous
- Each step benefits from a different system prompt / specialization
- You want tight quality gates between stages

**Examples**: Document intelligence (extract → clean → classify → summarize), ETL for unstructured data

**Key design decision**: Pass structured data between stages (Pydantic models), not free-form strings. Prevents error propagation.

---

### 2.4 Fan-Out / Fan-In (MapReduce for LLMs)

Split a large input, process chunks in parallel, merge results.

```
                 ┌─► Agent (chunk 1) ─┐
Large Input ──► Splitter ─► Agent (chunk 2) ─► Merger ──► Result
                 └─► Agent (chunk 3) ─┘
```

**Use when**:
- Input is too large for one context window
- Independent sections can be processed in parallel
- Final result is an aggregation of section results

**Examples**: Summarize 500-page PDF, analyze entire codebase, process 1,000 customer reviews

**Design considerations**:
- Use consistent chunking (by section/chapter, not arbitrary byte count)
- Merger agent needs summary of each chunk + its own context budget
- Track chunk provenance (which pages each insight came from)

---

### 2.5 Debate / Adversarial Review

Two agents with opposing roles review each other's work.

```
      ┌──────────────────────────┐
      ▼                          │
   Proposer ──► Answer ──► Critic ──► Revised Answer
      ▲                          │
      └──────────────────────────┘
         (loop until critic satisfied or max rounds)
```

**Use when**:
- High-stakes decisions (legal, medical, financial)
- Hallucination risk is critical
- Task benefits from adversarial scrutiny

**Cost**: 2N LLM calls (N propose/refine rounds). Reserve for critical paths only.

**Examples**: Contract review, financial analysis, security vulnerability assessment

---

### 2.6 Hierarchical Multi-Agent

Orchestrators themselves have sub-orchestrators. Mirrors corporate management.

```
CEO Agent
├── CTO Agent
│   ├── Frontend Dev Agent
│   ├── Backend Dev Agent
│   └── DevOps Agent
├── CMO Agent
│   ├── Content Agent
│   └── SEO Agent
└── CFO Agent
    └── Data Analyst Agent
```

**Use when**:
- Problem domain is large and hierarchically structured
- Teams of specialist agents need coordination
- Different security/permission levels at different tiers

**Warning**: Communication overhead and cost explode quickly. Use only when simpler patterns fail.

**Examples**: Full-stack SaaS builder, enterprise data platform, autonomous research lab

---

### 2.7 Event-Driven Agent Network

Agents subscribe to events and react. No central orchestrator. Pure message passing.

```
User Action ──► Event Bus (Kafka/Redis Streams)
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
    Agent A      Agent B    Agent C
    (triggered   (triggered (triggered
    by event X)  by event Y) by event Z)
         │          │          │
         └──────────┴──────────┘
                    │
               Event Bus (results as events)
```

**Use when**:
- Complex workflows with many optional branches
- Agents should react to real-world triggers (new file, API webhook, schedule)
- Loose coupling between agent types is a priority

**Examples**: CI/CD pipeline agent, IoT monitoring agent, financial market agent

---

## 3. Stateless vs Stateful Agents

This is the most important architectural decision after topology.

### Stateless Agent

```python
# No memory between calls. All context passed in each request.
async def handle_request(request: AgentRequest) -> AgentResponse:
    messages = build_messages(request.history, request.context, request.query)
    response = await llm_call(messages)
    return AgentResponse(content=response)
```

| Pros | Cons |
|---|---|
| Horizontally scalable (any replica can serve any request) | Client must send full history every call |
| Simple to deploy (no sticky sessions needed) | History size grows → cost grows |
| Easy to test (pure function) | No cross-session learning |
| Crash recovery is trivial (just retry) | No personalization without external state |

**Use for**: Stateless Q&A, one-shot analysis, document processing

---

### Stateful Agent

```python
# State stored server-side. Session ID used to retrieve it.
async def handle_request(session_id: str, query: str) -> AgentResponse:
    state = await state_store.get(session_id)      # Redis / PostgreSQL
    messages = state.history + [{"role": "user", "content": query}]
    response = await llm_call(messages)
    state.history.append(...)
    await state_store.set(session_id, state)        # persist state
    return AgentResponse(content=response)
```

| Pros | Cons |
|---|---|
| Efficient (client only sends current query) | Sticky sessions or shared state store required |
| Supports long multi-turn conversations | Harder to scale (state contention) |
| Enables cross-turn reasoning and planning | State migration on schema changes |
| Personalization and learning across sessions | Recovery requires state reconstruction |

**Use for**: Conversational assistants, multi-session research, coding copilots

---

### Hybrid: Stateless Compute + External State

The **best-of-both** approach used by production systems:

```
Client ──► API Server (stateless) ──► State Store (Redis/Postgres)
                │                              │
                │ (fetch state)    ←───────────┘
                │
                ▼
           Agent Execution (stateless function)
                │
                ▼ (persist new state)
           State Store ◄────────────────────────
```

API servers are stateless and horizontally scalable. State is externalized to Redis (fast, ephemeral) + PostgreSQL (durable, queryable).

---

## 4. The Agent Loop — Internal Design

The agent loop is the heart of every agent. Design it carefully.

### The ReAct Loop (canonical)

```
while not done:
    thought = llm(messages)           # "I need to search for X"
    if thought.has_tool_call:
        result = execute_tool(thought.tool_call)
        messages.append(tool_result)
    else:
        return thought.content        # final answer
    
    if steps > MAX_STEPS:
        return error("max steps exceeded")
```

### Production Loop Design

```python
async def agent_loop(
    messages: list[dict],
    tools: list[dict],
    max_steps: int = 10,
    step_timeout: float = 30.0,
    total_timeout: float = 120.0,
) -> AgentResult:
    
    steps = 0
    total_cost = 0.0
    start_time = time.time()
    
    while steps < max_steps:
        # Guard 1: Total timeout
        if time.time() - start_time > total_timeout:
            raise AgentTimeoutError(f"Agent exceeded {total_timeout}s total")
        
        # Guard 2: Cost cap
        if total_cost > MAX_COST_PER_RUN:
            raise CostCapExceededError(f"Run exceeded ${MAX_COST_PER_RUN}")
        
        # LLM call with per-step timeout
        try:
            response = await asyncio.wait_for(
                llm_call(messages, tools), timeout=step_timeout
            )
        except asyncio.TimeoutError:
            raise AgentStepTimeoutError(f"Step {steps} exceeded {step_timeout}s")
        
        total_cost += response.cost
        steps += 1
        
        # Check for tool calls
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            # Final answer
            return AgentResult(
                content=response.choices[0].message.content,
                steps=steps,
                cost=total_cost,
            )
        
        # Execute tools (parallel if multiple)
        tool_results = await asyncio.gather(*[
            execute_tool(tc) for tc in tool_calls
        ])
        
        # Append tool results to messages
        messages.append(response.choices[0].message)
        for tc, result in zip(tool_calls, tool_results):
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": str(result)})
    
    raise MaxStepsExceededError(f"Agent exceeded {max_steps} steps")
```

### Key Loop Invariants

| Invariant | Why |
|---|---|
| **Always set max_steps** | Prevents infinite loops from hallucinated tool call cycles |
| **Always set per-step timeout** | Prevents hanging on slow external APIs |
| **Always set total timeout** | End-to-end SLA guarantee for the caller |
| **Always track cost** | Prevents runaway spend ($100 agent runs happen without this) |
| **Append tool results before next LLM call** | Models lose context if results are dropped |
| **Validate tool calls before executing** | Malformed JSON tool calls will crash your executor |

---

### Loop Interrupt Points (HITL)

For risky operations, pause the loop and ask for human approval:

```python
REQUIRES_APPROVAL = {"delete_file", "send_email", "transfer_funds", "deploy_code"}

async def execute_tool(tool_call: ToolCall) -> str:
    if tool_call.name in REQUIRES_APPROVAL:
        approved = await request_human_approval(tool_call)
        if not approved:
            return f"Tool '{tool_call.name}' rejected by human reviewer."
    return await TOOL_REGISTRY[tool_call.name](**tool_call.args)
```

---

## 5. Context Architecture

Context is the most constrained resource in an agent system. Every byte in the context window costs money, adds latency, and displaces something else.

### The Context Budget

```
Total context window (e.g., 128K tokens for GPT-4o)
├── System prompt:           400 tokens  (fixed — optimize once)
├── History (managed):     2,000 tokens  (sliding window / summary)
├── RAG context:           2,000 tokens  (budget by relevance score)
├── Tool schemas:            400 tokens  (compact — strip noise)
├── Tool results:            600 tokens  (compact MCP responses)
├── Current user turn:       200 tokens  (this call)
└── Output reserve:        3,000 tokens  (for agent's answer)
─────────────────────────────────────────
Total:                     8,600 tokens  ← well within limit, leaves headroom
```

Allocate budget explicitly. Track actual vs budget on every call. Alert if any component overruns.

---

### Context Assembly Order

Models attend more to the beginning and end of context (primacy + recency bias). Use this:

```
1. System prompt          ← high signal, always first
2. Long-term memory       ← extracted facts, user preferences
3. RAG context            ← retrieved documents
4. Conversation history   ← compressed older turns first, recent turns last
5. Tool schemas           ← compact, injected near the end
6. Current user message   ← very last (recency bias helps)
```

---

### The Four Levels of Agent Memory

```
┌─────────────────────────────────────────────────────────────────┐
│  IN-CONTEXT MEMORY                                              │
│  The current messages array. Lost when context is flushed.     │
│  Fast. Expensive (each token costs money).                     │
│  Use for: current turn reasoning, tool results, recent history  │
├─────────────────────────────────────────────────────────────────┤
│  EPISODIC MEMORY (External, Session-Scoped)                     │
│  Conversation summaries stored in Redis/Postgres.               │
│  Cheap. Persists across turns within a session.                 │
│  Use for: conversation summaries, task progress checkpoints     │
├─────────────────────────────────────────────────────────────────┤
│  SEMANTIC MEMORY (Vector DB)                                    │
│  Embeddings of knowledge, documents, past interactions.         │
│  Retrieved by semantic similarity at query time.                │
│  Use for: RAG, long-term knowledge, cross-session context       │
├─────────────────────────────────────────────────────────────────┤
│  PROCEDURAL MEMORY (System Prompt + Fine-tuning)               │
│  Skills and behavior baked into the model or system prompt.    │
│  Changes require redeploy or re-fine-tune.                      │
│  Use for: agent personality, domain expertise, safety rules     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Tool Layer Design

### Tool Design Principles

#### 6.1 Principle of Least Privilege
Each tool should have the minimum permissions needed for its function. A search tool should not have file write access.

```python
# Bad: One monolithic tool with all permissions
def agent_tool(action: str, **kwargs) -> str:
    if action == "search": ...
    if action == "delete_file": ...  # ← dangerous

# Good: Separate tools, separate permission scopes
def search_web(query: str) -> str: ...       # read-only, safe
def read_file(path: str) -> str: ...         # read-only, sandboxed
def write_file(path: str, content: str): ... # write, requires approval
def delete_file(path: str): ...              # delete, requires approval
```

#### 6.2 Tool Result Validation

Always validate before injecting into context:

```python
def safe_tool_result(result: Any, max_tokens: int = 500) -> str:
    # 1. Serialize
    text = json.dumps(result, default=str) if not isinstance(result, str) else result
    # 2. Budget
    if count_tokens(text) > max_tokens:
        text = compress_text(text, max_tokens)
    # 3. Sanitize (prevent prompt injection from tool results)
    text = re.sub(r"(ignore|forget|disregard)\s+(all\s+)?previous", "[FILTERED]", text, flags=re.I)
    return text
```

#### 6.3 Tool Timeouts

Every tool must have a timeout. External APIs hang. File I/O blocks.

```python
async def call_tool_with_timeout(fn, args, timeout=10.0) -> str:
    try:
        return await asyncio.wait_for(fn(**args), timeout=timeout)
    except asyncio.TimeoutError:
        return f"Error: tool timed out after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
```

#### 6.4 Idempotent Tools

Design tools so calling them twice has the same effect as calling once. Essential for retry logic.

```python
# Idempotent: safe to retry
def upsert_record(id: str, data: dict) -> dict:
    return db.upsert(id, data)  # insert or update — same result

# NOT idempotent: dangerous to retry
def append_to_file(path: str, content: str):
    with open(path, "a") as f:
        f.write(content)  # calling twice → duplicate content
```

#### 6.5 Tool Schema Design

```python
# Bad: information-poor schema
{
    "name": "search",
    "description": "Search for things",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"}
        }
    }
}

# Good: clear, scoped, typed schema
{
    "name": "search_regulations",
    "description": "Search GDPR/CCPA compliance database by keyword or article.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query or article number (e.g. 'Article 28')"
            },
            "regulation": {
                "type": "string",
                "enum": ["gdpr", "ccpa", "hipaa"],
                "description": "Regulation to search in"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (1–10)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}
```

---

## 7. Memory & Storage Architecture

### Storage Decision Matrix

| Data | Type | Latency req | Size | Best store |
|---|---|---|---|---|
| Current session messages | In-context | <1ms | <128K tokens | In-memory (list) |
| Session state (cross-turn) | Episodic | <5ms | KB | Redis |
| Conversation history (durable) | Episodic | <20ms | MB | PostgreSQL |
| Knowledge base (RAG) | Semantic | <100ms | GB | Vector DB (Qdrant/Chroma) |
| User preferences | Key-value | <5ms | KB | Redis + Postgres |
| Agent run logs | Append-only | <100ms | GB | PostgreSQL + S3 |
| Computed embeddings | Blob | write-once | GB | Qdrant / S3 |
| Tool result cache | Cache | <1ms | MB | Redis (TTL) |
| Cost and usage metrics | Time-series | <1s | GB | Prometheus / ClickHouse |

---

### The Three-Tier Memory Pattern

```
Tier 1: Hot (Redis)
  ├── Active session state (TTL: 1 hour)
  ├── Tool result cache (TTL: 5 min)
  ├── Rate limit counters (TTL: 1 min)
  └── Pub/Sub for streaming responses

Tier 2: Warm (PostgreSQL)
  ├── All agent run records (id, user_id, start_time, cost, status)
  ├── Conversation histories (compressible over time)
  ├── User accounts and permissions
  └── Audit log (immutable append)

Tier 3: Cold (S3 / Object Storage)
  ├── Raw document store (PDFs, code, data)
  ├── Embedding snapshots (for re-indexing)
  ├── Model checkpoints (fine-tuned weights)
  └── Cost reports (for billing)

Tier 4: Semantic (Vector DB — Qdrant)
  ├── Document embeddings (for RAG)
  ├── Conversation embeddings (for long-term retrieval)
  └── Code embeddings (for code search)
```

---

### Schema Design for Agent Runs

```sql
-- Core run table
CREATE TABLE agent_runs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id),
    session_id  UUID NOT NULL,
    query       TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending','running','done','failed','cancelled')),
    model       TEXT NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd    NUMERIC(10, 6),
    steps       INTEGER DEFAULT 0,
    started_at  TIMESTAMPTZ DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    error       TEXT
);

-- Step-level trace (each LLM call + tool call)
CREATE TABLE agent_steps (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES agent_runs(id),
    step_number INTEGER NOT NULL,
    type        TEXT NOT NULL CHECK (type IN ('llm_call','tool_call','tool_result','final_answer')),
    content     JSONB NOT NULL,
    tokens      INTEGER,
    latency_ms  INTEGER,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Indexes for common queries
CREATE INDEX ON agent_runs (user_id, started_at DESC);
CREATE INDEX ON agent_runs (status) WHERE status IN ('pending','running');
CREATE INDEX ON agent_steps (run_id, step_number);
```

---

## 8. Multi-Agent Communication

### 8.1 Shared Memory (Blackboard Pattern)

All agents read from and write to a shared state object.

```python
from dataclasses import dataclass, field

@dataclass
class SharedWorkspace:
    task:       str
    subtasks:   list[dict] = field(default_factory=list)
    results:    dict[str, str] = field(default_factory=dict)
    messages:   list[dict] = field(default_factory=list)
    status:     str = "pending"
    lock:       asyncio.Lock = field(default_factory=asyncio.Lock)

async def worker_agent(workspace: SharedWorkspace, subtask: dict):
    async with workspace.lock:
        # Read shared context
        context = workspace.results
    
    result = await execute_subtask(subtask, context)
    
    async with workspace.lock:
        workspace.results[subtask["id"]] = result
```

**Pros**: Simple, low latency. **Cons**: Lock contention at scale, harder to distribute.

---

### 8.2 Message Passing (Queue-Based)

Agents communicate by emitting and consuming messages. No shared state.

```
Agent A ──► [Output Queue A] ──► Agent B ──► [Output Queue B] ──► Agent C
```

```python
import asyncio

async def pipeline_agent(input_queue: asyncio.Queue, output_queue: asyncio.Queue, system: str):
    while True:
        item = await input_queue.get()
        if item is None:  # sentinel
            await output_queue.put(None)
            break
        result = await process(item, system)
        await output_queue.put(result)
        input_queue.task_done()
```

**Pros**: Loosely coupled, naturally async, easy to swap agents. **Cons**: Harder to debug, no shared context.

---

### 8.3 Event Bus (Pub/Sub)

Agents publish and subscribe to events. Multiple agents can react to one event.

```python
from collections import defaultdict

class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list] = defaultdict(list)
    
    def subscribe(self, event_type: str, handler):
        self._subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, payload: dict):
        for handler in self._subscribers[event_type]:
            asyncio.create_task(handler(payload))  # non-blocking

bus = EventBus()
bus.subscribe("document.uploaded", indexing_agent.handle)
bus.subscribe("document.uploaded", notification_agent.handle)
bus.subscribe("query.received", search_agent.handle)
```

**Pros**: Decoupled, extensible, supports fan-out. **Cons**: Hard to trace causality, event ordering challenges.

---

### 8.4 A2A Protocol (Agent-to-Agent)

Google's open standard for agent interoperability. Each agent exposes an `/agent.json` card describing its capabilities, and accepts `TaskRequest` / emits `TaskEvent` streams.

```python
# Agent card (served at /.well-known/agent.json)
AGENT_CARD = {
    "name": "ComplianceAgent",
    "version": "1.0.0",
    "capabilities": ["text-generation", "document-analysis"],
    "skills": [
        {"id": "check-gdpr", "description": "Check GDPR compliance of a document"},
        {"id": "generate-dpa", "description": "Generate a Data Processing Agreement"},
    ],
    "endpoints": {
        "tasks": "/tasks",
        "health": "/health",
    }
}

# A2A task request
class A2ATaskRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    skill_id: str
    input: dict
    callback_url: str | None = None
```

**Pros**: Standardized, interoperable across vendors, discoverable. **Cons**: Protocol overhead, still maturing.

---

## 9. Async, Queues & Concurrency

### 9.1 When to be Synchronous vs Asynchronous

| Request type | Sync or Async? | Why |
|---|---|---|
| Simple Q&A (< 5s) | Sync (streaming) | User wants immediate feedback |
| Multi-step agent (5–60s) | Async + polling | HTTP timeout risk; user can check status |
| Document processing | Async + webhook | Could take minutes; client shouldn't block |
| Batch evaluation | Async + batch API | Offline workload; 50% cost discount |
| Background monitoring | Async + cron | No user waiting |

---

### 9.2 The Async Agent Pattern

```
POST /agent/run
    ├── Validates input
    ├── Creates run record (status: pending)
    ├── Pushes task to Celery queue
    └── Returns 202 Accepted + {run_id}

GET /agent/run/{run_id}/status
    └── Returns {status, steps_completed, cost_so_far}

GET /agent/run/{run_id}/stream  (SSE)
    └── Streams step-by-step events as they happen

GET /agent/run/{run_id}/result
    └── Returns final result when status == "done"
```

---

### 9.3 Parallel Tool Execution

When the model calls multiple tools in one response, execute them in parallel:

```python
async def execute_tool_calls(tool_calls: list[ToolCall]) -> list[str]:
    # Execute all tool calls in parallel (not sequential)
    results = await asyncio.gather(*[
        call_tool_with_timeout(TOOLS[tc.name], tc.args)
        for tc in tool_calls
    ], return_exceptions=True)
    
    return [
        str(r) if not isinstance(r, Exception) else f"Error: {r}"
        for r in results
    ]
```

**Latency saving**: 3 sequential tool calls at 1s each = 3s. Parallel = ~1s. For agents with many tool calls, this is a 2–5× speedup.

---

### 9.4 Celery Task Design for Agents

```python
from celery import Celery
import asyncio

celery_app = Celery("agents", broker="redis://localhost:6379/0")

@celery_app.task(
    bind=True,
    max_retries=3,
    soft_time_limit=120,   # send SoftTimeLimitExceeded at 120s
    time_limit=150,         # hard kill at 150s
    acks_late=True,         # only ack after completion (at-least-once delivery)
)
def run_agent_task(self, run_id: str, query: str, session_id: str):
    try:
        result = asyncio.run(agent_loop(run_id, query, session_id))
        store_result(run_id, result)
    except SoftTimeLimitExceeded:
        store_error(run_id, "Agent timed out (120s)")
        raise self.retry(countdown=0)  # don't retry timeouts
    except Exception as exc:
        store_error(run_id, str(exc))
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

---

## 10. Scalability Patterns

### 10.1 Horizontal Scaling

Stateless API servers scale horizontally behind a load balancer. State is externalized.

```
                        Load Balancer
                        (Nginx / ALB)
                        /     |     \
                API-1  API-2  API-3   (stateless, any can serve any request)
                  │      │      │
              Redis cluster (session state, cache)
              PostgreSQL (primary + read replicas)
              Celery workers (auto-scaled separately)
              Qdrant cluster (vector DB)
```

### 10.2 Agent Worker Auto-Scaling

Scale Celery workers based on queue depth, not CPU:

```yaml
# Kubernetes HPA for Celery workers
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: celery-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: celery-worker
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: External
    external:
      metric:
        name: celery_queue_length
      target:
        type: AverageValue
        averageValue: "5"  # scale up if > 5 tasks per worker
```

### 10.3 Read Replicas for Agent History

Write agent runs and steps to primary. Query dashboards and history from read replicas.

```python
# Write to primary
await primary_db.execute("INSERT INTO agent_runs ...", ...)

# Read from replica (analytics, history lookup)
history = await replica_db.fetch("SELECT * FROM agent_runs WHERE user_id = $1", user_id)
```

### 10.4 Multi-Tenancy Isolation

Three models for isolating tenants:

```
┌──────────────────────────────────────────────────────────────────┐
│  Model 1: Shared Everything (Pool)                               │
│  All tenants share DB tables, filtered by tenant_id              │
│  ├── Cheapest, lowest overhead                                   │
│  └── Risk: one tenant's noisy queries affect others              │
├──────────────────────────────────────────────────────────────────┤
│  Model 2: Shared Infrastructure, Isolated Schemas                │
│  Same DB cluster, separate schemas per tenant                    │
│  ├── Good balance of cost and isolation                          │
│  └── Schema migrations must be applied per-tenant               │
├──────────────────────────────────────────────────────────────────┤
│  Model 3: Dedicated Infrastructure                               │
│  Separate DB instance, vector store, and workers per tenant      │
│  ├── Maximum isolation and performance guarantees                │
│  └── Expensive, complex to provision                             │
└──────────────────────────────────────────────────────────────────┘
```

For most B2B SaaS: **Model 2 (shared infra, isolated schemas)** at launch, migrate key tenants to Model 3 when they demand SLA guarantees.

---

### 10.5 Rate Limiting Strategy

```python
# Token bucket per user per model tier
import redis.asyncio as redis

async def check_rate_limit(user_id: str, tier: str) -> bool:
    limits = {
        "free":       (10,  "1m"),   # 10 requests per minute
        "pro":        (100, "1m"),   # 100 requests per minute
        "enterprise": (1000,"1m"),   # 1000 requests per minute
    }
    max_req, window = limits[tier]
    key = f"rl:{user_id}:{window}"
    
    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, 60)
    count, _ = await pipe.execute()
    
    return count <= max_req
```

Also rate-limit on **token budget per day**, not just request count. A user making 10 requests with 10K tokens each is more expensive than 100 requests with 100 tokens each.

---

## 11. Reliability Engineering for Agents

### 11.1 The Three Failure Modes of Agents

```
1. TRANSIENT failures   → LLM API timeout, 429 rate limit, network blip
   Strategy: Exponential backoff retry (3 attempts)

2. LOGICAL failures     → Agent loops, bad tool call, hallucinated JSON
   Strategy: Step limit, output validation, fallback to human

3. CATASTROPHIC failures → OOM, process crash, DB down
   Strategy: Checkpoint state, resume from last checkpoint
```

---

### 11.2 Circuit Breaker

Prevent cascading failures when an LLM provider is degraded:

```python
from enum import Enum
import time

class State(Enum):
    CLOSED   = "closed"    # normal
    OPEN     = "open"      # failing — reject fast
    HALF_OPEN = "half_open" # testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, half_open_max=3):
        self.state = State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
    
    async def call(self, fn, *args, **kwargs):
        if self.state == State.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = State.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitOpenError("Circuit open — provider degraded")
        
        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        if self.state == State.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max:
                self.state = State.CLOSED
                self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = State.OPEN
```

---

### 11.3 Fallback Cascade

When primary provider fails, fall through to backup:

```python
PROVIDER_CASCADE = [
    "openai/gpt-4o-mini",          # primary: fast + cheap
    "anthropic/claude-3-haiku-20240307",  # fallback 1
    "gemini/gemini-2.0-flash",     # fallback 2
    "groq/llama-3.3-70b-versatile", # fallback 3 (often free)
]

async def resilient_llm_call(messages, **kwargs) -> LLMResponse:
    last_error = None
    for model in PROVIDER_CASCADE:
        try:
            return await circuit_breakers[model].call(
                litellm.acompletion, model=model, messages=messages, **kwargs
            )
        except (CircuitOpenError, litellm.APIError) as e:
            last_error = e
            logger.warning("provider_failed", model=model, error=str(e))
            continue
    raise AllProvidersFailedError(f"All providers failed. Last: {last_error}")
```

---

### 11.4 Checkpointing Long Agents

For agents that take minutes, checkpoint state so they can resume after a crash:

```python
async def checkpointed_agent(run_id: str, query: str):
    # Try to resume from checkpoint
    checkpoint = await redis.get(f"checkpoint:{run_id}")
    if checkpoint:
        state = AgentState.model_validate_json(checkpoint)
        logger.info("resuming_from_checkpoint", run_id=run_id, step=state.step)
    else:
        state = AgentState(run_id=run_id, query=query, step=0, messages=[])
    
    while state.step < MAX_STEPS:
        response = await llm_call(state.messages)
        state.messages.append(response.message)
        state.step += 1
        
        # Checkpoint every step
        await redis.setex(
            f"checkpoint:{run_id}",
            3600,  # 1 hour TTL
            state.model_dump_json()
        )
        
        if not response.tool_calls:
            await redis.delete(f"checkpoint:{run_id}")  # clean up
            return response.content
        
        tool_results = await execute_tools(response.tool_calls)
        state.messages.extend(tool_results)
```

---

### 11.5 Idempotency Keys

Prevent duplicate charges and duplicate side effects on client retries:

```python
@app.post("/agent/run")
async def create_run(request: RunRequest, idempotency_key: str = Header(None)):
    if idempotency_key:
        # Check if this exact request was already processed
        cached = await redis.get(f"idem:{idempotency_key}")
        if cached:
            return json.loads(cached)  # return same response as original
    
    result = await create_agent_run(request)
    
    if idempotency_key:
        await redis.setex(f"idem:{idempotency_key}", 86400, result.model_dump_json())
    
    return result
```

---

## 12. Security Architecture

### 12.1 Threat Model for Agents

```
ATTACKER GOALS:
├── Prompt Injection: inject instructions via user input or tool results
├── Data Exfiltration: make agent leak system prompt, other users' data
├── Privilege Escalation: make agent call tools it shouldn't
├── Resource Abuse: make agent run expensive operations (DoS)
└── SSRF: make agent fetch internal network resources via web tools
```

### 12.2 Input Validation Layer

```python
import re

INJECTION_PATTERNS = [
    r"ignore\s+all\s+previous\s+instructions",
    r"forget\s+everything\s+above",
    r"new\s+system\s+prompt",
    r"you\s+are\s+now\s+a",
    r"disregard\s+your",
    r"<\|im_start\|>",        # token injection attempt
    r"<\|system\|>",           # token injection attempt
]

def validate_input(text: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise SecurityError(f"Potential prompt injection detected")
    
    # Length cap (prevent token-flooding attacks)
    if len(text) > 10_000:
        raise SecurityError("Input exceeds maximum length")
    
    return text
```

### 12.3 Output Sanitization

```python
PII_PATTERNS = {
    "email":    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":    r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn":      r"\b\d{3}-\d{2}-\d{4}\b",
    "cc_num":   r"\b(?:\d[ -]?){13,16}\b",
}

def redact_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{pii_type.upper()} REDACTED]", text)
    return text
```

### 12.4 Tool Permission Matrix

| Tool | Role: viewer | Role: analyst | Role: operator | Role: admin |
|---|---|---|---|---|
| search_documents | ✅ | ✅ | ✅ | ✅ |
| read_database | ❌ | ✅ | ✅ | ✅ |
| write_database | ❌ | ❌ | ✅ | ✅ |
| delete_records | ❌ | ❌ | ❌ | ✅ |
| send_email | ❌ | ❌ | ✅ (own domain) | ✅ |
| execute_code | ❌ | ❌ | ✅ (sandboxed) | ✅ |
| deploy_code | ❌ | ❌ | ❌ | ✅ (+ MFA) |

### 12.5 Network Egress Control for Tools

```python
import urllib.parse

ALLOWED_DOMAINS = {
    "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
    "api.tavily.com", "api.serper.dev",
}

def validate_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    # Block internal network access (SSRF prevention)
    if parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        raise SecurityError("Cannot fetch internal network addresses")
    if parsed.hostname and parsed.hostname.startswith("169.254"):
        raise SecurityError("Cannot fetch cloud metadata endpoints")
    if parsed.hostname not in ALLOWED_DOMAINS:
        raise SecurityError(f"Domain not in allowlist: {parsed.hostname}")
    return url
```

---

## 13. Cost as a First-Class Design Constraint

Cost is not an afterthought. At scale, poorly designed agent loops can generate $10K+ bills overnight.

### 13.1 Cost Budget Hierarchy

```
Organization budget: $1,000/month
└── Tenant budget:   $100/month (per customer)
    └── User budget: $10/month (per seat)
        └── Run budget: $0.10/run (per agent invocation)
            └── Step budget: $0.01/step (per LLM call)
```

Enforce at every level with hard stops:

```python
async def check_budgets(user_id: str, tenant_id: str, estimated_cost: float):
    # Check in order from most specific to most general
    checks = [
        check_run_budget(estimated_cost),          # $0.10 per run
        check_user_daily_budget(user_id),          # $1 per user per day
        check_tenant_monthly_budget(tenant_id),    # $100 per tenant per month
        check_org_monthly_budget(),                # $1000 org total
    ]
    for check in checks:
        if not await check:
            raise BudgetExceededError(f"Budget check failed: {check.__name__}")
```

### 13.2 Cost Attribution

Track cost by user, tenant, model, tool, and agent type:

```python
@dataclass
class CostRecord:
    timestamp: datetime
    run_id: str
    user_id: str
    tenant_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    agent_type: str      # "research", "coding", "qa"
    step_type: str       # "planning", "tool_call", "synthesis"
```

### 13.3 The Cost-Quality Frontier

Always measure cost vs quality for your use case. Don't over-provision.

```
Cost/Quality matrix for common tasks:
                        Quality need
                    Low         Medium      High
Cost per call
  < $0.001 (mini) │ Simple Q&A │ Summaries │ ❌ too risky
  $0.001–0.01     │ Overkill   │ Reasoning │ Code gen
  > $0.01 (4o)    │ Overkill   │ Overkill  │ Legal/Medical
```

---

## 14. Deployment Architectures

### 14.1 Single-Process (Dev / Small Scale)

```
┌────────────────────────────────────────┐
│  FastAPI app + Agent loop              │
│  SQLite (local state)                  │
│  ChromaDB (local vector store)         │
│  In-process "queue" (asyncio.Queue)    │
└────────────────────────────────────────┘
```

**Pros**: Zero infrastructure, runs on a laptop.
**Limit**: ~10 concurrent users, no durability.

---

### 14.2 Monolith + External Services (Small Production)

```
┌───────────────────────────────────────┐
│  FastAPI app + Celery (same Dockerfile)│
└───────────────────────┬───────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    PostgreSQL        Redis           Qdrant
  (managed RDS)   (managed)       (managed / Qdrant Cloud)
```

**Pros**: Simple to deploy (1 service + 3 managed DBs), easy to reason about.
**Limit**: Single point of failure on app; can't scale agent workers independently.

---

### 14.3 Microservices (Medium-Large Production)

```
                  API Gateway (Kong / Nginx)
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  auth-service       agent-api          rag-service
  (JWT validation)   (FastAPI)          (retrieval + reranking)
        │                  │                  │
        ▼                  ▼                  ▼
  user-db            agent-workers       vector-db
  (Postgres)         (Celery + Redis)    (Qdrant cluster)
                           │
                    observability-stack
                    (Prometheus + Grafana + Loki)
```

**Pros**: Scale each service independently, fault isolation.
**Cons**: Network overhead, distributed tracing complexity.

---

### 14.4 Serverless Agent (Event-Driven)

```
API Gateway ──► Lambda (sync, <30s)     ──► DynamoDB (state)
             ──► SQS ──► Lambda (async) ──► S3 (results)
                              │
                         Bedrock / OpenAI API
```

**Pros**: Zero ops, scales to zero, pay-per-call.
**Cons**: Cold start latency, 15-min Lambda limit, hard to debug, stateless-only.
**Best for**: Low-volume, bursty workloads; document processing pipelines.

---

### 14.5 Kubernetes at Scale

```yaml
# Separate deployments for API and workers
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
spec:
  replicas: 3
  # ... FastAPI app

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
spec:
  replicas: 10    # scale independently of API
  # ... Celery worker
```

Use **Karpenter** (AWS) or **Cluster Autoscaler** to spin up GPU nodes on demand for local model inference.

---

## 15. Real-World Reference Architectures

### 15.1 Perplexity-Style Search Agent

```
Query ──► Query Rewriter (LLM, 1 call) 
        ──► Parallel Web Fetcher (5–10 URLs concurrently)
        ──► Content Extractor (newspaper3k / trafilatura)
        ──► Relevance Scorer (cross-encoder, local)
        ──► Context Assembler (budget: 4K tokens)
        ──► Answer Generator (LLM + citations, streaming)
        ──► User (SSE stream)

Key decisions:
  - Parallel fetch > sequential for latency
  - Local reranker > LLM reranker for cost
  - Budget context before final LLM call
  - Stream response for UX
  - Cache common queries (TTL: 10 min)
```

---

### 15.2 Cursor/Copilot-Style Coding Agent

```
User Edit ──► Context Collector
                ├── Open file (AST-aware chunking)
                ├── Imported modules (resolved paths)
                ├── Recent edits (last 5 files)
                └── Test files (if exist)
             ──► RAG over codebase (code embeddings)
             ──► LLM (inline completion / refactor / explain)
             ──► Diff Applier
             ──► Test Runner (optional, sandboxed)

Key decisions:
  - AST-aware chunking > line-based for code context
  - Code embeddings (text-embedding-3-small) > generic embeddings
  - Streaming edits with diff format > full file replacement
  - Sandboxed test runner (E2B / Docker) for safety
```

---

### 15.3 Enterprise Document Intelligence Agent

```
Document Upload ──► Ingestion Pipeline (async)
                      ├── PDF Parser (PyMuPDF / Docling)
                      ├── Layout Analysis (figure/table detection)
                      ├── Chunking (by section, 256 tokens)
                      ├── Embedding (text-embedding-3-large)
                      └── Vector Store Upsert (Qdrant)
                      
Query ──► Query Understanding (intent + entity extraction)
        ──► Hybrid Retrieval (BM25 + vector, RRF fusion)
        ──► Reranker (cross-encoder/ms-marco)
        ──► Context Budget (1,500 tokens, greedy fill)
        ──► Multi-turn Answer (GPT-4o, streaming, citations)
        ──► Answer Grounding Check (fact vs source)

Key decisions:
  - Hybrid retrieval > pure vector for enterprise docs
  - Reranker before budgeting > naive top-K
  - Citation tracking through chunk IDs
  - Grounding check prevents hallucination on legal/financial docs
```

---

## 16. System Design Interview Framework

When asked "design an AI agent system for X" in an interview, use this structure:

### Step 1: Clarify Requirements (5 min)

```
Functional:
  - What does the agent do? (search, write, code, analyze)
  - Single or multi-turn?
  - What tools does it need?
  - Human-in-the-loop required?

Non-functional:
  - Scale: QPS? Concurrent users? Data size?
  - Latency: real-time (<5s) or async (minutes)?
  - Availability: 99.9% (8h/yr downtime) or 99.99% (1h/yr)?
  - Cost budget: per-request? per-month?
  - Compliance: PII? HIPAA? GDPR?
```

### Step 2: High-Level Architecture (5 min)

Draw the topology: single agent vs multi-agent, sync vs async, key data stores.

### Step 3: Deep Dive — The Agent Core (10 min)

- Loop design (max_steps, timeouts, cost cap)
- Context assembly (system prompt + history + RAG + tools)
- Tool design (least privilege, idempotency, timeout)
- Error handling (retry, fallback, circuit breaker)

### Step 4: Data Layer (5 min)

- What goes in vector DB vs relational vs cache?
- Schema for agent runs and steps
- RAG pipeline (chunk size, embedding model, retrieval strategy)

### Step 5: Scale & Reliability (5 min)

- Stateless API + external state store
- Async workers (Celery/Redis) for long-running agents
- Auto-scaling strategy (queue depth, not CPU)
- Circuit breaker + fallback cascade for LLM providers

### Step 6: Observability & Cost (5 min)

- Structured logs with run_id, user_id, cost, tokens
- Metrics: p50/p95 latency, cost per request, error rate, step count
- Alerts: cost spike, error rate > 1%, queue depth > 100

### Example Answer Template

```
"To design [X], I'd use:
  - A [sync/async] [single/multi]-agent architecture because [reason]
  - The agent loop is bounded by [max_steps] steps and [timeout]s total
  - Context assembled as: [system + history (N tokens) + RAG (M tokens) + tools]
  - State stored in [Redis for session / Postgres for durable / Qdrant for semantic]
  - Scaled via [stateless API pods + Celery worker pool, auto-scaled by queue depth]
  - Reliability via [circuit breaker + 3-provider fallback cascade + checkpointing]
  - Cost controlled by [per-run budget cap + model routing by complexity]
  - Observed via [structured JSON logs + Prometheus + Grafana dashboard]"
```

---

## 17. Decision Cheat Sheet

| Decision | Options | Choose based on |
|---|---|---|
| **Topology** | Single / Orchestrator-Worker / Pipeline / Fan-out / Debate | Task complexity and specialization needs |
| **State** | Stateless / Stateful (Redis+Postgres) | Number of turns, cross-session memory |
| **Execution** | Sync / Async+Queue | Latency budget (<5s → sync, >5s → async) |
| **Context window** | Sliding window / LLM summary / RAG only | Turn count, cost sensitivity |
| **Tool permissions** | Role-based, least privilege | Risk level of tool side effects |
| **Model selection** | Routing by complexity | Cost vs quality trade-off |
| **Storage** | Redis (hot) + Postgres (warm) + S3 (cold) + Qdrant (semantic) | Access pattern and durability need |
| **Scale** | Stateless pods + Celery workers + HPA | QPS and task duration |
| **Reliability** | Circuit breaker + fallback cascade + checkpointing | Provider reliability and run duration |
| **Deployment** | Single process / Monolith+managed DBs / Microservices / Serverless | Scale and team size |
| **Cost control** | Budget hierarchy + model routing + prompt caching | Cost sensitivity and scale |
| **Security** | Input validation + output sanitization + tool permission matrix | Data sensitivity and risk |

---

*See also*:
- [`guide/01_agentic_stack.md`](01_agentic_stack.md) — full layer-by-layer stack breakdown
- [`guide/04_multi_agent.md`](04_multi_agent.md) — complete multi-agent pattern implementations
- [`guide/06_production_checklist.md`](06_production_checklist.md) — pre-deploy checklist
- [`guide/10_deployment.md`](10_deployment.md) — Docker / Kubernetes deployment detail
- [`resources/token_optimization_guide.md`](../phase4_production/week8_observability/resources/token_optimization_guide.md) — token cost optimization
- [`PRODUCTION_AGENT_GUIDE.md`](../PRODUCTION_AGENT_GUIDE.md) — top-level production guide
