# Week 16 — Graph RAG · Resilience · A2A · Multi-Tenancy · Advanced Reasoning

## What This Week Is About

The final five gaps for fully production-ready agents:

1. **Graph RAG** — multi-hop reasoning over knowledge graphs (where vector RAG fails)
2. **Failure resilience** — circuit breakers, fallback chains, saga pattern
3. **A2A protocol** — agent-to-agent communication across runtimes
4. **Multi-tenancy** — isolate state, rate limits, and cost per customer
5. **Advanced reasoning** — Tree of Thought, o3-style extended thinking

---

## 1. Graph RAG — When Vector RAG Fails

Vector RAG retrieves top-K similar chunks. It fails on:
- **Multi-hop questions**: "What policy does the vendor's parent company use?"
- **Relationship queries**: "Which contracts share the same legal entity?"
- **Aggregation**: "How many vendors in our system are GDPR non-compliant?"

**Solution**: Knowledge graph + LLM-generated Cypher queries.

```python
# pip install neo4j spacy
# python -m spacy download en_core_web_sm
import spacy
from neo4j import GraphDatabase

nlp = spacy.load("en_core_web_sm")

# 1. Entity extraction from documents
def extract_entities(text: str) -> list[dict]:
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char}
        for ent in doc.ents
        if ent.label_ in {"ORG", "PERSON", "GPE", "LAW", "DATE", "MONEY"}
    ]

# 2. Build knowledge graph
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def add_document_entities(doc_id: str, entities: list[dict], relationships: list[dict]):
    with driver.session() as session:
        session.run("MERGE (d:Document {id: $id})", id=doc_id)
        for ent in entities:
            session.run(
                "MERGE (e:Entity {name: $name, type: $type})"
                "MERGE (d:Document {id: $doc_id})-[:MENTIONS]->(e)",
                name=ent["text"], type=ent["label"], doc_id=doc_id,
            )

# 3. LLM generates Cypher query from natural language
async def graph_rag_query(question: str) -> str:
    schema = """
    Nodes: Document(id, title, type), Entity(name, type), Policy(id, name, regulation)
    Edges: MENTIONS, GOVERNED_BY, OWNED_BY, SHARES_ENTITY_WITH
    """
    cypher_prompt = f"""
    Schema: {schema}
    Question: {question}
    Write a Cypher query to answer this question. Return only the Cypher, no explanation.
    """
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": cypher_prompt}],
    )
    cypher = response.choices[0].message.content.strip()
    
    with driver.session() as session:
        result = session.run(cypher)
        records = [dict(r) for r in result]
    
    # Synthesize answer from graph results
    answer_prompt = f"Question: {question}\nGraph results: {records}\nAnswer concisely:"
    final = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
    )
    return final.choices[0].message.content
```

### Microsoft GraphRAG (Automated)

```python
# pip install graphrag
# graphrag init --root ./graph_rag_project
# graphrag index --root ./graph_rag_project
# → Auto-extracts entities, builds community summaries, enables global search

from graphrag.query.cli import run_global_search, run_local_search

# Global search: high-level questions about the entire corpus
result = await run_global_search("What are the main compliance risk themes across all contracts?")

# Local search: specific questions with entity context
result = await run_local_search("Which vendors are flagged for GDPR non-compliance?")
```

---

## 2. Failure Resilience Patterns

### Circuit Breaker

Stops calling a failing service. After N failures, opens the circuit for T seconds.

```python
# pip install tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: int = 30   # seconds
    _failures: int = field(default=0, init=False)
    _opened_at: datetime | None = field(default=None, init=False)
    
    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if datetime.now() - self._opened_at > timedelta(seconds=self.recovery_timeout):
            self._opened_at = None
            self._failures = 0
            return False   # half-open: let one request through
        return True
    
    def record_failure(self):
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = datetime.now()
    
    def record_success(self):
        self._failures = 0
        self._opened_at = None

circuit = CircuitBreaker()

async def call_with_circuit_breaker(model: str, messages: list) -> str:
    if circuit.is_open:
        raise RuntimeError(f"Circuit open — {model} is unavailable")
    try:
        result = await litellm.acompletion(model=model, messages=messages)
        circuit.record_success()
        return result.choices[0].message.content
    except Exception as e:
        circuit.record_failure()
        raise
```

### Fallback Model Chain

```python
FALLBACK_CHAIN = [
    "openai/gpt-4o-mini",     # primary
    "anthropic/claude-3-haiku",  # fallback 1
    "groq/llama-3.3-70b-versatile",  # fallback 2 (free)
    "ollama/llama3.2",           # fallback 3 (local, always available)
]

async def resilient_completion(messages: list, **kwargs) -> str:
    last_error = None
    for model in FALLBACK_CHAIN:
        try:
            result = await litellm.acompletion(model=model, messages=messages, **kwargs)
            if model != FALLBACK_CHAIN[0]:
                logger.warning("fallback_used", primary=FALLBACK_CHAIN[0], actual=model)
            return result.choices[0].message.content
        except Exception as e:
            logger.warning("model_failed", model=model, error=str(e))
            last_error = e
            continue
    raise RuntimeError(f"All models in fallback chain failed: {last_error}")
```

### Saga Pattern — Rollback Multi-Step Operations

```python
from dataclasses import dataclass
from typing import Callable, Awaitable

@dataclass
class SagaStep:
    name: str
    action: Callable[..., Awaitable]
    compensate: Callable[..., Awaitable]  # undo this step

async def execute_saga(steps: list[SagaStep], context: dict) -> dict:
    """Execute steps in order. If any fails, compensate all completed steps in reverse."""
    completed = []
    try:
        for step in steps:
            context = await step.action(context)
            completed.append(step)
        return context
    except Exception as e:
        # Compensate in reverse order
        for step in reversed(completed):
            try:
                await step.compensate(context)
            except Exception as comp_err:
                logger.error("compensation_failed", step=step.name, error=str(comp_err))
        raise RuntimeError(f"Saga failed at '{completed[-1].name}' + rolled back") from e

# Example: document review saga with rollback
steps = [
    SagaStep("reserve_slot", reserve_review_slot, release_review_slot),
    SagaStep("extract_doc", extract_document, delete_extraction),
    SagaStep("run_review", run_compliance_review, archive_failed_review),
    SagaStep("store_result", store_to_db, delete_from_db),
    SagaStep("notify_user", send_notification, send_failure_notification),
]
```

---

## 3. A2A Protocol — Agent-to-Agent Communication

Google's A2A protocol (2025) enables agents from different frameworks and vendors to call each other.

**Key concepts**:
- **Agent Card** (`/.well-known/agent.json`) — declares capabilities, auth, endpoint
- **Task** — structured unit of work sent between agents
- **Skill** — named capability an agent offers

```python
# Agent Card — make your agent discoverable
AGENT_CARD = {
    "name": "compliance-review-agent",
    "description": "Performs multi-jurisdiction compliance review on business documents",
    "url": "https://compliance.yourorg.com/a2a",
    "version": "1.0.0",
    "capabilities": {"streaming": True, "pushNotifications": False},
    "authentication": {"schemes": ["bearer"]},
    "skills": [
        {
            "id": "compliance_review",
            "name": "Compliance Review",
            "description": "Review a document for GDPR, SOX, HIPAA compliance",
            "inputModes": ["text/plain", "application/pdf"],
            "outputModes": ["application/json"],
        }
    ],
}

# FastAPI A2A endpoint
from fastapi import FastAPI, HTTPException
app = FastAPI()

@app.get("/.well-known/agent.json")
async def agent_card():
    return AGENT_CARD

@app.post("/a2a/tasks/send")
async def receive_task(task: dict, authorization: str = Header(...)):
    verify_bearer_token(authorization)   # JWT verification
    result = await compliance_agent.run(task["message"]["parts"][0]["text"])
    return {
        "id": task["id"],
        "status": {"state": "completed"},
        "result": {"parts": [{"type": "data", "data": result.model_dump()}]},
    }

# Calling another A2A agent
import httpx

async def delegate_to_legal_agent(document: str) -> dict:
    async with httpx.AsyncClient() as client:
        card = (await client.get("https://legal.yourorg.com/.well-known/agent.json")).json()
        task = {
            "id": str(uuid4()),
            "message": {"parts": [{"type": "text", "text": document}]},
            "skill": "legal_review",
        }
        response = await client.post(
            f"{card['url']}/tasks/send",
            json=task,
            headers={"Authorization": f"Bearer {get_service_token()}"},
        )
        return response.json()["result"]
```

---

## 4. Multi-Tenancy Patterns

```python
# Per-tenant LangGraph namespace isolation
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def get_tenant_graph(tenant_id: str):
    """Each tenant gets their own checkpoint namespace."""
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        graph = build_compliance_graph(checkpointer=saver)
        config = {
            "configurable": {
                "thread_id": f"tenant_{tenant_id}_session_1",
                "checkpoint_ns": f"tenant_{tenant_id}",  # namespace isolation
            }
        }
        return graph, config

# Per-tenant rate limiting
from collections import defaultdict
import time

class TenantRateLimiter:
    def __init__(self, requests_per_minute: dict[str, int]):
        self.limits = requests_per_minute   # {"free": 10, "pro": 100, "enterprise": 1000}
        self._windows: dict[str, list[float]] = defaultdict(list)
    
    def check(self, tenant_id: str, tier: str) -> bool:
        limit = self.limits.get(tier, 10)
        now = time.time()
        window = [t for t in self._windows[tenant_id] if now - t < 60]
        if len(window) >= limit:
            return False
        window.append(now)
        self._windows[tenant_id] = window
        return True

# Per-tenant cost allocation
from langfuse import Langfuse
langfuse = Langfuse()

def track_tenant_cost(tenant_id: str, session_id: str, doc_id: str):
    return langfuse.trace(
        name="compliance-review",
        session_id=session_id,
        user_id=tenant_id,
        tags=["tenant", tenant_id],
        metadata={"tenant_id": tenant_id, "doc_id": doc_id},
    )
```

---

## 5. Advanced Reasoning — Tree of Thought

For hard problems, explore multiple reasoning branches rather than one linear chain.

```python
from pydantic import BaseModel
from typing import Literal
import asyncio

class ThoughtNode(BaseModel):
    thought: str
    evaluation: Literal["promising", "partial", "dead_end"]
    score: float  # 0-1

async def tree_of_thought(problem: str, depth: int = 3, breadth: int = 3) -> str:
    """Explore multiple reasoning paths, select the best."""
    
    async def generate_thoughts(context: str) -> list[ThoughtNode]:
        """Generate multiple next steps from current context."""
        response = await litellm.acompletion(
            model="openai/gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Problem: {problem}\n\nCurrent thinking:\n{context}\n\n"
                           f"Generate {breadth} different next reasoning steps. "
                           "For each: state the thought, evaluate it (promising/partial/dead_end), "
                           "and score it 0-1. Return JSON array.",
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return [ThoughtNode(**t) for t in data["thoughts"]]
    
    # BFS: explore the most promising thoughts at each depth
    current_contexts = [problem]
    for _ in range(depth):
        all_thoughts = await asyncio.gather(*[
            generate_thoughts(ctx) for ctx in current_contexts
        ])
        # Keep top-breadth promising thoughts
        candidates = [
            (ctx, t) for ctx, thoughts in zip(current_contexts, all_thoughts)
            for t in thoughts
            if t.evaluation != "dead_end"
        ]
        candidates.sort(key=lambda x: x[1].score, reverse=True)
        current_contexts = [
            f"{ctx}\nStep: {t.thought}"
            for ctx, t in candidates[:breadth]
        ]
    
    # Synthesize best path into final answer
    best_chain = current_contexts[0]
    final = await litellm.acompletion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": f"Based on this reasoning chain, give a concise answer:\n\n{best_chain}"}],
    )
    return final.choices[0].message.content

# When to use o3/o3-mini extended thinking models
# Use o3 when: mathematical proofs, complex logic puzzles, multi-step planning
# Use o3-mini when: same but cost-sensitive
# Use gpt-4o when: creative tasks, conversational, straightforward generation
```

---

## Key Takeaways

1. **Graph RAG**: Use when questions require traversing relationships; Cypher queries bridge NL → graph
2. **Circuit breaker**: Open after N failures, recover after T seconds — prevents cascade failures
3. **Fallback chain**: Always have a local model as final fallback (always available, zero cost)
4. **Saga**: Treat multi-step agent actions as distributed transactions with compensation
5. **A2A agent card**: `/.well-known/agent.json` makes your agent discoverable and callable
6. **Multi-tenancy**: Namespace by `tenant_id` in checkpointer; rate limit and cost-track per tenant
7. **Tree of Thought**: BFS over reasoning branches — use for hard planning/logic tasks

---

## Exercises

- `ex1_graph_rag.py` — Extract entities → Neo4j graph → LLM-generated Cypher → multi-hop Q&A
- `ex2_resilience.py` — Circuit breaker + fallback chain + saga pattern for multi-step agent
