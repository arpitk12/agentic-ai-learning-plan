# Week 16 — Concept Guide: Graph RAG · Resilience · A2A · Multi-Tenancy · Reasoning

> **How to use this file**: Read this *before* `notes.md`. This file explains the *why* and the mental model in plain English — no code. Once you understand the concept, `notes.md` shows you the implementation.

---

## Concept 1 — Graph RAG: When Vector Search Is Not Enough

### The limits of vector similarity

Standard RAG retrieves chunks whose *text content* is similar to your query. This works when the answer is contained within a single chunk. It breaks when:

**Multi-hop queries**: "What is the privacy policy of the parent company of the vendor in our Q3 contract?"
To answer this, you need to:
1. Find the vendor in the Q3 contract
2. Find that vendor's parent company
3. Find the parent company's privacy policy
Each step requires the result of the previous step. Vector search has no way to "follow the chain."

**Relationship queries**: "Which of our contracts share the same legal entity?"
This is asking about connections between documents, not about any single document's content.

**Aggregate queries**: "How many vendors in our system are GDPR non-compliant?"
Counting across all documents requires visiting all of them — not just the top-K most similar.

### What a knowledge graph adds

A knowledge graph stores entities (things) and relationships (connections between things):

```
[Acme Corp] -[SIGNED]→ [Contract_Q3_2024]
[Acme Corp] -[SUBSIDIARY_OF]→ [GlobalTech Inc]
[GlobalTech Inc] -[HAS_POLICY]→ [GDPR_Policy_v2]
[Contract_Q3_2024] -[GOVERNED_BY]→ [EU_Law_2024]
```

You can now traverse this graph to answer multi-hop questions that are impossible for vector search.

### How Graph RAG combines both

Graph RAG uses:
1. **Vector search** to find the starting node (which document/entity is most relevant)
2. **Graph traversal** to follow relationships and gather connected information
3. **LLM** to synthesise a natural language answer from the graph results

Microsoft's GraphRAG paper showed that on multi-hop question answering benchmarks, graph-based retrieval significantly outperforms pure vector RAG — especially on questions about community structure and cross-document relationships.

---

## Concept 2 — Neo4j and Cypher

### What Neo4j is

Neo4j is a **graph database** — a database purpose-built to store and query nodes and relationships. It stores data as a graph natively, which makes traversal queries extremely fast (milliseconds for 5-hop traversals that would require complex JOINs in SQL).

### What Cypher is

Cypher is Neo4j's query language. Instead of table-based SQL, it uses a visual pattern syntax that looks like the graph itself:

SQL:
```sql
SELECT v.name FROM vendor v JOIN contract c ON c.vendor_id = v.id WHERE c.date > '2024-01-01'
```

Cypher:
```
MATCH (c:Contract)-[:SIGNED_BY]->(v:Vendor)
WHERE c.date > '2024-01-01'
RETURN v.name
```

The arrow `→` literally means "follow this relationship." The parentheses `()` are nodes. This visual syntax makes it easier to express complex graph patterns.

### Why LLMs can write Cypher

Cypher's pattern syntax is close to natural language descriptions of graph structure. LLMs trained on code can generate Cypher queries from English questions, given a schema description. This is how Graph RAG works in practice: you give the LLM the schema and it generates the traversal query.

---

## Concept 3 — Circuit Breakers: Stop Failing Fast

### The problem

Your agent calls an external service (an LLM API, a database, a third-party API). That service starts responding slowly. What happens?

Without protection:
- Every request hangs waiting for the slow service
- Your agent's thread pool fills up
- Your entire agent becomes unresponsive
- Users get timeout errors after a long wait

This is **cascading failure** — one slow service takes down everything that depends on it.

### What a circuit breaker does

A circuit breaker wraps calls to an external service and tracks the failure rate. It has three states:

**CLOSED** (normal operation)
- Requests pass through normally
- Breaker tracks failures and successes
- If failures exceed threshold → trip to OPEN

**OPEN** (service is failing)
- ALL requests immediately fail without trying the actual service
- Returns a fast error instead of waiting for timeout
- Fail fast protects your thread pool and other services
- After a timeout → move to HALF_OPEN

**HALF_OPEN** (testing if service has recovered)
- Allow a small number of test requests through
- If they succeed → close the breaker (service recovered)
- If they fail → go back to OPEN

**The key benefit**: Failing fast is better than failing slow. A 5ms error lets you try a fallback. A 30-second timeout has already wasted 30 seconds of your user's time.

### Analogy

A circuit breaker works exactly like an electrical circuit breaker:
- Normal: circuit closed, current (requests) flows
- Overload: circuit trips to open, current stops
- After cooldown: test if safe to restore

---

## Concept 4 — Fallback Chains

A fallback chain defines what to do when a primary system fails. It is a list of increasingly degraded but always available options:

```
Try: GPT-4o (best quality, may be slow or expensive)
  ↓ fails
Try: Claude Sonnet (slightly different quality)
  ↓ fails
Try: GPT-4o-mini (cheaper, lower quality but functional)
  ↓ fails
Return: cached answer from 10 minutes ago
  ↓ no cache
Return: static fallback message ("Service temporarily unavailable")
```

The key principle: **always have a response, even if it is degraded.** A graceful degradation is almost always better than an exception.

---

## Concept 5 — Saga Pattern: Distributed Transactions for Agents

### The problem

Your agent executes a multi-step workflow:
1. Send contract to legal review system
2. Update contract status in database
3. Notify client by email
4. Archive original contract

Steps 1-3 succeed. Step 4 fails. Now you have a half-completed state: the client was notified but the contract is not archived.

In a traditional database, you'd use a **transaction**: if any step fails, roll back all steps. But when each step is a call to a different service (LLM, database, email API, file system), there is no global transaction manager. You can't roll back an email already sent.

### What the Saga pattern is

The Saga pattern is the solution for long-running distributed transactions. For each step, you define:
- **Action**: what to do
- **Compensating action**: how to undo it if a later step fails

```
Step 1: Submit for legal review    ↔ Compensate: Withdraw submission
Step 2: Update status to "pending" ↔ Compensate: Revert status to "draft"
Step 3: Send notification          ↔ Compensate: Send cancellation email
Step 4: Archive document           ↔ Compensate: Restore from archive
```

If step 4 fails, the Saga executes compensating actions for steps 3, 2, 1 in reverse order — bringing the system back to a consistent state.

### Why agents specifically need Sagas

Agents execute long action sequences with real side effects (API calls, database writes, emails sent). Partial execution leaves systems in inconsistent states. Sagas give you a structured way to guarantee either full completion or full rollback.

---

## Concept 6 — Dead Letter Queue (DLQ)

### What it is

A Dead Letter Queue is a place where messages go when they have failed processing too many times. Instead of losing them (dropping the request) or getting stuck retrying forever, they are moved to the DLQ for inspection and manual or automated recovery.

### The pattern

```
Message arrives → Agent tries to process
    ↓ fails
    Retry after 1 second → fails
    Retry after 2 seconds → fails
    Retry after 4 seconds → fails
    → Move to Dead Letter Queue (don't retry again)
    → Alert operator
    → Operator inspects DLQ, fixes root cause, reprocesses
```

### Why this matters

Without a DLQ:
- Failed messages are silently dropped → data loss
- Or failed messages retry forever → infinite loops, resource waste

With a DLQ:
- No data loss (every failed message is stored)
- Automatic retry with backoff (usually exponential: 1s, 2s, 4s, 8s, 16s, ...)
- Visibility into what is failing and why
- Ability to replay messages after fixing the bug

---

## Concept 7 — Idempotency

### What it means

An operation is **idempotent** if executing it multiple times produces the same result as executing it once.

- **Idempotent**: "Set the contract status to APPROVED" (safe to repeat — result is the same)
- **Not idempotent**: "Send approval email" (repeating sends duplicate emails)

### Why agents need it

Agents retry failed operations. Networks have transient failures. The same message can arrive twice (exactly-once delivery is very hard). If your agent is not idempotent, retries cause:
- Duplicate emails sent
- Double charges processed
- Contracts reviewed twice and logged as two separate reviews

### How idempotency is implemented

Assign every operation a unique idempotency key (usually a hash of the inputs or a UUID). Before executing, check if you have already executed with this key. If yes, return the previously stored result. If no, execute and store the result.

The check-and-store must be atomic (one database transaction) to be reliable.

---

## Concept 8 — Agent-to-Agent (A2A) Protocol

### The problem

You have multiple AI agents built with different frameworks:
- A compliance agent built with LangGraph
- A contract analysis agent built with PydanticAI
- A notification agent built with a custom framework

You want them to collaborate. How do they discover each other? How do they communicate? How does authentication work when they call each other?

### What A2A is

Google's A2A (Agent-to-Agent) protocol is an open standard for inter-agent communication. It defines:

**Agent Cards** — a JSON document published at `/.well-known/agent.json` describing:
- What the agent can do (capabilities)
- What inputs it accepts
- How to authenticate (API keys, OAuth, JWT)
- Contact URL and metadata

**Task protocol** — a standard way to send tasks, poll for status, and receive results:
```
POST /tasks → { "task": "review this contract", "inputs": {...} }
     → returns task_id
GET /tasks/{task_id} → { "status": "completed", "result": {...} }
```

**Why this matters**: Without A2A, every agent-to-agent integration requires custom code. With A2A, any A2A-compliant agent can discover and call any other A2A-compliant agent through a standard interface.

### Analogy

A2A is the HTTP of the agent world. HTTP standardised how browsers and servers communicate. A2A standardises how agents communicate with each other.

---

## Concept 9 — Multi-Tenancy

### What multi-tenancy is

Multi-tenancy means a single instance of your software serves multiple customers (tenants), while keeping each customer's data and state completely isolated from others.

### The three problems to solve

**1. State isolation**
Customer A's conversation history must never be visible to Customer B. In LangGraph this means every customer gets a unique `thread_id` (namespace). In a vector store, every customer gets a separate collection or a metadata filter.

**2. Rate limiting**
Customer A should not be able to consume all your API capacity, leaving no capacity for Customer B. Solution: per-tenant token buckets enforced by Redis.

**Token bucket algorithm**:
- Each tenant has a bucket with capacity N tokens
- Each request consumes tokens proportional to usage
- Tokens refill at a fixed rate (e.g., 10,000 tokens/minute)
- When the bucket is empty, requests are rejected until tokens refill

**3. Cost tracking**
You need to know exactly how much each customer costs you (for billing, for usage limits, for capacity planning). Log every LLM call with the tenant ID, model, input tokens, and output tokens.

### The security principle

**Principle of least privilege**: Each tenant's requests should run with only the permissions needed for that tenant, never with admin-level access. A request from Customer A should be able to read only Customer A's data.

---

## Concept 10 — Tree of Thought (ToT) Reasoning

### The limit of linear reasoning

Chain-of-Thought (CoT) reasoning has the model think step by step:

```
Step 1 → Step 2 → Step 3 → Answer
```

The problem: if step 1 takes a wrong turn, every subsequent step is built on a bad foundation. The model cannot backtrack. This is like solving a maze by always going forward — you get stuck in dead ends.

### What Tree of Thought does

Tree of Thought generalises CoT by having the model maintain multiple reasoning branches simultaneously:

```
                    Question
                 /     |      \
           Path A   Path B   Path C
          (explore)(explore)(explore)
           /    \      |
        A.1    A.2   B.1   
       (prune) (keep) (keep)
                |       |
               A.2.1  B.1.1
                |       |
             Answer    Answer
```

After exploring each branch to some depth, a scoring function evaluates how promising each branch is. Low-scoring branches are pruned. High-scoring branches are explored further. This is essentially **best-first search** (or beam search) applied to reasoning.

### When ToT wins

ToT significantly outperforms CoT on:
- Mathematical problems (multiple solution approaches to try)
- Logic puzzles (requires backtracking when a path is wrong)
- Strategic planning (multiple plans to evaluate)
- Any task where the solution space has many local optima

### The cost

ToT requires multiple LLM calls per problem (one per branch, multiple levels deep). A tree of depth 3 with branching factor 3 might require 3 + 9 + 27 = 39 calls. Use ToT selectively for genuinely hard problems, not routine tasks.

### o3 and extended thinking

OpenAI's o3 and Anthropic's Claude extended thinking implement a learned version of tree-like reasoning. Instead of you managing the tree explicitly, the model has been trained to internally explore multiple paths and discard unpromising ones. This is more efficient but less controllable than explicit ToT.

---

## Key Takeaways

- **Graph RAG**: use when queries require following relationships across documents (multi-hop); combine with vector RAG for best results
- **Neo4j/Cypher**: graph database + query language; LLMs can generate Cypher queries from English descriptions
- **Circuit breaker**: CLOSED→OPEN→HALF_OPEN; fail fast to protect downstream services
- **Fallback chain**: always have a degraded response rather than an exception
- **Saga**: for multi-step agent workflows, define compensating actions for each step so you can roll back partially completed work
- **DLQ**: store failed messages for inspection; never silently drop or retry forever
- **Idempotency**: assign unique keys to operations; check before executing; safe to retry
- **A2A protocol**: Google's standard for agent-to-agent communication; agent cards + task protocol; enables cross-framework agent collaboration
- **Multi-tenancy**: isolate state by tenant (thread namespaces), rate-limit per tenant (Redis token bucket), track cost per tenant
- **Tree of Thought**: explores multiple reasoning branches simultaneously; prunes bad paths; significantly better than CoT on complex problems but costs many more LLM calls
