# Week 16 Resources — Graph RAG · Resilience · A2A · Multi-Tenancy · Advanced Reasoning

---

## 🏁 Start Here (Read in This Order)

1. **Martin Fowler — Circuit Breaker Pattern** (canonical description, 10-min read):
   https://martinfowler.com/bliki/CircuitBreaker.html

2. **Microsoft GraphRAG** — the paper that made Graph RAG mainstream:
   https://arxiv.org/abs/2404.16130

3. **Tree of Thought Paper** — understand the core idea in the abstract + Section 2:
   https://arxiv.org/abs/2305.10601

---

## Graph RAG and Knowledge Graphs

### Microsoft GraphRAG
- **GraphRAG Paper** (From Local to Global: A Graph RAG Approach to Query-Focused Summarisation):
  https://arxiv.org/abs/2404.16130
  *Read Section 1 (intro) and Section 3 (method). Explains clearly why vector RAG fails on global queries.*

- **Microsoft GraphRAG GitHub** (official implementation):
  https://github.com/microsoft/graphrag

- **Microsoft GraphRAG Docs**:
  https://microsoft.github.io/graphrag/

- **GraphRAG Accelerator** (Azure-hosted version for production):
  https://github.com/Azure-Samples/graphrag-accelerator

### Neo4j
- **Neo4j GraphAcademy** (free interactive courses, start with "Neo4j Fundamentals"):
  https://graphacademy.neo4j.com

- **Neo4j Docs — Cypher Manual**:
  https://neo4j.com/docs/cypher-manual/current/

- **Neo4j + LLM Integration Docs**:
  https://neo4j.com/docs/neo4j-graphrag-python/current/

- **Neo4j Vector Index** (for hybrid graph+vector search):
  https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/

- **"Building Knowledge Graphs with LLMs"** (Neo4j blog):
  https://neo4j.com/blog/developer/knowledge-graph-llm/

### spaCy for Entity Extraction
- **spaCy Named Entity Recognition**:
  https://spacy.io/usage/linguistic-features#named-entities

- **spaCy Models** (en_core_web_sm for speed, en_core_web_trf for quality):
  https://spacy.io/usage/models

- **spaCy Entity Ruler** (adding custom entity patterns):
  https://spacy.io/usage/rule-based-matching#entityruler

### LlamaIndex Graph RAG
- **LlamaIndex Knowledge Graphs Docs**:
  https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo/

---

## Resilience Patterns

### Circuit Breaker
- **Martin Fowler — Circuit Breaker Pattern**:
  https://martinfowler.com/bliki/CircuitBreaker.html
  *The canonical description. Read the whole article — it's only 1,200 words.*

- **"Release It!" Book** (Michael Nygard) — where circuit breakers in software originated:
  https://pragprog.com/titles/mnee2/release-it-second-edition/

- **Tenacity Library Docs** (retry + circuit-breaker-like patterns for Python):
  https://tenacity.readthedocs.io/en/latest/

- **Stamina** (modern alternative to Tenacity):
  https://stamina.hynek.me/en/stable/

### Saga Pattern
- **"Saga Pattern" on microservices.io** (Chris Richardson — canonical reference):
  https://microservices.io/patterns/data/saga.html

- **"Managing Distributed Transactions with the Saga Pattern"** (Microsoft Azure Docs):
  https://learn.microsoft.com/en-us/azure/architecture/reference-architectures/saga/saga

- **Choreography vs Orchestration Sagas** (which type to use when):
  https://temporal.io/blog/saga-pattern

- **Temporal** (workflow engine that implements Saga for Python natively):
  https://docs.temporal.io/develop/python

### Dead Letter Queue
- **"What is a Dead Letter Queue?"** (AWS explainer, readable even if you don't use AWS):
  https://aws.amazon.com/what-is/dead-letter-queue/

- **RQ (Redis Queue) — Python background jobs with DLQ support**:
  https://python-rq.org

- **Celery — Dead Letter Queue pattern**:
  https://docs.celeryq.dev/en/stable/userguide/tasks.html#retrying

### Idempotency
- **Stripe's guide to idempotency** (the gold standard example — their Idempotency-Key header):
  https://stripe.com/docs/api/idempotent_requests

- **"Idempotency Key Pattern"** (microservices.io):
  https://microservices.io/patterns/communication-style/idempotent-consumer.html

### Fallback and Bulkhead Patterns
- **"Bulkhead Pattern"** (Microsoft Docs):
  https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead

- **litellm Fallbacks** (built-in fallback routing across LLM providers):
  https://docs.litellm.ai/docs/routing#fallbacks

---

## Agent-to-Agent (A2A) Protocol

### Official Spec
- **Google A2A Protocol — Official Docs**:
  https://google.github.io/A2A/

- **A2A GitHub** (spec + reference implementation):
  https://github.com/google/A2A

- **A2A Python SDK**:
  https://github.com/google/A2A/tree/main/samples/python

### Agent Cards
- **A2A Agent Card Spec** (JSON schema for agent capability discovery):
  https://google.github.io/A2A/#/documentation?id=agent-card

### Related Standards
- **MCP (Model Context Protocol)** — Anthropic's standard for tool calling (complementary to A2A):
  https://modelcontextprotocol.io/introduction

- **"A2A vs MCP: What's the Difference?"** (good comparison):
  https://google.github.io/A2A/#/topics/a2a_and_mcp.md

### JWT Authentication (used in A2A exercises)
- **"JWT Introduction"** (jwt.io):
  https://jwt.io/introduction

- **PyJWT Library Docs**:
  https://pyjwt.readthedocs.io/en/stable/

---

## Multi-Tenancy

### Concepts and Patterns
- **"Multi-Tenancy Architecture Patterns"** (Stripe Engineering):
  https://stripe.com/blog/what-is-multitenancy

- **"LLM Applications: Multi-Tenant Design"** (practical guide):
  https://www.patterns.app/blog/2023/01/18/crm-ai-gpt3-email-summarization/

### LangGraph Multi-Tenancy
- **LangGraph State Namespacing** (thread_id isolation):
  https://langchain-ai.github.io/langgraph/concepts/persistence/

- **LangGraph Cloud — Multi-Tenant Support**:
  https://langchain-ai.github.io/langgraph/cloud/

### Rate Limiting
- **Redis Rate Limiting Docs** (token bucket + sliding window):
  https://redis.io/docs/manual/patterns/distributed-locks/

- **Limits Library** (Python rate limiter with Redis backend):
  https://limits.readthedocs.io/en/stable/

- **"Token Bucket vs Leaky Bucket vs Sliding Window"** (comparison article):
  https://blog.cloudflare.com/counting-things-a-lot-of-different-things/

### Cost Tracking
- **LangChain Callbacks for Cost Tracking**:
  https://python.langchain.com/docs/modules/callbacks/

- **Langfuse Cost Tracking** (per-user, per-model cost dashboards):
  https://langfuse.com/docs/model-usage-and-cost

---

## Advanced Reasoning

### Tree of Thought
- **Tree of Thought Paper** (Yao et al., 2023):
  https://arxiv.org/abs/2305.10601
  *Read the intro and Figures 1-2. The visual makes the concept immediately clear.*

- **Tree of Thought GitHub** (original implementation):
  https://github.com/princeton-nlp/tree-of-thought-llm

- **"Tree of Thoughts: A New Framework for LLM Reasoning"** (summary blog):
  https://www.promptingguide.ai/techniques/tot

### Graph of Thought and Beyond
- **Graph of Thought Paper** (extends ToT to non-linear graphs):
  https://arxiv.org/abs/2308.09687

- **Algorithm of Thought** (more efficient search-based reasoning):
  https://arxiv.org/abs/2308.10379

### o3 and Extended Thinking
- **Anthropic — Extended Thinking Docs** (Claude's built-in reasoning):
  https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking

- **OpenAI o3 System Card**:
  https://openai.com/index/openai-o3-system-card/

- **"How o3 Works"** (accessible explanation of RL + search):
  https://simonwillison.net/2024/Dec/20/o3/

### MCTS (Monte Carlo Tree Search)
- **"MCTS for LLM Reasoning"** (survey and intuition):
  https://arxiv.org/abs/2406.07394

- **MCTS Explained Simply** (Wikipedia — surprisingly clear):
  https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

### Prompt Chaining and Decomposition
- **"Chain of Thought Prompting"** (original paper, Wei et al. 2022):
  https://arxiv.org/abs/2201.11903

- **"Self-Consistency Improves CoT Reasoning"** (sample multiple CoT paths, vote):
  https://arxiv.org/abs/2203.11171

- **Reflexion** (agent self-reflection and correction without retraining):
  https://arxiv.org/abs/2303.11366

---

## Distributed Systems Background (If You Want It)

These are optional but give strong foundations for resilience patterns:

- **"Designing Data-Intensive Applications"** (Martin Kleppmann) — the best book on distributed systems:
  https://dataintensive.net
  *Chapter 7 (Transactions) and Chapter 8 (Trouble with Distributed Systems) are most relevant.*

- **"The Twelve-Factor App"** — production deployment principles:
  https://12factor.net

- **CAP Theorem Explained** (consistency/availability/partition tolerance trade-offs):
  https://www.ibm.com/topics/cap-theorem

---

## Videos and Courses

- **DeepLearning.AI — "Knowledge Graphs for RAG"** (free short course):
  https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/

- **Neo4j GraphAcademy — "Neo4j Fundamentals"** (free, interactive, ~2 hours):
  https://graphacademy.neo4j.com/courses/neo4j-fundamentals/

- **"Tree of Thoughts Explained"** (Yannic Kilcher):
  https://youtu.be/ut5kp56wW_4

- **"How A2A Protocol Works"** (Google Cloud Next talk):
  https://cloud.google.com/blog/products/ai-machine-learning/a2a-protocol-announcement

---

## Tools Checklist

| Tool | Purpose | Install |
|---|---|---|
| `neo4j` | Graph database Python driver | `pip install neo4j` |
| `spacy` | NER entity extraction | `pip install spacy` |
| `tenacity` | Retry with backoff | `pip install tenacity` |
| `redis` | Rate limiting (token bucket) | `pip install redis` |
| `pyjwt` | JWT auth for A2A | `pip install pyjwt` |
| `langchain-neo4j` | LangChain + Neo4j integration | `pip install langchain-neo4j` |
| `llama-index-graph-stores-neo4j` | LlamaIndex Neo4j support | `pip install llama-index-graph-stores-neo4j` |
| `temporal-sdk` | Workflow/Saga engine | `pip install temporalio` |
