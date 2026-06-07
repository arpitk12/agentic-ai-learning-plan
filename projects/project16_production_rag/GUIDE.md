# Build Guide — Production RAG Agent
### Project 16 · Step-by-step from zero to deployed

> **How to use this guide**  
> Work through each milestone in order. Each step tells you *what* to build and *why*, then points you at the corresponding stub file in `starter/`.  
> Check your work against `solution/` when you're stuck — but try first!

---

## Table of Contents

| # | Milestone | Est. time |
|---|-----------|-----------|
| [0](#0-prerequisites) | Prerequisites & environment | 15 min |
| [1](#1-project-setup) | Project setup & folder structure | 10 min |
| [2](#2-configuration--models) | Configuration & Pydantic models | 20 min |
| [3](#3-vector-store) | Vector store (ChromaDB wrapper) | 30 min |
| [4](#4-ingestion-pipeline) | Ingestion pipeline (load → chunk → embed → store) | 60 min |
| [5](#5-hybrid-retrieval) | Hybrid retrieval (BM25 + vector + RRF fusion) | 60 min |
| [6](#6-llm-reranker) | LLM reranker | 20 min |
| [7](#7-agents) | Agents (direct, RAG, orchestrator) | 45 min |
| [8](#8-mcp-server) | MCP server (FastMCP) | 20 min |
| [9](#9-fastapi-layer) | FastAPI layer (lifespan, middleware, routes) | 60 min |
| [10](#10-evaluation) | Evaluation suite (LLM-judge + CI gate) | 30 min |
| [11](#11-cli-scripts) | CLI scripts (ingest + evaluate) | 20 min |
| [12](#12-run--verify) | Run & verify end-to-end | 20 min |
| [13](#13-deployment) | Deployment options (local → Docker → cloud) | 30 min |
| [14](#14-ci-cd) | CI/CD with GitHub Actions | 20 min |

---

## 0. Prerequisites

### Software
```bash
python --version      # 3.11 or 3.12
git --version
docker --version      # optional, for containerised mode
```

### API key
This project uses [LiteLLM](https://docs.litellm.ai/) which is a unified wrapper.
Set whichever key you have:
```bash
export OPENAI_API_KEY="sk-..."       # OpenAI
# or
export ANTHROPIC_API_KEY="sk-ant-..." # Claude
# or any LiteLLM-supported provider
```

### Knowledge check
Before starting, make sure you understand:
- [ ] What a vector embedding is and why cosine similarity is useful
- [ ] What BM25 is (TF-IDF variant, good for keyword search)
- [ ] Why RAG (Retrieval-Augmented Generation) is better than relying on LLM memory
- [ ] What a FastAPI lifespan handler is

---

## 1. Project Setup

### 1.1 Clone and enter

```bash
cd projects/project16_production_rag/starter
```

### 1.2 Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
```

### 1.3 Install dependencies

```bash
pip install -r requirements.txt
```

Key packages and why we need them:

| Package | Purpose |
|---------|---------|
| `litellm` | Unified LLM client (OpenAI, Anthropic, etc.) |
| `fastapi` + `uvicorn` | Web framework + ASGI server |
| `chromadb` | Vector database (stores embeddings) |
| `sentence-transformers` | Local embedding model |
| `rank-bm25` | Keyword search index |
| `mcp` | Model Context Protocol server |
| `pytest` | Testing |

### 1.4 Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set MODEL and OPENAI_API_KEY
```

Important env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `gpt-4o-mini` | LiteLLM model string |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHROMA_PATH` | `data/chroma_db` | Local ChromaDB storage |
| `CHROMA_HOST` | *(empty)* | Set to use HTTP ChromaDB (Docker mode) |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `TOP_K` | `8` | Chunks to retrieve |
| `RERANK_TOP_N` | `3` | Chunks to keep after reranking |
| `EVAL_PASS_THRESHOLD` | `0.75` | Minimum score for CI gate |
| `RATE_LIMIT` | `60` | Max requests/minute per IP |

### 1.5 Verify folder structure

After setup you should have (in `starter/`):
```
src/
  config.py        ← given, complete
  models.py        ← given, complete
  store/
    chroma_store.py  ← TODO
  ingestion/
    loader.py        ← TODO
    chunker.py       ← TODO
    embedder.py      ← TODO
    pipeline.py      ← TODO
  retrieval/
    retriever.py     ← TODO
    reranker.py      ← TODO
  agents/
    direct_agent.py  ← TODO
    rag_agent.py     ← TODO
    orchestrator.py  ← TODO
  api/
    app.py           ← TODO
    middleware.py    ← TODO
    routes.py        ← TODO
  evaluation/
    evaluator.py     ← TODO
  mcp_server/
    server.py        ← TODO
scripts/
  ingest.py          ← TODO
  evaluate.py        ← TODO
data/sample_docs/    ← given (2 sample docs)
tests/               ← given (run these to verify your work)
```

---

## 2. Configuration & Models

### 2.1 Read `src/config.py`

This file is **given to you** — read it carefully. It defines a `Config` dataclass
that reads every setting from environment variables (or uses defaults).

The key singleton:
```python
cfg = Config()   # import this everywhere: from src.config import cfg
```

**Why a singleton?** Every module (embedder, retriever, agents, API) needs the same
settings. One import is cleaner than passing config objects everywhere.

### 2.2 Read `src/models.py`

Also **given**. This file defines all Pydantic models used across the system:

```
RawDocument      — raw file content before chunking
Chunk            — a text fragment after chunking
IngestionResult  — summary returned by the ingestion pipeline
RetrievedChunk   — a chunk with its relevance score
RetrievalResult  — wrapper around list[RetrievedChunk]
QueryRequest     — incoming API request body
QueryResponse    — API response with answer + citations
Citation         — a single source cited in the answer
EvalCase         — a golden QA pair
EvalCaseResult   — scores for one eval case
EvalReport       — aggregated evaluation report
```

**Why Pydantic?** Automatic validation, serialisation, and IDE autocomplete everywhere.

---

## 3. Vector Store

**File:** `src/store/chroma_store.py`  
**Tests:** `pytest tests/test_ingestion.py::TestPipeline`

### 3.1 Understand the design

```
VectorStore wraps ChromaDB. Nothing else in the codebase touches ChromaDB directly.
This is the "repository" pattern — swap ChromaDB for Qdrant/Pinecone by only
changing this one file.
```

### 3.2 Client modes

```python
# Local mode (default, no Docker needed)
chromadb.PersistentClient(path="data/chroma_db")

# HTTP mode (Docker, set CHROMA_HOST=chromadb)
chromadb.HttpClient(host="chromadb", port=8001)
```

The `_make_client` method checks `cfg.CHROMA_HOST` to decide which to use.

### 3.3 Collection metadata

When creating the collection, pass `metadata={"hnsw:space": "cosine"}`.  
This tells ChromaDB to use cosine distance (better than Euclidean for text embeddings).

### 3.4 Implement and test

After implementing all 10 TODOs in `chroma_store.py`, run:
```bash
python -c "from src.store.chroma_store import VectorStore; s = VectorStore(); print(s.count())"
```
Expected output: `0` (empty store on first run)

---

## 4. Ingestion Pipeline

This is the core of the offline pipeline. Documents flow through 4 stages:

```
disk files
    │
    ▼  src/ingestion/loader.py
RawDocument(doc_id, title, source, content)
    │
    ▼  src/ingestion/chunker.py
[Chunk(chunk_id, text, index), ...]
    │
    ▼  src/ingestion/embedder.py
[[0.1, -0.3, ...], ...]   ← 384-dim vectors
    │
    ▼  src/store/chroma_store.py
ChromaDB persisted to disk ✓
```

### 4.1 Loader (`src/ingestion/loader.py`)

**Key concept:** The `doc_id` must be stable so duplicate detection works.
```python
import hashlib
doc_id = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:16]
```

After implementing, test:
```bash
python -c "
from src.ingestion.loader import load_file, load_directory
docs = load_directory('data/sample_docs')
print(f'{len(docs)} docs loaded')
print(docs[0].title, '—', len(docs[0].content), 'chars')
"
```

### 4.2 Chunker (`src/ingestion/chunker.py`)

**Key concept:** Recursive splitting.

The algorithm tries separators in order — if text splits into pieces that are still
too large, it recurses with the next separator:

```
Separators: ["\n\n", "\n", ". ", " ", ""]

"A\n\nB\n\nC" → split on "\n\n" → ["A", "B", "C"]
"ABCDEF..." (very long, no newlines) → split on ". " → sentences
```

Overlap means the last `overlap` characters of chunk N are prepended to chunk N+1.
This prevents splitting a sentence's context:

```
chunk 0: "The rate limit is 1000 requests"
chunk 1: "1000 requests per hour for Professional accounts"
           ^^^^^^^^^^^^^ (overlapping part)
```

After implementing:
```bash
python -c "
from src.ingestion.loader import load_file
from src.ingestion.chunker import chunk_document
docs = load_file('data/sample_docs/product_overview.md')
chunks = chunk_document(docs[0])
print(f'{len(chunks)} chunks')
print('First chunk:', chunks[0].text[:100])
"
```

### 4.3 Embedder (`src/ingestion/embedder.py`)

**Key concept:** Singleton pattern.

The `SentenceTransformer` model takes ~2 seconds to load. We load it once and reuse:

```python
_model: SentenceTransformer | None = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(cfg.EMBED_MODEL)
    return _model
```

**Why `normalize_embeddings=True`?**  
With normalised vectors, cosine similarity = dot product. ChromaDB's HNSW index
with cosine distance works correctly with this.

After implementing:
```bash
python -c "
from src.ingestion.embedder import embed_query, embed_texts
v = embed_query('hello')
print(f'dim={len(v)}, norm≈{sum(x**2 for x in v)**0.5:.4f}')
"
```
Expected: `dim=384, norm≈1.0000`

### 4.4 Pipeline (`src/ingestion/pipeline.py`)

Two functions:
- `ingest_directory` — called once offline by `scripts/ingest.py`
- `ingest_text` — called at runtime by the `POST /ingest` API route

**Duplicate detection** (in `ingest_directory`):
```python
existing_ids = {d["metadata"]["doc_id"] for d in store.get_all_documents()}
if doc.doc_id in existing_ids and not replace_existing:
    skipped += 1
    continue
```

**Batch embedding** (important for performance):
```python
texts = [c.text for c in chunks]
embeddings = embed_texts(texts)   # ONE call for all chunks in a doc
```

After implementing, run the full offline ingestion:
```bash
python scripts/ingest.py --source data/sample_docs/ --verbose
```
Expected output:
```
Documents processed : 2
Chunks created      : ~15-25 (depends on chunk_size)
Total in store      : ~15-25
```

---

## 5. Hybrid Retrieval

**File:** `src/retrieval/retriever.py`

### 5.1 Why hybrid?

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| Vector | Semantic — "how many calls per hour" finds "rate limit 1000/hour" | Misses exact strings |
| BM25 | Exact keywords — "rate_limit" finds "rate_limit" | No understanding of meaning |
| **Hybrid** | Both | Slightly more complex |

### 5.2 BM25 index

BM25 (Best Match 25) is a probabilistic keyword retrieval algorithm. We use `rank-bm25`:
```python
from rank_bm25 import BM25Okapi
corpus = [doc["text"].lower().split() for doc in chunks]
bm25 = BM25Okapi(corpus)
scores = bm25.get_scores(query.lower().split())
```

**Important:** `rebuild_bm25()` must be called:
1. At startup (to load existing chunks from ChromaDB)
2. After any new document is ingested (to include new chunks)

### 5.3 Reciprocal Rank Fusion (RRF)

RRF merges two ranked lists without needing to normalise scores:
```
score(document) = 1/(k + rank_in_vector_results) + 1/(k + rank_in_bm25_results)
                   k = 60  (dampens the advantage of rank-1)
```

Example:
```
Document A: vector_rank=1, bm25_rank=3  → 1/61 + 1/63 = 0.0321
Document B: vector_rank=5, bm25_rank=1  → 1/65 + 1/61 = 0.0317
Document A wins slightly — it was top of both lists consistently
```

### 5.4 Implement and test

```bash
python -c "
from src.store.chroma_store import VectorStore
from src.retrieval.retriever import HybridRetriever
store = VectorStore()
r = HybridRetriever(store)
result = r.retrieve('what are the pricing tiers', top_k=3)
for c in result.chunks:
    print(f'[{c.score:.3f}] {c.text[:80]}')
"
```

---

## 6. LLM Reranker

**File:** `src/retrieval/reranker.py`

Retrieval returns chunks by approximate similarity. The LLM can judge relevance more
accurately by reading both the question and the chunk:

```
Query: "What is the rate limit for Professional plan?"

Chunk A (score 0.9): "rate limits vary by plan"        → LLM score: 6/10
Chunk B (score 0.85): "Professional: 1000 req/hour"    → LLM score: 10/10
Chunk C (score 0.8):  "see our pricing page for plans" → LLM score: 2/10

After reranking: B → A → C   (very different from retrieval order!)
```

**Performance note:** Reranking is an LLM call per chunk, so we only rerank top-K
(default 8) and keep top-N (default 3). The 5 discarded chunks never see LLM time.

---

## 7. Agents

### 7.1 Direct agent (`src/agents/direct_agent.py`)

Simple — just an LLM call with a helpful system prompt. No retrieval.
Used for general questions that don't need the knowledge base.

### 7.2 RAG agent (`src/agents/rag_agent.py`)

The grounded system prompt is critical:
```
"Answer using ONLY the context below. Cite [1], [2], etc. after each claim.
 If the context doesn't contain the answer, say 'I don't have that information.'

Context:
[1] (source: product_overview.md)
TechFlow offers three tiers: Starter, Professional, Enterprise...

[2] (source: api_reference.md)
Rate limit for Professional plan is 1000 requests per hour..."
```

**Why "ONLY the context"?** This prevents hallucination. The LLM must cite its sources
or say it doesn't know. This is the core value proposition of RAG.

### 7.3 Orchestrator (`src/agents/orchestrator.py`)

The classifier uses a cheap LLM call (max_tokens=5):
```
Prompt: "Does this question require product/API documentation?
         Reply 'rag' or 'direct'. Question: {question}"
Response: "rag"  (or "direct")
```

**Why not always use RAG?** For "what is 2+2?" retrieval adds latency and cost for
no benefit. The classifier prevents unnecessary work.

After implementing all three, test:
```bash
python -c "
import asyncio
from src.store.chroma_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.models import QueryRequest
from src.agents.orchestrator import handle

store = VectorStore()
retriever = HybridRetriever(store)
req = QueryRequest(question='What are the pricing tiers?')
resp = asyncio.run(handle(req, retriever, 'test'))
print(resp.answer[:200])
"
```

---

## 8. MCP Server

**File:** `src/mcp_server/server.py`

MCP (Model Context Protocol) lets LLM clients (Claude Desktop, custom agents) call
your RAG system as a tool — just like calling a function.

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("production-rag")

@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> str:
    """Search the knowledge base."""
    ...
```

The `@mcp.tool()` decorator handles all the MCP protocol plumbing.

Test by running the server standalone:
```bash
python -m src.mcp_server.server
```
It will block waiting for MCP client connections via stdio.

---

## 9. FastAPI Layer

### 9.1 Middleware (`src/api/middleware.py`)

Three responsibilities:
```
1. UUID request ID ── add to request.state + response header X-Request-ID
2. Latency header  ── measure time, add X-Latency-Ms to response
3. Rate limiter    ── sliding window (deque of timestamps per IP)
```

The sliding window rate limiter:
```python
window = self._windows[client_ip]   # deque of timestamps
now = time.time()
while window and now - window[0] > 60:   # drop old entries
    window.popleft()
if len(window) >= self._rate_limit:
    return 429 response
window.append(now)
```

### 9.2 App factory (`src/api/app.py`) — the most important file

**The lifespan pattern** is how FastAPI handles startup/shutdown:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ────────────────
    store     = VectorStore()
    retriever = HybridRetriever(store)   # builds BM25 from existing chunks
    app.state.store     = store
    app.state.retriever = retriever
    # ── APP IS LIVE ────────────
    yield
    # ── SHUTDOWN ───────────────
    # (ChromaDB handles its own cleanup)
```

**Why store on `app.state`?**  
Routes receive `request: Request`. They access shared objects via `request.app.state`.
This avoids global variables while sharing expensive objects (model, index) across requests.

### 9.3 Routes (`src/api/routes.py`)

```
GET  /health  → fast liveness probe, never blocks
GET  /stats   → vector store metadata
POST /query   → orchestrator.handle() → QueryResponse
POST /ingest  → ingest_text() + rebuild_bm25() → IngestionResult
GET  /eval    → run_eval() → EvalReport
```

After implementing, start the server:
```bash
uvicorn src.api.app:app --reload --port 8000
```

Test with curl:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the pricing tiers?"}'
```

Or open the interactive docs: http://localhost:8000/docs

---

## 10. Evaluation

**File:** `src/evaluation/evaluator.py`

### 10.1 LLM-judge pattern

Instead of string matching (brittle), we use an LLM to judge quality:

```
Faithfulness judge:
  "Context: {retrieved_chunks}
   Answer: {generated_answer}
   Score 0.0-1.0 how well the answer is grounded in context.
   1.0=fully grounded, 0.0=entirely fabricated. Reply ONLY a decimal."

Relevancy judge:
  "Question: {question}
   Answer: {generated_answer}
   Score 0.0-1.0 how well the answer addresses the question.
   Reply ONLY a decimal."
```

### 10.2 Golden dataset

The 5 QA pairs in `GOLDEN_CASES` are based on `data/sample_docs/`. They test:
1. Pricing tier names (product_overview.md)
2. API rate limits (api_reference.md)
3. Authentication method (api_reference.md)
4. Enterprise analytics features (product_overview.md)
5. REST endpoint for projects (api_reference.md)

**Why golden cases?** They let you measure quality regressions:
- Did changing the chunk_size break anything?
- Did switching models affect answer quality?

### 10.3 CI gate

The `scripts/evaluate.py` script exits with code 1 if score < threshold.
GitHub Actions treats exit code 1 as a build failure — this is how you enforce
quality in CI.

---

## 11. CLI Scripts

### 11.1 `scripts/ingest.py`

```bash
# Normal ingestion (skips duplicates)
python scripts/ingest.py --source data/sample_docs/

# Force re-embed all documents
python scripts/ingest.py --source data/sample_docs/ --replace

# Verbose output
python scripts/ingest.py --source data/sample_docs/ -v
```

### 11.2 `scripts/evaluate.py`

```bash
# Standard eval (uses cfg.EVAL_PASS_THRESHOLD = 0.75)
python scripts/evaluate.py

# Stricter gate
python scripts/evaluate.py --fail-below 0.85

# Save report without failing
python scripts/evaluate.py --output results/report.json --no-gate
```

---

## 12. Run & Verify

### Full end-to-end flow

```bash
# Step 1: Ingest sample documents
python scripts/ingest.py --source data/sample_docs/

# Step 2: Start the API server
uvicorn src.api.app:app --reload

# Step 3: Query via curl
curl -s -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the rate limit for the Professional API plan?"}' | python -m json.tool

# Step 4: Run evaluation
python scripts/evaluate.py --no-gate -v

# Step 5: Run tests
pytest tests/ -v
```

### Expected query response structure

```json
{
  "answer": "The Professional plan allows 1000 requests per hour [1].",
  "model": "gpt-4o-mini",
  "citations": [
    {
      "chunk_id": "abc123-0002",
      "text": "Professional: 1000 req/hour...",
      "source": "data/sample_docs/api_reference.md",
      "score": 0.92
    }
  ],
  "retrieval_mode": "hybrid",
  "request_id": "a1b2c3d4"
}
```

### Verification checklist

- [ ] `python scripts/ingest.py --source data/sample_docs/` runs without errors
- [ ] `GET /health` returns `{"status": "ok", "chunks_indexed": >0}`
- [ ] `POST /query` returns answer with citations
- [ ] `X-Request-ID` header present on every response
- [ ] Rate limit: 61st request in 60s returns 429
- [ ] `python scripts/evaluate.py` passes (score ≥ 0.75)
- [ ] `pytest tests/ -v` all green

---

## 13. Deployment

### Option A — Local (development)

```bash
# Already covered above — just uvicorn
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Limitations:** Single process, no persistence across restarts (ChromaDB files persist,
but BM25 index is rebuilt at startup — that's fine).

---

### Option B — Docker Compose (recommended for staging)

The `docker-compose.yml` in `solution/` defines two services:

```yaml
services:
  chromadb:       # ChromaDB runs as a separate HTTP service
    image: chromadb/chroma
    ports: ["8001:8000"]

  api:            # Your FastAPI app
    build: .
    ports: ["8000:8000"]
    environment:
      CHROMA_HOST: chromadb   # ← uses HTTP client mode
    depends_on:
      chromadb: { condition: service_healthy }
```

Copy the docker-compose from solution/ to your starter directory, then:

```bash
# Build and start both services
docker-compose up --build

# Ingest documents (runs inside the api container)
docker-compose exec api python scripts/ingest.py --source data/sample_docs/

# Query
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the pricing tiers?"}'

# Stop
docker-compose down
```

**Why separate ChromaDB service?** The API can be scaled horizontally (multiple
replicas) while ChromaDB remains a single source of truth.

---

### Option C — Railway (easiest cloud deploy, free tier available)

1. Push your code to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo
4. Add environment variables (MODEL, OPENAI_API_KEY, etc.)
5. Railway auto-detects the `Dockerfile` and builds

**For ChromaDB on Railway:**
- Option 1: Add a ChromaDB service from Railway's template marketplace
- Option 2: Use `CHROMA_PATH=/data/chroma_db` with a Railway persistent volume

Railway gives you a public HTTPS URL automatically.

---

### Option D — Render (free tier, good for demos)

1. Push to GitHub
2. [render.com](https://render.com) → New → Web Service → GitHub repo
3. Runtime: Docker (auto-detects Dockerfile)
4. Environment variables: add all from `.env.example`
5. For ChromaDB: add a Render Disk mounted at `/app/data/chroma_db`

**Free tier limitation:** Render spins down inactive services after 15 minutes.
The `GET /health` endpoint is perfect for an uptime monitor (e.g. UptimeRobot free tier)
to prevent spin-down.

---

### Option E — AWS ECS Fargate (production scale)

Architecture:
```
Internet
    │
    ▼
Application Load Balancer (ALB)
    │ HTTPS :443
    ▼
ECS Fargate Service (2+ tasks running your Docker image)
    │
    ├─► ChromaDB (ECS task on private subnet, or self-hosted on EC2)
    └─► S3 (for document storage, if needed)
```

Steps (high level):
```bash
# 1. Push image to ECR
aws ecr create-repository --repository-name production-rag
docker build -t production-rag .
docker tag production-rag:latest <account>.dkr.ecr.<region>.amazonaws.com/production-rag:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/production-rag:latest

# 2. Create ECS cluster
aws ecs create-cluster --cluster-name rag-cluster

# 3. Create task definition (JSON) referencing the ECR image
# 4. Create ALB + target group
# 5. Create ECS service pointing at ALB
```

Use AWS Secrets Manager for `OPENAI_API_KEY` — never put secrets in task definition JSON.

**For ChromaDB at scale:** Consider replacing ChromaDB with a managed vector database:
- [Pinecone](https://pinecone.io) (serverless, easy)
- [Qdrant Cloud](https://qdrant.tech) (open-source, self-host option)
- [Weaviate Cloud](https://weaviate.io)

Just update `src/store/chroma_store.py` — no other files need to change.

---

### Option F — Kubernetes (enterprise)

Example `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: production-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: production-rag
  template:
    spec:
      containers:
      - name: api
        image: your-registry/production-rag:latest
        ports: [{containerPort: 8000}]
        env:
        - name: CHROMA_HOST
          value: chromadb-service
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        livenessProbe:
          httpGet: {path: /health, port: 8000}
          initialDelaySeconds: 30
        readinessProbe:
          httpGet: {path: /health, port: 8000}
          initialDelaySeconds: 10
```

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml    # ClusterIP + Ingress
```

**Scaling considerations:**
- The BM25 index is in-memory per pod. All pods rebuild it at startup from ChromaDB.
- If ingest rate is high, add a Redis queue between the ingest endpoint and the
  embedding pipeline (convert `/ingest` to async background task with Celery).

---

## 14. CI/CD

Copy `.github/workflows/ci.yml` from `solution/` to your project root's `.github/` folder.

The pipeline runs 4 jobs on every push/PR:

```
lint ─────► test ─────► eval-gate ─────► docker-build
                         (fails if score < 0.75)
```

### Job breakdown

**lint:** `ruff check src/ scripts/ tests/` — catches style issues fast

**test:** 
```yaml
- run: python scripts/ingest.py --source data/sample_docs/
- run: pytest tests/ -v
```

**eval-gate:**
```yaml
- run: python scripts/evaluate.py --fail-below 0.75
```
If this exits with code 1, the docker-build job never runs. New code that breaks
RAG quality is blocked from being deployed.

**docker-build:**
```yaml
- run: docker build -t production-rag .
```

### Adding the workflow

```bash
mkdir -p .github/workflows
cp solution/.github/workflows/ci.yml .github/workflows/ci.yml
git add .github/
git commit -m "ci: add RAG quality gate pipeline"
git push
```

Go to your GitHub repo → Actions tab to see the pipeline run.

### Setting secrets

In your GitHub repo: Settings → Secrets and Variables → Actions → New repository secret:
- `OPENAI_API_KEY` = your key

In the CI YAML, secrets are accessed as:
```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Final Checklist

Before calling this production-ready:

### Functionality
- [ ] Ingest pipeline: load → chunk → embed → store works
- [ ] Hybrid retrieval returns relevant chunks
- [ ] Reranker improves ordering (check with verbose logs)
- [ ] RAG agent answers are grounded in context (no hallucination)
- [ ] Orchestrator correctly routes rag vs direct questions
- [ ] All API endpoints return correct responses
- [ ] MCP server runs and responds to tool calls

### Quality
- [ ] `pytest tests/ -v` — all tests pass
- [ ] `python scripts/evaluate.py` — score ≥ 0.75
- [ ] Rate limiter rejects the 61st request in 60s
- [ ] Every response has `X-Request-ID` header

### Deployment
- [ ] `docker-compose up --build` starts both services cleanly
- [ ] ChromaDB persists data between restarts
- [ ] Environment variables documented in `.env.example`
- [ ] `GET /health` returns 200 (for load balancer probes)

### Production extras (stretch goals)
- [ ] Add Prometheus metrics endpoint (`/metrics`) using `prometheus-client`
- [ ] Add structured JSON logging with `structlog`
- [ ] Add async background ingestion (Celery/ARQ) for large document batches
- [ ] Replace in-memory rate limiter with Redis-backed sliding window
- [ ] Add document chunking overlap visualisation (debugging tool)
- [ ] Add `/ingest/url` endpoint that fetches and ingests web pages

---

## Reference: Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         FastAPI Application           │
                    │                                       │
  HTTP Request ───► │  RequestMiddleware                   │
                    │   (UUID id · timing · rate limit)    │
                    │            │                          │
                    │            ▼                          │
                    │       Router                          │
                    │   /query  /ingest  /health  /eval    │
                    │            │                          │
                    │            ▼                          │
                    │      Orchestrator                     │
                    │   classify_intent()                   │
                    │      │           │                    │
                    │      ▼           ▼                    │
                    │  rag_agent   direct_agent             │
                    │      │                                │
                    │      ▼                                │
                    │  HybridRetriever                      │
                    │  BM25 ──┐                             │
                    │         ├──► RRF fusion               │
                    │  Vector ┘                             │
                    │      │                                │
                    │      ▼                                │
                    │  LLM Reranker                         │
                    │      │                                │
                    │      ▼                                │
                    │  LiteLLM (generate answer)            │
                    └─────────────────────────────────────┘
                              │               │
              ┌───────────────┘               └──────────────┐
              ▼                                              ▼
         ChromaDB                                    Offline Pipeline
      (vector storage)                    ingest.py → loader → chunker
                                          → embedder → ChromaDB
```

Good luck! Check `solution/` when stuck, and remember: the goal is to understand
*why* each component is built the way it is, not just to make the tests pass.
