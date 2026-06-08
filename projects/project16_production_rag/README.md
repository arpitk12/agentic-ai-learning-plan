# Project 16 — Production RAG Agent

A **production-grade, fully modular RAG system** that mirrors real engineering practice:
embedding runs in a separate async pipeline (Celery + Redis), retrieval happens at serve
time, and every layer is independently scalable. Includes a FastAPI service, Celery async
ingestion workers, Redis message queue, MCP server, multi-agent routing, hybrid search, an
evaluation pipeline, Docker deployment, and a CI/CD quality gate.

---

## 🏗 Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              OFFLINE INGESTION  (run once per doc update)                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   data/sample_docs/*.md                                                      ║
║           │                                                                  ║
║      loader.py        ← .md / .txt files → RawDocument                      ║
║           │                                                                  ║
║      chunker.py       ← recursive splitter (512 chars, 64 overlap)          ║
║           │                                                                  ║
║      embedder.py      ← sentence-transformers (local, no API cost, 384-dim) ║
║           │                                                                  ║
║      chroma_store.py  ← upsert to ChromaDB collection                       ║
║           │                                                                  ║
║   [ChromaDB persists to data/chroma_db/ or HTTP service]                    ║
║                                                                              ║
║   CLI:  python scripts/ingest.py --source data/sample_docs/ --verbose       ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║           ASYNC INGESTION PIPELINE  (background, non-blocking)              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   POST /ingest/async  ────────────────────────────────────────────────────  ║
║           │  returns {job_id} in <5ms                                        ║
║           ▼                                                                  ║
║      Redis Queue  ("ingestion")                                              ║
║           │                                                                  ║
║           ▼                                                                  ║
║   Celery Worker  (separate process / Docker container)                       ║
║       load → chunk → embed → store(ChromaDB)                                 ║
║       set  "bm25:stale" = "1"  in Redis  (TTL 2h)                           ║
║                                          │                                   ║
║                                          ▼                                   ║
║   POST /query  reads flag ───────────────┘                                   ║
║       → rebuilds BM25 index if stale  (transparent to caller)               ║
║       → deletes flag                                                         ║
║                                                                              ║
║   Track job:   GET /ingest/job/{job_id}   → {status, result, error}         ║
║   Monitor:     http://localhost:5555       (Flower UI)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                     SERVING PIPELINE  (every query)                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Client (curl / UI / MCP agent)                                              ║
║          │                                                                   ║
║          ▼  POST /query                                                      ║
║  RequestMiddleware                                                           ║
║    ├── attach X-Request-ID  (UUID)                                           ║
║    ├── measure latency  →  X-Latency-Ms header                               ║
║    └── sliding-window rate limit (60 req/min per IP)                         ║
║          │                                                                   ║
║     routes.py  →  _rebuild_bm25_if_stale()  →  orchestrator.handle()        ║
║          │                                                                   ║
║     orchestrator.py  (classify_intent — max_tokens=5 LLM call)              ║
║          │                                                                   ║
║     ┌────┴──────────────────────┐                                            ║
║     ▼                           ▼                                            ║
║  rag_agent.py             direct_agent.py                                    ║
║     │                           │                                            ║
║  retriever.py              LiteLLM (raw)                                     ║
║     │                                                                        ║
║  ┌──┴──────────────────────┐                                                 ║
║  BM25 keyword search       ChromaDB vector search                            ║
║  └──────────┬──────────────┘                                                 ║
║             ▼  RRF fusion  (1/(k+rank) per list)                             ║
║          top-K chunks                                                        ║
║             │                                                                ║
║        reranker.py  ← LLM scores each chunk 0-10, keeps top-N               ║
║             │                                                                ║
║        LiteLLM  ← grounded system prompt ("answer ONLY from context")       ║
║             │                                                                ║
║     QueryResponse(answer, citations, retrieval_mode, request_id)            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                     MCP SERVER  (optional — stdio subprocess)               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  src/mcp_server/server.py  (FastMCP)                                         ║
║    ├── search_docs(query, top_k)    ← calls retriever directly               ║
║    ├── ingest_text(text, title)     ← adds content at runtime                ║
║    └── get_stats()                  ← collection stats                       ║
║                                                                              ║
║  Run:  python -m src.mcp_server.server                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                     DOCKER COMPOSE — 5 SERVICES                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   chromadb :8001   — vector store (persistent volume)                        ║
║   redis    :6379   — message broker + result backend + BM25 staleness flag   ║
║   api      :8000   — FastAPI serving pipeline                                ║
║   worker           — Celery ingestion worker (--concurrency=2)               ║
║   flower   :5555   — Celery monitoring dashboard                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Folder Structure

```
project16_production_rag/
├── README.md
├── GUIDE.md                        ← step-by-step build guide + deployment options
├── starter/                        ← work here — stub files with TODOs
│   ├── requirements.txt
│   ├── .env.example
│   ├── docker-compose.yml          ← chromadb + redis + api + worker + flower (5 services)
│   ├── Dockerfile
│   ├── .github/
│   │   └── workflows/
│   │       └── ci.yml              ← lint → test → eval gate → docker build
│   ├── src/
│   │   ├── config.py               ← given complete
│   │   ├── models.py               ← given complete
│   │   ├── store/
│   │   │   └── chroma_store.py     ← TODO (10 tasks)
│   │   ├── ingestion/
│   │   │   ├── loader.py           ← TODO (10 tasks)
│   │   │   ├── chunker.py          ← TODO (8 tasks)
│   │   │   ├── embedder.py         ← TODO (4 tasks)
│   │   │   ├── pipeline.py         ← TODO (11 tasks)
│   │   │   └── tasks.py            ← TODO (13 tasks) ★ Celery async worker
│   │   ├── retrieval/
│   │   │   ├── retriever.py        ← TODO (15 tasks)
│   │   │   └── reranker.py         ← TODO (5 tasks)
│   │   ├── agents/
│   │   │   ├── orchestrator.py     ← TODO (7 tasks)
│   │   │   ├── rag_agent.py        ← TODO (8 tasks)
│   │   │   └── direct_agent.py     ← TODO (3 tasks)
│   │   ├── mcp_server/
│   │   │   └── server.py           ← TODO (5 tasks)
│   │   ├── api/
│   │   │   ├── app.py              ← TODO (9 tasks)
│   │   │   ├── routes.py           ← TODO (6 tasks) — sync routes + BM25 staleness check
│   │   │   ├── async_routes.py     ← TODO (16 tasks) ★ async ingestion endpoints
│   │   │   └── middleware.py       ← TODO (7 tasks)
│   │   └── evaluation/
│   │       └── evaluator.py        ← TODO (12 tasks)
│   ├── scripts/
│   │   ├── ingest.py               ← TODO (5 tasks) — offline CLI ingestion
│   │   └── evaluate.py             ← TODO (7 tasks) — CI quality gate
│   ├── tests/                      ← given complete (36 pytest cases)
│   └── data/sample_docs/           ← given (2 TechFlow docs)
│
└── solution/                       ← full working implementation — check when stuck
    ├── src/
    │   ├── ingestion/
    │   │   ├── pipeline.py
    │   │   └── tasks.py            ← Celery tasks: ingest_text_task, ingest_directory_task
    │   └── api/
    │       ├── app.py
    │       ├── routes.py           ← /query with _rebuild_bm25_if_stale()
    │       └── async_routes.py     ← /ingest/async, /ingest/job/{id}, /ingest/queue/stats
    └── docker-compose.yml          ← 5 services (chromadb, redis, api, worker, flower)
```

---

## ⚡ Production Patterns Demonstrated

| Pattern | Where |
|---------|-------|
| **Offline embed, online retrieve** | `scripts/ingest.py` runs before API; embeddings persist in ChromaDB |
| **Async ingestion queue** | `src/ingestion/tasks.py` — Celery workers, Redis broker, at-least-once delivery (`task_acks_late=True`) |
| **Non-blocking API** | `POST /ingest/async` returns `{job_id}` in <5ms; caller polls `GET /ingest/job/{id}` |
| **BM25 staleness flag** | Worker sets `bm25:stale` in Redis → `/query` rebuilds index on next request, then clears flag |
| **Graceful Redis degradation** | Staleness check is non-fatal — sync `/ingest` still works if Redis is down |
| **Local embeddings** (no API cost) | `src/ingestion/embedder.py` — sentence-transformers, runs offline |
| **Hybrid search** | `src/retrieval/retriever.py` — BM25 + ChromaDB vector + RRF fusion |
| **LLM reranker** | `src/retrieval/reranker.py` — top-K → top-N before generation |
| **Multi-agent routing** | `src/agents/orchestrator.py` — intent → rag \| direct |
| **MCP server** | `src/mcp_server/server.py` — RAG exposed as MCP tools |
| **Pydantic everywhere** | `src/models.py` — typed I/O for every interface |
| **Centralised config** | `src/config.py` — single `.env` → all modules via `cfg` |
| **Request middleware** | `src/api/middleware.py` — UUID request ID, latency header, rate limit |
| **LLM-judge eval gate** | `scripts/evaluate.py` → `sys.exit(1)` if quality < threshold |
| **Docker Compose (5 services)** | chromadb + redis + api + worker + flower |
| **GitHub Actions CI** | lint → test → eval gate → docker build |

---

## 🚀 Quick Start — Local (no Docker)

```bash
# 1. Enter the starter directory
cd projects/project16_production_rag/starter

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set MODEL and your LLM API key

# 5. Ingest documents (offline pipeline — run once)
python scripts/ingest.py --source data/sample_docs/ --verbose

# 6. Start Redis locally (needed for async routes)
#    macOS:  brew install redis && redis-server
#    Or skip — sync POST /ingest still works without Redis

# 7. Start Celery worker in a separate terminal (async ingestion)
celery -A src.ingestion.tasks worker --loglevel=info --queues=ingestion

# 8. Start the API
uvicorn src.api.app:app --reload --port 8000

# 9. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the API rate limits?"}'
```

## 🐳 Quick Start — Docker Compose (recommended)

```bash
cd projects/project16_production_rag/starter

# Build and start all 5 services
docker-compose up --build

# Ingest documents offline (run inside the worker container)
docker-compose exec worker python scripts/ingest.py --source data/sample_docs/

# Sync ingest — blocks until done, rebuilds BM25 inline
curl -X POST "http://localhost:8000/ingest?text=New+content&title=MyDoc"

# Async ingest — returns immediately, worker processes in background
curl -X POST http://localhost:8000/ingest/async \
  -H "Content-Type: application/json" \
  -d '{"text": "New product documentation...", "title": "v2 Changelog"}'
# → {"job_id": "abc-123", "status": "pending"}

# Poll job status until "success"
curl http://localhost:8000/ingest/job/abc-123
# → {"status": "success", "result": {"chunks_stored": 6, ...}, "ready": true}

# Celery monitoring dashboard
open http://localhost:5555

# Interactive API docs
open http://localhost:8000/docs
```

---

## 🔄 Sync vs Async Ingestion

| | `POST /ingest` (sync) | `POST /ingest/async` |
|---|---|---|
| **Returns when** | Pipeline complete | Job queued (<5ms) |
| **API thread blocked** | Yes (100ms–2s+) | No |
| **Track progress** | Not needed (inline) | `GET /ingest/job/{id}` |
| **Retries on failure** | No | Yes — 3× with exponential backoff |
| **BM25 rebuild** | Inline, immediate | Lazy on next `/query` |
| **Requires Redis** | No | Yes |
| **Best for** | Dev, small docs | Production, large batches |

---

## Milestones

### Milestone 1 — Ingestion Pipeline
Implement all files in `src/store/` and `src/ingestion/` (except `tasks.py`).
Run `python scripts/ingest.py --source data/sample_docs/ --verbose`.
Verify `GET /stats` shows `chunks > 0`.

### Milestone 2 — Hybrid Retrieval
Implement `src/retrieval/retriever.py` and `reranker.py`.
Compare `mode=vector` vs `mode=bm25` vs `mode=hybrid` on keyword vs semantic queries.

### Milestone 3 — Query API
Implement all agents and the synchronous API layer (app, middleware, routes).
`POST /query {"question": "..."}` returns a grounded answer with citations.

### Milestone 4 — Async Ingestion ★
Implement `src/ingestion/tasks.py` and `src/api/async_routes.py`.
Also implement `_rebuild_bm25_if_stale()` in `routes.py`.
Run `docker-compose up`, submit a document via `POST /ingest/async`, poll until done.
Verify the content appears in the next `/query` response (BM25 staleness check triggered).

### Milestone 5 — Evaluation Gate
Implement `src/evaluation/evaluator.py` and `scripts/evaluate.py`.
Run `python scripts/evaluate.py` — all 5 golden questions score ≥ 0.75.

### Milestone 6 — CI / Docker
Push `.github/workflows/ci.yml` (from `solution/`). Watch lint → test → eval → docker build
pass in GitHub Actions.

---

## Setup

```bash
pip install -r requirements.txt
```

> `sentence-transformers` downloads the embedding model (~90 MB) on first run.
> Cached in `~/.cache/huggingface/` after that.

> `celery[redis]` and `redis` are required for Milestone 4 (async ingestion).
> Milestones 1–3 work without Redis.
