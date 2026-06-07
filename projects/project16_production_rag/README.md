# Project 16 — Production RAG Agent

A **production-grade, fully modular RAG system** that mirrors real engineering practice:
embedding happens once (in an offline ingestion pipeline) and retrieval happens separately
at serve time. Includes a FastAPI service, MCP server, multi-agent routing, hybrid search,
an evaluation pipeline, Docker deployment, and a CI/CD quality gate.

---

## 🏗 Architecture

```
                  INGESTION PIPELINE (offline — runs on doc update)
                  ─────────────────────────────────────────────────
                  data/sample_docs/*.md
                          │
                     loader.py       ← loads .md / .txt / .pdf
                          │
                     chunker.py      ← recursive splitter (512 tok, 64 overlap)
                          │
                     embedder.py     ← sentence-transformers (local, no API cost)
                          │
                     chroma_store.py ← persists to data/chroma_db/
                          │
                   [ChromaDB on disk or Docker service]


                  SERVING PIPELINE (online — every query)
                  ─────────────────────────────────────────────────
  Client (curl / WhatsApp / UI)
          │
          ▼ POST /query
  FastAPI  ←── middleware.py (request ID, timing, rate limit)
          │
     routes.py
          │
     orchestrator.py   ← classify_intent → rag | direct | mcp
          │
    ┌─────┴────────────────────────┐
    ▼                              ▼
  rag_agent.py              direct_agent.py
    │                              │
  retriever.py (hybrid)       LiteLLM (direct)
    │
  ┌─┴───────────────────┐
  BM25 keyword search   ChromaDB vector search
    └──── RRF fusion ───┘
          │
     reranker.py      ← LLM-based cross-attention reranker
          │
     LiteLLM (generation with retrieved context)
          │
    structured response (Pydantic)


                  MCP SERVER (optional — stdio subprocess)
                  ─────────────────────────────────────────────────
  src/mcp_server/server.py
    ├── search_docs(query)     ← calls retriever directly
    ├── ingest_text(text)      ← adds content at runtime
    └── get_stats()            ← collection stats
```

---

## 📁 Folder Structure

```
project16_production_rag/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml          ← API + ChromaDB services
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml              ← lint → test → eval gate → build
├── src/
│   ├── config.py               ← all settings from env vars (single source of truth)
│   ├── models.py               ← Pydantic data models for all I/O
│   ├── store/
│   │   └── chroma_store.py     ← ChromaDB abstraction (local or HTTP)
│   ├── ingestion/
│   │   ├── loader.py           ← load .md/.txt files from directory
│   │   ├── chunker.py          ← recursive character splitter
│   │   ├── embedder.py         ← sentence-transformers wrapper (no API cost)
│   │   └── pipeline.py         ← orchestrates load→chunk→embed→store
│   ├── retrieval/
│   │   ├── retriever.py        ← hybrid BM25+vector search with RRF fusion
│   │   └── reranker.py         ← LLM-based reranker (top-k → top-n)
│   ├── agents/
│   │   ├── orchestrator.py     ← intent classifier → routes to correct agent
│   │   ├── rag_agent.py        ← retrieval-augmented generation
│   │   └── direct_agent.py     ← direct LLM (no retrieval, for simple queries)
│   ├── mcp_server/
│   │   └── server.py           ← FastMCP server exposing RAG as MCP tools
│   ├── api/
│   │   ├── app.py              ← FastAPI app with lifespan (loads retriever once)
│   │   ├── routes.py           ← /query /ingest /health /stats /eval
│   │   └── middleware.py       ← request ID, timing, structlog, rate limit
│   └── evaluation/
│       └── evaluator.py        ← faithfulness + relevancy + golden dataset eval
├── tests/
│   ├── test_ingestion.py       ← pytest: loader, chunker, embedder, pipeline
│   └── test_api.py             ← pytest: FastAPI routes with TestClient
├── scripts/
│   ├── ingest.py               ← CLI: python scripts/ingest.py --source data/sample_docs
│   └── evaluate.py             ← CLI: python scripts/evaluate.py (outputs JSON + exit code)
└── data/
    ├── sample_docs/            ← drop .md/.txt files here to index
    │   ├── product_overview.md
    │   └── api_reference.md
    └── chroma_db/              ← auto-created by ingestion pipeline (gitignored)
```

---

## ⚡ Production Patterns Demonstrated

| Pattern | Where |
|---------|-------|
| **Separated ingestion from serving** | `scripts/ingest.py` vs `src/api/` — embeddings written once, read many times |
| **Local embeddings** (no API cost) | `src/ingestion/embedder.py` — sentence-transformers, runs offline |
| **Hybrid search** (BM25 + vector) | `src/retrieval/retriever.py` — RRF fusion, best of both worlds |
| **LLM reranker** (top-k → top-n) | `src/retrieval/reranker.py` — cuts noise before generation |
| **Multi-agent routing** | `src/agents/orchestrator.py` — intent → rag \| direct \| mcp |
| **MCP server** (tool-based RAG) | `src/mcp_server/server.py` — exposes search as structured MCP tool |
| **Pydantic everywhere** | `src/models.py` — typed I/O for every interface |
| **Centralised config** | `src/config.py` — single `.env` → all modules via `cfg` |
| **Request middleware** | `src/api/middleware.py` — request ID, latency, structlog |
| **Eval CI gate** | `scripts/evaluate.py` → `sys.exit(1)` if quality falls below threshold |
| **Docker Compose** | `docker-compose.yml` — API + persistent ChromaDB service |
| **GitHub Actions CI** | `.github/workflows/ci.yml` — lint → test → eval → docker build |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env: set MODEL, optionally GEMINI_API_KEY / GROQ_API_KEY

# 3. Ingest documents (embedding pipeline — run once, or on doc update)
python scripts/ingest.py --source data/sample_docs --verbose

# 4. Run evaluation to check quality
python scripts/evaluate.py

# 5. Start the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# 6. Query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the API rate limits?"}'
```

## 🐳 Docker Quick Start

```bash
docker-compose up --build

# Ingest (separate container run)
docker-compose run --rm api python scripts/ingest.py --source data/sample_docs

# Then query as above
```

---

## Milestones

### Milestone 1 — Ingestion Pipeline
Run `python scripts/ingest.py --source data/sample_docs`.
Verify `data/chroma_db/` is created and `GET /stats` shows doc count > 0.

### Milestone 2 — Query API
Run `uvicorn src.api.app:app --reload`.
`POST /query {"question": "..."}` should return a grounded response with source citations.

### Milestone 3 — Hybrid Search
Compare `/query?mode=vector` vs `/query?mode=hybrid`.
Hybrid should retrieve different (often better) results for keyword-heavy queries.

### Milestone 4 — Evaluation Gate
Run `python scripts/evaluate.py`.
All 5 golden questions should pass faithfulness ≥ 0.75. Fix retrieval if they don't.

### Milestone 5 — Docker Deploy
`docker-compose up --build` → run ingestion → query the API.
Check `docker-compose logs api` for structured JSON logs.

### Milestone 6 — Extend
Add your own documents to `data/sample_docs/`, re-run ingestion, verify the new content appears in answers.
Add a new MCP tool to `src/mcp_server/server.py`.

---

## Setup

```bash
pip install -r requirements.txt
```

> ⚠️ `sentence-transformers` downloads the embedding model (~90MB) on first run.
> It is cached in `~/.cache/huggingface/` after that.
