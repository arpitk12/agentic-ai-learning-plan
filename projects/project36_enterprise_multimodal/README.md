# Project 36 — Enterprise Multimodal Compliance Agent

> **Enterprise-grade** AI agent that processes PDF documents, embedded images, and audio recordings for compliance analysis — powered by Graph RAG, 4-layer guardrails, Mem0 long-term memory, and production resilience patterns.

## What This Integrates

This capstone project is the synthesis of **every advanced technique from Phase 7**:

| Component | Technique | Week |
|---|---|---|
| **PDF Ingestion** | PyMuPDF text extraction + embedded image extraction | 14 |
| **Vision Analysis** | GPT-4o vision API (base64 image → structured JSON) | 14 |
| **Audio Transcription** | OpenAI Whisper (`audio.transcriptions.create`) | 14 |
| **Entity Extraction** | spaCy `en_core_web_sm` NER + LLM enhancement | 16 |
| **Graph RAG** | Neo4j MERGE + NL → Cypher + graph traversal | 16 |
| **Hybrid Retrieval** | ChromaDB vector + Neo4j graph + cross-encoder rerank | 16 |
| **Input Guardrails** | L1 regex injection · L2 PII scan · L3 LlamaGuard | 14 |
| **Output Guardrails** | PII redaction · faithfulness check on response | 14 |
| **Long-term Memory** | Mem0 episodic + semantic + procedural + profile | 13 |
| **Resilience** | Circuit breaker + ordered model fallback chain | 16 |
| **Cost Tracking** | Per-call token count + USD cost (input/output rates) | 13 |
| **FastAPI Serving** | Async endpoints + SSE streaming + rate limiting | 7 |
| **Structured Logging** | structlog with request IDs + latency + cost | 8 |

---

## Architecture

```
PDF / Images / Audio
        │
        ▼
┌────────────────────┐
│   Input Guardrails │  L1: regex injection detection (8 patterns)
│   (4-layer)        │  L2: PII scan + anonymize before LLM
└─────────┬──────────┘  L3: LlamaGuard / GPT-4o safety classifier
          │              L4: topic relevance gate
          ▼
┌────────────────────┐
│  Multimodal        │  PyMuPDF → text chunks (500 chars, 50 overlap)
│  Ingestion         │  GPT-4o vision → {description, type, data}
└─────────┬──────────┘  Whisper → timestamped transcript
          │
          ▼
┌──────────────────────────────────────┐
│          Dual Indexing               │
│  ┌────────────────┐  ┌────────────┐ │
│  │   ChromaDB     │  │   Neo4j    │ │
│  │   3 collections│  │   Knowledge│ │
│  │   text/img/au  │  │   Graph    │ │
│  └────────────────┘  └────────────┘ │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌────────────────────┐
│  Hybrid Retrieval  │  vector similarity (ChromaDB)
│                    │  + graph traversal (Cypher)
└─────────┬──────────┘  + cross-encoder rerank
          │
          ▼
┌────────────────────┐
│  Mem0 Memory       │  per-user episodic history
│  (long-term)       │  learned compliance preferences
└─────────┬──────────┘  procedural patterns + user profile
          │
          ▼
┌────────────────────┐
│  Resilient LLM     │  circuit breaker (trips at 3 failures)
│  (fallback chain)  │  GPT-4o → GPT-4o-mini → fine-tuned endpoint
└─────────┬──────────┘  retry with exponential backoff (3 attempts)
          │
          ▼
┌────────────────────┐
│  Output Guardrails │  PII redaction on response
│                    │  faithfulness score vs retrieved context
└─────────┬──────────┘  citation extraction + verification
          │
          ▼
      JSON Response + cost metadata
```

---

## Directory Structure

```
project36_enterprise_multimodal/
├── README.md                    This file
├── GUIDE.md                     Step-by-step implementation guide
├── starter/                     ← Solve this yourself (all TODOs)
│   ├── .env.example
│   ├── requirements.txt
│   ├── docker-compose.yml       Neo4j + Redis
│   ├── Dockerfile
│   ├── src/
│   │   ├── config.py            Environment + LLM configuration
│   │   ├── models.py            Pydantic request/response schemas
│   │   ├── ingestion/
│   │   │   ├── pdf_extractor.py     PDF → text chunks + image bytes
│   │   │   ├── vision_analyzer.py   Image → structured description
│   │   │   └── audio_transcriber.py Audio file → transcript
│   │   ├── graph/
│   │   │   ├── entity_extractor.py  Text → (entity, relation) tuples
│   │   │   └── neo4j_store.py       Load graph + generate Cypher
│   │   ├── retrieval/
│   │   │   ├── vector_store.py      ChromaDB 3-collection store
│   │   │   └── hybrid_retriever.py  Vector + graph hybrid search
│   │   ├── guardrails/
│   │   │   ├── injection_checker.py L1: prompt injection detection
│   │   │   ├── pii_scanner.py       L2: PII detect + anonymize
│   │   │   ├── safety_checker.py    L3: LlamaGuard / GPT-4o safety
│   │   │   └── pipeline.py          Orchestrate all 4 layers
│   │   ├── memory/
│   │   │   └── mem0_store.py        Mem0 long-term memory wrapper
│   │   ├── resilience/
│   │   │   ├── circuit_breaker.py   State machine + failure tracking
│   │   │   └── fallback_chain.py    Ordered model fallback
│   │   ├── agents/
│   │   │   └── multimodal_agent.py  Main orchestrator agent
│   │   ├── api/
│   │   │   ├── app.py               FastAPI application factory
│   │   │   ├── routes.py            /analyze /search /ingest /memories
│   │   │   └── middleware.py        Rate limiting + request logging
│   │   └── observability/
│   │       ├── logger.py            structlog setup
│   │       └── cost_tracker.py      Per-call token + USD tracking
│   ├── scripts/
│   │   ├── ingest.py            Bulk ingest a folder of documents
│   │   └── evaluate.py          Retrieval + guardrail + agent eval
│   └── tests/
│       ├── test_ingestion.py    Unit tests for ingestion pipeline
│       └── test_guardrails.py   Unit tests for guardrails pipeline
└── solution/                    ← Reference implementation (peek after trying)
    └── (mirrors starter/ structure, fully implemented)
```

---

## Prerequisites

```bash
# System dependencies
brew install ffmpeg            # Whisper audio decoding
python -m spacy download en_core_web_sm   # entity extraction

# Docker services
docker-compose up -d           # starts Neo4j (7474/7687) + Redis (6379)
```

## Setup

```bash
cp starter/.env.example starter/.env
# Edit .env: fill OPENAI_API_KEY, NEO4J_PASSWORD

pip install -r starter/requirements.txt
```

## Usage

```bash
# 1. Ingest a folder of PDFs (+ any audio files)
python starter/scripts/ingest.py --input ./sample_docs/

# 2. Start the API server
uvicorn starter.src.api.app:app --reload --port 8000

# 3. Analyze a compliance question
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "question": "What GDPR risks exist in our EU cloud contracts?",
    "include_graph": true
  }'
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Run full compliance analysis (guardrails → retrieval → LLM → output check) |
| `POST` | `/ingest/pdf` | Upload a PDF and ingest text + images |
| `POST` | `/ingest/audio` | Upload an audio file and ingest transcript |
| `GET` | `/search?q=...` | Hybrid vector + graph search (no LLM) |
| `GET` | `/graph/query?q=...` | NL → Cypher → graph results |
| `GET` | `/memories/{user_id}` | View user's long-term memories |
| `GET` | `/health` | Service health + circuit breaker states |
| `GET` | `/metrics` | Cost summary + latency percentiles |

## Evaluation

```bash
python starter/scripts/evaluate.py

# Report includes:
# Retrieval:  precision@5, recall@5, MRR
# Guardrails: injection block rate, PII detection rate, false positive rate
# Agent:      accuracy on 20-question golden dataset, p50/p95 latency, $/query
```

## Learning Objectives

After completing this project you will be able to:

1. **Build a multimodal ingestion pipeline** — extract text, images, and audio from documents and store them in separate vector collections
2. **Implement Graph RAG** — extract entities with spaCy, build a Neo4j knowledge graph, convert questions to Cypher, combine with vector search
3. **Deploy multi-layer guardrails** — detect injections, scan PII, call LlamaGuard, validate output faithfulness — all in a single async pipeline
4. **Add per-user long-term memory** — store episodic history, learned preferences, and compliance patterns with Mem0
5. **Engineer for production resilience** — circuit breakers, ordered fallback chains, retry with exponential backoff, dead-letter queues
6. **Observe costs and latency** — track tokens and USD per call, aggregate by user, serve Prometheus metrics
7. **Serve with FastAPI** — async endpoints, SSE streaming, rate limiting, structured logging with request IDs
