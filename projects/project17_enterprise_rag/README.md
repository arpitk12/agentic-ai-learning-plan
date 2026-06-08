# Project 17 — Enterprise RAG: 10M Docs, Zero Hallucination

A **production-grade RAG system** designed for real-world scale constraints:
10 million source documents, strict zero-hallucination guarantee, sub-second
query latency, and full observability. Every design decision is motivated by
a concrete production requirement.

---

## 🎯 Design Constraints

| Constraint | Requirement | Solution |
|---|---|---|
| **Scale** | 10M source docs → ~50M chunks | Qdrant + HNSW + INT8 quantization |
| **Throughput** | Ingest thousands of docs/min | Kafka pipeline with parallel consumers |
| **Hallucination** | Zero tolerance | NLI faithfulness check + abstain policy |
| **Latency** | < 500ms P99 for queries | Redis semantic cache + pre-built index |
| **Reliability** | No data loss during ingest | Kafka at-least-once + manual offsets |
| **Cost** | No embedding API cost | Local sentence-transformers model |
| **Observability** | Full trace of every answer | OpenTelemetry + Prometheus + Grafana |

---

## 🏗 Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║              INGESTION PIPELINE — Kafka-based, fully distributed                    ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  Data Sources  (S3 / DB / API / filesystem)                                          ║
║       │                                                                              ║
║       ▼  scripts/ingest.py  OR  POST /ingest                                        ║
║  kafka_producer.py                                                                   ║
║       │  publishes to:  raw-documents  (10 partitions, RF=3 in prod)                ║
║       ▼                                                                              ║
║  ┌───────────────────────────────────────────────────────────────────────────────┐  ║
║  │                      KAFKA BROKER CLUSTER                                     │  ║
║  │                                                                               │  ║
║  │  raw-documents (10 partitions)                                                │  ║
║  │       │                                                                       │  ║
║  │       ▼  [chunker-group — 10 consumers]                                       │  ║
║  │  chunk_consumer.py: load → split(512 chars, 64 overlap) → publish            │  ║
║  │       │  document-chunks (20 partitions)                                      │  ║
║  │       │                                                                       │  ║
║  │       ▼  [embedder-group — 20 consumers, GPU-scalable]                        │  ║
║  │  embed_consumer.py: batch(32) → check embedding cache → embed → publish      │  ║
║  │       │  embedded-chunks (20 partitions)                                      │  ║
║  │       │                                                                       │  ║
║  │       ▼  [indexer-group — 10 consumers]                                       │  ║
║  │  index_consumer.py: batch(256) → Qdrant upsert                               │  ║
║  │                                                                               │  ║
║  │  dlq-ingestion (5 partitions) ← failed messages after 3 retries              │  ║
║  └───────────────────────────────────────────────────────────────────────────────┘  ║
║                                          │                                           ║
║                                          ▼                                           ║
║                                   Qdrant Cluster                                     ║
║                                  (50M vectors, sharded)                             ║
║                                                                                      ║
║  Throughput math:  50M chunks ÷ (20 consumers × 1000 chunks/s) = 2500 s ≈ 42 min   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════════════╗
║              SERVING PIPELINE — Zero-Hallucination RAG                             ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  Client  →  POST /query  →  RequestMiddleware (ID, timing, rate limit)              ║
║                                    │                                                 ║
║                         ┌──────────▼──────────┐                                     ║
║                         │  Redis Semantic Cache │                                    ║
║                         │  cosine_sim ≥ 0.97?  │── HIT ──→  return cached answer   ║
║                         └──────────┬────────────┘                                   ║
║                                    │ MISS                                            ║
║                                    ▼                                                 ║
║                             HYBRID RETRIEVAL                                         ║
║                    ┌───────────────┴───────────────┐                                 ║
║                    ▼                               ▼                                 ║
║            Qdrant vector search          BM25 keyword search                         ║
║            (HNSW, ef=128, top-20)        (rank_bm25, top-20)                        ║
║                    └───────────────┬───────────────┘                                 ║
║                                    ▼  RRF fusion → top-20                           ║
║                               Reranker                                               ║
║                          (cross-encoder/ms-marco)                                    ║
║                               top-20 → top-5                                         ║
║                                    │                                                 ║
║                                    ▼                                                 ║
║                           ┌─────────────────┐                                        ║
║                           │  Abstain Policy │                                        ║
║                           │  top score<0.65?│── YES ──→  "No relevant info found"   ║
║                           └────────┬────────┘                                        ║
║                                    │ NO                                              ║
║                                    ▼                                                 ║
║                          LiteLLM Generation                                          ║
║                     (grounded prompt: "answer ONLY from context")                   ║
║                                    │                                                 ║
║                                    ▼                                                 ║
║                        ┌───────────────────────┐                                     ║
║                        │  Faithfulness Checker  │                                    ║
║                        │  NLI: cross-encoder/   │                                    ║
║                        │  nli-deberta-v3-base   │                                    ║
║                        │                        │                                    ║
║                        │  per sentence:         │                                    ║
║                        │  ENTAILMENT ≥ 0.75?    │                                    ║
║                        │   YES → keep           │                                    ║
║                        │   NO  → remove/abstain │                                    ║
║                        └──────────┬─────────────┘                                   ║
║                                   │  faithfulness_score < 0.80?                      ║
║                                   │   YES → abstain (return reason)                  ║
║                                   │   NO  → continue                                 ║
║                                   ▼                                                 ║
║                         Citation Verifier                                            ║
║                    (map each sentence → source chunk)                               ║
║                                   │                                                 ║
║                         Store in Redis cache                                         ║
║                                   │                                                 ║
║          QueryResponse(answer, citations, faithfulness_score, abstained=False)      ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════════════╗
║              DOCKER COMPOSE — 10 SERVICES                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   zookeeper           ← Kafka coordination                                           ║
║   kafka        :9092  ← Message broker (1 node dev, 3 prod)                        ║
║   kafka-ui     :8090  ← Kafka monitoring UI (topic browser, consumer lag)          ║
║   kafka-init          ← Creates topics on startup                                  ║
║   qdrant       :6333  ← Vector store (gRPC :6334)                                  ║
║   redis        :6379  ← Query cache + embedding cache + semantic cache             ║
║   api          :8000  ← FastAPI serving (scale horizontally behind nginx)          ║
║   chunk-consumer      ← Kafka → chunked documents (×2 replicas)                   ║
║   embed-consumer      ← Kafka → embeddings (×4 replicas — GPU scale point)        ║
║   index-consumer      ← Kafka → Qdrant upsert (×2 replicas)                       ║
║   prometheus   :9090  ← Metrics scrape                                             ║
║   grafana      :3000  ← Dashboards (no auth in dev)                               ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Folder Structure

```
project17_enterprise_rag/
├── README.md
├── GUIDE.md                        ← step-by-step build guide (start here)
│
├── starter/                        ← work here — stubs with TODOs
│   ├── requirements.txt
│   ├── .env.example
│   ├── docker-compose.yml          ← all 10 services
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── prometheus.yml
│   └── src/
│       ├── config.py               ← given complete (all settings via .env)
│       ├── models.py               ← given complete (Pydantic data models)
│       ├── store/
│       │   └── qdrant_store.py     ← TODO (12 tasks) — HNSW + INT8 quantization
│       ├── cache/
│       │   ├── redis_cache.py      ← TODO (8 tasks) — exact + embedding cache
│       │   └── semantic_cache.py   ← TODO (6 tasks) — similarity-based cache
│       ├── ingestion/
│       │   ├── loader.py           ← TODO (4 tasks)
│       │   ├── chunker.py          ← TODO (5 tasks)
│       │   ├── embedder.py         ← TODO (4 tasks)
│       │   └── kafka_producer.py   ← TODO (7 tasks) — publish to raw-documents
│       ├── consumers/
│       │   ├── chunk_consumer.py   ← TODO (8 tasks) — raw-docs → chunks
│       │   ├── embed_consumer.py   ← TODO (9 tasks) — chunks → embeddings
│       │   └── index_consumer.py   ← TODO (7 tasks) — embeddings → Qdrant
│       ├── retrieval/
│       │   ├── retriever.py        ← TODO (10 tasks) — hybrid BM25+Qdrant+RRF
│       │   └── reranker.py         ← TODO (5 tasks) — cross-encoder reranker
│       ├── hallucination/          ★ THE CORE INNOVATION
│       │   ├── faithfulness_checker.py  ← TODO (10 tasks) — NLI sentence check
│       │   ├── abstain_policy.py        ← TODO (5 tasks) — threshold-based refuse
│       │   └── citation_verifier.py     ← TODO (4 tasks) — sentence → chunk map
│       ├── agents/
│       │   ├── orchestrator.py     ← TODO (4 tasks)
│       │   └── rag_agent.py        ← TODO (8 tasks) — full pipeline
│       ├── api/
│       │   ├── app.py              ← TODO (5 tasks)
│       │   ├── routes.py           ← TODO (6 tasks)
│       │   └── middleware.py       ← TODO (5 tasks)
│       ├── observability/
│       │   └── metrics.py          ← TODO (5 tasks) — Prometheus counters/histograms
│       ├── scripts/
│       │   ├── ingest.py           ← TODO (4 tasks) — CLI offline ingestion
│       │   └── evaluate.py         ← TODO (5 tasks) — hallucination eval harness
│       └── tests/                  ← given complete
│           ├── test_hallucination.py
│           ├── test_pipeline.py
│           └── test_cache.py
│
└── solution/                       ← full implementation — check when stuck
    └── src/
        ├── store/qdrant_store.py
        ├── cache/{redis_cache,semantic_cache}.py
        ├── ingestion/kafka_producer.py
        ├── consumers/{chunk,embed,index}_consumer.py
        ├── retrieval/{retriever,reranker}.py
        ├── hallucination/{faithfulness_checker,abstain_policy,citation_verifier}.py
        ├── agents/{orchestrator,rag_agent}.py
        └── api/{app,routes,middleware}.py
```

---

## ⚡ Production Patterns Demonstrated

| Pattern | Where | Why |
|---------|-------|-----|
| **Kafka multi-stage pipeline** | `consumers/` | Decouple load/chunk/embed/index; replay on failure |
| **Dead letter queue** | `dlq-ingestion` topic | Failed messages don't block the pipeline |
| **Qdrant HNSW + INT8 quantization** | `qdrant_store.py` | 50M vectors in 18 GB vs 72 GB without quantization |
| **Payload index filtering** | `qdrant_store.py` | Pre-filter by source before vector search → 10× faster |
| **Embedding cache (Redis)** | `redis_cache.py` | Skip re-embedding identical text in consumers |
| **Exact query cache (Redis)** | `redis_cache.py` | Repeated questions return in <1ms |
| **Semantic cache** | `semantic_cache.py` | Similar questions (cosine ≥ 0.97) share cached answers |
| **NLI faithfulness check** | `faithfulness_checker.py` | Per-sentence entailment via DeBERTa cross-encoder |
| **Abstain policy** | `abstain_policy.py` | Return "I don't know" instead of hallucinating |
| **Citation pinning** | `citation_verifier.py` | Every sentence mapped to exact source chunk |
| **RRF hybrid search** | `retriever.py` | BM25 recall + vector semantic = best of both |
| **Cross-encoder reranker** | `reranker.py` | top-20 → top-5 precision before generation |
| **Circuit breaker** | `middleware.py` | API stays up even if Qdrant/LLM is slow |
| **Prometheus + Grafana** | `observability/` | Latency, cache hit rate, faithfulness score, abstain rate |
| **Manual Kafka offset commit** | consumers | Exactly-once processing guarantee |
| **Batch upsert (256 vectors)** | `index_consumer.py` | Qdrant throughput vs single-insert |

---

## 🔢 Scale Math

```
10M source docs
× 5 avg chunks/doc
= 50M chunks (vectors)

Vector memory (float32):   50M × 384 dims × 4 bytes = 72 GB
After INT8 quantization:   50M × 384 dims × 1 byte  = 18 GB  ← fits 24 GB GPU RAM

HNSW index overhead (m=16):  50M × 16 links × 8 bytes = 6.4 GB
Total Qdrant RAM estimate:    ~25 GB with quantization

Ingestion throughput (20 embed consumers, 1000 chunks/s/consumer):
  50M ÷ (20 × 1000) = 2500 s ≈ 42 min

Query latency budget (P99 < 500ms):
  Semantic cache hit:         < 5ms
  Qdrant vector search:      15-30ms (HNSW ef=128)
  BM25 search:               10ms
  Cross-encoder rerank (5):  50ms
  LLM generation:           200-400ms
  NLI check (5 sentences):   80ms
  Total (cache miss):        ~400ms ✓
```

---

## 🚀 Quick Start

```bash
cd projects/project17_enterprise_rag/starter

# 1. Create virtualenv and install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download spaCy model (needed for sentence splitting in faithfulness checker)
python -m spacy download en_core_web_sm

# 3. Configure environment
cp .env.example .env
# Edit .env — set MODEL and your LLM API key

# 4. Start all services
docker-compose up --build

# 5. Verify Kafka topics were created
open http://localhost:8090         # Kafka UI

# 6. Ingest documents (publishes to Kafka, consumed by workers)
python scripts/ingest.py --source data/sample_docs/ --verbose

# 7. Check ingestion progress
curl http://localhost:8000/stats

# 8. Query (with zero-hallucination guarantee)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the API rate limits?"}'

# 9. View metrics dashboard
open http://localhost:3000         # Grafana (admin/admin)

# 10. Run hallucination evaluation harness
python scripts/evaluate.py
```

---

## 🚫 Zero Hallucination — How It Works

```
Answer generation  →  NLI faithfulness check  →  Citation verification

For each sentence in the LLM's answer:
  1. Build premise = concatenated top-5 retrieved chunks
  2. Feed (premise, sentence) to cross-encoder/nli-deberta-v3-base
  3. Get probabilities: P(entailment), P(neutral), P(contradiction)

  If P(entailment) ≥ 0.75:  ✓  sentence is grounded — keep it
  If P(entailment) < 0.75:  ✗  sentence is NOT grounded — remove it

If fraction of grounded sentences < 0.80:
  → Abstain: return {"abstained": true, "abstain_reason": "insufficient_grounding"}

If retrieval max cosine similarity < 0.65:
  → Abstain: return {"abstained": true, "abstain_reason": "no_relevant_documents"}

Final answer = only grounded sentences + citations mapping each to its source chunk.
```

---

## Milestones

### Milestone 1 — Qdrant Store
Implement `src/store/qdrant_store.py`.
`docker-compose up qdrant`, create collection, upsert 100 test vectors, verify search returns correct results.

### Milestone 2 — Kafka Pipeline
Implement `kafka_producer.py` and all three consumers.
`docker-compose up kafka qdrant redis`, publish 10 test documents, watch them flow through chunker → embedder → indexer.
Verify `GET /stats` shows chunk count.

### Milestone 3 — Redis Caching
Implement `redis_cache.py` and `semantic_cache.py`.
Verify embedding cache hits on repeated text.
Verify query cache returns `cached: true` on repeated questions.
Verify semantic cache returns cached answer for paraphrased question.

### Milestone 4 — Zero Hallucination ★
Implement all files in `src/hallucination/`.
Test with a question that has no relevant documents → should abstain.
Test with a hallucination-prone question → faithfulness checker should remove invented sentences.
Run `python scripts/evaluate.py` — faithfulness ≥ 0.90 on the golden dataset.

### Milestone 5 — RAG Agent
Implement `src/retrieval/` and `src/agents/rag_agent.py`.
Full pipeline: question → hybrid retrieval → reranker → faithfulness → answer + citations.

### Milestone 6 — Production API
Implement `src/api/` and `src/observability/`.
`docker-compose up --build` — all 10 services running.
Run `python scripts/evaluate.py` with full stack.
View Grafana dashboard — latency, cache hit rate, abstain rate.

### Milestone 7 — Load Test
Run the benchmark: `python scripts/benchmark.py --rps 50 --duration 60`.
Verify P99 latency < 500ms with cache, < 600ms without.
Scale embed-consumer replicas to 8 and measure ingestion throughput improvement.

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) on first run.
> `cross-encoder/nli-deberta-v3-base` downloads ~750 MB on first faithfulness check.
> Both are cached in `~/.cache/huggingface/` after that.
>
> `confluent-kafka` requires `librdkafka`. Install via:
> macOS: `brew install librdkafka`
> Ubuntu: `apt-get install librdkafka-dev`
