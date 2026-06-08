# Enterprise RAG Build Guide — 10M Docs, Zero Hallucination

> **How to use this guide**
> Work through each phase in order. Each phase ends with a concrete checkpoint
> you can verify before moving on. Refer to `solution/` only when stuck.

---

## Prerequisites

```bash
# macOS
brew install librdkafka  # required by confluent-kafka Python package

# Ubuntu
apt-get install librdkafka-dev

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Phase 0 — Understand the Problem Space

Before writing any code, internalize the two hard constraints:

### Why 10M docs demands a different architecture

With ChromaDB / a local vector store on a single machine:
- 10M docs × 5 chunks = 50M vectors × 384 dims × 4 bytes = **72 GB** — doesn't fit in most server RAM
- A single Python embedding loop takes ~14 hours for 50M chunks at 1000/s
- A single HTTP API thread becomes the bottleneck for concurrent ingestion

The solution: **distributed vector store** (Qdrant with sharding) + **message queue pipeline** (Kafka) that scales embedding horizontally.

### Why zero hallucination demands an explicit verification step

LLMs hallucinate because:
1. They blend retrieved context with parametric knowledge (memorized facts)
2. They continue predicting tokens even when context doesn't support the claim
3. Grounding prompts ("answer ONLY from context") reduce but don't eliminate it

The solution: **NLI-based post-hoc verification** — after the LLM generates an answer, a separate model (cross-encoder/nli-deberta-v3-base) checks each sentence for entailment against the retrieved chunks. Sentences that are not entailed are removed or the entire answer is rejected.

---

## Phase 1 — Qdrant Vector Store

**File to implement:** `src/store/qdrant_store.py`

### 1.1 Why Qdrant over ChromaDB / Pinecone

| | ChromaDB | Pinecone | Qdrant |
|---|---|---|---|
| 50M vectors | ❌ memory-bound | ✓ (expensive) | ✓ (open source) |
| Self-hosted | ✓ | ❌ | ✓ |
| Quantization | ❌ | managed | INT8 / PQ |
| Payload filtering | basic | ✓ | ✓ fast indexed |
| gRPC | ❌ | ❌ | ✓ |
| Sharding | ❌ | managed | ✓ |

### 1.2 HNSW index explained

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where:
- Bottom layer: all vectors connected to their m=16 nearest neighbors
- Upper layers: progressively fewer vectors, used to navigate quickly

Key parameters:
```
m=16            — each vector stores 16 bi-directional links
                  higher m = better recall, more RAM
ef_construct=200 — accuracy of index build (more = slower build, better recall)
ef=128          — accuracy of search (higher = slower query, better recall)

Rule of thumb for production:
  m=16, ef_construct=200 → 95-99% recall
  m=32, ef_construct=400 → 99%+ recall (double the RAM)
```

### 1.3 INT8 scalar quantization math

```
float32 vector: 384 × 4 bytes = 1536 bytes per vector
int8 vector:    384 × 1 byte  =  384 bytes per vector  (4× compression)

50M vectors:
  float32: 50M × 1536 bytes = 72 GB
  int8:    50M × 384 bytes  = 18 GB  ← fits comfortably in 24 GB GPU

Recall trade-off: ~1% recall degradation at INT8
Mitigate with: rescore=True (re-rank top results with original float32 vectors)
```

### 1.4 Step-by-step implementation

**Step 1: Start Qdrant**
```bash
docker-compose up qdrant
curl http://localhost:6333/healthz  # → {"title":"qdrant - vector search engine",...}
```

**Step 2: Create collection**

Open `src/store/qdrant_store.py`. Implement `create_collection()`:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="enterprise_docs",
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=200,
        ),
    ),
    quantization_config=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,
        quantile=0.99,
        always_ram=True,  # keep quantized index in RAM
    ),
)
```

**Step 3: Create payload indexes** (critical for 10M docs — without this, filtering does a full scan)

```python
client.create_payload_index("enterprise_docs", "source", "keyword")
client.create_payload_index("enterprise_docs", "title", "keyword")
```

**Step 4: Implement batch upsert** (always batch — individual upserts are 10× slower)

```python
from qdrant_client.models import PointStruct
import uuid

points = [
    PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, item["chunk_id"])),
        vector=item["embedding"],
        payload={"text": item["text"], "source": item["source"], ...}
    )
    for item in batch
]
client.upsert(collection_name="enterprise_docs", points=points)
```

**Step 5: Implement search with oversampling** (get 2× more candidates, rescore with float32)

```python
results = client.search(
    collection_name="enterprise_docs",
    query_vector=query_embedding,
    limit=20,
    search_params=models.SearchParams(
        hnsw_ef=128,
        quantization=models.QuantizationSearchParams(
            rescore=True,       # compare with original float32 vectors
            oversampling=2.0,   # fetch 40, rescore, return 20
        ),
    ),
)
```

**Checkpoint:** Run `python -c "from src.store.qdrant_store import QdrantStore; s = QdrantStore('http://localhost:6333'); s.create_collection(); print(s.count())"` → `0`

---

## Phase 2 — Kafka Ingestion Pipeline

**Files to implement:** `src/ingestion/kafka_producer.py`, `src/consumers/*.py`

### 2.1 Why Kafka over Celery+Redis for this scale

| | Celery + Redis | Kafka |
|---|---|---|
| Message retention | No (consumed = gone) | Yes (configurable, default 7 days) |
| Replay on failure | ❌ | ✓ (reset offset) |
| Consumer groups | Basic | First-class, partition-level |
| Ordering guarantee | Per-queue | Per-partition |
| Throughput at scale | ~10K msg/s | ~1M msg/s |
| Backpressure | ❌ | ✓ (consumer lag) |
| At-least-once delivery | ❌ | ✓ (manual commit) |

For 50M messages (chunks), Kafka is the right tool. Celery is fine for Milestone-4-style async tasks.

### 2.2 Topic design

```
raw-documents      partitions=10   ← one partition per source/tenant for ordering
document-chunks    partitions=20   ← more parallelism for chunking
embedded-chunks    partitions=20   ← match embed consumer count
dlq-ingestion      partitions=5    ← dead letter queue (failed after 3 retries)
```

### 2.3 Manual offset commit — why it matters

```python
# ❌ WRONG: auto-commit (message marked consumed before processing)
consumer = Consumer({"enable.auto.commit": True})

# ✓ CORRECT: manual commit (only commit after successful processing + publish)
consumer = Consumer({"enable.auto.commit": False})
...
consumer.commit()  # ONLY after downstream publish succeeds
```

If the process crashes between consuming and committing, Kafka redelivers the message. This is **at-least-once** delivery — your consumers must be idempotent (Qdrant upsert is idempotent by chunk_id).

### 2.4 Step-by-step implementation

**Step 1: Start Kafka**
```bash
docker-compose up zookeeper kafka kafka-init kafka-ui
open http://localhost:8090   # Kafka UI — verify topics exist
```

**Step 2: Implement the producer** (`src/ingestion/kafka_producer.py`)

```python
from confluent_kafka import Producer
import json, uuid, time, hashlib

class DocumentProducer:
    def __init__(self, bootstrap_servers: str):
        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "acks": "all",              # wait for all replicas (reliability)
            "retries": 3,
            "retry.backoff.ms": 500,
        })

    def publish(self, text: str, title: str, source: str, metadata: dict = None) -> str:
        doc_id = str(uuid.uuid4())
        self.producer.produce(
            topic="raw-documents",
            key=doc_id.encode(),
            value=json.dumps({
                "doc_id": doc_id, "text": text, "title": title,
                "source": source, "metadata": metadata or {}, "ts": time.time()
            }).encode(),
            on_delivery=self._on_delivery,
        )
        self.producer.flush()
        return doc_id

    def _on_delivery(self, err, msg):
        if err:
            raise RuntimeError(f"Kafka delivery failed: {err}")
```

**Step 3: Implement the chunk consumer** (`src/consumers/chunk_consumer.py`)

```python
from confluent_kafka import Consumer, Producer

class ChunkConsumer:
    def run(self):
        consumer.subscribe(["raw-documents"])
        while True:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error():
                self._send_to_dlq(msg)
                consumer.commit()
                continue

            doc = json.loads(msg.value())
            chunks = chunker.chunk(doc["text"])

            for i, chunk in enumerate(chunks):
                producer.produce("document-chunks", value=json.dumps({
                    **doc,
                    "chunk_id": f"{doc['doc_id']}_{i}",
                    "chunk_index": i,
                    "text": chunk.text,
                }).encode())

            producer.flush()
            consumer.commit()   # ← only after successful publish
```

**Step 4: Implement the embed consumer** (`src/consumers/embed_consumer.py`)

Key optimization: batch 32 chunks before embedding, use embedding cache.

```python
# Batch accumulation pattern
batch = []
while True:
    msg = consumer.poll(0.05)   # 50ms poll timeout
    if msg and not msg.error():
        batch.append(json.loads(msg.value()))

    if len(batch) >= 32 or (batch and msg is None):
        # Check embedding cache first
        texts = [item["text"] for item in batch]
        embeddings = embedder.embed_batch(texts, use_cache=True)

        for item, emb in zip(batch, embeddings):
            producer.produce("embedded-chunks", value=json.dumps({**item, "embedding": emb}).encode())

        producer.flush()
        consumer.commit()
        batch.clear()
```

**Step 5: Implement the index consumer** (`src/consumers/index_consumer.py`)

```python
# Batch 256 vectors before upserting to Qdrant
batch = []
while True:
    msg = consumer.poll(0.05)
    if msg and not msg.error():
        batch.append(json.loads(msg.value()))

    if len(batch) >= 256:
        qdrant_store.upsert_batch(batch)
        consumer.commit()
        batch.clear()
```

**Checkpoint:**
```bash
# Publish a test document
python -c "
from src.ingestion.kafka_producer import DocumentProducer
p = DocumentProducer('localhost:9092')
p.publish('Hello world test document.', 'Test', 'test')
print('Published')
"
# Watch it flow through Kafka UI: localhost:8090
# Check Qdrant after ~5s: curl localhost:6333/collections/enterprise_docs
```

---

## Phase 3 — Redis Caching Layer

**Files to implement:** `src/cache/redis_cache.py`, `src/cache/semantic_cache.py`

### 3.1 Three-tier cache design

```
Tier 1: Exact query cache (Redis String)
  Key:    qcache:{sha256(normalized_question)[:16]}
  Value:  JSON of QueryResponse
  TTL:    3600 s (1 hour)
  Hit when: exact same question asked again

Tier 2: Semantic cache (in-memory + Redis)
  Store recent question embeddings in memory
  On new query: compute cosine similarity vs all cached embeddings
  Hit when: cosine_similarity ≥ 0.97
  (captures paraphrases: "API rate limits?" ≈ "how many API calls per minute?")

Tier 3: Embedding cache (Redis String)
  Key:    emb:{sha256(text)[:16]}
  Value:  JSON list of floats
  TTL:    86400 s (24 hours)
  Hit when: same text chunk is re-embedded (e.g., re-ingestion)
  Savings: 0 embedding model calls for duplicate text
```

### 3.2 Semantic cache implementation

The semantic cache compares embeddings in memory (fast, no Redis round-trip):

```python
import numpy as np
from collections import OrderedDict

class SemanticCache:
    def __init__(self, redis_client, embedder, threshold=0.97, max_size=5000):
        self._cache = OrderedDict()  # {norm_emb_bytes: answer_json}
        self._max = max_size
        self._threshold = threshold

    def get(self, question: str, q_emb: np.ndarray) -> str | None:
        q_norm = q_emb / np.linalg.norm(q_emb)
        for emb_bytes, answer in self._cache.items():
            cached_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = float(np.dot(q_norm, cached_emb))
            if sim >= self._threshold:
                return answer
        return None

    def set(self, question: str, q_emb: np.ndarray, answer: str):
        q_norm = (q_emb / np.linalg.norm(q_emb)).astype(np.float32)
        key = q_norm.tobytes()
        self._cache[key] = answer
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)  # evict oldest
```

> **Production note**: For multi-replica deployments, store embeddings in Redis with
> `HSET scache:{id} emb {bytes} ans {json}` and do the similarity computation server-side
> using **Redis Stack** with the `VSIM` command. This shares the cache across all API replicas.

**Checkpoint:**
```python
from src.cache.redis_cache import RedisCache
r = RedisCache("redis://localhost:6379/0")
r.set_query("what is the API rate limit?", '{"answer": "100 requests/min"}')
print(r.get_query("what is the API rate limit?"))   # → {"answer": ...}
print(r.get_query("unrelated question"))             # → None
```

---

## Phase 4 — Zero-Hallucination Layer ★

**Files to implement:** `src/hallucination/`

This is the most important phase. Take time to understand each component.

### 4.1 Why hallucination happens at the token level

```
Retrieved context: "The API allows 100 requests per minute."
LLM input:         [system prompt] + [context] + [question]

The LLM doesn't "read" context — it predicts the next token given all previous tokens.
If the context is ambiguous or the question primes a different answer, the model's
parametric memory (from pretraining) can dominate over the retrieved context.

Example:
  Question: "What is the API rate limit?"
  Context:  "The API allows 100 requests per minute per user."
  Hallucination: "The API allows 1000 requests per minute." (common API limit)

Post-hoc verification catches this: the NLI model sees
  premise = "The API allows 100 requests per minute per user."
  hypothesis = "The API allows 1000 requests per minute."
  → label: CONTRADICTION (entailment = 0.03) → sentence is rejected
```

### 4.2 NLI model: cross-encoder/nli-deberta-v3-base

Natural Language Inference classifies a (premise, hypothesis) pair as:
- **ENTAILMENT**: premise logically implies hypothesis
- **NEUTRAL**: premise neither implies nor contradicts hypothesis
- **CONTRADICTION**: premise contradicts hypothesis

We use it for faithfulness: premise = context chunks, hypothesis = generated sentence.

```python
from sentence_transformers import CrossEncoder
from scipy.special import softmax
import numpy as np

# Load once at startup (750 MB download, then cached)
model = CrossEncoder("cross-encoder/nli-deberta-v3-base", num_labels=3)

# Label order for this model: 0=contradiction, 1=entailment, 2=neutral
ENTAILMENT_IDX = 1

def entailment_score(premise: str, hypothesis: str) -> float:
    raw = model.predict([(premise, hypothesis)])
    probs = softmax(raw[0])
    return float(probs[ENTAILMENT_IDX])
```

### 4.3 Sentence splitting with spaCy

Sentence splitting is non-trivial (abbreviations, bullet points, etc.). Use spaCy:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def split_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]
```

### 4.4 Faithfulness checker — step by step

Implement `src/hallucination/faithfulness_checker.py`:

```python
class FaithfulnessChecker:
    def check(self, answer: str, context_chunks: list[str]) -> FaithfulnessResult:
        # 1. Build premise from top chunks (truncate to ~2000 chars)
        premise = " ".join(context_chunks[:5])[:2000]

        # 2. Split answer into sentences
        sentences = split_sentences(answer)

        # 3. Score each sentence
        sentence_results = []
        for sent in sentences:
            score = entailment_score(premise, sent)
            sentence_results.append(SentenceFaithfulness(
                sentence=sent,
                entailment_score=score,
                is_grounded=(score >= self.threshold),
            ))

        # 4. Compute faithfulness score
        n_grounded = sum(1 for s in sentence_results if s.is_grounded)
        faithfulness_score = n_grounded / len(sentence_results) if sentence_results else 0.0

        # 5. Build grounded answer (only supported sentences)
        grounded_answer = " ".join(s.sentence for s in sentence_results if s.is_grounded)

        return FaithfulnessResult(
            sentences=sentence_results,
            faithfulness_score=faithfulness_score,
            passed=(faithfulness_score >= self.overall_threshold),
            grounded_answer=grounded_answer,
        )
```

### 4.5 Abstain policy

Implement `src/hallucination/abstain_policy.py`:

```python
class AbstainPolicy:
    def check(
        self,
        retrieval_max_score: float,
        faithfulness_result: FaithfulnessResult,
    ) -> tuple[bool, str]:
        """Returns (should_abstain, reason)"""

        # Check 1: no relevant documents found
        if retrieval_max_score < self.min_retrieval_score:  # default: 0.65
            return True, "no_relevant_documents"

        # Check 2: NLI faithfulness too low
        if not faithfulness_result.passed:
            return True, "insufficient_grounding"

        # Check 3: all sentences were removed by faithfulness check
        if not faithfulness_result.grounded_answer.strip():
            return True, "all_sentences_ungrounded"

        return False, ""
```

### 4.6 Citation verifier

Map each sentence in the final answer to its best supporting chunk:

```python
class CitationVerifier:
    def verify(
        self, grounded_answer: str, chunks: list[dict]
    ) -> list[Citation]:
        sentences = split_sentences(grounded_answer)
        citations = []

        for sent in sentences:
            best_chunk = max(
                chunks,
                key=lambda c: entailment_score(" ".join(c["text"].split()[:100]), sent)
            )
            citations.append(Citation(
                sentence=sent,
                chunk_id=best_chunk["chunk_id"],
                document_title=best_chunk["title"],
                source=best_chunk["source"],
                chunk_text=best_chunk["text"][:200],
            ))

        return citations
```

**Checkpoint — hallucination eval:**
```bash
python scripts/evaluate.py
# Expected output:
# ✓ Question 1: faithfulness=0.92, abstained=False
# ✓ Question 2: faithfulness=0.88, abstained=False
# ✓ Question 3: abstained=True (reason=no_relevant_documents)
# Overall faithfulness: 0.91/1.00  [PASS ≥ 0.85]
# Abstain rate: 1/5 = 20%
```

---

## Phase 5 — RAG Agent

**Files to implement:** `src/retrieval/retriever.py`, `src/retrieval/reranker.py`, `src/agents/rag_agent.py`

### 5.1 Hybrid retrieval with RRF

```python
def hybrid_search(question: str, top_k: int = 20) -> list[dict]:
    q_emb = embedder.embed_text(question)

    # Parallel search
    vector_results = qdrant_store.search(q_emb, top_k=top_k)
    bm25_results = bm25_index.search(question, top_k=top_k)

    # RRF fusion: score = Σ 1/(k + rank_i),  k=60
    scores = defaultdict(float)
    for rank, item in enumerate(vector_results):
        scores[item["chunk_id"]] += 1 / (60 + rank + 1)
    for rank, item in enumerate(bm25_results):
        scores[item["chunk_id"]] += 1 / (60 + rank + 1)

    # Re-sort by RRF score
    all_chunks = {item["chunk_id"]: item for item in vector_results + bm25_results}
    ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [all_chunks[cid] for cid in ranked[:top_k]]
```

### 5.2 Cross-encoder reranker

The cross-encoder sees the full (question, chunk) pair — much more accurate than bi-encoder cosine similarity:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    pairs = [(question, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]
```

### 5.3 Full RAG agent pipeline

Implement `src/agents/rag_agent.py`:

```python
class RAGAgent:
    def answer(self, request: QueryRequest) -> QueryResponse:
        # 1. Check semantic cache
        q_emb = embedder.embed_text(request.question)
        cached = semantic_cache.get(request.question, q_emb)
        if cached:
            return QueryResponse(**cached, cached=True)

        # 2. Hybrid retrieval
        chunks = retriever.search(request.question, top_k=20)
        max_score = max(c["score"] for c in chunks) if chunks else 0.0

        # 3. Rerank top-20 → top-5
        chunks = reranker.rerank(request.question, chunks, top_n=5)

        # 4. Abstain if nothing relevant
        if max_score < cfg.min_retrieval_score:
            return QueryResponse(abstained=True, abstain_reason="no_relevant_documents", ...)

        # 5. Build grounded prompt
        context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
        prompt = GROUNDED_PROMPT.format(context=context, question=request.question)

        # 6. Generate
        answer = litellm.completion(model=cfg.model, messages=[{"role": "user", "content": prompt}])

        # 7. Faithfulness check
        result = faithfulness_checker.check(answer, [c["text"] for c in chunks])

        # 8. Abstain policy
        abstain, reason = abstain_policy.check(max_score, result)
        if abstain:
            return QueryResponse(abstained=True, abstain_reason=reason, ...)

        # 9. Citation verification
        citations = citation_verifier.verify(result.grounded_answer, chunks)

        # 10. Cache and return
        response = QueryResponse(
            answer=result.grounded_answer,
            citations=citations,
            faithfulness_score=result.faithfulness_score,
            retrieval_score=max_score,
        )
        semantic_cache.set(request.question, q_emb, response)
        return response
```

The grounded prompt template (critical for reducing hallucination BEFORE the NLI check):

```python
GROUNDED_PROMPT = """You are a precise assistant. Answer the question using ONLY the
information in the provided context. If the context does not contain the answer, write
exactly: "I cannot answer this based on the available documentation."
Do not add any information not present in the context.

Context:
{context}

Question: {question}

Answer:"""
```

**Checkpoint:**
```bash
python -c "
from src.agents.rag_agent import RAGAgent
agent = RAGAgent()
r = agent.answer_question('What is the API rate limit?')
print('Answer:', r.answer)
print('Faithfulness:', r.faithfulness_score)
print('Abstained:', r.abstained)
print('Citations:', len(r.citations))
"
```

---

## Phase 6 — Production API

**Files to implement:** `src/api/`

### 6.1 Circuit breaker for Qdrant / LLM

```python
from pybreaker import CircuitBreaker

# Opens after 5 failures, tries again after 30s
qdrant_breaker = CircuitBreaker(fail_max=5, reset_timeout=30)
llm_breaker = CircuitBreaker(fail_max=3, reset_timeout=60)

@qdrant_breaker
def _search_with_breaker(query_vector):
    return qdrant_store.search(query_vector)
```

If Qdrant is down, the circuit breaker opens — subsequent calls fail fast (< 1ms) instead
of timing out after 30s. The API returns a 503 with a clear error message.

### 6.2 Middleware stack

```python
# In app.py, order matters:
app.add_middleware(RateLimitMiddleware)   # ← outermost: reject before any processing
app.add_middleware(RequestIDMiddleware)  # ← attach X-Request-ID to every request
app.add_middleware(TimingMiddleware)     # ← add X-Latency-Ms to every response
```

### 6.3 Key API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | RAG query with faithfulness check |
| `POST` | `/ingest` | Publish document to Kafka |
| `GET`  | `/health` | Liveness probe (Kubernetes-ready) |
| `GET`  | `/stats` | Vector store stats + cache hit rate |
| `GET`  | `/metrics` | Prometheus metrics endpoint |

**Checkpoint:**
```bash
curl http://localhost:8000/health
# → {"status": "ok", "qdrant": "healthy", "redis": "healthy", "kafka": "healthy"}

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the API rate limit?"}'
# → {"answer": "...", "faithfulness_score": 0.91, "abstained": false, "citations": [...]}
```

---

## Phase 7 — Observability

**File to implement:** `src/observability/metrics.py`

### 7.1 Key metrics to expose

```python
from prometheus_client import Counter, Histogram, Gauge

# Query metrics
query_count = Counter("rag_queries_total", "Total queries", ["status"])
query_latency = Histogram("rag_query_latency_seconds", "Query latency",
                           buckets=[.05, .1, .25, .5, 1., 2., 5.])
faithfulness_score = Histogram("rag_faithfulness_score", "Faithfulness per query",
                                buckets=[.5, .6, .7, .8, .85, .9, .95, 1.0])

# Cache metrics
cache_hits = Counter("rag_cache_hits_total", "Cache hits", ["tier"])  # exact|semantic
cache_misses = Counter("rag_cache_misses_total", "Cache misses")

# Abstain metrics
abstain_count = Counter("rag_abstain_total", "Abstained queries", ["reason"])

# Ingestion metrics (in consumers)
docs_ingested = Counter("kafka_docs_ingested_total", "Documents ingested", ["stage"])
consumer_lag = Gauge("kafka_consumer_lag", "Consumer lag", ["topic", "group"])
```

### 7.2 Grafana dashboard panels

Import the provided `solution/grafana_dashboard.json` or build manually:

1. **P50/P95/P99 Query Latency** — histogram_quantile(0.99, rate(rag_query_latency...))
2. **Cache Hit Rate** — cache_hits / (cache_hits + cache_misses)
3. **Faithfulness Score Distribution** — heatmap of rag_faithfulness_score
4. **Abstain Rate** — abstain_count / query_count
5. **Kafka Consumer Lag** — kafka_consumer_lag by topic+group
6. **Documents Ingested Rate** — rate(kafka_docs_ingested_total[5m])

**Checkpoint:**
```bash
curl http://localhost:8000/metrics | grep rag_
# → rag_queries_total{status="success"} 42
# → rag_faithfulness_score_sum 38.2
# → rag_cache_hits_total{tier="semantic"} 7
```

---

## Phase 8 — Load Testing & Benchmarking

### 8.1 Ingestion throughput benchmark

```bash
# Measure how long it takes to ingest 10K documents end-to-end
python scripts/benchmark.py --mode ingest --count 10000

# Expected output:
# Ingestion throughput: 847 docs/s
# Chunking throughput: 4235 chunks/s
# Embedding throughput: 892 chunks/s (bottleneck — add more embed consumers)
# Qdrant upsert throughput: 3112 chunks/s
# Total pipeline latency (doc → indexed): 5.8s median
```

### 8.2 Query throughput benchmark

```bash
python scripts/benchmark.py --mode query --rps 50 --duration 60

# Expected output (cache warm):
# Requests:        3000
# Success:         2997 (99.9%)
# Abstained:        180 (6.0%)
# Cache hits:       720 (24.0%)
# P50 latency:     82ms
# P95 latency:    410ms
# P99 latency:    487ms   ← must be < 500ms
```

### 8.3 Scaling the embed consumer

The embed consumer is the throughput bottleneck. Each consumer runs one batch at a time.
To scale: increase `embed-consumer` replicas in docker-compose:

```yaml
embed-consumer:
  deploy:
    replicas: 8    # was 4 — doubles embedding throughput
```

With 8 replicas: `8 × 1000 chunks/s = 8000 chunks/s → 50M chunks in 6250s ≈ 104 min`

In Kubernetes, use HPA on Kafka consumer lag metric.

---

## Phase 9 — Docker Compose Deployment

```bash
# Full stack
docker-compose up --build

# Scale consumers
docker-compose up --scale embed-consumer=8 --scale chunk-consumer=4

# View consumer lag (tells you if ingestion is keeping up)
docker-compose exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe --group embedder-group

# Qdrant collection info
curl http://localhost:6333/collections/enterprise_docs \
  | python -m json.tool

# Prometheus targets
open http://localhost:9090/targets   # all services should be UP

# Grafana
open http://localhost:3000           # admin / admin
```

---

## Phase 10 — Production Checklist

Before shipping to prod, verify each item:

### Infrastructure
- [ ] Kafka: 3+ brokers, `replication-factor=3`, `min.insync.replicas=2`
- [ ] Qdrant: 4 shards, `replication_factor=2` (survives 1 node loss)
- [ ] Redis: Redis Sentinel or Cluster (no single point of failure)
- [ ] API: multiple replicas behind load balancer, readiness probe at `/health`

### Data pipeline
- [ ] DLQ monitored with alert on `dlq-ingestion` message count > 0
- [ ] Consumer lag alert: lag > 10K messages → page on-call
- [ ] Idempotent consumers: duplicate messages produce same result

### Zero-hallucination
- [ ] Faithfulness threshold set via `.env` (not hardcoded)
- [ ] Abstain rate monitored — spike means document coverage gap
- [ ] Regular evaluation runs on golden dataset in CI (reject deploy if score drops)
- [ ] All responses include `faithfulness_score` and `citations` in API response

### Caching
- [ ] Cache TTLs appropriate for your data update frequency
- [ ] Cache warming script runs after each ingestion batch
- [ ] Redis `maxmemory-policy: allkeys-lru` set (don't run out of memory)

### Observability
- [ ] Alerts: P99 > 500ms, error rate > 1%, abstain rate > 30%, consumer lag > 10K
- [ ] Every request has `X-Request-ID` (correlate API log → Kafka → Qdrant)
- [ ] Grafana dashboards reviewed and on-call team trained

### Security
- [ ] Kafka: SASL/TLS authentication enabled
- [ ] Qdrant: API key authentication enabled
- [ ] Redis: `requirepass` set
- [ ] Rate limiting: by IP + by API key
