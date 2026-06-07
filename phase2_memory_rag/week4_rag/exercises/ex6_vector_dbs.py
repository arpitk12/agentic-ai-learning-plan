"""
Exercise 6: Vector Database Comparison — FAISS vs ChromaDB vs Qdrant
Guide Sections: §2.12 (Qdrant), §2.13 (FAISS), §5 (Vector Search Reference)

Goal: Build the same search index in all three databases, run identical queries,
and understand the trade-offs through direct comparison.

Decision guide:
  FAISS    → research / batch processing / offline pipelines (no server needed)
  ChromaDB → development / prototyping / small production (<1M vectors)
  Qdrant   → production / large scale / advanced filtering / multi-tenant

pip install faiss-cpu chromadb sentence-transformers numpy
pip install qdrant-client   # optional — needs: docker run -p 6333:6333 qdrant/qdrant
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dotenv import load_dotenv
load_dotenv()

import faiss
import chromadb
import numpy as np
import time
from sentence_transformers import SentenceTransformer

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384  # all-MiniLM-L6-v2 output dimension


# ─── Dataset ──────────────────────────────────────────────────────────────────

DOCUMENTS = [
    {"text": "Python is a general-purpose language known for readability and simplicity.", "category": "programming", "year": 2023},
    {"text": "FastAPI builds high-performance REST APIs with automatic OpenAPI documentation.", "category": "web", "year": 2023},
    {"text": "Redis is an in-memory key-value store for caching, sessions, and pub/sub.", "category": "database", "year": 2023},
    {"text": "Docker containers package applications with all dependencies for portability.", "category": "devops", "year": 2024},
    {"text": "Kubernetes orchestrates containerized workloads across clusters of nodes.", "category": "devops", "year": 2024},
    {"text": "PostgreSQL is an ACID-compliant relational database with powerful JSON support.", "category": "database", "year": 2023},
    {"text": "LangGraph builds stateful AI agent workflows using typed directed graphs.", "category": "ai", "year": 2024},
    {"text": "RAG retrieves relevant documents to ground LLM answers in factual sources.", "category": "ai", "year": 2024},
    {"text": "Prometheus collects and stores time-series metrics using a pull-based model.", "category": "monitoring", "year": 2023},
    {"text": "Celery distributes background tasks across Python worker processes via Redis.", "category": "backend", "year": 2023},
    {"text": "FAISS performs efficient approximate similarity search on dense vectors.", "category": "ai", "year": 2023},
    {"text": "ChromaDB stores and queries embeddings locally for RAG development pipelines.", "category": "ai", "year": 2024},
    {"text": "Qdrant is a Rust-based production vector database with advanced filtering.", "category": "ai", "year": 2024},
    {"text": "Sentence transformers encode text into semantic embedding vectors.", "category": "ai", "year": 2023},
    {"text": "OpenTelemetry provides distributed tracing and metrics for microservices.", "category": "monitoring", "year": 2024},
    {"text": "Celery Flower is a web-based tool for monitoring Celery task queues.", "category": "backend", "year": 2023},
    {"text": "Grafana visualizes time-series data from Prometheus as dashboards.", "category": "monitoring", "year": 2023},
    {"text": "CrewAI orchestrates role-based multi-agent pipelines for complex tasks.", "category": "ai", "year": 2024},
    {"text": "Pydantic validates Python data structures using type annotations at runtime.", "category": "programming", "year": 2023},
    {"text": "asyncio enables concurrent I/O-bound tasks in a single Python thread.", "category": "programming", "year": 2023},
]

TEXTS = [d["text"] for d in DOCUMENTS]
# Normalize for cosine similarity (inner product = cosine for unit vectors)
EMBEDDINGS = EMBEDDER.encode(TEXTS, normalize_embeddings=True).astype(np.float32)


# ─── Database 1: FAISS ─────────────────────────────────────────────────────────

class FAISSVectorDB:
    """
    FAISS: Facebook AI Similarity Search (C++ with Python bindings)

    Architecture: Pure in-process library. No server, no persistence by default.
    Index types:
      IndexFlatIP    — exact search, dot product (= cosine if vectors are normalized)
      IndexHNSWFlat  — approximate, graph-based, fast queries, high memory
      IndexIVFFlat   — approximate, cluster-based, good for large datasets (needs training)

    We use IndexFlatIP (exact) + normalized vectors = cosine similarity.
    For datasets > 1M vectors, switch to IndexHNSWFlat or IndexIVFFlat.
    """

    def __init__(self, index_type: str = "flat"):
        if index_type == "flat":
            # Exact inner product — guaranteed correct results
            self.index = faiss.IndexFlatIP(DIM)
        elif index_type == "hnsw":
            # Approximate, graph-based — best for real-time serving
            self.index = faiss.IndexHNSWFlat(DIM, 32)     # 32 = graph connectivity
            self.index.hnsw.efConstruction = 200           # quality during build
            self.index.hnsw.efSearch = 50                  # quality during search
        self.texts = []
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, texts: list[str], metadata: list[dict]):
        """Add vectors to the index. Must add embeddings in batches."""
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query: str, k: int = 3, filter_category: str = None) -> list[dict]:
        """
        Search for the top-k most similar documents.
        
        Filtering: FAISS has NO native metadata filtering.
        Approach: retrieve more candidates, then filter in Python.
        This is less efficient than ChromaDB/Qdrant native filtering.
        """
        q_vec = EMBEDDER.encode([query], normalize_embeddings=True).astype(np.float32)
        # Fetch more if filtering, since some results will be discarded
        fetch_k = k * 5 if filter_category else k
        scores, indices = self.index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = {
                "text": self.texts[idx],
                "score": float(score),
                "category": self.metadata[idx].get("category"),
            }
            # Post-search Python filter (inefficient vs. native DB filtering)
            if filter_category and doc["category"] != filter_category:
                continue
            results.append(doc)
            if len(results) >= k:
                break

        return results

    def save(self, path: str):
        """Persist index to disk — manual step required in FAISS."""
        faiss.write_index(self.index, path)
        print(f"  FAISS index saved to {path}")

    def load(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(path)


# ─── Database 2: ChromaDB ──────────────────────────────────────────────────────

class ChromaVectorDB:
    """
    ChromaDB: developer-friendly embedded vector database

    Architecture: Runs as a Python library (in-process). Optional server mode.
    Persistence: automatic with PersistentClient(path="./chroma_db").
    Filtering: native where={} conditions on metadata fields.
    Scale: recommended up to ~1M vectors. Above that, migrate to Qdrant.
    """

    def __init__(self, collection_name: str = "vdb_comparison"):
        self.client = chromadb.Client()  # in-memory for this exercise
        # Production: chromadb.PersistentClient(path="./chroma_db")
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, embeddings: np.ndarray, texts: list[str], metadata: list[dict]):
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
            ids=ids,
        )

    def search(self, query: str, k: int = 3, filter_category: str = None) -> list[dict]:
        """
        Native metadata filtering: where={"category": {"$eq": "ai"}}
        This is faster than FAISS post-filtering because only matching documents
        are considered during the vector search.
        """
        q_vec = EMBEDDER.encode([query], normalize_embeddings=True).tolist()[0]

        where = None
        if filter_category:
            where = {"category": {"$eq": filter_category}}

        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )

        return [
            {
                "text": doc,
                "score": round(1 - dist, 4),  # ChromaDB returns distance, we convert
                "category": meta.get("category"),
            }
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]


# ─── Database 3: Qdrant (optional) ────────────────────────────────────────────

class QdrantVectorDB:
    """
    Qdrant: production-grade vector database written in Rust

    Architecture: Runs as a separate service (Docker or cloud).
    Features:
      - Complex filtering: must/should/must_not conditions, geo filters, ranges
      - Payload indexing: indexes metadata fields for fast filtering
      - Hybrid search: dense + sparse vectors (BM25 + semantic)
      - Quantization: reduces memory 4x with minimal quality loss
      - Horizontal scaling: cluster mode for billion-scale datasets
      - gRPC: high-throughput API in addition to REST

    To start: docker run -p 6333:6333 qdrant/qdrant

    This class gracefully skips if Qdrant is not running.
    """

    def __init__(self, collection_name: str = "vdb_comparison"):
        self.collection_name = collection_name
        self.available = False

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

            self.client = QdrantClient(host="localhost", port=6333, timeout=3)
            self.client.get_collections()  # test connection

            self._QdrantClient = QdrantClient
            self._Distance = Distance
            self._VectorParams = VectorParams
            self._PointStruct = PointStruct
            self._Filter = Filter
            self._FieldCondition = FieldCondition
            self._MatchValue = MatchValue
            self.available = True
            print("  ✓ Qdrant server connected")
        except Exception as e:
            print(f"  ⚠️  Qdrant not available ({type(e).__name__}). Skipping Qdrant.")
            print("     Start it with: docker run -p 6333:6333 qdrant/qdrant")

    def add(self, embeddings: np.ndarray, texts: list[str], metadata: list[dict]):
        if not self.available:
            return

        self.client.recreate_collection(
            self.collection_name,
            vectors_config=self._VectorParams(
                size=DIM,
                distance=self._Distance.COSINE,
            ),
        )

        points = [
            self._PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"text": texts[i], **metadata[i]},
            )
            for i in range(len(texts))
        ]
        self.client.upsert(self.collection_name, points=points)

        # Create payload index for fast category filtering
        # In production: also index user_id, created_at, etc.
        from qdrant_client.models import PayloadSchemaType
        self.client.create_payload_index(
            self.collection_name,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def search(self, query: str, k: int = 3, filter_category: str = None) -> list[dict]:
        if not self.available:
            return []

        q_vec = EMBEDDER.encode([query], normalize_embeddings=True).tolist()[0]

        q_filter = None
        if filter_category:
            q_filter = self._Filter(
                must=[self._FieldCondition(
                    key="category",
                    match=self._MatchValue(value=filter_category),
                )]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=q_vec,
            query_filter=q_filter,
            limit=k,
            with_payload=True,
        )

        return [
            {
                "text": r.payload["text"],
                "score": round(r.score, 4),
                "category": r.payload.get("category"),
            }
            for r in results
        ]


# ─── Benchmark Utility ────────────────────────────────────────────────────────

def run_benchmark(
    db,
    name: str,
    queries: list[str],
    filter_category: str = None,
) -> dict:
    """Time a set of queries and return timing + result summary."""
    if not getattr(db, "available", True):
        return {}

    label = f"{name} (filter={filter_category})" if filter_category else name
    print(f"\n{'─'*55}")
    print(f"[{label}]")

    total_ms = 0.0
    for q in queries:
        t0 = time.time()
        results = db.search(q, k=3, filter_category=filter_category)
        elapsed = (time.time() - t0) * 1000
        total_ms += elapsed

        print(f"\n  Query: '{q[:50]}'  ({elapsed:.0f}ms)")
        for r in results[:3]:
            cat = r.get("category", "?")
            print(f"    [{cat}] score={r['score']:.3f} | {r['text'][:65]}…")

    avg_ms = total_ms / len(queries)
    return {"avg_query_ms": round(avg_ms, 1)}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Vector Database Comparison: FAISS vs ChromaDB vs Qdrant ===\n")
    print("Building indexes…")

    # ── Populate all databases ──
    t0 = time.time()
    faiss_db = FAISSVectorDB(index_type="flat")
    faiss_db.add(EMBEDDINGS, TEXTS, DOCUMENTS)
    print(f"  FAISS:    setup in {(time.time()-t0)*1000:.0f}ms")

    t0 = time.time()
    chroma_db = ChromaVectorDB()
    chroma_db.add(EMBEDDINGS, TEXTS, DOCUMENTS)
    print(f"  ChromaDB: setup in {(time.time()-t0)*1000:.0f}ms")

    t0 = time.time()
    qdrant_db = QdrantVectorDB()
    if qdrant_db.available:
        qdrant_db.add(EMBEDDINGS, TEXTS, DOCUMENTS)
        print(f"  Qdrant:   setup in {(time.time()-t0)*1000:.0f}ms")

    queries = [
        "How do I run background tasks asynchronously in Python?",
        "What database should I use for storing AI embeddings?",
        "How do I monitor my microservice metrics?",
    ]

    # ── Test 1: Unfiltered search ──
    print("\n" + "="*55)
    print("TEST 1: Unfiltered semantic search")
    stats = {}
    stats["faiss_nofilter"] = run_benchmark(faiss_db, "FAISS", queries)
    stats["chroma_nofilter"] = run_benchmark(chroma_db, "ChromaDB", queries)
    if qdrant_db.available:
        stats["qdrant_nofilter"] = run_benchmark(qdrant_db, "Qdrant", queries)

    # ── Test 2: Category-filtered search ──
    print("\n" + "="*55)
    print("TEST 2: Filtered search — category='ai' only")
    filtered_query = ["What vector databases are available for production use?"]
    stats["faiss_filter"] = run_benchmark(faiss_db, "FAISS", filtered_query, filter_category="ai")
    stats["chroma_filter"] = run_benchmark(chroma_db, "ChromaDB", filtered_query, filter_category="ai")
    if qdrant_db.available:
        stats["qdrant_filter"] = run_benchmark(qdrant_db, "Qdrant", filtered_query, filter_category="ai")

    # ── Test 3: FAISS persistence ──
    print("\n" + "="*55)
    print("TEST 3: FAISS persistence (save / load)")
    faiss_db.save("/tmp/test_faiss.index")
    faiss_db2 = FAISSVectorDB()
    faiss_db2.texts = faiss_db.texts  # texts must be saved separately (e.g., pickle)
    faiss_db2.metadata = faiss_db.metadata
    faiss_db2.load("/tmp/test_faiss.index")
    t0 = time.time()
    r = faiss_db2.search("semantic vector search", k=1)
    print(f"  Loaded FAISS query: {(time.time()-t0)*1000:.0f}ms | {r[0]['text'][:70]}")

    # ── Summary ──
    print(f"\n{'='*55}")
    print("DECISION GUIDE")
    print(f"{'='*55}")
    print("""
  ┌─────────────┬────────────┬──────────────┬───────────────────────────────────┐
  │ Database    │ Setup      │ Filtering    │ Best For                          │
  ├─────────────┼────────────┼──────────────┼───────────────────────────────────┤
  │ FAISS       │ pip only   │ Post-search  │ Research, batch, offline pipelines │
  │ ChromaDB    │ pip only   │ Native (where│ Dev, prototyping, <1M vectors      │
  │ Qdrant      │ Docker     │ Native+index │ Production, >1M vectors, multi-tenant│
  └─────────────┴────────────┴──────────────┴───────────────────────────────────┘

  Filtering efficiency:
  - FAISS:    fetch 15 → filter 5 → show top 3  (wasted I/O)
  - ChromaDB: fetch 3 from filtered set          (efficient)
  - Qdrant:   fetch 3 using payload index         (most efficient, scales to billions)

  When metadata filtering is critical (user_id, tenant, date range):
  → use ChromaDB (dev) or Qdrant (prod). Avoid FAISS.
""")

    # ─── CHALLENGES ───────────────────────────────────────────────────────────
    # TODO: Switch FAISS to IndexHNSWFlat and compare query speed vs IndexFlatIP
    # TODO: Add 1000 random documents and measure setup + query time for each DB
    # TODO: Try Qdrant's multi-condition filter: must=[category="ai", year>=2024]
    # TODO: Implement FAISS IndexIVFFlat (requires training on vectors first)
