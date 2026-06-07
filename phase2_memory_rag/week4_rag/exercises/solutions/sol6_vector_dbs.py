"""
SOLUTION — Exercise 6: Vector Database Comparison (FAISS vs ChromaDB vs Qdrant)

Key concepts demonstrated:
- FAISS IndexFlatIP: exact cosine search (after L2 normalisation). No server needed.
- FAISS IndexHNSWFlat: approximate graph-based search. Faster for large sets.
- ChromaDB: embedded DB with native where={} metadata filtering.
- Qdrant: production DB with payload indexes for fast multi-condition filtering.
- Filtering: FAISS must post-filter (fetch more, discard); Chroma/Qdrant filter natively.
- Persistence: FAISS manual (write_index/read_index); Chroma auto; Qdrant auto.

Decision rule:
  FAISS    → research, offline batch, no server needed
  ChromaDB → dev, prototyping, production <1M vectors
  Qdrant   → production, >1M vectors, multi-tenant, advanced filtering

pip install faiss-cpu chromadb sentence-transformers numpy
pip install qdrant-client  # optional — needs: docker run -p 6333:6333 qdrant/qdrant
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dotenv import load_dotenv
load_dotenv()

import faiss
import chromadb
import numpy as np
import time
from sentence_transformers import SentenceTransformer

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384

DOCUMENTS = [
    {"text": "Python is a general-purpose language known for readability.",          "category": "programming", "year": 2023},
    {"text": "FastAPI builds high-performance REST APIs with auto OpenAPI docs.",    "category": "web",         "year": 2023},
    {"text": "Redis is an in-memory key-value store for caching and pub/sub.",      "category": "database",    "year": 2023},
    {"text": "Docker containers package apps with dependencies for portability.",    "category": "devops",      "year": 2024},
    {"text": "Kubernetes orchestrates containerised workloads across node clusters.","category": "devops",      "year": 2024},
    {"text": "PostgreSQL is ACID-compliant with powerful JSON and vector support.",  "category": "database",    "year": 2023},
    {"text": "LangGraph builds stateful AI agent workflows as directed graphs.",     "category": "ai",          "year": 2024},
    {"text": "RAG retrieves relevant documents to ground LLM answers in facts.",    "category": "ai",          "year": 2024},
    {"text": "Prometheus collects time-series metrics via a pull-based model.",     "category": "monitoring",  "year": 2023},
    {"text": "Celery distributes background tasks across Python workers via Redis.", "category": "backend",     "year": 2023},
    {"text": "FAISS performs efficient approximate similarity search on vectors.",   "category": "ai",          "year": 2023},
    {"text": "ChromaDB stores embeddings locally for RAG development pipelines.",   "category": "ai",          "year": 2024},
    {"text": "Qdrant is a Rust-based production vector DB with advanced filtering.", "category": "ai",          "year": 2024},
    {"text": "Sentence transformers encode text into semantic embedding vectors.",   "category": "ai",          "year": 2023},
    {"text": "OpenTelemetry provides distributed tracing for microservices.",       "category": "monitoring",  "year": 2024},
    {"text": "Grafana visualises Prometheus time-series data as dashboards.",       "category": "monitoring",  "year": 2023},
    {"text": "CrewAI orchestrates role-based multi-agent pipelines.",               "category": "ai",          "year": 2024},
    {"text": "Pydantic validates Python data with type annotations at runtime.",    "category": "programming", "year": 2023},
    {"text": "asyncio enables concurrent I/O-bound tasks in a single thread.",      "category": "programming", "year": 2023},
    {"text": "Celery Flower monitors Celery queues via a web dashboard.",           "category": "backend",     "year": 2023},
]

TEXTS = [d["text"] for d in DOCUMENTS]
# Normalise so dot-product == cosine similarity
VECS = EMBEDDER.encode(TEXTS, normalize_embeddings=True).astype(np.float32)


# ─── FAISS ────────────────────────────────────────────────────────────────────

class FAISSVectorDB:
    """
    Pure in-process library (C++ + Python bindings). No server.
    IndexFlatIP + normalised vectors = exact cosine similarity.

    Filtering: no native support. Post-filter: fetch k*multiplier, then discard.
    Persistence: manual write_index / read_index.
    """

    def __init__(self, index_type: str = "flat"):
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(DIM)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(DIM, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def add(self, vecs: np.ndarray, texts: list[str], meta: list[dict]):
        self.index.add(vecs)
        self.texts.extend(texts)
        self.metadata.extend(meta)

    def search(self, query: str, k: int = 3, filter_category: str = None) -> list[dict]:
        q_vec = EMBEDDER.encode([query], normalize_embeddings=True).astype(np.float32)
        fetch_k = k * 5 if filter_category else k
        scores, idxs = self.index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            doc = {"text": self.texts[idx], "score": float(score),
                   "category": self.metadata[idx].get("category")}
            if filter_category and doc["category"] != filter_category:
                continue
            results.append(doc)
            if len(results) >= k:
                break
        return results

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)


# ─── ChromaDB ─────────────────────────────────────────────────────────────────

class ChromaVectorDB:
    """
    Embedded vector DB (in-process). Auto-persistent with PersistentClient.
    Native where={} filtering — only matching docs searched.
    Recommended for dev and production up to ~1M vectors.
    """

    def __init__(self, name: str = "sol_vdb_cmp"):
        client = chromadb.Client()
        try:
            client.delete_collection(name)
        except Exception:
            pass
        self.coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    def add(self, vecs: np.ndarray, texts: list[str], meta: list[dict]):
        self.coll.add(
            embeddings=vecs.tolist(),
            documents=texts,
            metadatas=meta,
            ids=[f"d{i}" for i in range(len(texts))],
        )

    def search(self, query: str, k: int = 3, filter_category: str = None) -> list[dict]:
        q_vec = EMBEDDER.encode([query], normalize_embeddings=True).tolist()[0]
        where = {"category": {"$eq": filter_category}} if filter_category else None
        res = self.coll.query(
            query_embeddings=[q_vec], n_results=k, where=where,
            include=["documents", "distances", "metadatas"],
        )
        return [
            {"text": doc, "score": round(1 - dist, 4),
             "category": meta.get("category")}
            for doc, dist, meta in zip(
                res["documents"][0], res["distances"][0], res["metadatas"][0]
            )
        ]


# ─── Qdrant (optional) ────────────────────────────────────────────────────────

class QdrantVectorDB:
    """
    Rust-based production vector DB. Requires Docker or Qdrant Cloud.
    Features: payload indexes, hybrid search, quantisation, clustering.
    Start with: docker run -p 6333:6333 qdrant/qdrant
    """

    def __init__(self, name: str = "sol_vdb_cmp"):
        self.name = name
        self.available = False
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, PointStruct,
                Filter, FieldCondition, MatchValue,
            )
            self._client = QdrantClient(host="localhost", port=6333, timeout=3)
            self._client.get_collections()
            self._Distance = Distance
            self._VectorParams = VectorParams
            self._PointStruct = PointStruct
            self._Filter = Filter
            self._FieldCondition = FieldCondition
            self._MatchValue = MatchValue
            self.available = True
            print("  ✓ Qdrant connected")
        except Exception as e:
            print(f"  ⚠️  Qdrant unavailable ({type(e).__name__})")
            print("     Start: docker run -p 6333:6333 qdrant/qdrant")

    def add(self, vecs: np.ndarray, texts: list[str], meta: list[dict]):
        if not self.available:
            return
        from qdrant_client.models import PayloadSchemaType
        self._client.recreate_collection(
            self.name,
            vectors_config=self._VectorParams(size=DIM, distance=self._Distance.COSINE),
        )
        points = [
            self._PointStruct(id=i, vector=vecs[i].tolist(),
                              payload={"text": texts[i], **meta[i]})
            for i in range(len(texts))
        ]
        self._client.upsert(self.name, points=points)
        # Payload index → fast filtering without scanning all vectors
        self._client.create_payload_index(
            self.name, "category", PayloadSchemaType.KEYWORD
        )

    def search(self, query: str, k: int = 3, filter_category: str = None) -> list[dict]:
        if not self.available:
            return []
        q_vec = EMBEDDER.encode([query], normalize_embeddings=True).tolist()[0]
        q_filter = None
        if filter_category:
            q_filter = self._Filter(must=[
                self._FieldCondition(key="category",
                                     match=self._MatchValue(value=filter_category))
            ])
        results = self._client.search(
            self.name, query_vector=q_vec, query_filter=q_filter,
            limit=k, with_payload=True,
        )
        return [{"text": r.payload["text"], "score": round(r.score, 4),
                 "category": r.payload.get("category")} for r in results]


# ─── Benchmark helper ─────────────────────────────────────────────────────────

def bench(db, label: str, queries: list[str], filter_category: str = None) -> float:
    if not getattr(db, "available", True):
        return -1.0
    tag = f"{label}" + (f" [filter={filter_category}]" if filter_category else "")
    print(f"\n  ── {tag} ──")
    total = 0.0
    for q in queries:
        t0 = time.time()
        res = db.search(q, k=3, filter_category=filter_category)
        ms = (time.time() - t0) * 1000
        total += ms
        print(f"    Query: '{q[:45]}' ({ms:.0f}ms)")
        for r in res[:2]:
            print(f"      [{r['category']}] {r['score']:.3f} | {r['text'][:60]}…")
    return total / len(queries)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Vector DB Comparison: FAISS vs ChromaDB vs Qdrant ===\n")

    # Build indexes
    t = time.time()
    faiss_db = FAISSVectorDB("flat")
    faiss_db.add(VECS, TEXTS, DOCUMENTS)
    print(f"FAISS setup:    {(time.time()-t)*1000:.0f}ms")

    t = time.time()
    chroma_db = ChromaVectorDB()
    chroma_db.add(VECS, TEXTS, DOCUMENTS)
    print(f"ChromaDB setup: {(time.time()-t)*1000:.0f}ms")

    t = time.time()
    qdrant_db = QdrantVectorDB()
    if qdrant_db.available:
        qdrant_db.add(VECS, TEXTS, DOCUMENTS)
        print(f"Qdrant setup:   {(time.time()-t)*1000:.0f}ms")

    queries = [
        "How do I run async background jobs in Python?",
        "What database stores AI embeddings?",
        "How do I monitor microservice metrics?",
    ]

    # Test 1: unfiltered
    print("\n" + "="*55)
    print("TEST 1: Unfiltered semantic search")
    bench(faiss_db,  "FAISS",    queries)
    bench(chroma_db, "ChromaDB", queries)
    if qdrant_db.available:
        bench(qdrant_db, "Qdrant", queries)

    # Test 2: category filter
    print("\n" + "="*55)
    print("TEST 2: Filter — category='ai' only")
    q_ai = ["Which vector databases are suitable for production?"]
    bench(faiss_db,  "FAISS    (post-filter)", q_ai, "ai")
    bench(chroma_db, "ChromaDB (native)",      q_ai, "ai")
    if qdrant_db.available:
        bench(qdrant_db, "Qdrant   (payload-index)", q_ai, "ai")

    # Test 3: FAISS persistence
    print("\n" + "="*55)
    print("TEST 3: FAISS save / load")
    faiss_db.save("/tmp/sol_faiss.index")
    faiss_db2 = FAISSVectorDB()
    faiss_db2.texts = faiss_db.texts
    faiss_db2.metadata = faiss_db.metadata
    faiss_db2.load("/tmp/sol_faiss.index")
    t = time.time()
    r = faiss_db2.search("semantic vector search", k=1)
    print(f"  Loaded FAISS query: {(time.time()-t)*1000:.0f}ms | {r[0]['text'][:60]}…")

    # Summary
    print(f"""
{"="*55}
DECISION GUIDE
{"="*55}
  FAISS    — pip only, no server, no native filtering
             Use for: research, offline batch, fixed datasets

  ChromaDB — pip only, auto-persist, native where={{}} filtering
             Use for: dev, prototyping, production < 1M vectors

  Qdrant   — Docker/cloud, payload indexes, hybrid search
             Use for: production, multi-tenant, > 1M vectors,
             complex filtering (nested conditions, geo, ranges)

Filtering efficiency:
  FAISS:    fetch 15 → Python filter → return 3  (wasted vector I/O)
  ChromaDB: filter first → search in subset → return 3  (efficient)
  Qdrant:   payload index lookup → search subset → return 3  (fastest)
""")
