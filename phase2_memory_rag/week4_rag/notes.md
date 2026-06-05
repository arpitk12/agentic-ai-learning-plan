# Week 4 — RAG, Vector Databases & Agent Memory

## What This Week Is About
LLMs have two fundamental limitations: a knowledge cutoff and a finite context window. RAG (Retrieval-Augmented Generation) solves both. Agent memory — episodic, semantic, and procedural — is what transforms a stateless chatbot into an agent that learns. This week covers both from the ground up.

---

## 1. The RAG Architecture

**RAG** (Retrieval-Augmented Generation) is the pattern of finding relevant documents from a knowledge base and injecting them into the LLM prompt before generation.

```
User Query
    │
    ▼
[Embed query → vector]
    │
    ▼
[Vector DB: find top-K nearest documents]
    │
    ▼
[Inject retrieved docs into prompt]
    │
    ▼
[LLM generates answer grounded in docs]
    │
    ▼
Final Answer (with citations)
```

**Why RAG beats fine-tuning for most use cases:**
- ✅ No GPU required — runs on any machine
- ✅ Knowledge is updateable (add new docs, no retraining)
- ✅ Transparent (you can see what was retrieved and why)
- ✅ No hallucination about knowledge you don't have — it just retrieves

---

## 2. Embeddings — Converting Text to Vectors

**What an embedding is**: A list of 384–3072 floating-point numbers representing the semantic meaning of text. Semantically similar text → similar vectors (small cosine distance).

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions, ~80MB, fast

sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",      # semantically similar → close vector
    "The stock market fell 3% today",  # unrelated → far vector
]
vectors = model.encode(sentences)  # shape: (3, 384)
print(vectors.shape)
```

### Embedding Model Comparison

| Model | Dimensions | Size | Speed | Quality | Use When |
|-------|-----------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | 80MB | ⚡⚡⚡ | Good | Development, cost-sensitive |
| `all-mpnet-base-v2` | 768 | 420MB | ⚡⚡ | Better | Production quality |
| `text-embedding-3-small` (OpenAI API) | 1536 | API | ⚡⚡ | Excellent | Production, OpenAI ecosystem |
| `text-embedding-3-large` (OpenAI API) | 3072 | API | ⚡ | Best | High-accuracy retrieval |

**Critical**: Use the SAME embedding model for indexing and querying. Mixing models produces garbage results.

---

## 3. Chunking — Preparing Documents for RAG

Documents are too long to embed as-is. You must split them into chunks small enough to embed meaningfully and large enough to contain useful context.

### Chunking Strategies

**1. Fixed-size chunking** — split every N characters/tokens:
```python
def chunk_fixed(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks
```
- ✅ Simple and fast
- ❌ Splits mid-sentence — loses context

**2. Recursive character splitting** (LangChain's default):
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # tries in order
)
chunks = splitter.split_text(document_text)
```
- ✅ Respects natural text boundaries
- ✅ Overlap prevents context loss at boundaries

**3. Semantic chunking** — split when topic changes:
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(OpenAIEmbeddings())
chunks = splitter.split_text(text)  # groups sentences with similar meaning
```
- ✅ Best retrieval quality
- ❌ Slower, requires embedding every sentence

**Rules of thumb:**
- Start with recursive splitting, chunk_size=500–1000, overlap=10-20%
- Smaller chunks = more precise retrieval but less context
- Larger chunks = more context but noisier retrieval
- Always store chunk metadata: source file, page number, section heading

---

## 4. FAISS — Fast In-Memory Vector Search

**What it is**: Facebook AI's library for fast vector similarity search. Runs entirely in memory/on disk locally. No server needed.

**Purpose**: Development, small datasets (<1M vectors), and production when you control deployment.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = ["Python is a programming language", "FAISS is a vector search library", "RAG improves LLM accuracy"]

# 1. Embed documents
vectors = model.encode(docs).astype(np.float32)

# 2. Create FAISS index
dim = vectors.shape[1]  # 384
index = faiss.IndexFlatIP(dim)   # Inner Product (use with normalized vectors for cosine similarity)
faiss.normalize_L2(vectors)       # normalize for cosine
index.add(vectors)

# 3. Search
query = "What is vector search?"
q_vec = model.encode([query]).astype(np.float32)
faiss.normalize_L2(q_vec)
distances, indices = index.search(q_vec, k=2)  # top-2 results

for i, idx in enumerate(indices[0]):
    print(f"Match {i+1}: {docs[idx]} (score: {distances[0][i]:.3f})")

# 4. Persist
faiss.write_index(index, "my_index.faiss")
index = faiss.read_index("my_index.faiss")
```

---

## 5. ChromaDB — Persistent Vector Database

**What it is**: An open-source, developer-friendly vector database that runs locally or as a server. Includes metadata filtering, persistent storage, and a simple Python API.

**Purpose**: The recommended choice for development and small-to-medium production deployments. Easier than FAISS for documents with metadata.

```python
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")  # saves to disk
collection = client.get_or_create_collection("documents")

# Add documents (Chroma handles embedding if you provide an embedding function)
collection.add(
    documents=["Python is great", "RAG improves accuracy", "ChromaDB is a vector DB"],
    metadatas=[{"source": "doc1", "page": 1}, {"source": "doc1", "page": 2}, {"source": "doc2", "page": 1}],
    ids=["id1", "id2", "id3"]
)

# Query
results = collection.query(
    query_texts=["vector database"],
    n_results=2,
    where={"source": "doc1"}  # metadata filter
)
print(results["documents"])
print(results["distances"])
```

---

## 6. Qdrant — Production-Grade Vector Database

**What it is**: A Rust-based vector database designed for production scale. Supports billions of vectors, filtering, payload indexing, and hybrid search.

**Purpose**: When you need production reliability, horizontal scaling, and advanced filtering. The choice for serious production deployments.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")  # or QdrantClient(url="http://localhost:6333")

# Create collection
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Upsert documents
points = [
    PointStruct(id=1, vector=embed("Python tutorial"), payload={"text": "Python tutorial", "source": "docs"}),
    PointStruct(id=2, vector=embed("RAG system design"), payload={"text": "RAG system", "source": "blog"}),
]
client.upsert(collection_name="knowledge_base", points=points)

# Search with filter
results = client.search(
    collection_name="knowledge_base",
    query_vector=embed("how to use Python"),
    query_filter={"must": [{"key": "source", "match": {"value": "docs"}}]},
    limit=5
)
```

### Vector DB Comparison

| DB | Best For | Scaling | Setup | Cost |
|----|---------|---------|-------|------|
| **FAISS** | Local dev, research | Single node | pip install | Free |
| **ChromaDB** | Dev + small prod | Single node | pip install | Free |
| **Qdrant** | Production | Distributed | Docker/cloud | Free/cloud |
| **Pinecone** | Fully managed prod | Serverless | API only | $70+/mo |
| **Weaviate** | Production + GraphQL | Distributed | Docker/cloud | Free/cloud |

---

## 7. Hybrid Search: Vector + BM25

Pure vector search misses exact keyword matches. Pure BM25 (keyword) misses semantic meaning. **Hybrid search** combines both for best accuracy.

```python
from rank_bm25 import BM25Okapi

# BM25 keyword search
tokenized_corpus = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(query.split())

# Vector search scores (from FAISS/Chroma/Qdrant)
vector_scores = [0.92, 0.78, 0.45]

# Reciprocal Rank Fusion (RRF) — best way to combine rankings
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

---

## 8. Agent Memory Types

Agents need more than a vector DB — they need a full memory system:

| Memory Type | What It Stores | Storage | Example |
|-------------|---------------|---------|---------|
| **Working** | Current conversation, active context | In-memory (messages list) | "User just asked about X" |
| **Episodic** | Past conversations, what happened when | Vector DB + timestamps | "Last Tuesday user asked about Y" |
| **Semantic** | Facts, knowledge, beliefs | Vector DB | "User prefers Python, works at Acme Corp" |
| **Procedural** | How to do things | Code/prompts | Tool definitions, workflows |

### Persistent Memory with SQLite
```python
import sqlite3, json
from datetime import datetime

conn = sqlite3.connect("agent_memory.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY,
        user_id TEXT,
        memory_type TEXT,  -- episodic, semantic, procedural
        content TEXT,
        embedding BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        importance FLOAT DEFAULT 0.5
    )
""")

def save_memory(user_id: str, content: str, memory_type: str = "episodic"):
    embedding = embed(content)
    conn.execute(
        "INSERT INTO memories (user_id, memory_type, content, embedding) VALUES (?,?,?,?)",
        (user_id, memory_type, content, embedding.tobytes())
    )
    conn.commit()
```

---

## Full RAG Pipeline

```python
def rag_query(question: str, collection, k: int = 3) -> str:
    # 1. Retrieve relevant chunks
    results = collection.query(query_texts=[question], n_results=k)
    context_chunks = results["documents"][0]
    
    # 2. Build grounded prompt
    context = "\n\n---\n\n".join(context_chunks)
    messages = [{
        "role": "user",
        "content": f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}"""
    }]
    
    # 3. Generate grounded answer
    return get_text(chat(messages=messages))
```

---

## Tools & Libraries Used This Week — Deep Dive

### sentence-transformers — The Free Embedding Engine

**What it actually is**: A Python library wrapping Hugging Face Transformer models specifically fine-tuned for producing sentence-level embeddings. These models are trained with contrastive learning — similar sentences get vectors that point in similar directions.

**Why use it over OpenAI's embedding API?**
- **Cost**: Zero. No API calls, no usage fees.
- **Privacy**: Data never leaves your machine.
- **Speed**: Local CPU can embed ~1000 sentences/second with MiniLM.
- **No rate limits**: Embed 1M documents without throttling.

**The tradeoff**: OpenAI's `text-embedding-3-large` produces slightly better quality embeddings for semantic search tasks. For most production RAG systems, `all-MiniLM-L6-v2` or `all-mpnet-base-v2` is sufficient.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Model lifecycle — create once, reuse everywhere
# Loading takes 2-5 seconds. Creating a new instance per request is a bug.
model = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB, 384 dimensions

# Batch embedding — MUCH faster than one at a time
texts = ["This is sentence one.", "Another sentence here.", "Third example."]
vectors = model.encode(
    texts,
    batch_size=64,           # process 64 at a time (tune based on RAM)
    show_progress_bar=True,  # show progress for large batches
    normalize_embeddings=True,  # normalize to unit length for cosine similarity
    convert_to_numpy=True,   # return as numpy arrays (not tensors)
)
print(f"Shape: {vectors.shape}")   # (3, 384)
print(f"L2 norm: {np.linalg.norm(vectors[0]):.4f}")  # 1.0000 (normalized)

# Cosine similarity without a vector DB
from sklearn.metrics.pairwise import cosine_similarity

query = "How does machine learning work?"
docs = ["ML is a type of AI", "Python is popular", "Neural networks learn from data"]

query_vec = model.encode([query], normalize_embeddings=True)
doc_vecs = model.encode(docs, normalize_embeddings=True)

similarities = cosine_similarity(query_vec, doc_vecs)[0]
print(sorted(zip(similarities, docs), reverse=True))
# → [(0.78, "ML is a type of AI"), (0.71, "Neural networks..."), (0.12, "Python...")]
```

---

### FAISS — When to Use vs ChromaDB

**FAISS is for you when**:
- You're doing offline batch processing (not real-time serving)
- You need maximum raw speed on a fixed dataset
- The dataset fits in RAM (or on disk with memory-mapped indices)
- You DON'T need metadata filtering

**ChromaDB is for you when**:
- You need metadata filtering (filter by user_id, category, date)
- You need persistent storage without managing FAISS save/load
- You need a simple API without thinking about index types
- You're building a production RAG system (up to ~1M vectors)

```python
# FAISS index types — what to choose:

# IndexFlatIP — EXACT search, inner product (use with normalized vectors = cosine)
# Use when: small dataset (<100K), need 100% accurate results
index_exact = faiss.IndexFlatIP(384)

# IndexHNSWFlat — APPROXIMATE graph-based search
# Use when: real-time serving, <50M vectors, ~1% accuracy tradeoff acceptable  
index_hnsw = faiss.IndexHNSWFlat(384, 32)  # 32 = neighbor count per node
index_hnsw.hnsw.efConstruction = 200       # higher = better index quality, slower build
index_hnsw.hnsw.efSearch = 64             # higher = better search quality, slower query

# IndexIVFFlat — cluster-based approximate
# Use when: very large datasets (>10M vectors), batch workloads
n_clusters = int(np.sqrt(n_vectors))  # rule of thumb: sqrt(N) clusters
quantizer = faiss.IndexFlatL2(384)
index_ivf = faiss.IndexIVFFlat(quantizer, 384, n_clusters)
index_ivf.train(vectors)      # REQUIRED before add
index_ivf.nprobe = 20         # search this many clusters (higher = more accurate)
```

---

### ChromaDB — The Developer's Best Friend

**ChromaDB's under-the-hood**: ChromaDB uses SQLite for metadata storage and HNSW (from the `hnswlib` library) for vector indexing. The `PersistentClient` writes both to disk automatically.

**The embedding function pattern** — ChromaDB can embed FOR you:
```python
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,  # local model
    OpenAIEmbeddingFunction,               # OpenAI API
    GoogleGenerativeAiEmbeddingFunction,   # Google API
)

# Configure once — ChromaDB auto-embeds when you add/query text
ef = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

client = chromadb.PersistentClient("./chroma_db")
collection = client.get_or_create_collection("knowledge_base", embedding_function=ef)

# Add documents — ChromaDB embeds them for you
collection.add(
    documents=["Python is great", "FAISS is fast", "ChromaDB is easy"],
    ids=["doc_1", "doc_2", "doc_3"],
    metadatas=[{"topic": "language"}, {"topic": "search"}, {"topic": "db"}]
)

# Query — ChromaDB embeds the query for you  
results = collection.query(
    query_texts=["what are good programming languages?"],
    n_results=2,
    where={"topic": "language"},  # metadata filter (MongoDB-like)
    where_document={"$contains": "great"},  # filter by document content
)
```

**ChromaDB metadata filters** — powerful and underused:
```python
# Single condition
where={"user_id": "user_123"}

# AND condition
where={"$and": [{"user_id": "user_123"}, {"category": "technical"}]}

# OR condition
where={"$or": [{"category": "news"}, {"category": "research"}]}

# Range query (for numeric fields)
where={"year": {"$gte": 2024}}

# Content filter
where_document={"$contains": "machine learning"}
where_document={"$not_contains": "deprecated"}
```

---

### Qdrant — Production Vector Database Explained

**Why Qdrant over ChromaDB for production**:
- Written in Rust (10-50x less memory for same dataset)
- True horizontal scaling (add nodes to handle more data)
- Advanced filtering: nested conditions, geo-filters, multi-vector search
- On-disk storage with memory-mapped files (handles datasets larger than RAM)
- Built-in GRPC for high-throughput batch operations

**Key Qdrant concepts**:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Connection options:
# Local file storage (development)
client = QdrantClient(path="./qdrant_data")

# In-memory (testing)
client = QdrantClient(":memory:")

# Remote server (production)
client = QdrantClient(host="qdrant-server", port=6333, api_key="your-key")

# Collection = table in RDBMS terms
# Each point = a row (id + vector + payload)
client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Upsert (insert or update)
client.upsert("my_docs", points=[
    PointStruct(
        id=1,
        vector=[0.1, 0.2, ...],  # 384-dim vector
        payload={"text": "...", "user_id": 42, "source": "wiki"}  # arbitrary metadata
    )
])

# Search with filter — the killer feature
results = client.search(
    collection_name="my_docs",
    query_vector=query_embedding.tolist(),
    query_filter=Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=42))]
    ),
    limit=5,
    score_threshold=0.7,  # only return results with similarity >= 0.7
)
```

---

### rank-bm25 — Why BM25 Still Matters in 2025

**BM25 vs Vector Search — when each wins**:

```
Query: "Python 3.12 release date"
─────────────────────────────────
BM25: Excellent — exact keyword match on "Python 3.12" and "release date"
Vector: Good — finds semantically similar content but may miss exact version numbers

Query: "How does backpropagation work mathematically?"  
─────────────────────────────────
BM25: Poor — may miss documents that explain backpropagation using different words
Vector: Excellent — finds conceptually related explanations regardless of exact wording

Query: "AAPL stock price today"
─────────────────────────────────
BM25: Good — "AAPL" as a keyword is specific
Vector: Poor — "AAPL" has no semantic relationship to "Apple stock"
```

**Hybrid search in practice** (why it consistently outperforms either alone):
```python
from rank_bm25 import BM25Okapi
import numpy as np

# Build BM25 index
corpus = ["Machine learning is powerful", "Python is a programming language", 
          "Neural networks mimic the brain"]
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def hybrid_search_rrf(query: str, top_k: int = 3) -> list[str]:
    """
    Hybrid search using Reciprocal Rank Fusion.
    RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each ranking system i
    k=60 is the standard RRF constant (prevents top-ranked docs from dominating)
    """
    # BM25 scores
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranks = np.argsort(bm25_scores)[::-1]  # descending
    
    # Vector scores (simplified — use your embedding model)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_vec = model.encode([query], normalize_embeddings=True)
    d_vecs = model.encode(corpus, normalize_embeddings=True)
    vec_scores = (q_vec @ d_vecs.T)[0]
    vec_ranks = np.argsort(vec_scores)[::-1]
    
    # RRF combination
    k = 60
    rrf_scores = {}
    for rank, doc_idx in enumerate(bm25_ranks):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)
    for rank, doc_idx in enumerate(vec_ranks):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)
    
    top_docs = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    return [corpus[i] for i in top_docs]
```

---

## Memory Architecture — How Agents Remember Things

### The Four Memory Stores — With Implementation

```python
# 1. WORKING MEMORY — the messages list (always active)
# This IS the agent's short-term memory. Everything in here is "in mind."
working_memory = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user", "content": "What's my name?"}
]

# 2. EPISODIC MEMORY — what happened, when
# Store interaction summaries and retrieve by recency or similarity
import sqlite3
from datetime import datetime

def store_episode(user_id: str, summary: str, embedding: list):
    """Store an interaction episode."""
    db.execute(
        "INSERT INTO episodes (user_id, summary, embedding, timestamp) VALUES (?,?,?,?)",
        (user_id, summary, json.dumps(embedding), datetime.now().isoformat())
    )

def retrieve_recent_episodes(user_id: str, limit: int = 5) -> list[str]:
    """Get recent episodes for context."""
    rows = db.execute(
        "SELECT summary FROM episodes WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    return [r[0] for r in rows]

# 3. SEMANTIC MEMORY — facts and knowledge
# Store facts the agent learns about the user or domain
def store_fact(user_id: str, fact: str, category: str = "user_preference"):
    """Remember a fact: 'User prefers Python over JavaScript'"""
    embedding = model.encode(fact).tolist()
    collection.add(
        documents=[fact],
        metadatas=[{"user_id": user_id, "category": category}],
        ids=[f"{user_id}_{hash(fact)}"]
    )

def recall_facts(user_id: str, query: str, k: int = 3) -> list[str]:
    """Retrieve relevant facts for a query."""
    results = collection.query(
        query_texts=[query],
        where={"user_id": user_id},
        n_results=k
    )
    return results["documents"][0]

# 4. PROCEDURAL MEMORY — how to do things (tool definitions, workflows)
# This is your tool registry and system prompts
PROCEDURES = {
    "search_web": web_search,
    "run_python": execute_python,
    "query_database": run_sql,
}
```

---

## Common Pitfalls — Week 4

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Different embedding model for indexing vs querying | Terrible retrieval quality (random results) | Store model name in DB metadata, validate on load |
| No chunk overlap | Missing context at chunk boundaries | Use `chunk_overlap=200` (20% of chunk size) |
| Embedding the query as a document | Off-by-one in similarity math | Some models have separate `query_` and `passage_` prefixes |
| Not normalizing vectors before cosine similarity | Wrong similarity scores | Always `normalize_embeddings=True` or `faiss.normalize_L2()` |
| Chunk size too large | Noisy retrieval (chunks contain both relevant and irrelevant info) | Start with 500-1000 chars, tune based on evaluation |
| Chunk size too small | Insufficient context in LLM answer | Use parent-child chunking or increase chunk_size |
| ChromaDB collection name reuse with different model | Corruption — old embeddings don't match new queries | Include model name in collection name: `f"docs_{model_name}"` |
| Not storing source metadata | Can't cite sources | Always add `{"source": filename, "page": n}` metadata |
- `ex2_chunking_strategies.py` — compare fixed vs recursive vs semantic chunking
- `ex3_hybrid_search.py` — combine BM25 + vector for better retrieval
- `ex4_agent_memory.py` — agent that remembers facts about users across sessions

## Checklist
- [ ] Understood why cosine similarity works for text (normalized vectors)
- [ ] Ingested a real PDF into ChromaDB with metadata
- [ ] Compared retrieval quality with chunk_size 200 vs 1000 on same query
- [ ] Implemented hybrid search: BM25 + vector + RRF combination
- [ ] Built episodic memory: agent recalls what user said in previous session
