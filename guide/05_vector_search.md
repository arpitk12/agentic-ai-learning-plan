[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §4 Multi-Agent Patterns](guide/04_multi_agent.md) | [§6 Production Checklist →](guide/06_production_checklist.md)

---

## 5. Vector Search Reference — Choosing & Configuring Your Vector DB

### 5.1 Similarity Metrics — How Vectors Are Compared

Before choosing a DB, understand the math:

**Cosine Similarity** (most common for text):
$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}||\vec{B}|}$

Range: [-1, 1]. 1 = identical direction, 0 = orthogonal, -1 = opposite.
Use for: text, image features, anything semantic.

**Euclidean (L2) Distance**:
$d = \sqrt{\sum(A_i - B_i)^2}$

Range: [0, ∞). 0 = identical. Lower = more similar.
Use for: dense numerical features, not normalized embeddings.

**Dot Product (Inner Product)**:
$\text{sim} = \vec{A} \cdot \vec{B}$

Only meaningful for normalized vectors (equals cosine similarity when vectors are unit vectors). Fastest to compute.

**Rule**: Always normalize your embeddings (`faiss.normalize_L2()` or `normalize_embeddings=True`) and use cosine/dot product similarity. It's invariant to vector magnitude.

### 5.2 Database Selection — Full Comparison

| Criteria | FAISS | ChromaDB | Qdrant | Weaviate | Pinecone | pgvector |
|----------|-------|----------|--------|----------|----------|----------|
| **Setup complexity** | pip only | pip only | Docker | Docker/Cloud | API only | PostgreSQL |
| **Persistence** | Manual (save/load) | Auto | Auto | Auto | Auto | Auto |
| **Max scale** | Single node | ~5M vec | Billions | Billions | Serverless | ~10M vec |
| **Filtering** | ❌ | ✅ Basic | ✅ Advanced | ✅ GraphQL | ✅ | ✅ SQL |
| **Hybrid search** | ❌ | ❌ | ✅ Native | ✅ | ✅ | ✅ (with tsvector) |
| **Multi-tenancy** | Manual | Basic | ✅ | ✅ | ✅ | ✅ |
| **REST API** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Cloud managed** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ (Supabase) |
| **Cost** | Free | Free | Free/Cloud | Free/Cloud | $70+/mo | Free (PostgreSQL) |
| **Best for** | Batch/research | Dev/prototyping | Production | Production+ML | Serverless | SQL+vector |

### 5.3 When to Use Each Database

**FAISS** → Use when:
- Research or batch processing (offline, no serving)
- Need maximum raw speed for similarity search
- Small-medium dataset (fits in RAM)
- No need for metadata filtering or persistence

**ChromaDB** → Use when:
- Development and prototyping (zero config, just works)
- Small production (<500K vectors)
- Need metadata filtering
- Single-node deployment
- Just switched from in-memory to persistent

**Qdrant** → Use when:
- Production system with growth expected
- Need advanced filtering (nested conditions, geo filters)
- Want built-in hybrid search (dense + sparse)
- High query throughput required
- Need multi-tenant isolation

**pgvector** → Use when:
- Already using PostgreSQL (don't want another service)
- Small-medium scale
- Need ACID transactions with vector data
- SQL joins between vector data and relational data

**Pinecone** → Use when:
- Want fully managed (no infrastructure to run)
- Budget allows ($70+/month)
- Need serverless scaling

### 5.4 HNSW vs IVF — Index Types

**HNSW (Hierarchical Navigable Small World)** — the default for most DBs:
- Graph-based approximate nearest neighbor search
- Fast query time: O(log n)
- High memory usage (holds graph structure)
- Best for: real-time query serving, <100M vectors

**IVF (Inverted File Index)** — FAISS's production index:
- Divides vectors into clusters, searches only relevant clusters
- Lower memory than HNSW
- Requires training phase
- Best for: large datasets (>10M vectors), batch workloads

```python
# FAISS index selection
import faiss

dim = 384  # embedding dimensions

# Exact search — no approximation, always correct
index_flat = faiss.IndexFlatIP(dim)  # exact, cosine (after L2 normalize)

# HNSW — fast approximate, good for real-time
index_hnsw = faiss.IndexHNSWFlat(dim, 32)  # 32 = graph connectivity
index_hnsw.hnsw.efConstruction = 200  # quality during build (higher = better)
index_hnsw.hnsw.efSearch = 50        # quality during search (tune per use case)

# IVF — for large datasets
nlist = 100  # number of clusters (sqrt(n) is a good starting point)
quantizer = faiss.IndexFlatL2(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
index_ivf.train(vectors)  # required before adding
index_ivf.nprobe = 10    # clusters to search (higher = more accurate but slower)
```

### 5.5 Qdrant Production Setup

```python
# Full production Qdrant setup
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    HnswConfigDiff, OptimizersConfigDiff, ScalarQuantizationConfig, ScalarType
)

# Connect to production Qdrant
client = QdrantClient(
    url="http://qdrant-service:6333",
    api_key="your-qdrant-api-key",  # required if running with --api-key
    timeout=60,
    prefer_grpc=True,  # faster for large batch operations
)

# Create collection with production settings
client.recreate_collection(
    collection_name="prod_knowledge_base",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
        # Enable quantization to reduce memory by 4x
        quantization_config=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,  # keep quantized vectors in RAM
        ),
    ),
    hnsw_config=HnswConfigDiff(
        m=16,                  # graph connections per node (16-32 typical)
        ef_construct=200,      # quality during index build
        full_scan_threshold=10000,  # use brute force for collections < 10K
    ),
    optimizers_config=OptimizersConfigDiff(
        deleted_threshold=0.2,
        vacuum_min_vector_number=1000,
        default_segment_number=5,
    ),
)

# Create payload indexes for fast filtering
client.create_payload_index("prod_knowledge_base", "user_id", "keyword")
client.create_payload_index("prod_knowledge_base", "category", "keyword")
client.create_payload_index("prod_knowledge_base", "created_at", "integer")
```

### 5.6 pgvector — SQL + Vector Search

```sql
-- Install extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(384),     -- 384-dimensional vector
    user_id     BIGINT,
    category    VARCHAR(100),
    source_url  TEXT,
    created_at  TIMESTAMP DEFAULT NOW(),
    metadata    JSONB
);

-- Create HNSW index for fast similarity search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Similarity search with filters
SELECT 
    content,
    1 - (embedding <=> query_vector) AS similarity,  -- cosine similarity
    category,
    source_url
FROM documents
WHERE 
    user_id = 123                          -- filter by user (multi-tenant)
    AND category IN ('technical', 'docs')  -- filter by category
ORDER BY embedding <=> query_vector        -- order by cosine distance (ascending)
LIMIT 5;
```

```python
# Python pgvector usage
from pgvector.psycopg import register_vector
import psycopg
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

async def search(query: str, user_id: int, k: int = 5) -> list[dict]:
    query_vec = model.encode(query).tolist()
    
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await register_vector(conn)
        
        rows = await conn.execute("""
            SELECT content, 1 - (embedding <=> %s::vector) as similarity, source_url
            FROM documents
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vec, user_id, query_vec, k))
        
        return [{"content": r[0], "similarity": float(r[1]), "source": r[2]}
                async for r in rows]
```

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §4 Multi-Agent Patterns](guide/04_multi_agent.md) | [§6 Production Checklist →](guide/06_production_checklist.md)
