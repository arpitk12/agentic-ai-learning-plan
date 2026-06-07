[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §2 Framework Selection](guide/02_framework_selection.md) | [§4 Multi-Agent Patterns →](guide/04_multi_agent.md)

---

## 3. RAG Architecture Deep Dive

RAG (Retrieval-Augmented Generation) is the single most important pattern for making LLMs useful in production. This section covers the full pipeline from raw documents to accurate answers, with all the engineering details you need to build it right.

### 3.1 Why RAG? The Problem It Solves

| Problem | Without RAG | With RAG |
|---------|------------|---------|
| LLM knowledge cutoff | Can't answer about events after training | Retrieves current documents |
| Your company's private data | LLM doesn't know it | Retrieve from your knowledge base |
| Context window limits | Can't fit 10,000 pages in context | Retrieve only relevant 3-5 chunks |
| Hallucination | Makes up facts | Answers grounded in retrieved documents |
| Auditability | Can't explain why it said X | Shows exactly which documents were used |

**RAG vs Fine-tuning**:
- Fine-tuning: Trains new knowledge into model weights. Requires GPU, data, time. Knowledge is stale once trained.
- RAG: Retrieves knowledge at query time. Zero GPU required. Knowledge updates instantly. **Choose RAG for 95% of use cases.**

### 3.2 The Complete RAG Pipeline

```
═══════════════════ INGESTION PIPELINE (offline, run once) ══════════════════
Raw Files (PDF, DOCX, HTML, MD)
    │
    ▼
[Document Loader]      ← LangChain loaders, custom parsers
    │
    ▼
[Preprocessing]        ← strip HTML, fix encoding, clean whitespace
    │
    ▼
[Chunking]             ← split into 500-1000 token segments with overlap
    │
    ▼
[Embedding]            ← sentence-transformers / OpenAI → float[] vectors
    │
    ▼
[Vector DB Storage]    ← ChromaDB (dev) / Qdrant (prod) + metadata

═══════════════════ QUERY PIPELINE (online, every request) ══════════════════
User Question
    │
    ▼
[Query Analysis]       ← detect intent, extract entities, expand query
    │
    ▼
[Retrieval]            ← embed query → vector search → top-K chunks
    │
    ▼
[Reranking]            ← cross-encoder reranks top-K for precision
    │
    ▼
[Context Assembly]     ← format retrieved chunks into LLM prompt
    │
    ▼
[LLM Generation]       ← generate grounded answer with citations
    │
    ▼
Answer + Source Citations
```

### 3.3 Document Loading & Preprocessing

```python
from pathlib import Path
import re, hashlib
from typing import Generator

def load_pdf(path: str) -> str:
    """Load PDF using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)

def load_web_page(url: str) -> str:
    """Load and clean web page content."""
    import httpx
    from bs4 import BeautifulSoup
    
    response = httpx.get(url, timeout=10, follow_redirects=True)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    
    text = soup.get_text(separator="\n")
    # Clean whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse multiple newlines
    text = re.sub(r" {2,}", " ", text)       # collapse multiple spaces
    return text.strip()

def preprocess_document(text: str) -> str:
    """Clean text before chunking."""
    # Fix common encoding issues
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    # Remove null bytes
    text = text.replace("\x00", "")
    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)  # windows line endings
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()

def document_hash(content: str) -> str:
    """Unique ID for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### 3.4 Chunking Strategies — Deep Comparison

**Critical insight**: Chunk size is the most important RAG hyperparameter. Too small = not enough context. Too large = noisy retrieval. Always tune with your specific data.

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

# Strategy 1: Fixed-size chunking (simple, baseline)
def chunk_fixed(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split every N characters regardless of content boundaries."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size]
        if len(chunk.strip()) > 50:  # skip tiny chunks
            chunks.append(chunk.strip())
    return chunks

# Strategy 2: Recursive character splitting (RECOMMENDED DEFAULT)
def chunk_recursive(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Try splitting on: paragraph → sentence → word → character
    Best balance of quality and simplicity.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)

# Strategy 3: Token-based chunking (for LLM context budget control)
def chunk_by_tokens(text: str, max_tokens: int = 512) -> list[str]:
    """Split by actual token count — precise LLM context budgeting."""
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",  # GPT-4 tokenizer, also good for other models
        chunk_size=max_tokens,
        chunk_overlap=max_tokens // 10  # 10% overlap
    )
    return splitter.split_text(text)

# Strategy 4: Semantic chunking (BEST QUALITY, slowest)
def chunk_semantic(text: str) -> list[str]:
    """
    Group sentences by semantic similarity.
    Split when topic changes significantly.
    Requires embedding every sentence — 10x slower than recursive.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from sentence_transformers import SentenceTransformer
    
    # Use a lightweight model for chunking to keep it fast
    class LocalEmbeddings:
        def __init__(self):
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed_documents(self, texts):
            return self.model.encode(texts).tolist()
        def embed_query(self, text):
            return self.model.encode(text).tolist()
    
    splitter = SemanticChunker(
        LocalEmbeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,  # split on 95th percentile semantic distance
    )
    return splitter.split_text(text)

# Strategy 5: Parent-Child chunking (BEST for RAG)
def build_parent_child_index(text: str, parent_size: int = 2000, child_size: int = 400):
    """
    Small child chunks for precise retrieval.
    Large parent chunks for richer LLM context.
    
    Query → retrieve child chunk → return parent chunk to LLM
    """
    parents = chunk_recursive(text, chunk_size=parent_size, overlap=100)
    
    child_to_parent = {}
    all_children = []
    
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, chunk_size=child_size, overlap=40)
        for c_idx, child in enumerate(children):
            child_id = f"p{p_idx}_c{c_idx}"
            all_children.append({"id": child_id, "text": child, "parent_id": p_idx})
            child_to_parent[child_id] = parent
    
    return all_children, parents, child_to_parent

# Chunking decision guide
CHUNK_STRATEGY_GUIDE = {
    "FAQ/short docs": ("fixed", {"size": 300, "overlap": 30}),
    "Long articles/books": ("recursive", {"chunk_size": 1000, "overlap": 200}),
    "Code documentation": ("recursive", {"chunk_size": 800, "overlap": 100}),
    "Legal/technical PDFs": ("semantic", {}),
    "Production RAG": ("parent_child", {"parent_size": 2000, "child_size": 400}),
    "Token budget critical": ("tokens", {"max_tokens": 512}),
}
```

### 3.5 Embedding — Converting Text to Searchable Vectors

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

# Model selection guide:
EMBEDDING_MODELS = {
    "development": {
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "speed": "~1000 docs/sec on CPU",
        "quality": "Good for most use cases",
        "cost": "Free (local)",
    },
    "production_balanced": {
        "model": "all-mpnet-base-v2",
        "dimensions": 768,
        "speed": "~200 docs/sec on CPU",
        "quality": "Better semantic understanding",
        "cost": "Free (local)",
    },
    "production_best": {
        "model": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "speed": "~100 docs/sec on CPU",
        "quality": "State-of-the-art (2024)",
        "cost": "Free (local)",
    },
    "api_best": {
        "model": "text-embedding-3-large",
        "dimensions": 3072,
        "speed": "API speed",
        "quality": "Best available",
        "cost": "$0.13/1M tokens",
    },
    "multilingual": {
        "model": "paraphrase-multilingual-mpnet-base-v2",
        "dimensions": 768,
        "speed": "~150 docs/sec on CPU",
        "quality": "Good for 50+ languages",
        "cost": "Free (local)",
    },
}

class EmbeddingService:
    """Production embedding service with batching and caching."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a large list of texts efficiently in batches."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # normalize for cosine similarity
            convert_to_numpy=True,
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query for retrieval."""
        return self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    
    @lru_cache(maxsize=1000)  # cache frequent queries
    def embed_cached(self, text: str) -> tuple:
        """Cached embedding for repeated queries."""
        return tuple(self.embed_query(text).tolist())
```

### 3.6 Full Ingestion Pipeline

```python
import chromadb
from pathlib import Path
from typing import Iterator
import json, time

class RAGIngestionPipeline:
    """Complete document ingestion pipeline."""
    
    def __init__(self, collection_name: str, db_path: str = "./chroma_db"):
        self.embedder = EmbeddingService("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.stats = {"total_docs": 0, "total_chunks": 0, "failed": 0}
    
    def ingest_file(self, file_path: str, metadata: dict = None) -> int:
        """Ingest a single file. Returns number of chunks added."""
        path = Path(file_path)
        
        # Load based on file type
        if path.suffix == ".pdf":
            text = load_pdf(file_path)
        elif path.suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix == ".html":
            text = load_web_page(f"file://{path.absolute()}")
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        return self.ingest_text(text, metadata or {"source": str(path), "filename": path.name})
    
    def ingest_text(self, text: str, metadata: dict) -> int:
        """Ingest raw text."""
        text = preprocess_document(text)
        chunks = chunk_recursive(text, chunk_size=1000, overlap=200)
        
        # Filter tiny chunks
        chunks = [c for c in chunks if len(c.strip()) > 50]
        
        if not chunks:
            return 0
        
        # Embed all chunks
        embeddings = self.embedder.embed_batch(chunks)
        
        # Create IDs and metadata
        source_hash = document_hash(text)
        ids = [f"{source_hash}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_length": len(chunk),
            "ingested_at": time.time(),
        } for i, chunk in enumerate(chunks)]
        
        # Upsert (handles duplicates)
        self.collection.upsert(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
        
        self.stats["total_chunks"] += len(chunks)
        self.stats["total_docs"] += 1
        return len(chunks)
    
    def ingest_directory(self, dir_path: str, glob: str = "**/*.{pdf,md,txt}") -> dict:
        """Ingest all matching files in a directory."""
        path = Path(dir_path)
        files = list(path.glob(glob.replace("{pdf,md,txt}", "**")))
        # Simplified glob
        files = list(path.rglob("*.pdf")) + list(path.rglob("*.md")) + list(path.rglob("*.txt"))
        
        print(f"Found {len(files)} files to ingest")
        for f in files:
            try:
                n = self.ingest_file(str(f))
                print(f"  ✓ {f.name}: {n} chunks")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")
                self.stats["failed"] += 1
        
        return self.stats
```

### 3.7 Advanced Retrieval — HyDE, Multi-Query, Reranking

```python
from llm import chat, get_text

# Technique 1: HyDE (Hypothetical Document Embeddings)
# Problem: Query "when was Python released?" has poor overlap with
#          document "Python 1.0 was released in 1994."
# Solution: Generate a hypothetical answer, embed THAT for better retrieval.
def hyde_retrieve(question: str, collection, k: int = 5) -> list[str]:
    """
    Generate a hypothetical answer, embed it, use for retrieval.
    Usually improves retrieval quality by 5-15%.
    """
    hypothetical = get_text(chat([{
        "role": "user",
        "content": f"""Write a short, factual paragraph that would directly answer this question:
{question}

Write it as if it were from a textbook or documentation page. Be concise and specific."""
    }]))
    
    results = collection.query(query_texts=[hypothetical], n_results=k)
    return results["documents"][0]

# Technique 2: Multi-Query Retrieval
# Problem: One query may miss relevant docs that use different terminology
# Solution: Generate multiple query variations, retrieve for all, deduplicate
def multi_query_retrieve(question: str, collection, k: int = 5) -> list[str]:
    """
    Generate 3 query variations, retrieve for each, return unique top-K.
    Addresses vocabulary mismatch and improves recall.
    """
    variations_raw = get_text(chat([{
        "role": "user",
        "content": f"""Generate 3 different search queries to find information for this question:
{question}

Each query should use different words/angles. Output as JSON array: ["query1", "query2", "query3"]"""
    }]))
    
    import json, re
    clean = re.sub(r"```json?\s*|\s*```", "", variations_raw).strip()
    queries = json.loads(clean)
    queries.insert(0, question)  # include original
    
    # Retrieve for each query
    seen = set()
    all_docs = []
    for q in queries[:4]:  # limit to 4 queries to control cost
        results = collection.query(query_texts=[q], n_results=k)
        for doc in results["documents"][0]:
            if doc not in seen:
                seen.add(doc)
                all_docs.append(doc)
    
    return all_docs[:k * 2]  # return more candidates for reranking

# Technique 3: Reranking with Cross-Encoder
# Problem: Bi-encoder similarity is approximate — top-5 may not be best 5
# Solution: Use slower but more accurate cross-encoder to rerank candidates
def rerank(query: str, candidates: list[str], top_k: int = 3) -> list[str]:
    """
    Rerank retrieved candidates using a cross-encoder.
    Cross-encoder sees (query, document) together → much more accurate.
    Typical improvement: 10-30% over bi-encoder alone.
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    pairs = [(query, doc) for doc in candidates]
    scores = model.predict(pairs)
    
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# Complete retrieval pipeline
def retrieve_with_full_pipeline(
    question: str,
    collection,
    use_hyde: bool = True,
    use_multi_query: bool = True,
    use_rerank: bool = True,
    final_k: int = 3,
) -> list[str]:
    """Production-grade retrieval: HyDE + Multi-Query + Reranking."""
    
    # Step 1: Get candidates via multiple strategies
    candidates = []
    
    if use_multi_query:
        candidates.extend(multi_query_retrieve(question, collection, k=5))
    else:
        results = collection.query(query_texts=[question], n_results=5)
        candidates.extend(results["documents"][0])
    
    if use_hyde:
        hyde_results = hyde_retrieve(question, collection, k=3)
        for doc in hyde_results:
            if doc not in candidates:
                candidates.append(doc)
    
    if not candidates:
        return []
    
    # Step 2: Rerank candidates
    if use_rerank and len(candidates) > final_k:
        return rerank(question, candidates, top_k=final_k)
    
    return candidates[:final_k]

# The RAG answer generation function
def rag_answer(question: str, collection, k: int = 3) -> dict:
    """Generate a grounded answer with source attribution."""
    
    # Retrieve relevant chunks
    chunks = retrieve_with_full_pipeline(question, collection, final_k=k)
    
    if not chunks:
        return {"answer": "I couldn't find relevant information in the knowledge base.", "sources": []}
    
    # Format context
    context = "\n\n---\n\n".join([f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
    
    # Generate grounded answer
    answer = get_text(chat(
        messages=[{
            "role": "user",
            "content": f"""Answer the question using ONLY the provided sources below.

RULES:
1. If the answer isn't in the sources, say exactly: "I don't have that information in my knowledge base."
2. Always cite your sources using [Source N] notation
3. Be accurate and specific — don't add information not in the sources

Sources:
{context}

Question: {question}

Answer:"""
        }],
        system="You are a precise assistant. Answer only from provided sources. Always cite [Source N]."
    ))
    
    return {
        "answer": answer,
        "sources": chunks,
        "chunk_count": len(chunks),
    }
```

### 3.8 RAG Evaluation — Measuring Quality

```python
# Install: pip install ragas datasets
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

def evaluate_rag_pipeline(test_questions: list[dict], collection) -> dict:
    """
    Evaluate RAG quality using RAGAS metrics.
    
    test_questions format:
    [{"question": "...", "ground_truth": "..."}, ...]
    
    RAGAS Metrics:
    - faithfulness: Is the answer supported by the retrieved context? (0-1)
    - answer_relevancy: Is the answer relevant to the question? (0-1)
    - context_precision: Are the retrieved chunks relevant? (0-1)
    """
    from datasets import Dataset
    
    data = []
    for item in test_questions:
        result = rag_answer(item["question"], collection)
        data.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": result["sources"],
            "ground_truth": item["ground_truth"],
        })
    
    dataset = Dataset.from_list(data)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    
    return {
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_precision": scores["context_precision"],
    }

# RAG Quality Targets
RAG_QUALITY_BENCHMARKS = {
    "faithfulness": {"minimum": 0.80, "good": 0.90, "excellent": 0.95},
    "answer_relevancy": {"minimum": 0.75, "good": 0.85, "excellent": 0.93},
    "context_precision": {"minimum": 0.70, "good": 0.80, "excellent": 0.90},
}
```

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §2 Framework Selection](guide/02_framework_selection.md) | [§4 Multi-Agent Patterns →](guide/04_multi_agent.md)
