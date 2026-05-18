# Week 4 — Memory & RAG

## Topics
1. Memory types: in-context, external, episodic, semantic
2. Vector DBs: Chroma, Qdrant — embeddings fundamentals
3. RAG pipeline: chunk → embed → store → retrieve → augment
4. Hybrid search: BM25 + vector, rerankers

## Key Concepts

### RAG Pipeline
```
Documents → Chunking → Embedding → Vector Store
                                        ↓
User Query → Embed Query → Similarity Search → Top-K Chunks
                                                    ↓
                                          Augment Prompt → LLM → Answer
```

### Chunking Strategies (test all three)
- Fixed size: `chunk_size=512, overlap=50` — fast, dumb
- Sentence-based: preserves meaning, variable length
- Semantic: group related sentences — best quality, slowest

### Memory Types
| Type | Storage | Use Case |
|---|---|---|
| In-context | Message history | Short sessions |
| External | SQLite / Redis | Long-term user prefs |
| Episodic | Vector DB | "What did we discuss last week?" |
| Semantic | Vector DB + summaries | Knowledge base |

### Hybrid Search
Vector search alone misses exact keyword matches.
BM25 alone misses semantic similarity.
Combine both with a weighted score, then rerank with Cohere.

## Exercises
- `ex1_rag_basic.py` — ingest PDF, query it
- `ex2_chunking_compare.py` — compare 3 chunking strategies
- `ex3_persistent_memory.py` — agent remembers user across sessions
- `ex4_hybrid_search.py` — BM25 + vector with reranker

## Checklist
- [ ] Built basic RAG pipeline over a PDF
- [ ] Compared fixed vs sentence vs semantic chunking
- [ ] Implemented persistent memory in SQLite
- [ ] Added hybrid search and measured improvement
