"""
Exercise 4: Hybrid Search — BM25 + Vector with Score Fusion
Goal: Combine keyword (BM25) and semantic (vector) search for better retrieval.

Tasks:
  1. Complete bm25_search() — tokenize corpus, score with BM25Okapi, return top-k.
  2. Complete vector_search() — embed query + corpus, cosine similarity, return top-k.
  3. Complete hybrid_search() — fuse scores with: 0.4*bm25 + 0.6*vector (both normalized).
  4. Complete rag_answer() — retrieve via hybrid, pass chunks to LLM.
  5. Run compare() to measure: which strategy answers each question best?

Install: pip install rank-bm25 sentence-transformers

Expected output:
  Q: What is retrieval augmented generation?
  BM25 top: "Retrieval augmented generation combines..."
  Vector top: "RAG pipelines retrieve relevant chunks..."
  Hybrid top: "Retrieval augmented generation combines..."
  Answer: RAG combines a retrieval system with a language model...
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import math
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

# ── Sample Corpus ──────────────────────────────────────────────────────────────

CORPUS = [
    "Retrieval augmented generation (RAG) combines a retrieval system with a language model to answer questions from specific documents.",
    "Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbor search.",
    "BM25 is a probabilistic ranking function used in information retrieval based on term frequency and inverse document frequency.",
    "Chunking strategies include fixed-size, sentence-based, and semantic chunking. Each has tradeoffs for retrieval quality.",
    "Hybrid search fuses keyword search scores with vector similarity scores to get the best of both approaches.",
    "Embeddings are dense vector representations of text that capture semantic meaning rather than just keywords.",
    "Cosine similarity measures the angle between two vectors — 1.0 means identical direction, 0 means orthogonal.",
    "Reranking applies a second model to re-score the top-k retrieved chunks for better precision.",
    "Context window limits mean you can only feed a few chunks to the LLM — retrieval precision is critical.",
    "FAISS and Chroma are popular vector stores. FAISS is optimised for large-scale search; Chroma is developer-friendly.",
]

QUESTIONS = [
    "What is retrieval augmented generation?",
    "How does BM25 work?",
    "Why use hybrid search instead of pure vector search?",
    "What is cosine similarity?",
]


# ── Search Implementations ─────────────────────────────────────────────────────

def bm25_search(query: str, corpus: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """Return top-k (doc, score) pairs using BM25."""
    # TODO: from rank_bm25 import BM25Okapi
    # TODO: tokenize = [doc.lower().split() for doc in corpus]
    # TODO: bm25 = BM25Okapi(tokenize)
    # TODO: scores = bm25.get_scores(query.lower().split())
    # TODO: top indices = argsort descending, return top-k (corpus[i], scores[i])
    raise NotImplementedError


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using sentence-transformers."""
    # TODO: from sentence_transformers import SentenceTransformer
    # TODO: model = SentenceTransformer("all-MiniLM-L6-v2")
    # TODO: return model.encode(texts).tolist()
    raise NotImplementedError


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-8)


def vector_search(query: str, corpus: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    """Return top-k (doc, similarity) pairs using cosine similarity of embeddings."""
    # TODO: embed query + corpus
    # TODO: compute cosine_similarity(query_emb, doc_emb) for each doc
    # TODO: sort descending, return top-k
    raise NotImplementedError


def normalize(scores: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return {}
    vals = [s for _, s in scores]
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1.0
    return {doc: (score - mn) / rng for doc, score in scores}


def hybrid_search(query: str, corpus: list[str], top_k: int = 3,
                  bm25_weight: float = 0.4, vec_weight: float = 0.6) -> list[str]:
    """Fuse BM25 and vector scores. Return top-k doc strings."""
    # TODO: get bm25 results for all docs (top_k=len(corpus))
    # TODO: get vector results for all docs
    # TODO: normalize both score dicts
    # TODO: combined[doc] = bm25_weight * bm25_norm.get(doc, 0) + vec_weight * vec_norm.get(doc, 0)
    # TODO: sort by combined score descending, return top_k doc strings
    raise NotImplementedError


# ── RAG Answer ─────────────────────────────────────────────────────────────────

def rag_answer(question: str, strategy: str = "hybrid") -> str:
    """Retrieve chunks and answer the question."""
    if strategy == "bm25":
        chunks = [doc for doc, _ in bm25_search(question, CORPUS, top_k=3)]
    elif strategy == "vector":
        chunks = [doc for doc, _ in vector_search(question, CORPUS, top_k=3)]
    else:
        chunks = hybrid_search(question, CORPUS, top_k=3)

    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    response = chat(
        [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        system="Answer using only the provided context. Be concise.",
        max_tokens=256,
    )
    return get_text(response)


# ── Comparison ─────────────────────────────────────────────────────────────────

def compare():
    for q in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        bm25_top = bm25_search(q, CORPUS, top_k=1)
        vec_top = vector_search(q, CORPUS, top_k=1)
        hybrid_top = hybrid_search(q, CORPUS, top_k=1)
        print(f"  BM25   top: {bm25_top[0][0][:70]}...")
        print(f"  Vector top: {vec_top[0][0][:70]}...")
        print(f"  Hybrid top: {hybrid_top[0][:70]}...")
        answer = rag_answer(q, strategy="hybrid")
        print(f"  Answer: {answer[:120]}")


if __name__ == "__main__":
    compare()
