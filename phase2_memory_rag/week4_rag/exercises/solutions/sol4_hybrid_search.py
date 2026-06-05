"""
SOLUTION — Exercise 4: Hybrid Search — BM25 + Vector with Score Fusion
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import math
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

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


def bm25_search(query: str, corpus: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[rank-bm25 not installed] pip install rank-bm25")
        # Fallback: simple TF scoring
        words = set(query.lower().split())
        scores = [sum(1 for w in doc.lower().split() if w in words) for doc in corpus]
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(corpus[i], float(s)) for i, s in indexed[:top_k]]

    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(corpus[i], float(scores[i])) for i, _ in indexed[:top_k]]


def embed(texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts).tolist()
    except ImportError:
        print("[sentence-transformers not installed] Using TF fallback embeddings")
        vocab: dict[str, int] = {}
        for t in texts:
            for w in t.lower().split():
                if w not in vocab:
                    vocab[w] = len(vocab)
        vecs = []
        for t in texts:
            v = [0.0] * len(vocab)
            for w in t.lower().split():
                if w in vocab:
                    v[vocab[w]] += 1.0
            vecs.append(v)
        return vecs


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-8)


def vector_search(query: str, corpus: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    embeddings = embed([query] + corpus)
    q_emb = embeddings[0]
    doc_embs = embeddings[1:]
    scores = [cosine_similarity(q_emb, d) for d in doc_embs]
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(corpus[i], float(scores[i])) for i, _ in indexed[:top_k]]


def normalize(scores: list[tuple[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    vals = [s for _, s in scores]
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1.0
    return {doc: (score - mn) / rng for doc, score in scores}


def hybrid_search(query: str, corpus: list[str], top_k: int = 3,
                  bm25_weight: float = 0.4, vec_weight: float = 0.6) -> list[str]:
    bm25_results = bm25_search(query, corpus, top_k=len(corpus))
    vec_results = vector_search(query, corpus, top_k=len(corpus))
    bm25_norm = normalize(bm25_results)
    vec_norm = normalize(vec_results)
    all_docs = set(bm25_norm) | set(vec_norm)
    combined = {
        doc: bm25_weight * bm25_norm.get(doc, 0.0) + vec_weight * vec_norm.get(doc, 0.0)
        for doc in all_docs
    }
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def rag_answer(question: str, strategy: str = "hybrid") -> str:
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
