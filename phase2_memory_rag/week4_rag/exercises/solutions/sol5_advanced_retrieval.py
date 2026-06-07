"""
SOLUTION — Exercise 5: Advanced RAG Retrieval (HyDE + Multi-Query + Reranking)

Key concepts demonstrated:
- basic_retrieve: standard bi-encoder cosine similarity search (baseline)
- hyde_retrieve: embed a HYPOTHETICAL ANSWER instead of the raw question
    → bridges the gap between question vocabulary and document vocabulary
    → typical gain: 5-15% recall improvement
- multi_query_retrieve: generate 3 phrasings, retrieve for each, deduplicate
    → catches vocabulary-mismatch cases missed by a single query
    → typical gain: 10-20% recall improvement
- rerank: cross-encoder sees (query, document) TOGETHER → much more accurate
    → typical gain: 10-30% precision improvement
- advanced_retrieve: combines all three for production-grade retrieval

pip install sentence-transformers chromadb
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dotenv import load_dotenv
load_dotenv()

import json
import re
import time
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from llm import chat, get_text

# Bi-encoder: encode query and document SEPARATELY, compare embeddings.  Fast.
# Cross-encoder: encode (query, document) TOGETHER. Slow but much more accurate.
BI_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

CORPUS = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "FastAPI is a modern Python framework for building APIs with automatic OpenAPI docs.",
    "Kubernetes (K8s) orchestrates containerized workloads across clusters of nodes.",
    "Redis is an in-memory data structure store used as a cache, broker, and database.",
    "'Attention Is All You Need' by Vaswani et al. introduced the Transformer in 2017.",
    "Docker packages apps and dependencies into portable, isolated containers.",
    "PostgreSQL is an open-source relational database with 35+ years of development.",
    "LangGraph builds stateful AI agent workflows as directed graphs with typed state.",
    "Sentence transformers produce dense vectors for semantic similarity and retrieval.",
    "RAG (Retrieval-Augmented Generation) grounds LLM answers in retrieved documents.",
    "FAISS enables efficient approximate nearest-neighbor search on dense vector sets.",
    "Prometheus scrapes metrics from HTTP endpoints and stores time-series data.",
    "Celery processes async tasks across a distributed pool of Python workers via Redis.",
    "BM25 ranks documents by term frequency and inverse document frequency (TF-IDF).",
    "OpenTelemetry provides distributed tracing and metrics for cloud applications.",
]


def build_collection(docs: list[str], name: str = "adv_rag_sol") -> chromadb.Collection:
    client = chromadb.Client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    vecs = BI_ENCODER.encode(docs, normalize_embeddings=True).tolist()
    coll.add(documents=docs, embeddings=vecs, ids=[f"d{i}" for i in range(len(docs))])
    return coll


# ─── Technique 1: Basic (Baseline) ───────────────────────────────────────────

def basic_retrieve(question: str, coll: chromadb.Collection, k: int = 3) -> list[str]:
    """
    Standard bi-encoder retrieval.
    Fast (O(log n) with HNSW), but suffers from vocabulary mismatch:
    'Who invented Python?' has low overlap with 'Python was created by…'
    """
    q_vec = BI_ENCODER.encode(question, normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=[q_vec], n_results=k)
    return res["documents"][0]


# ─── Technique 2: HyDE ───────────────────────────────────────────────────────

def hyde_retrieve(
    question: str, coll: chromadb.Collection, k: int = 3
) -> tuple[list[str], str]:
    """
    HyDE: generate a hypothetical answer, embed THAT, use for retrieval.

    Why it works: the hypothetical answer uses the same vocabulary and style
    as the real document, making similarity search far more effective.
    The answer doesn't need to be factually correct — style matters.

    Cost: 1 extra LLM call per query.
    """
    hypothetical = get_text(chat([{
        "role": "user",
        "content": (
            "Write a short, factual paragraph that directly answers this question, "
            "as if copied from a technical encyclopedia. Use precise terminology.\n\n"
            f"Question: {question}\n\nAnswer (2-3 sentences):"
        ),
    }]))
    hyp_vec = BI_ENCODER.encode(hypothetical, normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=[hyp_vec], n_results=k)
    return res["documents"][0], hypothetical


# ─── Technique 3: Multi-Query ─────────────────────────────────────────────────

def multi_query_retrieve(
    question: str, coll: chromadb.Collection, k: int = 3
) -> tuple[list[str], list[str]]:
    """
    Generate 3 phrasings of the same question, retrieve for each, deduplicate.

    Why it works: different phrasings hit different documents because each
    phrasing's embedding points to a slightly different region of the vector space.
    Combining results gives broader recall than any single query.

    Cost: 1 LLM call + 4 vector searches.
    """
    raw = get_text(chat([{
        "role": "user",
        "content": (
            "Generate 3 different search queries to retrieve documents for this question.\n"
            "Rephrase using different words and angles.\n"
            'Output ONLY valid JSON: ["q1", "q2", "q3"]\n\n'
            f"Original: {question}"
        ),
    }]))
    clean = re.sub(r"```json?\s*|\s*```", "", raw).strip()
    try:
        variations = json.loads(clean)[:3]
    except json.JSONDecodeError:
        variations = []

    seen: set[str] = set()
    docs: list[str] = []
    queries = [question] + variations

    for q in queries:
        q_vec = BI_ENCODER.encode(q, normalize_embeddings=True).tolist()
        for doc in coll.query(query_embeddings=[q_vec], n_results=k)["documents"][0]:
            if doc not in seen:
                seen.add(doc)
                docs.append(doc)

    return docs, queries


# ─── Technique 4: Cross-Encoder Reranking ─────────────────────────────────────

def rerank(
    question: str, candidates: list[str], top_k: int = 3
) -> list[tuple[float, str]]:
    """
    Cross-encoder reranking: the most impactful single improvement.

    The cross-encoder attends to BOTH query and document simultaneously,
    enabling full interaction between their tokens. This is far more accurate
    than comparing separate embeddings, at the cost of O(n) inference.

    Never use cross-encoder for first-pass retrieval over large corpora.
    Always: bi-encode to get top 20, then cross-encode to rerank.
    """
    if not candidates:
        return []
    pairs = [(question, doc) for doc in candidates]
    scores = CROSS_ENCODER.predict(pairs)
    ranked = sorted(zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True)
    return ranked[:top_k]


# ─── Full Advanced Pipeline ────────────────────────────────────────────────────

def advanced_retrieve(
    question: str,
    coll: chromadb.Collection,
    top_k: int = 3,
) -> dict:
    """
    Production pipeline: Multi-Query + HyDE → pool → Cross-Encoder Rerank.

    Step 1: multi-query for broad recall
    Step 2: HyDE for vocabulary-mismatch cases
    Step 3: rerank the combined candidate pool for precision
    """
    pool: set[str] = set()

    mq_docs, queries = multi_query_retrieve(question, coll, k=5)
    pool.update(mq_docs)

    hyde_docs, hyp = hyde_retrieve(question, coll, k=5)
    pool.update(hyde_docs)

    ranked = rerank(question, list(pool), top_k=top_k)
    return {
        "documents": [doc for _, doc in ranked],
        "scores": [round(float(s), 4) for s, _ in ranked],
        "total_candidates": len(pool),
        "queries_used": queries,
        "hypothetical": hyp,
    }


# ─── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(question: str, expected_fragment: str, coll: chromadb.Collection):
    def hit(docs: list[str]) -> bool:
        return any(expected_fragment.lower() in d.lower() for d in docs)

    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"Expected fragment: '{expected_fragment}'")

    t0 = time.time()
    basic = basic_retrieve(question, coll)
    ms_basic = (time.time() - t0) * 1000
    print(f"\n[1] Basic        {ms_basic:5.0f}ms  {'✅' if hit(basic) else '❌'}")
    for d in basic[:2]:
        print(f"    · {d[:80]}…")

    t0 = time.time()
    hyde_docs, hyp = hyde_retrieve(question, coll)
    ms_hyde = (time.time() - t0) * 1000
    print(f"\n[2] HyDE         {ms_hyde:5.0f}ms  {'✅' if hit(hyde_docs) else '❌'}")
    print(f"    Hypothetical: {hyp[:80]}…")

    t0 = time.time()
    adv = advanced_retrieve(question, coll)
    ms_adv = (time.time() - t0) * 1000
    adv_docs = adv["documents"]
    print(f"\n[3] Advanced     {ms_adv:5.0f}ms  {'✅' if hit(adv_docs) else '❌'}")
    print(f"    Candidates pooled: {adv['total_candidates']}")
    for doc, score in zip(adv_docs[:3], adv["scores"][:3]):
        print(f"    · [{score:+.3f}] {doc[:75]}…")

    return {
        "basic": hit(basic),
        "hyde": hit(hyde_docs),
        "advanced": hit(adv_docs),
    }


if __name__ == "__main__":
    print("Building index…")
    coll = build_collection(CORPUS)
    print(f"Indexed {len(CORPUS)} documents\n")

    test_cases = [
        ("Who invented Python and when?",                           "Guido van Rossum"),
        ("Which algorithm is used for keyword-based document ranking?", "BM25"),
        ("What seminal paper introduced the attention mechanism?",  "Attention Is All You Need"),
        ("How can I run distributed background jobs in Python?",    "Celery"),
    ]

    all_results = []
    for q, expected in test_cases:
        r = benchmark(q, expected, coll)
        all_results.append(r)

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    for method in ("basic", "hyde", "advanced"):
        hits = sum(r[method] for r in all_results)
        print(f"  {method:<12}: {hits}/{len(all_results)} hits")

    print("""
Take-aways:
  • HyDE helps when the question phrasing differs from document phrasing
  • Multi-query expands recall by covering synonyms and reformulations
  • Cross-encoder reranking is the highest-ROI single improvement
  • Combine all three for production RAG where accuracy matters
""")
