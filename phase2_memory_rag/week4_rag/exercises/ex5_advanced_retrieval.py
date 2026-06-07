"""
Exercise 5: Advanced RAG Retrieval Techniques
Guide Section: §3.7 — HyDE, Multi-Query Retrieval, Cross-Encoder Reranking

Goal: Implement three retrieval improvements and compare their quality.

Why these matter:
- Basic vector search has ~60-70% recall — it misses relevant documents
- HyDE (+5-15%): embed a hypothetical answer instead of the raw question
- Multi-query (+10-20%): generate 3 phrasings, retrieve for all, deduplicate
- Reranking (+10-30%): cross-encoder sees (query+doc) together → much better precision

When to use:
- HyDE: when query vocabulary differs from document vocabulary
- Multi-query: when question can be asked many ways
- Reranking: always, as the final step before sending to LLM

pip install sentence-transformers chromadb
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dotenv import load_dotenv
load_dotenv()

import chromadb
import json, re, time
from sentence_transformers import SentenceTransformer, CrossEncoder
from llm import chat, get_text

# ─── Models ───────────────────────────────────────────────────────────────────
# Bi-encoder: fast, encodes query and document SEPARATELY → compare embeddings
# Cross-encoder: slow, sees (query, document) TOGETHER → far more accurate
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ─── Sample Corpus ────────────────────────────────────────────────────────────
SAMPLE_DOCS = [
    "Python was created by Guido van Rossum and first released in 1991. It emphasizes readability.",
    "FastAPI is a modern Python web framework for building APIs using type annotations.",
    "Kubernetes, also known as K8s, is used for orchestrating containerized applications at scale.",
    "Redis is an in-memory data structure store used as a database, cache, and message broker.",
    "The Transformer paper 'Attention Is All You Need' was published by Google Brain in 2017.",
    "Docker packages applications and their dependencies into portable, isolated containers.",
    "PostgreSQL is an open-source relational database with over 35 years of active development.",
    "LangGraph builds stateful AI agent workflows as directed graphs with typed state.",
    "Sentence transformers produce dense vector representations for semantic similarity tasks.",
    "RAG (Retrieval Augmented Generation) grounds LLM answers in retrieved documents.",
    "FAISS enables efficient approximate nearest-neighbor search on dense vector collections.",
    "Prometheus scrapes metrics from HTTP endpoints and stores time-series data.",
    "Celery processes asynchronous tasks across a distributed pool of Python workers.",
    "BM25 ranks documents by term frequency and inverse document frequency (keyword matching).",
    "OpenTelemetry provides distributed tracing, metrics, and logging for cloud applications.",
]


def build_collection(docs: list[str], name: str = "adv_rag") -> chromadb.Collection:
    """Build a ChromaDB collection from a list of texts."""
    client = chromadb.Client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    embeddings = bi_encoder.encode(docs, normalize_embeddings=True).tolist()
    coll.add(documents=docs, embeddings=embeddings, ids=[f"doc_{i}" for i in range(len(docs))])
    return coll


# ─── Technique 1: Basic Vector Search (Baseline) ──────────────────────────────

def basic_retrieve(question: str, coll: chromadb.Collection, k: int = 3) -> list[str]:
    """
    Standard bi-encoder retrieval.
    Fast: O(log n) with HNSW index.
    Weakness: vocabulary mismatch — if question uses different words than
    the document, similarity will be low even if meaning is the same.
    """
    q_vec = bi_encoder.encode(question, normalize_embeddings=True).tolist()
    results = coll.query(query_embeddings=[q_vec], n_results=k)
    return results["documents"][0]


# ─── Technique 2: HyDE — Hypothetical Document Embeddings ─────────────────────

def hyde_retrieve(question: str, coll: chromadb.Collection, k: int = 3) -> tuple[list[str], str]:
    """
    HyDE: Instead of embedding the question, embed a hypothetical answer.

    Why it works:
      - Question: "Who invented Python?" → short, interrogative → poor overlap
      - HyDE doc: "Python was invented by Guido van Rossum." → matches doc style
    
    The hypothetical answer doesn't need to be correct — its STYLE and VOCABULARY
    are what matters for retrieval. It acts as a "bridge" between the question
    embedding space and the document embedding space.

    Cost: 1 extra LLM call per query.
    Typical gain: 5-15% recall improvement.
    """
    hypothetical = get_text(chat([{
        "role": "user",
        "content": (
            f"Write a short, factual paragraph that would directly answer this question, "
            f"as if copied from a technical encyclopedia or documentation page. "
            f"Use precise terminology. Be specific.\n\n"
            f"Question: {question}\n\nAnswer (2-3 sentences):"
        )
    }]))

    # Embed the hypothetical answer, not the original question
    hyp_vec = bi_encoder.encode(hypothetical, normalize_embeddings=True).tolist()
    results = coll.query(query_embeddings=[hyp_vec], n_results=k)
    return results["documents"][0], hypothetical


# ─── Technique 3: Multi-Query Retrieval ───────────────────────────────────────

def multi_query_retrieve(
    question: str,
    coll: chromadb.Collection,
    k: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Multi-query: generate 3 phrasings of the same question, retrieve for each.

    Why it works:
      - Different phrasings hit different documents (vocabulary mismatch)
      - "Python creator" vs "who made Python" vs "Python history" → broader recall
      
    Results are deduplicated before returning.
    Cost: 1 LLM call (to generate queries) + 4 vector searches.
    Typical gain: 10-20% recall improvement.
    """
    raw = get_text(chat([{
        "role": "user",
        "content": (
            f"Generate 3 different search queries to find relevant documents for this question.\n"
            f"Each query should rephrase the question using different words or angles.\n"
            f"Output ONLY valid JSON array: [\"query1\", \"query2\", \"query3\"]\n\n"
            f"Original: {question}"
        )
    }]))

    clean = re.sub(r"```json?\s*|\s*```", "", raw).strip()
    try:
        variations = json.loads(clean)[:3]
    except json.JSONDecodeError:
        variations = []

    all_queries = [question] + variations  # include original
    seen: set[str] = set()
    all_docs: list[str] = []

    for q in all_queries:
        q_vec = bi_encoder.encode(q, normalize_embeddings=True).tolist()
        results = coll.query(query_embeddings=[q_vec], n_results=k)
        for doc in results["documents"][0]:
            if doc not in seen:
                seen.add(doc)
                all_docs.append(doc)

    return all_docs, all_queries  # return all candidates (more than k, for reranking)


# ─── Technique 4: Cross-Encoder Reranking ─────────────────────────────────────

def rerank(question: str, candidates: list[str], top_k: int = 3) -> list[tuple[float, str]]:
    """
    Cross-encoder reranking: the most powerful improvement.

    Bi-encoder limitation: embeds query and document SEPARATELY, then compares vectors.
    Cross-encoder: sees (query + document) TOGETHER in a single forward pass →
    full attention between query and document tokens → much more accurate relevance.

    Why it's used only for reranking (not first-pass retrieval):
    - Cross-encoder must see every (query, doc) pair — O(n) inference cost
    - Too slow to compare query vs all 1M docs (must bi-encode first)
    - Solution: bi-encode to get top 20 candidates, then cross-encode to rerank

    Typical improvement: 10-30% precision gain over bi-encoder alone.
    """
    if not candidates:
        return []
    pairs = [(question, doc) for doc in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True)
    return ranked[:top_k]


# ─── Full Advanced Pipeline ────────────────────────────────────────────────────

def advanced_retrieve(question: str, coll: chromadb.Collection, top_k: int = 3) -> dict:
    """
    Production retrieval: Multi-Query + HyDE → pool candidates → Cross-Encoder Rerank.
    
    1. Multi-query: broad recall across phrasings
    2. HyDE: catches vocabulary-mismatch cases
    3. Cross-encoder: precise reranking of combined candidates
    """
    candidates: set[str] = set()

    # Leg 1: Multi-query retrieval
    mq_docs, queries = multi_query_retrieve(question, coll, k=5)
    candidates.update(mq_docs)

    # Leg 2: HyDE retrieval
    hyde_docs, hypothetical = hyde_retrieve(question, coll, k=5)
    candidates.update(hyde_docs)

    candidate_list = list(candidates)

    # Final: cross-encoder reranking
    ranked = rerank(question, candidate_list, top_k=top_k)

    return {
        "documents": [doc for _, doc in ranked],
        "scores": [round(float(s), 4) for s, _ in ranked],
        "total_candidates": len(candidate_list),
        "queries_used": queries,
        "hypothetical_answer": hypothetical,
    }


# ─── Comparison Harness ───────────────────────────────────────────────────────

def compare_all(
    question: str,
    coll: chromadb.Collection,
    expected_fragment: str,
):
    """Run all four retrieval methods and compare hits, scores, and latency."""
    print(f"\n{'─'*60}")
    print(f"Query:    {question}")
    print(f"Expected: …{expected_fragment[:50]}…")
    print(f"{'─'*60}")

    def hit(docs: list[str]) -> bool:
        return any(expected_fragment.lower() in d.lower() for d in docs)

    # ── Basic ──
    t0 = time.time()
    basic_docs = basic_retrieve(question, coll, k=3)
    t_basic = (time.time() - t0) * 1000
    print(f"\n[1] Basic vector search  {t_basic:5.0f}ms  {'✅' if hit(basic_docs) else '❌'}")
    for d in basic_docs[:2]:
        print(f"    · {d[:80]}…")

    # ── HyDE ──
    t0 = time.time()
    hyde_docs, hyp = hyde_retrieve(question, coll, k=3)
    t_hyde = (time.time() - t0) * 1000
    print(f"\n[2] HyDE retrieval       {t_hyde:5.0f}ms  {'✅' if hit(hyde_docs) else '❌'}")
    print(f"    Hypothetical: {hyp[:80]}…")
    for d in hyde_docs[:2]:
        print(f"    · {d[:80]}…")

    # ── Advanced (Multi-Query + HyDE + Reranking) ──
    t0 = time.time()
    adv = advanced_retrieve(question, coll, top_k=3)
    t_adv = (time.time() - t0) * 1000
    adv_docs = adv["documents"]
    print(f"\n[3] Advanced pipeline    {t_adv:5.0f}ms  {'✅' if hit(adv_docs) else '❌'}")
    print(f"    Queries generated: {adv['queries_used'][1:]}")
    print(f"    Candidates before rerank: {adv['total_candidates']}")
    for doc, score in zip(adv_docs[:3], adv["scores"][:3]):
        print(f"    · [score={score:+.3f}] {doc[:75]}…")

    return {
        "basic": {"hit": hit(basic_docs), "ms": t_basic},
        "hyde": {"hit": hit(hyde_docs), "ms": t_hyde},
        "advanced": {"hit": hit(adv_docs), "ms": t_adv},
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building knowledge base…")
    coll = build_collection(SAMPLE_DOCS)
    print(f"Indexed {len(SAMPLE_DOCS)} documents\n")

    # Test cases: (question, fragment that should appear in top result)
    test_cases = [
        (
            "Who invented Python and when was it released?",
            "Guido van Rossum",
        ),
        (
            "Which algorithm do search engines traditionally use for keyword ranking?",
            "BM25",
        ),
        (
            "What was the seminal paper that introduced the attention mechanism?",
            "Attention Is All You Need",
        ),
        (
            "How can I run distributed background jobs in Python?",
            "Celery",
        ),
    ]

    results = []
    for question, expected in test_cases:
        r = compare_all(question, coll, expected)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Hits':>6} {'Avg ms':>8}")
    print(f"{'─'*42}")
    for method in ["basic", "hyde", "advanced"]:
        hits = sum(r[method]["hit"] for r in results)
        avg_ms = sum(r[method]["ms"] for r in results) / len(results)
        print(f"{method:<25} {hits:>4}/{len(results)}  {avg_ms:>7.0f}ms")

    print("""
Key takeaways:
  • Basic retrieval is fastest but misses vocabulary-mismatched queries
  • HyDE helps when the question phrasing differs from the document phrasing
  • Reranking improves precision — the #1 result is more likely to be correct
  • Advanced pipeline costs ~3-5s more per query but substantially improves recall

Production rule: use basic retrieval for latency-critical paths,
advanced pipeline for accuracy-critical paths (e.g., legal, medical, financial).
""")

    # ── CHALLENGES ────────────────────────────────────────────────────────────
    # TODO: Add more documents on topics you care about and test retrieval quality
    # TODO: Try a different cross-encoder: "cross-encoder/ms-marco-MiniLM-L-12-v2"
    # TODO: Implement Reciprocal Rank Fusion (RRF) to merge multi-query results
    # TODO: Compare HyDE vs no-HyDE on a domain-specific corpus (e.g., your code)
