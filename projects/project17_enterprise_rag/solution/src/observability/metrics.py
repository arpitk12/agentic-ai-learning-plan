"""
Prometheus metrics — export via /metrics (mounted in app.py).

Key metrics
───────────
  rag_queries_total           — total queries by status (success|abstained|error)
  rag_query_latency_seconds   — end-to-end query latency histogram
  rag_faithfulness_score      — per-query faithfulness score distribution
  rag_abstain_total           — abstained queries by reason
  rag_cache_hits_total        — cache hits by tier (exact|semantic)
"""
from prometheus_client import Counter, Gauge, Histogram

QUERY_COUNTER = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["status"],   # success | abstained | error
)

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0],
)

FAITHFULNESS_HISTOGRAM = Histogram(
    "rag_faithfulness_score",
    "NLI faithfulness score per query (non-abstained only)",
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)

ABSTAIN_COUNTER = Counter(
    "rag_abstain_total",
    "Abstained queries by reason",
    ["reason"],   # no_relevant_documents | insufficient_grounding | all_sentences_ungrounded
)

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Cache hits by tier",
    ["tier"],     # exact_or_semantic
)

VECTORS_INDEXED = Gauge(
    "rag_vectors_indexed_total",
    "Total vectors in Qdrant collection",
)

EMBED_CACHE_HIT_RATE = Gauge(
    "rag_embed_cache_hit_rate",
    "Embedding cache hit rate (semantic cache)",
)
