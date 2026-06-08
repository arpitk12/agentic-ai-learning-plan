"""Prometheus metrics — declare all counters/histograms here."""
from prometheus_client import Counter, Gauge, Histogram

# TODO 1: Define QUERY_COUNTER = Counter("rag_queries_total", "...", ["status"])
#         status labels: "success", "abstained", "error"
QUERY_COUNTER = None  # replace with Counter(...)

# TODO 2: Define QUERY_LATENCY = Histogram("rag_query_latency_seconds", "...",
#         buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0])
QUERY_LATENCY = None  # replace with Histogram(...)

# TODO 3: Define FAITHFULNESS_HISTOGRAM = Histogram("rag_faithfulness_score", "...",
#         buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
FAITHFULNESS_HISTOGRAM = None  # replace with Histogram(...)

# TODO 4: Define ABSTAIN_COUNTER = Counter("rag_abstain_total", "...", ["reason"])
#         reason labels: "no_relevant_documents", "insufficient_grounding", "all_sentences_ungrounded"
ABSTAIN_COUNTER = None  # replace with Counter(...)

# TODO 5: Define CACHE_HITS = Counter("rag_cache_hits_total", "...", ["tier"])
CACHE_HITS = None  # replace with Counter(...)

VECTORS_INDEXED = Gauge("rag_vectors_indexed_total", "Total vectors in Qdrant")
