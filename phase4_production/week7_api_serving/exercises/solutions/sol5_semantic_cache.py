"""
SOLUTION — Exercise 5: Semantic Cache for LLM Responses

Key concepts demonstrated:
- SemanticCacheStore: in-memory store with TTL eviction and LRU eviction
- SemanticCache.query(): embed query → cosine lookup → cache hit or LLM call
- Threshold tuning: 0.92 is the recommended starting point
  - Too high (0.99): almost no cache hits (only exact rephrases)
  - Too low (0.85): false hits — different questions share the same answer
- hit_rate, tokens_saved, cost_saved tracked across the session
- Threshold sensitivity demo: measure actual similarity for paired queries

Production swap:
  Replace the in-memory _entries list with Redis:
    import redis; r = redis.Redis(...)
    r.hset(key, mapping={...}); r.expire(key, ttl)
  Replace the linear scan with a FAISS or ChromaDB lookup for large caches.

pip install sentence-transformers numpy
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dotenv import load_dotenv
load_dotenv()

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from llm import chat, get_text, calc_cost, MODEL

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


# ─── Cache store ──────────────────────────────────────────────────────────────

class SemanticCacheStore:
    """
    In-memory store with:
    - O(n) linear similarity scan — fine up to ~500 entries
    - TTL-based expiry: evict_expired() called on every lookup
    - LRU-style eviction: oldest entry dropped when max_size reached
    - Hit counter per entry for analytics
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self._entries: list[dict] = []
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.total_hits = 0
        self.total_misses = 0

    def find_similar(self, q_vec: np.ndarray, threshold: float) -> str | None:
        self.evict_expired()
        best_score = 0.0
        best_response: str | None = None
        for entry in self._entries:
            cached = np.asarray(entry["embedding"])
            # Dot product of unit vectors == cosine similarity
            score = float(np.dot(q_vec, cached))
            if score > threshold and score > best_score:
                best_score = score
                best_response = entry["response"]
                entry["hits"] += 1
        return best_response

    def store(self, query: str, q_vec: np.ndarray, response: str):
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)  # drop oldest (LRU)
        self._entries.append({
            "query": query[:500],
            "embedding": q_vec.tolist(),
            "response": response,
            "hits": 0,
            "expires_at": time.time() + self.ttl,
        })

    def evict_expired(self):
        now = time.time()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["expires_at"] > now]
        if before != len(self._entries):
            print(f"  [cache] Evicted {before - len(self._entries)} expired entries")

    @property
    def size(self) -> int:
        return len(self._entries)

    def top_entries(self, n: int = 3) -> list[dict]:
        return sorted(self._entries, key=lambda e: e["hits"], reverse=True)[:n]


# ─── Semantic cache wrapper ────────────────────────────────────────────────────

class SemanticCache:
    """
    Wraps any LLM call with semantic caching.

    similarity_threshold guide:
      0.98  → nearly-identical queries only (very conservative)
      0.95  → same question rephrased with minor word changes
      0.92  → catches synonyms and paraphrases (recommended)
      0.88  → aggressive; risk of unrelated questions sharing answers
    """

    def __init__(self, similarity_threshold: float = 0.92, ttl_seconds: int = 3600):
        self.threshold = similarity_threshold
        self._store = SemanticCacheStore(ttl_seconds=ttl_seconds)
        self.tokens_saved = 0
        self.cost_saved_usd = 0.0
        self._latencies: list[float] = []

    def query(
        self,
        user_query: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> dict:
        """
        1. Embed the query (always needed for cache lookup)
        2. Find semantically similar cached response
        3. If miss → call LLM → store response
        """
        model = model or MODEL
        t0 = time.time()

        # Always embed — needed for both lookup and storage
        q_vec = EMBEDDER.encode(user_query, normalize_embeddings=True)

        cached_response = self._store.find_similar(q_vec, self.threshold)

        if cached_response is not None:
            # ── Cache HIT ──────────────────────────────────────────────────
            self._store.total_hits += 1
            ms = (time.time() - t0) * 1000
            self._latencies.append(ms)

            # Estimate cost saving (approximate token count)
            est_tokens = (len(user_query.split()) + len(cached_response.split())) * 2
            self.tokens_saved += est_tokens
            self.cost_saved_usd += calc_cost(model, est_tokens // 2, est_tokens // 2)

            return {
                "response": cached_response,
                "cached": True,
                "tokens_used": 0,
                "cost_usd": 0.0,
                "latency_ms": round(ms, 1),
            }

        # ── Cache MISS — call LLM ─────────────────────────────────────────
        self._store.total_misses += 1
        llm_resp = chat(
            [{"role": "user", "content": user_query}],
            system=system, model=model, max_tokens=max_tokens,
        )
        response_text = get_text(llm_resp)
        ms = (time.time() - t0) * 1000
        self._latencies.append(ms)

        usage = llm_resp.usage
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)

        self._store.store(user_query, q_vec, response_text)

        return {
            "response": response_text,
            "cached": False,
            "tokens_used": usage.total_tokens,
            "cost_usd": cost,
            "latency_ms": round(ms, 1),
        }

    @property
    def stats(self) -> dict:
        total = self._store.total_hits + self._store.total_misses
        avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0
        return {
            "total_queries":  total,
            "cache_hits":     self._store.total_hits,
            "cache_misses":   self._store.total_misses,
            "hit_rate_pct":   round(100 * self._store.total_hits / total, 1) if total else 0.0,
            "tokens_saved":   self.tokens_saved,
            "cost_saved_usd": round(self.cost_saved_usd, 6),
            "cache_entries":  self._store.size,
            "avg_latency_ms": round(avg_lat, 1),
            "threshold":      self.threshold,
        }

    def show_top_cached(self, n: int = 3):
        for e in self._store.top_entries(n):
            print(f"  [{e['hits']} hits] {e['query'][:55]}…")
            print(f"           → {e['response'][:65]}…")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Semantic Cache Solution ===\n")
    cache = SemanticCache(similarity_threshold=0.92, ttl_seconds=600)

    test_queries = [
        # Round 1 — warm up cache
        ("What is Python used for in software development?",         False),
        ("Explain how RAG works in AI applications",                 False),
        ("What is the difference between SQL and NoSQL databases?",  False),
        ("How does Redis work as a cache?",                          False),
        # Round 2 — paraphrases (should HIT)
        ("What are the main uses of Python as a programming language?", True),
        ("How does Retrieval Augmented Generation work?",               True),
        ("SQL vs NoSQL — what are the key differences?",               True),
        ("Explain Redis caching mechanism",                            True),
        # Round 3 — genuinely new (should MISS)
        ("What is Kubernetes and why is it used?",   False),
        ("How does Docker containerisation work?",   False),
    ]

    total_cost = 0.0
    print(f"{'Query':<55} {'Status':<10} {'Tokens':>7} {'Cost':>9} {'ms':>6}  Expected")
    print("─" * 100)

    for query, expected_hit in test_queries:
        r = cache.query(query, max_tokens=150)
        total_cost += r["cost_usd"]
        status = "✅ CACHED" if r["cached"] else "⬆ LLM"
        expected = "hit" if expected_hit else "miss"
        correct = "✓" if r["cached"] == expected_hit else "✗"
        print(f"{query[:54]:<55} {status:<10} {r['tokens_used']:>7} "
              f"${r['cost_usd']:>8.4f} {r['latency_ms']:>5.0f}ms  "
              f"[expected {expected}] {correct}")

    print()
    stats = cache.stats
    print("=" * 55)
    print("STATISTICS")
    print("=" * 55)
    for k, v in stats.items():
        print(f"  {k:<24}: {v}")

    print(f"\n  Total LLM cost:      ${total_cost:.4f}")
    print(f"  Cost without cache:  ${total_cost + stats['cost_saved_usd']:.4f}")
    print(f"  Savings:             ${stats['cost_saved_usd']:.4f}")

    print("\nTop cached entries by hit count:")
    cache.show_top_cached(3)

    # Threshold sensitivity
    print("\n" + "=" * 55)
    print("THRESHOLD SENSITIVITY")
    print("=" * 55)
    pairs = [
        ("What is Python?",            "What is the Python programming language?"),
        ("Explain machine learning.",   "How does machine learning work?"),
        ("What is a vector database?", "What is Kubernetes?"),  # should NOT match
    ]
    for q1, q2 in pairs:
        v1 = EMBEDDER.encode(q1, normalize_embeddings=True)
        v2 = EMBEDDER.encode(q2, normalize_embeddings=True)
        sim = float(np.dot(v1, v2))
        print(f"\n  sim={sim:.3f}  A: {q1}")
        print(f"               B: {q2}")
        for t in [0.95, 0.92, 0.88]:
            print(f"    threshold={t}: {'HIT' if sim >= t else 'miss'}")
