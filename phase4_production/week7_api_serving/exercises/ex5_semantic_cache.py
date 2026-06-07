"""
Exercise 5: Semantic Cache for LLM Responses
Guide Section: §7.2 — Cost Optimization: Semantic Caching (20-40% savings)

Goal: Cache LLM responses keyed by semantic similarity, not exact text match.
      "How tall is the Eiffel Tower?" → cache hit for "What's the Eiffel Tower height?"

Why semantic cache instead of exact cache?
  - Exact cache: matches "hello" == "hello" only (very low hit rate)
  - Semantic cache: matches queries with cosine similarity > threshold
  - Real workloads have many paraphrases of the same question
  - Typical hit rate in production: 20-35% of queries

When to use:
  - High query volume with repetitive questions (FAQ bot, support agent)
  - Expensive model (Claude-3.5-Sonnet, GPT-4o) → big savings per hit
  - When answers don't change frequently (static knowledge base)

When NOT to use:
  - Real-time data queries ("What's the price of AAPL right now?")
  - Personal/session-specific answers ("What did I ask 5 minutes ago?")
  - Very short TTL data (less than your cache TTL)

pip install sentence-transformers numpy
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dotenv import load_dotenv
load_dotenv()

import json, hashlib, time
import numpy as np
from sentence_transformers import SentenceTransformer
from llm import chat, get_text, calc_cost, MODEL

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


# ─── Cache Backend ─────────────────────────────────────────────────────────────
# We use an in-memory list here. In production, replace with Redis:
#   import redis
#   r = redis.Redis(host="localhost", port=6379, db=3)
#   r.hset(key, mapping={...}); r.expire(key, ttl)

class SemanticCacheStore:
    """
    In-memory cache store.
    Each entry: {query, embedding, response, hits, timestamp, expires_at}

    In production, replace with Redis using HSET/EXPIRE.
    The scan-based similarity lookup (O(n)) is fine up to ~500 cached entries.
    For large caches, use a vector DB (ChromaDB/FAISS) as the lookup engine.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self._entries: list[dict] = []
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.total_hits = 0
        self.total_misses = 0

    def find_similar(self, query_vec: np.ndarray, threshold: float) -> str | None:
        """
        Scan all cached entries and return the response for the most similar query.
        Returns None if no entry exceeds the threshold.
        
        Similarity is cosine similarity (dot product of normalized vectors).
        threshold=0.92 means "92% semantically similar" — catches paraphrases.
        threshold=0.99 means "nearly identical" — catches only minor rewordings.
        """
        self.evict_expired()  # clean up stale entries first

        best_score = 0.0
        best_response: str | None = None

        for entry in self._entries:
            cached_vec = np.asarray(entry["embedding"])
            # Cosine similarity: dot product of unit vectors
            score = float(np.dot(query_vec, cached_vec))

            if score > threshold and score > best_score:
                best_score = score
                best_response = entry["response"]
                entry["hits"] += 1  # track usage frequency

        return best_response

    def store(self, query: str, query_vec: np.ndarray, response: str):
        """Store a new (query, response) pair with its embedding."""
        # LRU-style eviction: remove oldest entry when full
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)

        self._entries.append({
            "query": query[:500],
            "embedding": query_vec.tolist(),
            "response": response,
            "hits": 0,
            "stored_at": time.time(),
            "expires_at": time.time() + self.ttl,
        })

    def evict_expired(self):
        """Remove entries older than TTL."""
        now = time.time()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["expires_at"] > now]
        evicted = before - len(self._entries)
        if evicted:
            print(f"  [cache] Evicted {evicted} expired entries")

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def top_entries(self, n: int = 5) -> list[dict]:
        """Return the n most-hit cache entries."""
        return sorted(self._entries, key=lambda e: e["hits"], reverse=True)[:n]


# ─── Semantic Cache Wrapper ────────────────────────────────────────────────────

class SemanticCache:
    """
    Wraps any LLM call with semantic caching.

    Similarity thresholds guide:
      0.98+  → only nearly-identical queries hit (very safe, low hit rate)
      0.95   → catches rephrased questions with same meaning (recommended start)
      0.92   → catches synonyms, spelling variations (good balance)
      0.88   → aggressive cache, risk of false hits (test carefully)
      < 0.85 → too loose, different questions may share cached answers
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
    ):
        self.threshold = similarity_threshold
        self._store = SemanticCacheStore(ttl_seconds=ttl_seconds)
        self.tokens_saved = 0
        self.cost_saved_usd = 0.0
        self.response_times: list[float] = []

    def query(
        self,
        user_query: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 512,
    ) -> dict:
        """
        Execute an LLM query with semantic caching.

        Returns:
          response     — the answer text (from cache or LLM)
          cached       — True if served from cache
          tokens_used  — 0 if cached, actual tokens if LLM was called
          cost_usd     — 0.0 if cached
          latency_ms   — response time
        """
        model = model or MODEL
        t0 = time.time()

        # Step 1: Embed the query (needed for cache lookup regardless)
        # Normalize for cosine similarity (dot product = cosine for unit vectors)
        query_vec = EMBEDDER.encode(user_query, normalize_embeddings=True)

        # Step 2: Look up semantic cache
        cached = self._store.find_similar(query_vec, self.threshold)

        if cached is not None:
            # ── Cache HIT ──
            self._store.total_hits += 1
            elapsed = (time.time() - t0) * 1000
            self.response_times.append(elapsed)

            # Estimate tokens/cost saved (approximate)
            approx_tokens = (len(user_query.split()) + len(cached.split())) * 2
            self.tokens_saved += approx_tokens
            self.cost_saved_usd += calc_cost(model, approx_tokens // 2, approx_tokens // 2)

            return {
                "response": cached,
                "cached": True,
                "tokens_used": 0,
                "cost_usd": 0.0,
                "latency_ms": round(elapsed, 1),
            }

        # ── Cache MISS — call LLM ──
        self._store.total_misses += 1
        messages = [{"role": "user", "content": user_query}]
        llm_resp = chat(messages, system=system, model=model, max_tokens=max_tokens)
        response_text = get_text(llm_resp)
        elapsed = (time.time() - t0) * 1000
        self.response_times.append(elapsed)

        usage = llm_resp.usage
        cost = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)

        # Step 3: Store the response
        self._store.store(user_query, query_vec, response_text)

        return {
            "response": response_text,
            "cached": False,
            "tokens_used": usage.total_tokens,
            "cost_usd": cost,
            "latency_ms": round(elapsed, 1),
        }

    @property
    def stats(self) -> dict:
        store = self._store
        total = store.total_hits + store.total_misses
        avg_latency = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        return {
            "total_queries":      total,
            "cache_hits":         store.total_hits,
            "cache_misses":       store.total_misses,
            "hit_rate_pct":       round(100 * store.total_hits / total, 1) if total else 0.0,
            "tokens_saved":       self.tokens_saved,
            "cost_saved_usd":     round(self.cost_saved_usd, 6),
            "cache_entries":      store.size,
            "avg_latency_ms":     round(avg_latency, 1),
            "threshold_used":     self.threshold,
        }

    def show_top_cached(self, n: int = 3):
        """Show the most frequently hit cache entries."""
        top = self._store.top_entries(n)
        if not top:
            print("Cache is empty.")
            return
        print("\nTop cache entries by hit count:")
        for e in top:
            print(f"  [{e['hits']} hits] {e['query'][:60]}…")
            print(f"           → {e['response'][:70]}…")


# ─── MAIN DEMO ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Semantic Cache Demo ===\n")
    print("Threshold: 0.92 (92% cosine similarity = treated as same query)\n")

    # Try different thresholds:
    # 0.95 = conservative (fewer hits, safer)
    # 0.92 = balanced (recommended)
    # 0.88 = aggressive (more hits, higher false-positive risk)
    cache = SemanticCache(similarity_threshold=0.92, ttl_seconds=600)

    # Test set: mix of originals, paraphrases (should hit), and new questions
    test_queries = [
        # --- Round 1: "warm up" the cache ---
        ("What is Python used for in software development?",          "python_q"),
        ("Explain how RAG works in AI applications",                  "rag_q"),
        ("What is the difference between SQL and NoSQL databases?",   "db_q"),
        ("How does Redis work as a cache?",                           "redis_q"),

        # --- Round 2: paraphrases — SHOULD hit cache ---
        ("What are the main uses of Python as a programming language?", "python_q"),  # ≈ query 1
        ("How does Retrieval Augmented Generation work?",               "rag_q"),     # ≈ query 2
        ("SQL vs NoSQL databases — what are the key differences?",      "db_q"),     # ≈ query 3
        ("Explain Redis caching mechanism",                             "redis_q"),  # ≈ query 4

        # --- Round 3: genuinely new questions — should MISS ---
        ("What is Kubernetes and why is it used?",     "new_1"),
        ("How does Docker containerization work?",     "new_2"),
    ]

    total_cost = 0.0
    print(f"{'Query':<57} {'Status':<10} {'Tokens':>7} {'Cost':>9} {'ms':>6}")
    print("─" * 95)

    for query, _ in test_queries:
        result = cache.query(query, max_tokens=150)
        total_cost += result["cost_usd"]

        status = "✅ CACHED" if result["cached"] else "⬆ LLM"
        tokens = result["tokens_used"]
        cost = result["cost_usd"]
        ms = result["latency_ms"]

        print(f"{query[:56]:<57} {status:<10} {tokens:>7} ${cost:>8.4f} {ms:>5.0f}ms")

    print()

    # ─── Stats ────────────────────────────────────────────────────────────────
    stats = cache.stats
    print(f"{'='*55}")
    print("CACHE STATISTICS")
    print(f"{'='*55}")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")

    print(f"\n  Total LLM cost this session:  ${total_cost:.4f}")
    print(f"  Cost if NO cache:             ${(total_cost + stats['cost_saved_usd']):.4f}")
    print(f"  Savings:                      ${stats['cost_saved_usd']:.4f}")

    cache.show_top_cached(3)

    # ─── Threshold Sensitivity Demo ───────────────────────────────────────────
    print(f"\n{'='*55}")
    print("THRESHOLD SENSITIVITY — how threshold affects hits")
    print(f"{'='*55}")

    control_pairs = [
        ("What is Python?",             "What is the Python programming language?"),
        ("Explain machine learning.",    "How does machine learning work?"),
        ("What is a vector database?",   "What is Kubernetes?"),  # should NOT match
    ]

    for q1, q2 in control_pairs:
        v1 = EMBEDDER.encode(q1, normalize_embeddings=True)
        v2 = EMBEDDER.encode(q2, normalize_embeddings=True)
        sim = float(np.dot(v1, v2))
        print(f"  Similarity: {sim:.3f}")
        print(f"    A: {q1}")
        print(f"    B: {q2}")
        for thresh in [0.95, 0.92, 0.88]:
            hit = "HIT" if sim >= thresh else "miss"
            print(f"      threshold={thresh}: {hit}")
        print()

    # ─── CHALLENGES ───────────────────────────────────────────────────────────
    # TODO: Swap InMemoryCache for Redis (docker run -p 6379:6379 redis:alpine)
    # TODO: Test threshold=0.85 — verify "Python" and "Kubernetes" DON'T share cache
    # TODO: Add TTL eviction test: set ttl_seconds=5, wait 6 seconds, re-query
    # TODO: Track cache hit rate in Prometheus (Counter: cache_hits_total)
    # TODO: For FAQ bots: pre-warm the cache with common Q&A pairs at startup
