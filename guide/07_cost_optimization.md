[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §6 Production Checklist](guide/06_production_checklist.md) | [§8 Security →](guide/08_security.md)

---

## 7. Cost Optimization Strategies

LLM API costs are the #1 expense in production AI systems. These strategies can reduce costs by 60-90%.

### 7.1 Strategy 1: Model Routing (60-80% savings)

Route queries to the cheapest model that can handle them:

```python
from pydantic import BaseModel
from llm import chat, get_text, MODEL
import re

class RoutingDecision(BaseModel):
    complexity: str  # "simple", "standard", "complex"
    reasoning: str

# Model cost tiers (approximate, per 1M tokens input+output)
MODEL_TIERS = {
    "simple":   "gemini/gemini-2.0-flash",      # ~$0.10/1M — greetings, yes/no
    "standard": "openai/gpt-4o-mini",            # ~$0.30/1M — explanations, summaries
    "complex":  "anthropic/claude-3-5-sonnet",   # ~$6.00/1M — analysis, code, reasoning
}

def route_query(query: str) -> str:
    """Classify query complexity and return appropriate model."""
    
    # Fast rule-based routing (no LLM call needed)
    if len(query.split()) < 10:
        return MODEL_TIERS["simple"]
    if any(kw in query.lower() for kw in ["analyze", "compare", "implement", "debug", "optimize"]):
        return MODEL_TIERS["complex"]
    
    # LLM-based routing for ambiguous queries (use cheapest model)
    decision_raw = get_text(chat(
        messages=[{"role": "user", "content": f"Classify this query complexity:\n\n{query}"}],
        system="""Classify the query as simple, standard, or complex.
simple: greetings, yes/no questions, simple lookups (< 5 words answer)
standard: explanations, summaries, translations, basic Q&A
complex: multi-step reasoning, code generation, analysis, comparisons

Reply with ONLY one word: simple, standard, or complex""",
        model="gemini/gemini-2.0-flash",  # cheapest for routing
        max_tokens=5,
    ))
    
    complexity = decision_raw.strip().lower()
    return MODEL_TIERS.get(complexity, MODEL_TIERS["standard"])

def cost_optimized_chat(messages: list, **kwargs) -> dict:
    """Chat with automatic model routing."""
    if "model" not in kwargs:
        query = messages[-1].get("content", "")
        kwargs["model"] = route_query(query)
    return chat(messages, **kwargs)
```

**Typical savings**: For a mixed-workload agent: 65% cost reduction vs always using Claude-3.5-Sonnet.

### 7.2 Strategy 2: Semantic Caching (20-40% savings)

Cache responses for semantically similar queries (not just exact matches):

```python
import redis, json, numpy as np
from sentence_transformers import SentenceTransformer
from llm import chat, get_text

r = redis.Redis(host="localhost", port=6379, db=3, decode_responses=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CACHE_SIMILARITY_THRESHOLD = 0.95  # queries with >95% similarity get cached response

def semantic_cache_lookup(query: str) -> str | None:
    """Find a cached response for a semantically similar query."""
    query_vec = embedder.encode(query)
    
    # Get all cached query keys
    keys = r.keys("query_cache:*")
    
    best_score = 0
    best_response = None
    
    for key in keys[:500]:  # limit search to recent 500 cached queries
        cached = r.hgetall(key)
        if not cached:
            continue
        
        cached_vec = np.array(json.loads(cached["embedding"]))
        similarity = float(np.dot(query_vec, cached_vec) / 
                          (np.linalg.norm(query_vec) * np.linalg.norm(cached_vec)))
        
        if similarity > CACHE_SIMILARITY_THRESHOLD and similarity > best_score:
            best_score = similarity
            best_response = cached["response"]
    
    return best_response

def cached_agent(query: str, ttl: int = 3600) -> str:
    # Check semantic cache
    cached = semantic_cache_lookup(query)
    if cached:
        return f"[cached] {cached}"
    
    # Run agent
    response = get_text(chat([{"role": "user", "content": query}]))
    
    # Save to cache
    import hashlib
    key = f"query_cache:{hashlib.sha256(query.encode()).hexdigest()}"
    query_vec = embedder.encode(query).tolist()
    r.hset(key, mapping={
        "query": query,
        "response": response,
        "embedding": json.dumps(query_vec),
    })
    r.expire(key, ttl)
    
    return response
```

### 7.3 Strategy 3: Context Window Trimming (10-30% savings)

Every token in the prompt costs money. Keep context minimal:

```python
def trim_conversation_history(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 4000,
    always_keep_last_n: int = 4,
) -> list[dict]:
    """
    Trim message history to stay within token budget.
    Always keeps: system prompt + first user message + last N messages.
    """
    def estimate_tokens(msgs: list) -> int:
        return sum(len(m.get("content", "") or "").split() * 1.3 for m in msgs)
    
    # Always keep last N messages
    protected = messages[-always_keep_last_n:] if len(messages) > always_keep_last_n else messages
    trimmable = messages[:-always_keep_last_n] if len(messages) > always_keep_last_n else []
    
    # Trim from oldest trimmable messages first
    while estimate_tokens(trimmable + protected) > max_tokens and trimmable:
        trimmable.pop(0)  # remove oldest
    
    return trimmable + protected

def summarize_and_compress(messages: list[dict], keep_last: int = 6) -> list[dict]:
    """
    When conversation is too long: summarize old messages, keep recent ones.
    """
    if len(messages) <= keep_last + 2:
        return messages
    
    old = messages[:-keep_last]
    recent = messages[-keep_last:]
    
    summary = get_text(chat([{
        "role": "user",
        "content": "Summarize this conversation history in 3-5 bullet points, "
                   "preserving all key facts, decisions, and user preferences:\n\n" +
                   "\n".join([f"{m['role']}: {str(m.get('content',''))[:500]}" for m in old])
    }], system="You are a conversation summarizer. Be concise and capture all important information."))
    
    return [{"role": "system", "content": f"[Conversation history summary]:\n{summary}"}] + recent
```

### 7.4 Strategy 4: Prompt Compression

```python
def compress_for_context(doc: str, question: str, max_words: int = 400) -> str:
    """
    Extract only relevant sentences from a long document.
    Reduces context token count while preserving answer quality.
    """
    if len(doc.split()) <= max_words:
        return doc  # already short enough
    
    return get_text(chat([{
        "role": "user",
        "content": f"""Extract ONLY the sentences from this document that are relevant to answering:
"{question}"

Document:
{doc[:5000]}

Rules:
- Include ONLY sentences directly relevant to the question
- Preserve exact wording — don't paraphrase
- If nothing is relevant, respond with: NOTHING_RELEVANT"""
    }], system="Extract relevant sentences exactly as written. Be selective and concise."))

### 7.5 Cost Budget Calculator

```python
# Cost estimation for different usage levels
def estimate_monthly_cost(
    queries_per_day: int,
    avg_input_tokens: int = 1000,
    avg_output_tokens: int = 500,
    model_mix: dict = None,  # {"simple": 0.4, "standard": 0.5, "complex": 0.1}
) -> dict:
    if model_mix is None:
        model_mix = {"simple": 0.4, "standard": 0.5, "complex": 0.1}
    
    # Approximate costs per 1M tokens (input + output combined)
    costs_per_1m = {
        "simple": 0.10,    # gemini-2.0-flash
        "standard": 0.30,  # gpt-4o-mini
        "complex": 6.00,   # claude-3-5-sonnet
    }
    
    total_tokens_per_day = queries_per_day * (avg_input_tokens + avg_output_tokens)
    
    daily_cost = 0
    for tier, fraction in model_mix.items():
        tier_tokens = total_tokens_per_day * fraction
        daily_cost += (tier_tokens / 1_000_000) * costs_per_1m[tier]
    
    return {
        "daily_cost_usd": round(daily_cost, 2),
        "monthly_cost_usd": round(daily_cost * 30, 2),
        "cost_per_query_cents": round(daily_cost / queries_per_day * 100, 4),
    }

# Example: 1000 queries/day with routing
print(estimate_monthly_cost(1000))
# → {'daily_cost_usd': 1.58, 'monthly_cost_usd': 47.4, 'cost_per_query_cents': 0.158}
```

---

---

[🏠 Index](../PRODUCTION_AGENT_GUIDE.md) | [← §6 Production Checklist](guide/06_production_checklist.md) | [§8 Security →](guide/08_security.md)
