# Token Optimization Guide — Reduce LLM Costs by 70–90%

> Tokens = money. Every character in your prompt and response has a price.  
> This guide covers the full stack: counting → compressing → routing → caching → monitoring.

Related: [`framework_selection_guide.md`](framework_selection_guide.md) · [`guide/07_cost_optimization.md`](guide/07_cost_optimization.md)

---

## Table of Contents

1. [What Is a Token?](#1-what-is-a-token)
2. [Count Tokens Before You Call](#2-count-tokens-before-you-call)
3. [Prompt Engineering for Token Reduction](#3-prompt-engineering-for-token-reduction)
4. [System Prompt Optimization](#4-system-prompt-optimization)
5. [Model Routing — The Biggest Lever](#5-model-routing--the-biggest-lever)
6. [Semantic Caching](#6-semantic-caching)
7. [Context Window Management](#7-context-window-management)
8. [Agent Loop Optimization](#8-agent-loop-optimization)
9. [RAG Token Optimization](#9-rag-token-optimization)
10. [Batching and Async](#10-batching-and-async)
11. [Budget Guardrails](#11-budget-guardrails)
12. [Exercises](#12-exercises)
13. [References and Resources](#13-references-and-resources)

---

## 1. What Is a Token?

A **token** is the basic unit of text that an LLM processes. Roughly:

```
1 token ≈ 4 characters ≈ 0.75 words (English)
```

Examples:
```
"Hello"              → 1 token
"Hello, world!"      → 4 tokens
"Uncharacteristically" → 6 tokens
"The quick brown fox" → 4 tokens
A Python function (50 lines) → ~500-800 tokens
```

### Why It Matters — Real Cost Table (June 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best for |
|---|---|---|---|
| `gemini/gemini-2.0-flash` | $0.075 | $0.30 | Routing, simple tasks, bulk |
| `openai/gpt-4o-mini` | $0.15 | $0.60 | Standard Q&A, summaries |
| `groq/llama-3.3-70b` | $0.59 | $0.79 | Complex reasoning (free tier) |
| `openai/gpt-4o` | $2.50 | $10.00 | High-stakes, complex tasks |
| `anthropic/claude-3-5-sonnet` | $3.00 | $15.00 | Long context, nuanced tasks |
| `openai/o3` | $10.00 | $40.00 | Deep reasoning only |

**Key insight**: output tokens cost 3–4× more than input tokens on most models. **Shorter answers = bigger savings**.

### Quick Cost Estimate Formula

$$\text{cost} = \frac{T_{in} \times P_{in} + T_{out} \times P_{out}}{1{,}000{,}000}$$

Where $T_{in}$, $T_{out}$ are token counts and $P_{in}$, $P_{out}$ are prices per 1M tokens.

```python
def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "openai/gpt-4o-mini",
) -> float:
    """Estimate USD cost for a single LLM call."""
    prices = {
        "openai/gpt-4o-mini":               (0.15,  0.60),
        "openai/gpt-4o":                    (2.50, 10.00),
        "anthropic/claude-3-5-sonnet":      (3.00, 15.00),
        "gemini/gemini-2.0-flash":          (0.075, 0.30),
        "groq/llama-3.3-70b-versatile":     (0.59,  0.79),
    }
    p_in, p_out = prices.get(model, (1.0, 3.0))
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
```

---

## 2. Count Tokens Before You Call

Never guess — measure. Use `tiktoken` (OpenAI) or `litellm.token_counter`.

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens for a string using the model's tokenizer."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # default for GPT-4 family
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """Count total tokens for a messages list (includes role overhead)."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        total += 4  # every message has role/content overhead
        total += len(enc.encode(str(msg.get("content", ""))))
    total += 2  # reply priming
    return total


# LiteLLM universal counter (works for any provider)
from litellm import token_counter

def count_litellm(messages: list[dict], model: str = "openai/gpt-4o-mini") -> int:
    return token_counter(model=model, messages=messages)
```

### Token Budget Pattern — Fail Fast Before Calling

```python
MAX_INPUT_TOKENS = 8_000  # leave room for output

def safe_llm_call(messages: list[dict], model: str = "openai/gpt-4o-mini") -> str:
    tokens = count_messages_tokens(messages, model)
    if tokens > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Input too large: {tokens} tokens (limit {MAX_INPUT_TOKENS}). "
            "Trim context before calling."
        )
    cost_estimate = estimate_cost(tokens, output_tokens=500, model=model)
    if cost_estimate > 0.05:  # $0.05 per call limit
        raise ValueError(f"Estimated cost ${cost_estimate:.4f} exceeds limit")
    # proceed with call ...
```

---

## 3. Prompt Engineering for Token Reduction

Small wording changes have a big impact at scale.

### 3.1 Eliminate Filler Words

```python
# ❌ VERBOSE — 47 tokens
verbose = """
I would like you to please provide me with a comprehensive and detailed summary
of the following text. Make sure to include all important points and key takeaways
that are present in the text below:
"""

# ✅ CONCISE — 8 tokens (83% reduction)
concise = "Summarize, key points only:"
```

### 3.2 Constrain Output Length Explicitly

```python
# ❌ Open-ended — model writes 800 tokens
"Explain transformer architecture."

# ✅ Bounded — model writes ~80 tokens
"Explain transformer architecture. Max 3 sentences."

# ✅ Even better — structured short answer
"Explain transformer architecture in exactly 2 sentences: one for the problem it solves, one for the mechanism."
```

### 3.3 Use Format Directives to Cut Output

| Directive | Token Reduction | Use when |
|---|---|---|
| `Reply in JSON only` | 20–40% | Need parseable output |
| `Max N words` | 30–60% | Summaries, labels |
| `One sentence` | 60–80% | Classification, routing |
| `Yes or No` | 90%+ | Binary decisions |
| `List format, no explanations` | 25–40% | Bullet point outputs |

### 3.4 Zero-Shot vs Few-Shot Trade-off

Few-shot examples add input tokens but often reduce output tokens (model follows the pattern more precisely):

```python
# Few-shot — +150 input tokens, saves ~200 output tokens (net positive)
few_shot_system = """Classify sentiment. Reply with exactly one word: positive, negative, or neutral.

Examples:
"Great product!" → positive
"Terrible service" → negative
"It arrived" → neutral"""

# Zero-shot — saves 150 tokens but model may over-explain
zero_shot_system = "Classify sentiment as positive, negative, or neutral. One word only."
```

**Rule of thumb**: Add few-shot examples only when zero-shot produces verbose or inconsistent output.

### 3.5 Remove Redundant Context

```python
import re

def strip_boilerplate(text: str) -> str:
    """Remove common filler patterns from user-submitted text."""
    patterns = [
        r"(?i)please\s+",
        r"(?i)could\s+you\s+(please\s+)?",
        r"(?i)i\s+would\s+like\s+(you\s+to\s+)?",
        r"(?i)can\s+you\s+(please\s+)?",
        r"(?i)i\s+need\s+you\s+to\s+",
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    return text.strip()
```

---

## 4. System Prompt Optimization

System prompts are sent on **every call** — they're the highest-leverage optimization target.

### 4.1 Audit Your System Prompt

```python
def audit_system_prompt(system: str) -> dict:
    """Analyse a system prompt for token waste."""
    tokens = count_tokens(system)
    words = len(system.split())
    sentences = system.count(".") + system.count("!") + system.count("?")

    issues = []
    if tokens > 500:
        issues.append(f"System prompt is {tokens} tokens — consider splitting or compressing")
    if "please" in system.lower():
        issues.append("'please' in system prompt — LLMs don't need politeness markers")
    if "you are an AI" in system.lower():
        issues.append("'you are an AI' is redundant — the model knows")
    if words / max(sentences, 1) > 25:
        issues.append("Very long sentences — break into bullet points for compression")

    return {"tokens": tokens, "words": words, "issues": issues}

# Example
system = """You are a helpful AI assistant. Please be kind and polite. 
You are an AI that helps users with their questions."""

print(audit_system_prompt(system))
# → {'tokens': 34, 'words': 26, 'issues': ["'please' in system prompt", "'you are an AI' is redundant"]}
```

### 4.2 Optimized vs Bloated System Prompts

```python
# ❌ BLOATED — 87 tokens
bloated = """
You are a helpful, friendly, and knowledgeable AI assistant. Your goal is to help 
users by providing accurate, clear, and concise answers to their questions. Please 
always be polite and respectful. If you don't know something, please say so honestly.
You should never make up information. Always be truthful and helpful.
"""

# ✅ OPTIMISED — 22 tokens (75% reduction, same behaviour)
optimised = """Answer accurately and concisely. Say "I don't know" if uncertain."""
```

### 4.3 Dynamic System Prompts — Load Only What's Needed

```python
BASE_SYSTEM = "You are a research assistant."

TOOL_INSTRUCTIONS = {
    "web_search": "\n- Use web_search for current events (post-2024).",
    "calculator": "\n- Use calculator for all numeric computations.",
    "rag": "\n- Use knowledge_base for company-specific questions.",
}

def build_system_prompt(available_tools: list[str]) -> str:
    """Only include tool instructions for tools actually available this call."""
    system = BASE_SYSTEM
    for tool in available_tools:
        system += TOOL_INSTRUCTIONS.get(tool, "")
    return system
```

---

## 5. Model Routing — The Biggest Lever

Routing simple queries to cheap models saves 80–95% on those queries.

```python
import os
from litellm import completion

# Cost tiers (adjust to your provider prices)
MODELS = {
    "nano":     "gemini/gemini-2.0-flash",     # ~$0.10/1M — routing, yes/no
    "mini":     "openai/gpt-4o-mini",           # ~$0.30/1M — summaries, Q&A
    "standard": "groq/llama-3.3-70b-versatile", # ~$0.80/1M — reasoning
    "pro":      "anthropic/claude-3-5-sonnet",  # ~$12/1M — complex analysis
}

# Rule-based routing (free — no LLM call)
def rule_based_route(query: str) -> str:
    q = query.lower().strip()
    word_count = len(q.split())

    # Nano: very short or binary questions
    if word_count <= 5 or q.endswith("?") and word_count <= 8:
        return MODELS["nano"]
    # Pro: explicit complexity signals
    if any(kw in q for kw in [
        "analyze", "compare", "debug", "implement", "architecture",
        "security", "optimize", "explain in depth", "research"
    ]):
        return MODELS["pro"]
    # Standard: code-adjacent
    if any(kw in q for kw in ["code", "function", "class", "sql", "query", "script"]):
        return MODELS["standard"]
    # Default: mini
    return MODELS["mini"]


# LLM-based routing (use when rule-based is insufficient)
def llm_route(query: str) -> str:
    resp = completion(
        model=MODELS["nano"],  # always use cheapest for routing
        messages=[{"role": "user", "content": query}],
        system="""Classify this query. Reply with ONLY one word:
nano    = greeting, yes/no, simple lookup
mini    = summary, translation, basic explanation
standard= code help, step-by-step instructions
pro     = analysis, research, complex reasoning""",
        max_tokens=5,
    )
    tier = resp.choices[0].message.content.strip().lower()
    return MODELS.get(tier, MODELS["mini"])


def routed_call(query: str, use_llm_routing: bool = False) -> str:
    """Call LLM with automatic model selection."""
    model = llm_route(query) if use_llm_routing else rule_based_route(query)
    resp = completion(
        model=model,
        messages=[{"role": "user", "content": query}],
        max_tokens=1024,
    )
    return resp.choices[0].message.content
```

### Routing Impact Calculator

| Traffic mix | No routing (always Pro) | With routing | Savings |
|---|---|---|---|
| 40% nano + 50% mini + 10% pro | $120/1M queries | $18/1M queries | **85%** |
| 20% mini + 70% standard + 10% pro | $120/1M queries | $52/1M queries | **57%** |
| 100% pro | $120/1M queries | $120/1M queries | 0% |

---

## 6. Semantic Caching

Avoid calling the LLM for questions you've already answered.

```python
import hashlib, json
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
_cache: dict = {}  # replace with Redis in production


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def semantic_cache_get(query: str, threshold: float = 0.93) -> str | None:
    """Return cached response if a similar query was seen before."""
    vec = embedder.encode(query)
    best_score, best_resp = 0.0, None
    for cached_vec, cached_resp in _cache.values():
        score = cosine_similarity(vec, cached_vec)
        if score > threshold and score > best_score:
            best_score, best_resp = score, cached_resp
    return best_resp


def semantic_cache_set(query: str, response: str) -> None:
    key = hashlib.sha256(query.encode()).hexdigest()
    _cache[key] = (embedder.encode(query), response)


def cached_llm_call(query: str, model: str = "openai/gpt-4o-mini") -> tuple[str, bool]:
    """Returns (response, was_cached)."""
    cached = semantic_cache_get(query)
    if cached:
        return cached, True

    from litellm import completion
    resp = completion(model=model, messages=[{"role": "user", "content": query}])
    answer = resp.choices[0].message.content
    semantic_cache_set(query, answer)
    return answer, False
```

**Cache hit rates by domain**: FAQ systems 60–80% · Documentation Q&A 40–60% · General chat 10–20%

---

## 7. Context Window Management

Every token in the conversation history is re-sent on every turn.

### 7.1 Sliding Window

```python
def sliding_window(messages: list[dict], max_tokens: int = 4000, keep_last: int = 6) -> list[dict]:
    """Keep only the most recent messages within the token budget."""
    # Always keep system message + last N messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    window = non_system[-keep_last:]
    while count_messages_tokens(system_msgs + window) > max_tokens and len(window) > 2:
        window.pop(0)

    return system_msgs + window
```

### 7.2 Summary Compression

```python
from litellm import completion

def compress_history(messages: list[dict], keep_recent: int = 4) -> list[dict]:
    """Summarize old messages, keep recent ones verbatim."""
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) <= keep_recent + 2:
        return messages  # not worth compressing

    old = non_system[:-keep_recent]
    recent = non_system[-keep_recent:]

    summary_prompt = "Summarize this conversation in ≤5 bullet points. Include: key facts, decisions, user preferences, unresolved questions.\n\n"
    summary_prompt += "\n".join(f"{m['role'].upper()}: {str(m.get('content',''))[:300]}" for m in old)

    resp = completion(
        model="gemini/gemini-2.0-flash",  # cheapest model for summarization
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=200,
    )
    summary = resp.choices[0].message.content

    return system_msgs + [{"role": "system", "content": f"[History summary]\n{summary}"}] + recent
```

### 7.3 Token Usage: Before vs After Compression

```
Turn 20 of a conversation:
  Without compression: 12,400 tokens/call  → $0.022/call → $26.4/1000 turns
  With sliding window:  3,100 tokens/call  → $0.006/call → $7.2/1000 turns  (73% saving)
  With summary:         1,800 tokens/call  → $0.004/call → $4.3/1000 turns  (84% saving)
```

---

## 8. Agent Loop Optimization

Agent loops are the biggest token sinks — every tool call adds input+output tokens.

### 8.1 Minimize Tool Call Round-Trips

```python
# ❌ Three separate calls to LLM
result1 = llm_call("Search for X")        # call 1
result2 = llm_call("Now summarize X")     # call 2
result3 = llm_call("Format as JSON")      # call 3

# ✅ One call with clear multi-step instruction
result = llm_call("""
1. Search for X (use web_search tool)
2. Summarize the findings in 3 bullet points
3. Return ONLY valid JSON: {"summary": "...", "sources": [...]}
""")
```

### 8.2 Limit Tool Output Size

```python
def truncate_tool_output(output: str, max_chars: int = 2000) -> str:
    """Prevent tool outputs from bloating the context."""
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"\n\n[... truncated {len(output) - max_chars} chars]"
```

### 8.3 Set `max_tokens` on Every Call

```python
# Without max_tokens: model may write 2000 tokens when 200 suffice
resp = completion(model="gpt-4o-mini", messages=messages)

# With max_tokens: hard cap on output cost
resp = completion(model="gpt-4o-mini", messages=messages, max_tokens=300)
```

### 8.4 Agent Token Budget Tracker

```python
from dataclasses import dataclass, field
from litellm import completion

@dataclass
class TokenBudget:
    limit: int = 50_000          # total token budget for this agent run
    used_input: int = 0
    used_output: int = 0
    calls: int = 0

    @property
    def used_total(self) -> int:
        return self.used_input + self.used_output

    @property
    def remaining(self) -> int:
        return self.limit - self.used_total

    def check(self, estimated_input: int = 0) -> None:
        if self.used_total + estimated_input > self.limit:
            raise RuntimeError(
                f"Token budget exceeded: {self.used_total}/{self.limit} used. "
                "Agent stopping to prevent runaway costs."
            )

    def record(self, usage) -> None:
        self.used_input += usage.prompt_tokens
        self.used_output += usage.completion_tokens
        self.calls += 1

    def report(self) -> str:
        return (
            f"Calls: {self.calls} | "
            f"Input: {self.used_input:,} | "
            f"Output: {self.used_output:,} | "
            f"Total: {self.used_total:,}/{self.limit:,} | "
            f"Remaining: {self.remaining:,}"
        )


def budget_aware_call(messages: list, budget: TokenBudget, model: str = "openai/gpt-4o-mini") -> str:
    estimated = count_messages_tokens(messages, model)
    budget.check(estimated_input=estimated)

    resp = completion(model=model, messages=messages, max_tokens=512)
    budget.record(resp.usage)
    return resp.choices[0].message.content
```

---

## 9. RAG Token Optimization

Retrieval-Augmented Generation has unique token costs: chunk size × number of chunks × calls.

### 9.1 Optimal Chunk Size

```
Chunk size trade-off:
  Small chunks (128 tokens) → More precise retrieval, but many chunks needed
  Large chunks (1024 tokens) → Fewer chunks needed, but more noise in context
  Optimal for most use cases: 256–512 tokens with 10–15% overlap
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Balanced: 512 tokens, 64 token overlap (~12%)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    length_function=count_tokens,  # use token-based splitting, not character-based
)
```

### 9.2 Retrieve Only What You Need

```python
# ❌ Retrieve 10 chunks, pass all to LLM — 5,000 tokens of context
chunks = retriever.invoke(query, k=10)

# ✅ Retrieve 10, rerank, keep top 3 — 1,500 tokens of context
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")

def retrieve_and_rerank(query: str, retriever, top_k_retrieve: int = 10, top_k_keep: int = 3):
    chunks = retriever.invoke(query, k=top_k_retrieve)
    pairs = [(query, c.page_content) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), reverse=True)
    return [c for _, c in ranked[:top_k_keep]]
```

### 9.3 Compress Retrieved Context

```python
def compress_context(chunks: list, query: str, max_tokens: int = 1500) -> str:
    """Extract only query-relevant sentences from retrieved chunks."""
    full_text = "\n\n".join(c.page_content for c in chunks)
    if count_tokens(full_text) <= max_tokens:
        return full_text  # already short enough

    # Use cheap model to extract relevant sentences
    resp = completion(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "user", "content":
            f"Extract ONLY sentences relevant to: '{query}'\n\n"
            f"Text:\n{full_text[:6000]}\n\n"
            "Output: exact sentences only, no commentary."
        }],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
```

---

## 10. Batching and Async

Concurrent calls with connection reuse dramatically improve throughput without adding tokens.

```python
import asyncio
from litellm import acompletion

async def batch_process(queries: list[str], model: str = "openai/gpt-4o-mini",
                        concurrency: int = 10) -> list[str]:
    """Process many queries concurrently, respecting rate limits."""
    sem = asyncio.Semaphore(concurrency)

    async def call(query: str) -> str:
        async with sem:
            resp = await acompletion(
                model=model,
                messages=[{"role": "user", "content": query}],
                max_tokens=256,
            )
            return resp.choices[0].message.content

    return await asyncio.gather(*[call(q) for q in queries])


# Usage
queries = ["Summarize X", "Explain Y", "Compare A and B"] * 100
results = asyncio.run(batch_process(queries, concurrency=20))
```

**Latency vs cost**: async batching doesn't reduce token count but reduces wall-clock time by 10–20×, enabling more efficient use of rate limits.

---

## 11. Budget Guardrails

Prevent runaway agents from spending hundreds of dollars.

```python
import time
from collections import defaultdict

class CostGuard:
    """Per-session and per-day spending limits."""

    PRICES = {
        "openai/gpt-4o-mini":           (0.15,  0.60),
        "openai/gpt-4o":                (2.50, 10.00),
        "anthropic/claude-3-5-sonnet":  (3.00, 15.00),
        "gemini/gemini-2.0-flash":      (0.075, 0.30),
        "groq/llama-3.3-70b-versatile": (0.59,  0.79),
    }

    def __init__(self, session_limit: float = 1.00, daily_limit: float = 10.00):
        self.session_limit = session_limit
        self.daily_limit = daily_limit
        self.session_spent = 0.0
        self.daily_spent = defaultdict(float)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p_in, p_out = self.PRICES.get(model, (1.0, 3.0))
        cost = (input_tokens * p_in + output_tokens * p_out) / 1_000_000
        self.session_spent += cost
        self.daily_spent[time.strftime("%Y-%m-%d")] += cost

        if self.session_spent > self.session_limit:
            raise RuntimeError(
                f"Session budget exceeded: ${self.session_spent:.4f} > ${self.session_limit}"
            )
        today = time.strftime("%Y-%m-%d")
        if self.daily_spent[today] > self.daily_limit:
            raise RuntimeError(
                f"Daily budget exceeded: ${self.daily_spent[today]:.4f} > ${self.daily_limit}"
            )
        return cost

    def report(self) -> dict:
        today = time.strftime("%Y-%m-%d")
        return {
            "session_spent_usd": round(self.session_spent, 6),
            "daily_spent_usd": round(self.daily_spent[today], 6),
            "session_remaining_usd": round(self.session_limit - self.session_spent, 6),
        }
```

---

## 12. Exercises

### Exercise 1 — Token Counter and Cost Estimator

**Goal**: Build a utility that counts tokens and estimates cost before any LLM call.

```python
# File: exercises/token_opt/ex1_token_counter.py
import tiktoken
from dataclasses import dataclass

PRICES = {
    "openai/gpt-4o-mini":          (0.15, 0.60),
    "openai/gpt-4o":               (2.50, 10.00),
    "gemini/gemini-2.0-flash":     (0.075, 0.30),
    "groq/llama-3.3-70b-versatile":(0.59, 0.79),
}

@dataclass
class TokenReport:
    input_tokens: int
    estimated_output_tokens: int
    model: str

    # TODO 1: Add a property 'estimated_cost_usd' that uses PRICES to compute cost

    # TODO 2: Add a __str__ method that prints a readable one-liner, e.g.:
    #   "gpt-4o-mini | 1,234 in + ~500 out = $0.0006"


def count_and_estimate(
    messages: list[dict],
    model: str = "openai/gpt-4o-mini",
    expected_output_tokens: int = 500,
) -> TokenReport:
    """
    TODO 3: Count tokens in the messages list using tiktoken.
    Return a TokenReport with the counts and model.

    Hint: use tiktoken.encoding_for_model() or get_encoding("cl100k_base") as fallback.
    Each message adds ~4 overhead tokens (role + separators).
    """
    raise NotImplementedError


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the transformer architecture in 3 sentences."},
    ]
    for model in PRICES:
        report = count_and_estimate(msgs, model=model)
        print(report)
```

---

### Exercise 2 — Prompt Compression Pipeline

**Goal**: Reduce a verbose user prompt by 40–60% while preserving its intent.

```python
# File: exercises/token_opt/ex2_prompt_compression.py
import re

FILLER_PATTERNS = [
    r"(?i)please\s+",
    r"(?i)could\s+you\s+(please\s+)?",
    r"(?i)i\s+would\s+like\s+(you\s+to\s+)?",
    r"(?i)can\s+you\s+(please\s+)?",
    r"(?i)i\s+(need|want)\s+(you\s+to\s+)?",
    r"(?i)as\s+(an?\s+)?(AI|language model|assistant)[,.]?\s*",
    r"(?i)you\s+are\s+an?\s+(AI|assistant|language model)[,.]?\s*",
]

def strip_filler(text: str) -> str:
    """
    TODO 1: Apply all FILLER_PATTERNS using re.sub and return cleaned text.
    """
    raise NotImplementedError


def compress_system_prompt(system: str) -> str:
    """
    TODO 2: Compress a system prompt:
    - Remove filler phrases
    - Convert long prose into bullet points (split on ". " and prefix with "- ")
    - Strip double spaces and leading/trailing whitespace
    """
    raise NotImplementedError


def enforce_output_constraints(prompt: str, max_words: int | None = None,
                               output_format: str | None = None) -> str:
    """
    TODO 3: Append constraint directives to the prompt:
    - If max_words is set, append: "Answer in at most {max_words} words."
    - If output_format is set (e.g. "JSON", "bullet list"), append the format instruction.
    """
    raise NotImplementedError


# ── Test ───────────────────────────────────────────────────────────────────
BLOATED = """
You are a helpful AI assistant. I would like you to please provide me with a 
comprehensive and detailed explanation of how neural networks learn. As an AI, 
you should make sure to cover backpropagation, gradient descent, and loss functions. 
Can you please also explain the role of activation functions?
"""

if __name__ == "__main__":
    compressed = compress_system_prompt(BLOATED)
    constrained = enforce_output_constraints(compressed, max_words=100, output_format="bullet list")
    
    original_tokens = count_tokens(BLOATED)
    compressed_tokens = count_tokens(constrained)
    reduction = (1 - compressed_tokens / original_tokens) * 100
    
    print(f"Original:   {original_tokens} tokens")
    print(f"Compressed: {compressed_tokens} tokens ({reduction:.0f}% reduction)")
    print(f"\nResult:\n{constrained}")
```

---

### Exercise 3 — Model Router

**Goal**: Build a two-stage router (rule-based + LLM fallback) and measure savings.

```python
# File: exercises/token_opt/ex3_model_router.py
from litellm import completion

MODELS = {
    "nano":     "gemini/gemini-2.0-flash",
    "mini":     "openai/gpt-4o-mini",
    "standard": "groq/llama-3.3-70b-versatile",
    "pro":      "anthropic/claude-3-5-sonnet",
}

COST_PER_1M = {
    "nano": 0.10, "mini": 0.30, "standard": 0.80, "pro": 12.00
}

def rule_based_route(query: str) -> str:
    """
    TODO 1: Return a tier name ("nano", "mini", "standard", "pro") based on:
    - Word count < 6 → "nano"
    - Contains any of: analyze, implement, debug, research, compare → "pro"
    - Contains any of: code, function, class, sql, script → "standard"
    - Default → "mini"
    """
    raise NotImplementedError


def llm_route(query: str) -> str:
    """
    TODO 2: Use the "nano" model to classify the query into a tier.
    System prompt must ask for exactly one word: nano / mini / standard / pro.
    Use max_tokens=5 to ensure a single-word response.
    Return MODELS[tier] — fall back to MODELS["mini"] if response is unexpected.
    """
    raise NotImplementedError


def routed_completion(query: str, use_llm_routing: bool = False) -> dict:
    """
    TODO 3: Call the LLM using the routed model and return:
    {
        "response": str,
        "model_used": str,
        "tier": str,
        "cost_per_1m": float,
    }
    """
    raise NotImplementedError


def simulate_savings(queries: list[str]) -> None:
    """
    TODO 4: For each query, get the routed tier and compare cost to always using "pro".
    Print: total queries, cost with routing, cost without routing, savings %.
    """
    raise NotImplementedError


# ── Test ───────────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "Hi",
    "What is 2 + 2?",
    "Summarize the French Revolution in 3 sentences.",
    "Write a Python function to parse a CSV file with error handling.",
    "Analyze the security vulnerabilities in this authentication flow.",
    "Compare transformer vs LSTM architectures for sequence modeling.",
    "Yes or no: is Python interpreted?",
]

if __name__ == "__main__":
    simulate_savings(TEST_QUERIES)
```

---

### Exercise 4 — Context Window Manager

**Goal**: Implement sliding window + summary compression, compare token usage.

```python
# File: exercises/token_opt/ex4_context_manager.py

def sliding_window(messages: list[dict], max_tokens: int = 3000,
                   always_keep_last: int = 4) -> list[dict]:
    """
    TODO 1: Return a trimmed messages list that:
    - Always keeps all system messages
    - Always keeps the last `always_keep_last` non-system messages
    - Trims oldest non-system messages until total tokens <= max_tokens
    Use count_messages_tokens() to measure.
    """
    raise NotImplementedError


def summarize_history(messages: list[dict], model: str = "gemini/gemini-2.0-flash",
                      keep_recent: int = 4) -> list[dict]:
    """
    TODO 2: Compress old messages into a summary:
    - Separate system messages from conversation
    - If conversation has <= keep_recent + 2 messages, return unchanged
    - Use the cheap model to summarize old messages (max_tokens=150)
    - Return: system_messages + [summary_message] + recent_messages
    """
    raise NotImplementedError


def compare_strategies(conversation: list[dict]) -> None:
    """
    TODO 3: Print a table showing token counts at each turn for:
    - No compression (raw history)
    - Sliding window (max_tokens=3000)
    - Summary compression (keep_recent=4)

    Format:
    Turn | Raw | Sliding Window | Summary
    -----|-----|----------------|--------
    1    | 120 | 120            | 120
    ...
    """
    raise NotImplementedError
```

---

### Exercise 5 — Full Cost Dashboard

**Goal**: Wrap all optimizations and build a cost report across 50 simulated queries.

```python
# File: exercises/token_opt/ex5_cost_dashboard.py
import time
from dataclasses import dataclass, field


@dataclass
class CallRecord:
    query: str
    model: str
    input_tokens: int
    output_tokens: int
    was_cached: bool
    latency_ms: float
    cost_usd: float


@dataclass
class Dashboard:
    records: list[CallRecord] = field(default_factory=list)

    # TODO 1: Add method total_cost() → float
    # TODO 2: Add method cache_hit_rate() → float (0.0 to 1.0)
    # TODO 3: Add method avg_latency_ms() → float
    # TODO 4: Add method cost_by_model() → dict[str, float]
    # TODO 5: Add method print_report() that prints a formatted summary table


def run_optimized_pipeline(queries: list[str]) -> Dashboard:
    """
    TODO 6: For each query:
    1. Check semantic cache → if hit, record with was_cached=True, cost=0
    2. Route to appropriate model
    3. Call LLM and record timing, tokens, cost
    4. Store in semantic cache for future calls
    5. Return Dashboard with all records
    """
    raise NotImplementedError


if __name__ == "__main__":
    import random
    base = [
        "What is machine learning?",
        "Explain gradient descent",
        "How does attention work?",
        "What is a transformer?",
        "Write a Python sort function",
    ]
    # Mix of new and repeated queries to test caching
    queries = base * 5 + [f"Variation {i}: {random.choice(base)}" for i in range(25)]
    random.shuffle(queries)

    dashboard = run_optimized_pipeline(queries)
    dashboard.print_report()
```

---

## 13. References and Resources

### 📄 Papers

| Paper | What it covers | Link |
|---|---|---|
| **Attention Is All You Need** (Vaswani et al., 2017) | Original transformer — foundation for understanding tokenization | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| **LLMLingua** (Jiang et al., 2023) | Learned prompt compression — 3–20× compression with <5% quality loss | [arxiv.org/abs/2310.05736](https://arxiv.org/abs/2310.05736) |
| **LLMLingua-2** (Pan et al., 2024) | Token-level compression via data distillation | [arxiv.org/abs/2403.12968](https://arxiv.org/abs/2403.12968) |
| **RECOMP** (Xu et al., 2023) | Compressive retrieval for RAG — compress retrieved docs before injection | [arxiv.org/abs/2310.04408](https://arxiv.org/abs/2310.04408) |
| **LongLLMLingua** (Jiang et al., 2023) | Key info extraction from long contexts for RAG | [arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825) |
| **Semantic Caching for LLMs** (Bang et al., 2023) | Vector similarity caching design and evaluation | [arxiv.org/abs/2304.10566](https://arxiv.org/abs/2304.10566) |

### 📚 Documentation

| Resource | Link |
|---|---|
| OpenAI Tokenizer (interactive) | [platform.openai.com/tokenizer](https://platform.openai.com/tokenizer) |
| tiktoken (Python tokenizer) | [github.com/openai/tiktoken](https://github.com/openai/tiktoken) |
| LiteLLM token_counter | [docs.litellm.ai/docs/completion/token_usage](https://docs.litellm.ai/docs/completion/token_usage) |
| LiteLLM cost tracking | [docs.litellm.ai/docs/completion/cost_estimation](https://docs.litellm.ai/docs/completion/cost_estimation) |
| Anthropic: long context best practices | [docs.anthropic.com/en/docs/build-with-claude/context-windows](https://docs.anthropic.com/en/docs/build-with-claude/context-windows) |
| OpenAI: prompt engineering guide | [platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering) |

### 🛠 Libraries

| Library | Purpose | Install |
|---|---|---|
| `tiktoken` | Fast OpenAI-compatible tokenizer | `pip install tiktoken` |
| `litellm` | Universal cost tracking + token counter | `pip install litellm` |
| `llmlingua` | Neural prompt compression (LLMLingua-2) | `pip install llmlingua` |
| `sentence-transformers` | Embeddings for semantic caching | `pip install sentence-transformers` |
| `langchain-core` | Token-aware splitters + callbacks | `pip install langchain-core` |
| `redis` | Production semantic cache backend | `pip install redis` |

### 🎓 Courses and Tutorials

| Resource | Format | Link |
|---|---|---|
| **DeepLearning.AI: Prompt Engineering for Developers** | Free course | [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) |
| **DeepLearning.AI: Building Systems with ChatGPT API** | Free course | [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) |
| **LiteLLM Cost Tracking Tutorial** | Docs + examples | [docs.litellm.ai](https://docs.litellm.ai/docs/proxy/cost_tracking) |
| **Chip Huyen: LLM Applications in Production** | Book chapter | [huyenchip.com/llmops](https://huyenchip.com/llmops/) |
| **Hamel Husain: Your AI Product Needs Evals** | Blog post | [hamel.dev/blog/posts/evals](https://hamel.dev/blog/posts/evals/) |

### 💰 Cost Monitoring Tools

| Tool | What it does | Link |
|---|---|---|
| **LangSmith** | Per-run cost tracking, token dashboards | [smith.langchain.com](https://smith.langchain.com) |
| **Helicone** | LLM observability + cost analytics (proxy) | [helicone.ai](https://helicone.ai) |
| **LiteLLM Proxy** | Centralized rate limiting + cost budgets | [docs.litellm.ai/docs/proxy](https://docs.litellm.ai/docs/proxy/quick_start) |
| **Portkey** | Cost guardrails + fallback routing | [portkey.ai](https://portkey.ai) |
| **OpenMeter** | Usage-based billing metering | [openmeter.io](https://openmeter.io) |

### 📊 Token Cost Calculators

- **LiteLLM cost calculator**: `python -c "from litellm import cost_per_token; print(cost_per_token('gpt-4o-mini', 1000, 500))"`
- **OpenAI pricing**: [openai.com/pricing](https://openai.com/pricing)
- **Anthropic pricing**: [anthropic.com/pricing](https://www.anthropic.com/pricing)
- **Together AI / Groq / Fireworks pricing**: compare at [artificialanalysis.ai](https://artificialanalysis.ai)

---

*Related guides in this repo:*
- *[`guide/07_cost_optimization.md`](guide/07_cost_optimization.md) — model routing, semantic caching, context trimming implementations*
- *[`framework_selection_guide.md`](framework_selection_guide.md) — choosing the right framework (also affects token efficiency)*
- *[`PRODUCTION_AGENT_GUIDE.md`](PRODUCTION_AGENT_GUIDE.md) — production deployment including cost monitoring*
