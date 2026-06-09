# Token Optimization — Comprehensive Strategy Guide

> **Companion to**: `exercises/ex5_token_optimizer.py`  
> **Goal**: Understand *why* and *how* to cut token usage across every layer of an agentic system without degrading answer quality.

---

## Table of Contents

1. [Why Token Optimization Matters](#1-why-token-optimization-matters)
2. [How Tokenization Works](#2-how-tokenization-works)
3. [Counting Tokens Accurately](#3-counting-tokens-accurately)
4. [System Prompt Optimization](#4-system-prompt-optimization)
5. [Prompt Compression Strategies](#5-prompt-compression-strategies)
6. [RAG Context Budgeting](#6-rag-context-budgeting)
7. [Tool Schema Compaction](#7-tool-schema-compaction)
8. [History & Conversation Management](#8-history--conversation-management)
9. [MCP / Tool-Response Compaction](#9-mcp--tool-response-compaction)
10. [Model Routing by Complexity](#10-model-routing-by-complexity)
11. [Prompt Caching (Anthropic / Google)](#11-prompt-caching-anthropic--google)
12. [Output Token Control](#12-output-token-control)
13. [Batching and Request-Level Savings](#13-batching-and-request-level-savings)
14. [Measuring Savings: Before/After Framework](#14-measuring-savings-beforeafter-framework)
15. [Production Checklist](#15-production-checklist)
16. [Further Reading](#16-further-reading)

---

## 1. Why Token Optimization Matters

### The Cost Equation

Every API call costs:

```
cost = (input_tokens × price_per_1k_in) + (output_tokens × price_per_1k_out)
```

| Model | Input (per 1K) | Output (per 1K) |
|---|---|---|
| GPT-4o | $0.0050 | $0.0150 |
| GPT-4o-mini | $0.0002 | $0.0006 |
| Gemini 2.0 Flash | $0.0001 | $0.0003 |
| Claude 3.5 Sonnet | $0.0030 | $0.0150 |
| Claude 3 Haiku | $0.0003 | $0.0015 |
| Groq Llama-3.3-70b | $0.0006 | $0.0008 |

**At scale, the math is brutal**: A 2,000-token context called 50,000×/day on GPT-4o = **$500/day** input alone. Cut that context by 60% → **$200/day saved**.

### The Latency Equation

Tokens don't just cost money — they add latency:
- More input tokens → longer TTFT (time-to-first-token) on streaming
- More output tokens → longer total response time
- Context window limits cap how much you can send at all (GPT-4o: 128K, Claude 3.5: 200K)

### The Quality Equation (the real constraint)

> **"The goal is not to minimize tokens — it is to maximize signal-to-noise ratio."**

Blindly cutting tokens degrades answers. The strategies below all maintain or improve quality by cutting *noise* rather than *signal*.

---

## 2. How Tokenization Works

### BPE (Byte-Pair Encoding) — the algorithm all major models use

BPE starts with characters and merges frequent pairs iteratively until a target vocabulary size is reached. The result:

- Common English words → 1 token (`the`, `and`, `is`)
- Rare words → multiple tokens (`tokenization` → `token` + `ization`)
- Code → often expensive (`function_name_with_underscores` → many tokens)
- Numbers → very expensive (each digit can be its own token)
- Non-English text → often 2–5× more tokens per word than English

### Encoding families

| Family | Models | Notes |
|---|---|---|
| `cl100k_base` | GPT-4, GPT-3.5, text-embedding-* | Default OpenAI |
| `o200k_base` | GPT-4o, GPT-4o-mini | ~30% more efficient on code |
| Gemini tokenizer | Gemini family | Not tiktoken-compatible |
| Claude tokenizer | Claude family | Anthropic internal, similar ratios |

**Practical rule**: When estimating cross-provider, tiktoken `cl100k_base` is a conservative upper bound.

### Token cost of common patterns

```python
# 1 token each (on cl100k):
"the", "and", "of", "to", "is"

# Multi-token words:
"tokenization"  → 3 tokens
"GDPR"          → 1 token (in vocabulary)
"UUID-4f2a..."  → many tokens (UUIDs are expensive)

# Structural overhead (JSON):
{"key": "value"}  → ~6 tokens for the brackets/colons/quotes

# Date formats:
"2024-01-15"    → 4–5 tokens (each number segment = token)
"Jan 15 2024"   → 4 tokens (cheaper!)

# Python code:
def foo():      → 4 tokens  
    pass        → 2 tokens
```

**Implication**: Replace UUIDs with short IDs in tool results. Use compact date formats. Prefer natural language for short enumerations.

---

## 3. Counting Tokens Accurately

### Using tiktoken

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    # Strip provider prefix (openai/, anthropic/, etc.)
    short = model.split("/")[-1]
    try:
        enc = tiktoken.encoding_for_model(short)
    except KeyError:
        # Unknown model — safe fallback
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
```

### Chat message overhead

The OpenAI chat format wraps each message in metadata tokens. The formula:

```
total = 2   # reply primer always added
      + sum(4 + len(message_tokens) for each message)
```

Where the `4` covers `<|im_start|>`, role, `\n\n`, `<|im_end|>`.

For tool calls, add the serialized tool call JSON tokens on top.

### When to count

- **Always count before sending** if you have a budget or routing decision
- **Count tool results** before injecting into context
- **Count RAG chunks** before assembling the final prompt

---

## 4. System Prompt Optimization

The system prompt is sent on **every single call**. A bloated 800-token system prompt costs more than a bloated 800-token user message, because the system prompt never benefits from conversation-level caching unless you use provider prompt caching (Section 11).

### Techniques

#### 4a. Deduplicate instructions
Bad:
```
Always be helpful. Always be concise. Always cite sources.
Be helpful and polite at all times. Cite your sources.
```
Good:
```
Be concise and helpful. Always cite sources.
```

#### 4b. Cut adjectives and hedging
Bad: `"You are an extremely helpful, professional, and knowledgeable AI assistant that always strives to provide accurate and comprehensive responses."`  
Good: `"You are a precise, citation-first compliance assistant."`

Saved: ~30 tokens, zero quality loss.

#### 4c. Use bullet lists over paragraphs

Bullet lists are both more token-efficient and better parsed by models:

```
# Bad (prose)
You should always answer in English. When the user asks about regulations,
cite the specific article number. Never speculate beyond what the documents say.

# Good (bullets)
Rules:
- Answer in English
- Cite regulation articles by number
- No speculation beyond source docs
```

#### 4d. Externalise static reference material

Don't embed reference tables or glossaries in the system prompt. Put them in RAG so they're only retrieved when needed.

#### 4e. Version and diff your system prompts

Track system prompt token counts in version control. A PR that adds 200 tokens to the system prompt should include a cost impact analysis.

---

## 5. Prompt Compression Strategies

### 5a. Extractive Compression (TF-IDF / Word Frequency)

Select the highest-information sentences without rewriting. Fast, deterministic, no LLM call required.

**Algorithm**:
1. Split text into sentences
2. Compute word frequency across all sentences
3. Score each sentence: `score = sum(freq[word] for word in sentence) / len(words)`
4. Greedy select top-scored sentences until token budget is hit
5. Re-sort selected sentences by original position to preserve flow

```python
def compress_text(text: str, target_tokens: int) -> str:
    sentences = split_sentences(text)
    freq = word_frequencies(sentences)
    ranked = sorted(sentences, key=lambda s: score(s, freq), reverse=True)
    
    kept, used = [], 0
    for i, sent in ranked:
        t = count_tokens(sent)
        if used + t <= target_tokens:
            kept.append((i, sent))
            used += t
    
    return " ".join(s for _, s in sorted(kept))
```

**When to use**: Compressing retrieved documents, tool results, MCP payloads, long user inputs.  
**Typical reduction**: 50–80% with ~5–10% quality loss.

### 5b. Abstractive Compression (LLM-based)

Ask a cheap, fast model to summarise the text before sending it to the expensive model.

```python
async def llm_compress(text: str, budget: int, model="gpt-4o-mini") -> str:
    resp = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "user", "content":
             f"Summarise this text in under {budget} tokens. Preserve key facts, "
             f"numbers, names, and dates:\n\n{text}"}
        ],
        max_tokens=budget + 50
    )
    return resp.choices[0].message.content
```

**When to use**: When extractive compression loses too much context (e.g., nuanced legal text). The cost of the mini-model summarization call is usually far less than the savings on the main model.  
**Cost model**: Spend $0.0002 on mini to save $0.005 on GPT-4o = 25× ROI.

### 5c. LLMLingua (Microsoft Research)

[LLMLingua](https://github.com/microsoft/LLMLingua) uses a small LM to score token importance and drop low-importance tokens at character level. More aggressive than sentence-level extractive compression.

```bash
pip install llmlingua
```

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
result = compressor.compress_prompt(text, ratio=0.5)  # 50% reduction
compressed = result["compressed_prompt"]
```

**Typical reduction**: 2–5× with minimal quality loss for retrieval context.  
**Caveat**: Adds a local model inference cost; best for high-throughput pipelines.

### 5d. Selective Compression (Hybrid)

Compress only the parts that are far from the query:

```python
def selective_compress(chunks: list[dict], query: str, budget: int) -> str:
    scored = [(semantic_score(c["text"], query), c) for c in chunks]
    result, used = [], 0
    for score, chunk in sorted(scored, reverse=True):
        t = count_tokens(chunk["text"])
        if used + t <= budget:
            # High-relevance chunk: include full text
            result.append(chunk["text"])
            used += t
        elif score > 0.5:
            # Medium-relevance: compress to 50%
            compressed = compress_text(chunk["text"], t // 2)
            result.append(compressed)
            used += count_tokens(compressed)
    return "\n".join(result)
```

---

## 6. RAG Context Budgeting

This is where the biggest real-world savings come from. Naïve RAG dumps all retrieved chunks into context. Budget RAG is disciplined.

### 6a. Greedy Fill by Relevance Score

```python
def budget_rag_context(chunks, token_budget=1500):
    sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
    selected, used = [], 0
    for chunk in sorted_chunks:
        t = count_tokens(chunk["text"])
        if used + t <= token_budget:
            selected.append(chunk)
            used += t
        elif t > token_budget * 0.4:
            # Large chunk just over budget: compress and include
            remaining = token_budget - used
            if remaining > 80:
                chunk["text"] = compress_text(chunk["text"], remaining)
                selected.append(chunk)
                used = token_budget  # budget exhausted
    return selected, used
```

**Key insight**: A relevance score of 0.85 chunk is worth more than four 0.60 chunks of the same total size. Sort by score, fill greedily.

### 6b. Rerank Before Budget

Retrieval (BM25/vector) returns candidates. Reranking (cross-encoder) scores relevance with the query in context. Only then budget:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(chunks, query, top_k=10):
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    for c, s in zip(chunks, scores):
        c["score"] = float(s)
    return sorted(chunks, key=lambda c: c["score"], reverse=True)[:top_k]
```

**Why this matters**: Vector search often returns off-topic chunks with high cosine similarity. A reranker catches this. Budget only the truly relevant chunks.

### 6c. Chunk Size Tuning

| Chunk size | Tokens | Best for |
|---|---|---|
| 128 tokens | ~100 words | Precise factual Q&A |
| 256 tokens | ~200 words | General Q&A |
| 512 tokens | ~400 words | Reasoning over paragraphs |
| 1024+ tokens | ~800+ words | Summarisation tasks |

Smaller chunks = better precision but more retrieval calls. Use 256 tokens for most agentic RAG.

### 6d. Parent-Child Chunking

Index small chunks (128 tokens) for retrieval, but inject the parent chunk (512 tokens) into context. Better recall, controlled context size.

```python
# At index time:
child_chunks = split(doc, size=128)
parent_chunk = doc  # or 512-token window

for child in child_chunks:
    child["parent_id"] = doc["id"]
    index(child)

# At query time:
hits = retrieve(query, k=20)
unique_parents = deduplicate([lookup_parent(h["parent_id"]) for h in hits])
selected = budget_rag_context(unique_parents, token_budget=1500)
```

### 6e. Compact Context Format

Every formatting choice costs tokens:

```
# Verbose XML (avoid):
<document>
  <title>GDPR Article 28</title>
  <content>
    The controller and processor shall...
  </content>
</document>

# Compact inline (use):
[1] gdpr_art28.pdf: The controller and processor shall...
```

Savings: ~25 tokens per chunk just from format.

---

## 7. Tool Schema Compaction

Tool schemas are injected on every agentic call — even when the tool isn't called. This is pure overhead for simple turns.

### 7a. First-Sentence Description (≤80 chars)

```python
# Before: 180 tokens for schema
"description": "This powerful search tool allows you to search through our comprehensive 
database of regulatory documents including GDPR, CCPA, HIPAA, and other compliance 
frameworks. It supports full-text search, metadata filtering, and semantic search. 
You can filter by jurisdiction, date range, article number, and document type."

# After: 12 tokens for description
"description": "Search regulatory compliance documents by keyword or article number."
```

### 7b. Remove Noise Fields

Strip these from every property definition:
- `"examples"` — demonstrates usage but tokens are expensive; move to system prompt
- `"default"` — models don't need defaults listed; they'll omit optional params
- `"title"` — redundant with property name
- `"$schema"` — developer tooling only

```python
def compact_tool_schemas(tools):
    for tool in copy.deepcopy(tools):
        for prop in tool["function"]["parameters"]["properties"].values():
            for key in ["examples", "default", "title", "$schema"]:
                prop.pop(key, None)
    return tools
```

### 7c. Abbreviate Long Enums

```python
# Before (lists all 20 options):
"enum": ["gdpr", "ccpa", "hipaa", "soc2", "pci-dss", "iso27001", "nist", ...]

# After:
"enum": ["gdpr", "ccpa", "hipaa"],
"description": "Regulation type (gdpr, ccpa, hipaa, and 17 more)"
```

### 7d. Dynamic Tool Loading

Only inject tools relevant to the current turn. Classify the query first, then select tools:

```python
ALL_TOOLS = {
    "search": search_tool_schema,
    "calculate": calc_tool_schema,
    "write_file": write_tool_schema,
    "send_email": email_tool_schema,
}

def select_tools(query: str) -> list[dict]:
    if any(kw in query.lower() for kw in ["search", "find", "look up", "what is"]):
        return [ALL_TOOLS["search"]]
    if any(kw in query.lower() for kw in ["calculate", "compute", "how much"]):
        return [ALL_TOOLS["calculate"], ALL_TOOLS["search"]]
    return list(ALL_TOOLS.values())  # complex query: all tools
```

**Savings**: Dropping 3 of 4 tool schemas can save 200–400 tokens per call.

### 7e. Tool-Free Turns

Not every turn needs tools. Classify first:

```python
def needs_tools(query: str, history: list) -> bool:
    # Pure conversational / factual from context
    conversational_patterns = [
        r"^(thanks|ok|got it|understood|sure)",
        r"^what did (you|i) (say|mean)",
        r"^(yes|no|maybe)$",
    ]
    return not any(re.match(p, query.lower()) for p in conversational_patterns)
```

---

## 8. History & Conversation Management

Chat history grows linearly with the conversation length. Without management, a 50-turn conversation balloons to 5,000+ tokens of history.

### 8a. Sliding Window

Keep only the last N turn-pairs. Fast, zero LLM cost, but loses early context:

```python
def sliding_window(messages, keep_last_n=4):
    system = [m for m in messages if m["role"] == "system"]
    others = [m for m in messages if m["role"] != "system"]
    return system + others[-(keep_last_n * 2):]
```

**When to use**: Conversational assistants where recent context matters most.

### 8b. LLM Summarisation (Episodic Memory)

Summarise older turns with a cheap model and inject as a system message:

```python
async def compress_history(old_turns, model="gpt-4o-mini"):
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old_turns)
    resp = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content":
            f"Summarise in 3 bullet points:\n{transcript}"}],
        max_tokens=150,
    )
    summary = resp.choices[0].message.content
    return {"role": "system", "content": f"[Past conversation]\n{summary}"}
```

**When to use**: Long research or planning sessions where early decisions matter.

### 8c. Key-Value Memory (Entity Extraction)

Instead of summarising turns, extract entities and store them:

```python
MEMORY = {}

async def extract_and_store(turn_pair):
    resp = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"Extract key facts as JSON {{entity: value}}:\n{turn_pair}"}],
        max_tokens=100
    )
    facts = json.loads(resp.choices[0].message.content)
    MEMORY.update(facts)

def inject_memory(query):
    relevant = {k: v for k, v in MEMORY.items()
                if k.lower() in query.lower()}
    if relevant:
        return f"[Known facts: {json.dumps(relevant)}]"
    return ""
```

**When to use**: Task-focused agents (code review, research) that accumulate facts over many turns.

### 8d. Adaptive Compression (Token-Budget Driven)

Compress based on current token count, not a fixed number of turns:

```python
async def adaptive_history(messages, max_tokens=2000, keep_last_n=4):
    while count_messages_tokens(messages) > max_tokens:
        system = [m for m in messages if m["role"] == "system"]
        non_sys = [m for m in messages if m["role"] != "system"]
        recent = non_sys[-(keep_last_n * 2):]
        older = non_sys[:-(keep_last_n * 2)]
        if not older:
            break  # Can't compress further without losing recent context
        summary = await compress_history(older)
        messages = system + [summary] + recent
    return messages
```

### 8e. Token Budget Per History Source

Allocate tokens across context sources before assembling:

```
Total budget:     8,000 tokens
├── System prompt:   400 tokens (fixed)
├── History:       2,000 tokens (managed)
├── RAG context:   2,000 tokens (budgeted)
├── Tool schemas:    400 tokens (compacted)
├── User message:    200 tokens (this turn)
└── Output reserve: 3,000 tokens (for answer)
```

---

## 9. MCP / Tool-Response Compaction

Tool responses can be enormous — a search API might return 50 results with full metadata. Almost all of it is noise.

### 9a. Strip Noise Fields

```python
NOISE_KEYS = {
    "_meta", "metadata", "request_id", "trace_id", "timing",
    "version", "api_version", "rate_limit_info", "x_request_id",
    "links", "_links", "href", "self", "etag", "last_modified",
}

def compact_mcp(response: dict) -> dict:
    return {
        k: compact_mcp(v) if isinstance(v, dict) else v
        for k, v in response.items()
        if k not in NOISE_KEYS
    }
```

### 9b. Truncate Long Lists

```python
def truncate_list(lst, max_items=10):
    if len(lst) <= max_items:
        return lst
    return lst[:max_items] + [{"_truncated": True, "_total": len(lst)}]
```

**Why 10?** Models read lists top-to-bottom. Items 11+ are rarely cited and almost never the correct answer. If they are, the query was too broad and should be refined.

### 9c. Whitelist Only What the Agent Needs

For known APIs, specify exactly which fields to keep:

```python
SEARCH_ESSENTIAL = ["results", "total", "query"]
RESULT_ESSENTIAL = ["title", "url", "snippet", "score"]

def compact_search_response(resp):
    clean = {k: resp[k] for k in SEARCH_ESSENTIAL if k in resp}
    clean["results"] = [
        {k: r[k] for k in RESULT_ESSENTIAL if k in r}
        for r in clean.get("results", [])[:5]
    ]
    return clean
```

### 9d. Hierarchical Compaction

For nested responses, apply compaction at each level:

```python
def deep_compact(obj, max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return str(obj)[:100] + "..."
    if isinstance(obj, dict):
        return {k: deep_compact(v, max_depth, current_depth + 1)
                for k, v in obj.items() if k not in NOISE_KEYS}
    if isinstance(obj, list):
        return [deep_compact(i, max_depth, current_depth + 1)
                for i in obj[:10]]
    return obj
```

---

## 10. Model Routing by Complexity

Not every query needs GPT-4o. Routing simple queries to cheaper models is the highest-ROI optimization after RAG budgeting.

### Routing Tiers

| Tier | Queries | Models | Relative cost |
|---|---|---|---|
| **Simple** | Factual Q&A, short answers, conversational | gpt-4o-mini, Haiku, Gemini Flash | 1× |
| **Medium** | Multi-step reasoning, tool use, ~1K context | gpt-4o-mini, Gemini Pro | 2–5× |
| **Complex** | Long context, chain-of-thought, code gen | gpt-4o, Claude 3.5 Sonnet | 25–75× |

### Heuristic Classifier

```python
def classify_complexity(query: str, context_tokens: int = 0) -> str:
    # Rule 1: Large context → complex
    if context_tokens > 3000:
        return "complex"

    # Rule 2: Query length
    words = query.split()
    if len(words) > 40:
        return "complex"
    if len(words) < 15 and context_tokens < 500:
        return "simple"

    # Rule 3: Complexity signal words
    complex_signals = {
        "compare", "contrast", "analyse", "analyze", "evaluate",
        "pros and cons", "step by step", "explain why", "trade-off"
    }
    if any(kw in query.lower() for kw in complex_signals):
        return "complex" if context_tokens > 500 else "medium"

    # Rule 4: Code content
    if re.search(r"```|def |class |SELECT |FROM ", query):
        return "medium"

    return "medium"
```

### LLM-based Routing (Meta-classifier)

For higher accuracy, use a tiny model to classify:

```python
ROUTING_PROMPT = """Classify query complexity for LLM routing.
Reply with exactly one word: simple, medium, or complex.

simple: greeting, factual lookup, yes/no, 1 sentence answer
medium: multi-step, tool use, code help, explanation
complex: long analysis, compare/contrast, chain-of-thought, large context

Query: {query}
Context tokens: {context_tokens}"""

async def llm_classify(query, context_tokens):
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content":
            ROUTING_PROMPT.format(query=query, context_tokens=context_tokens)}],
        max_tokens=5,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip().lower()
```

**Meta-routing ROI**: The classification call costs ~50 tokens × $0.0002/1K = $0.00001. If it routes even 30% of queries from GPT-4o to mini, savings are 30–75×.

### Cascade Routing

Start with a cheap model; retry with a powerful model if confidence is low:

```python
async def cascade_call(messages, tools=None):
    # Attempt 1: cheap model
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=messages, tools=tools, max_tokens=500)
    
    if is_confident(resp):
        return resp
    
    # Fallback: powerful model
    return await litellm.acompletion(
        model="openai/gpt-4o",
        messages=messages, tools=tools, max_tokens=1000)

def is_confident(resp) -> bool:
    content = resp.choices[0].message.content or ""
    uncertain = ["I'm not sure", "I don't know", "cannot determine",
                 "unclear", "insufficient information"]
    return not any(p in content.lower() for p in uncertain)
```

---

## 11. Prompt Caching (Anthropic / Google)

Provider-level caching lets you reuse the same prefix computation across calls at a fraction of the cost.

### Anthropic Prompt Caching

```python
import anthropic

client = anthropic.Anthropic()

# Mark the static prefix for caching
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a compliance assistant.",
        },
        {
            "type": "text",
            "text": "<entire_legal_corpus>...(50K tokens)...</entire_legal_corpus>",
            "cache_control": {"type": "ephemeral"},  # ← cache this prefix
        }
    ],
    messages=[{"role": "user", "content": user_query}]
)
```

**Economics**:
- Cache write: 1.25× normal input cost
- Cache read: 0.1× normal input cost (**10× cheaper**)
- Cache TTL: 5 minutes (refreshed on hit)

**Best for**: Long system prompts, large document corpora, repeated instructions.

### Google Gemini Context Caching

```python
import google.generativeai as genai

# Create a cached content object
cached = genai.caching.CachedContent.create(
    model="gemini-2.0-flash",
    contents=[large_document],
    ttl=datetime.timedelta(minutes=60),
)

# Use the cache in subsequent calls
model = genai.GenerativeModel.from_cached_content(cached)
response = model.generate_content("Summarise the key points.")
```

**Economics**: ~4× cheaper reads, 1-hour TTL.

### When Caching Helps Most

| Scenario | Savings |
|---|---|
| Same large system prompt, many users | Cache writes amortised across all calls |
| Document Q&A (same doc, many questions) | 90%+ input cost reduction after first call |
| Retrieval corpus that rarely changes | High hit rate → near-zero input costs |
| Every call has unique context | Low hit rate → not worth it |

---

## 12. Output Token Control

Output tokens are 2–3× more expensive than input tokens (per-token). Control them explicitly.

### 12a. Set max_tokens Appropriately

```python
# Bad: let the model ramble
response = await litellm.acompletion(model=model, messages=messages)

# Good: set a tight budget
response = await litellm.acompletion(
    model=model,
    messages=messages,
    max_tokens=300,  # based on expected answer length
    temperature=0.2,  # lower temp → shorter, more focused answers
)
```

### 12b. Instruct the Model to Be Concise

```
# System prompt addition:
"Answer in 2–3 sentences unless the user asks for detail.
 No preamble. No 'Great question!' Never repeat the question back."
```

Models trained on RLHF tend toward verbose, padding-heavy answers. Explicit instructions counter this.

### 12c. Structured Output (JSON Mode)

Structured outputs skip the prose and give you exactly the data you need:

```python
from pydantic import BaseModel
import instructor

client = instructor.from_litellm(litellm.completion)

class ComplianceCheck(BaseModel):
    compliant: bool
    violated_articles: list[str]
    recommendation: str  # max 1 sentence

result = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages,
    response_model=ComplianceCheck,
    max_tokens=150,  # JSON is compact
)
```

**Savings**: A structured 50-token JSON answer vs. a 300-token prose answer = 6× output reduction.

### 12d. Few-Shot Examples of Concise Answers

```python
messages = [
    {"role": "system", "content": "Answer compliance questions concisely."},
    {"role": "user", "content": "Does GDPR require a DPA?"},
    {"role": "assistant", "content": "Yes. Article 28 GDPR requires a Data Processing Agreement when a controller uses a processor."},  # ← 20 tokens, not 200
    {"role": "user", "content": actual_question},
]
```

---

## 13. Batching and Request-Level Savings

### 13a. Batch API (OpenAI)

Process non-time-sensitive requests in batches at **50% cost**:

```python
import openai

client = openai.OpenAI()

# Create a JSONL batch file
tasks = [
    {"custom_id": f"req-{i}", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "gpt-4o-mini", "messages": msg, "max_tokens": 200}}
    for i, msg in enumerate(all_messages)
]

batch_file = client.files.create(
    file=("\n".join(json.dumps(t) for t in tasks)).encode(),
    purpose="batch"
)

batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
```

**When to use**: Evaluation pipelines, nightly summaries, document processing, anything not user-facing.

### 13b. Parallel Tool Calls

Instead of sequential tool calls (each adding a round-trip), let the model call multiple tools at once:

```python
# With parallel_tool_calls=True (default in OpenAI):
response = await litellm.acompletion(
    model="openai/gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    parallel_tool_calls=True,  # model can call multiple tools in one turn
)
```

**Savings**: Reduces 3 sequential API calls to 1 call + 1 result-injection call = 2 vs 6 total round-trips.

### 13c. Request Deduplication

Cache identical or near-identical requests:

```python
import hashlib, functools

@functools.lru_cache(maxsize=1000)
def cached_llm_call(prompt_hash: str, model: str):
    # Not directly useful — need to store actual prompt
    pass

async def deduplicated_call(messages, model, ttl=300):
    key = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
    if key in CACHE and time.time() - CACHE[key]["ts"] < ttl:
        return CACHE[key]["response"]
    resp = await litellm.acompletion(model=model, messages=messages)
    CACHE[key] = {"response": resp, "ts": time.time()}
    return resp
```

---

## 14. Measuring Savings: Before/After Framework

You can't optimize what you don't measure. Build a savings report into every optimized call.

### The Report Structure

```python
def savings_report(original_tokens, optimized_tokens, output_tokens, model):
    orig_cost = estimate_cost(original_tokens, output_tokens, model)
    opt_cost  = estimate_cost(optimized_tokens, output_tokens, model)
    reduction = (original_tokens - optimized_tokens) / original_tokens

    return {
        "token_reduction_pct": f"{reduction:.0%}",
        "tokens_saved":        original_tokens - optimized_tokens,
        "cost_saved_per_call": orig_cost - opt_cost,
        "cost_saved_per_10k":  (orig_cost - opt_cost) * 10_000,
        "original_tokens":     original_tokens,
        "optimized_tokens":    optimized_tokens,
        "model":               model,
    }
```

### Attribution by Technique

Track which technique saved the most:

```python
savings_by_technique = {
    "system_prompt":   baseline_tokens - after_system_opt,
    "history_mgmt":    after_system_opt - after_history,
    "rag_budgeting":   after_history - after_rag,
    "tool_compaction": after_rag - after_tools,
    "mcp_compaction":  after_tools - final_tokens,
}
```

### Daily Savings Dashboard

```python
# Track in a simple JSON file, aggregate daily
def log_call_savings(report: dict):
    today = datetime.date.today().isoformat()
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "tokens_saved": report["tokens_saved"],
        "cost_saved": report["cost_saved_per_call"],
        "model": report["model"],
    }
    with open(f"savings_{today}.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

---

## 15. Production Checklist

Run through this before deploying any agent to production:

### Token Budget Setup
- [ ] Count tokens on every input before sending
- [ ] Set explicit `max_tokens` on every call — never let the model decide
- [ ] Allocate a total context budget and divide it across sources (system, history, RAG, tools, output)
- [ ] Alert when any single component exceeds its allocation

### System Prompt
- [ ] Under 400 tokens unless caching is enabled
- [ ] No duplicate instructions
- [ ] Explicit conciseness instruction ("2–3 sentences unless asked for more")
- [ ] Versioned in git with token count tracked

### RAG
- [ ] Relevance scores attached to all chunks
- [ ] Token budget enforced before assembling context
- [ ] Reranker applied before budgeting
- [ ] Compact context format (`[n] source: text`, not XML)
- [ ] Chunk size tuned to 256 tokens for precision Q&A

### Tool Schemas
- [ ] Descriptions ≤80 chars (first sentence only)
- [ ] `examples`, `default`, `title` stripped from all property schemas
- [ ] Long enums abbreviated to top 3–5 with a count note
- [ ] Dynamic tool selection based on query intent
- [ ] Tool-free turns detected and handled without injecting schemas

### History Management
- [ ] Sliding window set (default: last 4 turn-pairs)
- [ ] LLM summarisation on overflow (using cheap model)
- [ ] System messages always preserved
- [ ] Total history budget enforced (e.g., 2,000 tokens)

### Model Routing
- [ ] Complexity classifier in place (heuristic or LLM-based)
- [ ] Simple queries routed to cheapest model
- [ ] Cascade routing for uncertain confidence
- [ ] Cost-per-model tracked per day

### Measurement
- [ ] Savings report generated on every call
- [ ] Token savings attributed by technique
- [ ] Daily cost dashboard or alert
- [ ] Weekly review of per-technique savings to prioritise further work

### Caching
- [ ] Prompt caching enabled if static prefix > 1,000 tokens
- [ ] Response caching for deterministic queries (TTL: 5 min)
- [ ] Batch API used for all offline/async workloads

---

## 16. Further Reading

### Official Documentation
- **OpenAI Tokenizer (interactive)**: https://platform.openai.com/tokenizer
- **tiktoken (Python library)**: https://github.com/openai/tiktoken
- **Anthropic Prompt Caching**: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- **Google Gemini Context Caching**: https://ai.google.dev/gemini-api/docs/caching
- **OpenAI Batch API**: https://platform.openai.com/docs/guides/batch
- **OpenAI Parallel Tool Calls**: https://platform.openai.com/docs/guides/function-calling#parallel-function-calling
- **LiteLLM Cost Tracking**: https://docs.litellm.ai/docs/completion/token_usage

### Research Papers
- **LLMLingua: Compressing Prompts for Accelerated Inference** (Microsoft Research, 2023): https://arxiv.org/abs/2310.05736
- **LLMLingua-2** (2024, improved compression): https://arxiv.org/abs/2403.12968
- **Selective Context: Reducing LLM Token Usage** (2023): https://arxiv.org/abs/2304.12102
- **KV Cache Compression (H2O)**: https://arxiv.org/abs/2306.14048
- **RECOMP: Improving Retrieval via Abstractive Compression** (2023): https://arxiv.org/abs/2310.04408

### Tools and Libraries
- **LLMLingua** (token-level prompt compression): https://github.com/microsoft/LLMLingua
- **instructor** (structured outputs = fewer output tokens): https://python.useinstructor.com/
- **Ragatouille** (ColBERT reranking for RAG): https://github.com/bclavie/RAGatouille
- **Rerankers** (lightweight reranker library): https://github.com/answerdotai/rerankers
- **Helicone** (production token usage analytics): https://www.helicone.ai/
- **LangSmith** (trace-level cost attribution): https://smith.langchain.com/
- **mem0** (agent memory / entity extraction): https://github.com/mem0ai/mem0

### Blog Posts and Guides
- **Lilian Weng — Reducing LLM Hallucination** (covers context quality): https://lilianweng.github.io/posts/2024-02-05-human-data-quality/
- **Anthropic — Prompt Engineering Guide**: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- **Building RAG with Budget Control** (Pinecone blog): https://www.pinecone.io/learn/chunking-strategies/
- **Token Efficiency in Production LLM Apps**: https://hamel.dev/blog/posts/evals/
- **The Economics of LLMs** (Simon Willison): https://simonwillison.net/tags/llmprices/

### Courses
- **DeepLearning.AI — Building Systems with the ChatGPT API**: https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/
- **DeepLearning.AI — Advanced Retrieval for AI with Chroma**: https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/

---

*Last updated: June 2026 — prices and model capabilities change frequently. Always verify current pricing on the provider's official pricing page.*
