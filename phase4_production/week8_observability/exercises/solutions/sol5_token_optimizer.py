"""
sol5_token_optimizer.py — Reference solution for ex5_token_optimizer.py

Full-stack token optimization covering:
  1. tiktoken-based counting (per-model)
  2. TF-IDF extractive prompt compression
  3. RAG context budgeting with chunk-level trimming
  4. Tool schema compaction (5 techniques)
  5. Sliding-window history + LLM summarisation
  6. MCP response compaction (noise field removal + list truncation)
  7. Heuristic complexity routing
  8. Cost measurement and savings reporting
  9. Full optimised agent call wiring all techniques
"""
from __future__ import annotations
import copy, json, math, os, re, string
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()

COST_TABLE: dict[str, tuple[float, float]] = {
    "openai/gpt-4o":                (0.0050, 0.0150),
    "openai/gpt-4o-mini":           (0.0002, 0.0006),
    "openai/gpt-3.5-turbo":         (0.0005, 0.0015),
    "gemini/gemini-2.0-flash":      (0.0001, 0.0003),
    "groq/llama-3.3-70b-versatile": (0.0006, 0.0008),
}
DEFAULT_MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")

# ── Noise fields always stripped from MCP responses ───────────────────────────
_MCP_NOISE_KEYS = {
    "_meta", "metadata", "request_id", "trace_id", "timing", "version",
    "api_version", "rate_limit_info", "x_request_id", "links", "_links",
    "href", "self", "etag", "last_modified",
}

# ── Complexity signal words ────────────────────────────────────────────────────
_COMPLEX_WORDS = {
    "and then", "after that", "step by step", "compare", "contrast",
    "analyse", "analyze", "evaluate", "pros and cons", "trade-off",
    "trade off", "explain why", "break down", "walk me through",
}
_SIMPLE_BLOCKERS = {"and then", "compare", "contrast", "analyse", "analyze",
                    "evaluate", "explain why", "step by step"}


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Token Counting
# ══════════════════════════════════════════════════════════════════════════════

def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    import tiktoken  # type: ignore
    short = model.split("/")[-1]
    try:
        enc = tiktoken.encoding_for_model(short)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict], model: str = DEFAULT_MODEL) -> int:
    total = 2  # reply primer
    for m in messages:
        total += 4 + count_tokens(m.get("content") or "", model)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Prompt Compression (TF-IDF extractive)
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= 5]


def _word_freq(sentences: list[str]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for s in sentences:
        for word in s.lower().translate(str.maketrans("", "", string.punctuation)).split():
            freq[word] = freq.get(word, 0) + 1
    return freq


def compress_text(text: str, target_tokens: int, model: str = DEFAULT_MODEL) -> str:
    if count_tokens(text, model) <= target_tokens:
        return text

    sentences = _split_sentences(text)
    if not sentences:
        # No clean sentences — hard truncate
        enc_text = text
        while count_tokens(enc_text, model) > target_tokens and len(enc_text) > 20:
            enc_text = enc_text[:int(len(enc_text) * 0.85)]
        return enc_text

    freq = _word_freq(sentences)

    def score(s: str) -> float:
        words = s.lower().translate(str.maketrans("", "", string.punctuation)).split()
        return sum(freq.get(w, 0) for w in words) / max(len(words), 1)

    ranked = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)

    kept: list[tuple[int, str]] = []
    used = 0
    for orig_idx, sent in ranked:
        t = count_tokens(sent, model)
        if used + t <= target_tokens:
            kept.append((orig_idx, sent))
            used += t

    # Restore original reading order
    kept.sort(key=lambda x: x[0])
    return " ".join(s for _, s in kept)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — RAG Context Budgeting
# ══════════════════════════════════════════════════════════════════════════════

def budget_rag_context(
    chunks: list[dict],
    query: str,
    token_budget: int = 1500,
    model: str = DEFAULT_MODEL,
) -> tuple[list[dict], int]:
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
    selected: list[dict] = []
    used = 0
    for chunk in sorted_chunks:
        text = chunk["text"]
        t = count_tokens(text, model)
        if used + t <= token_budget:
            selected.append(chunk)
            used += t
        elif t > token_budget * 0.4:
            remaining = token_budget - used
            if remaining > 50:
                compressed = compress_text(text, remaining, model)
                ct = count_tokens(compressed, model)
                if ct + used <= token_budget:
                    selected.append({**chunk, "text": compressed})
                    used += ct
    return selected, used


def format_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = [f"[{i+1}] {c.get('source', 'doc')}: {c['text']}" for i, c in enumerate(chunks)]
    return "Context:\n" + "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Tool Schema Compaction
# ══════════════════════════════════════════════════════════════════════════════

def compact_tool_schemas(tools: list[dict]) -> list[dict]:
    result = copy.deepcopy(tools)
    for tool in result:
        fn = tool.get("function", {})

        # 4a. Truncate description to first sentence, max 80 chars
        desc = fn.get("description", "")
        first_sent = re.split(r"(?<=[.!?])\s", desc)[0]
        fn["description"] = first_sent[:80]

        params = fn.get("parameters", {})
        props = params.get("properties", {})

        for prop_name, prop in props.items():
            # 4b. Remove "examples" keys
            prop.pop("examples", None)
            # 4c. Remove "default" from required params
            if prop_name in params.get("required", []):
                prop.pop("default", None)
            # 4d. Abbreviate long enums
            enum = prop.get("enum", [])
            if len(enum) > 5:
                prop["enum"] = enum[:3]
                prop["description"] = (prop.get("description", "") + f" (and {len(enum)-3} more)")[:120]
            # 4e. Remove "title" keys
            prop.pop("title", None)

        # Remove title from params level too
        params.pop("title", None)

    return result


def measure_schema_tokens(tools: list[dict], model: str = DEFAULT_MODEL) -> int:
    return count_tokens(json.dumps(tools), model)


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Context History Management
# ══════════════════════════════════════════════════════════════════════════════

async def summarise_history(
    old_messages: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in old_messages
    )
    resp = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise summariser."},
            {"role": "user", "content":
             f"Summarise this conversation in 3 bullet points. Be concise.\n\n{transcript}"},
        ],
        max_tokens=150,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


async def manage_history(
    messages: list[dict],
    max_tokens: int = 2000,
    keep_last_n: int = 4,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    if count_messages_tokens(messages, model) <= max_tokens:
        return messages

    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system  = [m for m in messages if m["role"] != "system"]

    if len(non_system) <= keep_last_n * 2:
        # Nothing old to summarise — drop the oldest non-system message
        trimmed = system_msgs + non_system[1:]
        return await manage_history(trimmed, max_tokens, keep_last_n, model)

    # Keep most recent keep_last_n pairs (2× messages)
    recent = non_system[-(keep_last_n * 2):]
    older  = non_system[:-(keep_last_n * 2)]

    summary = await summarise_history(older, model)
    summary_msg = {"role": "system", "content": f"[Conversation summary]\n{summary}"}

    trimmed = system_msgs + [summary_msg] + recent
    # Recurse if still over budget (handles very long sessions)
    if count_messages_tokens(trimmed, model) > max_tokens:
        return await manage_history(trimmed, max_tokens, keep_last_n, model)
    return trimmed


# ══════════════════════════════════════════════════════════════════════════════
# 6 — MCP Response Compaction
# ══════════════════════════════════════════════════════════════════════════════

def compact_mcp_response(
    mcp_response: dict,
    essential_keys: list[str] | None = None,
    _max_list: int = 10,
) -> dict:
    if essential_keys:
        return {k: mcp_response[k] for k in essential_keys if k in mcp_response}

    result: dict = {}
    for k, v in mcp_response.items():
        if k in _MCP_NOISE_KEYS:
            continue
        if isinstance(v, dict):
            result[k] = compact_mcp_response(v, _max_list=_max_list)
        elif isinstance(v, list):
            if len(v) > _max_list:
                result[k] = v[:_max_list] + [{"_truncated": True, "_total": len(v)}]
            else:
                result[k] = v
        else:
            result[k] = v
    return result


def format_tool_result(
    tool_name: str, result: Any,
    max_tokens: int = 400,
    model: str = DEFAULT_MODEL,
) -> str:
    if isinstance(result, dict):
        result = compact_mcp_response(result)
    json_str = json.dumps(result, default=str)
    if count_tokens(json_str, model) > max_tokens:
        json_str = compress_text(json_str, max_tokens, model)
    return f"Tool '{tool_name}' returned:\n{json_str}"


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Model Routing
# ══════════════════════════════════════════════════════════════════════════════

def classify_query_complexity(query: str, context_tokens: int = 0) -> str:
    q = query.lower()
    has_complex = any(kw in q for kw in _COMPLEX_WORDS)
    has_code = bool(re.search(r"```|def |class |import ", query))
    word_count = len(query.split())

    if context_tokens > 3000:
        return "complex"
    if word_count > 40:
        return "complex"
    if has_complex and context_tokens > 1000:
        return "complex"

    if (word_count < 15
            and context_tokens < 500
            and not any(kw in q for kw in _SIMPLE_BLOCKERS)
            and not has_code):
        return "simple"

    return "medium"


def route_model(
    query: str,
    context_tokens: int = 0,
    simple_model:  str = "openai/gpt-4o-mini",
    medium_model:  str = "openai/gpt-4o-mini",
    complex_model: str = "openai/gpt-4o",
) -> str:
    complexity = classify_query_complexity(query, context_tokens)
    model_map = {"simple": simple_model, "medium": medium_model, "complex": complex_model}
    chosen = model_map[complexity]
    print(f"  [router] complexity={complexity} → {chosen}")
    return chosen


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Cost Measurement
# ══════════════════════════════════════════════════════════════════════════════

def estimate_cost(input_tokens: int, output_tokens: int, model: str = DEFAULT_MODEL) -> float:
    in_rate, out_rate = COST_TABLE.get(model, (0.0002, 0.0006))
    return (input_tokens / 1000 * in_rate) + (output_tokens / 1000 * out_rate)


def savings_report(
    original_input_tokens: int,
    optimized_input_tokens: int,
    output_tokens: int,
    model: str = DEFAULT_MODEL,
) -> dict:
    orig_cost = estimate_cost(original_input_tokens, output_tokens, model)
    opt_cost  = estimate_cost(optimized_input_tokens, output_tokens, model)
    saved_t   = original_input_tokens - optimized_input_tokens
    saved_pct = saved_t / max(original_input_tokens, 1) * 100
    cost_saved = orig_cost - opt_cost

    summary = (
        f"Tokens: {original_input_tokens}→{optimized_input_tokens} "
        f"(-{saved_pct:.0f}% / -{saved_t} tokens) | "
        f"Cost: ${orig_cost:.5f}→${opt_cost:.5f} (saved ${cost_saved:.5f})"
    )
    return {
        "original_tokens":    original_input_tokens,
        "optimized_tokens":   optimized_input_tokens,
        "tokens_saved":       saved_t,
        "savings_pct":        round(saved_pct, 1),
        "original_cost_usd":  round(orig_cost,  6),
        "optimized_cost_usd": round(opt_cost,   6),
        "cost_saved_usd":     round(cost_saved, 6),
        "model":              model,
        "summary":            summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9 — Full Optimised Agent Call
# ══════════════════════════════════════════════════════════════════════════════

async def optimised_agent_call(
    question: str,
    history: list[dict],
    rag_chunks: list[dict],
    tools: list[dict],
    rag_budget_tokens: int = 1500,
    history_max_tokens: int = 2000,
    model: str | None = None,
) -> dict:
    # 9a. RAG budgeting
    selected_chunks, rag_tokens = budget_rag_context(rag_chunks, question, rag_budget_tokens)
    rag_context = format_rag_context(selected_chunks)

    # 9b. History management
    trimmed_history = await manage_history(history, history_max_tokens)

    # 9c. Tool compaction
    compacted_tools = compact_tool_schemas(tools) if tools else []

    # 9d. Total context count for routing
    context_token_count = (count_messages_tokens(trimmed_history) +
                           count_tokens(rag_context) +
                           (measure_schema_tokens(compacted_tools) if compacted_tools else 0))

    # 9e. Model routing
    chosen_model = model or route_model(question, context_token_count)

    # 9f. Build messages
    system_content = (
        "You are an enterprise compliance AI assistant. "
        "Answer based on the retrieved context. Cite specific articles and sections.\n\n"
        + rag_context
    )
    messages = (
        [{"role": "system", "content": system_content}]
        + trimmed_history
        + [{"role": "user", "content": question}]
    )

    # 9g. Measure — what would the unoptimised call have cost?
    original_messages = (
        [{"role": "system", "content": system_content}]
        + history   # full unmanaged history
        + [{"role": "user", "content": question}]
    )
    original_tokens = (count_messages_tokens(original_messages) +
                       (measure_schema_tokens(tools) if tools else 0))
    optimized_tokens = (count_messages_tokens(messages) +
                        (measure_schema_tokens(compacted_tools) if compacted_tools else 0))

    # 9h. LLM call
    resp = await litellm.acompletion(
        model=chosen_model,
        messages=messages,
        tools=compacted_tools if compacted_tools else None,
        max_tokens=500,
        temperature=0.2,
    )
    reply = resp.choices[0].message.content or ""
    output_tokens = resp.usage.completion_tokens if resp.usage else count_tokens(reply)

    # 9j. Savings report
    report = savings_report(original_tokens, optimized_tokens, output_tokens, chosen_model)

    return {
        "reply":           reply,
        "model":           chosen_model,
        "report":          report,
        "selected_chunks": len(selected_chunks),
        "rag_tokens":      rag_tokens,
        "history_msgs":    len(trimmed_history),
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

# Import sample data from the exercise file
from ex5_token_optimizer import (  # type: ignore
    SAMPLE_TOOLS, SAMPLE_RAG_CHUNKS, SAMPLE_HISTORY, SAMPLE_MCP_RESPONSE,
)
import asyncio


async def main():
    print("=" * 64)
    print("Sol 5: Token Optimization — Reference Solution")
    print("=" * 64)

    # 1. Token counting
    print("\n── 1. Token Counting ──")
    sample = "GDPR Article 28 requires a Data Processing Agreement between controller and processor."
    tokens = count_tokens(sample, DEFAULT_MODEL)
    history_tokens = count_messages_tokens(SAMPLE_HISTORY, DEFAULT_MODEL)
    print(f"  Sentence tokens:  {tokens}")
    print(f"  History tokens:   {history_tokens} ({len(SAMPLE_HISTORY)} messages)")
    print(f"  Estimate cost if sent 1000×/day: "
          f"${estimate_cost(history_tokens*1000, 200*1000):.4f}")

    # 2. Compression
    print("\n── 2. Prompt Compression ──")
    long_text = " ".join(c["text"] for c in SAMPLE_RAG_CHUNKS)
    orig_t = count_tokens(long_text)
    compressed = compress_text(long_text, target_tokens=200)
    comp_t = count_tokens(compressed)
    print(f"  {orig_t} → {comp_t} tokens ({(1 - comp_t/orig_t):.0%} reduction)")
    print(f"  Sample: '{compressed[:120]}...'")

    # 3. RAG budgeting
    print("\n── 3. RAG Context Budgeting ──")
    all_t = sum(count_tokens(c["text"]) for c in SAMPLE_RAG_CHUNKS)
    selected, used = budget_rag_context(SAMPLE_RAG_CHUNKS, "DPA requirements GDPR Article 28", 600)
    print(f"  {all_t} total → {used} tokens used ({len(selected)}/{len(SAMPLE_RAG_CHUNKS)} chunks)")
    print(format_rag_context(selected[:2]))

    # 4. Tool compaction
    print("\n── 4. Tool Schema Compaction ──")
    orig_s  = measure_schema_tokens(SAMPLE_TOOLS)
    compact = compact_tool_schemas(SAMPLE_TOOLS)
    opt_s   = measure_schema_tokens(compact)
    print(f"  {orig_s} → {opt_s} tokens ({(1 - opt_s/orig_s):.0%} reduction)")
    print(f"  Compacted description: '{compact[0]['function']['description']}'")

    # 5. History management
    print("\n── 5. History Management ──")
    before_t = count_messages_tokens(SAMPLE_HISTORY)
    trimmed  = await manage_history(SAMPLE_HISTORY, max_tokens=300, keep_last_n=2)
    after_t  = count_messages_tokens(trimmed)
    print(f"  {before_t} → {after_t} tokens | "
          f"{len(SAMPLE_HISTORY)} → {len(trimmed)} messages")
    for m in trimmed:
        role = m["role"]
        preview = m["content"][:60].replace("\n", " ")
        print(f"    [{role}] {preview}...")

    # 6. MCP compaction
    print("\n── 6. MCP Response Compaction ──")
    mcp_before = count_tokens(json.dumps(SAMPLE_MCP_RESPONSE))
    compacted_mcp = compact_mcp_response(SAMPLE_MCP_RESPONSE)
    mcp_after  = count_tokens(json.dumps(compacted_mcp))
    print(f"  {mcp_before} → {mcp_after} tokens ({(1 - mcp_after/mcp_before):.0%} reduction)")
    tool_fmt = format_tool_result("search_regulations", SAMPLE_MCP_RESPONSE, max_tokens=150)
    print(f"  Formatted result: {count_tokens(tool_fmt)} tokens")
    print(f"  '{tool_fmt[:200]}'")

    # 7. Routing
    print("\n── 7. Model Routing ──")
    cases = [
        ("What is GDPR?", 80),
        ("Analyse and compare GDPR Article 28 legitimate interests step by step", 600),
        ("How do I implement DPA?", 300),
    ]
    for q, ctx in cases:
        complexity = classify_query_complexity(q, ctx)
        routed = route_model(q, ctx)
        print(f"  [{complexity:7}] {q[:50]:<52} → {routed}")

    # 8+9. Full optimised call
    print("\n── 9. Full Optimised Call ──")
    result = await optimised_agent_call(
        question="What are the key obligations under GDPR Article 28 for Data Processing Agreements?",
        history=SAMPLE_HISTORY,
        rag_chunks=SAMPLE_RAG_CHUNKS,
        tools=SAMPLE_TOOLS,
        rag_budget_tokens=600,
        history_max_tokens=300,
    )
    r = result["report"]
    print(f"\n  {r['summary']}")
    print(f"  Model:          {result['model']}")
    print(f"  RAG chunks:     {result['selected_chunks']} ({result['rag_tokens']} tokens)")
    print(f"  History msgs:   {result['history_msgs']}")
    print(f"\n  Answer preview: {result['reply'][:300]}...")

    # Final summary table
    print("\n" + "=" * 64)
    print("TECHNIQUE SAVINGS SUMMARY")
    print("=" * 64)
    techniques = [
        ("Prompt compression (2000→200 tokens)", "~90%"),
        ("RAG budgeting (full→budget)",           f"{(1 - used/max(all_t,1)):.0%}"),
        ("Tool schema compaction",                 f"{(1 - opt_s/max(orig_s,1)):.0%}"),
        ("History management",                     f"{(1 - after_t/max(before_t,1)):.0%}"),
        ("MCP compaction",                         f"{(1 - mcp_after/max(mcp_before,1)):.0%}"),
        ("Model routing (complex→mini)",           "~95% cost for simple queries"),
    ]
    for name, saving in techniques:
        print(f"  {name:<45} → {saving}")
    print(f"\n  End-to-end (this demo): {r['savings_pct']:.0f}% token reduction")
    print(f"  Cost saved per call:    ${r['cost_saved_usd']:.6f}")
    print(f"  Cost saved per 10k:     ${r['cost_saved_usd'] * 10_000:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
