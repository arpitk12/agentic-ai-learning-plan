"""
ex5_token_optimizer.py
─────────────────────
Exercise: Full-Stack Token Optimization
Goal: Reduce LLM API cost by 60–80% without degrading answer quality.

Techniques covered:
  1. Accurate token counting (tiktoken)
  2. Prompt compression (sentence scoring by TF-IDF importance)
  3. RAG context budgeting (score + rank + trim to token budget)
  4. Tool schema compaction (strip verbose descriptions, abbreviate)
  5. Context history management (sliding window + LLM summarisation)
  6. MCP response compaction (extract only essential fields)
  7. Model routing by query complexity
  8. Cost measurement and savings reporting

Run:
  pip install tiktoken litellm python-dotenv
  python ex5_token_optimizer.py
"""
from __future__ import annotations
import os, json, re
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Cost table: (input_per_1k_usd, output_per_1k_usd) ─────────────────────────
COST_TABLE: dict[str, tuple[float, float]] = {
    "openai/gpt-4o":              (0.0050, 0.0150),
    "openai/gpt-4o-mini":         (0.0002, 0.0006),
    "openai/gpt-3.5-turbo":       (0.0005, 0.0015),
    "gemini/gemini-2.0-flash":    (0.0001, 0.0003),
    "groq/llama-3.3-70b-versatile": (0.0006, 0.0008),
}
DEFAULT_MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")


# ══════════════════════════════════════════════════════════════════════════════
# TODO 1 — Accurate Token Counting
# ══════════════════════════════════════════════════════════════════════════════

def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Count tokens in `text` for a given model using tiktoken.

    Steps:
      1a. import tiktoken
      1b. Map the model name to an encoding:
          - models containing "gpt-4" or "gpt-3.5" → "cl100k_base"
          - models containing "gemini" or "llama"   → approximate with "cl100k_base"
          Use tiktoken.get_encoding("cl100k_base") as a safe default.
      1c. Try tiktoken.encoding_for_model(model_short_name) first
          (strip the "openai/" "groq/" prefix before looking up)
      1d. Return len(encoding.encode(text))

    Hint: tiktoken.encoding_for_model() raises KeyError for unknown models —
          catch it and fall back to get_encoding("cl100k_base").
    """
    # import tiktoken
    # short_name = model.split("/")[-1]
    # try:
    #     enc = tiktoken.encoding_for_model(short_name)
    # except KeyError:
    #     enc = tiktoken.get_encoding("cl100k_base")
    # return len(enc.encode(text))
    raise NotImplementedError


def count_messages_tokens(messages: list[dict], model: str = DEFAULT_MODEL) -> int:
    """
    Count total tokens for a list of chat messages (including role overhead).

    OpenAI overhead per message: 4 tokens for the envelope {"role": ..., "content": ...}
    Plus 2 tokens for the reply primer: <|im_start|>assistant

    Steps:
      1a. total = 2  (reply primer overhead)
      1b. For each message: total += 4 + count_tokens(message["content"], model)
      1c. Return total

    Reference: https://platform.openai.com/docs/guides/chat/managing-tokens
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TODO 2 — Prompt Compression (TF-IDF sentence ranking)
# ══════════════════════════════════════════════════════════════════════════════

def compress_text(text: str, target_tokens: int, model: str = DEFAULT_MODEL) -> str:
    """
    Compress `text` to approximately `target_tokens` tokens by keeping only
    the most important sentences (ranked by TF-IDF word importance).

    Steps:
      2a. If count_tokens(text) <= target_tokens: return text unchanged
      2b. Split text into sentences (split on '. ' / '.\n' / '! ' / '? ')
          Keep only sentences with >= 5 words.
      2c. Score each sentence:
          - Build word frequency dict across ALL sentences (lowercased, strip punct)
          - Score = sum of word frequencies for words in the sentence
            (high frequency = important concept repeated often)
      2d. Rank sentences by score descending
      2e. Greedily add highest-scoring sentences until target_tokens is reached
      2f. Re-sort kept sentences by their original position (preserve reading order)
      2g. Return " ".join(kept_sentences)

    Note: This is a simple extractive approach. For production consider
          LLMLingua (https://github.com/microsoft/LLMLingua) or Recomp.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TODO 3 — RAG Context Budgeting
# ══════════════════════════════════════════════════════════════════════════════

def budget_rag_context(
    chunks: list[dict],     # [{"text": str, "score": float, "source": str}]
    query: str,
    token_budget: int = 1500,
    model: str = DEFAULT_MODEL,
) -> tuple[list[dict], int]:
    """
    Select the best RAG chunks that fit within `token_budget` tokens.

    Steps:
      3a. Sort chunks by score descending (highest relevance first)
      3b. Greedily add chunks to the result list:
              chunk_tokens = count_tokens(chunk["text"], model)
              if used + chunk_tokens <= token_budget: add chunk, used += chunk_tokens
              else if chunk_tokens > token_budget * 0.4: try compressing the chunk
                  compressed = compress_text(chunk["text"], token_budget - used, model)
                  if count_tokens(compressed) + used <= token_budget: add compressed version
      3c. Return (selected_chunks, total_tokens_used)

    Why: Without budgeting, 10 retrieved chunks for a complex query can easily
         consume 4000+ tokens — dominating your prompt cost.
    """
    raise NotImplementedError


def format_rag_context(chunks: list[dict]) -> str:
    """
    Format selected RAG chunks as a compact context block.

    Instead of verbose XML tags, use compact line format:
      [1] {source}: {text}
      [2] {source}: {text}

    Steps:
      4a. For each (i, chunk): f"[{i+1}] {chunk.get('source','doc')}: {chunk['text']}"
      4b. Return "\n".join(lines)
      4c. Prefix with "Context:\n" (2 tokens vs 10+ for verbose XML wrappers)
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TODO 4 — Tool Schema Compaction
# ══════════════════════════════════════════════════════════════════════════════

def compact_tool_schemas(tools: list[dict]) -> list[dict]:
    """
    Reduce token cost of tool definitions sent to the LLM.

    Techniques:
      4a. Truncate descriptions: keep only the first sentence (up to 80 chars)
          tool["function"]["description"] = first_sentence[:80]
      4b. Remove "examples" keys from parameter schemas (not used by model)
      4c. Remove "default" keys from required parameters
      4d. Abbreviate enum values: if an enum has > 5 values, keep 3 + add "..."
          to the description
      4e. Remove "title" keys from parameter schemas (OpenAI ignores them)
      4f. Return compacted tools list (deep copy — don't mutate input)

    Why: Tool schemas can add 200–800 tokens per tool. In agents with 10+ tools,
         this is the #1 hidden cost driver.
    """
    raise NotImplementedError


def measure_schema_tokens(tools: list[dict], model: str = DEFAULT_MODEL) -> int:
    """Count how many tokens a tools list consumes when JSON-serialised."""
    return count_tokens(json.dumps(tools), model)


# ══════════════════════════════════════════════════════════════════════════════
# TODO 5 — Context History Management (sliding window + summarisation)
# ══════════════════════════════════════════════════════════════════════════════

async def summarise_history(
    old_messages: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Condense old conversation turns into a short summary.

    Steps:
      5a. Build a transcript string:
          "\n".join(f"{m['role'].upper()}: {m['content']}" for m in old_messages)
      5b. Call LLM: "Summarise this conversation in 3 bullet points. Be concise."
          Use max_tokens=150 to force a short reply.
      5c. Return the summary string

    Why: A 10-turn history can be 2000 tokens; its summary is ~100 tokens.
    """
    raise NotImplementedError


async def manage_history(
    messages: list[dict],
    max_tokens: int = 2000,
    keep_last_n: int = 4,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """
    Keep the conversation within `max_tokens` by summarising old turns.

    Strategy:
      5d. Count total tokens across all messages
      5e. If total <= max_tokens: return messages unchanged
      5f. Separate: system messages, recent N turns (keep_last_n pairs = 2N messages),
          older turns
      5g. If no older turns to summarise: trim oldest non-system message and retry
      5h. summary = await summarise_history(older_turns)
      5i. Inject summary as a system message:
          {"role": "system", "content": f"[Conversation summary]\n{summary}"}
      5j. Return: [system_messages] + [summary_message] + [recent_turns]
      5k. Recurse if still over max_tokens (handles very long sessions)

    Returns:
        Trimmed messages list guaranteed to fit within max_tokens
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TODO 6 — MCP Response Compaction
# ══════════════════════════════════════════════════════════════════════════════

def compact_mcp_response(mcp_response: dict, essential_keys: list[str] | None = None) -> dict:
    """
    Extract only the essential fields from an MCP tool response before injecting
    it back into the conversation context.

    MCP responses often return verbose metadata alongside the actual data.
    Injecting the raw response wastes tokens on fields the model doesn't use.

    Steps:
      6a. If essential_keys is provided: return {k: mcp_response[k] for k in essential_keys if k in mcp_response}
      6b. Otherwise apply heuristics to strip common noise fields:
          Remove keys: ["_meta", "metadata", "request_id", "trace_id", "timing",
                        "version", "api_version", "rate_limit_info", "x_request_id",
                        "links", "_links", "href", "self", "etag", "last_modified"]
      6c. Recursively compact nested dicts (same rules applied to sub-dicts)
      6d. For lists: if len > 10, keep first 10 and append {"_truncated": true, "_total": len}
      6e. Return the compacted dict

    Why: A search MCP response can include 3000 tokens of metadata around
         200 tokens of actual content. Compaction = 10–15× reduction.
    """
    raise NotImplementedError


def format_tool_result(tool_name: str, result: Any, max_tokens: int = 400,
                        model: str = DEFAULT_MODEL) -> str:
    """
    Convert a tool result to a compact string for the assistant message.

    Steps:
      6e. JSON-serialise result
      6f. If token count > max_tokens: compress_text(json_str, max_tokens, model)
      6g. Return f"Tool '{tool_name}' returned:\n{compact_result}"

    Why: Tool results injected as raw JSON often contain deeply nested structures
         the model ignores. A 2000-token JSON result can often compress to 200 tokens.
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TODO 7 — Model Routing by Complexity
# ══════════════════════════════════════════════════════════════════════════════

def classify_query_complexity(query: str, context_tokens: int = 0) -> str:
    """
    Classify a query as "simple", "medium", or "complex" without an LLM call.

    Rules (heuristic — tune thresholds for your workload):
      7a. SIMPLE if ALL of:
          - len(query.split()) < 15               (short question)
          - context_tokens < 500                  (small context)
          - no multi-step keywords ("and then", "after that", "first ... then",
            "compare", "analyse", "explain why", "evaluate")
          - no code patterns (```, def , class , import )
      7b. COMPLEX if ANY of:
          - context_tokens > 3000
          - len(query.split()) > 40
          - multi-hop keywords present ("and then", "compare", "contrast",
            "pros and cons", "evaluate all", "step by step")
          - code patterns AND reasoning patterns together
      7c. MEDIUM otherwise

    Returns: "simple" | "medium" | "complex"
    """
    raise NotImplementedError


def route_model(
    query: str,
    context_tokens: int = 0,
    simple_model:  str = "openai/gpt-4o-mini",
    medium_model:  str = "openai/gpt-4o-mini",
    complex_model: str = "openai/gpt-4o",
) -> str:
    """
    Return the appropriate model name based on query complexity.

    7d. complexity = classify_query_complexity(query, context_tokens)
    7e. Map: simple → simple_model, medium → medium_model, complex → complex_model
    7f. Log the routing decision with complexity label
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TODO 8 — Cost Measurement and Savings Report
# ══════════════════════════════════════════════════════════════════════════════

def estimate_cost(input_tokens: int, output_tokens: int, model: str = DEFAULT_MODEL) -> float:
    """
    Estimate USD cost for a single LLM call.

    Steps:
      8a. Look up (in_rate, out_rate) = COST_TABLE.get(model, (0.0002, 0.0006))
      8b. return (input_tokens / 1000 * in_rate) + (output_tokens / 1000 * out_rate)
    """
    raise NotImplementedError


def savings_report(
    original_input_tokens: int,
    optimized_input_tokens: int,
    output_tokens: int,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Produce a structured savings report comparing original vs optimised.

    Steps:
      8c. original_cost = estimate_cost(original_input_tokens, output_tokens, model)
      8d. optimized_cost = estimate_cost(optimized_input_tokens, output_tokens, model)
      8e. savings_pct = (original_input_tokens - optimized_input_tokens) / original_input_tokens * 100
      8f. Return dict with all fields + a formatted print-ready summary string

    Returns:
        {
          "original_tokens":   int,
          "optimized_tokens":  int,
          "tokens_saved":      int,
          "savings_pct":       float,
          "original_cost_usd": float,
          "optimized_cost_usd":float,
          "cost_saved_usd":    float,
          "model":             str,
          "summary":           str,  # one-line human readable
        }
    """
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TODO 9 — Full Optimised Agent Call
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
    """
    Make a single LLM call using ALL optimisation techniques from TODOs 1–8.

    Pipeline:
      9a. RAG:     budget_rag_context(rag_chunks, question, rag_budget_tokens)
                   format_rag_context(selected_chunks)
      9b. History: manage_history(history, history_max_tokens)
      9c. Tools:   compact_tool_schemas(tools)
      9d. Context count = count_messages_tokens(trimmed_history) + count_tokens(rag_context)
      9e. Routing: chosen_model = route_model(question, context_count)
                   if model is not None: override with provided model
      9f. Build messages:
           system = f"You are a helpful assistant.\n\n{rag_context}"
           messages = [{"role": "system", "content": system}] + trimmed_history
                    + [{"role": "user", "content": question}]
      9g. Measure original_tokens = count_messages_tokens(original_messages)
      9h. LLM call: litellm.acompletion(model=chosen_model, messages=messages,
                                         tools=compacted_tools if tools else None,
                                         max_tokens=500)
      9i. Extract reply, actual usage from response
      9j. Compute report = savings_report(original_tokens, count_messages_tokens(messages),
                                           response.usage.completion_tokens, chosen_model)
      9k. Return: {"reply": str, "model": str, "report": dict, "selected_chunks": int}
    """
    raise NotImplementedError


# ── Demo ──────────────────────────────────────────────────────────────────────

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": (
                "Search the regulatory database for compliance requirements, laws, directives, "
                "and regulatory guidance. This function queries across multiple jurisdictions "
                "including EU, US, UK, and APAC regions and returns the most relevant "
                "regulatory texts and summaries. Use this when you need to find specific "
                "regulation text or compliance requirements for a given topic."
            ),
            "parameters": {
                "type": "object",
                "title": "SearchRegulationsInput",
                "properties": {
                    "query": {
                        "type": "string",
                        "title": "Query",
                        "description": "The search query to find relevant regulations and compliance requirements",
                        "examples": ["GDPR data processing", "HIPAA breach notification"],
                    },
                    "jurisdiction": {
                        "type": "string",
                        "title": "Jurisdiction",
                        "description": "The legal jurisdiction to search within",
                        "enum": ["EU", "US", "UK", "APAC", "global", "CA", "AU", "SG"],
                        "default": "global",
                    },
                    "regulation_type": {
                        "type": "string",
                        "description": "The type of regulation to filter by",
                        "enum": ["privacy", "financial", "healthcare", "security", "employment", "environmental"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_compliance_status",
            "description": (
                "Check the current compliance status of an organisation against specified "
                "frameworks and standards. Returns a detailed assessment including gaps, "
                "risks, recommended actions, timeline estimates, and cost projections "
                "for achieving full compliance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "org_id": {
                        "type": "string",
                        "title": "Organisation ID",
                        "description": "The unique identifier for the organisation to check",
                        "examples": ["ORG-123", "acme-corp"],
                    },
                    "frameworks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of compliance frameworks to assess against",
                        "examples": [["GDPR", "SOC2"], ["HIPAA", "ISO27001"]],
                        "default": ["GDPR"],
                    },
                },
                "required": ["org_id"],
            },
        },
    },
]

SAMPLE_RAG_CHUNKS = [
    {"text": "GDPR Article 28 requires that controllers only use processors providing "
             "sufficient guarantees to implement appropriate technical and organisational "
             "measures ensuring processing meets GDPR requirements protecting data subject rights.",
     "score": 0.92, "source": "gdpr_article28.pdf"},
    {"text": "Under Article 28(3), the contract between controller and processor must set out "
             "the subject matter, duration, nature and purpose of the processing, the type of "
             "personal data and categories of data subjects, and the obligations and rights of "
             "the controller. This is typically documented in a Data Processing Agreement (DPA).",
     "score": 0.88, "source": "gdpr_article28.pdf"},
    {"text": "GDPR Recital 81 states that to ensure compliance with this Regulation in respect "
             "of the processing to be carried out by the processor on behalf of the controller, "
             "when entrusting a processor with processing activities, the controller should use "
             "only processors providing sufficient guarantees.",
     "score": 0.75, "source": "gdpr_recitals.pdf"},
    {"text": "Data Processing Agreements must include provisions for sub-processors. Article 28(2) "
             "specifies that a processor shall not engage another processor without prior specific "
             "or general written authorisation of the controller.",
     "score": 0.71, "source": "gdpr_article28.pdf"},
    {"text": "The ISO 27001 standard requires organisations to establish an Information Security "
             "Management System (ISMS). The standard covers 14 control domains including access "
             "control, cryptography, physical security, and supplier relationships.",
     "score": 0.45, "source": "iso27001_overview.pdf"},
]

SAMPLE_HISTORY = [
    {"role": "user",      "content": "What is GDPR?"},
    {"role": "assistant", "content": "GDPR (General Data Protection Regulation) is the EU's comprehensive data privacy law that came into force on 25 May 2018. It governs how organisations collect, store, and process personal data of EU residents. Key principles include lawfulness, fairness and transparency; purpose limitation; data minimisation; accuracy; storage limitation; integrity and confidentiality; and accountability."},
    {"role": "user",      "content": "What are the main rights of data subjects under GDPR?"},
    {"role": "assistant", "content": "Data subjects have eight main rights under GDPR: (1) Right to be informed, (2) Right of access (Article 15), (3) Right to rectification (Article 16), (4) Right to erasure / 'right to be forgotten' (Article 17), (5) Right to restrict processing (Article 18), (6) Right to data portability (Article 20), (7) Right to object (Article 21), and (8) Rights related to automated decision-making and profiling (Article 22). Organisations must respond to data subject requests within one calendar month."},
    {"role": "user",      "content": "How do these rights interact with legitimate interests?"},
    {"role": "assistant", "content": "Legitimate interests (Article 6(1)(f)) is a lawful basis for processing but must be balanced against data subjects' rights. Controllers must conduct a Legitimate Interests Assessment (LIA) — a three-part test: Purpose test (is there a legitimate interest?), Necessity test (is processing necessary?), Balancing test (does the interest override individual rights?). Crucially, data subjects retain the right to object to legitimate interests processing at any time, and the controller must cease unless compelling legitimate grounds override the individual's rights."},
    {"role": "user",      "content": "What are the penalties for GDPR violations?"},
    {"role": "assistant", "content": "GDPR provides two tiers of fines: Tier 1 (up to €10 million or 2% of global annual turnover, whichever is higher) for infringements of basic obligations such as child consent provisions, data protection by design, processor obligations, and breach notification requirements. Tier 2 (up to €20 million or 4% of global annual turnover) for the most serious violations including core principles, data subjects' rights, and international transfers. The UK's ICO can also fine up to £17.5 million or 4% of global turnover post-Brexit."},
]

SAMPLE_MCP_RESPONSE = {
    "_meta": {"request_id": "req_abc123", "trace_id": "tr_xyz789"},
    "api_version": "2024-01",
    "timing": {"total_ms": 234, "db_ms": 45, "cache_ms": 12},
    "rate_limit_info": {"remaining": 950, "reset_at": "2026-06-09T15:00:00Z"},
    "links": {"self": "/regulations/gdpr/28", "next": "/regulations/gdpr/29"},
    "etag": "a1b2c3d4",
    "data": {
        "regulation": "GDPR Article 28",
        "text": "Where processing is to be carried out on behalf of a controller...",
        "jurisdiction": "EU",
        "effective_date": "2018-05-25",
        "compliance_status": "active",
    },
    "related": [f"item_{i}" for i in range(50)],   # 50 related items — mostly noise
}


import asyncio

async def main():
    print("=" * 60)
    print("Exercise 5: Token Optimization")
    print("=" * 60)

    # TODO 1: Token counting
    print("\n── 1. Token Counting ──")
    sample = "GDPR Article 28 requires a Data Processing Agreement between controller and processor."
    tokens = count_tokens(sample, DEFAULT_MODEL)
    print(f"  Text: '{sample[:60]}...'")
    print(f"  Tokens: {tokens}")
    msg_tokens = count_messages_tokens(SAMPLE_HISTORY, DEFAULT_MODEL)
    print(f"  History ({len(SAMPLE_HISTORY)} messages): {msg_tokens} tokens")

    # TODO 2: Prompt compression
    print("\n── 2. Prompt Compression ──")
    long_text = " ".join(c["text"] for c in SAMPLE_RAG_CHUNKS)
    original_tokens = count_tokens(long_text, DEFAULT_MODEL)
    compressed = compress_text(long_text, target_tokens=200, model=DEFAULT_MODEL)
    compressed_tokens = count_tokens(compressed, DEFAULT_MODEL)
    print(f"  Original: {original_tokens} tokens → Compressed: {compressed_tokens} tokens")
    print(f"  Reduction: {(1 - compressed_tokens/original_tokens):.0%}")

    # TODO 3: RAG budgeting
    print("\n── 3. RAG Context Budgeting ──")
    all_rag_tokens = sum(count_tokens(c["text"]) for c in SAMPLE_RAG_CHUNKS)
    selected, used = budget_rag_context(SAMPLE_RAG_CHUNKS, "GDPR Article 28 DPA requirements", 600)
    print(f"  All chunks: {all_rag_tokens} tokens → Selected: {used} tokens ({len(selected)} chunks)")
    print(f"  Formatted:\n{format_rag_context(selected)[:300]}...")

    # TODO 4: Tool schema compaction
    print("\n── 4. Tool Schema Compaction ──")
    original_schema_tokens = measure_schema_tokens(SAMPLE_TOOLS)
    compacted = compact_tool_schemas(SAMPLE_TOOLS)
    compacted_schema_tokens = measure_schema_tokens(compacted)
    print(f"  Schema tokens: {original_schema_tokens} → {compacted_schema_tokens}")
    print(f"  Reduction: {(1 - compacted_schema_tokens/original_schema_tokens):.0%}")

    # TODO 5: History management
    print("\n── 5. History Management ──")
    trimmed = await manage_history(SAMPLE_HISTORY, max_tokens=300, keep_last_n=2)
    before = count_messages_tokens(SAMPLE_HISTORY)
    after = count_messages_tokens(trimmed)
    print(f"  History: {before} tokens → {after} tokens ({len(SAMPLE_HISTORY)}→{len(trimmed)} messages)")

    # TODO 6: MCP compaction
    print("\n── 6. MCP Response Compaction ──")
    before_mcp = count_tokens(json.dumps(SAMPLE_MCP_RESPONSE))
    compacted_mcp = compact_mcp_response(SAMPLE_MCP_RESPONSE)
    after_mcp = count_tokens(json.dumps(compacted_mcp))
    print(f"  MCP response: {before_mcp} → {after_mcp} tokens")
    fmt = format_tool_result("search_regulations", compacted_mcp, max_tokens=150)
    print(f"  Formatted tool result: {count_tokens(fmt)} tokens")

    # TODO 7: Model routing
    print("\n── 7. Model Routing ──")
    queries = [
        ("What is GDPR?", 100),
        ("Analyse and compare all GDPR legitimate interests balancing tests step by step", 500),
        ("Extract entities", 50),
    ]
    for q, ctx_tokens in queries:
        model = route_model(q, ctx_tokens)
        complexity = classify_query_complexity(q, ctx_tokens)
        print(f"  [{complexity:7}] '{q[:50]}' → {model}")

    # TODO 8 + 9: Full optimised call
    print("\n── 8+9. Full Optimised Agent Call ──")
    result = await optimised_agent_call(
        question="What are the key obligations under GDPR Article 28 for Data Processing Agreements?",
        history=SAMPLE_HISTORY,
        rag_chunks=SAMPLE_RAG_CHUNKS,
        tools=SAMPLE_TOOLS,
        rag_budget_tokens=600,
        history_max_tokens=300,
    )
    r = result["report"]
    print(f"\n  Reply: {result['reply'][:200]}...")
    print(f"\n  {r['summary']}")
    print(f"  RAG chunks used: {result['selected_chunks']}")
    print(f"  Model routed to: {result['model']}")


if __name__ == "__main__":
    asyncio.run(main())
