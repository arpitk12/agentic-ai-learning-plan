"""
solution/src/agents/multimodal_agent.py — Full implementation.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass

from src.guardrails.pipeline import run_pipeline  # type: ignore
from src.retrieval.hybrid_retriever import hybrid_search  # type: ignore
from src.memory.mem0_store import search_memories, inject_into_prompt, add_memory  # type: ignore
from src.guardrails.pii_scanner import scan_and_anonymize  # type: ignore


@dataclass
class AgentDependencies:
    collections: dict
    neo4j_driver: object
    neo4j_schema: str
    fallback_chain: object
    mem0_client: object
    cost_tracker: object
    logger: object


def build_context_string(search_results: dict) -> str:
    parts: list[str] = []
    if search_results.get("text"):
        parts.append("=== Document Context ===")
        for h in search_results["text"]:
            meta = h.get("metadata", {})
            parts.append(f"[Document] (score={h['score']:.2f}, page={meta.get('page', '?')})")
            parts.append(h["text"])
    if search_results.get("images"):
        parts.append("\n=== Image Context ===")
        for h in search_results["images"]:
            parts.append(f"[Image] (score={h['score']:.2f})")
            parts.append(h["text"])   # text = image description
            meta = h.get("metadata", {})
            if meta.get("key_data"):
                parts.append(f"  Key data: {meta['key_data']}")
    if search_results.get("audio"):
        parts.append("\n=== Audio Transcript Context ===")
        for h in search_results["audio"]:
            meta = h.get("metadata", {})
            t = f"t={meta.get('start_time', 0):.1f}s" if meta.get("start_time") else ""
            parts.append(f"[Audio] (score={h['score']:.2f} {t})")
            parts.append(h["text"])
    if search_results.get("graph"):
        parts.append("\n=== Knowledge Graph Facts ===")
        for item in search_results["graph"]:
            parts.append(f"[Graph] {item['fact']}")
    if not parts:
        return ""
    return "RETRIEVED CONTEXT:\n" + "\n".join(parts)


async def analyze(
    user_id: str,
    question: str,
    deps: AgentDependencies,
    include_graph: bool = True,
    top_k: int = 5,
) -> dict:
    t0 = time.time()

    # 1. Input guardrails
    guard = await run_pipeline(question)
    if not guard.safe:
        return {
            "error": f"Request blocked by guardrail {guard.blocked_layer}",
            "blocked_layer": guard.blocked_layer,
            "issues": guard.issues,
        }

    # 2. Hybrid retrieval
    search_results = await hybrid_search(
        guard.sanitized_text,
        deps.collections,
        deps.neo4j_driver,
        deps.neo4j_schema,
        top_k=top_k,
        include_graph=include_graph,
    )
    context_str = build_context_string(search_results)

    # 3. Memory retrieval
    memories = search_memories(deps.mem0_client, guard.sanitized_text, user_id, limit=5)
    memory_str = inject_into_prompt(memories)

    # 4. Build messages
    system = (
        "You are an enterprise compliance AI assistant. "
        "Provide precise, actionable compliance guidance based on the retrieved context. "
        "Always cite specific regulations and document sections when possible.\n\n"
        f"{memory_str}\n"
        f"{context_str}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": guard.sanitized_text},
    ]

    # 5. Resilient LLM call
    reply, model_used = await deps.fallback_chain.call(messages, temperature=0.2)

    # 6. Output guardrails: PII scan on response
    clean_reply, reply_pii = scan_and_anonymize(reply)

    # 7. Store to memory (episodic)
    add_memory(
        deps.mem0_client, user_id,
        [{"role": "user", "content": question},
         {"role": "assistant", "content": clean_reply}],
        memory_type="episodic",
    )

    # 8. Cost tracking (rough estimate without usage data)
    rough_in = len(" ".join(m["content"] for m in messages).split())
    rough_out = len(clean_reply.split())
    cost = deps.cost_tracker.record(user_id, model_used, rough_in, rough_out) \
           if hasattr(deps.cost_tracker, "record") else 0.0

    latency_ms = int((time.time() - t0) * 1000)

    sources = search_results.get("text", [])[:3]
    graph_facts = [g["fact"] for g in search_results.get("graph", [])]

    return {
        "answer": clean_reply,
        "sources": sources,
        "graph_facts": graph_facts,
        "memories_used": len(memories),
        "model_used": model_used,
        "cost_usd": cost,
        "latency_ms": latency_ms,
        "pii_sanitized": bool(guard.pii_types_found or reply_pii),
    }
