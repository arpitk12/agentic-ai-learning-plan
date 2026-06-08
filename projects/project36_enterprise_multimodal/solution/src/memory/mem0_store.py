"""
solution/src/memory/mem0_store.py — Full implementation.
"""
from __future__ import annotations
import asyncio
import json
import time
import litellm  # type: ignore

MEMORY_TYPES = ("episodic", "semantic", "procedural", "profile")


def create_client(api_key: str = ""):
    from mem0 import Memory  # type: ignore
    if api_key:
        return Memory(api_key=api_key)
    return Memory()


def add_memory(client, user_id: str, messages: list[dict], memory_type: str = "episodic") -> str:
    if memory_type not in MEMORY_TYPES:
        memory_type = "episodic"
    try:
        result = client.add(
            messages=messages,
            user_id=user_id,
            metadata={"type": memory_type, "timestamp": time.time()},
        )
        # Handle mem0 v1 (list) vs v2 (dict)
        if isinstance(result, list):
            return result[0].get("id", "") if result else ""
        if isinstance(result, dict):
            if "id" in result:
                return result["id"]
            results = result.get("results", [])
            return results[0].get("id", "") if results else ""
        return ""
    except Exception:
        return ""


def search_memories(client, query: str, user_id: str,
                    limit: int = 5, memory_type: str | None = None) -> list[dict]:
    try:
        raw = client.search(query=query, user_id=user_id, limit=limit * 2)
        # Normalise format
        if isinstance(raw, dict):
            items = raw.get("results", [])
        elif isinstance(raw, list):
            items = raw
        else:
            items = []
        results = []
        for item in items:
            text = item.get("memory") or item.get("text") or ""
            score = float(item.get("score", 0.0))
            mtype = item.get("metadata", {}).get("type", "episodic")
            if memory_type and mtype != memory_type:
                continue
            results.append({"text": text, "score": score, "type": mtype,
                             "id": item.get("id", "")})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
    except Exception:
        return []


def inject_into_prompt(memories: list[dict]) -> str:
    if not memories:
        return ""
    groups: dict[str, list[str]] = {}
    for m in memories:
        mtype = m.get("type", "episodic")
        groups.setdefault(mtype, []).append(m["text"])

    labels = {
        "episodic": "Past conversations",
        "semantic": "Known facts",
        "procedural": "Learned procedures",
        "profile": "User preferences",
    }
    parts = ["Relevant context from your long-term memory:"]
    for mtype, texts in groups.items():
        label = labels.get(mtype, mtype.title())
        parts.append(f"\n[{label}]")
        for t in texts:
            parts.append(f"• {t}")
    return "\n".join(parts) + "\n"


async def consolidate(client, user_id: str, max_memories: int = 50,
                       model: str = "openai/gpt-4o-mini") -> int:
    try:
        all_memories = client.get_all(user_id=user_id)
        if isinstance(all_memories, dict):
            all_memories = all_memories.get("results", [])
        if len(all_memories) <= max_memories:
            return 0

        # Sort oldest first, take the oldest 20
        sorted_mems = sorted(all_memories, key=lambda m: m.get("metadata", {}).get("timestamp", 0))
        to_consolidate = sorted_mems[:20]
        texts = [m.get("memory") or m.get("text") or "" for m in to_consolidate]

        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content":
                f"Summarise these memories into 3 key points:\n" + "\n".join(f"- {t}" for t in texts)}],
            temperature=0.2,
        )
        summary = resp.choices[0].message.content

        # Add summary + delete originals
        add_memory(client, user_id, [{"role": "assistant", "content": summary}], "semantic")
        for m in to_consolidate:
            try:
                client.delete(m["id"])
            except Exception:
                pass

        return len(to_consolidate)
    except Exception:
        return 0
