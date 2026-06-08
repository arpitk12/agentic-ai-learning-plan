"""
SOLUTION — Exercise 2: Long-Term Memory Agent with Mem0
Phase 7 / Week 13

How this solution works:
  TODO 1: Memory() with default config uses in-memory Chroma + GPT-4o-mini as the
           extraction LLM. For production, pass MemoryConfig with Qdrant.
  TODO 2: add() with role messages mimics a real conversation — Mem0 automatically
           extracts the key facts and stores them as embeddings.
  TODO 3: Semantic preferences are just messages — Mem0 handles extraction.
  TODO 4: Procedural patterns: same API, differentiated only by metadata type.
  TODO 5: search() returns top-K memories by cosine similarity; filter by metadata type.
  TODO 6: Inject retrieved memories into the system prompt before each LLM call;
           store result as new episodic memory to build up history over time.
  TODO 7: get_all() → filter by timestamp → LLM summarises → delete originals → store summary.
  TODO 8: User profile stored as JSON in a memory message; delete-before-upsert prevents dups.
"""
from __future__ import annotations
import os, json, asyncio
from datetime import datetime, timezone, timedelta
from typing import Any
from pydantic import BaseModel
import litellm
from dotenv import load_dotenv

load_dotenv()


# ── TODO 1 SOLUTION: Initialize Mem0 ─────────────────────────────────────────

def create_memory_client():
    from mem0 import Memory   # type: ignore
    # Default config: in-memory Chroma vector store, GPT-4o-mini for extraction
    # Works out of the box with just OPENAI_API_KEY set
    m = Memory()
    print("Mem0 memory client created (local in-memory config)")
    return m

# Production config (uncomment to use with Qdrant):
# def create_memory_client_production():
#     from mem0 import Memory
#     config = {
#         "vector_store": {
#             "provider": "qdrant",
#             "config": {"host": "localhost", "port": 6333, "collection_name": "compliance_memory"},
#         },
#         "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
#         "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
#     }
#     return Memory.from_config(config)


# ── TODO 2 SOLUTION: Episodic memory ─────────────────────────────────────────

def add_episodic_memory(
    memory_client,
    user_id: str,
    doc_id: str,
    doc_type: str,
    risk_level: str,
    summary: str,
) -> str:
    result = memory_client.add(
        messages=[
            {"role": "user", "content": f"I reviewed document {doc_id} ({doc_type})"},
            {"role": "assistant", "content": f"Risk level: {risk_level}. {summary}"},
        ],
        user_id=user_id,
        metadata={
            "type": "episodic",
            "doc_id": doc_id,
            "doc_type": doc_type,
            "risk_level": risk_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    # mem0 v1 returns list of results, v2 returns dict with "results" key
    mem_id = (
        result[0]["id"] if isinstance(result, list)
        else result.get("id", result.get("results", [{}])[0].get("id", ""))
    )
    print(f"  [Episodic] stored '{doc_id}' → {risk_level} risk | id={mem_id[:8]}...")
    return mem_id


# ── TODO 3 SOLUTION: Semantic memory — user preferences ──────────────────────

def store_preference(memory_client, user_id: str, preference: str) -> str:
    result = memory_client.add(
        messages=[{"role": "user", "content": preference}],
        user_id=user_id,
        metadata={"type": "semantic", "category": "preference"},
    )
    mem_id = (
        result[0]["id"] if isinstance(result, list)
        else result.get("id", "")
    )
    print(f"  [Semantic] preference stored | id={mem_id[:8] if mem_id else '?'}...")
    return mem_id


# ── TODO 4 SOLUTION: Procedural memory — learned workflows ───────────────────

def store_learned_pattern(memory_client, user_id: str, pattern: str) -> str:
    result = memory_client.add(
        messages=[{"role": "user", "content": f"When reviewing documents, I follow this pattern: {pattern}"}],
        user_id=user_id,
        metadata={"type": "procedural", "category": "workflow"},
    )
    mem_id = (
        result[0]["id"] if isinstance(result, list)
        else result.get("id", "")
    )
    print(f"  [Procedural] pattern stored | id={mem_id[:8] if mem_id else '?'}...")
    return mem_id


# ── TODO 5 SOLUTION: Search relevant memories ────────────────────────────────

def search_memory(
    memory_client,
    user_id: str,
    query: str,
    limit: int = 5,
    memory_type: str | None = None,
) -> list[dict]:
    results = memory_client.search(query=query, user_id=user_id, limit=limit * 2)
    # Normalise: some mem0 versions return {"results": [...]}
    if isinstance(results, dict):
        results = results.get("results", [])

    if memory_type:
        results = [
            m for m in results
            if m.get("metadata", {}).get("type") == memory_type
        ]

    results = results[:limit]
    print(f"  [Search] '{query[:40]}...' → {len(results)} memories found")
    for m in results:
        score = m.get("score", m.get("similarity", "?"))
        print(f"    • score={score:.3f if isinstance(score, float) else score} | {m.get('memory', '')[:80]}")
    return results


# ── TODO 6 SOLUTION: Memory-augmented agent ───────────────────────────────────

async def compliance_agent_with_memory(
    memory_client,
    user_id: str,
    document: str,
    document_type: str,
    doc_id: str,
) -> dict:
    # a) Retrieve relevant memories for this user + document type
    memories = search_memory(
        memory_client,
        user_id=user_id,
        query=f"{document_type} compliance review preferences and patterns",
        limit=5,
    )

    # b) Build dynamic system prompt that injects memories
    memory_block = "\n".join(f"- {m.get('memory', '')}" for m in memories)
    system_prompt = f"""You are a compliance review assistant.

Your memory of past interactions with this user:
{memory_block if memory_block else "(no relevant memories yet)"}

Review the document for compliance risks. Return JSON:
{{"risk_level": "low|medium|high|critical", "reasoning": "detailed explanation", "summary": "one sentence"}}"""

    # c) Call LLM with memory-enriched system prompt
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document type: {document_type}\n\n{document}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    # d) Parse result
    result = json.loads(resp.choices[0].message.content)

    # e) Store this run as a new episodic memory so history builds up
    add_episodic_memory(
        memory_client,
        user_id=user_id,
        doc_id=doc_id,
        doc_type=document_type,
        risk_level=result["risk_level"],
        summary=result["summary"],
    )

    # f) Return the parsed result
    print(f"  [Agent] {doc_id} → {result['risk_level']} risk")
    return result


# ── TODO 7 SOLUTION: Memory consolidation ────────────────────────────────────

async def consolidate_old_memories(
    memory_client, user_id: str, older_than_days: int = 7
) -> int:
    # a) Get all memories for this user
    all_memories = memory_client.get_all(user_id=user_id)
    if isinstance(all_memories, dict):
        all_memories = all_memories.get("results", [])

    # b) Filter memories older than the threshold
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    old: list[dict] = []
    for m in all_memories:
        ts_str = m.get("metadata", {}).get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    old.append(m)
            except ValueError:
                pass

    # c) Nothing to consolidate
    if not old:
        print(f"  [Consolidate] No memories older than {older_than_days} days")
        return 0

    # d) Ask LLM to summarise the old memories
    memory_texts = "\n".join(f"- {m.get('memory', '')}" for m in old)
    prompt = f"Summarize these agent memory entries in 3-5 concise bullet points:\n{memory_texts}"
    summary_resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    summary = summary_resp.choices[0].message.content.strip()

    # e) Add consolidated summary as a new memory
    memory_client.add(
        messages=[{"role": "system", "content": f"Consolidated memory summary: {summary}"}],
        user_id=user_id,
        metadata={"type": "consolidated", "timestamp": datetime.now(timezone.utc).isoformat()},
    )

    # f) Delete the old individual memories
    deleted = 0
    for m in old:
        try:
            memory_client.delete(m["id"])
            deleted += 1
        except Exception:
            pass

    print(f"  [Consolidate] Deleted {deleted} old memories, stored 1 summary")
    return deleted


# ── TODO 8 SOLUTION: User profile memory ─────────────────────────────────────

class UserProfile(BaseModel):
    user_id: str
    role: str
    department: str
    risk_tolerance: str
    notification_preference: str

def upsert_user_profile(memory_client, profile: UserProfile) -> None:
    # Delete existing profile memory to avoid duplicates
    all_mems = memory_client.get_all(user_id=profile.user_id)
    if isinstance(all_mems, dict):
        all_mems = all_mems.get("results", [])
    for m in all_mems:
        if m.get("metadata", {}).get("type") == "user_profile":
            memory_client.delete(m["id"])
            print(f"  [Profile] Deleted old profile for {profile.user_id}")

    # Store new profile as JSON in memory
    memory_client.add(
        messages=[{
            "role": "user",
            "content": f"My profile: {profile.model_dump_json()}",
        }],
        user_id=profile.user_id,
        metadata={"type": "user_profile"},
    )
    print(f"  [Profile] Stored profile for {profile.user_id}: {profile.role} @ {profile.department}")

def get_user_profile(memory_client, user_id: str) -> UserProfile | None:
    all_mems = memory_client.get_all(user_id=user_id)
    if isinstance(all_mems, dict):
        all_mems = all_mems.get("results", [])
    for m in all_mems:
        if m.get("metadata", {}).get("type") == "user_profile":
            try:
                # The memory text may contain JSON — try to parse it
                mem_text = m.get("memory", "")
                # Find the JSON blob in the memory text
                start = mem_text.find("{")
                if start != -1:
                    profile_data = json.loads(mem_text[start:])
                    return UserProfile(**profile_data)
            except (json.JSONDecodeError, Exception):
                pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Mem0 Long-Term Memory Agent — SOLUTION ===\n")

    memory = create_memory_client()
    user_id = "analyst_007"

    print("1. Storing initial memories...")
    store_preference(memory, user_id, "I prefer detailed reasoning with specific regulatory citations")
    store_preference(memory, user_id, "Flag missing DPA clauses as high risk, not medium")
    store_learned_pattern(memory, user_id, "For vendor contracts over $500k, always escalate to legal review")
    add_episodic_memory(memory, user_id, "DOC-001", "contract", "high",
                        "Missing SOX §404 controls disclosure. Escalated to legal.")
    print()

    print("2. Setting user profile...")
    upsert_user_profile(memory, UserProfile(
        user_id=user_id,
        role="Senior Compliance Analyst",
        department="Legal & Compliance",
        risk_tolerance="conservative",
        notification_preference="immediate",
    ))
    profile = get_user_profile(memory, user_id)
    print(f"   Profile retrieved: {profile.role if profile else 'NOT FOUND'}\n")

    print("3. Running memory-augmented agent on new document...")
    result = await compliance_agent_with_memory(
        memory, user_id,
        document="This vendor agreement between TechCorp and CloudStorage Ltd processes "
                 "EU customer personal data. No Data Processing Agreement (DPA) is attached. "
                 "Contract value: $2,000,000. Payment terms: net-30. Signed March 2026.",
        document_type="contract",
        doc_id="DOC-NEW",
    )
    print(f"\n   Result: {result}\n")

    print("4. Searching episodic memories...")
    episodes = search_memory(memory, user_id, "vendor contract review", memory_type="episodic")
    print(f"   Found {len(episodes)} episodic memories\n")

    print("5. Memory consolidation (simulating 7+ day old memories)...")
    # Note: fresh memories won't be consolidated; in production this runs on a schedule
    deleted = await consolidate_old_memories(memory, user_id, older_than_days=0)  # 0 = consolidate all
    print(f"   Deleted {deleted} memories, replaced with 1 summary\n")

if __name__ == "__main__":
    asyncio.run(main())
