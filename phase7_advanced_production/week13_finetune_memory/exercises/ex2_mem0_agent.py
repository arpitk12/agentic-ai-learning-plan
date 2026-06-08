"""
Exercise 2: Long-Term Memory Agent with Mem0
Phase 7 / Week 13 — Fine-Tuning + Long-Term Memory

Goal: Build a compliance assistant that remembers user preferences, past document
      reviews, and learned policies across sessions using all four Mem0 memory types.

Stack: mem0ai · litellm · pydantic · asyncio

pip install mem0ai litellm pydantic python-dotenv

TODOs:
  1. Initialize Mem0 with a local vector store config
  2. Build add_memory() that stores agent run results as episodic memory
  3. Build store_preference() for user preference (semantic memory)
  4. Build store_learned_pattern() for procedural memory
  5. Build search_memory() to retrieve relevant memories before agent call
  6. Build a full conversation agent that injects memories into system prompt
  7. Implement memory consolidation: compress memories older than 7 days
  8. BONUS: User profile memory — persist role, department, risk tolerance
"""
from __future__ import annotations
import os, json, asyncio
from datetime import datetime, timezone, timedelta
from typing import Any
from pydantic import BaseModel
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── TODO 1: Initialize Mem0 ────────────────────────────────────────────────────

def create_memory_client():
    """
    TODO 1: Create and return a Mem0 Memory instance.

    from mem0 import Memory

    For local dev, use the default config (in-memory vector store, litellm LLM):
      m = Memory()

    For production config with Qdrant vector store:
      from mem0 import Memory, MemoryConfig
      config = MemoryConfig(
          vector_store={"provider": "qdrant", "config": {...}},
          llm={"provider": "litellm", "config": {"model": "gpt-4o-mini"}},
          embedder={"provider": "huggingface", "config": {"model": "BAAI/bge-small-en-v1.5"}},
      )
      m = Memory.from_config(config)

    Return the Memory instance.
    """
    # TODO 1: implement here
    raise NotImplementedError

# ── TODO 2: Episodic memory — store past agent runs ───────────────────────────

def add_episodic_memory(
    memory_client,
    user_id: str,
    doc_id: str,
    doc_type: str,
    risk_level: str,
    summary: str,
) -> str:
    """
    TODO 2: Store a past document review as episodic memory.

    Call memory_client.add() with:
      messages=[
          {"role": "user", "content": f"I reviewed document {doc_id} ({doc_type})"},
          {"role": "assistant", "content": f"Risk level: {risk_level}. {summary}"},
      ]
      user_id=user_id
      metadata={"type": "episodic", "doc_id": doc_id, "risk_level": risk_level,
                 "timestamp": datetime.now(timezone.utc).isoformat()}

    Return the memory ID (result["id"] or result[0]["id"] depending on mem0 version).
    Print what was stored.
    """
    # TODO 2: implement here
    raise NotImplementedError

# ── TODO 3: Semantic memory — user preferences ────────────────────────────────

def store_preference(memory_client, user_id: str, preference: str) -> str:
    """
    TODO 3: Store a user preference as semantic memory.

    Examples of preferences:
      "I want detailed reasoning chains in risk reports"
      "Flag anything with missing DPA clauses immediately"
      "I prefer concise summaries, not bullet lists"

    Call memory_client.add() with:
      messages=[{"role": "user", "content": preference}]
      user_id=user_id
      metadata={"type": "semantic", "category": "preference"}

    Return the memory ID.
    """
    # TODO 3: implement here
    raise NotImplementedError

# ── TODO 4: Procedural memory — learned workflows ─────────────────────────────

def store_learned_pattern(memory_client, user_id: str, pattern: str) -> str:
    """
    TODO 4: Store a learned workflow pattern as procedural memory.

    Examples of patterns:
      "For vendor contracts, always check SOX §404 before GDPR"
      "If risk_level=critical and amount>$1M, escalate to legal immediately"
      "Contracts from APAC region require additional data residency check"

    Store with metadata={"type": "procedural", "category": "workflow"}.
    Return the memory ID.
    """
    # TODO 4: implement here
    raise NotImplementedError

# ── TODO 5: Search relevant memories ─────────────────────────────────────────

def search_memory(
    memory_client,
    user_id: str,
    query: str,
    limit: int = 5,
    memory_type: str | None = None,
) -> list[dict]:
    """
    TODO 5: Search memories relevant to a query.

    Call memory_client.search(query=query, user_id=user_id, limit=limit).
    
    If memory_type is provided (e.g., "episodic", "semantic", "procedural"),
    filter results: [m for m in results if m.get("metadata", {}).get("type") == memory_type]

    Return the filtered (or all) results as a list of dicts.
    Each result has: {"memory": str, "score": float, "metadata": dict}

    Print the number of memories found and their scores.
    """
    # TODO 5: implement here
    raise NotImplementedError

# ── TODO 6: Memory-augmented agent ────────────────────────────────────────────

async def compliance_agent_with_memory(
    memory_client,
    user_id: str,
    document: str,
    document_type: str,
    doc_id: str,
) -> dict:
    """
    TODO 6: Full compliance agent that uses memory in its system prompt.

    Steps:
    a) Search for relevant memories for this user + document type
       (search for: f"{document_type} compliance review preferences")

    b) Build dynamic system prompt:
       f\"\"\"You are a compliance review assistant.

       Your memory of past interactions with this user:
       {chr(10).join(f'- {m["memory"]}' for m in memories)}

       Review the document for compliance risks. Return JSON:
       {{"risk_level": "low|medium|high|critical", "reasoning": "...", "summary": "..."}}
       \"\"\"

    c) Call litellm.acompletion with gpt-4o-mini and the dynamic system prompt.

    d) Parse the JSON result.

    e) Store this run as episodic memory (call add_episodic_memory).

    f) Return the parsed result dict.
    """
    # TODO 6: implement here
    raise NotImplementedError

# ── TODO 7: Memory consolidation ─────────────────────────────────────────────

async def consolidate_old_memories(
    memory_client, user_id: str, older_than_days: int = 7
) -> int:
    """
    TODO 7: Consolidate old memories into a summary, then delete originals.

    Steps:
    a) Get all memories: memory_client.get_all(user_id=user_id)

    b) Filter memories older than older_than_days using metadata["timestamp"].
       Parse timestamps with datetime.fromisoformat().

    c) If no old memories, return 0.

    d) Ask the LLM to summarize the old memories:
       prompt = f"Summarize these agent memories in 3-5 bullet points:\n{chr(10).join(m['memory'] for m in old)}"
       summary = (await litellm.acompletion(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])).choices[0].message.content

    e) Add the summary as a new consolidated memory:
       memory_client.add([{"role":"system","content":f"Consolidated memory: {summary}"}],
                         user_id=user_id, metadata={"type":"consolidated"})

    f) Delete old memories: memory_client.delete(m["id"]) for each old memory.

    g) Return the count of deleted memories.
    """
    # TODO 7: implement here
    raise NotImplementedError

# ── TODO 8 (BONUS): User profile memory ──────────────────────────────────────

class UserProfile(BaseModel):
    user_id: str
    role: str
    department: str
    risk_tolerance: str   # "conservative" | "moderate" | "aggressive"
    notification_preference: str  # "immediate" | "daily_digest"

def upsert_user_profile(memory_client, profile: UserProfile) -> None:
    """
    TODO 8: Store/update user profile as long-term memory.

    Serialize profile to JSON and store with metadata={"type": "user_profile"}.
    Before adding, delete any existing user_profile memory for this user_id
    to avoid duplicates:
      all_mems = memory_client.get_all(user_id=profile.user_id)
      for m in all_mems:
          if m.get("metadata", {}).get("type") == "user_profile":
              memory_client.delete(m["id"])

    Then add the new profile.
    """
    # TODO 8: implement here
    raise NotImplementedError

def get_user_profile(memory_client, user_id: str) -> UserProfile | None:
    """
    TODO 8 (continued): Retrieve user profile from memory.

    Search for memory with metadata type == "user_profile" for this user.
    Parse JSON and return UserProfile, or None if not found.
    """
    # TODO 8: implement here
    raise NotImplementedError

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Mem0 Long-Term Memory Agent ===\n")

    memory = create_memory_client()
    user_id = "analyst_007"

    # Seed some memories
    print("1. Storing initial memories...")
    store_preference(memory, user_id, "I prefer detailed reasoning with specific regulatory citations")
    store_preference(memory, user_id, "Flag missing DPA clauses as high risk, not medium")
    store_learned_pattern(memory, user_id, "For vendor contracts over $500k, always escalate to legal review")
    add_episodic_memory(memory, user_id, "DOC-001", "contract", "high",
                        "Missing SOX §404 controls disclosure. Escalated to legal.")
    print("   ✓ 4 memories stored\n")

    # User profile
    print("2. Setting user profile...")
    upsert_user_profile(memory, UserProfile(
        user_id=user_id,
        role="Senior Compliance Analyst",
        department="Legal & Compliance",
        risk_tolerance="conservative",
        notification_preference="immediate",
    ))
    profile = get_user_profile(memory, user_id)
    print(f"   Profile: {profile.role} | {profile.department}\n")

    # Run agent with memory
    print("3. Running memory-augmented compliance review...")
    test_doc = """
    VENDOR SERVICES AGREEMENT
    This agreement between Acme Corp and DataVendor Ltd establishes data processing
    terms for CRM analytics services. Payment: $750,000 annually. Term: 3 years.
    Data: customer PII including email, phone, purchase history.
    DPA: To be attached as Exhibit C (pending legal review).
    """
    result = await compliance_agent_with_memory(
        memory, user_id, test_doc, "contract", "DOC-002"
    )
    print(f"   Risk level: {result.get('risk_level', 'unknown')}")
    print(f"   Reasoning:  {result.get('reasoning', '')[:120]}...\n")

    # Search memories
    print("4. Searching relevant memories...")
    hits = search_memory(memory, user_id, "vendor contract DPA requirements", limit=3)
    print(f"   Found {len(hits)} relevant memories")
    for m in hits:
        print(f"   [{m.get('score', 0):.2f}] {m['memory'][:80]}")

    # Memory consolidation (simulate)
    print("\n5. Memory consolidation (7-day-old memories)...")
    deleted = await consolidate_old_memories(memory, user_id, older_than_days=0)  # 0=all for demo
    print(f"   Consolidated and deleted {deleted} old memory entries")

    print("\n✅ Memory agent exercise complete!")

if __name__ == "__main__":
    asyncio.run(main())
