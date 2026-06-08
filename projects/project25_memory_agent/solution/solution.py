"""
Project 25 SOLUTION — Long-Term Memory Agent
Mem0 agent with episodic, semantic, procedural, and user-profile memory.

Architecture:
  - create_memory_client()   → Mem0 with local vector store
  - store_episode()          → episodic: past doc reviews
  - store_preference()       → semantic: user preferences
  - store_workflow_pattern() → procedural: learned workflows
  - upsert_profile()         → user profile: role/dept/risk_tolerance
  - compliance_agent()       → retrieves relevant memories → injects → calls LLM → stores result
  - consolidate_memories()   → compresses memories older than N days into summary
  - FastAPI REST API         → /analyze, /memories, /profile endpoints
"""
from __future__ import annotations
import os, json, asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import litellm
from dotenv import load_dotenv

load_dotenv()


# ── Memory Client ─────────────────────────────────────────────────────────────

def create_memory_client():
    from mem0 import Memory  # type: ignore
    return Memory()


# ── Episodic Memory ───────────────────────────────────────────────────────────

def store_episode(memory, user_id: str, doc_id: str, doc_type: str, risk: str, summary: str) -> str:
    result = memory.add(
        messages=[
            {"role": "user", "content": f"I reviewed document {doc_id} ({doc_type})"},
            {"role": "assistant", "content": f"Risk: {risk}. {summary}"},
        ],
        user_id=user_id,
        metadata={
            "type": "episodic", "doc_id": doc_id,
            "risk_level": risk,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result[0]["id"] if isinstance(result, list) else result.get("id", "")


# ── Semantic Memory ───────────────────────────────────────────────────────────

def store_preference(memory, user_id: str, preference: str) -> str:
    result = memory.add(
        messages=[{"role": "user", "content": preference}],
        user_id=user_id,
        metadata={"type": "semantic", "category": "preference"},
    )
    return result[0]["id"] if isinstance(result, list) else result.get("id", "")


# ── Procedural Memory ────────────────────────────────────────────────────────

def store_workflow_pattern(memory, user_id: str, pattern: str) -> str:
    result = memory.add(
        messages=[{"role": "user", "content": f"My workflow pattern: {pattern}"}],
        user_id=user_id,
        metadata={"type": "procedural", "category": "workflow"},
    )
    return result[0]["id"] if isinstance(result, list) else result.get("id", "")


# ── User Profile Memory ───────────────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id: str
    role: str
    department: str
    risk_tolerance: str  # conservative | moderate | aggressive

def upsert_profile(memory, profile: UserProfile) -> None:
    all_mems = memory.get_all(user_id=profile.user_id)
    if isinstance(all_mems, dict):
        all_mems = all_mems.get("results", [])
    for m in all_mems:
        if m.get("metadata", {}).get("type") == "user_profile":
            memory.delete(m["id"])
    memory.add(
        messages=[{"role": "user", "content": json.dumps({
            "user_id": profile.user_id, "role": profile.role,
            "department": profile.department, "risk_tolerance": profile.risk_tolerance,
        })}],
        user_id=profile.user_id,
        metadata={"type": "user_profile"},
    )

def get_profile(memory, user_id: str) -> UserProfile | None:
    all_mems = memory.get_all(user_id=user_id)
    if isinstance(all_mems, dict):
        all_mems = all_mems.get("results", [])
    for m in all_mems:
        if m.get("metadata", {}).get("type") == "user_profile":
            try:
                text = m.get("memory", "")
                start = text.find("{")
                if start != -1:
                    data = json.loads(text[start:])
                    return UserProfile(**data)
            except Exception:
                pass
    return None


# ── Memory-Augmented Agent ────────────────────────────────────────────────────

async def compliance_agent(
    memory, user_id: str, document: str, document_type: str, doc_id: str
) -> dict:
    # Retrieve relevant memories
    search_results = memory.search(
        query=f"{document_type} compliance review",
        user_id=user_id, limit=5,
    )
    if isinstance(search_results, dict):
        search_results = search_results.get("results", [])

    memory_block = "\n".join(f"- {m.get('memory', '')}" for m in search_results)

    system_prompt = f"""You are a compliance review assistant.

Memory from past interactions:
{memory_block or "(no relevant memories yet)"}

Review the document and return JSON:
{{"risk_level": "low|medium|high|critical", "reasoning": "...", "summary": "one sentence"}}"""

    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document type: {document_type}\n\n{document}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    result = json.loads(resp.choices[0].message.content)

    # Store result as new episodic memory
    store_episode(memory, user_id, doc_id, document_type,
                  result["risk_level"], result["summary"])
    return result


# ── Memory Consolidation ──────────────────────────────────────────────────────

async def consolidate_memories(memory, user_id: str, older_than_days: int = 7) -> int:
    all_mems = memory.get_all(user_id=user_id)
    if isinstance(all_mems, dict):
        all_mems = all_mems.get("results", [])

    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    old = []
    for m in all_mems:
        ts = m.get("metadata", {}).get("timestamp")
        if ts:
            try:
                t = datetime.fromisoformat(ts)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if t < cutoff:
                    old.append(m)
            except ValueError:
                pass

    if not old:
        return 0

    texts = "\n".join(f"- {m.get('memory', '')}" for m in old)
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize in 3-5 bullet points:\n{texts}"}],
    )
    summary = resp.choices[0].message.content.strip()

    memory.add(
        messages=[{"role": "system", "content": f"Consolidated summary: {summary}"}],
        user_id=user_id,
        metadata={"type": "consolidated", "timestamp": datetime.now(timezone.utc).isoformat()},
    )
    for m in old:
        try:
            memory.delete(m["id"])
        except Exception:
            pass
    return len(old)


# ── FastAPI REST API ──────────────────────────────────────────────────────────

def create_app():
    from fastapi import FastAPI, HTTPException  # type: ignore
    from pydantic import BaseModel as PM

    app = FastAPI(title="Memory Agent API")
    _memory = create_memory_client()

    class AnalyzeRequest(PM):
        user_id: str
        doc_id: str
        document_type: str
        document: str

    class PreferenceRequest(PM):
        user_id: str
        preference: str

    @app.post("/analyze")
    async def analyze(req: AnalyzeRequest):
        return await compliance_agent(_memory, req.user_id, req.document, req.document_type, req.doc_id)

    @app.get("/memories/{user_id}")
    async def get_memories(user_id: str):
        results = _memory.get_all(user_id=user_id)
        if isinstance(results, dict):
            results = results.get("results", [])
        return {"user_id": user_id, "memories": results, "count": len(results)}

    @app.post("/preferences")
    async def add_preference(req: PreferenceRequest):
        mem_id = store_preference(_memory, req.user_id, req.preference)
        return {"memory_id": mem_id}

    @app.get("/profile/{user_id}")
    async def get_user_profile(user_id: str):
        profile = get_profile(_memory, user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile

    return app


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 25: Memory Agent SOLUTION ===\n")

    memory = create_memory_client()
    user_id = "analyst_001"

    print("1. Setting up user profile and seed memories...")
    upsert_profile(memory, UserProfile(
        user_id=user_id, role="Compliance Analyst",
        department="Legal", risk_tolerance="conservative",
    ))
    store_preference(memory, user_id, "Always cite the specific regulatory clause")
    store_workflow_pattern(memory, user_id, "For contracts >$500k, escalate to legal review")
    store_episode(memory, user_id, "DOC-HIST-001", "contract", "high",
                  "Missing GDPR Art.28 DPA. Escalated to legal.")
    print("   ✓ Profile and 3 seed memories stored\n")

    print("2. Running memory-augmented compliance analysis...")
    result = await compliance_agent(
        memory, user_id,
        document="Cloud services agreement with AWS for EU data processing. "
                 "No Data Processing Agreement attached. Contract value: $1.2M annually.",
        document_type="contract",
        doc_id="DOC-NEW-001",
    )
    print(f"   Risk: {result['risk_level']}")
    print(f"   Reasoning: {result['reasoning'][:120]}...\n")

    print("3. Memory consolidation (consolidating all memories for demo)...")
    deleted = await consolidate_memories(memory, user_id, older_than_days=0)
    print(f"   Consolidated {deleted} memories into summary\n")

    print("4. To run the FastAPI server:")
    print("   app = create_app()")
    print("   uvicorn.run(app, host='0.0.0.0', port=8000)")

if __name__ == "__main__":
    asyncio.run(main())
