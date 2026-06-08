# Project 25 — Long-Term Memory Agent (Mem0 + All 4 Memory Types)

> **Stack**: Mem0 · LiteLLM · Qdrant · FastAPI · asyncio  
> **Phase 7 — Advanced Production** | Priority: P0 🔴

---

## What You'll Build

A personal assistant agent that remembers **everything** across sessions, users, and weeks — using all four memory types that mirror human long-term memory.

```
User Session 1 (Monday)
  User: "I prefer detailed reasoning in risk reports"
  Agent: stores → semantic memory (preference)
  Agent: reviews DOC-001, risk=high → stores → episodic memory

User Session 2 (Thursday — new process started)
  Agent: retrieves relevant memories before responding
  Agent: "Based on your preference for detailed reasoning..."
         automatically provides more thorough analysis
         "DOC-001 context: you reviewed a similar contract last Monday..."
```

---

## The Four Memory Types

| Type | What | Example |
|---|---|---|
| **Episodic** | Past events | "On Monday I reviewed contract DOC-001 and flagged it high risk" |
| **Semantic** | Facts + preferences | "User prefers detailed reasoning. User is conservative risk-tolerant." |
| **Procedural** | Learned workflows | "For vendor contracts >$500k, escalate to legal review" |
| **User profile** | Persistent attributes | role=analyst, dept=legal, timezone=UTC+1 |

---

## Milestones

### Milestone 1 — Mem0 Setup + Basic Add/Search
Install and configure Mem0 with a local vector store. Add 10 memories of each type. Search and verify retrieval quality (score > 0.8 for relevant queries).

### Milestone 2 — Memory-Augmented System Prompt
Build a function that retrieves relevant memories for a given user+query, formats them into a system prompt, and runs an LLM call. Verify the agent's responses change based on retrieved memories.

### Milestone 3 — Episodic Memory from Agent Runs
After every compliance review, automatically extract and store: document type, risk level, key concerns, and outcome. Build a retriever that can answer "what did we review last week?"

### Milestone 4 — Memory Consolidation
Implement a nightly consolidation job: compress memories older than 7 days into summaries. Verify the total memory count stays bounded while preserving key information.

### Milestone 5 — User Profile Memory
Build a user profile that persists across sessions. Store and retrieve: role, department, risk tolerance, preferred output format, notification preferences. Auto-inject profile into every session.

### Milestone 6 — Multi-User Isolation
Verify that User A's memories never appear in User B's context. Build a test that confirms cross-user isolation even when both users have reviewed similar documents.

### Milestone 7 — FastAPI Endpoints
Expose the memory agent via REST:
- `POST /chat` — chat with memory injection
- `GET /memories/{user_id}` — list all memories
- `DELETE /memories/{user_id}` — clear all memories
- `POST /memories/{user_id}/consolidate` — trigger consolidation

---

## Setup

```bash
cd projects/project25_memory_agent
pip install mem0ai litellm qdrant-client fastapi uvicorn pydantic python-dotenv

# Optional: Qdrant for production vector store
docker run -p 6333:6333 qdrant/qdrant
```

---

## Expected Output

```
=== Memory Agent Demo ===

Session 1 (storing memories):
  ✓ Stored preference: "detailed reasoning in risk reports"
  ✓ Stored pattern: "contracts >$500k → escalate to legal"
  ✓ Stored episode: DOC-001 review (high risk, missing DPA)

Session 2 (memory retrieval):
  Searching: "vendor contract compliance review"
  Found 4 relevant memories (scores: 0.94, 0.91, 0.87, 0.82)

  Agent response (with memory):
  "Based on your preference for detailed reasoning, and noting that
   you reviewed a similar vendor contract (DOC-001) last Monday
   which was flagged high risk for missing DPA — this contract
   shows similar characteristics..."

Consolidation:
  Compressed 12 old memories → 2 summary entries
  Memory count: 12 → 6 (50% reduction, 0% information loss)
```
