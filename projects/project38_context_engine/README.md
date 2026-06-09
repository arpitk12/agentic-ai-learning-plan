# Project 38 — Context Budget Engine

> **Stack**: tiktoken · LiteLLM · asyncio · Python 3.11+  
> **Theme**: System Design — Chapter 5 of `guide/13_system_design.md`  
> **Companion guide**: [`guide/13_system_design.md §5`](../../guide/13_system_design.md) | [`token_optimization_guide.md`](../../phase4_production/week8_observability/resources/token_optimization_guide.md)

---

## What You'll Build

A **production context budget engine** that precisely allocates the context window across 5 sources, enforces budgets, manages all 4 memory levels, and produces a before/after measurement report — as a standalone reusable module.

```
Total Context Window (e.g. 8,000 tokens)
       │
       ├── ALLOCATOR
       │     ├── System prompt:    400 tokens  (measured, optimized)
       │     ├── Long-term memory: 300 tokens  (retrieved from store)
       │     ├── RAG context:    2,000 tokens  (greedy fill by score)
       │     ├── History:        2,000 tokens  (managed, compressed)
       │     ├── Tool schemas:     400 tokens  (compacted)
       │     └── Output reserve: 2,900 tokens  (never touches this)
       │
       ├── ASSEMBLER
       │     └── Final messages array, in optimal order
       │
       └── REPORTER
             ├── Per-source token counts (actual vs budget)
             ├── Budget overrun warnings
             └── Before/after comparison (unmanaged vs managed)
```

The engine is a real module you can `import` and drop into any of the Phase 4–7 agents.

---

## Why This Project Matters

Every agent in this repo assembles context in an ad-hoc way. This project builds the missing infrastructure:
- **Prevents silent overruns** — no more hitting the context limit at call time
- **Attributes cost** — know exactly what each source costs
- **Forces discipline** — context is a shared resource; budget it like memory or disk

After this project, you'll be able to retrofit any agent with measured, enforced context management.

---

## System Design Concepts Covered

| Concept | Where in code |
|---|---|
| Token budget allocation (per source) | `ContextBudget` dataclass |
| The 4 memory levels | `ContextEngine.build_context()` |
| Context assembly order (primacy/recency) | `ContextEngine.assemble()` |
| Before/after measurement | `ContextReport.diff()` |
| Budget enforcement (hard stops) | `ContextAllocator.allocate()` |
| RAG greedy fill by relevance score | `ContextEngine._fill_rag()` |
| History sliding window + LLM summary | `ContextEngine._manage_history()` |
| Tool schema compaction (inline) | `ContextEngine._compact_tools()` |
| Long-term memory retrieval | `MemoryStore.retrieve()` |

---

## Milestones

### Milestone 1 — Token Counting
Implement `count_tokens(text, model)` using tiktoken with model→encoding mapping and `cl100k_base` fallback. Implement `count_messages_tokens(messages, model)` with the 4-token-per-message overhead formula.

### Milestone 2 — Context Budget
Implement `ContextBudget` dataclass with fields: `total`, `system_prompt`, `long_term_memory`, `rag_context`, `history`, `tool_schemas`, `output_reserve`. Add `compute_available()` → sum of non-output sources. Add `validate()` → raises `BudgetError` if sources + reserve > total.

### Milestone 3 — Memory Store
Implement `MemoryStore` with in-memory storage (for this project). Methods:
- `store(key, value, memory_type)` — store a fact
- `retrieve(query, top_k=5)` — return most relevant memories (simple keyword overlap scoring)
- `get_all(memory_type)` — return all memories of a type

### Milestone 4 — RAG Greedy Fill
Implement `ContextEngine._fill_rag(chunks, budget)`. Sort chunks by `score` field descending. Greedily add chunks until budget is exhausted. If a chunk is too large but has score > 0.7, compress with `compress_text()`. Return selected chunks and tokens used.

### Milestone 5 — History Management
Implement `ContextEngine._manage_history(messages, budget)`. Preserve system messages. Keep last `keep_last_n=4` turn-pairs. If over budget: summarise older turns with LLM (cheap model), inject as system message. Recurse until under budget.

### Milestone 6 — Tool Schema Compaction
Implement `ContextEngine._compact_tools(tools, budget)`. Strip descriptions to first sentence ≤80 chars. Remove `examples`, `default`, `title`. Abbreviate enums > 5. Return compacted tools and token count.

### Milestone 7 — Context Assembly
Implement `ContextEngine.assemble(...)`. Takes: system_prompt, history, rag_chunks, tools, question, memory_store. Returns `AssembledContext` with `messages` array and `report`. Assembly order: system → long-term memory → RAG → compressed history → (tool schemas: note in system) → current user message.

### Milestone 8 — Context Report
Implement `ContextReport` with fields per source (budget, actual, overrun). Methods:
- `render()` — formatted table
- `diff(unmanaged_tokens, managed_tokens)` — % savings
- `warnings()` — list sources that exceeded their budget

### Milestone 9 — Integration Demo
Wire everything together in `main()`. Show:
1. Unmanaged context (no engine) — count total tokens
2. Managed context (engine) — count total tokens
3. Print report with per-source breakdown and savings

---

## Setup

```bash
pip install litellm tiktoken python-dotenv pydantic
cp ../../.env.example ../../.env
python starter/starter.py
```

---

## Expected Output

```
═══════════════════════════════════════════════════════
 Context Budget Engine — Assembly Report
═══════════════════════════════════════════════════════

 Source            │ Budget   │ Actual   │ Status
──────────────────────────────────────────────────────
 System prompt     │   400    │   287    │ ✅ OK
 Long-term memory  │   300    │   198    │ ✅ OK
 RAG context       │  2,000   │  1,847   │ ✅ OK  (4/5 chunks)
 History           │  2,000   │  1,643   │ ✅ OK  (summarised 6 → 3 msgs)
 Tool schemas      │   400    │   312    │ ✅ OK  (compacted 2 tools)
 Current message   │   200    │    42    │ ✅ OK
 Output reserve    │  2,900   │    —     │ reserved
──────────────────────────────────────────────────────
 TOTAL             │  8,000   │  4,329   │ ✅ 46% headroom

Before/After:
  Unmanaged: 7,841 tokens   → ${0.00392} per call
  Managed:   4,329 tokens   → ${0.00217} per call
  Savings:   44.8%  (-3,512 tokens, -${0.00175}/call)
  At 10k calls/day:         -${17.50}/day  → -${525}/month
```

---

## Stretch Goals

- [ ] Persist `MemoryStore` to SQLite (use `aiosqlite`)
- [ ] Embed memories with sentence-transformers and retrieve by cosine similarity
- [ ] Add a `ContextBudgetProfiler` that runs 100 real agent calls and reports average per-source usage vs budget
- [ ] Export assembled context to JSON for debugging / replay
- [ ] Add a `strict_mode` that raises if any source exceeds its budget (vs silently truncating)
