# Project 19 — LangGraph: Stateful Code Review Workflow

A **production-grade LangGraph application** demonstrating stateful multi-step workflows:
a code review pipeline where each stage is a graph node, edges encode routing logic,
human approval can pause and resume the graph, and every state is checkpointed to SQLite
for crash recovery.

---

## 🎯 What You Learn

| Concept | Where |
|---------|-------|
| **StateGraph** — define graph with typed state | `src/graph/graph.py` |
| **TypedDict state** — shared memory between nodes | `src/graph/state.py` |
| **Nodes** — functions that update state | `src/graph/nodes.py` |
| **Conditional edges** — routing based on state | `src/graph/graph.py` |
| **Checkpointer** — SQLite persistence + replay | `src/graph/checkpointer.py` |
| **Human-in-the-loop** — `interrupt()` + resume | `src/graph/nodes.py` |
| **Streaming** — `stream()` event-by-event | `src/api/routes.py` |
| **Subgraphs** — compose graphs | `src/graph/subgraphs.py` |

---

## 🏗 Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║              CODE REVIEW STATE GRAPH                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   START                                                          ║
║     │                                                            ║
║   [parse_code]        ← extract language, functions, patterns   ║
║     │                                                            ║
║   [analyze_security]  ← check for vulnerabilities (OWASP top 10)║
║     │                                                            ║
║   [analyze_quality]   ← check style, complexity, coverage hints ║
║     │                                                            ║
║   [generate_review]   ← LLM writes full review with suggestions ║
║     │                                                            ║
║   [human_approval] ←──── INTERRUPT ────────────────────────────║
║     │  human edits / approves / rejects                         ║
║     │                                                            ║
║   ┌─┴──────────────────────────────────┐                        ║
║   │  route_approval (conditional edge) │                        ║
║   └─────┬─────────────────────┬────────┘                        ║
║         │ approved            │ needs_revision                  ║
║         │                     │                                  ║
║   [finalize_review]   [revise_review] ──► back to [generate]   ║
║         │                                                        ║
║       END                                                        ║
║                                                                  ║
║  Checkpointer: SqliteSaver — survives restarts, enables replay  ║
║  State: TypedDict with Annotated[list, operator.add] reducers   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📁 Folder Structure

```
project19_langgraph_workflow/
├── README.md
├── GUIDE.md
├── starter/
│   ├── requirements.txt
│   ├── .env.example
│   └── src/
│       ├── graph/
│       │   ├── state.py          ← TODO (5 tasks) — TypedDict state schema
│       │   ├── nodes.py          ← TODO (10 tasks) — all node functions
│       │   ├── graph.py          ← TODO (8 tasks) — build + compile graph
│       │   └── checkpointer.py   ← TODO (3 tasks) — SQLite checkpointer
│       ├── tools/
│       │   └── code_tools.py     ← TODO (4 tasks) — AST analysis tools
│       └── api/
│           └── routes.py         ← TODO (4 tasks) — FastAPI streaming endpoint
└── solution/
    └── src/
```

---

## ⚡ Key LangGraph Patterns

| Pattern | Code | Why |
|---------|------|-----|
| State schema | `class State(TypedDict): messages: Annotated[list, operator.add]` | Reducer merges lists |
| Add node | `graph.add_node("name", fn)` | fn: State → dict |
| Conditional edge | `graph.add_conditional_edges("node", router_fn, {"a": "nodeA", "b": "nodeB"})` | Dynamic routing |
| Compile | `graph.compile(checkpointer=SqliteSaver("db.sqlite"))` | Enable persistence |
| Invoke | `graph.invoke(state, config={"configurable": {"thread_id": "t1"}})` | Per-thread state isolation |
| Interrupt | `interrupt("reason")` inside node | Pauses for human input |
| Resume | `graph.invoke(Command(resume=human_input), config=...)` | Continues from interrupt |
| Stream | `for event in graph.stream(state, config=...): ...` | Real-time node events |

---

## 🚀 Quick Start

```bash
cd projects/project19_langgraph_workflow/starter
pip install -r requirements.txt
cp .env.example .env

# Run a review (non-interactive)
python -m src.main --code "def add(a, b): return a + b" --auto-approve

# Run with human approval gate
python -m src.main --code "$(cat my_file.py)" --interactive

# FastAPI server with streaming
uvicorn src.api.app:app --reload
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"code": "def divide(a,b): return a/b", "language": "python"}'
```

---

## Milestones

1. **State** — implement `state.py`, verify TypedDict + reducers
2. **Nodes** — implement each node function, test individually
3. **Graph** — wire nodes + edges, compile, run without checkpointer
4. **Checkpointer** — add SQLite, verify state survives restart
5. **Human-in-the-loop** — add `interrupt()`, test pause + resume
6. **API** — expose via FastAPI with streaming events
