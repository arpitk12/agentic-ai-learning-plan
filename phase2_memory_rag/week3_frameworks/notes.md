# Week 3 — LangGraph & Agent Frameworks

## Topics
1. LangChain: chains, LCEL pipeline syntax
2. LangGraph: nodes, edges, state machines
3. When to use a framework vs raw SDK
4. Tracing & observability with LangSmith

## Key Concepts

### LangGraph State Machine
Every agent is a graph:
- **Nodes**: Python functions that transform state
- **Edges**: routing logic (conditional or fixed)
- **State**: a TypedDict shared across all nodes

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_results: list
    final_answer: str | None

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", run_tools)
graph.add_conditional_edges("llm", route_after_llm)
graph.add_edge("tools", "llm")
graph.set_entry_point("llm")
app = graph.compile()
```

### Framework vs Raw SDK
| Use Framework | Use Raw SDK |
|---|---|
| Complex branching logic | Simple linear chains |
| Need built-in persistence | Full control needed |
| Team project (consistency) | Learning / prototyping |
| Human-in-the-loop flows | Cost optimization |

## Exercises
- `ex1_langgraph_basic.py` — rewrite Week 2 ReAct in LangGraph
- `ex2_conditional_routing.py` — branch on tool failure
- `ex3_langsmith_tracing.py` — trace an agent run

## Checklist
- [ ] Rewrote ReAct agent using LangGraph state machine
- [ ] Added fallback branch when tool fails
- [ ] Traced a full agent run in LangSmith
- [ ] Understood state persistence between nodes
