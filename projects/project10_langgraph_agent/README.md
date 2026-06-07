# Project 10 — LangGraph Decision Agent (HITL + Checkpointing)

## What You Build

A multi-step research-and-execute agent using **LangGraph's StateGraph**. The agent researches a topic, creates an action plan with a risk assessment, and routes HIGH/MEDIUM risk plans through a human approval node before executing. State is checkpointed so the agent can resume from any interruption.

## Production Skills Practised

| Skill | Guide Section |
|-------|--------------|
| LangGraph `StateGraph` + `TypedDict` state | §2.4 |
| `add_conditional_edges()` for routing | §2.4 |
| `MemorySaver` checkpointing (resume on crash) | §2.4 |
| `interrupt_before` for HITL pause | §2.4, §4.7 |
| Risk-aware routing (LOW → auto-execute, HIGH → human) | §4.7 |
| Structured LLM output with Pydantic + JSON parsing | §2.8 |

## Architecture — Graph Structure

```
[RESEARCH node]
      │
      ▼
[PLAN node]
      │
      ▼  route_after_plan()
      ├─── risk=LOW  ─────────────────► [EXECUTE node] ─► END
      └─── risk=MED/HIGH ──► [APPROVE node]
                                   │
                                   ▼  route_after_approval()
                                   ├─── approved=True  ─► [EXECUTE node] ─► END
                                   └─── approved=False ─► [REJECT node]  ─► END
```

## State Schema

```python
class ResearchState(TypedDict):
    query:            str           # original user query
    research_results: list[str]     # gathered information
    plan:             str           # the proposed action plan
    risk_level:       str           # "LOW" | "MEDIUM" | "HIGH"
    approved:         bool          # human approval result
    execution_result: str           # final output
    steps:            int           # step counter
```

## Setup

```bash
pip install litellm python-dotenv pydantic langgraph
```

## Usage

```bash
# A low-risk research task (auto-executes)
python starter.py "Summarise the top 3 multi-agent design patterns"

# A higher-stakes task (triggers HITL approval)
python starter.py "Draft and send a public technical blog post about AI safety"

# Full solution with checkpointing + resumption
python solution.py "Analyse the trade-offs of LangGraph vs CrewAI"
```

## What To Implement (5 TODOs)

1. **`research_node(state)`** — gather information, update `research_results`
2. **`plan_node(state)`** — LLM creates plan + risk level (JSON output)
3. **`route_after_plan(state)`** — return `"human_approval"` or `"execute"`
4. **`execute_node(state)`** — implement the plan using research context
5. **Build the graph** — `add_node`, `add_edge`, `add_conditional_edges`, `compile`

## Key Insight: Why LangGraph?

With a raw loop you get: works → breaks → impossible to debug → impossible to resume.
With LangGraph you get:
- **Explicit state**: every node gets typed state, every output is a state patch
- **Checkpointing**: if the agent crashes at step 3 of 5, it resumes from step 3
- **HITL natively**: `interrupt_before=["execute"]` pauses graph before risky steps
- **Visual graph**: call `agent.get_graph().draw_mermaid()` to see the exact flow
