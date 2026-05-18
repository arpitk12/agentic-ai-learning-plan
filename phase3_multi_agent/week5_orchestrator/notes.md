# Week 5 — Orchestrator / Subagent Patterns

## Topics
1. Orchestrator–worker pattern, task delegation
2. Subagent specialization: planner, executor, critic, verifier
3. Message passing: shared state vs message bus
4. Supervisor pattern with human-in-the-loop checkpoints

## Key Concepts

### Orchestrator Pattern
```
User → Orchestrator → [Planner, Executor, Critic] → Result
```
The orchestrator:
- Breaks work into subtasks
- Assigns to specialized agents
- Aggregates results
- Decides when work is complete

### Specialist Agent Roles
| Role | Responsibility |
|---|---|
| Planner | Decompose task into steps |
| Executor | Carry out a single step |
| Critic | Review output for quality/errors |
| Verifier | Check facts, run tests |
| Summarizer | Synthesize multiple outputs |

### Human-in-the-Loop
Add approval gates before destructive actions:
```python
def should_interrupt(state):
    return state["action_type"] in ["delete", "send_email", "deploy"]

graph.add_conditional_edges("executor", should_interrupt, 
    {True: "human_approval", False: "critic"})
```

## Exercises
- `ex1_three_agent_pipeline.py` — Planner → Executor → Critic
- `ex2_human_approval.py` — interrupt before destructive actions
- `ex3_debate_pattern.py` — 2 agents debate, 3rd votes

## Checklist
- [ ] Built 3-agent pipeline in LangGraph
- [ ] Added human approval checkpoint
- [ ] Implemented debate pattern
- [ ] Measured quality improvement vs single agent
