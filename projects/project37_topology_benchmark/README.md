# Project 37 — Agent Topology Benchmark

> **Stack**: LiteLLM · asyncio · Python 3.11+ · rich (terminal tables)  
> **Theme**: System Design — Chapter 2 of `guide/13_system_design.md`  
> **Companion guide**: [`guide/13_system_design.md §2`](../../guide/13_system_design.md)

---

## What You'll Build

A benchmark harness that runs the **same task** through five distinct agent topologies and produces a side-by-side decision matrix — quality score, latency, cost, LLM calls, and step count.

```
                      Task: "Research and write a 200-word explainer on GDPR Article 28"
                                              │
              ┌──────────────┬───────────────┼──────────────┬──────────────┐
              ▼              ▼               ▼              ▼              ▼
        Single Agent   Orchestrator   Sequential     Fan-Out       Debate
        (ReAct loop)    + Workers      Pipeline      Parallel     (Adversarial)
              │              │               │              │              │
              └──────────────┴───────────────┴──────────────┴──────────────┘
                                              │
                                   Evaluation + Scoring
                                   (LLM judge: 0–10 quality)
                                              │
                                   Decision Matrix Report:
                              ┌──────────────────────────────────────┐
                              │ Topology       │ Quality │ Cost  │ ms │
                              │ Single-ReAct   │  7.2    │ $0.003│ 4.2│
                              │ Orch-Worker    │  8.8    │ $0.012│12.1│
                              │ Pipeline       │  8.1    │ $0.009│ 8.7│
                              │ Fan-Out        │  8.4    │ $0.011│ 5.3│
                              │ Debate         │  9.1    │ $0.021│18.4│
                              └──────────────────────────────────────┘
```

---

## Why This Project Matters

You can read about agent patterns in any blog post. **Actually measuring them** on identical tasks reveals:
- When the extra cost of orchestrator-worker is worth it vs single ReAct
- How much quality lift debate adds (and whether you can afford it)
- Why fan-out doesn't always win on latency (coordination overhead)
- Which topology to default to for YOUR use case

This is the "show, don't tell" version of `guide/13_system_design.md §2`.

---

## System Design Concepts Covered

| Concept | Where in code |
|---|---|
| ReAct loop internals (think → act → observe) | `SingleReActAgent` |
| Orchestrator decomposition → synthesis | `OrchestratorWorkerTopology` |
| Pipeline (structured data, not free-form strings) | `SequentialPipelineTopology` |
| asyncio fan-out with `gather()` | `FanOutTopology` |
| Adversarial review loop + quality gate | `DebateTopology` |
| Max steps, per-step timeout, cost cap | All topologies via `AgentGuards` |
| LLM-as-judge evaluation | `Evaluator.score()` |
| Structured benchmark reporting | `BenchmarkReport.render()` |

---

## Milestones

### Milestone 1 — AgentGuards
Build a `AgentGuards` dataclass that tracks steps, elapsed time, and running cost. Raises `StepLimitError`, `TimeoutError`, or `CostCapError` when limits are breached. Used by all topologies.

### Milestone 2 — Single ReAct Agent
Implement `SingleReActAgent.run(task)`. Standard ReAct loop: think → tool call (mock `web_search`) → observe → final answer. Guards: max 8 steps, 30s timeout, $0.05 cost cap.

### Milestone 3 — Orchestrator-Worker
Implement `OrchestratorWorkerTopology.run(task)`. Planner LLM decomposes task into 3 subtasks (JSON), runs each subtask with a specialist prompt in sequence, synthesizer LLM merges results.

### Milestone 4 — Sequential Pipeline
Implement `SequentialPipelineTopology.run(task)`. Three fixed stages with Pydantic models as handoffs:
- Stage 1 (Researcher): returns `ResearchOutput(facts: list[str], sources: list[str])`
- Stage 2 (Writer): consumes `ResearchOutput`, returns `DraftOutput(text: str, word_count: int)`
- Stage 3 (Editor): consumes `DraftOutput`, returns `FinalOutput(text: str, changes: list[str])`

### Milestone 5 — Fan-Out Parallel
Implement `FanOutTopology.run(task)`. Split task into 3 aspects, run all 3 specialist agents in parallel with `asyncio.gather()`. Merger agent combines into final answer.

### Milestone 6 — Debate
Implement `DebateTopology.run(task)`. Proposer writes initial answer. Critic identifies flaws (rated CRITICAL/HIGH/MEDIUM/LOW). Proposer revises. Loop until no CRITICAL/HIGH issues or max 3 rounds.

### Milestone 7 — Evaluator
Implement `Evaluator.score(task, answer)` using LLM-as-judge. Score 0–10 on: accuracy, completeness, conciseness, structure. Return structured `EvalResult`.

### Milestone 8 — Benchmark Runner
Implement `BenchmarkRunner.run(tasks, topologies)`. For each task × topology: run with timing, catch errors, record cost, call Evaluator. Return `BenchmarkReport`.

### Milestone 9 — Decision Matrix
Render a rich terminal table + markdown export showing all topologies side-by-side across all metrics. Include a "recommended for" row based on measured trade-offs.

---

## Setup

```bash
pip install litellm python-dotenv pydantic rich asyncio
cp ../../.env.example ../../.env
# Edit .env — any LLM works (Groq is free)
python starter/starter.py
```

---

## Expected Output

```
═══════════════════════════════════════════════════════════════
 AGENT TOPOLOGY BENCHMARK — Task: GDPR Article 28 Explainer
═══════════════════════════════════════════════════════════════

┌──────────────────────┬─────────┬────────┬──────────┬────────┬──────────┐
│ Topology             │ Quality │  Cost  │ Latency  │ Calls  │  Steps   │
│                      │  /10    │  USD   │    s     │  LLM   │  Agent   │
├──────────────────────┼─────────┼────────┼──────────┼────────┼──────────┤
│ Single ReAct         │   7.2   │ $0.003 │   4.2    │   3    │    3     │
│ Orchestrator-Worker  │   8.8   │ $0.012 │  12.1    │   5    │    5     │
│ Sequential Pipeline  │   8.1   │ $0.009 │   8.7    │   4    │    3     │
│ Fan-Out Parallel     │   8.4   │ $0.011 │   5.3    │   5    │    2     │
│ Debate (3 rounds)    │   9.1   │ $0.021 │  18.4    │   7    │    6     │
└──────────────────────┴─────────┴────────┴──────────┴────────┴──────────┘

Recommendations:
  Best quality/cost:  Fan-Out Parallel  (quality 8.4, cost $0.011)
  Fastest:            Single ReAct      (4.2s, quality 7.2)
  Highest quality:    Debate            (9.1, 3× cost of fan-out)
  Safest default:     Sequential Pipeline (predictable, structured)

Decision Matrix (when to use each):
  Single ReAct      → Simple tasks, tight latency (<5s), cost-sensitive
  Orch-Worker       → Complex tasks with distinct specializations
  Sequential        → Quality gates required between stages
  Fan-Out           → Independent subtasks, moderate latency budget
  Debate            → High-stakes decisions, quality >> cost
```

---

## Stretch Goals

- [ ] Add a 6th topology: **Map-Reduce** (chunk a 500-word doc, summarize each chunk, reduce)
- [ ] Run on 5 different tasks and average results (research / coding / analysis / creative / QA)
- [ ] Export benchmark results to JSON for further analysis
- [ ] Plot latency vs quality scatter (matplotlib / plotly)
- [ ] Add cascade routing: try single ReAct first, escalate to debate if quality < 7.0
