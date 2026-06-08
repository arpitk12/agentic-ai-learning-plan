# Project 35 — Advanced Reasoning (Tree of Thought + o3 + Process Reward Models)

> **Stack**: LiteLLM · asyncio · Pydantic · OpenAI o3  
> **Phase 7 — Advanced Production** | Priority: P3 🟢

---

## What You'll Build

Advanced reasoning strategies for agent tasks that standard ReAct/chain-of-thought fails on: multi-step planning, logical deduction across long chains, and knowing when to use extended-thinking models vs fast models.

---

## Reasoning Strategy Selection

```
Task complexity assessment
    │
    ├─ Simple (classification, extraction) → standard gpt-4o-mini (fast, cheap)
    ├─ Medium (analysis, multi-step)       → gpt-4o with chain-of-thought
    ├─ Hard (planning, logic, math)        → Tree of Thought (parallel branches)
    └─ Very hard (proofs, complex plans)   → o3-mini with extended thinking
```

---

## Milestones

### Milestone 1 — Tree of Thought Implementation
Build a BFS Tree of Thought that explores `breadth=3` reasoning branches at `depth=3` levels. At each node: generate 3 continuations, evaluate (promising/partial/dead_end/score), prune dead ends, keep top 3.

### Milestone 2 — Thought Evaluation
Train a few-shot evaluator (using GPT-4o-mini) to score intermediate reasoning steps on: logical consistency, evidence support, progress toward answer. Compare: ToT with learned evaluator vs random evaluator.

### Milestone 3 — o3 Integration + Routing
Build a task complexity classifier that routes to: gpt-4o-mini (fast), gpt-4o (medium), o3-mini (hard). Test with a benchmark of 30 tasks at different complexity levels. Measure cost vs accuracy tradeoff.

### Milestone 4 — Process Reward Model
Collect reasoning traces (correct and incorrect) for 50 problems. Fine-tune a reward model (using QLoRA from Project 24) to score intermediate steps. Use PRM to guide ToT branch selection.

### Milestone 5 — Self-Consistency Sampling
For uncertain tasks: generate 5 independent reasoning chains, take the majority vote answer. Measure: single-chain accuracy vs 5-sample majority vote accuracy on a logic benchmark.

### Milestone 6 — Monte Carlo Tree Search (MCTS)
Implement basic MCTS for multi-step planning tasks: `UCT = Q/N + C*sqrt(ln(N_parent)/N)`. Use for: long-horizon task planning, game-like scheduling problems, resource allocation.

### Milestone 7 — Benchmark Suite
Create 30 benchmark problems across 3 categories:
- **Logic**: syllogisms, constraint satisfaction
- **Planning**: multi-step scheduling, resource allocation
- **Math**: word problems, proof steps

Run all 4 strategies (CoT, ToT, self-consistency, o3) and print a comparison table.

---

## Setup

```bash
pip install litellm openai pydantic asyncio python-dotenv scipy
# o3 requires OpenAI API key with Tier 3+ access
# For o3 without access: use anthropic/claude-3-5-sonnet as substitute
```

---

## Expected Output

```
=== Advanced Reasoning Benchmark (30 problems) ===

Logic problems (10):
  Standard CoT (gpt-4o-mini): 5/10 (50%) — avg $0.0003/problem
  Tree of Thought (depth=3):   8/10 (80%) — avg $0.0024/problem
  Self-consistency (n=5):      7/10 (70%) — avg $0.0015/problem
  o3-mini:                     9/10 (90%) — avg $0.0180/problem

Planning problems (10):
  Standard CoT:         3/10 (30%)
  Tree of Thought:      7/10 (70%) ← best value (3x better, 8x cost)
  o3-mini:              9/10 (90%)

Math problems (10):
  Standard CoT:         6/10 (60%)
  Self-consistency:     8/10 (80%)
  o3-mini:             10/10 (100%)

Recommendation:
  - Logic < medium: gpt-4o-mini CoT (fast + cheap)
  - Logic > medium + Planning: Tree of Thought (3x accuracy, 8x cost → worth it)
  - Math + Proofs: o3-mini (no alternative — 40% accuracy gap vs CoT)
  - Cost-sensitive planning: ToT with gpt-4o-mini thoughts (best value)
```
