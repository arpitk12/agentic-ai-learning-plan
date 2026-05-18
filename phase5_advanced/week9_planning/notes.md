# Week 9 — Planning & Self-Correction

## Topics
1. Plan-and-Execute vs ReAct — tradeoffs
2. Tree of Thought, LATS (LLM-as-tree-search)
3. Self-reflection loops: agent critiques its own output
4. Reflexion pattern: learn from failures across episodes

## Key Concepts

### Plan-and-Execute vs ReAct
| ReAct | Plan-and-Execute |
|---|---|
| Interleaved reasoning + action | Plan all steps first, then execute |
| Adapts to tool results dynamically | More predictable, easier to debug |
| Better for exploratory tasks | Better for well-defined multi-step tasks |
| Can get stuck in loops | Can fail if plan is wrong upfront |

### Self-Reflection Loop
```
Generate answer
    → Critic agent: score 1-10 with reasoning
    → If score < threshold: regenerate with critic feedback
    → Loop until score >= threshold or max retries
```

### Reflexion Pattern
Unlike one-shot self-reflection, Reflexion persists lessons:
```python
# After each failed attempt
failure_log.append({
    "attempt": attempt_num,
    "action": what_was_tried,
    "outcome": what_went_wrong,
    "lesson": llm_derived_lesson
})
# Next attempt includes full failure_log in context
```

### Tree of Thought
Generate multiple reasoning branches:
```
Question
├── Approach A → step 1a → step 2a → answer_a
├── Approach B → step 1b → step 2b → answer_b  ← best score
└── Approach C → step 1c → (pruned, low score)
```
Evaluate each branch and continue most promising.

## Exercises
- `ex1_plan_execute.py` — plan first, then execute each step
- `ex2_self_reflection.py` — critic loop with retry
- `ex3_reflexion.py` — multi-episode with failure memory
- `ex4_tree_of_thought.py` — branching search

## Checklist
- [ ] Implemented plan-and-execute for a 5-step task
- [ ] Critic loop improves output quality measurably
- [ ] Reflexion agent learns from failure across 3 episodes
- [ ] Tree-of-thought outperforms single-shot on hard problem
