# Week 9 — Advanced Planning: Plan-Execute, Tree of Thought & Reflexion

## What This Week Is About
Basic ReAct agents think one step at a time. For complex, multi-step problems — writing a 10-page report, debugging a complex codebase, planning a multi-week project — you need explicit planning strategies. This week covers Plan-and-Execute, Tree of Thought, Reflexion, and LATS (Language Agent Tree Search).

---

## 1. The Limits of Pure ReAct

ReAct works well when:
- The next step is obvious from the current state
- Tasks are 3-7 steps long
- Errors are recoverable

ReAct fails when:
- Tasks need upfront planning (e.g., "write a thesis on AI ethics")
- The agent needs to explore multiple solution paths
- Mistakes in early steps compound catastrophically
- The optimal sequence requires foresight

**The insight**: Humans don't just react — we plan first, then execute.

---

## 2. Plan-and-Execute Architecture

Split the agent into two distinct phases:
1. **Planner**: Takes the goal, produces an ordered list of steps (no tool calls)
2. **Executor**: Works through each step, using tools, updating the plan if needed

```python
from llm import chat, get_text
import json

def planner(goal: str, context: str = "") -> list[dict]:
    """
    Creates a detailed execution plan for achieving the goal.
    Returns a list of steps with assigned agents and expected outputs.
    """
    system = """You are an expert project planner. Given a goal, create a detailed, 
executable plan as a JSON list of steps.

Each step must have:
- step_id: integer (1, 2, 3...)
- description: what to do in this step
- tool: which tool/agent handles this ("web_search", "code_writer", "analyzer", "synthesizer")
- depends_on: list of step_ids that must complete first
- expected_output: what this step should produce

Think carefully about ordering and dependencies. Steps with no dependencies can run in parallel."""
    
    response = get_text(chat(
        messages=[{
            "role": "user",
            "content": f"Create an execution plan for this goal:\n{goal}\n\nContext:\n{context}"
        }],
        system=system
    ))
    
    # Parse JSON plan
    clean = response.strip().strip("```json").strip("```").strip()
    plan = json.loads(clean)
    return plan if isinstance(plan, list) else plan.get("steps", [])

def executor(step: dict, completed_steps: dict) -> str:
    """Execute a single step using the appropriate tool/agent."""
    tool = step["tool"]
    
    # Build context from completed steps
    context = "\n\n".join([
        f"Step {sid} result: {result}"
        for sid, result in completed_steps.items()
        if sid in step.get("depends_on", [])
    ])
    
    task = f"Task: {step['description']}\n\nContext from previous steps:\n{context}"
    
    if tool == "web_search":
        return web_search_agent(task)
    elif tool == "code_writer":
        return code_agent(task)
    elif tool == "analyzer":
        return analyzer_agent(task)
    elif tool == "synthesizer":
        return synthesizer_agent(task, context)
    else:
        return get_text(chat([{"role": "user", "content": task}]))

def plan_and_execute(goal: str, max_replanning: int = 2) -> str:
    """Full Plan-and-Execute agent."""
    plan = planner(goal)
    completed = {}
    
    print(f"Plan created: {len(plan)} steps")
    
    for step in plan:
        step_id = step["step_id"]
        
        # Check dependencies
        missing_deps = [d for d in step.get("depends_on", []) if d not in completed]
        if missing_deps:
            print(f"Waiting for steps: {missing_deps}")
            continue
        
        print(f"Executing step {step_id}: {step['description']}")
        result = executor(step, completed)
        completed[step_id] = result
        
        # Check if replanning is needed
        if "CANNOT COMPLETE" in result or "FAILED" in result:
            if max_replanning > 0:
                print(f"Step {step_id} failed. Replanning...")
                new_context = f"Previous attempt failed at step {step_id}: {result}"
                plan = planner(goal, context=new_context)
                max_replanning -= 1
    
    # Final synthesis
    all_results = "\n\n".join(f"Step {sid}: {res}" for sid, res in completed.items())
    return get_text(chat([{
        "role": "user",
        "content": f"Goal: {goal}\n\nResults from all steps:\n{all_results}\n\nSynthesize a final answer."
    }]))
```

---

## 3. Tree of Thought (ToT)

Instead of a linear plan, explore multiple solution paths in parallel and pick the best one.

**The idea**: For each decision point, generate N candidate thoughts. Evaluate each. Keep the most promising. Continue expanding the tree.

```python
import asyncio

async def tree_of_thought(problem: str, branches: int = 3, depth: int = 3) -> str:
    """
    Tree of Thought reasoning.
    branches: number of candidate thoughts per level
    depth: how many levels deep to explore
    """
    
    async def generate_thoughts(parent_thought: str, problem: str, n: int) -> list[str]:
        """Generate n candidate next-step thoughts."""
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"""Problem: {problem}

Current reasoning: {parent_thought}

Generate {n} different, distinct next reasoning steps. 
Number each one (1., 2., 3., ...).
Each should explore a different approach or angle."""
            }],
        )
        text = response.choices[0].message.content
        # Parse numbered thoughts
        thoughts = [line.strip() for line in text.split("\n") if line.strip() and line[0].isdigit()]
        return thoughts[:n]
    
    async def evaluate_thought(thought: str, problem: str) -> float:
        """Score a thought from 0-10 for promise."""
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"""Problem: {problem}

Reasoning step: {thought}

Rate this reasoning step from 0-10 for:
- Logical correctness
- Progress toward solution  
- Avoiding dead ends

Reply with just a number (0-10)."""
            }],
            max_tokens=10
        )
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 5.0
    
    # Initialize with root thoughts
    current_thoughts = await generate_thoughts("", problem, branches)
    
    # Expand tree
    for level in range(depth - 1):
        # Evaluate all current thoughts
        scores = await asyncio.gather(*[evaluate_thought(t, problem) for t in current_thoughts])
        
        # Keep top-K thoughts (beam search)
        scored = sorted(zip(scores, current_thoughts), reverse=True)
        best_thoughts = [t for _, t in scored[:branches]]
        
        print(f"Level {level+1}: Best thought score = {scored[0][0]:.1f}")
        
        # Expand best thoughts
        expanded = await asyncio.gather(*[
            generate_thoughts(thought, problem, branches)
            for thought in best_thoughts
        ])
        current_thoughts = [t for sublist in expanded for t in sublist][:branches * 2]
    
    # Final answer from best thought
    best_final_thought = current_thoughts[0]
    return await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": f"Problem: {problem}\n\nBest reasoning path:\n{best_final_thought}\n\nProvide the final answer."}]
    ).then(lambda r: r.choices[0].message.content)
```

---

## 4. Reflexion — Learning from Mistakes

**Reflexion** (Shinn et al., 2023) makes agents learn from failure by generating verbal feedback about what went wrong and using it to retry.

```
Attempt 1 → Failed
     ↓
Reflection: "I searched too broadly. I should search for the specific paper title."
     ↓
Attempt 2 → Failed
     ↓
Reflection: "The paper is from 2019, I should add the year to my search."
     ↓
Attempt 3 → Success
```

```python
def reflexion_agent(task: str, max_attempts: int = 3) -> str:
    reflections = []
    
    for attempt in range(max_attempts):
        # Build context from past reflections
        reflection_context = ""
        if reflections:
            reflection_context = "\n\nLearnings from previous attempts:\n" + "\n".join(
                f"Attempt {i+1}: {r}" for i, r in enumerate(reflections)
            )
        
        # Attempt the task
        print(f"\nAttempt {attempt + 1}/{max_attempts}")
        result = react_agent(task + reflection_context)
        
        # Evaluate the result
        evaluation = get_text(chat(
            messages=[{
                "role": "user",
                "content": f"""Task: {task}
                
Result achieved: {result}

Evaluate if this result fully and correctly completes the task.
Reply with JSON: {{"success": true|false, "reason": "explanation of any gaps or errors"}}"""
            }],
            system="Be strict. Only mark success if the task is fully and correctly completed."
        ))
        
        import json, re
        clean = re.sub(r"```json?\s*|\s*```", "", evaluation).strip()
        eval_result = json.loads(clean)
        
        if eval_result["success"]:
            print(f"Task completed successfully on attempt {attempt + 1}")
            return result
        
        # Generate reflection for next attempt
        reflection = get_text(chat(
            messages=[{
                "role": "user",
                "content": f"""Task: {task}
                
My attempt: {result}
Why it failed: {eval_result['reason']}

Write a short, actionable reflection on what I should do differently in my next attempt.
Focus on concrete changes to strategy, search terms, approach, or reasoning."""
            }],
            system="Be specific and actionable. Write 2-3 sentences."
        ))
        
        reflections.append(f"Failed because: {eval_result['reason']}. Next time: {reflection}")
        print(f"Reflection: {reflection}")
    
    return f"Could not complete task after {max_attempts} attempts. Final attempt: {result}"
```

---

## 5. LATS — Language Agent Tree Search

**LATS** (Zhou et al., 2023) combines Tree of Thought with Reflexion and Monte Carlo Tree Search (MCTS). It's the most sophisticated planning framework.

Key ideas:
- **Expansion**: Generate candidate next actions (like ToT)
- **Simulation**: Run each candidate to a terminal state
- **Evaluation**: Score the terminal states
- **Backpropagation**: Update parent node scores
- **Backtracking**: Revisit better paths when current path fails

LATS achieves state-of-the-art performance on programming tasks (HumanEval) and web navigation.

```python
# Simplified LATS concept
class LATSNode:
    def __init__(self, state: dict, parent=None):
        self.state = state      # current messages/context
        self.parent = parent    # parent node
        self.children = []      # child nodes
        self.visits = 0         # how many times explored
        self.value = 0.0        # cumulative score
        self.reflection = ""    # verbal feedback from failure
    
    def ucb_score(self, exploration_weight: float = 1.4) -> float:
        """Upper Confidence Bound — balances exploitation vs exploration."""
        if self.visits == 0:
            return float("inf")
        import math
        return (self.value / self.visits) + exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
```

---

## 6. When to Use Each Planning Strategy

| Strategy | Best For | Overhead | Implementation |
|----------|---------|---------|---------------|
| **ReAct** | Short tasks, clear next steps | Minimal | Already built |
| **Plan-Execute** | Multi-step tasks, parallel subtasks | Medium | This week |
| **Tree of Thought** | Open-ended problems, design decisions | High | Async required |
| **Reflexion** | Tasks requiring iteration, research | Medium | 2-3x cost |
| **LATS** | Complex coding, long-horizon tasks | Very High | Advanced |

**Rule**: Start with ReAct. Add planning only when you measure agent failures on complex tasks.

---

## Tools & Libraries Used This Week — Deep Dive

### Plan-Execute vs. ReAct — The Fundamental Trade-off

The core question is: **should the agent decide what to do next at each step, or plan everything upfront?**

**ReAct (Reactive)**: Decision made step-by-step. After each observation, the model decides the next action.
- ✅ Adapts to new information mid-task
- ✅ Works for short tasks
- ❌ Can't see ahead — makes suboptimal sequences
- ❌ Fails on tasks requiring coordination across many steps

**Plan-Execute**: Plan all steps upfront, execute them. Replan only if something fails.
- ✅ More coherent multi-step workflows
- ✅ Can execute independent steps in parallel
- ✅ Better for tasks requiring foresight
- ❌ Plan can be wrong → wastes tokens before realizing
- ❌ More expensive (planning LLM call + all execution calls)

**Mental model**: ReAct is improvisation. Plan-Execute is jazz (structured improvisation within a plan). Tree of Thought is algorithmic exploration. Reflexion is iteration with learning.

---

### Tree of Thought — When Exploration Matters

**The insight**: For complex problems, the "obviously right" first approach often isn't the best. Humans explore multiple approaches mentally before committing. ToT does this explicitly.

**Where ToT consistently outperforms ReAct**:
1. **Creative problem solving**: Multiple valid approaches exist, need the most creative
2. **Mathematical proofs**: Many proof paths, some dead ends early
3. **Design decisions**: Architecture choices with long-term consequences
4. **Game playing**: Chess, puzzles where lookahead is critical

**Where ToT is overkill**:
1. Clear factual queries (just answer it)
2. Tasks with one obvious approach
3. When cost/latency matters more than quality

```python
# The beam search variant of ToT — more efficient
async def beam_search_tot(
    problem: str,
    beam_width: int = 3,    # keep top-N thoughts at each level
    depth: int = 4,          # search N levels deep
) -> str:
    """
    Beam search variant of Tree of Thought.
    More efficient than full tree (explores beam_width^depth thoughts).
    """
    # Initialize beams with initial thoughts
    beams = []
    for _ in range(beam_width):
        thought = await generate_thought(problem, context="")
        score = await evaluate_thought(problem, thought)
        beams.append((score, thought, [thought]))  # (score, current, history)
    
    for level in range(depth - 1):
        candidates = []
        for score, current_thought, history in beams:
            # Expand each beam
            context = "\n→ ".join(history)
            for _ in range(beam_width):
                new_thought = await generate_thought(problem, context=context)
                new_score = await evaluate_thought(problem, new_thought)
                candidates.append((new_score, new_thought, history + [new_thought]))
        
        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
        print(f"Level {level+1}: Best score = {beams[0][0]:.2f}")
    
    # Return the best final thought
    best_score, best_thought, best_history = beams[0]
    context = "\n→ ".join(best_history)
    
    return await generate_final_answer(problem, context)
```

---

### Reflexion — The Science of Learning from Failure

**The paper**: Reflexion (Shinn et al., 2023) showed that verbal self-reflection — storing natural language feedback from failures — dramatically improves agent performance on coding and decision-making tasks.

**Key finding**: 3 Reflexion iterations on HumanEval (code generation benchmark) achieved **91% pass@1** vs **67%** for standard ReAct. This is a 24-point improvement from just letting the agent reflect on its mistakes.

**Why verbal reflection works better than just retrying**:
- Simple retry: same strategy, different random sample → same failure mode
- With reflection: agent explicitly identifies what went wrong → different strategy

```python
# The reflection generation prompt is crucial
REFLECTION_PROMPT = """You attempted this task:
{task}

Your attempt:
{attempt}

Evaluator feedback: {feedback}

Write a SHORT, SPECIFIC reflection on:
1. What specifically went wrong (not "it failed" but exactly what and why)
2. What you will do DIFFERENTLY in the next attempt (concrete strategy change)
3. Any constraints or edge cases you missed

DO NOT just say "try harder" or "be more careful" — be specific about strategy changes."""

# Example good reflection:
# "I searched for 'Python release date' which returned general Python info.
# The question is specifically about Python 3.12. Next time I'll search
# 'Python 3.12 release date official' and also check python.org directly."

# Example bad reflection (not useful):
# "I need to search more carefully and provide accurate information."
```

---

### LATS — The Research Frontier

LATS (Language Agent Tree Search) combines:
- **MCTS** (Monte Carlo Tree Search) for exploring action space
- **LLM for action generation** (what to do next)
- **LLM for state evaluation** (how good is the current state)
- **Reflexion for learning from failures** (verbal backpropagation)

**MCTS key formula** — Upper Confidence Bound (UCB):
$$\text{UCB}(s) = \frac{V(s)}{N(s)} + c\sqrt{\frac{\ln N(\text{parent})}{N(s)}}$$

Where:
- $V(s)$ = cumulative value of node s
- $N(s)$ = visit count of node s
- $c$ = exploration constant (~1.4)

The first term **exploits** (go to high-value nodes). The second term **explores** (go to less-visited nodes). This balance is what makes MCTS so effective.

**When you'd actually use LATS in production**: Almost never — it's extremely expensive (many LLM calls per action decision) and designed for research benchmarks. Reflexion with 3 iterations gives 80% of the benefit at 10% of the cost.

---

### Planning Strategy Selection Flowchart

```
Task received
│
├─ "Answer a simple question"
│    └─ ReAct (2-4 steps) — cheapest, fastest
│
├─ "Research and write a report" (5-15 steps)
│    └─ Plan-Execute — parallel subtasks, structured output
│
├─ "Find the best solution to [open-ended design problem]"
│    └─ Tree of Thought (depth=3, beam=3) — explore options
│
├─ "Generate [code/analysis] that must be correct"
│    └─ Reflexion (max 3 attempts) — iterate to quality
│
├─ "Solve [difficult programming challenge]"
│    └─ Reflexion or LATS — most accurate
│
└─ "Just answer yes/no"
     └─ Single LLM call — no agent needed
```

---

## Common Pitfalls — Week 9

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Plan too rigid — no replanning | Agent fails when step 3 of 10 fails | Always check step outcome, replan if failure detected |
| ToT evaluator too simple | All thoughts get same score, no pruning | Ask model to score 1-10 with specific criteria |
| Reflexion without evaluator | Reflection generated even on success | Always evaluate first, only reflect on failure |
| ToT beam width too large | Cost explosion (width^depth calls) | Start with beam=2, depth=3 — only scale if needed |
| No "bailout" on Reflexion | Costs 3x even when first try is fine | Check success first — most tasks succeed on attempt 1 |
| JSON parse on plan fails | Agent crashes at planning stage | Always retry JSON parse 3 times with error feedback |
- `ex2_self_reflection.py` — Reflexion loop with 3 attempts and verbal feedback
- `ex3_tree_of_thought.py` — ToT with beam search, depth 3, branches 3
- `ex4_compare_strategies.py` — benchmark ReAct vs Plan-Execute on same task

## Checklist
- [ ] Plan-Execute: planner produces JSON plan with dependencies
- [ ] Executor respects step dependencies before running
- [ ] Reflexion: agent generates and uses reflections across 3 attempts
- [ ] ToT: evaluated and scored thoughts to pick best path
- [ ] Benchmarked: measured which strategy works best on your test problems
