"""
Exercise 1: Plan-and-Execute Agent
Goal: Separate planning from execution — generate a full plan first, then execute each step.

Contrast with ReAct (which interleaves reasoning and action):
  Plan-and-Execute: Plan all steps → Execute step 1 → Execute step 2 → ... → Synthesize
  ReAct:            Think → Act → Observe → Think → Act → ...

Tasks:
  1. Complete planner() — given a task, return a list of concrete steps (JSON array).
  2. Complete executor() — given one step + context so far, return the result.
  3. Complete synthesizer() — given task + all step results, return a final answer.
  4. Complete run_plan_execute() — wire planner → executor loop → synthesizer.
  5. (Bonus) Add a re-planner: if a step fails, ask the LLM to revise the remaining plan.

Expected output:
  Task: Write a 3-point summary of quantum computing
  PLAN (4 steps):
    1. Define quantum computing
    2. Explain qubits and superposition
    3. Describe key applications
    4. Synthesize into 3 points
  EXECUTING step 1/4...
  EXECUTING step 2/4...
  ...
  FINAL ANSWER: 1. Quantum computers use qubits...
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

# ── Agent Prompts ──────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a task planner. Break the given task into 3-5 concrete, sequential steps.
Each step should be specific and self-contained.
Return ONLY a JSON array of strings. No markdown, no explanation.
Example: ["Step 1: Define X", "Step 2: Explain Y", "Step 3: Summarize"]"""

EXECUTOR_SYSTEM = """You are a task executor. You will be given:
- A specific step to execute
- Context from previous steps (may be empty)
Execute ONLY the given step. Be specific and thorough. 2-3 paragraphs max."""

SYNTHESIZER_SYSTEM = """You are a synthesis expert. Given the original task and results from
each step, produce a coherent, well-structured final answer. Be concise and direct."""


# ── Agent Functions ────────────────────────────────────────────────────────────

def planner(task: str) -> list[str]:
    """Return a list of steps to complete the task."""
    # TODO: call chat() with PLANNER_SYSTEM
    # TODO: raw = get_text(response)
    # TODO: parse JSON: steps = json.loads(raw[raw.find("["):raw.rfind("]")+1])
    # TODO: return steps
    raise NotImplementedError


def executor(step: str, context: str) -> str:
    """Execute one step given accumulated context from previous steps."""
    user_msg = f"Step to execute: {step}"
    if context:
        user_msg += f"\n\nContext from previous steps:\n{context}"
    # TODO: call chat() with EXECUTOR_SYSTEM and user_msg
    # TODO: return get_text(response)
    raise NotImplementedError


def synthesizer(task: str, step_results: list[tuple[str, str]]) -> str:
    """Synthesize all step results into a final answer."""
    steps_text = "\n\n".join(
        f"Step {i+1}: {step}\nResult: {result}"
        for i, (step, result) in enumerate(step_results)
    )
    user_msg = f"Original task: {task}\n\nStep results:\n{steps_text}"
    # TODO: call chat() with SYNTHESIZER_SYSTEM and user_msg
    # TODO: return get_text(response)
    raise NotImplementedError


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_plan_execute(task: str) -> str:
    print(f"\nTask: {task}")
    print("─" * 50)

    # Phase 1: Plan
    print("PLANNING...")
    steps = planner(task)
    print(f"PLAN ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    # Phase 2: Execute
    print()
    step_results: list[tuple[str, str]] = []
    context = ""

    for i, step in enumerate(steps, 1):
        print(f"EXECUTING step {i}/{len(steps)}: {step[:60]}...")
        result = executor(step, context)
        step_results.append((step, result))
        context += f"\n\nStep {i} ({step}):\n{result}"
        print(f"  → {result[:100]}...")

    # Phase 3: Synthesize
    print("\nSYNTHESIZING...")
    final = synthesizer(task, step_results)

    print(f"\n{'='*50}")
    print("FINAL ANSWER:")
    print(final)
    return final


TASKS = [
    "Explain the three main types of machine learning with a practical example of each",
    "Write a 5-step guide to debugging a slow Python program",
    "Compare microservices vs monolithic architecture: pros, cons, and when to use each",
]

if __name__ == "__main__":
    import sys as _sys
    task = " ".join(_sys.argv[1:]) if len(_sys.argv) > 1 else TASKS[0]
    run_plan_execute(task)
