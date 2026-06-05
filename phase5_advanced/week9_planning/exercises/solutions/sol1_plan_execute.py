"""
SOLUTION — Exercise 1: Plan-and-Execute Agent
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

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


def planner(task: str) -> list[str]:
    response = chat(
        [{"role": "user", "content": f"Task: {task}"}],
        system=PLANNER_SYSTEM,
        max_tokens=512,
    )
    raw = get_text(response)
    start = raw.find("[")
    end = raw.rfind("]") + 1
    steps = json.loads(raw[start:end])
    return [s for s in steps if isinstance(s, str)]


def executor(step: str, context: str) -> str:
    user_msg = f"Step to execute: {step}"
    if context:
        user_msg += f"\n\nContext from previous steps:\n{context}"
    response = chat(
        [{"role": "user", "content": user_msg}],
        system=EXECUTOR_SYSTEM,
        max_tokens=600,
    )
    return get_text(response)


def synthesizer(task: str, step_results: list[tuple[str, str]]) -> str:
    steps_text = "\n\n".join(
        f"Step {i+1}: {step}\nResult: {result}"
        for i, (step, result) in enumerate(step_results)
    )
    user_msg = f"Original task: {task}\n\nStep results:\n{steps_text}"
    response = chat(
        [{"role": "user", "content": user_msg}],
        system=SYNTHESIZER_SYSTEM,
        max_tokens=800,
    )
    return get_text(response)


def run_plan_execute(task: str) -> str:
    print(f"\nTask: {task}")
    print("─" * 50)

    print("PLANNING...")
    steps = planner(task)
    print(f"PLAN ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    print()
    step_results: list[tuple[str, str]] = []
    context = ""

    for i, step in enumerate(steps, 1):
        print(f"EXECUTING step {i}/{len(steps)}: {step[:60]}...")
        result = executor(step, context)
        step_results.append((step, result))
        context += f"\n\nStep {i} ({step}):\n{result}"
        print(f"  → {result[:100]}...")

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
