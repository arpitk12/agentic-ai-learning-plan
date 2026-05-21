"""
SOLUTION — Exercise 1: Orchestrator + Specialist Agents
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()


def _call(system: str, user: str, max_tokens: int = 1024) -> str:
    r = chat([{"role": "user", "content": user}], system=system, max_tokens=max_tokens)
    return get_text(r)


def planner_agent(task: str) -> list[str]:
    system = (
        "You are a planning expert. Break the given task into 3-5 concrete, "
        "actionable steps. Return a JSON array of strings only. No markdown."
    )
    raw = _call(system, f"Task: {task}")
    # Find JSON array in response
    start = raw.find("[")
    end = raw.rfind("]") + 1
    return json.loads(raw[start:end])


def executor_agent(step: str, context: str) -> str:
    system = (
        "You are an expert writer and researcher. Execute the given task step "
        "thoroughly and professionally. Use the provided context for continuity."
    )
    return _call(system, f"Context so far:\n{context}\n\nExecute this step: {step}", max_tokens=2048)


def critic_agent(content: str, step: str) -> tuple[int, str]:
    system = (
        "You are a quality critic. Review the given content against the step it was meant to fulfill. "
        "Return JSON: {\"score\": <int 1-10>, \"feedback\": \"<str>\"}. No markdown."
    )
    raw = _call(system, f"Step: {step}\n\nContent:\n{content}")
    start = raw.find("{")
    end = raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    return int(data["score"]), data["feedback"]


def run_orchestrator(task: str) -> str:
    print(f"\n🎯 Task: {task}")
    steps = planner_agent(task)
    print(f"📋 Plan ({len(steps)} steps):")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    results = []
    context = f"Task: {task}\n\n"

    for i, step in enumerate(steps):
        print(f"\n⚙ Executing step {i+1}: {step}")
        output = ""
        for attempt in range(3):
            output = executor_agent(step, context)
            score, feedback = critic_agent(output, step)
            print(f"  Score: {score}/10 | {feedback[:70]}")
            if score >= 7:
                break
            context += f"\nCritic feedback: {feedback}\n"
            print(f"  ↩ Retrying (attempt {attempt+2}/3)...")
        results.append(output)
        context += f"\nStep {i+1} result:\n{output}\n"

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    result = run_orchestrator("Research and write a short blog post about the future of quantum computing")
    print("\n" + "=" * 60 + "\nFINAL OUTPUT:\n" + "=" * 60)
    print(result)
