"""
Exercise 1: Orchestrator + Specialist Agents
Goal: Build a 3-agent system: Planner → Executor → Critic.

Scenario: A user asks to "research and write a blog post about quantum computing".
  - Planner: breaks task into 3-5 steps
  - Executor: executes each step (has web_search and draft_section tools)
  - Critic: reviews each section and scores quality 1-10

Tasks:
  1. Implement three agent functions: planner_agent, executor_agent, critic_agent.
  2. Each takes a state dict and returns an updated state dict.
  3. Wire them together: planner → executor (loop per step) → critic → final.
  4. If critic score < 7, send back to executor with feedback.
  5. Print a final report with all sections concatenated.

Note: Use the raw Anthropic SDK (no frameworks) to understand the pattern.
"""
import json
from llm import chat, get_text, get_tool_calls, stop_reason, MODEL


def planner_agent(task: str) -> list[str]:
    """Decompose a task into a list of concrete steps."""
    raise NotImplementedError


def executor_agent(step: str, context: str) -> str:
    """Execute a single step given the context of prior steps."""
    raise NotImplementedError


def critic_agent(content: str, step: str) -> tuple[int, str]:
    """Review content quality for the given step. Return (score 1-10, feedback)."""
    raise NotImplementedError


def run_orchestrator(task: str) -> str:
    """Orchestrate the full Planner → Executor → Critic pipeline."""
    print(f"\n🎯 Task: {task}")

    steps = planner_agent(task)
    print(f"📋 Plan ({len(steps)} steps):")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
    results = []
    context = f"Task: {task}\n\n"

    for i, step in enumerate(steps):
        print(f"\n⚙ Executing step {i+1}: {step}")
        for attempt in range(3):
            output = executor_agent(step, context)
            score, feedback = critic_agent(output, step)
            print(f"  Score: {score}/10 | Feedback: {feedback[:60]}")
            if score >= 7:
                break
            print(f"  Retrying (attempt {attempt+2}/3)...")
            context += f"\nFeedback for retry: {feedback}\n"
        results.append(output)
        context += f"\nStep {i+1} result:\n{output}\n"

    final = "\n\n---\n\n".join(results)
    return final


if __name__ == "__main__":
    result = run_orchestrator("Research and write a short blog post about the future of quantum computing")
    print("\n" + "="*60)
    print("FINAL OUTPUT:")
    print("="*60)
    print(result)
