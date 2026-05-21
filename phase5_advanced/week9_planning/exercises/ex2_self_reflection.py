"""
Exercise 2: Self-Reflection / Critic Loop
Goal: Agent generates output, critic scores it, regenerate if below threshold.

pip install anthropic python-dotenv pydantic
"""
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from pydantic import BaseModel
from llm import chat, get_text, MODEL


class CriticScore(BaseModel):
    score: int           # 1-10
    strengths: list[str]
    weaknesses: list[str]
    improvement_prompt: str  # instruction for next attempt


GENERATOR_SYSTEM = """You are an expert writer and analyst.
Produce high-quality, detailed, accurate responses."""

CRITIC_SYSTEM = """You are a strict quality critic. Evaluate the given response and 
return ONLY valid JSON (no markdown):
{
  "score": <1-10>,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvement_prompt": "Specific instruction to improve the response"
}

Scoring guide:
1-3: Major errors or missing key content
4-6: Acceptable but needs improvement  
7-8: Good quality, minor improvements possible
9-10: Excellent, publish-ready"""


def generate(query: str, feedback: str | None = None) -> str:
    """Generate a response. If feedback is provided, use it as improvement guidance."""
    raise NotImplementedError


def critique(query: str, response: str) -> CriticScore:
    """Evaluate the response against the original query and return a CriticScore."""
    raise NotImplementedError


def self_reflecting_agent(
    query: str,
    threshold: int = 7,
    max_attempts: int = 3
) -> tuple[str, list[dict]]:
    """
    Run generator → critic → regenerate loop until score >= threshold.
    Returns (final_answer, attempt_log).
    """
    attempt_log = []

    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_attempts} ===")

        # TODO: Generate a response, using feedback from the previous attempt if available
        # TODO: Critique the response
        # TODO: Log the attempt (score, response snippet, feedback)
        # TODO: Return early if score >= threshold
        # TODO: Otherwise carry the improvement prompt into the next iteration

        feedback = None  # replace this

    print(f"\nMax attempts reached. Returning best response.")
    return "Not implemented", attempt_log


if __name__ == "__main__":
    query = "Explain the CAP theorem and its practical implications for distributed systems."
    answer, log = self_reflecting_agent(query, threshold=8, max_attempts=3)

    print(f"\n{'='*50}")
    print(f"Final Answer (after {len(log)} attempts):")
    print(answer)

    print(f"\nAttempt scores: {[a['score'] for a in log]}")
