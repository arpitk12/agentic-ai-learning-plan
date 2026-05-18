"""
SOLUTION — Exercise 2: Self-Reflection / Critic Loop
"""
import json
from pydantic import BaseModel, ValidationError
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()


class CriticScore(BaseModel):
    score: int
    strengths: list[str]
    weaknesses: list[str]
    improvement_prompt: str


GENERATOR_SYSTEM = "You are an expert writer and analyst. Produce high-quality, detailed, accurate responses."

CRITIC_SYSTEM = """You are a strict quality critic. Evaluate the given response and
return ONLY valid JSON (no markdown):
{
  "score": <1-10>,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvement_prompt": "Specific instruction to improve the response"
}
Scoring: 1-3 = major issues, 4-6 = acceptable, 7-8 = good, 9-10 = excellent."""


def generate(query: str, feedback: str | None = None) -> str:
    user_content = query
    if feedback:
        user_content += f"\n\n[IMPROVEMENT GUIDANCE]: {feedback}"
    r = client.messages.create(
        model="claude-opus-4-5", max_tokens=1024,
        system=GENERATOR_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return r.content[0].text


def critique(query: str, response: str) -> CriticScore:
    prompt = f"Original request: {query}\n\nResponse to evaluate:\n{response}"
    r = client.messages.create(
        model="claude-opus-4-5", max_tokens=512,
        system=CRITIC_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text
    start = raw.find("{")
    end = raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    return CriticScore(**data)


def self_reflect(query: str, threshold: int = 7, max_attempts: int = 3) -> str:
    print(f"\n🎯 Query: {query}")
    feedback = None

    for attempt in range(1, max_attempts + 1):
        print(f"\n✍ Attempt {attempt}/{max_attempts}")
        response = generate(query, feedback)

        score_obj = critique(query, response)
        score = score_obj.score
        print(f"  Score: {score}/10")
        print(f"  Strengths: {score_obj.strengths}")
        print(f"  Weaknesses: {score_obj.weaknesses}")

        if score >= threshold:
            print(f"  ✅ Accepted (score {score} >= threshold {threshold})")
            return response

        feedback = score_obj.improvement_prompt
        print(f"  ↩ Retrying with feedback: {feedback[:80]}")

    print(f"  ⚠ Max attempts reached. Returning best attempt.")
    return response


if __name__ == "__main__":
    result = self_reflect(
        "Explain the concept of attention mechanisms in transformers, including how they work mathematically.",
        threshold=7,
        max_attempts=3,
    )
    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(result)
