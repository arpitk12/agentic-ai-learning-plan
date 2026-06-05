"""
SOLUTION — Exercise 3: Debate Pattern — Two Agents Argue, Third Votes
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()


class JudgeVerdict(BaseModel):
    winner: str
    for_score: int
    against_score: int
    reasoning: str
    final_answer: str


FOR_SYSTEM = """You are a skilled debate champion arguing FOR the given position.
Make your best case with 3 strong arguments. Be persuasive, use evidence and logic.
Keep your argument under 200 words."""

AGAINST_SYSTEM = """You are a skilled debate champion arguing AGAINST the given position.
Make your best case with 3 strong arguments. Be persuasive, use evidence and logic.
Keep your argument under 200 words."""

JUDGE_SYSTEM = """You are an impartial debate judge. Evaluate both sides on logic, evidence, and persuasiveness.
Return ONLY valid JSON (no markdown):
{
  "winner": "for" | "against" | "tie",
  "for_score": <1-10>,
  "against_score": <1-10>,
  "reasoning": "2-3 sentences explaining your decision",
  "final_answer": "Your own synthesized conclusion on the topic"
}"""


def agent_argue(topic: str, side: str) -> str:
    system = FOR_SYSTEM if side == "for" else AGAINST_SYSTEM
    response = chat(
        [{"role": "user", "content": f"Debate topic: {topic}"}],
        system=system,
        max_tokens=300,
    )
    return get_text(response)


def judge_debate(topic: str, for_argument: str, against_argument: str) -> JudgeVerdict:
    prompt = (
        f"Topic: {topic}\n\n"
        f"FOR argument:\n{for_argument}\n\n"
        f"AGAINST argument:\n{against_argument}"
    )
    response = chat(
        [{"role": "user", "content": prompt}],
        system=JUDGE_SYSTEM,
        max_tokens=400,
    )
    raw = get_text(response)
    start = raw.find("{")
    end = raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    return JudgeVerdict(**data)


def run_debate(topic: str) -> JudgeVerdict:
    print(f"\n{'='*60}")
    print(f"DEBATE: {topic}")
    print("="*60)

    print("\n🟢 FOR:")
    for_arg = agent_argue(topic, "for")
    print(for_arg)

    print("\n🔴 AGAINST:")
    against_arg = agent_argue(topic, "against")
    print(against_arg)

    print("\n⚖ JUDGE deliberating...")
    verdict = judge_debate(topic, for_arg, against_arg)

    print(f"\n{'─'*60}")
    print(f"VERDICT: {verdict.winner.upper()} wins")
    print(f"  FOR score:     {verdict.for_score}/10")
    print(f"  AGAINST score: {verdict.against_score}/10")
    print(f"  Reasoning: {verdict.reasoning}")
    print(f"\n💡 Judge's conclusion:\n{verdict.final_answer}")
    return verdict


DEBATE_TOPICS = [
    "LLM agents will replace traditional software engineers within 5 years",
    "AI systems should be required to identify themselves as AI in all conversations",
    "Open-source AI models are safer than closed-source models",
]

if __name__ == "__main__":
    import sys as _sys
    topic = " ".join(_sys.argv[1:]) if len(_sys.argv) > 1 else DEBATE_TOPICS[0]
    verdict = run_debate(topic)
    print(f"\nFinal winner: {verdict.winner}")
