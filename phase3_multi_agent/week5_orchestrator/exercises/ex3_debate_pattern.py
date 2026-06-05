"""
Exercise 3: Debate Pattern — Two Agents Argue, Third Votes
Goal: Use disagreement between agents to surface better answers.

Pattern:
  1. Agent A argues FOR a position
  2. Agent B argues AGAINST (or takes an opposing view)
  3. Judge agent reads both and picks the stronger argument + gives final verdict

Tasks:
  1. Complete agent_argue() — given a topic and side ("for"/"against"), return argument.
  2. Complete judge_debate() — given topic + both arguments, return a JudgeVerdict.
  3. Complete run_debate() — orchestrate the full flow and print a formatted report.
  4. (Bonus) Add a third round: let each agent respond to the other's argument.

Expected output:
  Topic: "LLM agents should replace traditional software engineers"
  FOR:  Agents can write, test, and deploy code autonomously...
  AGAINST: Agents lack accountability, make subtle errors...
  VERDICT: Against wins (7/10 vs 5/10). Reason: ...
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()


# ── Output Schema ──────────────────────────────────────────────────────────────

class JudgeVerdict(BaseModel):
    winner: str          # "for" | "against" | "tie"
    for_score: int       # 1-10
    against_score: int   # 1-10
    reasoning: str       # 2-3 sentences
    final_answer: str    # The judge's own synthesized conclusion


# ── Agents ─────────────────────────────────────────────────────────────────────

FOR_SYSTEM = """You are a skilled debate champion arguing FOR the given position.
Make your best case with 3 strong arguments. Be persuasive, use evidence and logic.
Keep your argument under 200 words."""

AGAINST_SYSTEM = """You are a skilled debate champion arguing AGAINST the given position.
Make your best case with 3 strong arguments. Be persuasive, use evidence and logic.
Keep your argument under 200 words."""

JUDGE_SYSTEM = """You are an impartial debate judge. You have heard both sides of a debate.
Evaluate each argument on: logic, evidence quality, and persuasiveness.
Return ONLY valid JSON (no markdown):
{
  "winner": "for" | "against" | "tie",
  "for_score": <1-10>,
  "against_score": <1-10>,
  "reasoning": "2-3 sentences explaining your decision",
  "final_answer": "Your own synthesized conclusion on the topic"
}"""


def agent_argue(topic: str, side: str) -> str:
    """Return a debate argument. side = 'for' or 'against'."""
    system = FOR_SYSTEM if side == "for" else AGAINST_SYSTEM
    # TODO: call chat() with system prompt and user message = f"Debate topic: {topic}"
    # TODO: return get_text(response)
    raise NotImplementedError


def judge_debate(topic: str, for_argument: str, against_argument: str) -> JudgeVerdict:
    """Return a structured verdict from the judge agent."""
    prompt = (
        f"Topic: {topic}\n\n"
        f"FOR argument:\n{for_argument}\n\n"
        f"AGAINST argument:\n{against_argument}"
    )
    # TODO: call chat() with JUDGE_SYSTEM
    # TODO: parse JSON from get_text(response)
    # TODO: return JudgeVerdict(**data)
    raise NotImplementedError


def run_debate(topic: str) -> JudgeVerdict:
    """Run the full debate and print a formatted report."""
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
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEBATE_TOPICS[0]
    verdict = run_debate(topic)
    print(f"\nFinal winner: {verdict.winner}")
