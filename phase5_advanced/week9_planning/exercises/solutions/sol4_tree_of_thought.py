"""
SOLUTION — Exercise 4: Tree of Thought — Explore Multiple Reasoning Branches
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()


@dataclass
class Thought:
    content: str
    score: float = 0.0
    depth: int = 0
    children: list["Thought"] = field(default_factory=list)

    def short(self) -> str:
        return self.content[:80] + "..." if len(self.content) > 80 else self.content


GENERATOR_SYSTEM = """You are a creative problem solver. Generate {n} distinct approaches or partial answers
to the question. Each approach should explore a DIFFERENT angle or strategy.
Return ONLY a JSON array of strings (one string per approach). No markdown."""

EVALUATOR_SYSTEM = """You are a quality evaluator. Score this approach/answer for the given question.
Return ONLY JSON: {"score": <1-10>, "reason": "<15 words max>"}
Criteria: relevance, depth, correctness, clarity."""

EXPANDER_SYSTEM = """You are an expert writer. Take this initial approach and develop it into a
complete, high-quality answer. Be specific, concrete, and thorough. Max 300 words."""


def generate_thoughts(question: str, context: str = "", n: int = 3) -> list[str]:
    user_msg = f"Question: {question}"
    if context:
        user_msg += f"\n\nBuild on this approach: {context}"
    response = chat(
        [{"role": "user", "content": user_msg}],
        system=GENERATOR_SYSTEM.format(n=n),
        max_tokens=400,
    )
    raw = get_text(response)
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        thoughts = json.loads(raw[start:end])
        return [t for t in thoughts if isinstance(t, str)]
    except Exception:
        return [raw]


def evaluate_thought(question: str, thought: str) -> tuple[float, str]:
    prompt = f"Question: {question}\n\nApproach/Answer:\n{thought}"
    response = chat(
        [{"role": "user", "content": prompt}],
        system=EVALUATOR_SYSTEM,
        max_tokens=100,
    )
    raw = get_text(response)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return float(data["score"]), data.get("reason", "")
    except Exception:
        return 5.0, "could not parse"


def expand_thought(question: str, thought: str) -> str:
    prompt = f"Question: {question}\n\nInitial approach: {thought}\n\nDevelop this into a full answer."
    response = chat(
        [{"role": "user", "content": prompt}],
        system=EXPANDER_SYSTEM,
        max_tokens=400,
    )
    return get_text(response)


def tree_of_thought(question: str, branching: int = 3, depth: int = 2,
                    keep_top: int = 2) -> tuple[str, list[Thought]]:
    print(f"\nQuestion: {question}")
    print(f"ToT config: branching={branching}, depth={depth}, keep_top={keep_top}\n")

    all_thoughts: list[Thought] = []

    print(f"Level 1: generating {branching} candidate approaches...")
    initial = generate_thoughts(question, n=branching)
    thoughts = [Thought(content=t, depth=1) for t in initial]

    for i, t in enumerate(thoughts):
        t.score, reason = evaluate_thought(question, t.content)
        all_thoughts.append(t)
        print(f"  T{i+1} (score {t.score:.0f}): {t.short()} — {reason}")

    current_level = sorted(thoughts, key=lambda t: t.score, reverse=True)[:keep_top]

    for d in range(2, depth + 1):
        print(f"\nLevel {d}: expanding {len(current_level)} best thought(s)...")
        next_level: list[Thought] = []

        for parent in current_level:
            expanded = expand_thought(question, parent.content)
            child = Thought(content=expanded, depth=d)
            child.score, reason = evaluate_thought(question, expanded)
            parent.children.append(child)
            all_thoughts.append(child)
            print(f"  Expanded (score {child.score:.0f}): {child.short()}")
            next_level.append(child)

        current_level = sorted(next_level, key=lambda t: t.score, reverse=True)[:keep_top]

    best = max(all_thoughts, key=lambda t: t.score)
    print(f"\n✅ Best answer (score {best.score:.0f}/10):")
    print(best.content)

    return best.content, all_thoughts


def compare_approaches(question: str):
    print("="*60)
    print("SINGLE-SHOT:")
    response = chat([{"role": "user", "content": question}], max_tokens=300)
    single_answer = get_text(response)
    single_score, _ = evaluate_thought(question, single_answer)
    print(f"Score: {single_score}/10")
    print(single_answer[:200])

    print("\n" + "="*60)
    print("TREE OF THOUGHT:")
    tot_answer, _ = tree_of_thought(question, branching=3, depth=2, keep_top=1)
    tot_score, _ = evaluate_thought(question, tot_answer)

    print(f"\n{'='*60}")
    print(f"Single-shot score: {single_score}/10")
    print(f"ToT score:         {tot_score}/10")
    improvement = tot_score - single_score
    print(f"Improvement: {'+' if improvement >= 0 else ''}{improvement:.1f} points")


TEST_QUESTION = "What are three non-obvious strategies a startup should use to retain its first 100 customers?"

if __name__ == "__main__":
    compare_approaches(TEST_QUESTION)
