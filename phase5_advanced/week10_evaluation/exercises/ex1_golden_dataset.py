"""
Exercise 1: Build a Golden Evaluation Dataset
Goal: Create a 30-question dataset that tests your agent across multiple categories.

Tasks:
  1. Expand GOLDEN_DATASET to at least 30 cases (10 categories, 3 per category).
  2. Complete generate_adversarial() — ask LLM to generate tricky/edge-case questions.
  3. Complete save_dataset() and load_dataset() — persist to JSON.
  4. Complete compute_baseline() — run each question through a simple agent and record answers.
  5. Run and save the baseline so future runs can compare against it (regression detection).

Categories to cover:
  - math: arithmetic, algebra
  - geography: capitals, countries
  - science: physics, biology, chemistry
  - tech: programming, computer science
  - history: dates, events, people
  - language: grammar, definitions
  - logic: deduction, riddles
  - coding: what does this code do
  - current_events: (careful — model may not know)
  - edge_cases: trick questions, ambiguous inputs
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

DATASET_PATH = "golden_dataset.json"


# ── Dataset Schema ─────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: str
    question: str
    expected: str
    category: str
    difficulty: str = "medium"    # "easy" | "medium" | "hard"
    notes: str = ""               # hints for the judge, e.g. "accept any year in 1939-1945"


# ── Seed Dataset ───────────────────────────────────────────────────────────────
# TODO: Expand to 30+ cases. Add 3+ cases per category below.

GOLDEN_DATASET: list[EvalCase] = [
    # Math
    EvalCase("m001", "What is 17 × 23?", "391", "math", "easy"),
    EvalCase("m002", "What is the square root of 169?", "13", "math", "easy"),
    EvalCase("m003", "If a train travels 60 mph for 2.5 hours, how far does it go?", "150 miles", "math", "medium"),

    # Geography
    EvalCase("g001", "What is the capital of Japan?", "Tokyo", "geography", "easy"),
    EvalCase("g002", "Which is the largest country by area?", "Russia", "geography", "easy"),
    EvalCase("g003", "What river runs through Egypt?", "Nile", "geography", "easy"),

    # Science
    EvalCase("s001", "What is the chemical formula for water?", "H2O", "science", "easy"),
    EvalCase("s002", "What is the speed of light in m/s?", "299792458", "science", "medium",
             notes="Accept ~3×10^8 or 299,792,458"),
    EvalCase("s003", "What force keeps planets in orbit?", "gravity", "science", "easy"),

    # Tech
    EvalCase("t001", "What does CPU stand for?", "Central Processing Unit", "tech", "easy"),
    EvalCase("t002", "What is the time complexity of binary search?", "O(log n)", "tech", "medium"),
    EvalCase("t003", "What programming language is Django written in?", "Python", "tech", "easy"),

    # History
    EvalCase("h001", "In what year did World War 2 end?", "1945", "history", "easy"),
    EvalCase("h002", "Who was the first person to walk on the Moon?", "Neil Armstrong", "history", "easy"),
    EvalCase("h003", "What year did the Berlin Wall fall?", "1989", "history", "medium"),

    # Logic
    EvalCase("l001", "If all A are B, and all B are C, are all A also C?", "yes", "logic", "easy"),
    EvalCase("l002", "I have 3 apples. I give 2 away and get 5 back. How many do I have?", "6", "logic", "easy"),
    EvalCase("l003", "A rooster lays an egg on a roof. Which way does it roll?", "roosters don't lay eggs", "logic", "hard",
             notes="Trick question"),

    # TODO: Add more cases for: language, coding, edge_cases categories
]


# ── Dataset Operations ─────────────────────────────────────────────────────────

def save_dataset(cases: list[EvalCase], path: str = DATASET_PATH):
    """Save dataset to JSON file."""
    # TODO: write [asdict(c) for c in cases] to path as pretty JSON
    raise NotImplementedError


def load_dataset(path: str = DATASET_PATH) -> list[EvalCase]:
    """Load dataset from JSON file."""
    # TODO: read JSON, reconstruct EvalCase objects
    raise NotImplementedError


# ── Adversarial Generator ──────────────────────────────────────────────────────

ADVERSARIAL_SYSTEM = """Generate {n} tricky evaluation questions that test an AI assistant's
honesty and reasoning. Focus on:
- Questions with counterintuitive answers
- Questions that require careful reading
- Questions where the obvious answer is wrong

Return ONLY a JSON array:
[{"question": "...", "expected": "...", "category": "edge_cases", "difficulty": "hard", "notes": "..."}, ...]"""


def generate_adversarial(n: int = 5) -> list[EvalCase]:
    """Use LLM to generate adversarial test cases."""
    # TODO: call chat() with ADVERSARIAL_SYSTEM.format(n=n)
    # TODO: parse JSON array from response
    # TODO: convert to EvalCase objects with auto-generated IDs: f"adv{i:03d}"
    raise NotImplementedError


# ── Baseline Computation ───────────────────────────────────────────────────────

def agent_under_test(question: str) -> str:
    """Simple agent — no tools, just direct LLM answer."""
    response = chat(
        [{"role": "user", "content": question}],
        system="Answer concisely and directly. If unsure, say so.",
        max_tokens=128,
    )
    return get_text(response)


def compute_baseline(cases: list[EvalCase]) -> dict:
    """Run all cases through agent, record answers. Return results dict."""
    results = {"answers": {}, "categories": {}}
    print(f"\nComputing baseline on {len(cases)} cases...")

    for case in cases:
        print(f"  [{case.id}] {case.question[:50]}...", end=" ", flush=True)
        answer = agent_under_test(case.question)
        results["answers"][case.id] = {
            "question": case.question,
            "expected": case.expected,
            "actual": answer,
            "category": case.category,
        }
        print(f"→ {answer[:40]}")

    # Count by category
    for case_id, data in results["answers"].items():
        cat = data["category"]
        results["categories"].setdefault(cat, {"total": 0, "ids": []})
        results["categories"][cat]["total"] += 1
        results["categories"][cat]["ids"].append(case_id)

    return results


if __name__ == "__main__":
    # Build and save the dataset
    dataset = GOLDEN_DATASET.copy()

    # Generate some adversarial cases
    print("Generating adversarial cases...")
    adversarial = generate_adversarial(n=5)
    dataset.extend(adversarial)

    print(f"\nDataset: {len(dataset)} cases across {len(set(c.category for c in dataset))} categories")
    save_dataset(dataset)
    print(f"Saved to {DATASET_PATH}")

    # Compute baseline
    baseline = compute_baseline(dataset)
    baseline_path = DATASET_PATH.replace(".json", "_baseline.json")
    Path(baseline_path).write_text(json.dumps(baseline, indent=2))
    print(f"\nBaseline saved to {baseline_path}")
    print(f"Total questions: {len(dataset)}")
    for cat, info in baseline["categories"].items():
        print(f"  {cat}: {info['total']} questions")
