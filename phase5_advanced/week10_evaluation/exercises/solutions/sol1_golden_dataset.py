"""
SOLUTION — Exercise 1: Build a Golden Evaluation Dataset
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

DATASET_PATH = "golden_dataset.json"


@dataclass
class EvalCase:
    id: str
    question: str
    expected: str
    category: str
    difficulty: str = "medium"
    notes: str = ""


GOLDEN_DATASET: list[EvalCase] = [
    # Math (5 cases)
    EvalCase("m001", "What is 17 × 23?", "391", "math", "easy"),
    EvalCase("m002", "What is the square root of 169?", "13", "math", "easy"),
    EvalCase("m003", "If a train travels 60 mph for 2.5 hours, how far does it go?", "150 miles", "math", "medium"),
    EvalCase("m004", "What is 15% of 240?", "36", "math", "easy"),
    EvalCase("m005", "What is 2 to the power of 10?", "1024", "math", "easy"),

    # Geography (4 cases)
    EvalCase("g001", "What is the capital of Japan?", "Tokyo", "geography", "easy"),
    EvalCase("g002", "Which is the largest country by area?", "Russia", "geography", "easy"),
    EvalCase("g003", "What river runs through Egypt?", "Nile", "geography", "easy"),
    EvalCase("g004", "What is the capital of Australia?", "Canberra", "geography", "medium",
             notes="Common wrong answer is Sydney"),

    # Science (4 cases)
    EvalCase("s001", "What is the chemical formula for water?", "H2O", "science", "easy"),
    EvalCase("s002", "What is the speed of light in m/s?", "299792458", "science", "medium",
             notes="Accept ~3×10^8 or 299,792,458"),
    EvalCase("s003", "What force keeps planets in orbit?", "gravity", "science", "easy"),
    EvalCase("s004", "What is the atomic number of carbon?", "6", "science", "medium"),

    # Tech (4 cases)
    EvalCase("t001", "What does CPU stand for?", "Central Processing Unit", "tech", "easy"),
    EvalCase("t002", "What is the time complexity of binary search?", "O(log n)", "tech", "medium"),
    EvalCase("t003", "What programming language is Django written in?", "Python", "tech", "easy"),
    EvalCase("t004", "What does HTTP stand for?", "HyperText Transfer Protocol", "tech", "easy"),

    # History (3 cases)
    EvalCase("h001", "In what year did World War 2 end?", "1945", "history", "easy"),
    EvalCase("h002", "Who was the first person to walk on the Moon?", "Neil Armstrong", "history", "easy"),
    EvalCase("h003", "What year did the Berlin Wall fall?", "1989", "history", "medium"),

    # Logic (4 cases)
    EvalCase("l001", "If all A are B, and all B are C, are all A also C?", "yes", "logic", "easy"),
    EvalCase("l002", "I have 3 apples. I give 2 away and get 5 back. How many do I have?", "6", "logic", "easy"),
    EvalCase("l003", "A rooster lays an egg on a roof. Which way does it roll?",
             "roosters don't lay eggs", "logic", "hard", notes="Trick question"),
    EvalCase("l004", "What is always in front of you but cannot be seen?", "the future", "logic", "medium"),

    # Language (3 cases)
    EvalCase("la001", "What is a synonym for 'happy'?", "joyful", "language", "easy",
             notes="Accept: glad, cheerful, elated, content"),
    EvalCase("la002", "What part of speech is the word 'quickly'?", "adverb", "language", "easy"),
    EvalCase("la003", "What is the plural of 'analysis'?", "analyses", "language", "medium"),

    # Coding (3 cases)
    EvalCase("c001", "What does `len([1, 2, 3])` return in Python?", "3", "coding", "easy"),
    EvalCase("c002", "What is the output of `print(type(42))`?", "<class 'int'>", "coding", "easy"),
    EvalCase("c003", "What does `list(range(3))` return?", "[0, 1, 2]", "coding", "easy"),

    # Edge cases (3 cases)
    EvalCase("e001", "How many months have 28 days?", "all of them", "edge_cases", "hard",
             notes="Trick: all months have at least 28 days"),
    EvalCase("e002", "If you have a 10-foot rope and cut it in half, how many pieces do you have?",
             "2", "edge_cases", "easy"),
    EvalCase("e003", "Is it possible for a man to marry his widow's sister in most countries?",
             "no, he would be dead", "edge_cases", "hard", notes="Trick question"),
]


def save_dataset(cases: list[EvalCase], path: str = DATASET_PATH):
    data = [asdict(c) for c in cases]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(cases)} cases to {path}")


def load_dataset(path: str = DATASET_PATH) -> list[EvalCase]:
    with open(path) as f:
        data = json.load(f)
    return [EvalCase(**d) for d in data]


ADVERSARIAL_SYSTEM = """Generate {n} tricky evaluation questions that test an AI assistant's
honesty and reasoning. Focus on counterintuitive answers or careful reading required.
Return ONLY a JSON array:
[{{"question": "...", "expected": "...", "category": "edge_cases", "difficulty": "hard", "notes": "..."}}]"""


def generate_adversarial(n: int = 5) -> list[EvalCase]:
    response = chat(
        [{"role": "user", "content": f"Generate {n} tricky evaluation questions."}],
        system=ADVERSARIAL_SYSTEM.format(n=n),
        max_tokens=600,
    )
    raw = get_text(response)
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        items = json.loads(raw[start:end])
        return [
            EvalCase(
                id=f"adv{i:03d}",
                question=item["question"],
                expected=item["expected"],
                category=item.get("category", "edge_cases"),
                difficulty=item.get("difficulty", "hard"),
                notes=item.get("notes", ""),
            )
            for i, item in enumerate(items, 1)
        ]
    except Exception as e:
        print(f"[parse error] {e}")
        return []


JUDGE_SYSTEM = """You are an evaluator. Does the answer correctly address the question?
The expected answer is provided for reference. Be lenient about phrasing if the meaning is correct.
Return ONLY JSON: {"correct": true/false, "score": <0-1>, "note": "<brief reason>"}"""


def judge_answer(question: str, expected: str, answer: str, notes: str = "") -> tuple[bool, float, str]:
    prompt = f"Question: {question}\nExpected: {expected}\nActual answer: {answer}"
    if notes:
        prompt += f"\nNote: {notes}"
    response = chat(
        [{"role": "user", "content": prompt}],
        system=JUDGE_SYSTEM,
        max_tokens=100,
    )
    raw = get_text(response)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return bool(data["correct"]), float(data.get("score", 1.0 if data["correct"] else 0.0)), data.get("note", "")
    except Exception:
        return False, 0.0, "parse error"


def compute_baseline(cases: list[EvalCase]) -> dict:
    """Run each question through a basic agent and record pass/fail."""
    results = []
    passed = 0

    print(f"Running baseline on {len(cases)} cases...")
    for case in cases:
        response = chat(
            [{"role": "user", "content": case.question}],
            system="Answer concisely and directly. Give the answer only, no explanation.",
            max_tokens=100,
        )
        answer = get_text(response).strip()
        correct, score, note = judge_answer(case.question, case.expected, answer, case.notes)

        if correct:
            passed += 1
        status = "✅" if correct else "❌"
        print(f"  {status} [{case.id}] {case.question[:50]}...")
        print(f"     Expected: {case.expected[:60]} | Got: {answer[:60]}")
        if note:
            print(f"     Note: {note}")

        results.append({
            "id": case.id,
            "category": case.category,
            "difficulty": case.difficulty,
            "question": case.question,
            "expected": case.expected,
            "actual": answer,
            "correct": correct,
            "score": score,
        })

    accuracy = passed / len(cases) if cases else 0
    print(f"\n{'='*50}")
    print(f"Baseline accuracy: {passed}/{len(cases)} = {accuracy:.1%}")

    # Category breakdown
    from collections import defaultdict
    cat_stats: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat_stats[r["category"]]["total"] += 1
        if r["correct"]:
            cat_stats[r["category"]]["correct"] += 1

    print("\nBy category:")
    for cat, stats in sorted(cat_stats.items()):
        pct = stats["correct"] / stats["total"]
        print(f"  {cat:<15} {stats['correct']}/{stats['total']} = {pct:.0%}")

    return {"accuracy": accuracy, "passed": passed, "total": len(cases), "results": results}


if __name__ == "__main__":
    # Save the dataset
    save_dataset(GOLDEN_DATASET)
    print(f"Dataset: {len(GOLDEN_DATASET)} cases across {len({c.category for c in GOLDEN_DATASET})} categories\n")

    # Generate adversarial cases
    print("Generating adversarial cases...")
    adversarial = generate_adversarial(n=3)
    if adversarial:
        all_cases = GOLDEN_DATASET + adversarial
        save_dataset(all_cases, "golden_dataset_with_adversarial.json")
        print(f"Extended dataset: {len(all_cases)} cases\n")

    # Compute baseline
    print("Computing baseline (first 10 cases only for speed)...")
    compute_baseline(GOLDEN_DATASET[:10])
