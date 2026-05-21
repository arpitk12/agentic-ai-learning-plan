"""
Exercise 2: LLM-as-Judge Evaluation Pipeline
Goal: Automatically score agent outputs against a golden dataset.

pip install anthropic python-dotenv pydantic rich
"""
import json
import csv
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from pydantic import BaseModel
from llm import chat, get_text, MODEL


# --- Data Models ---
class EvalCase(BaseModel):
    id: str
    question: str
    expected: str
    category: str = "general"


class EvalResult(BaseModel):
    id: str
    question: str
    expected: str
    actual: str
    score: int        # 1-5
    reasoning: str
    passed: bool      # score >= 3


class EvalReport(BaseModel):
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    by_category: dict[str, dict]
    failed_cases: list[EvalResult]


# --- Golden Dataset (extend this!) ---
GOLDEN_DATASET = [
    EvalCase(id="g001", question="What is 15% of 240?", expected="36", category="math"),
    EvalCase(id="g002", question="What is the capital of Japan?", expected="Tokyo", category="geography"),
    EvalCase(id="g003", question="What does CPU stand for?", expected="Central Processing Unit", category="tech"),
    EvalCase(id="g004", question="Who wrote Hamlet?", expected="William Shakespeare", category="literature"),
    EvalCase(id="g005", question="What year did World War 2 end?", expected="1945", category="history"),
    # TODO: Add 45 more cases across categories
]


# --- Agent Under Test ---
def agent_under_test(question: str) -> str:
    """
    TODO: Replace with your actual agent.
    For now, a simple LLM call.
    """
    response = chat(
        messages=[{"role": "user", "content": question}],
        max_tokens=256,
    )
    return get_text(response)


# --- LLM Judge ---
JUDGE_SYSTEM = """You are an objective evaluator. Compare the actual answer to the expected answer.
Be lenient with phrasing differences — focus on factual correctness and completeness.
Respond ONLY with valid JSON:
{"score": <1-5>, "reasoning": "<one sentence explanation>"}

Scoring:
5 = Correct and complete
4 = Correct with minor omissions
3 = Partially correct
2 = Mostly wrong but shows some understanding
1 = Completely wrong or refused to answer"""


def llm_judge(case: EvalCase, actual: str) -> tuple[int, str]:
    """TODO: Use LLM to score actual vs expected. Return (score, reasoning)."""
    prompt = f"""Question: {case.question}
Expected answer: {case.expected}
Actual answer: {actual}"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        system=JUDGE_SYSTEM,
        max_tokens=256,
    )
    data = json.loads(get_text(response))
    return data["score"], data["reasoning"]


# --- Eval Runner ---
def run_eval(
    dataset: list[EvalCase],
    pass_threshold: int = 3
) -> EvalReport:
    results: list[EvalResult] = []

    for i, case in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] {case.id}: {case.question[:60]}...")

        actual = agent_under_test(case.question)
        score, reasoning = llm_judge(case, actual)
        passed = score >= pass_threshold

        result = EvalResult(
            id=case.id,
            question=case.question,
            expected=case.expected,
            actual=actual,
            score=score,
            reasoning=reasoning,
            passed=passed
        )
        results.append(result)
        print(f"  Score: {score}/5 | {'✓ PASS' if passed else '✗ FAIL'} | {reasoning}")

    # TODO: Aggregate results into EvalReport
    # Group by category, compute pass_rate and avg_score per category
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)

    return EvalReport(
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=round(passed_count / total, 3) if total else 0,
        avg_score=round(sum(r.score for r in results) / total, 2) if total else 0,
        by_category={},  # TODO: fill this in
        failed_cases=[r for r in results if not r.passed]
    )


if __name__ == "__main__":
    print("Running evaluation...\n")
    report = run_eval(GOLDEN_DATASET)

    print(f"\n{'='*50}")
    print(f"EVAL REPORT")
    print(f"{'='*50}")
    print(f"Total:     {report.total}")
    print(f"Passed:    {report.passed}")
    print(f"Failed:    {report.failed}")
    print(f"Pass rate: {report.pass_rate:.1%}")
    print(f"Avg score: {report.avg_score}/5")

    if report.failed_cases:
        print(f"\nFailed cases:")
        for r in report.failed_cases:
            print(f"  [{r.id}] {r.question[:50]}... → Score {r.score}: {r.reasoning}")

    # Save report
    Path("eval_report.json").write_text(
        json.dumps(report.model_dump(), indent=2)
    )
    print("\nSaved to eval_report.json")
