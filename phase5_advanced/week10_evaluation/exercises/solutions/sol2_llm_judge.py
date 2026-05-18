"""
SOLUTION — Exercise 2: LLM-as-Judge Evaluation Pipeline
"""
import json
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()


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
    score: int
    reasoning: str
    passed: bool


class EvalReport(BaseModel):
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    by_category: dict
    failed_cases: list[EvalResult]


GOLDEN_DATASET = [
    EvalCase(id="g001", question="What is 15% of 240?", expected="36", category="math"),
    EvalCase(id="g002", question="What is the capital of Japan?", expected="Tokyo", category="geography"),
    EvalCase(id="g003", question="What does CPU stand for?", expected="Central Processing Unit", category="tech"),
    EvalCase(id="g004", question="Who wrote Hamlet?", expected="William Shakespeare", category="literature"),
    EvalCase(id="g005", question="What year did World War 2 end?", expected="1945", category="history"),
    EvalCase(id="g006", question="What is the speed of light in m/s?", expected="299,792,458", category="science"),
    EvalCase(id="g007", question="What programming language is Django written in?", expected="Python", category="tech"),
    EvalCase(id="g008", question="How many bytes in a kilobyte?", expected="1024", category="tech"),
    EvalCase(id="g009", question="What is the largest planet in our solar system?", expected="Jupiter", category="science"),
    EvalCase(id="g010", question="What is the chemical symbol for gold?", expected="Au", category="science"),
]

JUDGE_SYSTEM = """You are an objective evaluator. Compare the actual answer to the expected answer.
Return ONLY JSON: {"score": <1-5>, "reasoning": "<str>"}
Scoring: 5=perfect, 4=correct but verbose/imprecise, 3=partially correct, 2=mostly wrong, 1=wrong/irrelevant"""


def agent_under_test(question: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": question}],
    )
    return r.content[0].text


def llm_judge(case: EvalCase, actual: str) -> EvalResult:
    prompt = (
        f"Question: {case.question}\n"
        f"Expected: {case.expected}\n"
        f"Actual: {actual}"
    )
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=256,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text
    s = raw.find("{"); e = raw.rfind("}") + 1
    data = json.loads(raw[s:e])
    return EvalResult(
        id=case.id,
        question=case.question,
        expected=case.expected,
        actual=actual,
        score=data["score"],
        reasoning=data["reasoning"],
        passed=data["score"] >= 3,
    )


def run_eval(dataset: list[EvalCase]) -> EvalReport:
    results: list[EvalResult] = []
    for case in dataset:
        print(f"  Evaluating {case.id}: {case.question[:50]}...")
        actual = agent_under_test(case.question)
        result = llm_judge(case, actual)
        results.append(result)
        icon = "✅" if result.passed else "❌"
        print(f"    {icon} Score: {result.score}/5 | {result.reasoning[:60]}")

    passed = [r for r in results if r.passed]
    avg = sum(r.score for r in results) / len(results) if results else 0

    by_cat: dict = {}
    for r in results:
        cat = next((c.category for c in dataset if c.id == r.id), "unknown")
        by_cat.setdefault(cat, {"total": 0, "passed": 0})
        by_cat[cat]["total"] += 1
        if r.passed:
            by_cat[cat]["passed"] += 1

    return EvalReport(
        total=len(results),
        passed=len(passed),
        failed=len(results) - len(passed),
        pass_rate=len(passed) / len(results) if results else 0,
        avg_score=round(avg, 2),
        by_category=by_cat,
        failed_cases=[r for r in results if not r.passed],
    )


if __name__ == "__main__":
    print(f"Running evaluation on {len(GOLDEN_DATASET)} cases...\n")
    report = run_eval(GOLDEN_DATASET)
    print(f"\n{'='*50}")
    print(f"EVAL REPORT")
    print(f"{'='*50}")
    print(f"Pass rate:  {report.pass_rate:.0%} ({report.passed}/{report.total})")
    print(f"Avg score:  {report.avg_score}/5")
    print(f"\nBy category:")
    for cat, stats in report.by_category.items():
        pct = stats['passed'] / stats['total']
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({pct:.0%})")
    if report.failed_cases:
        print(f"\nFailed cases:")
        for f in report.failed_cases:
            print(f"  [{f.id}] {f.question[:50]} → {f.reasoning[:60]}")
