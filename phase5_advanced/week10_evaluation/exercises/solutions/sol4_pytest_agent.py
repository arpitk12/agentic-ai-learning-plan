"""
SOLUTION — Exercise 4: pytest Regression Suite for Agents

Run: pytest sol4_pytest_agent.py -v
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import json
import pytest
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

SYSTEM = "You are a helpful AI assistant. Answer questions accurately and concisely."


def agent(question: str) -> str:
    r = chat(
        [{"role": "user", "content": question}],
        system=SYSTEM,
        max_tokens=300,
    )
    return get_text(r)


def exact_match_eval(answer: str, expected_keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in expected_keywords)


def llm_judge_eval(question: str, answer: str, threshold: int = 7) -> tuple[bool, int, str]:
    JUDGE_SYSTEM = (
        "You are an impartial evaluator. Rate the quality of an answer 1-10.\n"
        'Respond ONLY with JSON: {"score": <int>, "reasoning": "<string>"}'
    )
    prompt = f"Question: {question}\n\nAnswer: {answer}"
    try:
        r = chat(
            [{"role": "user", "content": prompt}],
            system=JUDGE_SYSTEM,
            max_tokens=150,
        )
        raw = get_text(r)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        score = int(data["score"])
        reasoning = data.get("reasoning", "")
        return score >= threshold, score, reasoning
    except Exception as e:
        return False, 5, f"Parse error: {e}"


EXACT_CASES = [
    pytest.param("What is 15 + 27?", ["42"], id="math-simple"),
    pytest.param("What is the capital of France?", ["paris"], id="geography-capital"),
    pytest.param("What does HTTP stand for?", ["hypertext", "transfer", "protocol"], id="acronym-http"),
    pytest.param("What is the chemical symbol for water?", ["h2o"], id="chemistry-water"),
    pytest.param("What does CPU stand for?", ["central", "processing", "unit"], id="acronym-cpu"),
    pytest.param("What is 2 to the power of 10?", ["1024"], id="math-power"),
]

LLM_JUDGED_CASES = [
    pytest.param("Explain object-oriented programming in simple terms.", 7, id="explain-oop"),
    pytest.param("What are the pros and cons of using microservices?", 6, id="microservices-tradeoffs"),
    pytest.param("How would you debug a memory leak in a Python application?", 7, id="debug-memory-leak"),
    pytest.param("Explain the difference between TCP and UDP.", 6, id="tcp-vs-udp"),
    pytest.param("What is the purpose of a load balancer?", 7, id="load-balancer"),
]


@pytest.fixture(scope="session")
def response_cache() -> dict:
    return {}


@pytest.mark.parametrize("question,keywords", EXACT_CASES)
def test_exact_cases(question: str, keywords: list[str], response_cache: dict):
    if question not in response_cache:
        response_cache[question] = agent(question)
    answer = response_cache[question]
    result = exact_match_eval(answer, keywords)
    assert result, f"Answer '{answer[:80]}' missing keywords {keywords}"


@pytest.mark.parametrize("question,threshold", LLM_JUDGED_CASES)
def test_llm_judged_cases(question: str, threshold: int, response_cache: dict):
    if question not in response_cache:
        response_cache[question] = agent(question)
    answer = response_cache[question]
    passed, score, reasoning = llm_judge_eval(question, answer, threshold)
    print(f"\nScore {score}/10: {reasoning}")
    assert passed, f"Score {score} < threshold {threshold}. Reasoning: {reasoning}"


if __name__ == "__main__":
    print("=== Exact Match Tests ===")
    for params in EXACT_CASES:
        q, kw = params.values[0], params.values[1]
        ans = agent(q)
        ok = exact_match_eval(ans, kw)
        print(f"  {'✓' if ok else '✗'} {params.id}: {ans[:60]}...")

    print("\n=== LLM Judge Tests ===")
    for params in LLM_JUDGED_CASES:
        q, thresh = params.values[0], params.values[1]
        ans = agent(q)
        passed, score, reasoning = llm_judge_eval(q, ans, thresh)
        print(f"  {'✓' if passed else '✗'} {params.id}: score={score}/10 — {reasoning[:60]}...")
