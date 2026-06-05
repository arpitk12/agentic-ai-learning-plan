"""
Exercise 4: pytest Regression Suite for Agents
Goal: Write deterministic and LLM-judged tests for your agent.

Run with:
  pytest ex4_pytest_agent.py -v
  pytest ex4_pytest_agent.py -v -k "math"          # filter by name
  pytest ex4_pytest_agent.py --tb=short -q          # quiet mode

Tasks:
  1. Complete exact_match_eval() — case-insensitive keyword check.
  2. Complete llm_judge_eval() — LLM rates answer 1-10, return True if >= threshold.
  3. Complete the parametrized test_exact_cases() test.
  4. Complete the parametrized test_llm_judged_cases() test.
  5. Add 2 more cases to EXACT_CASES and LLM_JUDGED_CASES.
  6. (Bonus) Add a @pytest.fixture that caches LLM responses to avoid repeat API calls.

Expected output:
  test_exact_cases[math-simple] PASSED
  test_exact_cases[geography-capital] PASSED
  test_llm_judged_cases[explain-oop] PASSED
  ...
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
import pytest
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

# ── Agent Under Test ──────────────────────────────────────────────────────────

SYSTEM = "You are a helpful AI assistant. Answer questions accurately and concisely."


def agent(question: str) -> str:
    r = chat(
        [{"role": "user", "content": question}],
        system=SYSTEM,
        max_tokens=300,
    )
    return get_text(r)


# ── Evaluation Helpers ────────────────────────────────────────────────────────

def exact_match_eval(answer: str, expected_keywords: list[str]) -> bool:
    """
    Return True if ALL expected_keywords appear in the answer (case-insensitive).
    TODO: implement this (one line is enough)
    """
    raise NotImplementedError


def llm_judge_eval(question: str, answer: str, threshold: int = 7) -> tuple[bool, int, str]:
    """
    Ask an LLM to rate the answer 1–10. Return (passed, score, reasoning).
    
    TODO:
    1. Build a prompt asking the LLM to rate `answer` for `question` as JSON:
       {"score": <int 1-10>, "reasoning": "<string>"}
    2. Parse the JSON response.
    3. Return (score >= threshold, score, reasoning).
    
    Hint: Wrap the chat call in try/except and default to score=5 on parse error.
    """
    JUDGE_SYSTEM = (
        "You are an impartial evaluator. Rate the quality of an answer 1-10.\n"
        "Respond ONLY with JSON: {\"score\": <int>, \"reasoning\": \"<string>\"}"
    )
    raise NotImplementedError


# ── Test Cases ────────────────────────────────────────────────────────────────

# Deterministic tests — answer must contain these keywords
EXACT_CASES = [
    pytest.param(
        "What is 15 + 27?",
        ["42"],
        id="math-simple",
    ),
    pytest.param(
        "What is the capital of France?",
        ["paris"],
        id="geography-capital",
    ),
    pytest.param(
        "What does HTTP stand for?",
        ["hypertext", "transfer", "protocol"],
        id="acronym-http",
    ),
    pytest.param(
        "What is the chemical symbol for water?",
        ["h2o"],
        id="chemistry-water",
    ),
    # TODO: add 2 more exact-match cases
]

# LLM-judged tests — open-ended, judged by another LLM call
LLM_JUDGED_CASES = [
    pytest.param(
        "Explain object-oriented programming in simple terms.",
        7,  # minimum score
        id="explain-oop",
    ),
    pytest.param(
        "What are the pros and cons of using microservices?",
        6,
        id="microservices-tradeoffs",
    ),
    pytest.param(
        "How would you debug a memory leak in a Python application?",
        7,
        id="debug-memory-leak",
    ),
    # TODO: add 2 more LLM-judged cases
]


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def response_cache() -> dict:
    """Cache LLM responses for the test session to avoid duplicate API calls."""
    return {}


@pytest.mark.parametrize("question,keywords", EXACT_CASES)
def test_exact_cases(question: str, keywords: list[str], response_cache: dict):
    """
    TODO:
    1. Check response_cache first; if question is cached, use the cached answer.
    2. Otherwise call agent(question) and store in cache.
    3. Call exact_match_eval(answer, keywords).
    4. Assert the result is True.
    """
    raise NotImplementedError


@pytest.mark.parametrize("question,threshold", LLM_JUDGED_CASES)
def test_llm_judged_cases(question: str, threshold: int, response_cache: dict):
    """
    TODO:
    1. Check response_cache first; if cached, use cached answer.
    2. Otherwise call agent(question) and store in cache.
    3. Call llm_judge_eval(question, answer, threshold).
    4. Print f"Score {score}/10: {reasoning}" for visibility.
    5. Assert passed is True.
    """
    raise NotImplementedError


# ── Standalone Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke-test without pytest
    print("=== Exact Match Tests ===")
    for params in EXACT_CASES:
        q, kw = params.values[0], params.values[1]
        ans = agent(q)
        ok = exact_match_eval(ans, kw)
        status = "✓" if ok else "✗"
        print(f"  [{status}] {params.id}: {ans[:60]}...")

    print("\n=== LLM Judge Tests ===")
    for params in LLM_JUDGED_CASES:
        q, thresh = params.values[0], params.values[1]
        ans = agent(q)
        passed, score, reasoning = llm_judge_eval(q, ans, thresh)
        status = "✓" if passed else "✗"
        print(f"  [{status}] {params.id}: score={score}/10 — {reasoning[:60]}...")
