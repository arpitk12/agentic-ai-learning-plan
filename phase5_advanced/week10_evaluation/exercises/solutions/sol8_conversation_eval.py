"""
Solution 8: Multi-Turn Conversation Evaluation
"""

import os, sys, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

SYSTEM = "You are a helpful AI assistant. Remember everything in the conversation and be consistent."


@dataclass
class Turn:
    user:              str
    expected_keywords: list[str]
    must_reference:    list[str] = field(default_factory=list)
    context_check:     str       = ""


@dataclass
class Scenario:
    id:    str
    turns: list[Turn]


@dataclass
class TurnResult:
    turn_idx:  int
    user:      str
    response:  str
    passed:    bool
    reason:    str


@dataclass
class ScenarioResult:
    scenario_id:    str
    turn_results:   list[TurnResult]
    contradictions: list[str]
    pass_rate:      float
    overall_passed: bool


SCENARIOS: list[Scenario] = [
    Scenario("python_basics", [
        Turn("What is a Python list?",
             expected_keywords=["mutable", "ordered", "elements"],
             context_check="list"),
        Turn("Can you give me an example of creating one?",
             expected_keywords=["[", "]"],
             must_reference=["list"]),
        Turn("Now what's the difference between a list and a tuple?",
             expected_keywords=["immutable", "tuple"],
             must_reference=["list"]),
        Turn("Which one should I use if my data won't change?",
             expected_keywords=["tuple"],
             context_check="tuple"),
        Turn("Can you show me how to create a tuple from the list example you gave?",
             expected_keywords=["tuple(", "list"],
             must_reference=["tuple", "list"]),
    ]),

    Scenario("topic_switch", [
        Turn("Tell me about Python dictionaries.",
             expected_keywords=["key", "value", "dict"],
             context_check="dict"),
        Turn("How do I iterate over the keys and values at the same time?",
             expected_keywords=[".items()", "for"],
             must_reference=["dict"]),
        Turn("Completely different topic — what is Docker used for?",
             expected_keywords=["container", "docker", "image"],
             context_check="container"),
        Turn("How does that relate to virtual environments in Python?",
             expected_keywords=["isolation", "dependencies"],
             must_reference=["docker", "container", "virtual"]),
    ]),

    Scenario("coreference", [
        Turn("I have two functions: add(a, b) and multiply(a, b). What do they probably do?",
             expected_keywords=["add", "multiply", "sum", "product"]),
        Turn("Which one would I use if I want to combine two numbers?",
             expected_keywords=["add"],
             must_reference=["add", "multiply"]),
        Turn("And the other one?",
             expected_keywords=["multiply"],
             context_check="multiply"),
        Turn("If I call them both on inputs 3 and 4, what do I get?",
             expected_keywords=["7", "12"],
             must_reference=["add", "multiply"]),
    ]),

    Scenario("context_continuity", [
        Turn("My application is slow. I'm using a Python Flask API with a PostgreSQL database.",
             expected_keywords=["flask", "postgresql", "performance"],
             context_check="flask"),
        Turn("I think the bottleneck is in the database queries. What should I look at first?",
             expected_keywords=["index", "query", "explain"],
             must_reference=["postgresql", "database"]),
        Turn("I added indexes but it's still slow. What next?",
             expected_keywords=["cache", "connection pool", "n+1"],
             must_reference=["index"]),
        Turn("Tell me more about connection pooling specifically.",
             expected_keywords=["pool", "connections"],
             must_reference=["connection pool", "pool"]),
        Turn("How would I implement that in my Flask app?",
             expected_keywords=["flask", "sqlalchemy", "pool"],
             must_reference=["flask", "connection"]),
    ]),
]


# ── Agent ──────────────────────────────────────────────────────────────────────

async def agent_turn(history: list[dict]) -> str:
    response = await achat(history, system=SYSTEM, max_tokens=400)
    return get_text(response)


# ── Solution implementations ───────────────────────────────────────────────────

def check_turn(turn: Turn, response: str) -> tuple[bool, str]:
    """Evaluate a single turn response; return (passed, reason)."""
    lower = response.lower()

    # Check all expected keywords
    missing_kw = [kw for kw in turn.expected_keywords if kw.lower() not in lower]
    if missing_kw:
        return False, f"Missing expected keywords: {missing_kw}"

    # Check all must_reference items (context memory check)
    missing_ref = [r for r in turn.must_reference if r.lower() not in lower]
    if missing_ref:
        return False, f"Missing context references: {missing_ref}"

    # Check context_check phrase
    if turn.context_check and turn.context_check.lower() not in lower:
        return False, f"Missing context_check: {turn.context_check!r}"

    return True, ""


async def detect_contradiction(stmt1: str, stmt2: str) -> bool:
    """Ask the LLM if two statements contradict each other."""
    prompt = (
        f"Do these two statements contradict each other?\n"
        f"Statement 1: {stmt1[:300]}\n"
        f"Statement 2: {stmt2[:300]}\n"
        f"Answer ONLY 'YES' or 'NO'."
    )
    response = await achat([{"role": "user", "content": prompt}], max_tokens=10)
    text = get_text(response).strip().upper()
    return "YES" in text


async def eval_scenario(scenario: Scenario) -> ScenarioResult:
    """Run a full multi-turn conversation and score every turn."""
    history:       list[dict]       = []
    turn_results:  list[TurnResult] = []
    asst_responses: list[str]       = []  # for contradiction check

    for i, turn in enumerate(scenario.turns):
        history.append({"role": "user", "content": turn.user})
        response = await agent_turn(history)
        history.append({"role": "assistant", "content": response})
        asst_responses.append(response)

        passed, reason = check_turn(turn, response)
        turn_results.append(TurnResult(
            turn_idx=i, user=turn.user,
            response=response, passed=passed, reason=reason,
        ))

    # Check for contradictions between adjacent assistant turns
    contradictions: list[str] = []
    for i in range(len(asst_responses) - 1):
        try:
            is_contra = await detect_contradiction(asst_responses[i], asst_responses[i + 1])
            if is_contra:
                contradictions.append(f"Turn {i+1} vs Turn {i+2} contradict")
        except Exception:
            pass  # don't let contradiction check fail the whole eval

    passed_count = sum(t.passed for t in turn_results)
    pass_rate    = passed_count / len(turn_results) if turn_results else 0.0
    overall_pass = pass_rate >= 0.80 and not contradictions

    return ScenarioResult(
        scenario_id=scenario.id,
        turn_results=turn_results,
        contradictions=contradictions,
        pass_rate=pass_rate,
        overall_passed=overall_pass,
    )


async def run_all_scenarios(scenarios: list[Scenario]) -> list[ScenarioResult]:
    """Evaluate all scenarios sequentially (conversations are order-dependent)."""
    results = []
    for scenario in scenarios:
        print(f"  Running scenario: {scenario.id}...")
        result = await eval_scenario(scenario)
        results.append(result)
    return results


def print_report(results: list[ScenarioResult]):
    print("\nMULTI-TURN CONVERSATION EVALUATION")
    print("=" * 56)
    total_turns, total_passed = 0, 0
    for r in results:
        n      = len(r.turn_results)
        passed = sum(t.passed for t in r.turn_results)
        icon   = "✅" if r.overall_passed else "⚠️ "
        contra = f"  ⚡ {len(r.contradictions)} contradiction(s)" if r.contradictions else "  no contradictions"
        print(f"  [{r.scenario_id:<24}]  {passed}/{n} turns  {r.pass_rate:5.1%}  {icon}{contra}")
        for t in r.turn_results:
            if not t.passed:
                print(f"      turn {t.turn_idx+1} FAIL: {t.reason}")
        total_turns  += n
        total_passed += passed

    overall_rate = total_passed / total_turns if total_turns else 0.0
    overall_pass = overall_rate >= 0.90
    print(f"\nOverall: {total_passed}/{total_turns} turns ({overall_rate:.1%})  "
          f"{'✅ PASS' if overall_pass else '❌ FAIL'}")


async def main():
    results = await run_all_scenarios(SCENARIOS)
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
