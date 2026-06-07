"""
Exercise 8: Multi-Turn Conversation Evaluation
Goal: Evaluate agent quality across a full conversation session, not just single turns.

Theory (from §12.6 Challenge 4 of the Production Agent Guide):
  Multi-turn evaluation must check:
  - Turn accuracy          — each individual turn answers correctly
  - Context continuity     — agent remembers and uses earlier turns
  - Context drift          — agent doesn't contradict what it said earlier
  - Topic switching        — agent handles a change of subject gracefully
  - Coreference resolution — agent understands "it" / "that" / "the first one"

Why this matters:
  A single-turn eval score of 95% does NOT mean your chatbot is 95% good.
  If the agent loses context on turn 3, the rest of the conversation is broken.
  You must evaluate the *session* as a unit.

Tasks:
  1. Complete check_turn()      — score a single turn (keywords present? refusal? context ref correct?).
  2. Complete eval_scenario()   — run a full conversation, score every turn, return ScenarioResult.
  3. Complete detect_contradiction() — ask LLM if two statements contradict each other.
  4. Complete run_all_scenarios()   — evaluate all scenarios, return summary stats.
  5. Add one new conversation scenario of your own (topic of your choice, min 4 turns).

Run:
  python ex8_conversation_eval.py

Expected output:
  MULTI-TURN CONVERSATION EVALUATION
  ====================================
  [python_basics]        5/5 turns  100.0%  ✅  no contradictions
  [topic_switch]         4/4 turns   75.0%  ⚠️   turn 3 failed: ...
  [coreference]          4/4 turns  100.0%  ✅  no contradictions
  [context_continuity]   5/5 turns  100.0%  ✅  no contradictions

  Overall: 18/18 turns  (100.0%)  ✅ PASS
"""

import os, sys, re, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

SYSTEM = "You are a helpful AI assistant. Remember everything in the conversation and be consistent."


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Turn:
    user:             str
    expected_keywords: list[str]    # all must appear in response (case-insensitive)
    must_reference:   list[str] = field(default_factory=list)   # earlier concepts that must be mentioned
    context_check:    str        = ""  # optional: substring that proves context was remembered


@dataclass
class Scenario:
    id:     str
    turns:  list[Turn]


@dataclass
class TurnResult:
    turn_idx:  int
    user:      str
    response:  str
    passed:    bool
    reason:    str          # why it failed (empty if passed)


@dataclass
class ScenarioResult:
    scenario_id:     str
    turn_results:    list[TurnResult]
    contradictions:  list[str]   # pairs of contradicting statements
    pass_rate:       float
    overall_passed:  bool


# ── Conversation scenarios ─────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [

    Scenario("python_basics", [
        Turn("What is a Python list?",
             expected_keywords=["mutable", "ordered", "elements"],
             context_check="list"),
        Turn("Can you give me an example of creating one?",
             expected_keywords=["[", "]"],
             must_reference=["list"]),           # "it" refers to list from prev turn
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
             expected_keywords=["pool", "connections", "pg"],
             must_reference=["connection pool", "pool"]),
        Turn("How would I implement that in my Flask app?",
             expected_keywords=["flask", "sqlalchemy", "pool"],
             must_reference=["flask", "connection"]),
    ]),

    # TODO: Add one more scenario of your own (min 4 turns).
    # Ideas: debugging session, recipe help with ingredient substitution,
    #        coding a class step-by-step, travel planning with preferences remembered.
]


# ── Agent ──────────────────────────────────────────────────────────────────────

async def agent_turn(history: list[dict]) -> str:
    """Call the LLM with the full conversation history."""
    response = await achat(history, system=SYSTEM, max_tokens=400)
    return get_text(response)


# ─────────────────────────────────────────────────────────────────────────────
# TODO 1: Complete check_turn()
# ─────────────────────────────────────────────────────────────────────────────

def check_turn(turn: Turn, response: str) -> tuple[bool, str]:
    """
    Evaluate a single turn response.
    Returns (passed: bool, reason: str).

    Rules:
    1. All expected_keywords must appear in the response (case-insensitive).
    2. All must_reference items must appear in the response (case-insensitive).
    3. If context_check is set, it must appear in the response (case-insensitive).

    On first failure, return (False, <reason string explaining what was missing>).
    Return (True, "") if everything passes.

    TODO: implement using any/all and .lower() string comparisons.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: Complete detect_contradiction()
# ─────────────────────────────────────────────────────────────────────────────

async def detect_contradiction(stmt1: str, stmt2: str) -> bool:
    """
    Ask the LLM: do these two statements contradict each other?
    Return True if they contradict, False if they're consistent.

    TODO:
    1. Build a prompt: "Do these two statements contradict each other?
       Statement 1: {stmt1}
       Statement 2: {stmt2}
       Answer ONLY 'YES' or 'NO'."
    2. Call achat with max_tokens=10.
    3. Return True if "YES" is in the response (case-insensitive), False otherwise.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Complete eval_scenario()
# ─────────────────────────────────────────────────────────────────────────────

async def eval_scenario(scenario: Scenario) -> ScenarioResult:
    """
    Run a full multi-turn conversation and score every turn.

    Algorithm:
    1. history = []
    2. For each turn in scenario.turns:
       a. Append {"role": "user", "content": turn.user} to history.
       b. Call agent_turn(history) → response.
       c. Append {"role": "assistant", "content": response} to history.
       d. Call check_turn(turn, response) → (passed, reason).
       e. Append TurnResult to turn_results.
    3. Check for contradictions between adjacent assistant responses
       (i-th and (i+1)-th assistant turns):
       - Call detect_contradiction(resp_i, resp_i+1) for each adjacent pair.
       - If True, add f"Turn {i} vs {i+1}: ..." to contradictions list.
    4. Compute pass_rate = passed_turns / total_turns.
    5. overall_passed = pass_rate >= 0.80 and no contradictions.
    6. Return ScenarioResult.

    TODO: implement the conversation loop and scoring.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 4: Complete run_all_scenarios()
# ─────────────────────────────────────────────────────────────────────────────

async def run_all_scenarios(scenarios: list[Scenario]) -> list[ScenarioResult]:
    """
    Evaluate all scenarios sequentially (not concurrently — each depends on turn order).
    Return list of ScenarioResult.

    TODO: use a simple for loop + await, collect results.
    """
    raise NotImplementedError


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results: list[ScenarioResult]):
    print("\nMULTI-TURN CONVERSATION EVALUATION")
    print("=" * 50)
    total_turns, total_passed = 0, 0
    for r in results:
        n       = len(r.turn_results)
        passed  = sum(t.passed for t in r.turn_results)
        rate    = r.pass_rate
        icon    = "✅" if r.overall_passed else "⚠️ "
        contra  = f"  ⚡ {len(r.contradictions)} contradiction(s)" if r.contradictions else "  no contradictions"
        print(f"  [{r.scenario_id:<24}]  {passed}/{n} turns  {rate:5.1%}  {icon} {contra}")
        for t in r.turn_results:
            if not t.passed:
                print(f"      turn {t.turn_idx+1} FAIL: {t.reason}")
        total_turns  += n
        total_passed += passed

    overall_rate = total_passed / total_turns if total_turns else 0
    overall_pass = overall_rate >= 0.90
    print(f"\nOverall: {total_passed}/{total_turns} turns ({overall_rate:.1%})  "
          f"{'✅ PASS' if overall_pass else '❌ FAIL'}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    results = await run_all_scenarios(SCENARIOS)
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
