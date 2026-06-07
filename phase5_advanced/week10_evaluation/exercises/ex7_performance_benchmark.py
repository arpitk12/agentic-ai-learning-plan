"""
Exercise 7: Performance & Cost Benchmarking
Goal: Measure latency percentiles, cost per run, and step efficiency for your agent.

Theory (from §12.2.6 of the Production Agent Guide):
  - Latency P50 / P95 / P99  — end-to-end response time percentiles
  - Cost per successful run   — USD tokens consumed per completed task
  - Step Count Efficiency     — actual steps ÷ minimum expected steps (lower = better)
  - Token budget compliance   — % of runs that stay within per-run token budget

Why this matters:
  A 10-step sequential agent at 3 s/step = 30 s average latency.
  At P95 (8 s/step) = 80 s — completely unacceptable for an API product.
  Measuring percentiles, not averages, reveals the real user experience.

Tasks:
  1. Complete TimedRun dataclass — add a property cost_usd that calls calc_cost().
  2. Complete run_timed()        — run a single agent call, capture wall time + token counts.
  3. Complete percentile()       — compute Pth percentile of a sorted list.
  4. Complete compute_perf_report() — aggregate N TimedRuns into PerformanceReport.
  5. Complete print_perf_report()   — print formatted table with pass/fail indicators.
  6. (Bonus) Run 20 concurrent requests and measure whether latency stays flat.

Run:
  python ex7_performance_benchmark.py

Expected output (rough — actual depends on your network + model):
  Benchmarking agent with 15 runs...
  ┌─────────────────────────────────────┐
  │  PERFORMANCE REPORT (n=15)          │
  ├─────────────────────────────────────┤
  │  Latency P50         :   3.24 s     │
  │  Latency P95         :   7.81 s  ✅ │
  │  Latency P99         :  12.33 s  ✅ │
  │  Cost / run (avg)    :  $0.0003     │
  │  Cost / run (P95)    :  $0.0008     │
  │  Token budget comply :   93.3%  ✅  │
  │  Step efficiency     :   1.43 ×  ✅ │
  │  Success rate        :  100.0%  ✅  │
  └─────────────────────────────────────┘
"""

import os, sys, time, asyncio, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import achat, get_text, calc_cost, MODEL

load_dotenv()

# ── Thresholds ─────────────────────────────────────────────────────────────────

LATENCY_P95_TARGET_S  = 30.0     # P95 latency must be ≤ 30 s
LATENCY_P99_TARGET_S  = 60.0     # P99 latency must be ≤ 60 s
COST_PER_RUN_USD      = 0.01     # per-run token budget ceiling
STEP_EFFICIENCY_MAX   = 2.0      # actual/ideal steps — must be ≤ 2×
TOKEN_BUDGET_PER_RUN  = 4000     # total tokens (in + out) budget per run

# ── Benchmark query corpus ─────────────────────────────────────────────────────

BENCHMARK_QUERIES = [
    ("q01", "What is the time complexity of merge sort?",                        1),
    ("q02", "Explain Python's GIL in 3 sentences.",                              1),
    ("q03", "Write a Python function to flatten a nested list.",                 1),
    ("q04", "What is the difference between TCP and UDP?",                       1),
    ("q05", "Summarise the CAP theorem.",                                        1),
    ("q06", "What are the SOLID principles? Give one sentence per principle.",   1),
    ("q07", "Explain how a hash map handles collisions.",                        1),
    ("q08", "Write a SQL query that finds the top 5 customers by revenue.",      1),
    ("q09", "What is the difference between a process and a thread?",            1),
    ("q10", "Explain eventual consistency with a real-world example.",           1),
    ("q11", "What is the purpose of the Python __slots__ declaration?",          1),
    ("q12", "Write a regular expression that matches valid email addresses.",    1),
    ("q13", "What are the trade-offs of microservices vs monoliths?",            1),
    ("q14", "Explain the difference between authentication and authorisation.",  1),
    ("q15", "What is backpressure in reactive systems?",                         1),
]
# Third element = expected_steps (minimum ideal steps for a simple Q&A = 1)

SYSTEM = "You are a helpful AI assistant. Answer questions accurately and concisely."


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TimedRun:
    query_id:    str
    latency_s:   float
    input_tokens:  int
    output_tokens: int
    response:    str
    steps:       int = 1          # actual LLM calls made (1 for simple Q&A)
    ideal_steps: int = 1          # minimum steps needed
    success:     bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # TODO 1: Add a property cost_usd
    # ─────────────────────────────────────────────────────────────────────────
    # Use calc_cost(MODEL, input_tokens, output_tokens) from llm.py.
    # Property name: cost_usd
    # Hint: @property decorator, call calc_cost with self.input_tokens and self.output_tokens.
    # TODO: implement this property.


@dataclass
class PerformanceReport:
    n:                    int
    latency_p50_s:        float
    latency_p95_s:        float
    latency_p99_s:        float
    avg_cost_usd:         float
    p95_cost_usd:         float
    token_budget_comply:  float   # fraction within budget
    step_efficiency:      float   # avg (actual / ideal)
    success_rate:         float


# ─────────────────────────────────────────────────────────────────────────────
# TODO 2: Complete run_timed()
# ─────────────────────────────────────────────────────────────────────────────

async def run_timed(query_id: str, query: str, ideal_steps: int = 1) -> TimedRun:
    """
    Run one LLM call, capture wall-clock time and token counts.

    TODO:
    1. Record start = time.monotonic().
    2. Call achat([{"role": "user", "content": query}], system=SYSTEM, max_tokens=500).
    3. Record end = time.monotonic().
    4. Extract text with get_text().
    5. Extract token counts from the response object:
       - response.usage.prompt_tokens     → input_tokens
       - response.usage.completion_tokens → output_tokens
       (Fall back to 0 if .usage is None.)
    6. Return a TimedRun with latency_s = end - start, steps=1, success=True.

    Wrap in try/except Exception → return TimedRun with success=False, latency_s=0.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 3: Complete percentile()
# ─────────────────────────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float:
    """
    Compute the Pth percentile of a list of floats (p in 0..100).

    TODO:
    1. Sort the list.
    2. Calculate index = (p / 100) * (len - 1).
    3. Interpolate between floor and ceil indices.

    Hint: Use int() for floor, min(floor+1, len-1) for ceil.
          result = lower + (upper - lower) * fraction
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 4: Complete compute_perf_report()
# ─────────────────────────────────────────────────────────────────────────────

def compute_perf_report(runs: list[TimedRun]) -> PerformanceReport:
    """
    Aggregate a list of TimedRuns into a PerformanceReport.

    TODO:
    1. latency_p50/p95/p99: call percentile() on [r.latency_s for r in runs].
    2. avg_cost_usd:        mean of [r.cost_usd for r in runs].
    3. p95_cost_usd:        percentile(costs, 95).
    4. token_budget_comply: fraction where (r.input_tokens + r.output_tokens) <= TOKEN_BUDGET_PER_RUN.
    5. step_efficiency:     mean of [r.steps / r.ideal_steps for r in runs].
    6. success_rate:        fraction where r.success is True.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# TODO 5: Complete print_perf_report()
# ─────────────────────────────────────────────────────────────────────────────

def print_perf_report(report: PerformanceReport):
    """
    Print a formatted performance report table with ✅/❌ against targets.

    TODO: print each metric with its target and pass/fail indicator.
    Use the LATENCY_P95_TARGET_S, LATENCY_P99_TARGET_S, COST_PER_RUN_USD,
    STEP_EFFICIENCY_MAX, TOKEN_BUDGET_PER_RUN constants.

    Format example:
      Latency P50         :   3.24 s
      Latency P95         :   7.81 s  ✅  (target ≤ 30.0 s)
      Cost per run (avg)  :  $0.0003
    """
    raise NotImplementedError


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    print(f"Benchmarking agent with {len(BENCHMARK_QUERIES)} queries...")
    print("(Runs are concurrent — total wall time ≈ max individual latency)\n")

    tasks = [run_timed(qid, q, ideal) for qid, q, ideal in BENCHMARK_QUERIES]
    runs  = list(await asyncio.gather(*tasks))

    report = compute_perf_report(runs)
    print_perf_report(report)

    # Bonus: show per-query latency
    print("\nPer-query latency:")
    for r in sorted(runs, key=lambda x: -x.latency_s):
        bar = "█" * int(r.latency_s * 2)
        ok  = "✅" if r.success else "❌"
        print(f"  {r.query_id}  {r.latency_s:5.2f}s  {bar}  {ok}")


if __name__ == "__main__":
    asyncio.run(main())
