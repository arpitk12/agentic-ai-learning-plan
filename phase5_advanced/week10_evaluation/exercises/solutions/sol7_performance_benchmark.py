"""
Solution 7: Performance & Cost Benchmarking
"""

import os, sys, time, asyncio, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dataclasses import dataclass
from dotenv import load_dotenv
from llm import achat, get_text, calc_cost, MODEL

load_dotenv()

LATENCY_P95_TARGET_S = 30.0
LATENCY_P99_TARGET_S = 60.0
COST_PER_RUN_USD     = 0.01
STEP_EFFICIENCY_MAX  = 2.0
TOKEN_BUDGET_PER_RUN = 4000

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

SYSTEM = "You are a helpful AI assistant. Answer questions accurately and concisely."


@dataclass
class TimedRun:
    query_id:      str
    latency_s:     float
    input_tokens:  int
    output_tokens: int
    response:      str
    steps:         int  = 1
    ideal_steps:   int  = 1
    success:       bool = True

    @property
    def cost_usd(self) -> float:
        return calc_cost(MODEL, self.input_tokens, self.output_tokens)


@dataclass
class PerformanceReport:
    n:                   int
    latency_p50_s:       float
    latency_p95_s:       float
    latency_p99_s:       float
    avg_cost_usd:        float
    p95_cost_usd:        float
    token_budget_comply: float
    step_efficiency:     float
    success_rate:        float


# ── Solution implementations ───────────────────────────────────────────────────

async def run_timed(query_id: str, query: str, ideal_steps: int = 1) -> TimedRun:
    """Run one LLM call and capture latency + token counts."""
    try:
        start    = time.monotonic()
        response = await achat(
            [{"role": "user", "content": query}],
            system=SYSTEM,
            max_tokens=500,
        )
        latency  = time.monotonic() - start
        text     = get_text(response)

        usage         = getattr(response, "usage", None)
        input_tokens  = getattr(usage, "prompt_tokens",     0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return TimedRun(
            query_id=query_id,
            latency_s=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response=text,
            steps=1,
            ideal_steps=ideal_steps,
            success=True,
        )
    except Exception as e:
        return TimedRun(
            query_id=query_id,
            latency_s=0.0,
            input_tokens=0,
            output_tokens=0,
            response=str(e),
            success=False,
        )


def percentile(values: list[float], p: float) -> float:
    """Compute the Pth percentile of a list (p in 0..100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n           = len(sorted_vals)
    idx         = (p / 100) * (n - 1)
    floor       = int(idx)
    ceil        = min(floor + 1, n - 1)
    fraction    = idx - floor
    return sorted_vals[floor] + (sorted_vals[ceil] - sorted_vals[floor]) * fraction


def compute_perf_report(runs: list[TimedRun]) -> PerformanceReport:
    """Aggregate TimedRuns into a PerformanceReport."""
    if not runs:
        return PerformanceReport(0, 0, 0, 0, 0, 0, 0, 0, 0)

    latencies = [r.latency_s for r in runs]
    costs     = [r.cost_usd  for r in runs]
    tokens    = [r.input_tokens + r.output_tokens for r in runs]

    return PerformanceReport(
        n=len(runs),
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        latency_p99_s=percentile(latencies, 99),
        avg_cost_usd=sum(costs) / len(costs),
        p95_cost_usd=percentile(costs, 95),
        token_budget_comply=sum(1 for t in tokens if t <= TOKEN_BUDGET_PER_RUN) / len(tokens),
        step_efficiency=sum(r.steps / r.ideal_steps for r in runs) / len(runs),
        success_rate=sum(r.success for r in runs) / len(runs),
    )


def print_perf_report(report: PerformanceReport):
    """Print formatted performance report with pass/fail."""
    width = 45
    print("┌" + "─" * width + "┐")
    print(f"│  {'PERFORMANCE REPORT':^{width-2}}  │")
    print(f"│  {f'n={report.n} queries benchmarked':^{width-2}}  │")
    print("├" + "─" * width + "┤")

    def row(label: str, value: str, ok: bool | None = None):
        icon = ("  ✅" if ok else "  ❌") if ok is not None else ""
        line = f"  {label:<22}: {value}{icon}"
        print(f"│{line:<{width}}│")

    row("Latency P50",         f"{report.latency_p50_s:6.2f} s")
    row("Latency P95",         f"{report.latency_p95_s:6.2f} s",
        report.latency_p95_s <= LATENCY_P95_TARGET_S)
    row("Latency P99",         f"{report.latency_p99_s:6.2f} s",
        report.latency_p99_s <= LATENCY_P99_TARGET_S)
    row("Cost/run (avg)",      f"${report.avg_cost_usd:.5f}")
    row("Cost/run (P95)",      f"${report.p95_cost_usd:.5f}",
        report.p95_cost_usd <= COST_PER_RUN_USD)
    row("Token budget comply", f"{report.token_budget_comply:6.1%}",
        report.token_budget_comply >= 0.95)
    row("Step efficiency",     f"{report.step_efficiency:6.2f}×",
        report.step_efficiency <= STEP_EFFICIENCY_MAX)
    row("Success rate",        f"{report.success_rate:6.1%}",
        report.success_rate >= 0.99)

    print("└" + "─" * width + "┘")


async def main():
    print(f"Benchmarking agent with {len(BENCHMARK_QUERIES)} queries (concurrent)...\n")

    tasks = [run_timed(qid, q, ideal) for qid, q, ideal in BENCHMARK_QUERIES]
    runs  = list(await asyncio.gather(*tasks))

    report = compute_perf_report(runs)
    print_perf_report(report)

    print("\nPer-query latency (slowest first):")
    for r in sorted(runs, key=lambda x: -x.latency_s):
        bar = "█" * int(r.latency_s * 2)
        ok  = "✅" if r.success else "❌"
        print(f"  {r.query_id}  {r.latency_s:5.2f}s  {bar:30s}  {ok}  ${r.cost_usd:.5f}")


if __name__ == "__main__":
    asyncio.run(main())
