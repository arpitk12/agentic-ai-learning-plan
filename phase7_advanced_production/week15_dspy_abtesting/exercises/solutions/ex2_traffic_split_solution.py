"""
SOLUTION — Exercise 2: A/B Testing Router + Shadow Mode + Statistical Significance
Phase 7 / Week 15

How this solution works:
  TODO 1: MD5 hash of "{experiment_id}:{user_id}" converted to integer, then mod 100
           gives a deterministic bucket 0-99. Same input → same bucket every time.
  TODO 2: ExperimentTracker uses SQLite with a single `results` table. No extra deps.
  TODO 3: shadow_call runs both models concurrently via asyncio.gather; user only
           sees control's response — treatment runs silently for data collection.
  TODO 4: ab_router assigns variant via hash, calls the right model, logs to tracker.
  TODO 5: chi-square test on contingency table [[A_success, A_fail], [B_success, B_fail]].
  TODO 6: Sequential experiment runs until min_samples reached and p < alpha.
  TODO 7: Bayesian test samples from Beta(alpha, beta) posterior and estimates
           P(treatment > control) via Monte Carlo integration.
"""
from __future__ import annotations
import os, json, hashlib, asyncio, sqlite3, time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Callable, Awaitable
from pydantic import BaseModel
import litellm
from dotenv import load_dotenv

load_dotenv()

Variant = Literal["control", "treatment"]

@dataclass
class CallResult:
    variant: Variant
    model: str
    output: str
    latency_ms: float
    cost_usd: float
    success: bool
    error: str | None = None

class ExperimentConfig(BaseModel):
    experiment_id: str
    control_model: str = "openai/gpt-4o-mini"
    treatment_model: str = "openai/gpt-4o-mini"
    treatment_pct: int = 10
    min_samples: int = 100
    alpha: float = 0.05


# ── TODO 1 SOLUTION: Hash-Based Variant Assignment ───────────────────────────

def assign_variant(user_id: str, experiment_id: str, treatment_pct: int = 10) -> Variant:
    """
    Deterministic assignment: same user+experiment always gets the same variant.
    Uses MD5 (not for security — just for fast, uniform distribution).
    """
    hash_input = f"{experiment_id}:{user_id}".encode()
    hash_hex = hashlib.md5(hash_input).hexdigest()
    bucket = int(hash_hex, 16) % 100          # bucket in [0, 99]
    return "treatment" if bucket < treatment_pct else "control"


def verify_assignment_distribution(n_users: int = 1000, treatment_pct: int = 10) -> dict:
    counts: dict[Variant, int] = {"control": 0, "treatment": 0}
    for i in range(n_users):
        variant = assign_variant(f"user_{i}", "exp_verify", treatment_pct)
        counts[variant] += 1

    actual_pct = counts["treatment"] / n_users * 100
    within_tolerance = abs(actual_pct - treatment_pct) <= 2.0
    print(f"  Distribution check: {counts['treatment']}/{n_users} treatment "
          f"({actual_pct:.1f}% vs {treatment_pct}% expected) — "
          f"{'✓ within ±2%' if within_tolerance else '✗ outside tolerance'}")
    return {
        "treatment_count": counts["treatment"],
        "treatment_actual_pct": actual_pct,
        "expected_pct": treatment_pct,
    }


# ── TODO 2 SOLUTION: Experiment Tracker ──────────────────────────────────────

class ExperimentTracker:
    def __init__(self, db_path: str = "./ab_experiment.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                model TEXT NOT NULL,
                success INTEGER NOT NULL,
                latency_ms REAL,
                cost_usd REAL,
                output_quality REAL,
                timestamp TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def record(
        self,
        experiment_id: str,
        user_id: str,
        variant: Variant,
        result: CallResult,
        output_quality: float | None = None,
    ) -> None:
        self.conn.execute("""
            INSERT INTO results
              (experiment_id, user_id, variant, model, success, latency_ms, cost_usd, output_quality, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, user_id, variant, result.model,
            1 if result.success else 0,
            result.latency_ms, result.cost_usd, output_quality,
            datetime.now(timezone.utc).isoformat(),
        ))
        self.conn.commit()

    def get_summary(self, experiment_id: str) -> dict:
        rows = self.conn.execute("""
            SELECT variant, COUNT(*) as n, SUM(success) as successes,
                   AVG(latency_ms) as avg_latency, AVG(cost_usd) as avg_cost
            FROM results WHERE experiment_id = ?
            GROUP BY variant
        """, (experiment_id,)).fetchall()

        summary: dict[str, dict] = {}
        for row in rows:
            variant, n, successes, avg_lat, avg_cost = row
            summary[variant] = {
                "n": n,
                "successes": successes or 0,
                "failures": n - (successes or 0),
                "success_rate": (successes or 0) / n,
                "avg_latency_ms": round(avg_lat or 0, 1),
                "avg_cost_usd": round(avg_cost or 0, 6),
            }
        return summary

    def close(self):
        self.conn.close()


# ── TODO 3 SOLUTION: Shadow Mode ─────────────────────────────────────────────

async def shadow_call(
    messages: list[dict],
    config: ExperimentConfig,
    user_id: str,
    tracker: ExperimentTracker,
) -> CallResult:
    """
    Run both models concurrently. Return control's result to the user.
    Treatment runs silently — its result is only logged.
    """
    async def call_model(model: str, variant: Variant) -> CallResult:
        t0 = time.perf_counter()
        try:
            resp = await litellm.acompletion(model=model, messages=messages, temperature=0.0)
            latency = (time.perf_counter() - t0) * 1000
            cost = litellm.completion_cost(resp) if hasattr(litellm, "completion_cost") else 0.0
            result = CallResult(
                variant=variant, model=model,
                output=resp.choices[0].message.content.strip(),
                latency_ms=latency, cost_usd=cost, success=True,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            result = CallResult(
                variant=variant, model=model, output="",
                latency_ms=latency, cost_usd=0.0, success=False, error=str(e),
            )
        tracker.record(config.experiment_id, user_id, variant, result)
        return result

    # Run both concurrently — user waits only for control
    control_result, treatment_result = await asyncio.gather(
        call_model(config.control_model, "control"),
        call_model(config.treatment_model, "treatment"),
    )
    # Only return control; treatment is logged silently
    print(f"  [Shadow] control={control_result.success} treatment={treatment_result.success} "
          f"| latency: {control_result.latency_ms:.0f}ms vs {treatment_result.latency_ms:.0f}ms")
    return control_result


# ── TODO 4 SOLUTION: Main A/B Router ─────────────────────────────────────────

async def ab_router(
    messages: list[dict],
    config: ExperimentConfig,
    user_id: str,
    tracker: ExperimentTracker,
) -> CallResult:
    variant = assign_variant(user_id, config.experiment_id, config.treatment_pct)
    model = config.treatment_model if variant == "treatment" else config.control_model

    t0 = time.perf_counter()
    try:
        resp = await litellm.acompletion(model=model, messages=messages, temperature=0.0)
        latency = (time.perf_counter() - t0) * 1000
        cost = litellm.completion_cost(resp) if hasattr(litellm, "completion_cost") else 0.0
        result = CallResult(
            variant=variant, model=model,
            output=resp.choices[0].message.content.strip(),
            latency_ms=latency, cost_usd=cost, success=True,
        )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        result = CallResult(
            variant=variant, model=model, output="",
            latency_ms=latency, cost_usd=0.0, success=False, error=str(e),
        )

    tracker.record(config.experiment_id, user_id, variant, result)
    return result


# ── TODO 5 SOLUTION: Chi-Square Significance Test ────────────────────────────

def chi_square_test(summary: dict) -> dict:
    from scipy import stats  # type: ignore

    ctrl = summary.get("control", {})
    trt = summary.get("treatment", {})

    if not ctrl or not trt:
        return {"error": "Missing data for one or both variants"}

    ctrl_success = int(ctrl["successes"])
    ctrl_fail = int(ctrl["failures"])
    trt_success = int(trt["successes"])
    trt_fail = int(trt["failures"])

    total = ctrl_success + ctrl_fail + trt_success + trt_fail
    if total < 2:
        return {"error": "Insufficient data"}

    contingency = [[ctrl_success, ctrl_fail], [trt_success, trt_fail]]
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency, correction=True)

    ctrl_rate = ctrl["success_rate"]
    trt_rate = trt["success_rate"]
    lift = (trt_rate - ctrl_rate) / ctrl_rate * 100 if ctrl_rate > 0 else 0

    significant = p_value < 0.05
    winner = "treatment" if (trt_rate > ctrl_rate and significant) else \
             "control" if (ctrl_rate > trt_rate and significant) else "inconclusive"

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": dof,
        "significant": significant,
        "winner": winner,
        "control_success_rate": round(ctrl_rate, 4),
        "treatment_success_rate": round(trt_rate, 4),
        "lift_pct": round(lift, 2),
        "control_n": ctrl_success + ctrl_fail,
        "treatment_n": trt_success + trt_fail,
    }


def print_experiment_report(config: ExperimentConfig, tracker: ExperimentTracker) -> None:
    summary = tracker.get_summary(config.experiment_id)
    stats_result = chi_square_test(summary)

    print(f"\n{'='*60}")
    print(f"Experiment: {config.experiment_id}")
    print(f"{'='*60}")
    for variant, data in summary.items():
        print(f"\n  {variant.upper()}: n={data['n']} | "
              f"success_rate={data['success_rate']:.1%} | "
              f"avg_latency={data['avg_latency_ms']:.0f}ms")
    print(f"\n  Chi-square: χ²={stats_result.get('chi2','?')} "
          f"p={stats_result.get('p_value','?')} "
          f"significant={stats_result.get('significant','?')}")
    print(f"  Winner: {stats_result.get('winner', '?').upper()} "
          f"(lift: {stats_result.get('lift_pct', '?')}%)")
    print(f"{'='*60}\n")


# ── TODO 6 SOLUTION: Sequential Experiment Runner ────────────────────────────

async def run_sequential_experiment(
    config: ExperimentConfig,
    tracker: ExperimentTracker,
    prompt_generator: Callable[[], list[dict]],
    n_max: int = 200,
) -> dict:
    """Run experiment until significance achieved or n_max reached."""
    print(f"\nRunning sequential experiment '{config.experiment_id}'...")
    users_processed = 0

    for i in range(n_max):
        user_id = f"user_{i:04d}"
        messages = prompt_generator()
        await ab_router(messages, config, user_id, tracker)
        users_processed += 1

        # Check significance every 50 requests after min_samples
        if users_processed >= config.min_samples and users_processed % 50 == 0:
            summary = tracker.get_summary(config.experiment_id)
            result = chi_square_test(summary)
            print(f"  [{users_processed} requests] p={result.get('p_value', '?'):.4f} "
                  f"winner={result.get('winner', '?')}")
            if result.get("significant"):
                print(f"  ✓ Statistical significance reached at {users_processed} requests!")
                return result

    print(f"  Max requests ({n_max}) reached without significance")
    return chi_square_test(tracker.get_summary(config.experiment_id))


# ── TODO 7 SOLUTION: Bayesian A/B Test ───────────────────────────────────────

def bayesian_ab_test(summary: dict, n_samples: int = 50_000) -> dict:
    """
    Compute P(treatment > control) using Beta posterior sampling.
    Beta(alpha, beta) where alpha=successes+1, beta=failures+1 (Jeffreys prior).
    """
    import numpy as np  # type: ignore

    ctrl = summary.get("control", {})
    trt = summary.get("treatment", {})
    if not ctrl or not trt:
        return {"error": "Missing data"}

    ctrl_alpha = ctrl["successes"] + 1     # successes + prior
    ctrl_beta = ctrl["failures"] + 1       # failures + prior
    trt_alpha = trt["successes"] + 1
    trt_beta = trt["failures"] + 1

    # Sample from posterior distributions
    rng = np.random.default_rng(seed=42)
    ctrl_samples = rng.beta(ctrl_alpha, ctrl_beta, n_samples)
    trt_samples = rng.beta(trt_alpha, trt_beta, n_samples)

    # P(treatment > control)
    prob_trt_better = (trt_samples > ctrl_samples).mean()
    # P(treatment > control by at least 1%)
    prob_trt_better_1pct = (trt_samples > ctrl_samples + 0.01).mean()
    # Expected lift
    expected_lift = (trt_samples - ctrl_samples).mean()

    result = {
        "prob_treatment_better": round(float(prob_trt_better), 4),
        "prob_treatment_better_by_1pct": round(float(prob_trt_better_1pct), 4),
        "expected_lift": round(float(expected_lift), 4),
        "control_posterior_mean": round(ctrl_alpha / (ctrl_alpha + ctrl_beta), 4),
        "treatment_posterior_mean": round(trt_alpha / (trt_alpha + trt_beta), 4),
        "recommendation": "ship treatment" if prob_trt_better > 0.95 else
                          "ship control" if prob_trt_better < 0.05 else
                          "need more data",
    }
    print(f"\n  Bayesian result: P(treatment>control)={result['prob_treatment_better']:.1%} "
          f"→ {result['recommendation'].upper()}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== A/B Testing Router — SOLUTION ===\n")

    print("1. Verifying hash assignment distribution...")
    verify_assignment_distribution(1000, treatment_pct=10)

    config = ExperimentConfig(
        experiment_id="compliance_model_v2_test",
        control_model="openai/gpt-4o-mini",
        treatment_model="openai/gpt-4o-mini",  # same model — use different prompts in real test
        treatment_pct=30,
        min_samples=20,
        alpha=0.05,
    )
    tracker = ExperimentTracker("./ab_test_solution.db")

    COMPLIANCE_PROMPT = [
        {"role": "system", "content": "You are a compliance expert."},
        {"role": "user", "content": "Is 'Vendor processes EU PII without DPA' a compliance risk? Answer: yes/no"},
    ]

    print("\n2. Running shadow mode (10 requests)...")
    for i in range(5):
        await shadow_call(COMPLIANCE_PROMPT, config, f"shadow_user_{i}", tracker)

    print("\n3. Running sequential A/B experiment (30 requests)...")
    def prompt_gen():
        return COMPLIANCE_PROMPT
    final = await run_sequential_experiment(config, tracker, prompt_gen, n_max=30)

    print_experiment_report(config, tracker)

    print("4. Bayesian analysis...")
    summary = tracker.get_summary(config.experiment_id)
    bayesian_ab_test(summary)

    tracker.close()

if __name__ == "__main__":
    asyncio.run(main())
