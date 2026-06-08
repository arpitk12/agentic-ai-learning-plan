"""
Exercise 2: A/B Testing Router + Shadow Mode + Statistical Significance
Phase 7 / Week 15 — DSPy + A/B Testing

Goal: Build a production-grade A/B testing system for LLM agents that:
      - deterministically assigns users to variants by hash
      - runs shadow mode (zero user impact)
      - tracks metrics
      - tests for statistical significance before calling a winner

Stack: litellm · scipy · pydantic · sqlite3 (no extra deps)

pip install litellm scipy pydantic python-dotenv

TODOs:
  1. Implement hash-based deterministic variant assignment
  2. Build ExperimentTracker (SQLite-backed metric store)
  3. Implement shadow mode caller (control + treatment in parallel)
  4. Build the main A/B router (traffic split at configurable %)
  5. Implement chi-square significance test
  6. Implement a sequential experiment runner with early stopping
  7. BONUS: Bayesian A/B test using Beta distribution (no p-value required)
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

# ── Types ─────────────────────────────────────────────────────────────────────

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
    treatment_model: str = "openai/gpt-4o-mini"   # swap to your new model
    treatment_pct: int = 10     # % of traffic to treatment
    min_samples: int = 100      # minimum before calling significance
    alpha: float = 0.05         # significance level

# ── TODO 1: Hash-Based Variant Assignment ─────────────────────────────────────

def assign_variant(
    user_id: str,
    experiment_id: str,
    treatment_pct: int = 10,
) -> Variant:
    """
    TODO 1: Deterministically assign a user to control or treatment.

    Algorithm:
    a) hash_input = f"{experiment_id}:{user_id}".encode()
    b) hash_hex = hashlib.md5(hash_input).hexdigest()
    c) bucket = int(hash_hex, 16) % 100
    d) return "treatment" if bucket < treatment_pct else "control"

    The same user_id + experiment_id always gets the same variant.
    This is important for consistency — users shouldn't flip between variants.

    Test your implementation with:
      - assert assign_variant("user_1", "exp_001", 10) in ["control", "treatment"]
      - assert assign_variant("user_1", "exp_001", 10) == assign_variant("user_1", "exp_001", 10)
      - Verify ~10% go to treatment when called with 1000 different user_ids
    """
    # TODO 1: implement here
    raise NotImplementedError

def verify_assignment_distribution(n_users: int = 1000, treatment_pct: int = 10) -> dict:
    """
    TODO 1 (continued): Verify hash assignment gives the expected distribution.

    Generate n_users fake user IDs, assign each, count treatment count.
    Return {"treatment_count": N, "treatment_actual_pct": float, "expected_pct": treatment_pct}
    Print whether the actual % is within ±2% of expected.
    """
    # TODO 1: implement here
    raise NotImplementedError

# ── TODO 2: Experiment Tracker ────────────────────────────────────────────────

class ExperimentTracker:
    """
    TODO 2: SQLite-backed metric tracker for A/B experiments.

    In __init__(self, db_path="ab_experiments.db"):
      - Connect to SQLite and create table:
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT,
            variant TEXT,
            model TEXT,
            success INTEGER,  -- 1/0
            latency_ms REAL,
            cost_usd REAL,
            quality_score REAL,  -- optional, from evaluator
            timestamp TEXT
        )

    Implement:
    a) record(self, experiment_id, result: CallResult, quality_score=None)
       - INSERT a row into the table

    b) get_stats(self, experiment_id) -> dict
       - SELECT and compute:
         - control_total, control_successes, control_avg_latency, control_avg_cost
         - treatment_total, treatment_successes, treatment_avg_latency, treatment_avg_cost
       - Return as a dict

    c) get_samples(self, experiment_id) -> tuple[list[int], list[int]]
       - Return (control_successes_list, treatment_successes_list)
         where each is a list of 1s and 0s (for the chi-square test)
    """
    def __init__(self, db_path: str = "ab_experiments.db"):
        # TODO 2: implement here
        raise NotImplementedError

    def record(self, experiment_id: str, result: CallResult, quality_score: float | None = None):
        # TODO 2: implement here
        raise NotImplementedError

    def get_stats(self, experiment_id: str) -> dict:
        # TODO 2: implement here
        raise NotImplementedError

# ── TODO 3: Shadow Mode Caller ────────────────────────────────────────────────

async def shadow_call(
    user_id: str,
    messages: list[dict],
    config: ExperimentConfig,
    tracker: ExperimentTracker,
) -> CallResult:
    """
    TODO 3: Run control + treatment simultaneously; user always gets control.

    Steps:
    a) Create two async tasks:
       control_task = asyncio.create_task(call_model(config.control_model, messages))
       shadow_task = asyncio.create_task(call_model(config.treatment_model, messages))

    b) Gather both with return_exceptions=True:
       control_result, shadow_result = await asyncio.gather(control_task, shadow_task, return_exceptions=True)

    c) Record shadow_result in tracker (if not an exception):
       tracker.record(config.experiment_id, shadow_result)
       Log whether control.output == shadow_result.output (exact match rate)

    d) Always return control_result to the caller (user sees control only).

    Implement helper:
    async def call_model(model: str, messages: list[dict]) -> CallResult:
        start = time.perf_counter()
        try:
            r = await litellm.acompletion(model=model, messages=messages, max_tokens=200)
            return CallResult(
                variant="control" if model == config.control_model else "treatment",
                model=model,
                output=r.choices[0].message.content,
                latency_ms=(time.perf_counter() - start) * 1000,
                cost_usd=litellm.completion_cost(r) or 0.0,
                success=True,
            )
        except Exception as e:
            return CallResult(variant=..., model=model, output="", latency_ms=...,
                              cost_usd=0.0, success=False, error=str(e))
    """
    # TODO 3: implement here
    raise NotImplementedError

# ── TODO 4: A/B Router ────────────────────────────────────────────────────────

async def ab_router(
    user_id: str,
    messages: list[dict],
    config: ExperimentConfig,
    tracker: ExperimentTracker,
    quality_evaluator: Callable[[str, str], float] | None = None,
) -> CallResult:
    """
    TODO 4: Route traffic based on variant assignment.

    a) variant = assign_variant(user_id, config.experiment_id, config.treatment_pct)

    b) model = config.treatment_model if variant == "treatment" else config.control_model

    c) result = await call_model(model, messages)   # re-use helper from TODO 3

    d) quality_score = None
       If quality_evaluator is provided:
         quality_score = quality_evaluator(messages[-1]["content"], result.output)
         (The evaluator takes (user_query, agent_response) and returns 0.0-1.0)

    e) tracker.record(config.experiment_id, result, quality_score=quality_score)

    f) Return result (the user gets whichever variant they were assigned to).
    """
    # TODO 4: implement here
    raise NotImplementedError

# ── TODO 5: Statistical Significance Testing ──────────────────────────────────

def chi_square_test(
    control_successes: int, control_total: int,
    treatment_successes: int, treatment_total: int,
    alpha: float = 0.05,
) -> dict:
    """
    TODO 5: Run a chi-square test for proportion difference.

    from scipy import stats

    a) Build 2×2 contingency table:
       [[control_successes, control_total - control_successes],
        [treatment_successes, treatment_total - treatment_successes]]

    b) chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    c) control_rate = control_successes / control_total
       treatment_rate = treatment_successes / treatment_total
       relative_lift = (treatment_rate - control_rate) / control_rate

    d) Return:
    {
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
        "control_rate": round(control_rate, 4),
        "treatment_rate": round(treatment_rate, 4),
        "relative_lift": round(relative_lift, 4),
        "winner": "treatment" if (treatment_rate > control_rate and p_value < alpha)
                   else "control" if (control_rate > treatment_rate and p_value < alpha)
                   else "inconclusive",
        "recommendation": str,  # human-readable decision
    }
    """
    # TODO 5: implement here
    raise NotImplementedError

def print_experiment_report(tracker: ExperimentTracker, config: ExperimentConfig) -> None:
    """
    TODO 5 (continued): Print a full experiment report.

    Get stats: tracker.get_stats(config.experiment_id)
    Run significance test.
    Print a formatted table showing:
      | Metric           | Control        | Treatment      |
      |------------------|----------------|----------------|
      | Samples          | N              | N              |
      | Success Rate     | XX.X%          | XX.X%          |
      | Avg Latency      | XXXms          | XXXms          |
      | Avg Cost/call    | $0.000X        | $0.000X        |
      | p-value          | (significance) |                |
      | Winner           | (winner)       |                |
    """
    # TODO 5: implement here
    raise NotImplementedError

# ── TODO 6: Sequential Test with Early Stopping ───────────────────────────────

async def run_sequential_experiment(
    config: ExperimentConfig,
    messages_list: list[list[dict]],
    tracker: ExperimentTracker,
    check_interval: int = 50,
) -> dict:
    """
    TODO 6: Run the experiment, checking significance every check_interval calls.

    Steps:
    a) Loop over messages_list, assigning fake user_ids:
       user_id = f"user_{i:04d}"
       result = await ab_router(user_id, messages, config, tracker)

    b) Every check_interval calls, run chi_square_test using tracker.get_stats().
       If significant: break early and return the result.

    c) If loop ends without significance: return {"winner": "inconclusive", ...}

    d) Print progress every check_interval: "Checked N calls... p={p_value:.4f}"

    Return the chi_square_test result dict.
    """
    # TODO 6: implement here
    raise NotImplementedError

# ── TODO 7 (BONUS): Bayesian A/B Test ─────────────────────────────────────────

def bayesian_ab_test(
    control_successes: int, control_total: int,
    treatment_successes: int, treatment_total: int,
    n_samples: int = 10_000,
) -> dict:
    """
    TODO 7: Bayesian A/B test using Beta distribution.

    No p-value. Instead, compute the probability that treatment is better.

    import numpy as np
    from scipy.stats import beta

    # Beta distribution: Beta(successes + 1, failures + 1) — uniform prior
    control_dist = beta(control_successes + 1, control_total - control_successes + 1)
    treatment_dist = beta(treatment_successes + 1, treatment_total - treatment_successes + 1)

    # Monte Carlo sampling
    control_samples = control_dist.rvs(n_samples)
    treatment_samples = treatment_dist.rvs(n_samples)
    prob_treatment_better = (treatment_samples > control_samples).mean()

    # Expected lift
    expected_lift = (treatment_samples - control_samples).mean()

    Return:
    {
        "prob_treatment_better": round(prob_treatment_better, 4),
        "expected_lift": round(expected_lift, 4),
        "winner": "treatment" if prob_treatment_better > 0.95
                   else "control" if prob_treatment_better < 0.05
                   else "inconclusive",
        "confidence": f"{max(prob_treatment_better, 1-prob_treatment_better):.1%}",
    }
    """
    # TODO 7: implement here
    raise NotImplementedError

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== A/B Testing Router Exercise ===\n")

    config = ExperimentConfig(
        experiment_id="compliance-model-v2",
        control_model="openai/gpt-4o-mini",
        treatment_model="openai/gpt-4o-mini",  # same model for demo; swap in real experiment
        treatment_pct=20,
        min_samples=50,
    )
    tracker = ExperimentTracker()

    # Verify variant distribution
    print("1. Verifying hash-based assignment distribution...")
    dist = verify_assignment_distribution(n_users=1000, treatment_pct=config.treatment_pct)
    print(f"   Expected: {config.treatment_pct}% | Actual: {dist['treatment_actual_pct']:.1f}%\n")

    # Simulate experiment with 100 calls
    print("2. Running 100 simulated A/B calls...")
    test_queries = [
        [{"role": "user", "content": f"Review document {i} for GDPR compliance. Rate risk: low/medium/high/critical."}]
        for i in range(100)
    ]
    result = await run_sequential_experiment(config, test_queries, tracker, check_interval=25)
    print(f"   Experiment result: {result.get('winner', 'unknown')}")
    print(f"   p-value: {result.get('p_value', 'N/A')}\n")

    # Print full report
    print("3. Experiment Report:")
    print_experiment_report(tracker, config)

    # Bayesian test on final numbers
    stats = tracker.get_stats(config.experiment_id)
    print("\n4. Bayesian Test:")
    bayesian = bayesian_ab_test(
        stats["control_successes"], stats["control_total"],
        stats["treatment_successes"], stats["treatment_total"],
    )
    print(f"   P(treatment > control): {bayesian['prob_treatment_better']:.1%}")
    print(f"   Winner: {bayesian['winner']}")
    print(f"   Confidence: {bayesian['confidence']}")

if __name__ == "__main__":
    asyncio.run(main())
