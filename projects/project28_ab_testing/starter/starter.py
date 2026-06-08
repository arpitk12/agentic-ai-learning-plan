"""Project 28 — A/B Testing + Model Management: Starter File
pip install litellm scipy mlflow fastapi uvicorn pydantic python-dotenv
"""
from __future__ import annotations
import os, hashlib, asyncio, sqlite3, time
from dataclasses import dataclass
from typing import Literal
import litellm
from dotenv import load_dotenv
load_dotenv()

Variant = Literal["control", "treatment"]

@dataclass
class ExperimentConfig:
    experiment_id: str
    control_model: str = "openai/gpt-4o-mini"
    treatment_model: str = "openai/gpt-4o-mini"
    treatment_pct: int = 10
    min_samples: int = 100
    alpha: float = 0.05

# TODO 1: Deterministic variant assignment
# hash(f"{experiment_id}:{user_id}") % 100 < treatment_pct → "treatment"
def assign_variant(user_id: str, experiment_id: str, treatment_pct: int = 10) -> Variant:
    """TODO 1: Hash-based deterministic assignment. Same user always gets same variant."""
    # YOUR CODE HERE
    raise NotImplementedError

def verify_distribution(n: int = 1000, pct: int = 10) -> dict:
    """TODO 1 (cont): Verify ~pct% go to treatment. Return actual distribution stats."""
    raise NotImplementedError

# TODO 2: SQLite experiment tracker
class ExperimentTracker:
    """TODO 2: Track calls per variant. Schema: experiment_id, variant, success, latency, cost."""
    def __init__(self, db_path: str = "ab_experiments.db"):
        # YOUR CODE HERE
        raise NotImplementedError
    def record(self, experiment_id: str, variant: Variant, success: bool, latency_ms: float, cost: float):
        raise NotImplementedError
    def get_stats(self, experiment_id: str) -> dict:
        """Return control/treatment counts, success rates, avg latency + cost."""
        raise NotImplementedError

# TODO 3: Shadow mode (control + treatment in parallel, user gets control)
async def shadow_call(user_id: str, messages: list, config: ExperimentConfig, tracker: ExperimentTracker) -> str:
    """TODO 3: Run both models concurrently. Log shadow result. Return control response."""
    raise NotImplementedError

# TODO 4: A/B router (live traffic split)
async def ab_router(user_id: str, messages: list, config: ExperimentConfig, tracker: ExperimentTracker) -> str:
    """TODO 4: Assign variant, call appropriate model, track result. Return response."""
    raise NotImplementedError

# TODO 5: Statistical significance test
def chi_square_test(ctrl_success: int, ctrl_n: int, trt_success: int, trt_n: int, alpha: float = 0.05) -> dict:
    """TODO 5: Chi-square test. Return {"significant": bool, "p_value": float, "winner": str, "lift": float}"""
    # from scipy import stats
    raise NotImplementedError

def print_report(tracker: ExperimentTracker, config: ExperimentConfig):
    """TODO 5 (cont): Print full experiment report with significance test result."""
    raise NotImplementedError

# TODO 6: MLflow model registry integration
def log_to_mlflow(experiment_id: str, model_name: str, metrics: dict, params: dict):
    """TODO 6: Log experiment to MLflow. Register winning model if significant."""
    # import mlflow
    raise NotImplementedError

# TODO 7 (BONUS): Bayesian A/B test
def bayesian_test(ctrl_success: int, ctrl_n: int, trt_success: int, trt_n: int) -> dict:
    """TODO 7: Beta distribution Monte Carlo. Return {"prob_treatment_better": float, "winner": str}"""
    # import numpy as np; from scipy.stats import beta
    raise NotImplementedError

async def main():
    print("=== Project 28: A/B Testing ===\n")
    config = ExperimentConfig("test-experiment", treatment_pct=20)
    tracker = ExperimentTracker()
    print("Distribution check:", verify_distribution(1000, config.treatment_pct))
    msgs = [{"role": "user", "content": "Classify this contract as low/medium/high risk."}]
    for i in range(60):
        await ab_router(f"user_{i:04d}", msgs, config, tracker)
    print_report(tracker, config)

if __name__ == "__main__":
    asyncio.run(main())
