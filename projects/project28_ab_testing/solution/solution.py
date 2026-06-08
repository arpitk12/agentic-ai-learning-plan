"""
Project 28 SOLUTION — A/B Testing Framework for LLM Agents
Hash-based traffic split + shadow mode + chi-square + Bayesian significance testing.
"""
from __future__ import annotations
import os, json, hashlib, asyncio, sqlite3, time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
import litellm
from dotenv import load_dotenv

load_dotenv()

Variant = Literal["control", "treatment"]

# ── Hash-based assignment ─────────────────────────────────────────────────────

def assign_variant(user_id: str, experiment_id: str, treatment_pct: int = 10) -> Variant:
    h = hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest()
    return "treatment" if int(h, 16) % 100 < treatment_pct else "control"


# ── SQLite Tracker ────────────────────────────────────────────────────────────

class ExperimentTracker:
    def __init__(self, db: str = "./ab_project28.db"):
        self.conn = sqlite3.connect(db, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT, user_id TEXT, variant TEXT, model TEXT,
                success INTEGER, latency_ms REAL, cost_usd REAL, timestamp TEXT
            )
        """)
        self.conn.commit()

    def record(self, experiment_id: str, user_id: str, variant: Variant,
               model: str, success: bool, latency_ms: float, cost_usd: float = 0.0):
        self.conn.execute(
            "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?)",
            (experiment_id, user_id, variant, model, 1 if success else 0,
             latency_ms, cost_usd, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def summary(self, experiment_id: str) -> dict:
        rows = self.conn.execute("""
            SELECT variant, COUNT(*), SUM(success), AVG(latency_ms)
            FROM results WHERE experiment_id=? GROUP BY variant
        """, (experiment_id,)).fetchall()
        return {
            r[0]: {"n": r[1], "successes": r[2] or 0, "failures": r[1] - (r[2] or 0),
                   "success_rate": (r[2] or 0) / r[1], "avg_latency_ms": r[3] or 0}
            for r in rows
        }

    def close(self): self.conn.close()


# ── Shadow Mode ───────────────────────────────────────────────────────────────

async def shadow_call(messages: list[dict], control_model: str, treatment_model: str,
                      experiment_id: str, user_id: str, tracker: ExperimentTracker):
    async def call(model: str, variant: Variant):
        t0 = time.perf_counter()
        try:
            resp = await litellm.acompletion(model=model, messages=messages, temperature=0.0)
            latency = (time.perf_counter() - t0) * 1000
            tracker.record(experiment_id, user_id, variant, model, True, latency)
            return resp.choices[0].message.content.strip(), True
        except Exception:
            latency = (time.perf_counter() - t0) * 1000
            tracker.record(experiment_id, user_id, variant, model, False, latency)
            return "", False

    (ctrl_out, ctrl_ok), (trt_out, trt_ok) = await asyncio.gather(
        call(control_model, "control"),
        call(treatment_model, "treatment"),
    )
    return ctrl_out   # user sees only control


# ── A/B Router ────────────────────────────────────────────────────────────────

async def ab_router(messages: list[dict], control_model: str, treatment_model: str,
                    experiment_id: str, user_id: str, treatment_pct: int,
                    tracker: ExperimentTracker) -> str:
    variant = assign_variant(user_id, experiment_id, treatment_pct)
    model = treatment_model if variant == "treatment" else control_model
    t0 = time.perf_counter()
    try:
        resp = await litellm.acompletion(model=model, messages=messages, temperature=0.0)
        latency = (time.perf_counter() - t0) * 1000
        tracker.record(experiment_id, user_id, variant, model, True, latency)
        return resp.choices[0].message.content.strip()
    except Exception:
        latency = (time.perf_counter() - t0) * 1000
        tracker.record(experiment_id, user_id, variant, model, False, latency)
        return ""


# ── Statistical Significance ──────────────────────────────────────────────────

def chi_square_test(summary: dict) -> dict:
    from scipy import stats  # type: ignore
    ctrl, trt = summary.get("control", {}), summary.get("treatment", {})
    if not ctrl or not trt:
        return {"error": "Missing data"}
    table = [[int(ctrl["successes"]), int(ctrl["failures"])],
             [int(trt["successes"]), int(trt["failures"])]]
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=True)
    lift = (trt["success_rate"] - ctrl["success_rate"]) / max(ctrl["success_rate"], 1e-6) * 100
    return {
        "chi2": round(chi2, 4), "p_value": round(p, 6), "significant": p < 0.05,
        "winner": "treatment" if trt["success_rate"] > ctrl["success_rate"] and p < 0.05 else
                  "control" if ctrl["success_rate"] > trt["success_rate"] and p < 0.05 else "inconclusive",
        "lift_pct": round(lift, 2),
        "control_n": int(ctrl["n"]), "treatment_n": int(trt["n"]),
    }


def bayesian_ab_test(summary: dict, n_samples: int = 50_000) -> dict:
    import numpy as np  # type: ignore
    ctrl, trt = summary.get("control", {}), summary.get("treatment", {})
    if not ctrl or not trt:
        return {"error": "Missing data"}
    rng = np.random.default_rng(42)
    ctrl_samples = rng.beta(ctrl["successes"] + 1, ctrl["failures"] + 1, n_samples)
    trt_samples = rng.beta(trt["successes"] + 1, trt["failures"] + 1, n_samples)
    prob = float((trt_samples > ctrl_samples).mean())
    return {
        "prob_treatment_better": round(prob, 4),
        "recommendation": "ship treatment" if prob > 0.95 else "ship control" if prob < 0.05 else "need more data",
        "expected_lift": round(float((trt_samples - ctrl_samples).mean()), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 28: A/B Testing Framework SOLUTION ===\n")

    EXP_ID = "compliance_v2"
    CTRL = "openai/gpt-4o-mini"
    TRTM = "openai/gpt-4o-mini"
    MSG = [{"role": "user", "content": "Is 'no DPA for EU data processing' a critical compliance risk? Answer: yes/no and why in one sentence."}]

    tracker = ExperimentTracker()

    print("1. Running shadow mode (5 requests)...")
    for i in range(5):
        out = await shadow_call(MSG, CTRL, TRTM, EXP_ID, f"shadow_{i}", tracker)

    print("\n2. Running A/B experiment (20 requests, 30% treatment)...")
    for i in range(20):
        await ab_router(MSG, CTRL, TRTM, EXP_ID, f"user_{i}", 30, tracker)

    print("\n3. Results:")
    summary = tracker.summary(EXP_ID)
    for v, s in summary.items():
        print(f"  {v}: n={s['n']} success_rate={s['success_rate']:.1%} latency={s['avg_latency_ms']:.0f}ms")

    print("\n4. Chi-square significance test:")
    chi = chi_square_test(summary)
    print(f"  p={chi.get('p_value')} significant={chi.get('significant')} winner={chi.get('winner')} lift={chi.get('lift_pct')}%")

    print("\n5. Bayesian A/B test:")
    bayes = bayesian_ab_test(summary)
    print(f"  P(treatment > control) = {bayes['prob_treatment_better']:.1%}")
    print(f"  Recommendation: {bayes['recommendation'].upper()}")

    tracker.close()

if __name__ == "__main__":
    asyncio.run(main())
