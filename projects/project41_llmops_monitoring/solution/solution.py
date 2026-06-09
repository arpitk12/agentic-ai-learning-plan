"""
Project 41 — LLMOps Monitoring Hub (SOLUTION)
==============================================
Full implementation: TraceLogger, CostCalculator, EmbeddingDriftDetector,
QualityDriftMonitor, SLOTracker, AlertManager, MonitoredAgent, FastAPI.

Run:  python solution.py           (simulation)
      python solution.py --server  (API at :8000)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
    REGISTRY,
)
import litellm

load_dotenv()

# ─────────────────────────────────────────────────
# Sample data
# ─────────────────────────────────────────────────
REFERENCE_QUERIES = [
    "What does Article 28 of GDPR require for data processors?",
    "Explain the right to erasure under GDPR Article 17.",
    "What is a Data Processing Agreement and when is it required?",
    "Does GDPR apply to companies outside the EU?",
    "What are the penalties for GDPR non-compliance?",
    "What personal data categories require explicit consent?",
    "Explain legitimate interest as a lawful basis under GDPR.",
    "What is the difference between a data controller and a processor?",
    "How long can personal data be retained under GDPR?",
    "What rights do data subjects have under GDPR?",
] * 5  # 50

DRIFTED_QUERIES = [
    "Write me a poem about the ocean.",
    "What is the capital of France?",
    "How do I make pasta carbonara?",
    "Explain quantum entanglement.",
    "What are the best hiking trails in Colorado?",
    "How does photosynthesis work?",
    "What is the plot of Hamlet?",
    "How do I train a machine learning model?",
    "What is blockchain technology?",
    "Recommend a good science fiction novel.",
] * 15  # 150


# ══════════════════════════════════════════════════════════════════════════════
# 1 — TraceLogger
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CallTrace:
    trace_id:       str
    model:          str
    prompt_name:    str
    prompt_version: str
    user_id:        str
    tokens_in:      int   = 0
    tokens_out:     int   = 0
    latency_ms:     float = 0.0
    cost_usd:       float = 0.0
    error:          str | None = None
    quality_score:  float | None = None
    timestamp:      str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TraceLogger:
    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT, timestamp TEXT, model TEXT,
                prompt_name TEXT, prompt_version TEXT, user_id TEXT,
                tokens_in INT, tokens_out INT, latency_ms REAL,
                cost_usd REAL, error TEXT, quality_score REAL
            )
        """)
        self.db.commit()

    def log(self, trace: CallTrace):
        self.db.execute(
            "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (trace.trace_id, trace.timestamp, trace.model,
             trace.prompt_name, trace.prompt_version, trace.user_id,
             trace.tokens_in, trace.tokens_out, trace.latency_ms,
             trace.cost_usd, trace.error, trace.quality_score),
        )
        self.db.commit()

    def get_recent(self, n: int = 100) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM traces ORDER BY rowid DESC LIMIT ?", (n,)
        ).fetchall()
        cols = ["trace_id","timestamp","model","prompt_name","prompt_version",
                "user_id","tokens_in","tokens_out","latency_ms","cost_usd","error","quality_score"]
        return [dict(zip(cols, r)) for r in rows]

    def get_stats(self, hours: int = 24) -> dict:
        rows = self.db.execute("SELECT latency_ms, cost_usd, error, quality_score FROM traces").fetchall()
        if not rows:
            return {"avg_latency_ms": 0, "total_cost_usd": 0, "error_count": 0,
                    "call_count": 0, "avg_quality_score": None, "p95_latency_ms": 0}
        latencies = sorted(r[0] for r in rows)
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        costs    = [r[1] for r in rows]
        errors   = [r[2] for r in rows if r[2]]
        qualities = [r[3] for r in rows if r[3] is not None]
        return {
            "avg_latency_ms":    sum(latencies) / len(latencies),
            "p95_latency_ms":    p95,
            "total_cost_usd":    sum(costs),
            "error_count":       len(errors),
            "call_count":        len(rows),
            "avg_quality_score": sum(qualities) / len(qualities) if qualities else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2 — CostCalculator
# ══════════════════════════════════════════════════════════════════════════════

class CostCalculator:
    # Per-million tokens (in USD)
    PRICES: dict[str, dict[str, float]] = {
        "gpt-4o-mini":     {"in": 0.15,  "out": 0.60},
        "gpt-4o":          {"in": 2.50,  "out": 10.00},
        "gpt-4.1":         {"in": 2.00,  "out": 8.00},
        "claude-haiku-4":  {"in": 0.80,  "out": 4.00},
        "claude-sonnet-4": {"in": 3.00,  "out": 15.00},
        "claude-opus-4":   {"in": 15.00, "out": 75.00},
    }

    def calculate(self, model: str, tokens_in: int, tokens_out: int) -> float:
        clean = model.split("/")[-1] if "/" in model else model
        prices = self.PRICES.get(clean, self.PRICES["gpt-4o-mini"])
        return (tokens_in * prices["in"] + tokens_out * prices["out"]) / 1_000_000


# ══════════════════════════════════════════════════════════════════════════════
# 3 — EmbeddingDriftDetector (TF-IDF + PCA + KS test)
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingDriftDetector:
    def __init__(
        self,
        reference_size: int = 50,
        window_size:    int = 50,
        check_interval: int = 25,
        significance:   float = 0.05,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer     = TfidfVectorizer(max_features=200, stop_words="english")
        self.reference_size = reference_size
        self.window_size    = window_size
        self.check_interval = check_interval
        self.significance   = significance
        self._reference: list[str] = []
        self._window:    list[str] = []
        self._locked     = False
        self._call_count = 0
        self._pca_component: np.ndarray | None = None

    def add(self, text: str):
        self._call_count += 1
        if not self._locked:
            self._reference.append(text)
            if len(self._reference) >= self.reference_size:
                # Fit vectorizer and compute first PCA component
                matrix = self.vectorizer.fit_transform(self._reference).toarray()
                mat_c  = matrix - matrix.mean(axis=0)
                _, _, vt = np.linalg.svd(mat_c, full_matrices=False)
                self._pca_component = vt[0]
                self._locked = True
        else:
            self._window.append(text)
            if len(self._window) > self.window_size:
                self._window.pop(0)

    def _project(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts).toarray()
        mat_c  = matrix - self.vectorizer.transform(self._reference).toarray().mean(axis=0)
        return mat_c @ self._pca_component

    def check(self) -> dict:
        if not self._locked or len(self._window) < 10:
            return {"drift": False, "reason": "building reference",
                    "n_reference": len(self._reference), "n_current": len(self._window)}
        from scipy.stats import ks_2samp
        ref_proj = self._project(self._reference)
        cur_proj = self._project(self._window)
        stat, p_value = ks_2samp(ref_proj, cur_proj)
        drift = p_value < self.significance
        return {
            "drift":       drift,
            "ks_stat":     float(stat),
            "p_value":     float(p_value),
            "n_reference": len(self._reference),
            "n_current":   len(self._window),
            "reason": (
                f"Drift detected! KS={stat:.3f}, p={p_value:.4f}" if drift
                else f"No drift. KS={stat:.3f}, p={p_value:.4f}"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4 — QualityDriftMonitor
# ══════════════════════════════════════════════════════════════════════════════

class QualityDriftMonitor:
    def __init__(
        self,
        judge_model:    str   = "gpt-4o-mini",
        sample_rate:    float = 0.05,
        window_size:    int   = 50,
        drop_threshold: float = 0.5,
    ):
        self.judge_model    = judge_model
        self.sample_rate    = sample_rate
        self.scores: deque  = deque(maxlen=window_size)
        self.baseline: float | None = None
        self.drop_threshold = drop_threshold

    def _llm_judge(self, query: str, response: str) -> float:
        try:
            resp = litellm.completion(
                model=self.judge_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rate this response for accuracy and helpfulness on a scale of 1-10. "
                        f"ONLY output a single number.\n\nQuery: {query}\nResponse: {response}"
                    ),
                }],
                max_tokens=5,
                temperature=0.0,
            )
            return float(resp.choices[0].message.content.strip().split()[0])
        except Exception:
            return 7.0

    def maybe_score(self, query: str, response: str) -> float | None:
        if random.random() > self.sample_rate:
            return None
        score = self._llm_judge(query, response)
        self.scores.append(score)
        return score

    def check(self) -> dict:
        if len(self.scores) < 5:
            return {"drift": False, "reason": "collecting data",
                    "sample_count": len(self.scores)}
        avg = sum(self.scores) / len(self.scores)
        if self.baseline is None:
            self.baseline = avg
            return {"drift": False, "reason": f"Baseline set: {self.baseline:.2f}/10",
                    "current_avg": avg, "baseline": self.baseline,
                    "drop": 0.0, "sample_count": len(self.scores)}
        drop  = self.baseline - avg
        drift = drop > self.drop_threshold
        return {
            "drift":        drift,
            "current_avg":  avg,
            "baseline":     self.baseline,
            "drop":         drop,
            "sample_count": len(self.scores),
            "reason": (
                f"Quality drift! Score dropped {drop:.2f} pts ({self.baseline:.2f}→{avg:.2f}/10)"
                if drift else f"Quality OK: {avg:.2f}/10 (baseline {self.baseline:.2f})"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5 — SLOTracker (Prometheus)
# ══════════════════════════════════════════════════════════════════════════════

_LATENCY = Histogram(
    "llm_request_duration_seconds", "LLM request latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
_ERRORS  = Counter(
    "llm_request_errors_total", "LLM errors", labelnames=["error_type"],
)
_QUALITY = Gauge("llm_quality_score", "Current rolling avg quality score")
_COST    = Histogram(
    "llm_cost_per_call_usd", "Cost per call",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05],
)
_TOKENS  = Counter("llm_tokens_total", "Token totals", labelnames=["direction"])


class SLOTracker:
    SLO_LIMITS = {
        "p95_latency_s":       5.0,
        "error_rate_pct":      0.5,
        "quality_score_min":   7.5,
        "cost_per_call_usd_max": 0.015,
    }

    def __init__(self):
        self._latencies: deque = deque(maxlen=100)
        self._errors    = 0
        self._calls     = 0
        self._last_quality: float | None = None

    def track(self, latency_s: float, cost_usd: float, tokens_in: int, tokens_out: int,
              error: str | None = None, quality: float | None = None):
        _LATENCY.observe(latency_s)
        _COST.observe(cost_usd)
        _TOKENS.labels(direction="in").inc(tokens_in)
        _TOKENS.labels(direction="out").inc(tokens_out)
        self._latencies.append(latency_s)
        self._calls += 1
        if error:
            _ERRORS.labels(error_type=error).inc()
            self._errors += 1
        if quality is not None:
            _QUALITY.set(quality)
            self._last_quality = quality

    def slo_status(self) -> dict:
        lats   = sorted(self._latencies)
        p95    = lats[int(len(lats) * 0.95)] if lats else 0.0
        err_rt = (self._errors / self._calls * 100) if self._calls else 0.0

        def _status(value, limit, lower_better=True):
            ok = value <= limit if lower_better else value >= limit
            return "OK" if ok else "WARN"

        return {
            "p95_latency":   {"value": round(p95, 2), "limit": self.SLO_LIMITS["p95_latency_s"],       "status": _status(p95, self.SLO_LIMITS["p95_latency_s"])},
            "error_rate":    {"value": round(err_rt, 2), "limit": self.SLO_LIMITS["error_rate_pct"],   "status": _status(err_rt, self.SLO_LIMITS["error_rate_pct"])},
            "quality_score": {"value": self._last_quality, "limit": self.SLO_LIMITS["quality_score_min"], "status": _status(self._last_quality or 10, self.SLO_LIMITS["quality_score_min"], lower_better=False)},
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6 — AlertManager
# ══════════════════════════════════════════════════════════════════════════════

class AlertManager:
    COOLDOWN_MINUTES = 30

    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT, severity TEXT, message TEXT)"
        )
        self.db.commit()

    def check_and_alert(self, slo_status: dict, drift_status: dict, quality_status: dict) -> list[str]:
        fired = []
        rules = [
            ("WARNING",  quality_status.get("drift", False),       f"Quality drift: {quality_status.get('reason', '')}"),
            ("WARNING",  drift_status.get("drift", False),          f"Embedding drift: {drift_status.get('reason', '')}"),
            ("CRITICAL", slo_status.get("error_rate", {}).get("status") == "WARN", "Error rate SLO breach"),
            ("CRITICAL", slo_status.get("p95_latency", {}).get("status") == "WARN", "P95 latency SLO breach"),
        ]
        for severity, condition, message in rules:
            if condition and self._fire(severity, message):
                fired.append(f"[{severity}] {message}")
        return fired

    def _fire(self, severity: str, message: str) -> bool:
        # Check cooldown
        cutoff = datetime.now(timezone.utc).timestamp() - self.COOLDOWN_MINUTES * 60
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        existing = self.db.execute(
            "SELECT 1 FROM alerts WHERE message=? AND timestamp > ?", (message, cutoff_iso)
        ).fetchone()
        if existing:
            return False
        ts = datetime.now(timezone.utc).isoformat()
        self.db.execute("INSERT INTO alerts (timestamp, severity, message) VALUES (?,?,?)",
                        (ts, severity, message))
        self.db.commit()
        self.send_slack(f"[{severity}] {message}")
        return True

    def send_slack(self, message: str):
        webhook = os.getenv("SLACK_WEBHOOK_URL")
        if webhook:
            try:
                data = json.dumps({"text": message}).encode()
                req  = urllib.request.Request(webhook, data=data,
                                              headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                print(f"  [Slack send failed: {e}]")
        else:
            print(f"  🔔 ALERT: {message}")

    def get_alert_history(self, n: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT timestamp, severity, message FROM alerts ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [{"timestamp": r[0], "severity": r[1], "message": r[2]} for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# 7 — MonitoredAgent
# ══════════════════════════════════════════════════════════════════════════════

class MonitoredAgent:
    def __init__(
        self,
        model:          str = "gpt-4o-mini",
        system_prompt:  str = "You are a helpful compliance assistant.",
        prompt_name:    str = "compliance-agent",
        prompt_version: str = "v1",
    ):
        self.model          = model
        self.system_prompt  = system_prompt
        self.prompt_name    = prompt_name
        self.prompt_version = prompt_version
        self.tracer         = TraceLogger()
        self.cost_calc      = CostCalculator()
        self.drift_detector = EmbeddingDriftDetector()
        self.quality_mon    = QualityDriftMonitor()
        self.slo            = SLOTracker()
        self.alerts         = AlertManager()
        self._call_count    = 0

    def query(self, user_query: str, user_id: str = "anonymous") -> dict:
        trace_id = hashlib.md5(f"{user_query}{time.time()}".encode()).hexdigest()[:8]
        start    = time.time()
        error    = None
        answer   = ""
        tok_in = tok_out = 0

        try:
            resp = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_query},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            answer  = resp.choices[0].message.content or ""
            tok_in  = resp.usage.prompt_tokens     if resp.usage else 0
            tok_out = resp.usage.completion_tokens if resp.usage else 0
        except Exception as e:
            error  = str(e)
            answer = ""

        latency_ms  = (time.time() - start) * 1000
        cost        = self.cost_calc.calculate(self.model, tok_in, tok_out)

        trace = CallTrace(
            trace_id=trace_id, model=self.model, prompt_name=self.prompt_name,
            prompt_version=self.prompt_version, user_id=user_id,
            tokens_in=tok_in, tokens_out=tok_out,
            latency_ms=latency_ms, cost_usd=cost, error=error,
        )

        # Quality sampling
        if answer:
            score = self.quality_mon.maybe_score(user_query, answer)
            if score is not None:
                trace.quality_score = score

        self.tracer.log(trace)
        self.slo.track(latency_ms / 1000, cost, tok_in, tok_out, error=error,
                       quality=trace.quality_score)
        self.drift_detector.add(user_query)
        self._call_count += 1

        if self._call_count % 25 == 0:
            self._run_periodic_checks()

        return {
            "answer":     answer,
            "trace_id":   trace_id,
            "latency_ms": latency_ms,
            "cost_usd":   cost,
            "tokens_in":  tok_in,
            "tokens_out": tok_out,
        }

    def _run_periodic_checks(self):
        drift   = self.drift_detector.check()
        quality = self.quality_mon.check()
        slo     = self.slo.slo_status()
        fired   = self.alerts.check_and_alert(slo, drift, quality)

        print(f"\n  [Periodic check @ call #{self._call_count}]")
        print(f"    Embedding drift: {drift['reason']}")
        print(f"    Quality:         {quality.get('reason', 'N/A')}")
        if fired:
            for f in fired:
                print(f"    🔔 {f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8 — FastAPI
# ══════════════════════════════════════════════════════════════════════════════

app   = FastAPI(title="LLMOps Monitoring Hub")
AGENT = MonitoredAgent(system_prompt="You are a helpful assistant. Answer concisely.")


class QueryRequest(BaseModel):
    query:   str
    user_id: str = "anonymous"


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    return AGENT.query(req.query, req.user_id)


@app.get("/metrics")
async def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dashboard")
async def dashboard():
    return {
        "stats":   AGENT.tracer.get_stats(24),
        "slo":     AGENT.slo.slo_status(),
        "drift":   AGENT.drift_detector.check(),
        "quality": AGENT.quality_mon.check(),
    }


@app.get("/drift")
async def drift_status():
    return {
        "embedding": AGENT.drift_detector.check(),
        "quality":   AGENT.quality_mon.check(),
    }


@app.get("/alerts")
async def alert_history():
    return {"alerts": AGENT.alerts.get_alert_history(50)}


@app.get("/traces")
async def traces(n: int = 100):
    return {"traces": AGENT.tracer.get_recent(n)}


# ══════════════════════════════════════════════════════════════════════════════
# Simulation runner
# ══════════════════════════════════════════════════════════════════════════════

def simulate_traffic(agent: MonitoredAgent, n_reference: int = 50, n_drifted: int = 150):
    print("=== LLMOps Monitoring Hub — Traffic Simulation ===\n")

    print(f"Phase 1: {n_reference} reference queries (compliance)…")
    for i, query in enumerate(REFERENCE_QUERIES[:n_reference]):
        result = agent.query(query, user_id=f"user-{i % 10}")
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_reference}] latency={result['latency_ms']:.0f}ms cost=${result['cost_usd']:.5f}")

    print(f"\nPhase 2: {n_drifted} drifted queries (off-topic)…")
    for i, query in enumerate(DRIFTED_QUERIES[:n_drifted]):
        result = agent.query(query, user_id=f"user-{i % 10}")
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n_drifted}] latency={result['latency_ms']:.0f}ms cost=${result['cost_usd']:.5f}")

    print("\n=== Final Summary ===")
    stats   = agent.tracer.get_stats(24)
    drift   = agent.drift_detector.check()
    quality = agent.quality_mon.check()
    alerts  = agent.alerts.get_alert_history()

    print(f"  Total calls:       {stats['call_count']}")
    print(f"  Total cost:        ${stats['total_cost_usd']:.4f}")
    print(f"  Avg latency:       {stats['avg_latency_ms']:.0f}ms")
    print(f"  Embedding drift:   {'YES ⚠️' if drift['drift'] else 'No ✅'} ({drift.get('reason','')})")
    print(f"  Quality status:    {quality.get('reason', 'N/A')}")
    print(f"  Alerts fired:      {len(alerts)}")
    for a in alerts[:5]:
        print(f"    [{a['severity']}] {a['message']}")


if __name__ == "__main__":
    import sys
    if "--server" in sys.argv:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        agent = MonitoredAgent(
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant. Answer concisely.",
            prompt_name="demo-agent",
            prompt_version="v1",
        )
        simulate_traffic(agent, n_reference=50, n_drifted=150)
