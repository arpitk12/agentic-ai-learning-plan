"""
Project 42 — Prompt Version Control System (SOLUTION)
======================================================
Full implementation: PromptVersion, PromptRegistry, ABRouter, CanaryRouter,
MetricsCollector, MonitoredRouter, FastAPI.

Run:  python solution.py           (demo)
      python solution.py --server  (API at :8001)
"""
from __future__ import annotations

import difflib
import hashlib
import json
import random
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Callable

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, model_validator

load_dotenv()

# ─────────────────────────────────────────────────
# Sample prompts
# ─────────────────────────────────────────────────
PROMPT_V1 = """You are a compliance assistant. Answer questions about GDPR."""

PROMPT_V2 = """You are a senior compliance attorney with 20 years of experience.
Always cite the specific article number when answering.
Keep answers under 150 words unless complexity requires more.
If uncertain, say so explicitly rather than guessing."""

PROMPT_V3 = """You are a senior compliance attorney specializing in EU data protection law.
Always cite the specific GDPR article number when answering.
Structure your response: [Article Reference] [Plain explanation] [Practical implication].
Keep answers under 200 words. If uncertain, say "I recommend consulting a qualified attorney for..."
Always consider cross-border implications for multinational companies."""

TEST_QUERIES = [
    "What does Article 28 of GDPR require?",
    "When is a Data Protection Impact Assessment required?",
    "What is the right to erasure?",
    "How long can personal data be retained?",
    "What constitutes legitimate interest under GDPR?",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1 — PromptVersion
# ══════════════════════════════════════════════════════════════════════════════

class PromptVersion(BaseModel):
    name:          str
    version:       str
    stage:         str  = "development"
    text:          str
    model:         str  = "gpt-4o-mini"
    temperature:   float = 0.0
    max_tokens:    int   = 512
    created_at:    str  = ""
    author:        str  = ""
    commit_sha:    str  = ""
    hash:          str  = ""
    changelog:     str  = ""
    metrics:       dict = {
        "accuracy": 0.0, "quality_avg": 0.0,
        "latency_p95_ms": 0.0, "cost_per_call_usd": 0.0, "total_calls": 0
    }

    @model_validator(mode="after")
    def _auto_fields(self) -> "PromptVersion":
        if not self.hash:
            self.hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 2 — PromptRegistry
# ══════════════════════════════════════════════════════════════════════════════

def _row_to_pv(row: tuple) -> PromptVersion:
    cols = ["id","name","version","stage","text","model","temperature","max_tokens",
            "created_at","author","commit_sha","hash","changelog","metrics"]
    d = dict(zip(cols, row))
    d.pop("id", None)
    d["metrics"] = json.loads(d["metrics"] or "{}")
    return PromptVersion(**d)


class PromptRegistry:
    def __init__(self, db_path: str = "prompt_registry.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                version     TEXT NOT NULL,
                stage       TEXT NOT NULL DEFAULT 'development',
                text        TEXT NOT NULL,
                model       TEXT NOT NULL,
                temperature REAL NOT NULL,
                max_tokens  INTEGER NOT NULL,
                created_at  TEXT,
                author      TEXT,
                commit_sha  TEXT,
                hash        TEXT,
                changelog   TEXT,
                metrics     TEXT DEFAULT '{}',
                UNIQUE(name, version)
            )
        """)
        self.db.commit()

    def create(
        self,
        name: str,
        text: str,
        model:        str   = "gpt-4o-mini",
        temperature:  float = 0.0,
        max_tokens:   int   = 512,
        author:       str   = "",
        commit_sha:   str   = "",
        changelog:    str   = "",
    ) -> str:
        count   = self.db.execute("SELECT COUNT(*) FROM prompts WHERE name=?", (name,)).fetchone()[0]
        version = f"v{count + 1}"
        h       = hashlib.sha256(text.encode()).hexdigest()[:16]
        ts      = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT INTO prompts (name,version,stage,text,model,temperature,max_tokens,"
            "created_at,author,commit_sha,hash,changelog,metrics) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, version, "development", text, model, temperature, max_tokens,
             ts, author, commit_sha, h, changelog,
             json.dumps({"accuracy": 0.0, "quality_avg": 0.0, "latency_p95_ms": 0.0,
                         "cost_per_call_usd": 0.0, "total_calls": 0}))
        )
        self.db.commit()
        return version

    def get(self, name: str, version: str = "latest") -> PromptVersion:
        if version == "latest":
            row = self.db.execute(
                "SELECT * FROM prompts WHERE name=? ORDER BY id DESC LIMIT 1", (name,)
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT * FROM prompts WHERE name=? AND version=?", (name, version)
            ).fetchone()
        if not row:
            raise KeyError(f"Prompt {name!r} version {version!r} not found")
        return _row_to_pv(row)

    def get_by_stage(self, name: str, stage: str) -> PromptVersion:
        row = self.db.execute(
            "SELECT * FROM prompts WHERE name=? AND stage=? ORDER BY id DESC LIMIT 1",
            (name, stage)
        ).fetchone()
        if not row:
            raise KeyError(f"Prompt {name!r} has no version in stage {stage!r}")
        return _row_to_pv(row)

    def list_versions(self, name: str) -> list[PromptVersion]:
        rows = self.db.execute(
            "SELECT * FROM prompts WHERE name=? ORDER BY id DESC", (name,)
        ).fetchall()
        return [_row_to_pv(r) for r in rows]

    def promote(self, name: str, version: str, to_stage: str) -> None:
        # Archive current holder
        self.db.execute(
            "UPDATE prompts SET stage='archived' WHERE name=? AND stage=?", (name, to_stage)
        )
        self.db.execute(
            "UPDATE prompts SET stage=? WHERE name=? AND version=?", (to_stage, name, version)
        )
        self.db.commit()

    def rollback(self, name: str) -> str:
        try:
            prod = self.get_by_stage(name, "production")
        except KeyError:
            raise RuntimeError(f"No production version for {name!r}")
        # Find the most recent archived version before current production id
        prod_id = self.db.execute(
            "SELECT id FROM prompts WHERE name=? AND version=?", (name, prod.version)
        ).fetchone()[0]
        row = self.db.execute(
            "SELECT * FROM prompts WHERE name=? AND stage='archived' AND id < ? ORDER BY id DESC LIMIT 1",
            (name, prod_id)
        ).fetchone()
        if not row:
            raise RuntimeError(f"No archived version available for rollback for {name!r}")
        prev = _row_to_pv(row)
        self.promote(name, prev.version, "production")
        return prev.version

    def update_metrics(self, name: str, version: str, metrics: dict) -> None:
        row = self.db.execute(
            "SELECT metrics FROM prompts WHERE name=? AND version=?", (name, version)
        ).fetchone()
        if not row:
            return
        existing = json.loads(row[0] or "{}")
        existing.update(metrics)
        self.db.execute(
            "UPDATE prompts SET metrics=? WHERE name=? AND version=?",
            (json.dumps(existing), name, version)
        )
        self.db.commit()

    def diff(self, name: str, v1: str, v2: str) -> str:
        pv1 = self.get(name, v1)
        pv2 = self.get(name, v2)
        lines = list(difflib.unified_diff(
            pv1.text.splitlines(keepends=True),
            pv2.text.splitlines(keepends=True),
            fromfile=f"{name} {v1}",
            tofile=f"{name} {v2}",
        ))
        return "".join(lines) if lines else "(no changes)"


# ══════════════════════════════════════════════════════════════════════════════
# 3 — ABRouter
# ══════════════════════════════════════════════════════════════════════════════

class ABRouter:
    def __init__(self, registry: PromptRegistry):
        self.registry    = registry
        self.call_counts: dict[str, int] = {}

    def route(self, user_id: str, name: str, v_a: str, v_b: str, split: float = 0.5) -> PromptVersion:
        bucket  = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        version = v_a if bucket < int(split * 100) else v_b
        self.call_counts[version] = self.call_counts.get(version, 0) + 1
        return self.registry.get(name, version)

    def stats(self) -> dict[str, int]:
        return dict(self.call_counts)


# ══════════════════════════════════════════════════════════════════════════════
# 4 — CanaryRouter
# ══════════════════════════════════════════════════════════════════════════════

class CanaryRouter:
    STAGES                = [0.10, 0.50, 1.00]
    MIN_CALLS_PER_STAGE   = 50

    def __init__(self, registry: PromptRegistry):
        self.registry = registry
        # {name → {canary_version, current_pct, stage_idx, call_count, metrics}}
        self._active: dict[str, dict] = {}

    def start_canary(self, name: str, canary_version: str) -> None:
        self._active[name] = {
            "canary_version": canary_version,
            "current_pct":    self.STAGES[0],
            "stage_idx":      0,
            "call_count":     0,
            "metrics":        {"qualities": [], "latencies": []},
        }

    def route(self, user_id: str, name: str) -> PromptVersion:
        state = self._active.get(name)
        if not state:
            return self.registry.get_by_stage(name, "production")
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        if bucket < int(state["current_pct"] * 100):
            return self.registry.get(name, state["canary_version"])
        return self.registry.get_by_stage(name, "production")

    def record_call(self, name: str, version: str, quality: float, latency_ms: float) -> None:
        state = self._active.get(name)
        if not state or version != state["canary_version"]:
            return
        state["call_count"] += 1
        state["metrics"]["qualities"].append(quality)
        state["metrics"]["latencies"].append(latency_ms)
        if state["call_count"] < self.MIN_CALLS_PER_STAGE:
            return
        avg_quality  = sum(state["metrics"]["qualities"]) / len(state["metrics"]["qualities"])
        avg_latency  = sum(state["metrics"]["latencies"]) / len(state["metrics"]["latencies"])
        metrics_ok   = avg_quality >= 7.5 and avg_latency <= 5000
        if not metrics_ok:
            print(f"  ⚠️  Canary {version} failed metrics (q={avg_quality:.2f}, l={avg_latency:.0f}ms). Rolling back.")
            try:
                self.registry.rollback(name)
            except Exception:
                pass
            del self._active[name]
            return
        state["stage_idx"] += 1
        if state["stage_idx"] >= len(self.STAGES):
            # Full promotion
            self.registry.promote(name, version, "production")
            del self._active[name]
            print(f"  ✅ Canary {version} fully promoted to production.")
            return
        next_pct = self.STAGES[state["stage_idx"]]
        state["current_pct"] = next_pct
        state["call_count"]  = 0
        state["metrics"]     = {"qualities": [], "latencies": []}
        print(f"  🎯 Canary {version} advanced to {int(next_pct*100)}%.")

    def status(self, name: str) -> dict:
        state = self._active.get(name)
        if not state:
            return {"active": False}
        return {"active": True, **state}


# ══════════════════════════════════════════════════════════════════════════════
# 5 — MetricsCollector
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    def __init__(self, db_path: str = "prompt_metrics.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT,
                version       TEXT,
                timestamp     TEXT,
                latency_ms    REAL,
                cost_usd      REAL,
                quality_score REAL,
                error         INTEGER DEFAULT 0
            )
        """)
        self.db.commit()

    def record(
        self,
        name: str,
        version: str,
        latency_ms:    float,
        cost_usd:      float,
        quality_score: float | None = None,
        error:         bool         = False,
    ) -> None:
        self.db.execute(
            "INSERT INTO metrics (name,version,timestamp,latency_ms,cost_usd,quality_score,error) "
            "VALUES (?,?,?,?,?,?,?)",
            (name, version, datetime.now(timezone.utc).isoformat(),
             latency_ms, cost_usd, quality_score, int(error)),
        )
        self.db.commit()

    def get_report(self, name: str, hours: int = 24) -> dict[str, dict]:
        rows = self.db.execute(
            "SELECT version, latency_ms, cost_usd, quality_score, error FROM metrics WHERE name=?",
            (name,)
        ).fetchall()
        per_version: dict[str, list] = {}
        for r in rows:
            per_version.setdefault(r[0], []).append(r[1:])
        report = {}
        for ver, data in per_version.items():
            lats  = sorted(d[0] for d in data)
            costs = [d[1] for d in data]
            quals = [d[2] for d in data if d[2] is not None]
            errs  = [d[3] for d in data]
            n     = len(data)
            p95   = lats[int(n * 0.95)] if lats else 0
            report[ver] = {
                "count":       n,
                "avg_latency": sum(lats) / n if lats else 0,
                "p95_latency": p95,
                "avg_cost":    sum(costs) / n if costs else 0,
                "avg_quality": sum(quals) / len(quals) if quals else None,
                "error_rate":  sum(errs) / n if n else 0,
            }
        return report

    def compare(self, name: str, v1: str, v2: str) -> dict:
        report = self.get_report(name)
        m1 = report.get(v1, {})
        m2 = report.get(v2, {})
        winner: dict[str, str] = {}
        for metric, lower_better in [
            ("avg_latency", True), ("error_rate", True),
            ("avg_cost", True), ("avg_quality", False),
        ]:
            val1 = m1.get(metric) or 0
            val2 = m2.get(metric) or 0
            if lower_better:
                winner[metric] = v1 if val1 <= val2 else v2
            else:
                winner[metric] = v1 if val1 >= val2 else v2
        return {"v1": m1, "v2": m2, "winner": winner}


# ══════════════════════════════════════════════════════════════════════════════
# 6 — MonitoredRouter
# ══════════════════════════════════════════════════════════════════════════════

class MonitoredRouter:
    def __init__(
        self,
        registry: PromptRegistry,
        metrics:  MetricsCollector,
        model:    str = "gpt-4o-mini",
    ):
        self.registry   = registry
        self.metrics    = metrics
        self.model      = model
        self.ab_router  = ABRouter(registry)
        self.canary     = CanaryRouter(registry)
        self._calls     = 0

    def query(
        self,
        user_query:  str,
        user_id:     str,
        prompt_name: str,
        agent_fn:    Callable[[str, str], str],
        ab_versions: tuple[str, str] | None = None,
    ) -> dict:
        # Route
        if ab_versions:
            pv = self.ab_router.route(user_id, prompt_name, ab_versions[0], ab_versions[1])
        else:
            try:
                pv = self.canary.route(user_id, prompt_name)
            except KeyError:
                pv = self.registry.get(prompt_name)

        start     = time.time()
        answer    = agent_fn(pv.text, user_query)
        latency   = (time.time() - start) * 1000
        cost      = 0.003  # mock cost
        quality   = None

        self.metrics.record(prompt_name, pv.version, latency, cost, quality_score=quality)
        self._calls += 1

        if self._calls % 100 == 0:
            report = self.metrics.get_report(prompt_name)
            for ver, m in report.items():
                self.registry.update_metrics(prompt_name, ver, {
                    "quality_avg":      m.get("avg_quality") or 0.0,
                    "latency_p95_ms":   m.get("p95_latency") or 0.0,
                    "cost_per_call_usd": m.get("avg_cost") or 0.0,
                    "total_calls":      m.get("count") or 0,
                })

        return {
            "answer":         answer,
            "prompt_version": pv.version,
            "latency_ms":     latency,
            "cost_usd":       cost,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7 — FastAPI Application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Prompt Registry API", version="1.0.0")

_registry = PromptRegistry(db_path=":memory:")
_metrics  = MetricsCollector(db_path=":memory:")
_router   = MonitoredRouter(_registry, _metrics)


class CreatePromptRequest(BaseModel):
    text:        str
    model:       str   = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens:  int   = 512
    author:      str   = ""
    changelog:   str   = ""

class PromoteRequest(BaseModel):
    version:  str
    to_stage: str

class CanaryRequest(BaseModel):
    version: str


@app.get("/prompts/{name}")
async def get_prompt(name: str, stage: str = "production"):
    try:
        return _registry.get_by_stage(name, stage).model_dump()
    except KeyError:
        try:
            return _registry.get(name).model_dump()
        except KeyError:
            raise HTTPException(404, f"Prompt {name!r} not found")


@app.get("/prompts/{name}/versions")
async def list_versions(name: str):
    return [pv.model_dump() for pv in _registry.list_versions(name)]


@app.post("/prompts/{name}", status_code=201)
async def create_prompt(name: str, req: CreatePromptRequest):
    version = _registry.create(
        name=name, text=req.text, model=req.model, temperature=req.temperature,
        max_tokens=req.max_tokens, author=req.author, changelog=req.changelog,
    )
    return {"name": name, "version": version, "status": "created"}


@app.post("/prompts/{name}/promote")
async def promote(name: str, req: PromoteRequest):
    try:
        _registry.promote(name, req.version, req.to_stage)
        return {"name": name, "version": req.version, "stage": req.to_stage}
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/prompts/{name}/rollback")
async def rollback(name: str):
    try:
        prev = _registry.rollback(name)
        return {"name": name, "rolled_back_to": prev}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/prompts/{name}/diff")
async def diff(name: str, v1: str = Query(...), v2: str = Query(...)):
    try:
        return {"diff": _registry.diff(name, v1, v2)}
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/prompts/{name}/compare")
async def compare(name: str, v1: str = Query(...), v2: str = Query(...)):
    return _metrics.compare(name, v1, v2)


@app.post("/prompts/{name}/canary")
async def start_canary(name: str, req: CanaryRequest):
    _router.canary.start_canary(name, req.version)
    return {"name": name, "canary_version": req.version, "pct": "10%", "status": "started"}


@app.get("/prompts/{name}/metrics")
async def metrics_report(name: str, hours: int = 24):
    return _metrics.get_report(name, hours)


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Demo runner
# ══════════════════════════════════════════════════════════════════════════════

def simple_agent(system_prompt: str, user_query: str) -> str:
    try:
        resp = litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_query},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[Error: {e}]"


def run_demo():
    print("=== Prompt Registry Demo ===\n")

    registry = PromptRegistry(db_path=":memory:")
    metrics  = MetricsCollector(db_path=":memory:")
    router   = MonitoredRouter(registry, metrics)

    # 8a — Create versions
    v1 = registry.create("compliance", PROMPT_V1, author="demo", changelog="Initial version")
    v2 = registry.create("compliance", PROMPT_V2, author="demo", changelog="Add article citation + word limit")
    v3 = registry.create("compliance", PROMPT_V3, author="demo", changelog="Add cross-border guidance + structure")
    print(f"Created: {v1}, {v2}, {v3}")

    # 8b — Promote v2 to staging → production
    registry.promote("compliance", v2, "staging")
    print(f"Promoted {v2} → staging")
    registry.promote("compliance", v2, "production")
    print(f"Promoted {v2} → production")

    # 8c — Diff v1 → v2
    print("\n--- Diff v1 → v2 ---")
    print(registry.diff("compliance", "v1", "v2"))

    # 8d — A/B test: 50 queries between v1 and v2
    print("\n--- A/B Test (50 queries) ---")
    ab_router = ABRouter(registry)
    for i, q in enumerate(TEST_QUERIES * 10):
        user_id = f"user-{i}"
        pv      = ab_router.route(user_id, "compliance", "v1", "v2")
        answer  = simple_agent(pv.text, q)
        latency = random.uniform(200, 800)
        quality = random.uniform(6.5, 9.5)
        metrics.record("compliance", pv.version, latency, 0.003, quality)
    stats = ab_router.stats()
    print(f"  A/B call distribution: {stats}")
    report = metrics.get_report("compliance")
    for ver, m in report.items():
        print(f"  {ver}: calls={m['count']}, avg_q={m['avg_quality']:.2f}, avg_l={m['avg_latency']:.0f}ms")

    # 8e — Canary v3 at 10%
    print("\n--- Canary: v3 at 10% ---")
    canary = CanaryRouter(registry)
    canary.start_canary("compliance", "v3")
    for i in range(200):
        user_id = f"user-canary-{i}"
        pv      = canary.route(user_id, "compliance")
        q       = random.choice(TEST_QUERIES)
        latency = random.uniform(200, 600)
        quality = random.uniform(7.8, 9.5)
        canary.record_call("compliance", pv.version, quality, latency)

    # 8f — Rollback
    print("\n--- Rollback ---")
    try:
        prev = registry.rollback("compliance")
        print(f"  Rolled back to {prev}")
    except RuntimeError as e:
        print(f"  Rollback: {e}")

    # 8g — Final version list
    print("\n=== Final version list ===")
    for pv in registry.list_versions("compliance"):
        print(f"  {pv.version}: stage={pv.stage} hash={pv.hash}")


if __name__ == "__main__":
    import sys
    if "--server" in sys.argv:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
    else:
        run_demo()
