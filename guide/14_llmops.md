# Guide 14 — LLMOps: Deploying, Monitoring & Iterating LLM Agents in Production

> **Read time**: ~2 hours | **Level**: Intermediate–Advanced
> **Prerequisites**: [§9 Observability](09_observability.md) · [§10 Deployment](10_deployment.md) · [§12 Evaluation](12_evaluation.md)

**Projects that apply this guide**:
- [project41 — LLMOps Monitoring Hub](../../projects/project41_llmops_monitoring/)
- [project42 — Prompt Registry](../../projects/project42_prompt_registry/)
- [project43 — Continuous Eval Scheduler](../../projects/project43_continuous_eval/)

---

## Table of Contents

| § | Topic | Read time |
|---|---|---|
| 1 | LLMOps vs MLOps vs AIOps | 10 min |
| 2 | The LLMOps Lifecycle | 10 min |
| 3 | Prompt Registry & Versioning | 20 min |
| 4 | Experiment Tracking | 15 min |
| 5 | Continuous Evaluation in Production | 20 min |
| 6 | Drift Detection | 20 min |
| 7 | Model Registry & Promotion Workflow | 15 min |
| 8 | SLOs for LLM Systems | 15 min |
| 9 | LLMOps Platforms Compared | 10 min |
| 10 | Production Monitoring Stack | 15 min |
| 11 | CI/CD for LLM Agents | 15 min |
| 12 | Cost Governance | 10 min |
| 13 | Feature Stores for LLM Context | 10 min |
| 14 | Reference Architecture | 10 min |
| 15 | Interview Framework | 10 min |
| 16 | Decision Cheat Sheet | 5 min |

---

## 1. LLMOps vs MLOps vs AIOps

### 1.1 Definitions

| Term | What it covers |
|---|---|
| **MLOps** | Training pipelines · feature stores · model versioning · serving · data drift |
| **LLMOps** | Prompt versioning · LLM eval · token cost · context management · hallucination monitoring |
| **AIOps** | Applying AI *to* IT operations — log analysis, anomaly detection, auto-remediation |

LLMOps is a **specialisation of MLOps** for foundation-model applications. You rarely train from scratch, so the "train → serve" loop is replaced by a **"prompt → eval → deploy → monitor → iterate"** loop.

### 1.2 What's Different for LLMs

| MLOps concern | LLM equivalent | Why it's harder |
|---|---|---|
| Model weights | Prompt text | Prompts are informal — no gradient, no loss function |
| Feature engineering | Context assembly | Context window is finite; ordering matters |
| Model accuracy | LLM-judge score / task completion | Ground truth is expensive/ambiguous |
| Model drift | Prompt sensitivity drift | API model updates silently change behaviour |
| Serving latency | TTFT + total generation time | Varies with output length — unpredictable |
| Inference cost | Token cost (input + output) | Cost is a first-class operational metric |
| A/B testing | Prompt A/B + model A/B | Statistical significance harder with human-eval tasks |

### 1.3 The New Failure Modes

```
Prompt injection          → Input validation + guardrails
Hallucination             → RAG + faithfulness eval + citation enforcement
Context poisoning         → Sanitise retrieved chunks; length limits
Silent API model updates  → Pin model versions; schedule regression evals
Runaway costs             → Per-call budget caps; alert on daily spend
Jailbreak / misuse        → LlamaGuard + NeMo + output filtering
```

---

## 2. The LLMOps Lifecycle

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                      LLMOps Lifecycle                               │
 │                                                                     │
 │  1. Prompt          2. Experiment       3. Eval Gate                │
 │  Development        Tracking            (CI/CD)                     │
 │  ──────────         ──────────          ──────────                  │
 │  Write prompt  ──►  Log to MLflow  ──►  Run golden  ──►  Pass?     │
 │  Iterate v1→v2      (params +           dataset eval       │        │
 │  Store in registry  metrics +           LLM-judge score    │        │
 │                     artifacts)          Exit 0 or 1        │        │
 │                                                            ▼        │
 │  7. Iterate         6. Drift            5. Monitor         4. Deploy│
 │  ──────────         Detection           ───────────        ─────────│
 │  Open PR with  ◄──  Alert fires   ◄──  Prometheus +  ◄──  Canary   │
 │  new prompt v       Regression          Grafana           10%→100%  │
 │  trigger eval       detected            Langfuse                    │
 │  gate again         Rollback?           traces                      │
 └─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Key Insight: Prompts are the New Model Weights

In classic ML, you retrain a model. In LLM apps, you **retune a prompt**. This means:

- Every prompt change is a **deployment** — needs a gate
- You need **version history** so you can roll back
- You need **metric tracking** per version so you know if v2 is better than v1
- Silent model API updates (OpenAI, Anthropic) can **break your prompts without your knowledge**

---

## 3. Prompt Registry & Versioning

### 3.1 Why You Need It

Without a prompt registry:
```
# The chaos of unversioned prompts
system_prompt_v3_FINAL.txt
system_prompt_v3_FINAL_REAL.txt
system_prompt_v3_FINAL_REAL_use_this_one.txt
```

With a prompt registry:
```python
prompt = registry.get("compliance-classifier", version="v4", stage="production")
# → always deterministic; rollback in one call
```

### 3.2 What to Track Per Prompt Version

```python
@dataclass
class PromptVersion:
    name: str           # "compliance-classifier"
    version: str        # "v4"
    stage: str          # "development" | "staging" | "production" | "archived"
    text: str           # full prompt text
    model: str          # "gpt-4o-mini" — the model it was tuned for
    temperature: float  # 0.0–2.0
    created_at: str     # ISO timestamp
    author: str         # who created it
    commit_sha: str     # git commit that triggered this version
    metrics: dict       # {"accuracy": 0.91, "latency_p95_ms": 1800, "cost_per_call": 0.004}
    hash: str           # SHA-256 of prompt text — detect accidental changes
    changelog: str      # human-readable description of what changed
```

### 3.3 SQLite-Backed Prompt Registry (Production Pattern)

```python
import sqlite3, hashlib, json
from datetime import datetime, timezone

class PromptRegistry:
    def __init__(self, db_path: str = "prompt_registry.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                version     TEXT NOT NULL,
                stage       TEXT NOT NULL DEFAULT 'development',
                text        TEXT NOT NULL,
                model       TEXT NOT NULL,
                temperature REAL NOT NULL DEFAULT 0.0,
                created_at  TEXT NOT NULL,
                author      TEXT,
                commit_sha  TEXT,
                metrics     TEXT DEFAULT '{}',
                hash        TEXT NOT NULL,
                changelog   TEXT,
                UNIQUE(name, version)
            )
        """)
        self.db.commit()

    def create(self, name: str, text: str, model: str, temperature: float = 0.0,
               author: str = "", commit_sha: str = "", changelog: str = "") -> str:
        """Create a new version (auto-numbered)."""
        # Get next version number
        row = self.db.execute(
            "SELECT COUNT(*) FROM prompts WHERE name = ?", (name,)
        ).fetchone()
        version = f"v{row[0] + 1}"
        
        prompt_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        self.db.execute("""
            INSERT INTO prompts (name, version, text, model, temperature,
                                 created_at, author, commit_sha, hash, changelog)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, version, text, model, temperature,
              datetime.now(timezone.utc).isoformat(),
              author, commit_sha, prompt_hash, changelog))
        self.db.commit()
        return version

    def get(self, name: str, version: str = "latest", stage: str = None) -> dict:
        """Retrieve a prompt. version='latest' returns highest version number."""
        if stage:
            row = self.db.execute(
                "SELECT * FROM prompts WHERE name=? AND stage=? ORDER BY id DESC LIMIT 1",
                (name, stage)
            ).fetchone()
        elif version == "latest":
            row = self.db.execute(
                "SELECT * FROM prompts WHERE name=? ORDER BY id DESC LIMIT 1", (name,)
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT * FROM prompts WHERE name=? AND version=?", (name, version)
            ).fetchone()
        
        if not row:
            raise KeyError(f"Prompt '{name}' version '{version}' not found")
        
        cols = ["id","name","version","stage","text","model","temperature",
                "created_at","author","commit_sha","metrics","hash","changelog"]
        result = dict(zip(cols, row))
        result["metrics"] = json.loads(result["metrics"])
        return result

    def promote(self, name: str, version: str, to_stage: str) -> None:
        """Promote a version to a new stage. Demotes previous holder."""
        # Demote current holder of that stage
        self.db.execute(
            "UPDATE prompts SET stage='archived' WHERE name=? AND stage=?",
            (name, to_stage)
        )
        self.db.execute(
            "UPDATE prompts SET stage=? WHERE name=? AND version=?",
            (to_stage, name, version)
        )
        self.db.commit()

    def rollback(self, name: str) -> str:
        """Roll back production to previous production version."""
        prod = self.get(name, stage="production")
        # Find the version before current production
        prev = self.db.execute("""
            SELECT version FROM prompts
            WHERE name=? AND id < (SELECT id FROM prompts WHERE name=? AND version=?)
            ORDER BY id DESC LIMIT 1
        """, (name, name, prod["version"])).fetchone()
        
        if not prev:
            raise ValueError("No previous version to roll back to")
        
        self.promote(name, prev[0], "production")
        return prev[0]

    def update_metrics(self, name: str, version: str, metrics: dict) -> None:
        self.db.execute(
            "UPDATE prompts SET metrics=? WHERE name=? AND version=?",
            (json.dumps(metrics), name, version)
        )
        self.db.commit()

    def diff(self, name: str, v1: str, v2: str) -> str:
        """Show line-level diff between two versions."""
        import difflib
        p1 = self.get(name, v1)["text"]
        p2 = self.get(name, v2)["text"]
        return "\n".join(difflib.unified_diff(
            p1.splitlines(), p2.splitlines(),
            fromfile=f"{name}@{v1}", tofile=f"{name}@{v2}", lineterm=""
        ))
```

### 3.4 Options for Prompt Storage

| Option | Pros | Cons | Best for |
|---|---|---|---|
| **Git files** | Free, diffs built-in, PR reviews | No metric tracking, no API | Small teams, simple workflows |
| **SQLite registry** (above) | Full control, offline, fast | DIY, no UI | Single-server deployments |
| **LangSmith Prompt Hub** | UI, versioning, LangChain native | Vendor lock-in, cost | LangChain shops |
| **PromptLayer** | Proxy-based, zero-code | Limited rollback, cost | Quick wins |
| **MLflow** | Metrics + artifacts + stages | Heavy, not prompt-native | Teams already using MLflow |

---

## 4. Experiment Tracking

### 4.1 What to Log for Every Experiment

```python
import mlflow

with mlflow.start_run(run_name="compliance-v4-gpt4o-mini"):
    # Parameters — inputs you controlled
    mlflow.log_params({
        "prompt_name": "compliance-classifier",
        "prompt_version": "v4",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 512,
        "dataset": "compliance_golden_v2.jsonl",
        "dataset_size": 100,
    })
    
    # Metrics — what you measured
    mlflow.log_metrics({
        "accuracy": 0.91,
        "precision": 0.89,
        "recall": 0.93,
        "f1": 0.91,
        "faithfulness": 0.88,
        "latency_p50_ms": 820,
        "latency_p95_ms": 1800,
        "cost_per_call_usd": 0.0042,
        "tokens_in_avg": 312,
        "tokens_out_avg": 48,
    })
    
    # Artifacts — files that are part of the experiment
    mlflow.log_artifact("eval_report.json")
    mlflow.log_artifact("prompt_v4.txt")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Tags — searchable labels
    mlflow.set_tags({
        "author": "alice",
        "experiment": "latency-reduction",
        "git_sha": "a3b4c5d",
        "passed_eval_gate": "true",
    })
```

### 4.2 MLflow vs W&B vs LangSmith

| Feature | MLflow | W&B | LangSmith |
|---|---|---|---|
| Self-hosted | ✅ Free | ✅ (paid) | ❌ (SaaS only) |
| Prompt tracking | ✅ (as artifact) | ✅ | ✅ Native |
| LLM trace capture | Limited | ✅ Weave | ✅ Native |
| Dataset management | Basic | ✅ Artifacts | ✅ Native |
| Model registry | ✅ Mature | ✅ | Limited |
| Evaluation UI | ✅ mlflow.evaluate() | ✅ | ✅ Best-in-class |
| Cost | Free OSS | Free tier → paid | Free tier → paid |
| Best for | Enterprise, model registry | Research, rich viz | LangChain + evals |

### 4.3 Minimum Viable Experiment Discipline

Even without a tracking server, capture this in every eval run:

```python
# eval_run.py — minimum viable experiment log
import json, hashlib
from datetime import datetime, timezone

def log_run(config: dict, results: dict, output_file: str = "runs.jsonl"):
    record = {
        "run_id": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "results": results,
    }
    with open(output_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[run {record['run_id']}] accuracy={results.get('accuracy', '?'):.2%}")
```

---

## 5. Continuous Evaluation in Production

### 5.1 Why CI Evals Are Not Enough

| CI eval gate | Production continuous eval |
|---|---|
| Runs on every PR | Runs on a schedule (hourly/daily) |
| Tests code changes | Tests production behaviour |
| Fixed dataset | Growing golden dataset |
| Developer-triggered | Always-on |
| Detects regressions from code | Detects regressions from model API changes |

A CI gate catches your bugs. A continuous eval catches **OpenAI silently updating their model**.

### 5.2 The Continuous Eval Architecture

```
┌────────────────────────────────────────────────────────┐
│               Continuous Eval Scheduler                │
│                                                        │
│  APScheduler (cron: every 6h)                          │
│       │                                                │
│       ▼                                                │
│  Golden Dataset  ──►  Agent Under Test  ──►  LLM Judge │
│  (100 fixed QA              ▲                   │      │
│   pairs)              Production             scores    │
│                        endpoint                  │     │
│                                                  ▼     │
│                                          RegressionDetector│
│                                              │         │
│                     ┌────────────────────────┤         │
│                     ▼                        ▼         │
│               No regression          Score < threshold │
│               Log to DB              AlertChannel      │
│                                      (Slack/email/PD)  │
└────────────────────────────────────────────────────────┘
```

### 5.3 Building the Golden Dataset

A golden dataset is a fixed set of test cases with **known expected behaviours** — not exact strings, but behavioural assertions:

```python
GOLDEN_DATASET = [
    {
        "id": "compliance-001",
        "input": "Does Article 28 of GDPR require a DPA?",
        "assertions": {
            "must_contain": ["data processing agreement", "processor"],
            "must_not_contain": ["I don't know", "I cannot"],
            "min_length": 50,
            "llm_judge": "Answer accurately describes the DPA requirement under Article 28",
        },
        "category": "regulatory",
        "priority": "high",
    },
    # ... 99 more test cases
]
```

**Rules for good golden datasets**:
1. Cover all **task categories** (query types your users actually send)
2. Include **edge cases** (ambiguous queries, borderline cases)
3. Include **adversarial inputs** (injection attempts, out-of-scope)
4. **Never modify** existing cases — only add new ones
5. Version the dataset itself (golden_v1, golden_v2...)

### 5.4 Detecting Regressions

```python
class RegressionDetector:
    def __init__(self, db_path: str, threshold: float = 0.05):
        """threshold: max allowed drop from baseline (e.g. 0.05 = 5%)"""
        self.db_path = db_path
        self.threshold = threshold

    def check(self, current_score: float, prompt_name: str) -> tuple[bool, str]:
        """Returns (is_regression, reason)."""
        baseline = self._get_baseline(prompt_name)
        if baseline is None:
            self._set_baseline(prompt_name, current_score)
            return False, "First run — baseline set"
        
        drop = baseline - current_score
        if drop > self.threshold:
            return True, (
                f"Regression: score dropped {drop:.1%} "
                f"({baseline:.2%} → {current_score:.2%})"
            )
        
        # Also check 7-day trend (gradual degradation)
        week_scores = self._get_last_n_scores(prompt_name, n=7)
        if len(week_scores) >= 3:
            trend = week_scores[-1] - week_scores[0]
            if trend < -0.03:  # 3% drop over a week
                return True, f"Gradual degradation: {trend:.1%} over 7 runs"
        
        return False, "OK"
```

---

## 6. Drift Detection

### 6.1 Four Types of Drift in LLM Systems

```
Type 1: Input Drift     — what users are asking has changed
Type 2: Embedding Drift — the semantic distribution of inputs has shifted
Type 3: Quality Drift   — response quality is degrading over time
Type 4: Latency/Cost    — P95 latency or cost/call is creeping up
```

All four are independent and need separate detection strategies.

### 6.2 Input Drift — Topic Distribution

```python
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TopicDriftDetector:
    def __init__(self, window_size: int = 1000):
        self.reference_distribution = None  # established in first N requests
        self.window_size = window_size
        self.recent_queries: list[str] = []

    def record(self, query: str):
        self.recent_queries.append(query)
        if len(self.recent_queries) > self.window_size:
            self.recent_queries.pop(0)

    def set_reference(self, queries: list[str]):
        """Call once on initial production traffic sample."""
        vec = TfidfVectorizer(max_features=100, stop_words="english")
        self.vectorizer = vec
        self.reference_distribution = vec.fit_transform(queries).toarray().mean(axis=0)

    def check_drift(self) -> dict:
        if self.reference_distribution is None or len(self.recent_queries) < 100:
            return {"drift": False, "reason": "insufficient data"}
        
        current = self.vectorizer.transform(self.recent_queries).toarray().mean(axis=0)
        # Cosine similarity between reference and current topic distribution
        similarity = np.dot(self.reference_distribution, current) / (
            np.linalg.norm(self.reference_distribution) * np.linalg.norm(current) + 1e-9
        )
        
        drift = similarity < 0.7  # configurable threshold
        return {
            "drift": drift,
            "similarity": float(similarity),
            "reason": f"Topic similarity: {similarity:.2%}" if not drift else
                      f"Topic drift detected! Similarity dropped to {similarity:.2%}",
        }
```

### 6.3 Embedding Drift — Semantic Distribution

```python
import numpy as np
from scipy.stats import ks_2samp

class EmbeddingDriftDetector:
    """
    Kolmogorov-Smirnov test on the first principal component of embeddings.
    Flags when recent inputs are semantically different from the reference window.
    """
    def __init__(self, embed_fn, reference_size: int = 500, window_size: int = 200):
        self.embed_fn = embed_fn  # callable: str → list[float]
        self.reference_embeddings: list[list[float]] = []
        self.reference_size = reference_size
        self.window_size = window_size
        self.current_window: list[list[float]] = []
        self.reference_locked = False

    def add(self, text: str):
        embedding = self.embed_fn(text)
        if not self.reference_locked:
            self.reference_embeddings.append(embedding)
            if len(self.reference_embeddings) >= self.reference_size:
                self.reference_locked = True
        else:
            self.current_window.append(embedding)
            if len(self.current_window) > self.window_size:
                self.current_window.pop(0)

    def check(self) -> dict:
        if not self.reference_locked or len(self.current_window) < 50:
            return {"drift": False, "reason": "building reference"}
        
        # Use first principal component (most variance)
        ref = np.array(self.reference_embeddings)
        cur = np.array(self.current_window)
        
        # Project onto first component of reference
        ref_mean = ref.mean(axis=0)
        ref_centered = ref - ref_mean
        u, s, vt = np.linalg.svd(ref_centered, full_matrices=False)
        first_pc = vt[0]
        
        ref_proj = ref_centered @ first_pc
        cur_proj = (cur - ref_mean) @ first_pc
        
        stat, p_value = ks_2samp(ref_proj, cur_proj)
        drift = p_value < 0.05  # standard significance threshold
        
        return {
            "drift": drift,
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "reason": (
                f"Embedding drift detected (KS={stat:.3f}, p={p_value:.4f})"
                if drift else
                f"No embedding drift (KS={stat:.3f}, p={p_value:.4f})"
            ),
        }
```

### 6.4 Quality Drift — Monitoring LLM-Judge Scores

```python
class QualityDriftMonitor:
    """
    Samples N% of production traffic and scores it with an LLM judge.
    Detects when rolling average drops below threshold.
    """
    def __init__(self, judge_fn, sample_rate: float = 0.05, window: int = 100):
        self.judge_fn = judge_fn    # callable: (query, response) → float (0-10)
        self.sample_rate = sample_rate
        self.scores: list[float] = []
        self.window = window
        self.baseline: float | None = None

    def maybe_score(self, query: str, response: str) -> float | None:
        import random
        if random.random() > self.sample_rate:
            return None
        score = self.judge_fn(query, response)
        self.scores.append(score)
        if len(self.scores) > self.window:
            self.scores.pop(0)
        return score

    def check(self) -> dict:
        if len(self.scores) < 10:
            return {"drift": False, "reason": "collecting data"}
        
        current_avg = sum(self.scores) / len(self.scores)
        
        if self.baseline is None:
            self.baseline = current_avg
            return {"drift": False, "reason": f"Baseline set: {self.baseline:.2f}/10"}
        
        drop = self.baseline - current_avg
        drift = drop > 0.5  # configurable: 0.5 points on a 10-point scale
        return {
            "drift": drift,
            "current_avg": current_avg,
            "baseline": self.baseline,
            "drop": drop,
            "sample_count": len(self.scores),
            "reason": (
                f"Quality drift! Score dropped {drop:.2f} pts "
                f"({self.baseline:.2f} → {current_avg:.2f}/10)"
                if drift else
                f"Quality OK: {current_avg:.2f}/10 (baseline {self.baseline:.2f})"
            ),
        }
```

### 6.5 Drift Response Playbook

| Drift Type | Detected by | Auto-response | Manual response |
|---|---|---|---|
| Input topic drift | KS test on TF-IDF | Alert → human review | Add new golden dataset cases |
| Embedding drift | KS test on projections | Alert | Analyse new query cluster |
| Quality drop >5% | Rolling LLM-judge | Auto-rollback prompt | Investigate model API change |
| Latency P95 >threshold | Prometheus alert | Scale out workers | Profile slow tool calls |
| Cost/call up >20% | Daily spend monitor | Alert → review | Check if output length increased |

---

## 7. Model Registry & Promotion Workflow

### 7.1 The Promotion Pipeline

```
Development  ──►  Staging  ──►  Shadow  ──►  Canary (10%)  ──►  Production
     │                │              │              │                 │
  Local tests   Eval gate runs   Parallel run    Compare           Full traffic
  No gate       score > 0.85     compare vs      metrics           Rollback
                                 prod (no        if OK → 50%       in < 60s
                                 traffic yet)    → 100%
```

**Shadow mode** is the safest technique: the new prompt/model handles the request but its response is **discarded**. You only use it for metric comparison. Zero risk to users.

### 7.2 MLflow Promotion Workflow

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

def promote_to_production(run_id: str, model_name: str, min_accuracy: float = 0.90):
    """
    1. Register the run as a model version
    2. Transition it through Staging → Production
    3. Archive the old production version
    """
    # Step 1: Register
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version

    # Step 2: Promote to Staging
    client.transition_model_version_stage(model_name, version, "Staging")
    
    # Step 3: Run eval gate
    accuracy = run_eval_on_staging(model_name, version)
    
    if accuracy < min_accuracy:
        client.transition_model_version_stage(model_name, version, "Archived")
        raise ValueError(f"Eval gate failed: {accuracy:.2%} < {min_accuracy:.2%}")
    
    # Step 4: Archive current production
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    for v in current_prod:
        client.transition_model_version_stage(model_name, v.version, "Archived")
    
    # Step 5: Promote to Production
    client.transition_model_version_stage(model_name, version, "Production")
    print(f"✅ {model_name} v{version} promoted to Production (accuracy: {accuracy:.2%})")

def rollback(model_name: str) -> str:
    """Roll back to the most recent Archived version."""
    archived = client.get_latest_versions(model_name, stages=["Archived"])
    if not archived:
        raise ValueError("No archived version to roll back to")
    
    latest_archived = sorted(archived, key=lambda v: int(v.version), reverse=True)[0]
    
    # Archive current production
    prod = client.get_latest_versions(model_name, stages=["Production"])
    for v in prod:
        client.transition_model_version_stage(model_name, v.version, "Archived")
    
    # Promote previous
    client.transition_model_version_stage(
        model_name, latest_archived.version, "Production"
    )
    return latest_archived.version
```

### 7.3 Canary Deployment with Hash-Based Routing

```python
import hashlib

def canary_route(user_id: str, canary_version: str, canary_pct: float,
                 registry: PromptRegistry) -> dict:
    """Deterministic routing: same user always gets same version."""
    # Hash user_id to 0-100 bucket
    bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    
    if bucket < canary_pct * 100:
        return registry.get("agent-prompt", version=canary_version)
    else:
        return registry.get("agent-prompt", stage="production")
```

---

## 8. SLOs for LLM Systems

### 8.1 What to Define

An SLO (Service Level Objective) has three parts:
1. **Metric** — what you measure
2. **Target** — the threshold
3. **Window** — over how long

```
SLO: P95 latency < 5 000 ms measured over a rolling 24-hour window
SLO: Error rate < 0.5% measured over a rolling 7-day window
SLO: LLM-judge quality score ≥ 7.5/10 measured over a rolling 24-hour window
SLO: Cost per call < $0.015 measured over a rolling 7-day window
```

### 8.2 Recommended SLO Targets for Agent Systems

| Metric | Good | Acceptable | Alert threshold |
|---|---|---|---|
| P50 latency | < 800 ms | < 1.5 s | > 2 s |
| P95 latency | < 3 s | < 5 s | > 8 s |
| P99 latency | < 8 s | < 15 s | > 20 s |
| Error rate | < 0.1% | < 0.5% | > 1% |
| Tool failure rate | < 1% | < 3% | > 5% |
| Quality score | > 8.5/10 | > 7.5/10 | < 7.0/10 |
| Cost per call | < $0.005 | < $0.015 | > $0.03 |
| Token budget adherence | > 95% | > 90% | < 85% |

### 8.3 Error Budget and Burn Rate

```
If your SLO target is 99.5% availability over 30 days:
  Error budget = 0.5% × 30 days × 24 hrs × 60 min = 216 minutes/month

Burn rate: 
  1× = consuming budget at exactly the SLO rate (fine)
  2× = consuming at 2× rate → burn 216 min in 15 days (alert)
  14× = consuming at 14× rate → burn entire budget in 36 hours (page)
```

### 8.4 Implementing SLO Tracking

```python
from prometheus_client import Histogram, Counter, Gauge
import time

# Metrics
REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
REQUEST_ERRORS = Counter(
    "llm_request_errors_total",
    "Total LLM request errors",
    labelnames=["error_type"]
)
QUALITY_SCORE = Gauge(
    "llm_quality_score_current",
    "Current rolling average quality score"
)
COST_PER_CALL = Histogram(
    "llm_cost_per_call_usd",
    "Cost per LLM call in USD",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
)

class SLOTracker:
    def track(self, latency_s: float, cost_usd: float,
              error: str | None = None, quality: float | None = None):
        REQUEST_LATENCY.observe(latency_s)
        COST_PER_CALL.observe(cost_usd)
        if error:
            REQUEST_ERRORS.labels(error_type=error).inc()
        if quality is not None:
            QUALITY_SCORE.set(quality)
```

---

## 9. LLMOps Platforms Compared

| Platform | Open-source | Tracing | Eval | Prompt Mgmt | Drift | Registry | Best for |
|---|---|---|---|---|---|---|---|
| **Langfuse** | ✅ Self-host | ✅ | ✅ Datasets | ✅ | ❌ | ❌ | Privacy-first, free |
| **LangSmith** | ❌ SaaS | ✅ | ✅ Best | ✅ | ❌ | Limited | LangChain shops |
| **Arize Phoenix** | ✅ | ✅ | ✅ | ❌ | ✅ Best | ❌ | RAG debugging, drift |
| **Helicone** | ✅ Proxy | ✅ | Basic | ❌ | ❌ | ❌ | Cost visibility fast |
| **W&B Weave** | ❌ | ✅ | ✅ | Limited | ❌ | ✅ | Research teams |
| **MLflow** | ✅ | Limited | ✅ mlflow.evaluate | As artifacts | ❌ | ✅ Best | Enterprise, model registry |

### 9.1 Langfuse Setup (Fastest Path to Traces)

```python
# pip install langfuse litellm
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import litellm

langfuse = Langfuse(
    public_key="lf-pub-...",
    secret_key="lf-sk-...",
    host="https://cloud.langfuse.com",  # or self-hosted
)

@observe()  # ← this decorator captures the full trace automatically
def run_agent(user_query: str) -> str:
    langfuse_context.update_current_trace(
        name="compliance-agent",
        user_id="user-123",
        tags=["production", "compliance"],
        metadata={"query_type": "regulatory"}
    )
    
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_query}]
    )
    
    result = response.choices[0].message.content
    
    # Score this trace (LLM-judge can do this async)
    langfuse_context.score_current_trace(
        name="quality",
        value=score_with_llm_judge(user_query, result),
        comment="LLM-judge evaluation"
    )
    
    return result
```

### 9.2 Arize Phoenix for Embedding Drift

```python
# pip install arize-phoenix openinference-instrumentation-litellm
import phoenix as px
from openinference.instrumentation.litellm import LiteLLMInstrumentor

# Start Phoenix (local, no API key needed)
px.launch_app()

# Auto-instrument litellm — all calls are traced + embeddings captured
LiteLLMInstrumentor().instrument()

# Now all your litellm calls appear in Phoenix UI at http://localhost:6006
# Phoenix automatically visualises embedding clusters and drift over time
```

---

## 10. Production Monitoring Stack

### 10.1 What to Emit from Every Agent Call

```python
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class CallTrace:
    trace_id: str
    prompt_name: str
    prompt_version: str
    model: str
    user_id: str
    # Timing
    start_ts: float = field(default_factory=time.time)
    end_ts: float = 0.0
    # Tokens
    tokens_in: int = 0
    tokens_out: int = 0
    # Results
    error: str | None = None
    quality_score: float | None = None
    # Tools
    tools_called: list[str] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000

    @property
    def cost_usd(self) -> float:
        # gpt-4o-mini pricing
        in_cost = self.tokens_in * 0.15 / 1_000_000
        out_cost = self.tokens_out * 0.60 / 1_000_000
        return in_cost + out_cost
```

### 10.2 Recommended Grafana Dashboard Layout

```
Row 1 — Traffic
  Panel: requests/min (last 1h)   │  Panel: error rate %   │  Panel: active users

Row 2 — Latency
  Panel: P50/P95/P99 latency heatmap (last 24h)

Row 3 — Quality
  Panel: quality score rolling avg │  Panel: quality score heatmap by hour

Row 4 — Cost
  Panel: cost/hour                 │  Panel: tokens in/out/hr  │  Panel: daily spend

Row 5 — Drift
  Panel: embedding KS stat         │  Panel: topic similarity   │  Panel: regression alerts
```

### 10.3 OpenTelemetry Integration with litellm

```python
# pip install opentelemetry-sdk opentelemetry-exporter-otlp litellm
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure OTEL
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agent")

# litellm has native OTEL support
import litellm
litellm.success_callback = ["otel"]  # built-in integration
# Sets OTEL_EXPORTER_OTLP_ENDPOINT env var to point to your collector
```

---

## 11. CI/CD for LLM Agents

### 11.1 Full GitHub Actions Workflow

```yaml
# .github/workflows/llm_ci.yml
name: LLM Agent CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  eval-gate:
    needs: test
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements.txt
      
      - name: Run golden dataset eval
        run: |
          python eval/run_eval.py \
            --dataset eval/golden_v2.jsonl \
            --threshold 0.85 \
            --output eval_report.json
      
      - name: Check eval gate
        run: |
          python -c "
          import json, sys
          r = json.load(open('eval_report.json'))
          if r['accuracy'] < 0.85:
              print(f'FAIL: accuracy {r[\"accuracy\"]:.2%} < 85%')
              sys.exit(1)
          if r['quality_score'] < 7.5:
              print(f'FAIL: quality {r[\"quality_score\"]:.1f} < 7.5')
              sys.exit(1)
          print(f'PASS: accuracy={r[\"accuracy\"]:.2%}, quality={r[\"quality_score\"]:.1f}')
          "
      
      - name: Upload eval report
        uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: eval_report.json

  docker:
    needs: eval-gate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gistry,ref=ghcr.io/${{ github.repository }}:buildcache
          cache-to: type=registry,ref=ghcr.io/${{ github.repository }}:buildcache,mode=max

  deploy:
    needs: docker
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Rolling update
        run: |
          kubectl set image deployment/agent-api \
            agent=ghcr.io/${{ github.repository }}:${{ github.sha }}
          kubectl rollout status deployment/agent-api --timeout=5m
      
      - name: Smoke test
        run: |
          sleep 10
          response=$(curl -sf -X POST https://api.example.com/agent \
            -H "Content-Type: application/json" \
            -d '{"query": "smoke test query"}')
          echo "$response" | python -c "
          import json, sys
          r = json.load(sys.stdin)
          assert r.get('answer'), 'No answer in response'
          print('Smoke test passed')
          "
      
      - name: Rollback on failure
        if: failure()
        run: kubectl rollout undo deployment/agent-api
```

### 11.2 Eval Script Pattern

```python
# eval/run_eval.py
import json, sys, argparse, time
from pathlib import Path

def run_eval(dataset_path: str, threshold: float, output: str):
    dataset = [json.loads(l) for l in Path(dataset_path).read_text().splitlines()]
    
    results = []
    for case in dataset:
        start = time.time()
        response = call_agent(case["input"])
        latency = time.time() - start
        
        passed = evaluate_case(case, response)
        quality = llm_judge(case["input"], response)
        
        results.append({
            "id": case["id"],
            "passed": passed,
            "quality": quality,
            "latency_ms": latency * 1000,
        })
    
    accuracy = sum(r["passed"] for r in results) / len(results)
    quality_avg = sum(r["quality"] for r in results) / len(results)
    
    report = {
        "accuracy": accuracy,
        "quality_score": quality_avg,
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "failed_cases": [r["id"] for r in results if not r["passed"]],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    
    Path(output).write_text(json.dumps(report, indent=2))
    
    # Exit code for CI
    if accuracy < threshold:
        print(f"EVAL GATE FAILED: {accuracy:.2%} < {threshold:.2%}")
        sys.exit(1)
    
    print(f"EVAL GATE PASSED: accuracy={accuracy:.2%}, quality={quality_avg:.1f}/10")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output", default="eval_report.json")
    args = parser.parse_args()
    run_eval(args.dataset, args.threshold, args.output)
```

---

## 12. Cost Governance

### 12.1 The Cost Attribution Model

```
Level 1: Per-call cost     = tokens_in × in_price + tokens_out × out_price
Level 2: Per-session cost  = Σ per-call costs in a session
Level 3: Per-user cost     = Σ per-session costs for a user (daily/monthly)
Level 4: Per-team cost     = Σ per-user costs by team tag (chargeback)
Level 5: Per-feature cost  = Σ per-call costs tagged with feature name
```

### 12.2 Cost Budget Enforcement

```python
class CostBudgetEnforcer:
    """Hard and soft limits on LLM spending."""
    
    def __init__(self, db_path: str):
        self.limits = {
            "per_call_usd": 0.10,       # Hard: kill call if over
            "per_user_daily_usd": 2.00,  # Soft: warn user
            "per_team_daily_usd": 50.00, # Soft: alert team lead
            "total_daily_usd": 500.00,   # Hard: stop all calls
        }
    
    def check_per_call(self, estimated_cost: float) -> None:
        if estimated_cost > self.limits["per_call_usd"]:
            raise ValueError(
                f"Estimated cost ${estimated_cost:.4f} exceeds "
                f"per-call limit ${self.limits['per_call_usd']:.2f}. "
                "Compress context or use a cheaper model."
            )
    
    def check_daily_total(self) -> bool:
        spent = self._get_today_total()
        if spent > self.limits["total_daily_usd"]:
            # This is a hard stop — log to alerting and block
            self._alert(f"HARD STOP: Daily spend ${spent:.2f} exceeded limit")
            return False
        elif spent > self.limits["total_daily_usd"] * 0.8:
            self._alert(f"WARNING: At 80% of daily budget (${spent:.2f})")
        return True
```

### 12.3 Model Tiering for Cost Control

```python
# Route queries to the cheapest model that can handle them
MODEL_TIERS = {
    "nano":   {"model": "gpt-4o-mini",    "cost_per_mtok_in": 0.15,  "max_complexity": 3},
    "small":  {"model": "gpt-4.1-mini",   "cost_per_mtok_in": 0.40,  "max_complexity": 6},
    "medium": {"model": "gpt-4o",         "cost_per_mtok_in": 2.50,  "max_complexity": 8},
    "large":  {"model": "claude-opus-4",  "cost_per_mtok_in": 15.00, "max_complexity": 10},
}

def select_model(complexity: int, max_cost_usd: float | None = None) -> str:
    for tier_name, tier in MODEL_TIERS.items():
        if complexity <= tier["max_complexity"]:
            return tier["model"]
    return MODEL_TIERS["large"]["model"]
```

---

## 13. Feature Stores for LLM Context

### 13.1 Vector Store as Live Feature Store

In classic ML, a feature store provides pre-computed features at low latency. For LLM agents, the **vector store is your feature store**:

```
Classic ML:            LLM Agent:
User features          User history (episodic memory)
Item features          Domain knowledge (RAG chunks)
Interaction features   Tool results (cached)
Session features       Conversation context
```

### 13.2 Cache Layers for Context

```python
import hashlib
from functools import lru_cache

class SemanticCache:
    """
    Cache LLM responses by semantic similarity of the query.
    Avoids redundant LLM calls for similar queries.
    """
    def __init__(self, embed_fn, similarity_threshold: float = 0.95):
        self.embed_fn = embed_fn
        self.threshold = similarity_threshold
        self.cache: list[dict] = []  # {embedding, response, query}

    def get(self, query: str) -> str | None:
        import numpy as np
        q_emb = np.array(self.embed_fn(query))
        for item in self.cache:
            similarity = np.dot(q_emb, item["embedding"]) / (
                np.linalg.norm(q_emb) * np.linalg.norm(item["embedding"]) + 1e-9
            )
            if similarity > self.threshold:
                return item["response"]
        return None

    def set(self, query: str, response: str):
        self.cache.append({
            "embedding": self.embed_fn(query),
            "response": response,
            "query": query,
        })
```

### 13.3 Retrieval-Augmented Context as Features

```
Query ─► Embed ─► Vector Search ─► Top-K chunks   ← "Features" for this query
                                        │
                                        ▼
                                   Reranker          ← Feature selection
                                        │
                                        ▼
                               Context Assembly      ← Feature engineering
                                        │
                                        ▼
                              LLM (uses features)    ← Inference
```

The vector store update pipeline (embed new docs → upsert) is your **feature pipeline**. Keep it fresh (Celery + nightly re-embed or triggered on doc upload).

---

## 14. Reference Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Full Production LLMOps Stack                      │
│                                                                          │
│  Developer Workflow                                                       │
│  ─────────────────                                                        │
│  Prompt v1 → PromptRegistry ──► Git PR ──► GitHub Actions CI            │
│                                                  │                       │
│                              ┌───────────────────┤                       │
│                              ▼                   ▼                       │
│                       pytest pass?       Eval gate pass?                 │
│                              │           (golden dataset,                │
│                              │            LLM-judge ≥ 7.5)              │
│                              └───────────────────┤                       │
│                                                  ▼                       │
│                                          Docker build → ghcr.io          │
│                                                  │                       │
│                                                  ▼                       │
│                                       Canary deploy (10%)                │
│                                                  │                       │
│                                                  ▼                       │
│  Runtime Stack                          Metrics compare                  │
│  ─────────────                         canary vs production              │
│                                                  │                       │
│  User ──► FastAPI ──► Agent ──► litellm ──► OpenAI/Anthropic           │
│                │          │                      │                       │
│                │          └──► Tool calls         │                      │
│                │          │   (web, SQL, MCP)     │                      │
│                │          │                       │                       │
│                ▼          ▼                       ▼                      │
│           Langfuse     Redis cache           Cost tracker                │
│           (traces)     (semantic)            (SQLite)                    │
│                │                                  │                       │
│                ▼                                  ▼                      │
│           Prometheus ──────────────────► Grafana dashboard               │
│           metrics:                       │                               │
│           - latency P50/P95/P99          ▼                               │
│           - error rate               PagerDuty / Slack                   │
│           - quality score rolling    (SLO breach alert)                  │
│           - cost/call                                                    │
│           - token usage                                                  │
│                │                                                         │
│                ▼                                                         │
│       Drift Detectors ──────────────► Alert on:                         │
│       - EmbeddingDrift                  - Topic drift                   │
│       - TopicDrift                      - Quality drift                 │
│       - QualityDrift                    - Latency drift                 │
│                │                              │                          │
│                ▼                              ▼                          │
│       ContinuousEvalScheduler ──► Regression detected ──► Rollback      │
│       (APScheduler, every 6h)      trigger prompt v-1                   │
│                                    or model rollback                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Interview Framework

When asked about LLMOps in an interview, structure your answer as: **"We manage four lifecycles simultaneously."**

```
Lifecycle 1: Prompt Lifecycle
  develop → version → eval → stage → canary → production → monitor → iterate

Lifecycle 2: Model Lifecycle
  select → experiment track → eval gate → registry → promote → monitor → rollback

Lifecycle 3: Data Lifecycle
  golden dataset → continuous eval → drift detection → dataset expansion → re-eval

Lifecycle 4: Cost Lifecycle
  budget → per-call cap → daily limit → model tiering → chargeback → report
```

### 15.1 Common Interview Questions

**Q: "How do you handle silent model API updates breaking your prompts?"**
> "We pin model versions (`gpt-4o-mini-2024-07-18` not `gpt-4o-mini`), run continuous evals every 6 hours against a golden dataset, and have a rollback mechanism that re-routes traffic to the previous prompt version within 60 seconds of a regression alert."

**Q: "How do you know if your LLM agent is getting worse in production?"**
> "Three signals: (1) LLM-judge quality scores tracked by Prometheus, (2) embedding drift detector using KS test on input distribution, (3) continuous eval scheduler comparing to golden dataset baseline. Any of the three dropping triggers a Slack alert."

**Q: "How do you A/B test prompt changes safely?"**
> "Hash-based traffic splitting — same user always gets the same variant. New prompt gets 10% of traffic in shadow mode first (response computed but discarded), then 10% live, then compare P95 latency + quality score over 24 hours. If both within tolerance, auto-promote to 100%."

**Q: "What's your cost control strategy?"**
> "Four layers: per-call cost estimate before calling (block if over $0.10), query complexity classifier routing to cheapest adequate model, daily spend limit with hard stop at threshold, and team-level chargeback tagging. Weekly cost report to each team's Slack channel."

---

## 16. Decision Cheat Sheet

| Decision | Question | Answer |
|---|---|---|
| **Prompt storage** | Team size < 5? | Git files + simple JSON metadata |
| **Prompt storage** | Team size ≥ 5 or multiple services? | SQLite registry or LangSmith |
| **Experiment tracking** | Already using MLflow? | Keep MLflow, add LLM metrics |
| **Experiment tracking** | Starting fresh? | Langfuse (free, self-host) |
| **Tracing platform** | Need embedding drift viz? | Arize Phoenix |
| **Tracing platform** | Need team eval datasets? | LangSmith |
| **Tracing platform** | Need cost visibility ASAP? | Helicone (proxy-based, zero-code) |
| **Eval schedule** | How often? | Every 6h for user-facing; daily for batch |
| **Drift detection** | What to detect first? | Quality drift (highest signal) |
| **Canary %** | New prompt first deploy? | 5% for 2h → 20% for 6h → 100% |
| **Rollback trigger** | Quality threshold? | Drop > 5% from baseline auto-rolls back |
| **Cost control** | Highest bang-for-buck fix? | Model routing (nano vs large) saves 80%+ |

---

## 17. Production Checklist

### Before Deploying a New Prompt Version
- [ ] Prompt stored in registry with version number and changelog
- [ ] Eval gate runs on golden dataset (accuracy ≥ 85%, quality ≥ 7.5)
- [ ] Token count verified (won't overflow context window)
- [ ] Latency regression check (P95 within 20% of baseline)
- [ ] Cost per call estimate documented
- [ ] Rollback plan documented (which version to roll back to)

### Monitoring Setup (Before Going Live)
- [ ] Langfuse/LangSmith tracing connected and emitting traces
- [ ] Prometheus scraping latency, error rate, cost, token usage
- [ ] Grafana dashboard shows all 4 rows (traffic, latency, quality, cost)
- [ ] PagerDuty/Slack alerts configured for SLO breaches
- [ ] Drift detectors running (embedding + quality)
- [ ] Continuous eval scheduler configured (golden dataset → LLM judge)

### Weekly Operations
- [ ] Review drift detector outputs
- [ ] Check cost trend (is $/call creeping up?)
- [ ] Review quality score trend
- [ ] Add new failure cases from production to golden dataset
- [ ] Archive old prompt versions

---

*Cross-references*: [§09 Observability](09_observability.md) — tracing and metrics setup |
[§10 Deployment](10_deployment.md) — Docker/K8s | [§12 Evaluation](12_evaluation.md) — evaluation metrics and LLM-judge |
[§13 System Design](13_system_design.md) — architecture patterns
