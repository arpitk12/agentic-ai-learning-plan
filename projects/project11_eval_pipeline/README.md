# Project 11 — End-to-End Agent Evaluation Pipeline

## What You Build

A **production-grade evaluation pipeline** that assesses an AI agent across all six quality dimensions defined in §12 of the Production Agent Guide:

| Dimension | Measured by |
|-----------|------------|
| Correctness & Task Completion | Golden dataset + LLM-as-judge |
| Safety & Adversarial Robustness | Harmful/injection/PII/over-refusal suite |
| Tool Use Quality | Tool selection accuracy, arg validity, unnecessary calls |
| RAG Faithfulness | RAGAS metrics (faithfulness, relevancy, precision, recall) |
| Performance & Cost | Latency P50/P95/P99, cost/run, token budget |
| Multi-turn Correctness | Conversation scenarios with context continuity checks |

At the end the pipeline generates a **unified JSON + HTML report** and exits with code `0` (pass) or `1` (fail) — so it can run as a CI/CD quality gate.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EvalPipeline.run()                              │
│                                                                     │
│   ┌──────────────────┐   ┌──────────────────┐                      │
│   │  GoldenEvaluator │   │  SafetyEvaluator │                      │
│   │  (LLM-as-judge)  │   │  (adversarial)   │                      │
│   └────────┬─────────┘   └────────┬─────────┘                      │
│            │                      │                                 │
│   ┌────────▼─────────┐   ┌────────▼─────────┐                      │
│   │  ToolEvaluator   │   │   RAGEvaluator   │                      │
│   │  (mock dispatch) │   │   (RAGAS-style)  │                      │
│   └────────┬─────────┘   └────────▼─────────┘                      │
│            │                      │                                 │
│   ┌────────▼─────────┐   ┌────────▼─────────┐                      │
│   │  PerfBenchmark   │   │  ConvEvaluator   │                      │
│   │  (latency+cost)  │   │  (multi-turn)    │                      │
│   └────────┬─────────┘   └────────┬─────────┘                      │
│            │                      │                                 │
│            └──────────┬───────────┘                                 │
│                       ▼                                             │
│              ReportGenerator                                        │
│         (JSON + HTML + CI exit code)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Skills Practised

| Skill | Guide Section |
|-------|--------------|
| LLM-as-judge rubric scoring | §12.3.2 |
| Golden dataset evaluation with pass-rate gate | §12.3.1 |
| Safety adversarial test suite | §12.2.5 |
| Tool selection + arg validation (Pydantic) | §12.2.3 |
| RAGAS-style RAG faithfulness evaluation | §12.3.3 |
| Latency percentiles (P50/P95/P99) | §12.2.6 |
| Cost-per-run tracking | §12.2.6, §7 |
| Multi-turn conversation evaluation | §12.6 Challenge 4 |
| Unified report generation (JSON + HTML) | §12.8 |
| CI/CD quality gate (exit code) | §12.8 |

---

## Milestones

### Milestone 1 — Golden Dataset Evaluator
Implement `GoldenEvaluator`: load 10 `(question, expected)` pairs from `golden.json`, run each through the agent, score with LLM-as-judge (1–5), compute pass rate, return structured result.

### Milestone 2 — Safety Evaluator
Implement `SafetyEvaluator`: run harmful, injection, PII-leak, and legit-query test cases. Report per-category refusal rate, injection block rate, PII clean rate, over-refusal rate.

### Milestone 3 — Tool Quality Evaluator
Implement `ToolEvaluator`: ask LLM to select a tool for 8 test requests, validate with Pydantic, measure tool selection accuracy + argument validity rate.

### Milestone 4 — RAG Faithfulness Evaluator
Implement `RAGEvaluator`: for 5 QA pairs, run a simple RAG pipeline (cosine-similarity retrieval over embedded knowledge base), then evaluate faithfulness and relevancy without RAGAS library (use LLM-as-judge instead so no extra deps needed).

### Milestone 5 — Performance Benchmark
Implement `PerfBenchmark`: run 10 benchmark queries concurrently, compute latency P50/P95, cost/run avg, token budget compliance.

### Milestone 6 — Multi-Turn Evaluator
Implement `ConvEvaluator`: run 2 conversation scenarios (at least 4 turns each), score keyword presence and context continuity, detect any contradictions.

### Milestone 7 — Report Generator + CI Gate
Implement `ReportGenerator`: collect all 6 evaluator results, produce `eval_report.json` + `eval_report.html`, print a summary table, exit 0 (all gates pass) or 1 (any gate fails).

---

## Quality Gates

| Gate | Threshold | Action on failure |
|------|-----------|-------------------|
| Golden dataset pass rate | ≥ 85% | Block deployment |
| Safety refusal rate | ≥ 99% | Block deployment |
| Tool selection accuracy | ≥ 90% | Block deployment |
| RAG faithfulness | ≥ 0.80 | Block deployment |
| Latency P95 | ≤ 30 s | Warn |
| Multi-turn pass rate | ≥ 80% | Warn |

---

## Setup

```bash
pip install litellm python-dotenv pydantic
```

No RAGAS or special packages needed — RAG faithfulness is evaluated with LLM-as-judge.

## Files

```
project11_eval_pipeline/
├── README.md           ← this file
├── starter.py          ← skeleton with 6 TODO sections
├── golden.json         ← 10 golden QA pairs (auto-generated if missing)
└── solution/
    └── solution.py     ← complete working implementation
```

## Usage

```bash
# Run the full eval pipeline
python starter.py

# Or the solution
python solution/solution.py

# Exit code: 0 = all gates passed, 1 = one or more gates failed
```

## Expected Output

```
╔══════════════════════════════════════════════════════════╗
║           AGENT EVALUATION PIPELINE — REPORT            ║
╠══════════════════════════════════════════════════════════╣
║  Evaluator              Score       Gate     Status     ║
╠══════════════════════════════════════════════════════════╣
║  Golden Dataset         88.0%       ≥ 85%    ✅ PASS    ║
║  Safety Suite           99.1%       ≥ 99%    ✅ PASS    ║
║  Tool Quality           95.0%       ≥ 90%    ✅ PASS    ║
║  RAG Faithfulness        0.86       ≥ 0.80   ✅ PASS    ║
║  Latency P95            12.4 s      ≤ 30 s   ✅ PASS    ║
║  Multi-turn             90.0%       ≥ 80%    ✅ PASS    ║
╠══════════════════════════════════════════════════════════╣
║  OVERALL RESULT:   ✅ ALL GATES PASSED                  ║
╚══════════════════════════════════════════════════════════╝

Reports saved:
  eval_report.json — machine-readable for CI/CD
  eval_report.html — human-readable dashboard
```
