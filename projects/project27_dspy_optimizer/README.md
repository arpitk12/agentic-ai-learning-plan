# Project 27 — DSPy Prompt Optimizer

> **Stack**: DSPy · LiteLLM · SciPy · Pydantic  
> **Phase 7 — Advanced Production** | Priority: P1 🟠

---

## What You'll Build

A systematic prompt engineering pipeline using DSPy that replaces hand-crafted few-shot examples with algorithmically selected ones — improving accuracy by 10-20% without changing the model.

---

## Why DSPy?

Hand-crafted prompts plateau around 70-75% accuracy and are brittle. DSPy treats prompts as compiled programs: you define the task (Signature), provide labeled examples, and DSPy searches for the best prompt automatically.

**Time saved**: 4 hours of manual prompt tweaking → 20 minutes of DSPy optimization.

---

## Milestones

### Milestone 1 — Signature + Module Definition
Define a `ComplianceClassifier` DSPy Signature with four fields: `document`, `document_type` (inputs), `risk_level`, `key_concern` (outputs). Wrap it in a `ComplianceModule` using `ChainOfThought`.

### Milestone 2 — Dev Set Construction
Build 60 labeled examples (40 train / 20 test). Cover all risk levels and document types. Verify the metric function returns 1.0 for exact match, 0.5 for adjacent level.

### Milestone 3 — BootstrapFewShot Optimization
Run `BootstrapFewShot` with 4 demos. Measure: compile time, selected examples, before/after accuracy on test set.

### Milestone 4 — MIPROv2 Optimization
Run `MIPROv2(auto="medium")`. Compare: instruction text before/after (MIPROv2 changes the instruction, not just the examples). Measure accuracy improvement.

### Milestone 5 — Comparison Table
Evaluate all three approaches on test set. Print a table: baseline / BootstrapFewShot / MIPROv2, with accuracy and relative gain.

### Milestone 6 — Production Integration
Save the optimized module with `module.save()`. Show how to load and use it as a drop-in replacement for manual prompting. Verify loaded module gives identical outputs.

### Milestone 7 — Continuous Optimization
Build a script that: collects human-corrected outputs in production, re-runs DSPy optimization weekly, compares new module vs current production module, auto-promotes if improvement > 5%.

---

## Setup

```bash
pip install dspy-ai litellm pydantic python-dotenv scipy
```

---

## Expected Output

```
=== DSPy Optimization Results ===

Dev set: 40 train / 20 test

Baseline (no optimization):
  Accuracy: 72.5% exact | 88.0% adjacent

BootstrapFewShot (4 demos selected):
  Accuracy: 83.0% exact | 94.0% adjacent  (+10.5%)
  Compile time: 42s
  Selected demos: [low: 1, medium: 1, high: 1, critical: 1]

MIPROv2 (medium budget, 15 candidates):
  Accuracy: 91.0% exact | 97.0% adjacent  (+18.5%)
  Compile time: 8m 12s
  Optimized instruction: "You are a compliance risk expert..."

Recommendation: Use MIPROv2 module in production.
```
