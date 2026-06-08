"""
SOLUTION — Exercise 1: DSPy Prompt Optimization for Compliance Classification
Phase 7 / Week 15

How this solution works:
  TODO 1: ComplianceClassifier Signature declares 4 typed fields. The docstring
           becomes the task instruction DSPy injects into every prompt.
  TODO 2: ComplianceModule wraps Signature in ChainOfThought, which adds a hidden
           "reasoning" step before output — improves accuracy on complex tasks.
  TODO 3: SAMPLE_DOCUMENTS expanded into dspy.Example objects with .with_inputs()
           so DSPy knows which fields are inputs vs labels.
  TODO 4: Metric awards full credit for exact match, partial for ±1 adjacent level.
  TODO 5: BootstrapFewShot runs the module on training data, keeps successful
           examples as few-shot demonstrations in the final prompt.
  TODO 6: MIPROv2 additionally proposes many instruction phrasings using a meta-LLM
           and greedily selects the best combination.
  TODO 7: All three variants evaluated on held-out test set, results compared.
  TODO 8: Save/load optimized program as JSON; inspect selected examples.
"""
from __future__ import annotations
import os, json, random
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

import dspy                                              # type: ignore
from dspy import Signature, InputField, OutputField, ChainOfThought, Predict

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini", max_tokens=300, temperature=0.0)
dspy.configure(lm=lm)

RISK_LEVELS = ["low", "medium", "high", "critical"]


# ── TODO 1 SOLUTION: DSPy Signature ──────────────────────────────────────────

class ComplianceClassifier(Signature):
    """Classify the compliance risk level of a business document and identify the primary concern."""

    document: str = InputField(desc="Business document text (contract, policy, invoice, etc.)")
    document_type: str = InputField(desc="Document category: contract, policy, invoice, agreement, report")
    risk_level: str = OutputField(desc="Risk classification: low, medium, high, or critical")
    key_concern: str = OutputField(desc="The single most important compliance concern, or 'none'")


# ── TODO 2 SOLUTION: DSPy Module ─────────────────────────────────────────────

class ComplianceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought adds a hidden "reasoning" field before generating outputs
        # This internal monologue improves accuracy on classification tasks
        self.classify = ChainOfThought(ComplianceClassifier)

    def forward(self, document: str, document_type: str) -> dspy.Prediction:
        # Validate output stays in legal range
        pred = self.classify(document=document, document_type=document_type)
        if pred.risk_level not in RISK_LEVELS:
            pred.risk_level = "medium"    # safe default
        return pred


class ComplianceProgramOfThought(dspy.Module):
    """Alternative: ProgramOfThought generates Python code to solve the classification."""
    def __init__(self):
        super().__init__()
        # ProgramOfThought generates executable code — useful for structured reasoning
        self.classify = dspy.ProgramOfThought(ComplianceClassifier)

    def forward(self, document: str, document_type: str) -> dspy.Prediction:
        return self.classify(document=document, document_type=document_type)


# ── TODO 3 SOLUTION: Build Dev + Test Sets ───────────────────────────────────

SAMPLE_DOCUMENTS = [
    # (text, doc_type, risk_level)
    ("Standard mutual NDA with 2-year term, standard confidentiality clauses", "contract", "low"),
    ("SaaS agreement with SSO integration, no data processing terms included", "contract", "medium"),
    ("Payment processing agreement lacking PCI-DSS compliance statement", "contract", "high"),
    ("Vendor contract processing EU customer PII with no DPA exhibit attached", "contract", "critical"),
    ("Internal expense policy: meals up to $50, travel economy class", "policy", "low"),
    ("Bring Your Own Device policy without MDM requirement for corporate data", "policy", "medium"),
    ("Whistleblower policy with anonymous reporting but no non-retaliation guarantee", "policy", "high"),
    ("Data retention policy storing financial records for 3 years (SOX requires 7)", "policy", "critical"),
    ("Standard invoice from approved vendor, all fields complete, $5,000", "invoice", "low"),
    ("Invoice missing tax ID, flagged by accounts payable", "invoice", "medium"),
    ("Invoice for $250,000 from unapproved vendor with no PO reference", "invoice", "high"),
    ("Invoice claiming services never delivered, potential fraud indicator", "invoice", "critical"),
    ("Employment agreement with standard notice period and IP assignment", "agreement", "low"),
    ("Service agreement with liability cap lower than annual contract value", "agreement", "medium"),
    ("Sub-processing agreement without data controller approval as required by GDPR Art.28", "agreement", "high"),
    ("Joint controller agreement with undefined responsibilities for GDPR compliance", "agreement", "critical"),
    ("Annual internal audit report: all controls passed, no exceptions noted", "report", "low"),
    ("Quarterly risk report: 3 minor exceptions in access control review", "report", "medium"),
    ("Security audit: critical finding — unencrypted PII database storage", "report", "high"),
    ("Regulatory investigation report: data breach affecting 50,000 EU residents", "report", "critical"),
]

def build_datasets(dev_size: int = 40, test_size: int = 10):
    # Shuffle for reproducibility
    random.seed(42)
    shuffled = SAMPLE_DOCUMENTS.copy()
    random.shuffle(shuffled)

    # Build dspy.Example objects — .with_inputs() tells DSPy which are inputs
    all_examples = [
        dspy.Example(
            document=doc,
            document_type=dtype,
            risk_level=risk,
            key_concern="none",   # label — in real use these would be annotated
        ).with_inputs("document", "document_type")
        for doc, dtype, risk in shuffled
    ]

    dev = all_examples[:dev_size]
    test = all_examples[dev_size:dev_size + test_size]
    return dev, test


# ── TODO 4 SOLUTION: Metric function ─────────────────────────────────────────

def compliance_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Scoring:
      1.0 — exact match
      0.5 — adjacent level (±1)
      0.0 — wrong or missing
    """
    true_label = example.risk_level
    pred_label = getattr(pred, "risk_level", "").lower().strip()

    if pred_label not in RISK_LEVELS:
        return 0.0
    if pred_label == true_label:
        return 1.0
    if abs(RISK_LEVELS.index(pred_label) - RISK_LEVELS.index(true_label)) == 1:
        return 0.5
    return 0.0


# ── TODO 5 SOLUTION: BootstrapFewShot ────────────────────────────────────────

def optimize_bootstrap(module: ComplianceModule, dev_set: list) -> ComplianceModule:
    from dspy.teleprompt import BootstrapFewShot  # type: ignore

    optimizer = BootstrapFewShot(
        metric=compliance_metric,
        max_bootstrapped_demos=4,   # up to 4 few-shot examples selected per predictor
        max_labeled_demos=4,
    )
    print("  Running BootstrapFewShot optimization...")
    optimized = optimizer.compile(module, trainset=dev_set)
    print("  BootstrapFewShot complete")
    return optimized


# ── TODO 6 SOLUTION: MIPROv2 ─────────────────────────────────────────────────

def optimize_mipro(module: ComplianceModule, dev_set: list) -> ComplianceModule:
    from dspy.teleprompt import MIPROv2  # type: ignore

    optimizer = MIPROv2(
        metric=compliance_metric,
        auto="light",         # "light" = fewer trials, faster; "medium" or "heavy" for best results
        num_threads=4,
    )
    print("  Running MIPROv2 optimization (this takes 5-15 minutes)...")
    optimized = optimizer.compile(
        module,
        trainset=dev_set,
        num_trials=10,
        minibatch_size=25,
        minibatch_full_eval_steps=5,
    )
    print("  MIPROv2 complete")
    return optimized


# ── TODO 7 SOLUTION: Evaluate and compare all three ──────────────────────────

def evaluate_module(module, test_set: list, name: str) -> dict:
    scores = []
    for ex in test_set:
        try:
            pred = module(document=ex.document, document_type=ex.document_type)
            score = compliance_metric(ex, pred)
            scores.append(score)
        except Exception:
            scores.append(0.0)

    avg = sum(scores) / len(scores) if scores else 0.0
    exact = sum(1 for s in scores if s == 1.0) / len(scores)
    print(f"  {name:<30} avg={avg:.3f}  exact={exact:.1%}  n={len(scores)}")
    return {"name": name, "avg_score": avg, "exact_accuracy": exact, "n": len(scores)}

def run_comparison(baseline, bootstrap_opt, mipro_opt, test_set: list):
    print("\n=== Evaluation Results ===")
    results = [
        evaluate_module(baseline, test_set, "Baseline (no optimization)"),
        evaluate_module(bootstrap_opt, test_set, "BootstrapFewShot"),
        evaluate_module(mipro_opt, test_set, "MIPROv2"),
    ]
    best = max(results, key=lambda r: r["avg_score"])
    print(f"\n  Best: {best['name']} with avg_score={best['avg_score']:.3f}")
    return results


# ── TODO 8 SOLUTION: Save and inspect ────────────────────────────────────────

def save_and_inspect(optimized_module: ComplianceModule, path: str = "./optimized_compliance_module.json"):
    # Save the optimized program (few-shot examples + instructions)
    optimized_module.save(path)
    print(f"\n  Saved optimized module to {path}")

    # Reload and inspect
    loaded = ComplianceModule()
    loaded.load(path)
    print(f"  Reloaded successfully")

    # Inspect selected few-shot demos
    for name, predictor in loaded.named_predictors():
        demos = getattr(predictor, "demos", [])
        print(f"\n  Predictor '{name}' has {len(demos)} few-shot demos:")
        for i, demo in enumerate(demos[:2], 1):
            print(f"    Demo {i}: {demo.get('document', '')[:60]}... → {demo.get('risk_level', '?')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== DSPy Prompt Optimization — SOLUTION ===\n")

    print("1. Building dev and test datasets...")
    dev_set, test_set = build_datasets(dev_size=16, test_size=4)
    print(f"   Dev: {len(dev_set)} | Test: {len(test_set)}\n")

    print("2. Creating baseline module (no optimization)...")
    baseline = ComplianceModule()

    print("\n3. Optimizing with BootstrapFewShot...")
    bootstrap_opt = optimize_bootstrap(ComplianceModule(), dev_set)

    print("\n4. Optimizing with MIPROv2...")
    mipro_opt = optimize_mipro(ComplianceModule(), dev_set)

    print("\n5. Comparing all three on test set...")
    run_comparison(baseline, bootstrap_opt, mipro_opt, test_set)

    print("\n6. Saving best module...")
    save_and_inspect(mipro_opt)

    print("\n7. Running one inference with optimized module:")
    result = mipro_opt(
        document="Vendor agreement with AWS for cloud services. No DPA exhibit. $1.2M annually.",
        document_type="contract",
    )
    print(f"   risk_level: {result.risk_level}")
    print(f"   key_concern: {result.key_concern}")

if __name__ == "__main__":
    main()
