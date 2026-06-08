"""
Exercise 1: DSPy Prompt Optimization for Compliance Classification
Phase 7 / Week 15 — DSPy + A/B Testing

Goal: Use DSPy to automatically optimize prompts for a compliance classifier,
      replacing hand-crafted few-shot examples with algorithmically selected ones.

Stack: dspy-ai · litellm · pydantic · datasets

pip install dspy-ai litellm pydantic python-dotenv scipy

TODOs:
  1. Define a DSPy Signature for compliance classification
  2. Build a DSPy Module using ChainOfThought (adds internal reasoning)
  3. Create a dev set of 50 labeled examples
  4. Define a metric function (exact match + adjacent-level partial credit)
  5. Optimize with BootstrapFewShot (fast baseline optimizer)
  6. Optimize with MIPROv2 (also optimizes the instruction text)
  7. Evaluate all three (baseline, BootstrapFewShot, MIPROv2) on test set
  8. BONUS: Save + load optimized module; show the selected few-shot examples
"""
from __future__ import annotations
import os, json, asyncio, random
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# pip install dspy-ai
import dspy
from dspy import Signature, InputField, OutputField, ChainOfThought, Predict

# ── Configure DSPy LM ─────────────────────────────────────────────────────────

lm = dspy.LM("openai/gpt-4o-mini", max_tokens=300, temperature=0.0)
dspy.configure(lm=lm)

RISK_LEVELS = ["low", "medium", "high", "critical"]

# ── TODO 1: Define DSPy Signature ─────────────────────────────────────────────

class ComplianceClassifier(Signature):
    """
    TODO 1: Define a DSPy Signature for compliance risk classification.

    A Signature defines the input and output fields of the task.
    DSPy uses this to generate and optimize prompts.

    Add these fields using InputField and OutputField:
      - document: str = InputField(desc="Business document text (contract, policy, invoice, etc.)")
      - document_type: str = InputField(desc="Document category: contract, policy, invoice, agreement, report")
      - risk_level: str = OutputField(desc="Risk classification: low, medium, high, or critical")
      - key_concern: str = OutputField(desc="The single most important compliance concern, or 'none'")

    The docstring becomes the task description in the prompt.
    Change the docstring to: "Classify the compliance risk level of a business document
    and identify the primary concern."
    """
    # TODO 1: replace this placeholder with actual fields
    document: str = InputField(desc="TODO: add description")
    risk_level: str = OutputField(desc="TODO: add description")

# ── TODO 2: Build DSPy Module ─────────────────────────────────────────────────

class ComplianceModule(dspy.Module):
    """
    TODO 2: Create a Module that wraps the Signature in a reasoning strategy.

    In __init__, create:
      self.classify = ChainOfThought(ComplianceClassifier)

    ChainOfThought adds a hidden "reasoning" field that DSPy uses internally
    before generating the final output fields — improves accuracy on complex tasks.

    In forward(self, document, document_type):
      - Call self.classify(document=document, document_type=document_type)
      - Return the prediction

    Also add a second Module called ComplianceProgramOfThought that uses
      self.classify = dspy.ProgramOfThought(ComplianceClassifier)
    for comparison (ProgramOfThought generates Python code to solve the task).
    """
    def __init__(self):
        # TODO 2: implement here
        raise NotImplementedError

    def forward(self, document: str, document_type: str) -> dspy.Prediction:
        # TODO 2: implement here
        raise NotImplementedError

# ── TODO 3: Build Dev + Test Sets ─────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    ("Standard mutual NDA with 2-year term, standard confidentiality clauses", "contract", "low"),
    ("SaaS agreement with SSO integration, no data processing terms", "contract", "medium"),
    ("Payment processing agreement lacking PCI-DSS compliance statement", "contract", "high"),
    ("Vendor contract processing EU customer PII with no DPA exhibit attached", "contract", "critical"),
    ("Internal expense policy: meals up to $50, travel economy class", "policy", "low"),
    ("Bring Your Own Device policy without MDM requirement", "policy", "medium"),
    ("Whistleblower policy with anonymous reporting but no non-retaliation guarantee", "policy", "high"),
    ("Data retention policy deleting financial records after 3 years (SOX requires 7)", "policy", "critical"),
    ("Invoice for office supplies $450 from approved vendor", "invoice", "low"),
    ("Invoice for consulting services $95,000 with no SOW attached", "invoice", "medium"),
    ("Invoice from vendor flagged in sanctions list screening", "invoice", "critical"),
    ("Software license agreement with source code escrow, standard terms", "agreement", "low"),
    ("Revenue sharing agreement with 40% commission, no audit rights clause", "agreement", "medium"),
    ("Joint venture agreement in jurisdiction with restricted data transfer rules", "agreement", "high"),
    ("Q3 financial report with revenue figures, unaudited, material discrepancy noted", "report", "high"),
]

def build_datasets(n_train: int = 40, n_test: int = 15) -> tuple[list, list]:
    """
    TODO 3: Create train and test sets as lists of dspy.Example objects.

    For each sample in SAMPLE_DOCUMENTS, create:
      dspy.Example(
          document=doc,
          document_type=doc_type,
          risk_level=risk,
          key_concern="auto",  # placeholder; DSPy optimizers don't need ground truth for all fields
      ).with_inputs("document", "document_type")

    If len(SAMPLE_DOCUMENTS) < n_train + n_test, duplicate and shuffle to reach the count.
    Use random.seed(42) for reproducibility.

    Return (train_set[:n_train], test_set[:n_test]).
    """
    random.seed(42)
    # TODO 3: implement here
    raise NotImplementedError

# ── TODO 4: Define Metric Function ────────────────────────────────────────────

def compliance_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
) -> float:
    """
    TODO 4: Score a prediction against the gold label.

    Scoring:
      - Exact match (gold.risk_level == pred.risk_level): return 1.0
      - Off by one level (adjacent): return 0.5
        (low↔medium, medium↔high, high↔critical)
      - Off by 2+ levels: return 0.0
      - Unparseable (pred.risk_level not in RISK_LEVELS): return 0.0

    Clean pred.risk_level: strip whitespace, lowercase, take first word.
    """
    # TODO 4: implement here
    raise NotImplementedError

# ── TODO 5: Optimize with BootstrapFewShot ────────────────────────────────────

def optimize_bootstrap(train_set: list, test_set: list) -> tuple[dspy.Module, float]:
    """
    TODO 5: Run BootstrapFewShot optimization.

    from dspy.teleprompt import BootstrapFewShot

    bootstrap = BootstrapFewShot(
        metric=compliance_metric,
        max_bootstrapped_demos=4,   # add up to 4 examples to the prompt
        max_labeled_demos=4,
        max_rounds=1,
    )
    optimized = bootstrap.compile(ComplianceModule(), trainset=train_set)

    Then evaluate on test_set using dspy.evaluate.Evaluate:
      from dspy.evaluate import Evaluate
      evaluator = Evaluate(devset=test_set, metric=compliance_metric, num_threads=4)
      score = evaluator(optimized)

    Print: "BootstrapFewShot score: XX.X%"
    Return (optimized_module, score).
    """
    # TODO 5: implement here
    raise NotImplementedError

# ── TODO 6: Optimize with MIPROv2 ─────────────────────────────────────────────

def optimize_mipro(train_set: list, test_set: list) -> tuple[dspy.Module, float]:
    """
    TODO 6: Run MIPROv2 optimization (also optimizes the instruction text).

    from dspy.teleprompt import MIPROv2

    mipro = MIPROv2(
        metric=compliance_metric,
        auto="light",     # "light" for speed; use "medium" or "heavy" for production
        num_threads=4,
        verbose=False,
    )
    optimized = mipro.compile(
        ComplianceModule(),
        trainset=train_set,
        valset=test_set[:10],
        num_trials=10,    # number of instruction candidates to evaluate
    )

    Evaluate on test_set and return (optimized_module, score).
    Print: "MIPROv2 score: XX.X%"
    """
    # TODO 6: implement here
    raise NotImplementedError

# ── TODO 7: Evaluate all three ────────────────────────────────────────────────

def run_comparison(train_set: list, test_set: list) -> dict:
    """
    TODO 7: Compare baseline, BootstrapFewShot, and MIPROv2.

    from dspy.evaluate import Evaluate
    evaluator = Evaluate(devset=test_set, metric=compliance_metric, num_threads=4, display_progress=False)

    a) Baseline: evaluator(ComplianceModule())
    b) Bootstrap: run optimize_bootstrap, then evaluator(bootstrap_module)
    c) MIPROv2: run optimize_mipro, then evaluator(mipro_module)

    Print a comparison table:
    | Method         | Exact Acc. | Gain  |
    |----------------|------------|-------|
    | Baseline       | 71.0%      | —     |
    | BootstrapFewShot | 82.0%   | +11%  |
    | MIPROv2        | 89.0%      | +18%  |

    Return {"baseline": float, "bootstrap": float, "mipro": float}
    """
    # TODO 7: implement here
    raise NotImplementedError

# ── TODO 8 (BONUS): Save + inspect optimized module ──────────────────────────

def save_and_inspect(module: dspy.Module, path: str = "compliance_optimized.json") -> None:
    """
    TODO 8: Save the optimized module and display the selected few-shot examples.

    a) module.save(path)
       print(f"Saved to {path}")

    b) Load it back: loaded = ComplianceModule(); loaded.load(path)

    c) Inspect the demos that DSPy selected:
       demos = module.classify.demos   # list of dspy.Example
       print(f"\\nSelected {len(demos)} few-shot examples:")
       for i, demo in enumerate(demos):
           print(f"  [{i+1}] {demo.document[:60]}... → {demo.risk_level}")

    d) Print the optimized instruction (if MIPROv2 was used):
       print("\\nOptimized instruction:")
       print(module.classify.signature.instructions)
    """
    # TODO 8: implement here
    raise NotImplementedError

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== DSPy Prompt Optimizer Exercise ===\n")

    # Build datasets
    print("1. Building dev + test datasets...")
    train_set, test_set = build_datasets(n_train=40, n_test=15)
    print(f"   Train: {len(train_set)} | Test: {len(test_set)}\n")

    # Run full comparison
    print("2. Running optimization comparison...")
    results = run_comparison(train_set, test_set)

    print("\n3. Saving best model...")
    # Run MIPROv2 again to get the module object for saving
    best_module, _ = optimize_mipro(train_set, test_set)
    save_and_inspect(best_module)

    print("\n✅ DSPy optimization exercise complete!")
    print(f"   Best accuracy: {max(results.values()):.1%}")

if __name__ == "__main__":
    main()
