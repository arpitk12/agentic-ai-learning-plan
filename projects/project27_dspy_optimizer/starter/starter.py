"""Project 27 — DSPy Prompt Optimizer: Starter File
pip install dspy-ai litellm pydantic python-dotenv scipy
"""
from __future__ import annotations
import dspy
from dspy import Signature, InputField, OutputField, ChainOfThought
from dotenv import load_dotenv
load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini", max_tokens=300, temperature=0.0)
dspy.configure(lm=lm)
RISK_LEVELS = ["low", "medium", "high", "critical"]

# TODO 1: Define ComplianceClassifier Signature
# Fields: document (input), document_type (input), risk_level (output), key_concern (output)
# Docstring: "Classify the compliance risk level of a business document and identify the primary concern."
class ComplianceClassifier(Signature):
    """TODO 1: Replace this docstring. Add InputField and OutputField declarations."""
    document: str = InputField(desc="TODO")   # TODO 1
    risk_level: str = OutputField(desc="TODO") # TODO 1

# TODO 2: Build ComplianceModule using ChainOfThought(ComplianceClassifier)
class ComplianceModule(dspy.Module):
    def __init__(self):
        # TODO 2: self.classify = ChainOfThought(ComplianceClassifier)
        raise NotImplementedError
    def forward(self, document: str, document_type: str) -> dspy.Prediction:
        # TODO 2: return self.classify(document=document, document_type=document_type)
        raise NotImplementedError

# TODO 3: Build 60 labeled examples as dspy.Example objects
# Use SAMPLE_DOCUMENTS list from phase7/week15/ex1 for variety
SAMPLE_DOCUMENTS = [
    ("Standard NDA 2-year term", "contract", "low"),
    ("Payment agreement lacking PCI-DSS statement", "contract", "high"),
    ("Vendor processing EU PII with no DPA", "contract", "critical"),
    ("Internal expense policy $50 meal limit", "policy", "low"),
    ("Data retention deleting records after 3yr (SOX requires 7yr)", "policy", "critical"),
    ("Invoice from sanctioned vendor", "invoice", "critical"),
    ("Software license with escrow, standard terms", "agreement", "low"),
]

def build_datasets(n_train: int = 40, n_test: int = 20) -> tuple[list, list]:
    """TODO 3: Build train + test dspy.Example sets. Return (train, test)."""
    # YOUR CODE HERE
    raise NotImplementedError

# TODO 4: Metric function — 1.0 exact, 0.5 adjacent, 0.0 wrong
def compliance_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """TODO 4: Score prediction. Adjacent means risk levels off by 1 (low↔medium, etc)."""
    # YOUR CODE HERE
    raise NotImplementedError

# TODO 5: BootstrapFewShot optimization
def optimize_bootstrap(train_set: list, test_set: list) -> tuple:
    """TODO 5: Run BootstrapFewShot(max_bootstrapped_demos=4). Return (module, score)."""
    # from dspy.teleprompt import BootstrapFewShot
    # YOUR CODE HERE
    raise NotImplementedError

# TODO 6: MIPROv2 optimization
def optimize_mipro(train_set: list, test_set: list) -> tuple:
    """TODO 6: Run MIPROv2(auto="light", num_threads=4). Return (module, score)."""
    # from dspy.teleprompt import MIPROv2
    # YOUR CODE HERE
    raise NotImplementedError

# TODO 7: Compare all three and print results table
def run_comparison(train_set: list, test_set: list) -> dict:
    """TODO 7: Evaluate baseline, bootstrap, mipro. Print comparison table. Return scores dict."""
    # from dspy.evaluate import Evaluate
    # YOUR CODE HERE
    raise NotImplementedError

# TODO 8 (BONUS): Save optimized module + inspect selected demos
def save_and_inspect(module, path: str = "compliance_optimized.json"):
    """TODO 8: Save module, reload, print selected demos and optimized instruction."""
    # YOUR CODE HERE
    raise NotImplementedError

def main():
    print("=== Project 27: DSPy Optimizer ===\n")
    train_set, test_set = build_datasets()
    results = run_comparison(train_set, test_set)
    best = max(results, key=results.get)
    print(f"\nBest: {best} at {results[best]:.1%}")

if __name__ == "__main__":
    main()
