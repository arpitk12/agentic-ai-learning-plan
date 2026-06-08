# Week 15 — DSPy Prompt Optimization + A/B Testing

## What This Week Is About

1. **DSPy** — compile and auto-optimize your prompts using a dev set, replacing hand-crafted few-shot examples with algorithmically selected ones
2. **A/B testing and model management** — safely roll out new models and prompts in production without breaking existing users

---

## 1. What Is DSPy?

Traditional LLM development:
```
You → write prompt → test → tweak prompt → test → repeat forever
```

DSPy development:
```
You → define Signature (input/output spec) + metric → DSPy optimizes prompts for you
```

DSPy compiles your program into optimized prompts + few-shot examples by searching over the space of possible prompts using your evaluation metric.

**When DSPy wins over hand-crafted prompts:**
- You have a dev set of 50+ examples to optimize against
- Task is well-defined (clear metric)
- You're iterating frequently and want systematic improvement
- You want a reproducible, version-controlled prompt pipeline

---

## 2. DSPy Core Concepts

```python
# pip install dspy-ai
import dspy
from dspy import Signature, InputField, OutputField, ChainOfThought

# Configure LM
lm = dspy.LM("openai/gpt-4o-mini", max_tokens=500)
dspy.configure(lm=lm)

# 1. Signature — declare the task (like a function type signature)
class ComplianceClassifier(Signature):
    """Classify a business document's compliance risk level."""
    document: str = InputField(desc="Business document text to classify")
    document_type: str = InputField(desc="Type: contract, policy, invoice, etc.")
    risk_level: str = OutputField(desc="One of: low, medium, high, critical")
    reasoning: str = OutputField(desc="One sentence explaining the classification")

# 2. Module — wrap a Signature in a reasoning strategy
class ComplianceModule(dspy.Module):
    def __init__(self):
        self.classify = ChainOfThought(ComplianceClassifier)  # adds internal reasoning
    
    def forward(self, document: str, document_type: str) -> dspy.Prediction:
        return self.classify(document=document, document_type=document_type)

# 3. Use (before optimization)
module = ComplianceModule()
result = module(document="This vendor agreement lacks DPA...", document_type="contract")
print(result.risk_level)    # → "high"
print(result.reasoning)     # → "Missing DPA clause violates GDPR Art. 28"
```

---

## 3. Building a Dev Set and Metric

```python
# Dev set: examples where you know the correct answer
dev_set = [
    dspy.Example(
        document="Standard NDA with mutual confidentiality, 2-year term",
        document_type="contract",
        risk_level="low",
    ).with_inputs("document", "document_type"),
    dspy.Example(
        document="Payment processing agreement lacking PCI-DSS compliance statement",
        document_type="contract",
        risk_level="high",
    ).with_inputs("document", "document_type"),
    # ... 50+ examples
]

# Metric: how to score a prediction
def compliance_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Return 1.0 if risk_level matches exactly, 0.5 if adjacent level, 0.0 if far off."""
    levels = ["low", "medium", "high", "critical"]
    gold_idx = levels.index(gold.risk_level)
    try:
        pred_idx = levels.index(pred.risk_level.lower().strip())
    except ValueError:
        return 0.0
    diff = abs(gold_idx - pred_idx)
    return {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}[diff]
```

---

## 4. Optimizing with DSPy Optimizers

```python
from dspy.teleprompt import BootstrapFewShot, MIPROv2

# Optimizer A: BootstrapFewShot — generate + select best few-shot examples
bootstrap = BootstrapFewShot(
    metric=compliance_metric,
    max_bootstrapped_demos=4,  # add up to 4 examples to the prompt
    max_labeled_demos=4,
    max_rounds=1,
)
optimized_module = bootstrap.compile(
    ComplianceModule(),
    trainset=dev_set[:40],
)

# Optimizer B: MIPROv2 — also optimizes the instruction text (stronger but slower)
mipro = MIPROv2(
    metric=compliance_metric,
    auto="medium",         # "light" / "medium" / "heavy" — controls search budget
    num_threads=4,
)
optimized_v2 = mipro.compile(
    ComplianceModule(),
    trainset=dev_set[:40],
    valset=dev_set[40:],
)

# Save for production
optimized_v2.save("compliance_classifier_optimized.json")

# Load in production
production_module = ComplianceModule()
production_module.load("compliance_classifier_optimized.json")
```

---

## 5. Evaluate Before vs After

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=dev_set[40:],   # held-out test set
    metric=compliance_metric,
    num_threads=4,
    display_progress=True,
)

baseline_score = evaluator(ComplianceModule())
optimized_score = evaluator(optimized_v2)

print(f"Baseline:  {baseline_score:.1%}")   # e.g., 71.0%
print(f"Optimized: {optimized_score:.1%}")  # e.g., 88.5%
print(f"Gain:      +{(optimized_score - baseline_score):.1%}")
```

---

## 6. A/B Testing Agent Models in Production

**The problem**: You have a working agent using `gpt-4o-mini`. You want to test `llama-3.1-8B-fine-tuned`. If you swap directly, 100% of traffic hits the new model — any regression affects all users.

**Traffic splitting pattern**:

```python
import hashlib
from enum import Enum

class ModelVariant(str, Enum):
    CONTROL = "openai/gpt-4o-mini"        # existing model
    TREATMENT = "openai/compliance-ft-3b"  # new fine-tuned model

def assign_variant(user_id: str, experiment_id: str, treatment_pct: int = 10) -> ModelVariant:
    """Deterministic assignment: same user always gets same variant."""
    hash_input = f"{experiment_id}:{user_id}".encode()
    bucket = int(hashlib.md5(hash_input).hexdigest(), 16) % 100
    return ModelVariant.TREATMENT if bucket < treatment_pct else ModelVariant.CONTROL

async def agent_call(user_id: str, document: str) -> dict:
    variant = assign_variant(user_id, experiment_id="compliance-v2", treatment_pct=10)
    
    result = await run_agent(document, model=variant.value)
    
    # Log to Langfuse with experiment metadata
    langfuse.trace(
        name="compliance-review",
        metadata={"variant": variant.value, "experiment": "compliance-v2"},
        tags=["ab-test"],
    )
    
    return result
```

---

## 7. Shadow Mode — Zero-Risk Model Testing

Run the new model in parallel, compare outputs, never affect users:

```python
import asyncio

async def shadow_call(user_id: str, document: str) -> dict:
    """Run control + treatment simultaneously. User always gets control response."""
    control_task = asyncio.create_task(
        run_agent(document, model="openai/gpt-4o-mini")
    )
    shadow_task = asyncio.create_task(
        run_agent(document, model="openai/compliance-ft-3b")
    )
    
    control_result, shadow_result = await asyncio.gather(
        control_task, shadow_task, return_exceptions=True
    )
    
    # Log shadow comparison (never shown to user)
    if not isinstance(shadow_result, Exception):
        log_shadow_comparison(
            user_id=user_id,
            control=control_result,
            shadow=shadow_result,
            match=(control_result["risk_level"] == shadow_result["risk_level"]),
        )
    
    return control_result  # user always gets control
```

---

## 8. Statistical Significance Testing

Don't call a winner until you have statistical confidence:

```python
from scipy import stats

def check_significance(
    control_conversions: int, control_total: int,
    treatment_conversions: int, treatment_total: int,
    alpha: float = 0.05,
) -> dict:
    """Chi-square test for conversion rate difference."""
    contingency = [
        [control_conversions, control_total - control_conversions],
        [treatment_conversions, treatment_total - treatment_conversions],
    ]
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    control_rate = control_conversions / control_total
    treatment_rate = treatment_conversions / treatment_total
    
    return {
        "significant": p_value < alpha,
        "p_value": round(p_value, 4),
        "control_rate": round(control_rate, 4),
        "treatment_rate": round(treatment_rate, 4),
        "lift": round((treatment_rate - control_rate) / control_rate, 4),
        "winner": "treatment" if treatment_rate > control_rate and p_value < alpha else "inconclusive",
    }

# Example:
result = check_significance(
    control_conversions=850, control_total=1000,    # 85% accuracy
    treatment_conversions=920, treatment_total=1000, # 92% accuracy
)
# → {"significant": True, "winner": "treatment", "lift": 0.0824}
```

---

## 9. Model Registry with MLflow

```python
import mlflow
from mlflow.tracking import MlflowClient

# Log experiment results
with mlflow.start_run(run_name="compliance-ft-v2"):
    mlflow.log_params({"model": "llama-3.1-8B", "lora_r": 16, "dataset_size": 500})
    mlflow.log_metrics({"accuracy": 0.92, "avg_latency_ms": 340, "cost_per_call": 0.0002})
    mlflow.pyfunc.log_model("model", python_model=ComplianceModel())
    mlflow.set_tag("experiment", "compliance-classifier-v2")

# Register winning model
client = MlflowClient()
client.create_registered_model("compliance-classifier")
client.create_model_version(
    name="compliance-classifier",
    source="runs:/abc123/model",
    run_id="abc123",
)
client.transition_model_version_stage(
    name="compliance-classifier", version=1, stage="Production"
)
```

---

## Key Takeaways

1. **DSPy Signatures**: define input/output spec; DSPy finds the best prompt for it
2. **BootstrapFewShot**: fastest optimizer, adds labeled demos — good starting point
3. **MIPROv2**: stronger, also tunes instruction text — use when you have compute budget
4. **Traffic split**: hash `user_id + experiment_id` → deterministic bucket assignment
5. **Shadow mode**: zero-risk testing — run both, log comparison, user sees control
6. **p < 0.05**: minimum bar before calling a winner; collect at least 500 samples per variant

---

## Exercises

- `ex1_dspy_optimizer.py` — DSPy Signature → BootstrapFewShot → MIPROv2 → before/after eval
- `ex2_traffic_split.py` — Hash-based A/B router + shadow mode + chi-square significance test
