"""
SOLUTION — Exercise 1: QLoRA Fine-Tuning Pipeline for Compliance Classification
Phase 7 / Week 13

How this solution works:
  TODO 1: Prompt GPT-4o-mini with high temperature to generate diverse labeled examples.
           Use asyncio.gather for parallel generation (100× faster than sequential).
  TODO 2: Unsloth's FastLanguageModel.from_pretrained with load_in_4bit=True loads
           the full model quantised to 4-bit, fitting on a single 24GB GPU.
  TODO 3: FastLanguageModel.get_peft_model adds low-rank (r=16) adapter matrices to
           every attention projection and MLP gate — only 0.5% of parameters are trained.
  TODO 4: SFTTrainer handles chat-format data, gradient accumulation, and checkpointing.
  TODO 5: Evaluate accuracy against ground truth labels; track exact + ±1 adjacent.
  TODO 6: Load saved adapter, enable 2× speed inference with FastLanguageModel.for_inference.
  TODO 7: Print cost table — fine-tuned 3B model is ~95% cheaper than GPT-4o-mini.
"""
from __future__ import annotations
import os, json, asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Types (same as starter) ───────────────────────────────────────────────────

class ComplianceLabel(BaseModel):
    document: str
    document_type: Literal["contract", "policy", "invoice", "agreement", "report"]
    risk_level: Literal["low", "medium", "high", "critical"]
    reason: str = Field(description="One sentence explaining the classification")

@dataclass
class TrainingExample:
    document: str
    document_type: str
    risk_level: str
    reason: str

    def to_chat_messages(self) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a compliance classifier. "
                    "Given a business document, classify its risk level as: "
                    "low, medium, high, or critical. "
                    "Respond with JSON: {\"risk_level\": \"...\", \"reason\": \"...\"}"
                ),
            },
            {
                "role": "user",
                "content": f"Document type: {self.document_type}\n\n{self.document}",
            },
            {
                "role": "assistant",
                "content": json.dumps({"risk_level": self.risk_level, "reason": self.reason}),
            },
        ]


# ── TODO 1 SOLUTION: Generate synthetic training dataset ─────────────────────

_GENERATION_PROMPT = """\
Generate ONE synthetic business compliance document example for training a classifier.

Requirements:
- document: 100-200 word realistic excerpt from a business document
- document_type: one of [contract, policy, invoice, agreement, report]
- risk_level: one of [low, medium, high, critical]
- reason: one sentence explaining the specific risk

Risk level guidelines:
  low: standard terms, no missing clauses, routine document
  medium: minor issues — unclear terms, missing standard clauses
  high: significant issues — missing DPA, lacking mandatory disclosures, ambiguous liability
  critical: severe — missing SOX controls, unencrypted PII processing, GDPR violations

Vary document types and industries. Return ONLY valid JSON with the four fields."""

async def generate_one_example() -> TrainingExample:
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": _GENERATION_PROMPT}],
        response_format={"type": "json_object"},
        temperature=0.9,   # high temperature → diverse examples
    )
    data = json.loads(resp.choices[0].message.content)
    label = ComplianceLabel(**data)
    return TrainingExample(
        document=label.document,
        document_type=label.document_type,
        risk_level=label.risk_level,
        reason=label.reason,
    )

async def generate_dataset(n: int = 100) -> tuple[list[TrainingExample], list[TrainingExample]]:
    # Gather all n examples concurrently — ~10s instead of 100s sequential
    examples: list[TrainingExample] = list(
        await asyncio.gather(*[generate_one_example() for _ in range(n)])
    )
    split = int(n * 0.9)
    train, test = examples[:split], examples[split:]
    print(f"Generated {len(train)} train + {len(test)} test examples")
    return train, test


# ── TODO 2 SOLUTION: Load model in QLoRA ─────────────────────────────────────

def load_qlora_model(model_name: str = "unsloth/Llama-3.2-3B-Instruct"):
    # Requires GPU. On CPU/Mac: use "unsloth/Llama-3.2-1B-Instruct" and load_in_4bit=False
    from unsloth import FastLanguageModel  # type: ignore
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,         # auto: float16 on Ampere GPUs, bfloat16 on newer
        load_in_4bit=True,  # QLoRA: base weights in 4-bit NF4 format
    )
    print(f"Base model loaded: {model_name}")
    return model, tokenizer


# ── TODO 3 SOLUTION: Add LoRA adapters ───────────────────────────────────────

def add_lora_adapters(model):
    from unsloth import FastLanguageModel  # type: ignore
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention projections
            "gate_proj", "up_proj", "down_proj",        # MLP layers
        ],
        lora_alpha=16,                    # scaling = alpha/r; keep equal to r
        lora_dropout=0,                   # 0 is optimal per Unsloth benchmarks
        bias="none",
        use_gradient_checkpointing="unsloth",   # saves ~30% more VRAM
        random_state=42,
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = trainable / total * 100
    print(f"LoRA added — trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
    return model


# ── TODO 4 SOLUTION: Train with SFTTrainer ────────────────────────────────────

def train_model(model, tokenizer, train_examples: list[TrainingExample], test_examples: list[TrainingExample]):
    from unsloth.chat_templates import get_chat_template   # type: ignore
    from datasets import Dataset                            # type: ignore
    from trl import SFTTrainer, TrainingArguments           # type: ignore

    # Apply the Llama-3 chat template to the tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    def to_text(examples):
        """Apply chat template to convert messages list → single training string."""
        return {
            "text": [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                for msgs in examples["messages"]
            ]
        }

    train_ds = Dataset.from_list(
        [{"messages": ex.to_chat_messages()} for ex in train_examples]
    ).map(to_text, batched=True)

    val_ds = Dataset.from_list(
        [{"messages": ex.to_chat_messages()} for ex in test_examples]
    ).map(to_text, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,   # packing=True for faster training with short sequences
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,   # effective batch = 8
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            output_dir="./compliance_ft",
            logging_steps=10,
            warmup_ratio=0.05,
        ),
    )

    print("Starting training (3 epochs)...")
    stats = trainer.train()
    print(f"Training done — eval loss: {stats.metrics.get('eval_loss', 'n/a'):.4f}")

    trainer.save_model("./compliance_ft/final")
    tokenizer.save_pretrained("./compliance_ft/final")
    print("Adapter saved to ./compliance_ft/final")
    return stats


# ── TODO 5 SOLUTION: Evaluate before/after ───────────────────────────────────

RISK_ORDER = ["low", "medium", "high", "critical"]

async def evaluate_model(
    test_examples: list[TrainingExample],
    model_endpoint: str | None = None,
) -> dict:
    async def eval_one(ex: TrainingExample) -> str:
        try:
            kwargs: dict = dict(
                model="openai/gpt-4o-mini" if model_endpoint is None else "openai/compliance-ft",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a compliance classifier. "
                            "Return JSON: {\"risk_level\": \"low|medium|high|critical\", \"reason\": \"...\"}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Document type: {ex.document_type}\n\n{ex.document}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            if model_endpoint:
                kwargs["api_base"] = model_endpoint
            resp = await litellm.acompletion(**kwargs)
            pred = json.loads(resp.choices[0].message.content).get("risk_level", "").lower()
            if pred == ex.risk_level:
                return "exact"
            if pred in RISK_ORDER and ex.risk_level in RISK_ORDER:
                if abs(RISK_ORDER.index(pred) - RISK_ORDER.index(ex.risk_level)) <= 1:
                    return "adjacent"
            return "wrong"
        except Exception as e:
            print(f"  eval error: {e}")
            return "error"

    results = list(await asyncio.gather(*[eval_one(ex) for ex in test_examples]))
    n = len(test_examples)
    exact = sum(1 for r in results if r == "exact")
    adjacent = sum(1 for r in results if r in ("exact", "adjacent"))
    errors = sum(1 for r in results if r == "error")
    return {
        "exact_accuracy": exact / n,
        "adjacent_accuracy": adjacent / n,
        "total": n,
        "errors": errors,
    }


# ── TODO 6 SOLUTION: Inference with saved adapter ────────────────────────────

def load_and_infer(adapter_path: str, document: str, document_type: str) -> dict:
    from unsloth import FastLanguageModel   # type: ignore
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        adapter_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)   # enables 2× speed via compiled kernels

    messages = [
        {
            "role": "system",
            "content": "You are a compliance classifier. Return JSON: {\"risk_level\": \"...\", \"reason\": \"...\"}",
        },
        {"role": "user", "content": f"Document type: {document_type}\n\n{document}"},
    ]
    import torch
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100, use_cache=True)

    # Decode only the newly generated tokens (skip the prompt tokens)
    generated = outputs[0][inputs.shape[-1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return json.loads(decoded)


# ── TODO 7 SOLUTION: Cost comparison table ───────────────────────────────────

def print_cost_comparison(n_docs_per_month: int = 10_000):
    models = [
        ("GPT-4o (production)",      0.0025,  "~1200ms"),
        ("GPT-4o-mini (baseline)",   0.00060, "~800ms"),
        ("Fine-tuned Llama-3.2-3B",  0.00003, "~340ms"),
    ]
    header = f"{'Model':<30} {'$/call':>9} {'Monthly (10k docs)':>22} {'Latency':>10}"
    sep = "-" * 76
    print(f"\n{header}\n{sep}")
    for name, cost, latency in models:
        monthly = cost * n_docs_per_month
        print(f"{name:<30} {cost:>9.5f} {'$' + f'{monthly:.2f}':>22} {latency:>10}")
    print(sep)
    baseline_m = 0.00060 * n_docs_per_month
    ft_m = 0.00003 * n_docs_per_month
    savings = (baseline_m - ft_m) / baseline_m * 100
    print(f"Fine-tune vs GPT-4o-mini: {savings:.0f}% cost reduction | ${baseline_m - ft_m:.2f}/month saved\n")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== QLoRA Fine-Tuning Exercise — SOLUTION ===\n")

    print("1. Generating synthetic dataset (100 examples)...")
    train_data, test_data = await generate_dataset(n=100)

    print("\n2. Loading QLoRA model...")
    model, tokenizer = load_qlora_model()
    model = add_lora_adapters(model)

    print("\n3. Training (3 epochs)...")
    train_model(model, tokenizer, train_data, test_data)

    print("\n4. Evaluating baseline (gpt-4o-mini)...")
    baseline = await evaluate_model(test_data, model_endpoint=None)
    print(f"   Baseline exact accuracy:   {baseline['exact_accuracy']:.1%}")
    print(f"   Baseline adjacent accuracy:{baseline['adjacent_accuracy']:.1%}")

    print("\n5. Evaluating fine-tuned model (requires vLLM at localhost:8001)...")
    print("   Start vLLM: python -m vllm.entrypoints.openai.api_server \\")
    print("               --model ./compliance_ft/final --port 8001")
    # Uncomment when vLLM server is running:
    # ft = await evaluate_model(test_data, model_endpoint="http://localhost:8001/v1")
    # print(f"   Fine-tuned exact accuracy:  {ft['exact_accuracy']:.1%}")

    print("\n6. Cost comparison:")
    print_cost_comparison()

if __name__ == "__main__":
    asyncio.run(main())
