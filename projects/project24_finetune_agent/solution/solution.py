"""
Project 24 SOLUTION — Fine-Tuning Agent
Complete QLoRA + DPO pipeline for compliance classification.

Architecture:
  1. generate_dataset()     → GPT-4o-mini generates 500 labeled examples (teacher model)
  2. setup_qlora()          → Llama-3.2-3B in 4-bit + LoRA adapters (0.5% params)
  3. train()                → SFTTrainer 3 epochs, saves best checkpoint
  4. generate_preference_pairs() → GPT-4o-mini generates chosen/rejected pairs for DPO
  5. run_dpo()              → DPOTrainer aligns output format and reasoning quality
  6. evaluate()             → compare GPT-4o-mini baseline vs fine-tuned model
  7. serve via vLLM         → OpenAI-compatible endpoint at localhost:8001
"""
from __future__ import annotations
import os, json, asyncio
from pathlib import Path
import litellm
from dotenv import load_dotenv

load_dotenv()

RISK_LEVELS = ["low", "medium", "high", "critical"]
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# ── 1: Dataset Generation ─────────────────────────────────────────────────────

_GEN_PROMPT = """\
Generate ONE synthetic business compliance document example for training a classifier.
Return ONLY valid JSON with exactly these fields:
{
  "document": "<100-200 word realistic business document excerpt>",
  "document_type": "<one of: contract, policy, invoice, agreement, report>",
  "risk_level": "<one of: low, medium, high, critical>",
  "reason": "<one sentence explaining the risk level>"
}
Vary document type, industry, and risk level. Be specific about which clause or requirement causes the risk."""

async def _generate_one() -> dict:
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": _GEN_PROMPT}],
        response_format={"type": "json_object"},
        temperature=0.9,
    )
    return json.loads(resp.choices[0].message.content)

async def generate_dataset(n: int = 500) -> tuple[list, list]:
    print(f"  Generating {n} examples concurrently...")
    examples = list(await asyncio.gather(*[_generate_one() for _ in range(n)]))

    split = int(n * 0.9)
    train_data, test_data = examples[:split], examples[split:]

    # Save to JSONL
    with open(DATA_DIR / "train.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    with open(DATA_DIR / "test.jsonl", "w") as f:
        for ex in test_data:
            f.write(json.dumps(ex) + "\n")

    print(f"  Saved {len(train_data)} train + {len(test_data)} test to ./data/")
    return train_data, test_data


# ── 2: QLoRA Setup ────────────────────────────────────────────────────────────

def setup_qlora(model_name: str = "unsloth/Llama-3.2-3B-Instruct"):
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
    return model, tokenizer


# ── 3: SFT Training ───────────────────────────────────────────────────────────

def _to_messages(ex: dict) -> list[dict]:
    return [
        {"role": "system", "content": "Classify compliance risk. Return JSON: {\"risk_level\": \"low|medium|high|critical\", \"reason\": \"...\"}"},
        {"role": "user", "content": f"Document type: {ex['document_type']}\n\n{ex['document']}"},
        {"role": "assistant", "content": json.dumps({"risk_level": ex["risk_level"], "reason": ex["reason"]})},
    ]

def train(model, tokenizer, train_data: list, val_data: list):
    from unsloth.chat_templates import get_chat_template  # type: ignore
    from datasets import Dataset                            # type: ignore
    from trl import SFTTrainer, TrainingArguments           # type: ignore

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    def apply_template(batch):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]}

    train_ds = Dataset.from_list([{"messages": _to_messages(ex)} for ex in train_data]).map(apply_template, batched=True)
    val_ds = Dataset.from_list([{"messages": _to_messages(ex)} for ex in val_data]).map(apply_template, batched=True)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        dataset_text_field="text", max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4, fp16=True,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True,
            output_dir="./compliance_ft",
        ),
    )
    result = trainer.train()
    trainer.save_model("./compliance_ft/final")
    print(f"  Saved to ./compliance_ft/final | eval_loss={result.metrics.get('eval_loss', '?'):.4f}")
    return result.metrics


# ── 4: DPO Preference Pairs ───────────────────────────────────────────────────

_DPO_PROMPT = """\
Generate a DPO training pair for a compliance document classifier.
Return JSON:
{
  "prompt": "<100-word contract excerpt with a compliance issue>",
  "document_type": "contract",
  "chosen": "<correct JSON response: {\"risk_level\": \"high\", \"reason\": \"specific regulatory violation\"}>",
  "rejected": "<incorrect response: vague, wrong level, or missing reason>"
}"""

async def generate_preference_pairs(n: int = 50) -> list:
    print(f"  Generating {n} preference pairs...")
    pairs = list(await asyncio.gather(*[
        litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": _DPO_PROMPT}],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        for _ in range(n)
    ]))
    return [json.loads(p.choices[0].message.content) for p in pairs]

def run_dpo(model, tokenizer, preference_data: list):
    from datasets import Dataset  # type: ignore
    from trl import DPOTrainer, DPOConfig  # type: ignore

    ds = Dataset.from_list([{
        "prompt": f"Document type: {p.get('document_type', 'contract')}\n\n{p['prompt']}",
        "chosen": p["chosen"],
        "rejected": p["rejected"],
    } for p in preference_data])

    trainer = DPOTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds,
        args=DPOConfig(
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=5e-5,
            output_dir="./compliance_dpo",
        ),
    )
    trainer.train()
    trainer.save_model("./compliance_dpo/final")
    print("  DPO training complete. Saved to ./compliance_dpo/final")


# ── 5: Evaluation ─────────────────────────────────────────────────────────────

async def evaluate(test_data: list, model_endpoint: str | None = None) -> dict:
    async def eval_one(ex: dict) -> str:
        try:
            kwargs: dict = dict(
                model="openai/gpt-4o-mini" if model_endpoint is None else "openai/compliance-ft",
                messages=[
                    {"role": "system", "content": "Classify compliance risk. Return JSON: {\"risk_level\": \"...\"}"},
                    {"role": "user", "content": f"Document type: {ex['document_type']}\n\n{ex['document']}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            if model_endpoint:
                kwargs["api_base"] = model_endpoint
            resp = await litellm.acompletion(**kwargs)
            pred = json.loads(resp.choices[0].message.content).get("risk_level", "").lower()
            if pred == ex["risk_level"]:
                return "exact"
            if pred in RISK_LEVELS and ex["risk_level"] in RISK_LEVELS:
                if abs(RISK_LEVELS.index(pred) - RISK_LEVELS.index(ex["risk_level"])) <= 1:
                    return "adjacent"
            return "wrong"
        except Exception:
            return "error"

    results = list(await asyncio.gather(*[eval_one(ex) for ex in test_data]))
    n = len(results)
    return {
        "exact": sum(1 for r in results if r == "exact") / n,
        "adjacent": sum(1 for r in results if r in ("exact", "adjacent")) / n,
        "n": n,
    }


# ── 6: vLLM Serving (instructions) ───────────────────────────────────────────

def print_serving_instructions():
    print("""
vLLM Serving Instructions:
  1. Install: pip install vllm
  2. Start server:
       python -m vllm.entrypoints.openai.api_server \\
         --model ./compliance_ft/final \\
         --port 8001

  3. Test with litellm:
       import litellm
       resp = litellm.completion(
           model="openai/compliance-ft",
           api_base="http://localhost:8001/v1",
           messages=[{"role": "user", "content": "Classify: missing DPA clause in vendor contract"}]
       )

  4. Cost comparison:
     GPT-4o-mini:     ~$0.00060/call → $6.00/month (10k docs)
     Fine-tuned 3B:   ~$0.00003/call → $0.30/month (10k docs)
     Savings: 95% cost reduction, 2.4× lower latency
""")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 24: Fine-Tuning Agent SOLUTION ===\n")

    print("Step 1: Generating dataset (n=20 for demo, use 500 for production)...")
    train_data, test_data = await generate_dataset(n=20)

    print("\nStep 2: Setting up QLoRA model (requires GPU)...")
    print("  [Skipping on CPU — uncomment to run on GPU]")
    # model, tokenizer = setup_qlora()

    print("\nStep 3: Training (requires GPU)...")
    # train(model, tokenizer, train_data, test_data)

    print("\nStep 4: Generating DPO preference pairs...")
    # pairs = await generate_preference_pairs(n=10)

    print("\nStep 5: DPO training (requires GPU)...")
    # run_dpo(model, tokenizer, pairs)

    print("\nStep 6: Evaluating baseline (gpt-4o-mini)...")
    baseline = await evaluate(test_data[:5])
    print(f"  Baseline exact accuracy: {baseline['exact']:.1%}")
    print(f"  Baseline adjacent accuracy: {baseline['adjacent']:.1%}")

    print_serving_instructions()

if __name__ == "__main__":
    asyncio.run(main())
