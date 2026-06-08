"""
Exercise 1: QLoRA Fine-Tuning Pipeline for Compliance Classification
Phase 7 / Week 13 — Fine-Tuning + Long-Term Memory

Goal: Build a complete QLoRA fine-tuning pipeline to teach llama-3.2-3B-Instruct
      to classify compliance documents as well as GPT-4o at ~5% of the inference cost.

Stack: unsloth · transformers · trl · datasets · pydantic · litellm

pip install unsloth[colab-new] transformers trl datasets pydantic litellm python-dotenv

TODOs:
  1. Generate a synthetic training dataset using GPT-4o-mini (100 examples)
  2. Load llama-3.2-3B in 4-bit QLoRA with Unsloth
  3. Configure LoRA adapters (r=16, target attention + MLP layers)
  4. Train with SFTTrainer for 3 epochs
  5. Evaluate before vs after on a held-out test set
  6. Save adapter and show how to load for inference
  7. BONUS: Compare latency and cost: fine-tuned 3B vs GPT-4o-mini baseline
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

# ── Types ────────────────────────────────────────────────────────────────────

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
        """Convert to chat format for SFTTrainer."""
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

# ── TODO 1: Generate synthetic training dataset ───────────────────────────────

async def generate_one_example() -> TrainingExample:
    """
    TODO 1: Use litellm.acompletion with gpt-4o-mini to generate ONE training example.

    Prompt the model to return JSON with fields:
      - document: str (100-200 words, realistic business document excerpt)
      - document_type: one of [contract, policy, invoice, agreement, report]
      - risk_level: one of [low, medium, high, critical]
      - reason: str (one sentence explanation)

    Use response_format={"type": "json_object"}.
    Parse the JSON and return a TrainingExample.
    """
    # TODO 1: implement here
    raise NotImplementedError

async def generate_dataset(n: int = 100) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """
    Generate n examples concurrently and split 90/10 train/test.

    TODO: Call generate_one_example() n times concurrently using asyncio.gather,
          then split: first 90% = train, last 10% = test.
    """
    # TODO: implement concurrent generation + split
    raise NotImplementedError

# ── TODO 2: Load model in QLoRA ───────────────────────────────────────────────

def load_qlora_model(model_name: str = "unsloth/Llama-3.2-3B-Instruct"):
    """
    TODO 2: Load the model and tokenizer using Unsloth's FastLanguageModel.

    from unsloth import FastLanguageModel
    import torch

    Call FastLanguageModel.from_pretrained with:
      - model_name=model_name
      - max_seq_length=2048
      - dtype=None (auto-detect)
      - load_in_4bit=True (QLoRA)

    Return (model, tokenizer).

    Note: requires GPU. If running on CPU/Mac, set load_in_4bit=False and
          use the smaller model: "unsloth/Llama-3.2-1B-Instruct"
    """
    # TODO 2: implement here
    raise NotImplementedError

# ── TODO 3: Add LoRA adapters ─────────────────────────────────────────────────

def add_lora_adapters(model):
    """
    TODO 3: Use FastLanguageModel.get_peft_model to add LoRA adapters.

    Parameters:
      - r=16 (rank)
      - target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
      - lora_alpha=16
      - lora_dropout=0
      - bias="none"
      - use_gradient_checkpointing="unsloth"

    Print the number of trainable parameters and the % of total.
    Return the peft model.
    """
    # TODO 3: implement here
    raise NotImplementedError

# ── TODO 4: Train with SFTTrainer ─────────────────────────────────────────────

def train_model(model, tokenizer, train_examples: list[TrainingExample], test_examples: list[TrainingExample]):
    """
    TODO 4: Set up and run SFTTrainer.

    Steps:
    a) Apply chat template to tokenizer:
       from unsloth.chat_templates import get_chat_template
       tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    b) Convert examples to Dataset:
       from datasets import Dataset
       train_data = Dataset.from_list([{"messages": ex.to_chat_messages()} for ex in train_examples])
       test_data  = Dataset.from_list([{"messages": ex.to_chat_messages()} for ex in test_examples])

    c) Create SFTTrainer with TrainingArguments:
       - per_device_train_batch_size=2
       - gradient_accumulation_steps=4
       - num_train_epochs=3
       - learning_rate=2e-4
       - eval_strategy="epoch"
       - output_dir="./compliance_ft"

    d) Run trainer.train() and print final eval loss.
    e) Save model to "./compliance_ft/final"
    """
    # TODO 4: implement here
    raise NotImplementedError

# ── TODO 5: Evaluate before/after ─────────────────────────────────────────────

async def evaluate_model(
    test_examples: list[TrainingExample],
    model_endpoint: str | None = None,  # None = use base model baseline
) -> dict:
    """
    TODO 5: Evaluate accuracy on test_examples.

    If model_endpoint is None: use litellm with gpt-4o-mini as the baseline
    If model_endpoint is set: call the fine-tuned model served at that endpoint
       (use litellm with model="openai/compliance-ft", api_base=model_endpoint)

    For each test example:
      - Send document + document_type to the model
      - Parse the JSON response for risk_level
      - Compare to test_example.risk_level
      - Track exact matches and adjacent-level matches (±1)

    Return:
    {
        "exact_accuracy": float,
        "adjacent_accuracy": float,  # exact or off-by-one level
        "total": int,
        "errors": int  # unparseable responses
    }
    """
    # TODO 5: implement here
    raise NotImplementedError

# ── TODO 6: Inference with saved adapter ──────────────────────────────────────

def load_and_infer(adapter_path: str, document: str, document_type: str) -> dict:
    """
    TODO 6: Load the saved LoRA adapter and run inference.

    Steps:
    a) Load with FastLanguageModel.from_pretrained(adapter_path, load_in_4bit=True)
    b) Apply get_chat_template
    c) Enable 2x speed with FastLanguageModel.for_inference(model)
    d) Tokenize the user message using the chat template
    e) Generate with max_new_tokens=100
    f) Decode and parse the JSON risk_level + reason
    g) Return {"risk_level": ..., "reason": ...}
    """
    # TODO 6: implement here
    raise NotImplementedError

# ── TODO 7 (BONUS): Cost comparison ───────────────────────────────────────────

def print_cost_comparison(n_docs_per_month: int = 10_000):
    """
    TODO 7: Print a cost comparison table showing:

    | Model                  | Cost/call | Monthly (10k docs) | Latency |
    |------------------------|-----------|--------------------|---------|
    | GPT-4o-mini (baseline) | ~$0.0006  | ~$6.00             | ~800ms  |
    | Fine-tuned llama-3.2-3B| ~$0.00003 | ~$0.30             | ~340ms  |
    | Savings                | 95%       | $5.70/mo           | 2.4x    |

    Use actual numbers from your litellm cost tracking if available.
    """
    # TODO 7: implement here
    raise NotImplementedError

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== QLoRA Fine-Tuning Exercise ===\n")

    # Step 1: Generate dataset
    print("1. Generating synthetic dataset (100 examples)...")
    train_data, test_data = await generate_dataset(n=100)
    print(f"   Train: {len(train_data)} | Test: {len(test_data)}")

    # Step 2-3: Load + configure model
    print("2. Loading QLoRA model...")
    model, tokenizer = load_qlora_model()
    model = add_lora_adapters(model)

    # Step 4: Train
    print("3. Training...")
    train_model(model, tokenizer, train_data, test_data)

    # Step 5: Evaluate
    print("4. Evaluating baseline (gpt-4o-mini)...")
    baseline = await evaluate_model(test_data, model_endpoint=None)
    print(f"   Baseline: {baseline['exact_accuracy']:.1%} exact accuracy")

    print("5. Evaluating fine-tuned model (requires vLLM server)...")
    # Start vLLM: python -m vllm.entrypoints.openai.api_server --model ./compliance_ft/final --port 8001
    ft_result = await evaluate_model(test_data, model_endpoint="http://localhost:8001/v1")
    print(f"   Fine-tuned: {ft_result['exact_accuracy']:.1%} exact accuracy")

    # Step 7: Cost comparison
    print_cost_comparison()

if __name__ == "__main__":
    asyncio.run(main())
