"""
Project 24 — Fine-Tuning Agent: Starter File
QLoRA fine-tuning pipeline for compliance classification.

pip install unsloth[colab-new] trl transformers datasets pydantic litellm ragas python-dotenv

Complete the TODOs below. Reference: phase7_advanced_production/week13_finetune_memory/
"""
from __future__ import annotations
import os, json, asyncio
from pathlib import Path
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── TODO 1: Synthetic Dataset Generation ─────────────────────────────────────
# Use litellm.acompletion with gpt-4o-mini to generate 500 training examples.
# Each example: {document, document_type, risk_level, reason}
# Run concurrently with asyncio.gather for speed.
# Save to ./data/train.jsonl and ./data/test.jsonl (90/10 split)

async def generate_dataset(n: int = 500) -> tuple[list, list]:
    """TODO 1: Generate n labeled training examples using GPT-4o-mini as teacher."""
    # YOUR CODE HERE
    raise NotImplementedError

# ── TODO 2: QLoRA Model Setup ─────────────────────────────────────────────────
# from unsloth import FastLanguageModel
# Load "unsloth/Llama-3.2-3B-Instruct" in 4-bit (load_in_4bit=True)
# Add LoRA adapters: r=16, lora_alpha=16, target attention + MLP modules
# Print trainable parameter count

def setup_qlora(model_name: str = "unsloth/Llama-3.2-3B-Instruct"):
    """TODO 2: Load model in 4-bit + add LoRA adapters. Return (model, tokenizer)."""
    # YOUR CODE HERE
    raise NotImplementedError

# ── TODO 3: Training Pipeline ─────────────────────────────────────────────────
# Apply chat template to tokenizer
# Convert examples to Dataset with "messages" field
# SFTTrainer: 3 epochs, lr=2e-4, gradient_accumulation_steps=4
# Save best checkpoint to ./compliance_ft/final

def train(model, tokenizer, train_data: list, val_data: list):
    """TODO 3: Run SFTTrainer and save model. Return trainer stats."""
    # YOUR CODE HERE
    raise NotImplementedError

# ── TODO 4: DPO Alignment ─────────────────────────────────────────────────────
# Generate 50 preference pairs (chosen vs rejected outputs)
# Use DPOTrainer with beta=0.1
# This step aligns output format and reasoning quality

async def generate_preference_pairs(n: int = 50) -> list:
    """TODO 4: Generate (prompt, chosen, rejected) pairs for DPO. Return list of dicts."""
    # YOUR CODE HERE
    raise NotImplementedError

def run_dpo(model, tokenizer, preference_data: list):
    """TODO 4 (cont): Run DPO training on preference pairs."""
    # YOUR CODE HERE
    raise NotImplementedError

# ── TODO 5: Evaluation ────────────────────────────────────────────────────────
# Evaluate on test set: exact accuracy + adjacent accuracy
# Compare baseline (gpt-4o-mini) vs fine-tuned model
# Use RAGAS if available for additional metrics

async def evaluate(
    test_data: list,
    model_endpoint: str | None = None,  # None = use gpt-4o-mini baseline
) -> dict:
    """TODO 5: Evaluate accuracy. Return {"exact": float, "adjacent": float, "n": int}"""
    # YOUR CODE HERE
    raise NotImplementedError

# ── TODO 6: vLLM Serving + LiteLLM Integration ───────────────────────────────
# Document how to: python -m vllm.entrypoints.openai.api_server --model ./compliance_ft/final
# Then use litellm with api_base="http://localhost:8001/v1"

def print_serving_instructions(adapter_path: str = "./compliance_ft/final"):
    """TODO 6: Print instructions for serving the fine-tuned model with vLLM."""
    # YOUR CODE HERE
    raise NotImplementedError

# ── TODO 7: Cost Comparison Report ───────────────────────────────────────────
# Print a table comparing: cost/call, latency P50, monthly cost at 10k docs/day
# Use real litellm cost tracking numbers if available

def cost_comparison_report(n_docs_per_day: int = 10_000):
    """TODO 7: Print before/after cost comparison table."""
    # YOUR CODE HERE
    raise NotImplementedError

# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 24: Fine-Tuning Agent ===\n")
    train_data, test_data = await generate_dataset(n=500)
    print(f"Dataset: {len(train_data)} train | {len(test_data)} test")
    model, tokenizer = setup_qlora()
    train(model, tokenizer, train_data, test_data[:50])
    baseline = await evaluate(test_data)
    print(f"Baseline accuracy: {baseline['exact']:.1%}")
    cost_comparison_report()

if __name__ == "__main__":
    asyncio.run(main())
