# Project 24 — Fine-Tuning Agent (QLoRA / DPO)

> **Stack**: Unsloth · HuggingFace PEFT · TRL · Datasets · vLLM · RAGAS  
> **Phase 7 — Advanced Production** | Priority: P0 🔴

---

## What You'll Build

A complete fine-tuning pipeline that teaches `llama-3.2-3B-Instruct` to classify compliance documents as accurately as GPT-4o, at **~5% of the inference cost**.

Pipeline:
```
GPT-4o-mini (teacher)
     ↓ generates
Synthetic dataset (500 labeled examples)
     ↓ trains
llama-3.2-3B via QLoRA (4-bit, LoRA r=16)
     ↓ evaluated
RAGAS quality metrics + cost comparison
     ↓ served via
vLLM OpenAI-compatible endpoint
     ↓ used by
Production compliance agent (drop-in LiteLLM swap)
```

---

## Why This Matters

| Metric | GPT-4o-mini | Fine-tuned llama-3.2-3B |
|---|---|---|
| Cost per 1M tokens | ~$0.30 | ~$0.015 (self-hosted) |
| Latency (P50) | ~800ms | ~180ms (vLLM) |
| Monthly cost at 10k docs/day | ~$900 | ~$45 |
| Accuracy on your domain | 78% | 91% (after fine-tune) |

---

## Milestones

### Milestone 1 — Synthetic Dataset Generation
Use GPT-4o-mini as a teacher model to generate 500 labeled training examples. Each example: document excerpt + document type → risk classification + reasoning. Split 80/10/10 train/val/test.

### Milestone 2 — QLoRA Setup
Load `unsloth/Llama-3.2-3B-Instruct` in 4-bit. Add LoRA adapters to attention + MLP layers. Print trainable parameter count (target: < 1% of total).

### Milestone 3 — SFT Training
Train with `SFTTrainer` for 3 epochs. Use gradient checkpointing. Target: eval loss < 0.5. Save best checkpoint.

### Milestone 4 — DPO Alignment (optional)
Generate preference pairs (chosen vs rejected) for 50 edge cases. Fine-tune further with DPO to align output format and reasoning quality.

### Milestone 5 — Evaluation
Run RAGAS metrics on the test set. Compare: baseline GPT-4o-mini vs fine-tuned model. Target: ≥ 85% exact accuracy on risk level classification.

### Milestone 6 — vLLM Serving
Serve the fine-tuned adapter with vLLM as an OpenAI-compatible endpoint. Verify: `litellm.completion(model="openai/compliance-ft", api_base="http://localhost:8001/v1", ...)` works identically to the original model.

### Milestone 7 — Cost Report
Generate a before/after cost comparison showing monthly savings at production volume.

---

## Setup

```bash
cd projects/project24_finetune_agent
pip install unsloth[colab-new] trl transformers datasets pydantic litellm ragas vllm
# GPU required for training (Google Colab free tier works)
# CPU-only: use Milestone 5 + 6 only (skip training, use pre-trained adapter)
```

---

## File Structure

```
project24_finetune_agent/
├── README.md
├── starter/
│   ├── requirements.txt
│   ├── ex1_generate_dataset.py    — TODOs 1-4: synthetic data generation
│   └── ex2_train_and_serve.py     — TODOs 5-9: QLoRA training + vLLM serving
└── solution/
    ├── generate_dataset.py
    ├── train_qlora.py
    ├── evaluate.py
    └── serve.py
```

---

## Expected Output

```
=== Fine-Tuning Results ===

Dataset: 400 train / 50 val / 50 test examples

Training:
  Epoch 1: train_loss=1.23, eval_loss=0.89
  Epoch 2: train_loss=0.67, eval_loss=0.54
  Epoch 3: train_loss=0.41, eval_loss=0.38 ← best

Evaluation (test set, 50 examples):
  Baseline (gpt-4o-mini):  78% exact accuracy, avg latency 820ms
  Fine-tuned llama-3.2-3B: 91% exact accuracy, avg latency 180ms

Cost Comparison (10,000 docs/day):
  Baseline:     $27.00/day  →  $810/month
  Fine-tuned:    $1.35/day  →   $40.5/month
  Monthly savings: $769.50 (95.0%)
```
