# Week 13 Resources — Fine-Tuning + Long-Term Memory

---

## 🏁 Start Here (Read in This Order)

1. **Maxime Labonne — LLM Fine-Tuning Overview** (best single intro):
   https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_Unsloth.html

2. **Hugging Face — PEFT / LoRA Conceptual Guide**:
   https://huggingface.co/docs/peft/conceptual_guides/lora

3. **Mem0 Quickstart** (30-min to working memory layer):
   https://docs.mem0.ai/quickstart

---

## LoRA and QLoRA

### Foundational Papers
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021):
  https://arxiv.org/abs/2106.09685
  *The original LoRA paper. Read the first 5 pages — explains the rank decomposition idea clearly.*

- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023):
  https://arxiv.org/abs/2305.14314
  *Introduces 4-bit NormalFloat (NF4) quantization + nested quantization. Appendix B has ablation results.*

### Practical Guides
- **Unsloth GitHub** — fastest QLoRA trainer, 2× speed vs HuggingFace:
  https://github.com/unslothai/unsloth

- **Unsloth Notebooks** — ready-to-run Colab notebooks for Llama 3, Mistral, Gemma:
  https://github.com/unslothai/unsloth?tab=readme-ov-file#-finetune-for-free

- **Hugging Face PEFT Docs** — official LoRA/QLoRA API reference:
  https://huggingface.co/docs/peft/index

- **TRL (SFTTrainer) Docs** — training loop library used in exercises:
  https://huggingface.co/docs/trl/sft_trainer

- **Alpaca-LoRA repo** — original open-source fine-tune of LLaMA (good code reference):
  https://github.com/tloen/alpaca-lora

### Videos
- **Andrej Karpathy — "State of GPT"** (covers fine-tuning pipeline clearly):
  https://youtu.be/bZQun8Y4L2A

- **Maxime Labonne — Fine-tuning Llama 3.1 with Unsloth** (hands-on walkthrough):
  https://youtu.be/5QHUlG5Hkzw

- **DeepLearning.AI — "Finetuning Large Language Models"** (free short course):
  https://www.deeplearning.ai/short-courses/finetuning-large-language-models/

---

## DPO (Direct Preference Optimisation)

### Paper
- **DPO: Direct Preference Optimization** (Rafailov et al., 2023):
  https://arxiv.org/abs/2305.18290
  *Section 3 (just 2 pages) gives the full mathematical intuition. The rest is experiments.*

### Practical Guides
- **HuggingFace TRL — DPOTrainer Docs**:
  https://huggingface.co/docs/trl/dpo_trainer

- **Maxime Labonne — Fine-tune Llama 3 with DPO** (step-by-step with code):
  https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_Unsloth.html

- **Hugging Face Alignment Handbook** — production DPO recipes from the HF team:
  https://github.com/huggingface/alignment-handbook

### Dataset
- **Anthropic HH-RLHF** — the standard human preference dataset used in DPO research:
  https://huggingface.co/datasets/Anthropic/hh-rlhf

---

## Dataset Construction

- **Alpaca dataset format** — the standard instruction fine-tuning data format:
  https://github.com/tatsu-lab/stanford_alpaca#data-release

- **Hugging Face Datasets Docs** — loading and formatting data for training:
  https://huggingface.co/docs/datasets/index

- **Lilac** — data curation and quality filtering tool:
  https://www.lilacml.com

- **Argilla** — open-source annotation platform for building fine-tuning datasets:
  https://docs.argilla.io

---

## vLLM Serving

- **vLLM Project Site + Docs**:
  https://docs.vllm.ai/en/latest/

- **vLLM GitHub**:
  https://github.com/vllm-project/vllm

- **vLLM — Serving LoRA Adapters** (multiple adapters on one base model):
  https://docs.vllm.ai/en/latest/features/lora.html

- **PagedAttention Paper** — the core technique that makes vLLM fast:
  https://arxiv.org/abs/2309.06180

---

## Long-Term Memory with Mem0

### Official Docs
- **Mem0 Documentation**:
  https://docs.mem0.ai

- **Mem0 GitHub**:
  https://github.com/mem0ai/mem0

- **Mem0 Quickstart Notebook**:
  https://docs.mem0.ai/quickstart

### Background Reading
- **Episodic vs Semantic Memory** (cognitive science basis for the Mem0 model):
  https://en.wikipedia.org/wiki/Episodic_memory

- **Generative Agents: Interactive Simulacra of Human Behavior** (Stanford, 2023) — the seminal paper on agent memory that Mem0 builds on:
  https://arxiv.org/abs/2304.03442

- **MemGPT Paper** — another long-term memory architecture (complementary reading):
  https://arxiv.org/abs/2310.08560

---

## Quantisation (Background)

- **Hugging Face — Quantization Concepts Guide**:
  https://huggingface.co/docs/transformers/en/quantization/overview

- **bitsandbytes library** — provides the 4-bit/8-bit quantization used in QLoRA:
  https://github.com/TimDettmers/bitsandbytes

- **GGUF/llama.cpp** — alternative quantization format for local inference (complementary):
  https://github.com/ggerganov/llama.cpp

---

## Key Reference Models and Hubs

- **Hugging Face Open LLM Leaderboard** — compare fine-tuned models:
  https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

- **Unsloth pre-quantized models** — ready-to-fine-tune base models:
  https://huggingface.co/unsloth

- **Llama 3.1 8B Instruct** — the base model used in exercises:
  https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

---

## Blog Posts Worth Reading

- **"Fine-tuning is overrated (mostly)"** — pragmatic view on when NOT to fine-tune:
  https://hamel.dev/blog/posts/fine-tuning/

- **Hamel Husain — Evaluation-driven fine-tuning workflow**:
  https://hamel.dev/notes/llm/finetuning/03_eval_data.html

- **Sebastian Raschka — "Fine-Tuning LLMs: A Practical Guide"**:
  https://sebastianraschka.com/blog/2023/llm-finetuning-llama.html

---

## Tools Checklist

| Tool | Purpose | Install |
|---|---|---|
| `unsloth` | Fast QLoRA trainer | `pip install unsloth` |
| `trl` | SFTTrainer + DPOTrainer | `pip install trl` |
| `peft` | LoRA adapters | `pip install peft` |
| `bitsandbytes` | 4-bit quantization | `pip install bitsandbytes` |
| `vllm` | Production model serving | `pip install vllm` |
| `mem0ai` | Long-term memory layer | `pip install mem0ai` |
| `datasets` | Dataset loading/formatting | `pip install datasets` |
