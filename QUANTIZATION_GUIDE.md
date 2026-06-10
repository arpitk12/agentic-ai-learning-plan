# Quantization of Models — Explained

## Quick Answer

**Quantization** is reducing the precision of model weights and computations to save memory and speed up inference.

**Example**: Store weights as 8-bit integers instead of 32-bit floats → **4× less memory** with ~1% quality loss.

---

## Why Quantize?

### Problem
A 7B parameter model weights ~28 GB in float32:
```
7 billion params × 4 bytes/param = 28 GB  ← doesn't fit on most GPUs
```

### Solution: Quantization
```
7 billion params × 1 byte/param  = 7 GB   ← INT8 quantization
7 billion params × 0.5 byte/param = 3.5 GB ← 4-bit quantization (QLoRA)
```

### When to Use
- **Running large models locally** → Use quantization (Ollama, llama.cpp, QLoRA)
- **Production inference at scale** → Use quantization to reduce GPU memory
- **Fine-tuning on consumer GPU** → QLoRA (4-bit) + LoRA adapters
- **Cloud API** → Usually don't need quantization (pay per token)

---

## Quantization Types (Covered in This Curriculum)

### 1. **INT8 Scalar Quantization** (Vector Database Scale)
**Where**: [`projects/project17_enterprise_rag/`](projects/project17_enterprise_rag/)  
**Use case**: Storing 50M embedding vectors efficiently

| Aspect | Before | After |
|--------|--------|-------|
| **Format** | float32 | int8 |
| **Bytes per dimension** | 4 | 1 |
| **50M vectors (384 dims) memory** | 72 GB | 18 GB |
| **Quality loss** | — | ~1% recall drop |

**How it works**:
```python
# From Qdrant documentation
from qdrant_client.models import ScalarQuantizationConfig, ScalarType

config = ScalarQuantizationConfig(
    type=ScalarType.INT8,      # Store as 8-bit integers
    quantile=0.99,             # Clip top 1% outliers
    always_ram=True,           # Keep quantized vectors in RAM
)
```

**Math**:
```
1. Get min/max of float32 vector: [-2.5, 3.1]
2. Map range to [-128, 127] (int8 range)
3. Clip outliers: only use values between quantile 1% and 99%
4. Store as bytes; reconstruct via reverse mapping on search
5. Re-score top-k with float32 for precision (hybrid)
```

---

### 2. **4-bit Quantization (QLoRA)** (Fine-Tuning Scale)
**Where**: [`phase7_advanced_production/week13_finetune_memory/`](phase7_advanced_production/week13_finetune_memory/)  
**Use case**: Fine-tuning 7–70B models on consumer GPUs

| Aspect | Full Precision | QLoRA |
|--------|---|---|
| **Base model** | float16 | 4-bit NormalFloat (NF4) |
| **LoRA adapters** | — | float16 (trainable) |
| **Memory for 7B model** | 14 GB | 2–3 GB + 0.5 GB adapters |
| **Training on 24GB GPU** | ❌ | ✅ (RTX 4090 / A100) |

**How QLoRA works**:
```
1. Load base model in 4-bit (frozen, no gradients)
2. Attach small LoRA adapters (trainable, full precision)
3. During backward pass: compute gradients only for adapters
4. Only save/deploy the LoRA weights (~1–10 MB, not 28 GB)
5. At inference: merge adapters into base model or use separately
```

**Example from curriculum**:
```python
# phase7_advanced_production/week13_finetune_memory/notes.md
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,        # Nested quantization
    bnb_4bit_quant_type="nf4",             # NormalFloat-4 format
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
)
```

---

### 3. **GGUF Quantization** (Local Inference)
**Where**: [`LOCAL_LLM_SETUP.md`](LOCAL_LLM_SETUP.md), Ollama  
**Use case**: Running models with llama.cpp locally (CPU/Apple Silicon)

**GGUF Variants**:
```
Q4_K_M  — 4-bit with medium optimization (good balance)
Q4_K_S  — 4-bit with small optimization (smaller file)
Q8_0    — 8-bit (larger, better quality)
Q3_K    — 3-bit (smallest, less accurate)
F16     — full float16 (largest, best quality)
```

**Example**:
```bash
# Run a GGUF model locally
./server -m compliance_classifier.Q4_K_M.gguf -c 2048 --port 8001
```

**Memory usage**:
```
llama2-7b.Q4_K_M.gguf  ≈ 3.8 GB  ← runs on 8 GB RAM
llama2-7b.Q8_0.gguf    ≈ 6.7 GB  ← runs on 12 GB RAM
llama2-7b-f16.gguf     ≈ 13 GB   ← requires 16 GB RAM
```

---

## Quality vs. Speed vs. Memory Trade-off

| Quantization | Memory | Speed | Quality | Use Case |
|---|---|---|---|---|
| **float32** | 1.0x | Baseline | 100% | Training (rare) |
| **float16** | 0.5x | ~1.5x faster | 99.8% | Fine-tuning |
| **bfloat16** | 0.5x | ~1.5x faster | 99.7% | Better numerical stability |
| **INT8** | 0.25x | ~2x faster | 98–99% | Vector search, inference |
| **4-bit (NF4)** | 0.125x | ~3x faster | 96–98% | Fine-tuning on small GPU |
| **GGUF Q4** | 0.125x | ~4x faster | 95–97% | Local inference |
| **GGUF Q3** | 0.09x | ~5x faster | 90–95% | Extreme mobile only |

---

## Where Quantization Is Covered in This Curriculum

### 1. **Vector DB Quantization** (Project 17 — Enterprise RAG)
- **Section**: `projects/project17_enterprise_rag/GUIDE.md` § 1.3
- **Topics**:
  - INT8 scalar quantization math
  - Qdrant implementation with HNSW
  - Recall degradation estimates
  - Re-scoring strategy
- **Code**: `projects/project17_enterprise_rag/solution/src/store/qdrant_store.py`

### 2. **Fine-Tuning Quantization** (Phase 7, Week 13)
- **Section**: `phase7_advanced_production/week13_finetune_memory/notes.md`
- **Topics**:
  - QLoRA + 4-bit quantization
  - Training 7B models on single GPU
  - bfloat16 vs float16 vs NF4
  - Memory calculations
- **Code**: `phase7_advanced_production/week13_finetune_memory/exercises/ex1_qlora_finetune_solution.py`
- **Resources**: Full Hugging Face quantization guide in links.md

### 3. **Local Model Quantization** (LOCAL_LLM_SETUP.md)
- **Ollama models**: Already quantized (Q4, Q8 variants)
- **GGUF format**: For running via llama.cpp
- **Trade-offs**: Memory vs. quality per model

---

## Quantization Techniques by Category

### A. **Uniform Quantization**
Map all values evenly across the range:
```
[-2.5, 3.1] → -128...127  (equally spaced)
```
**Pros**: Fast, simple  
**Cons**: Outliers hurt precision

### B. **Non-Uniform Quantization (NF4, NF8)**
Use a learned distribution that better matches neural net weights:
```
Uses normal distribution mapped to int8 buckets
Better for outlier-heavy data
```
**Pros**: Better quality at 4-bit  
**Cons**: Slightly slower

### C. **Dynamic vs. Static Quantization**
- **Static**: Quantization parameters computed once from training data
- **Dynamic**: Recomputed at runtime per-batch
**Trade-off**: Static faster but less adaptive

### D. **Symmetric vs. Asymmetric**
- **Symmetric**: Map [-max, max] → [-128, 128]
- **Asymmetric**: Map [min, max] → [-128, 128] (tighter fit)
**Trade-off**: Asymmetric more precise but requires offset storage

---

## When NOT to Quantize

❌ **Don't quantize if**:
- You're using a cloud API (costs are already optimized)
- The task is extremely quality-sensitive (medical diagnosis, legal review)
- You have unlimited GPU memory
- Latency doesn't matter (batch processing)
- Model is already optimized for your hardware

✅ **DO quantize if**:
- Running on consumer GPU (24 GB limit)
- Need to fit model in edge device memory
- Batch inference can tolerate 1–3% quality loss
- Cost per token matters (dense retrieval on 50M vectors)
- Training budget is constrained

---

## Quick Reference: Which Quantization to Use?

```
┌─ Are you fine-tuning?
│  ├─ YES → Use QLoRA (4-bit, bitsandbytes)
│  └─ NO → Continue below
│
├─ Are you storing embeddings/vectors?
│  ├─ YES → Use INT8 scalar quantization (Qdrant/Pinecone)
│  └─ NO → Continue below
│
├─ Are you running locally (CPU/Apple Silicon)?
│  ├─ YES → Use GGUF quantization (llama.cpp, Ollama)
│  └─ NO → Continue below
│
├─ Are you doing GPU inference?
│  ├─ Large batch (>10 examples) → Use INT8 (torch.quantization)
│  ├─ Small batch, high throughput → Use float16 (standard)
│  └─ Memory-constrained → Use INT8
│
└─ Cloud API or unlimited resources?
   └─ Just use native precision, don't quantize
```

---

## Resources in This Curriculum

### Code References
- **Vector DB quantization**: `projects/project17_enterprise_rag/solution/src/store/qdrant_store.py`
- **QLoRA fine-tuning**: `phase7_advanced_production/week13_finetune_memory/exercises/ex1_qlora_finetune_solution.py`
- **Ollama (GGUF)**: Models available via `ollama pull llama2` (already quantized)

### Guides
- **Production vector search**: `projects/project17_enterprise_rag/GUIDE.md` § 1.3 (INT8 math)
- **Fine-tuning**: `phase7_advanced_production/week13_finetune_memory/concepts.md`
- **Local setup**: `LOCAL_LLM_SETUP.md`

### Papers
- **QLoRA** (Dettmers et al., 2023): Efficient fine-tuning of quantized LLMs
- **GPTQ** (Frantar et al., 2023): Post-training quantization for LLMs
- **AWQ** (Lin et al., 2023): Activation-aware weight quantization

---

## TL;DR

| Scenario | Quantization Type | Memory Savings | Quality Impact |
|----------|---|---|---|
| Fine-tune 7B on RTX 4090 | QLoRA 4-bit | 4–6x | 96–98% |
| Store 50M embeddings | INT8 vector quantization | 4x | 99% (recall) |
| Run model locally on MacBook | GGUF Q4 | 4x | 95–97% |
| Production inference at scale | INT8 + batch | 4x | 98% |
| No constraints (cloud API) | None (keep native) | — | 100% |
