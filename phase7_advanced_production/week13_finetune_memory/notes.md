# Week 13 — Fine-Tuning Agents + Long-Term Memory

## What This Week Is About

Two techniques that slash cost and unlock persistent intelligence:

1. **Fine-tuning** — teach a small open-source model to do your specific task as well as GPT-4o, at 5–20× lower inference cost
2. **Long-term memory** — give your agent a memory that survives across sessions, users, and weeks

---

## 1. When to Fine-Tune (vs RAG vs Prompt Engineering)

| Situation | Best approach |
|---|---|
| Public knowledge, needs retrieval at runtime | **RAG** |
| Need consistent output format / tone | **Prompt engineering first** |
| Task is narrow, repetitive, domain-specific | **Fine-tune** |
| Need to reduce inference cost >50% | **Fine-tune smaller model** |
| Model must learn private data inaccessible at inference | **Fine-tune** |

**Decision rule**: If a human expert reading only the prompt could do the task correctly, prompt engineering wins. If the task requires knowledge baked into weights (style, domain jargon, consistent reasoning traces), fine-tune.

---

## 2. LoRA and QLoRA

**LoRA (Low-Rank Adaptation)** — instead of updating all 7B parameters, add small low-rank matrices (rank r=8..64) to each attention layer and train only those. 99% of parameters frozen.

```
Parameters trained:  ~0.1–1% of total
Memory reduction:    4–8× vs full fine-tune
Quality:             Near-identical to full fine-tune on narrow tasks
```

**QLoRA** — LoRA + 4-bit quantization of the frozen base weights. Train a 7B model on a single 24GB GPU.

```python
# Install
# pip install unsloth[colab-new] transformers datasets trl

from unsloth import FastLanguageModel
import torch

# 1. Load base model in 4-bit (QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    dtype=None,          # auto-detect: float16 on Ampere, bfloat16 on newer
    load_in_4bit=True,   # QLoRA: 4-bit base weights
)

# 2. Add LoRA adapters (only these parameters will be trained)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                              # rank — higher = more capacity, more memory
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,                     # scaling factor (keep = r)
    lora_dropout=0,                    # 0 is optimal per Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% memory reduction
    random_state=42,
)

print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
# → ~41M of 8B total (~0.5%)
```

---

## 3. Building a Fine-Tuning Dataset

Good fine-tuning data is more important than the training loop.

```python
from datasets import Dataset

# Format: list of conversation dicts
def make_example(document: str, classification: str, reasoning: str) -> dict:
    """One training example in chat format."""
    return {
        "messages": [
            {"role": "system", "content": "You are a compliance classifier. Classify documents as low/medium/high/critical risk with a brief reason."},
            {"role": "user", "content": f"Classify this document:\n\n{document}"},
            {"role": "assistant", "content": f'{{"risk_level": "{classification}", "reason": "{reasoning}"}}'},
        ]
    }

# Synthetic data generation using a stronger model
async def generate_synthetic_dataset(n: int = 500) -> list[dict]:
    """Use GPT-4o to generate training examples for llama-3B fine-tune."""
    import litellm
    examples = []
    for _ in range(n):
        prompt = "Generate a realistic business document (contract excerpt, policy statement, or vendor agreement, 100-200 words) and its correct compliance risk classification (low/medium/high/critical) with a one-sentence reason."
        r = await litellm.acompletion(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        data = json.loads(r.choices[0].message.content)
        examples.append(make_example(data["document"], data["risk_level"], data["reason"]))
    return examples

dataset = Dataset.from_list(generate_synthetic_dataset(500))
dataset = dataset.train_test_split(test_size=0.1)
```

---

## 4. Training with SFTTrainer

```python
from trl import SFTTrainer, TrainingArguments
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="messages",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,    # effective batch = 8
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        output_dir="./compliance_classifier",
    ),
)

trainer.train()
trainer.save_model("./compliance_classifier/final")
```

---

## 5. DPO — Direct Preference Optimization

Use DPO when you have preference data (human rated "response A is better than response B").

```python
from trl import DPOTrainer, DPOConfig

# Preference dataset format
preference_data = [
    {
        "prompt": "Classify this vendor agreement...",
        "chosen": '{"risk_level": "high", "reason": "Missing DPA clause, SOX §404 applies"}',
        "rejected": '{"risk_level": "medium", "reason": "Looks okay"}',
    },
    ...
]

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,   # use implicit reference (β-DPO)
    args=DPOConfig(
        beta=0.1,     # KL penalty strength — higher = stay closer to base model
        max_length=512,
        max_prompt_length=256,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        output_dir="./compliance_dpo",
    ),
    train_dataset=Dataset.from_list(preference_data),
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

---

## 6. Serving Fine-tuned Models

```bash
# Option A: vLLM (best throughput, GPU required)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model ./compliance_classifier/final \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --port 8001
# → Drop-in OpenAI-compatible endpoint at localhost:8001

# Option B: llama.cpp (CPU/Mac M-series, no GPU needed)
# Convert to GGUF first, then:
./server -m compliance_classifier.Q4_K_M.gguf -c 2048 --port 8001

# Option C: Modal (serverless GPU in cloud)
# See project24 solution for Modal deployment code
```

**Connecting to LiteLLM**:
```python
import litellm
response = litellm.completion(
    model="openai/compliance-classifier",   # custom model name
    api_base="http://localhost:8001/v1",
    api_key="not-needed",
    messages=[{"role": "user", "content": "Classify this..."}],
)
```

---

## 7. Long-Term Agent Memory with Mem0

**Why sessions-only memory breaks products**: A user tells your agent their preferences on Monday. On Wednesday, they return and the agent has no memory. This feels broken.

**The four memory types**:
| Type | What it stores | Example |
|---|---|---|
| **Episodic** | Past events and agent runs | "Last Tuesday I processed invoice INV-9923" |
| **Semantic** | Facts and knowledge learned | "User prefers risk_level=high threshold at 0.8" |
| **Procedural** | Learned workflows | "For contract reviews, always check SOX first" |
| **User profile** | Persistent user attributes | role="analyst", department="legal", timezone="UTC+1" |

```python
# pip install mem0ai
from mem0 import Memory

m = Memory()

# Add memory after agent run
m.add(
    messages=[
        {"role": "user", "content": "I prefer detailed reasoning in risk reports"},
        {"role": "assistant", "content": "Noted. I'll include full reasoning chains."},
    ],
    user_id="analyst_007",
    metadata={"source": "compliance_agent", "session": "2026-06-08"},
)

# Retrieve relevant memories before next run
memories = m.search(
    query="user preferences for risk reporting",
    user_id="analyst_007",
    limit=5,
)
# → [{"memory": "User prefers detailed reasoning in risk reports", "score": 0.97}, ...]

# Use in system prompt
relevant = "\n".join(f"- {m['memory']}" for m in memories)
system_prompt = f"You are a compliance agent.\n\nUser preferences:\n{relevant}"
```

---

## 8. Memory Architecture for Production

```python
from mem0 import Memory, MemoryConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig

# Production config: Qdrant vector store + PostgreSQL metadata
config = MemoryConfig(
    vector_store={
        "provider": "qdrant",
        "config": QdrantConfig(
            collection_name="agent_memory",
            host="localhost",
            port=6333,
        ),
    },
    llm={"provider": "litellm", "config": {"model": "gpt-4o-mini"}},
    embedder={"provider": "huggingface", "config": {"model": "BAAI/bge-small-en-v1.5"}},
)

memory = Memory.from_config(config)

# Memory consolidation — compress memories older than 30 days
def consolidate_old_memories(user_id: str, days: int = 30) -> int:
    all_memories = memory.get_all(user_id=user_id)
    old = [m for m in all_memories if is_older_than(m["created_at"], days)]
    if not old:
        return 0
    summary = llm_summarize([m["memory"] for m in old])
    memory.add([{"role": "system", "content": f"Consolidated memory: {summary}"}], user_id=user_id)
    for m in old:
        memory.delete(m["id"])
    return len(old)
```

---

## Key Takeaways

1. **QLoRA**: 4-bit quantized fine-tune — train 8B models on a single GPU
2. **Data quality > training tricks**: 500 high-quality examples > 5000 noisy ones
3. **DPO**: Preferred when you have human preference ratings, not just correct answers
4. **Mem0**: Four memory types — use all four for production agents
5. **Consolidation**: Compress old memories to prevent memory store bloat

---

## Exercises

- `ex1_qlora_finetune.py` — QLoRA fine-tune llama-3B on synthetic compliance dataset
- `ex2_mem0_agent.py` — Build an agent with all four memory types using Mem0
