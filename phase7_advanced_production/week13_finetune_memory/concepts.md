# Week 13 — Concept Guide: Fine-Tuning + Long-Term Memory

> **How to use this file**: Read this *before* `notes.md`. This file explains the *why* and the mental model in plain English — no code. Once you understand the concept, `notes.md` shows you the implementation.

---

## Concept 1 — Why Fine-Tuning Exists

### The problem with prompting

Prompts live outside the model. When you write a long system prompt full of examples, you are paying token cost on every single API call — even for tasks the model has already "seen" a thousand times. At scale this is expensive. More importantly, prompts have a ceiling: no matter how many examples you stuff in, there are tasks where the model just cannot consistently produce the exact format or reasoning style you need.

### The intuition

Think of prompting as giving a contractor a 10-page instruction manual every time they show up. Fine-tuning is like hiring that contractor full-time and training them until the behaviour is automatic — they don't need the manual anymore.

Fine-tuning updates the model's weights so the desired behaviour is baked in. The model no longer needs examples in the prompt because the task knowledge is part of its parameters.

### When fine-tuning makes sense vs not

| Situation | Use |
|---|---|
| Knowledge is public and retrievable at runtime | RAG (not fine-tune) |
| You need a specific output format or tone consistently | Fine-tune |
| The task is narrow and repetitive (e.g., classify every invoice) | Fine-tune |
| You want to cut inference cost by switching from GPT-4o to a 7B model | Fine-tune smaller model |
| The model needs to know private data it can't access at inference | Fine-tune |
| You're experimenting or building a prototype | Prompt engineering first |

**Rule of thumb**: If a human expert reading only the prompt could answer correctly every time, prompt engineering is enough. If the model needs to have internalised a style, domain vocabulary, or reasoning pattern, fine-tune.

---

## Concept 2 — LoRA: How Fine-Tuning Became Affordable

### The old problem

A model like Llama 3 8B has ~8 billion parameters. Full fine-tuning means updating all 8B parameters on each training step. That requires multiple A100 GPUs (expensive) and produces a completely separate copy of the model for each use case.

### What LoRA does (in plain English)

LoRA stands for **Low-Rank Adaptation**. The key insight is:

> The change a fine-tuning run needs to make to a model is low-rank — it can be expressed as the product of two small matrices rather than one large one.

Instead of modifying the original weight matrix W (which is huge), LoRA adds two small matrices A and B:

```
New output = W·x + (B·A)·x
```

Where:
- W is frozen (never touched during training)
- A and B are tiny (rank 8 or 16 vs the full dimension of thousands)
- Only A and B are trained

**Result**: You train ~0.1–1% of parameters, use 4–8× less GPU memory, and get quality nearly identical to full fine-tuning on narrow tasks. When you are done, you can discard A and B or swap them per-request (multiple LoRA adapters on one base model).

### What QLoRA adds

QLoRA = LoRA + 4-bit quantization of the base model's frozen weights.

Quantization means storing weights in 4 bits instead of 16 bits (a 4× memory reduction). Combined with LoRA's parameter reduction, QLoRA lets you fine-tune a 7–8B model on a single consumer GPU (24GB VRAM, such as an RTX 3090 or 4090) or a free Google Colab notebook.

**Mental model**: QLoRA is like compressing the heavy textbook you're not writing in (the base weights) so it fits in your bag, while you only scribble notes (LoRA adapters) on a small notepad.

---

## Concept 3 — SFT vs DPO: Two Training Objectives

### Supervised Fine-Tuning (SFT)

The simplest form of fine-tuning. You give the model:
- Input: a prompt
- Target: the exact output you want

The model learns to reproduce your target output. You need examples of the *correct* answer. Good for: format compliance, domain jargon, consistent reasoning traces.

### DPO (Direct Preference Optimisation)

DPO teaches the model what is *preferred* vs what is *rejected*. Instead of correct/wrong, you provide:
- Input: a prompt
- Chosen: the better response
- Rejected: the worse response

The model learns to increase the probability of chosen responses relative to rejected ones. Good for: alignment, tone, safety, quality ranking when there is no single "correct" answer.

**Analogy**: SFT is like showing a student the answer key. DPO is like showing them two essays and saying "this one is better than that one — learn to write like the better one."

**Practical tip**: Run SFT first to get the task format right, then DPO to improve quality and align style.

---

## Concept 4 — Training Dataset Quality

This is the single most important factor in fine-tuning outcomes — more important than hyperparameters, more important than model size.

### What makes a good training example

1. **Diversity**: Cover the full range of inputs the model will see in production (easy cases, edge cases, ambiguous cases).
2. **Consistency**: If example A classifies "missing DPA clause" as *high risk* and example B classifies the same language as *medium risk*, the model learns noise.
3. **Correct reasoning, not just correct answers**: Include chain-of-thought traces if you want the model to reason, not just label.
4. **Minimum 200 examples**: Fewer than 100 examples often fails to shift the model's behaviour reliably.

### Golden rule

"Garbage in, garbage out" applies 10× harder to fine-tuning than to prompting. A model trained on inconsistent data learns inconsistency.

---

## Concept 5 — What "Long-Term Memory" Means for Agents

### The problem with stateless agents

Every LLM API call starts from a blank slate. The model doesn't remember:
- What you discussed last week
- Your preferences ("always respond in bullet points")
- That you are allergic to certain solutions
- Patterns from 1,000 prior user sessions

All existing memory techniques (conversation history, RAG, LangGraph state) are *session-scoped*. They vanish when the conversation ends.

### Mem0 — the four memory types

Mem0 is a memory layer that sits alongside your agent and persists memory to a vector database (Qdrant, Pinecone, etc.):

| Memory type | What it stores | Example |
|---|---|---|
| **Episodic** | Specific past events | "On 2025-03-14 the user asked about GDPR fines" |
| **Semantic** | General facts the agent learned | "Acme Corp is a cloud vendor headquartered in Dublin" |
| **Procedural** | How to do tasks | "When user asks for a contract review, always check the termination clause first" |
| **User profile** | Stable user preferences | "User prefers concise answers, hates jargon, timezone is IST" |

### How memory retrieval works

When a new user message arrives:
1. Mem0 embeds the message
2. Searches the vector store for semantically relevant past memories
3. Injects the relevant memories into the system prompt automatically
4. After the conversation, Mem0 extracts new facts and adds them to memory

**Result**: After 10 conversations, the agent behaves like it's been working with you for 10 conversations — because it has.

---

## Concept 6 — vLLM: Serving Your Fine-Tuned Model

Once you fine-tune a model, you need to serve it. vLLM is the standard high-performance inference server for open-source models.

### Why not just use Hugging Face `model.generate()`?

The standard Hugging Face generation is not optimised for concurrent requests. Under load it becomes slow and memory-inefficient.

vLLM implements **PagedAttention** — a technique that manages the KV (key-value) cache like a virtual memory system, dramatically increasing throughput (tokens per second) and concurrent user capacity.

**Mental model**: vLLM is to model inference what Nginx is to web serving — it handles concurrency, batching, and memory efficiently so your model can serve many users at once.

### How you use it

After fine-tuning, you push your LoRA adapter to Hugging Face Hub, then run:
```
vllm serve my-org/my-finetuned-model --port 8000
```
This gives you an OpenAI-compatible REST API (`/v1/chat/completions`) that any existing code using `openai` client can call — just by changing the `base_url`.

---

## Key Takeaways

- **Fine-tune when**: you need consistent format/style at scale, or want to cut inference cost by switching to a smaller model
- **QLoRA**: makes fine-tuning accessible on a single GPU — you don't need cloud TPUs
- **SFT first, DPO second**: SFT for format, DPO for quality
- **Dataset quality > hyperparameters**: 500 clean examples beat 5,000 noisy ones
- **Mem0**: four memory types that persist across sessions — episodic, semantic, procedural, user profile
- **vLLM**: the production inference server for open-source models; OpenAI-compatible API
