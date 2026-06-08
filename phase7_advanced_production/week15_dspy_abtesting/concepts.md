# Week 15 — Concept Guide: DSPy + A/B Testing

> **How to use this file**: Read this *before* `notes.md`. This file explains the *why* and the mental model in plain English — no code. Once you understand the concept, `notes.md` shows you the implementation.

---

## Concept 1 — The Problem DSPy Solves

### Hand-crafted prompts are engineering debt

Consider this workflow that most teams follow today:

```
1. Write a prompt
2. Test it on 10 examples
3. Find failures
4. Manually tweak wording
5. Test again
6. Repeat for weeks
7. Deploy
8. Model version changes → start over
```

This is fragile for several reasons:
- **You are optimising by intuition**, not by systematic search
- **Prompts are brittle**: changing one sentence can improve performance on some inputs and hurt others
- **Not reproducible**: two engineers making different edits produce different results
- **Not transferable**: when you switch from GPT-4o to Claude or Gemini, your carefully crafted prompts often need rework

### What DSPy does differently

DSPy treats prompt engineering as a **compilation problem**, not a craft problem.

You define:
1. **What** the task is (a Signature: inputs and outputs)
2. **How to measure success** (a metric function)
3. **Training examples** (a small dev set)

DSPy's optimizers then **automatically search** over the space of possible prompts, few-shot examples, and instructions to find the combination that maximises your metric on your dev set.

**Analogy**: DSPy is like a compiler. You write high-level code (what you want). The compiler (DSPy) translates it into low-level machine instructions (prompts + few-shot examples) optimised for your target hardware (the specific LLM you're using).

---

## Concept 2 — DSPy Building Blocks

### Signature — the task contract

A Signature is a declaration of what goes in and what comes out. It says nothing about *how* to do the task — that's the module's job.

```
Input: document text + document type
Output: risk_level + reasoning
```

Think of it as a type signature for an AI task — like defining a function's parameters and return type before writing the body.

### Module — the reasoning strategy

A Module wraps a Signature with a reasoning strategy:
- `Predict` — direct input-to-output, no explicit reasoning
- `ChainOfThought` — forces the model to reason step-by-step before producing the output
- `ReAct` — alternating reasoning and tool-use steps
- `MultiChainComparison` — generates multiple candidate answers, then selects the best

**Key insight**: You can swap reasoning strategies without changing your Signature. This makes it easy to experiment.

### Optimizer — the thing that improves your program

An optimizer takes your unoptimised module and improves it by finding better prompts and examples:

**BootstrapFewShot**
- Runs your module on training examples
- Keeps the examples where the module succeeds (these become few-shot examples for the prompt)
- Fast, simple, works well with 20–50 training examples
- Best starting point

**MIPROv2** (Multi-prompt Instruction PRoposal Optimizer v2)
- Proposes many candidate instruction phrasings using a meta-LLM
- Evaluates each candidate on your dev set
- Selects the combination of instructions + examples that maximises your metric
- Slower but achieves higher accuracy gains (typically +10–30% over BootstrapFewShot)
- Use when you have 50+ examples and want maximum performance

---

## Concept 3 — What DSPy Is Not

**DSPy is not a prompt template library.** It doesn't give you better prompts to copy-paste. It finds prompts automatically.

**DSPy is not RAG** (but it works great with RAG). DSPy optimises how your agent uses a retriever, not the retriever itself.

**DSPy is not magic.** It needs:
- A clear, measurable metric (if you can't measure success, DSPy can't optimise for it)
- At least 20 labelled examples to optimise against (50+ for MIPROv2)
- Time to run the optimisation (minutes to hours depending on dataset size)

---

## Concept 4 — A/B Testing for LLM Applications

### Why you need A/B testing

You have a working agent with prompt v1. You believe prompt v2 is better. How do you know?

- Testing on a handful of examples is not statistically reliable
- "It feels better" is not a measurement
- Deploying v2 to all users risks degrading experience if you're wrong

A/B testing lets you answer: "Is v2 actually better, and how confident am I in that claim?"

### The mechanics

**Traffic splitting**: When a user request arrives, you assign it to either version A (control) or version B (treatment) based on a deterministic hash of the user ID. The same user always sees the same version — otherwise your measurements are noisy.

**Shadow mode**: Before fully splitting traffic, run both models on every request. Show the user only version A's response. Collect version B's response silently for comparison. Zero user risk, full measurement capability.

**Metrics to measure**: Not just "is the output correct" — also:
- Latency (p50, p95, p99)
- Cost per request
- User engagement (did they follow up? did they rate it positively?)
- Error rate
- Safety violation rate

---

## Concept 5 — Statistical Significance: Chi-Square Test

### The core question

You ran 1,000 requests through version A and 1,000 through version B. Version A was correct 720 times. Version B was correct 750 times. Is B actually better, or is the difference just random noise?

This is the question the chi-square test answers.

### What chi-square tests

The chi-square test for independence checks: "Could this difference in rates have arisen by chance if the two versions were actually identical?"

The test produces a **p-value**: the probability that you'd see this large a difference by pure chance if the two versions were actually identical.

- **p < 0.05**: Less than 5% chance the difference is random. Convention: call it "statistically significant."
- **p < 0.01**: Less than 1% chance. Stronger evidence.
- **p > 0.05**: The difference could easily be noise. Don't ship yet.

### Common mistakes

- **Stopping too early**: If you check significance every 100 requests and stop when p < 0.05, you'll often get false positives. Decide your sample size *before* starting.
- **Measuring the wrong thing**: A model can score higher on your automatic metric and still get lower user satisfaction scores.
- **Ignoring practical significance**: p=0.001 but the improvement is 0.3% accuracy. Statistically significant ≠ worth shipping.

---

## Concept 6 — Bayesian A/B Testing

### What it is and why it's better

Classical A/B testing (chi-square) gives you p-values: "Is there an effect?" Bayesian A/B testing gives you a direct probability: "What's the probability that version B is better than version A by at least X%?"

### The Beta distribution

A Beta distribution models our belief about a probability (like success rate) when we have seen some successes and failures.

Think of it this way:
- Before you run any tests: your belief about version A's true success rate is a flat distribution — it could be anywhere from 0 to 1
- After seeing 720 successes out of 1,000: your belief concentrates around 72%, with some uncertainty on either side
- The Beta distribution is the mathematical tool that represents this belief

**Beta(α, β)** where:
- α = number of successes + 1
- β = number of failures + 1

### Why Bayesian is more useful in practice

With chi-square, you get: "The difference is significant at p=0.03"

With Bayesian, you get: "There is a 94.7% probability that version B is better than version A"

The second statement is what business stakeholders actually want to know. It also allows you to quantify how much better: "There is a 78% probability that B is better by at least 2%."

---

## Concept 7 — MLflow Model Registry

### What it is

MLflow Model Registry is a centralised store that tracks:
- Every version of every model/prompt you've registered
- Who registered it and when
- Its lifecycle stage: Staging → Production → Archived
- Metrics from evaluation runs

### Why you need it

Without a model registry, your prompt versions live in:
- Git commits (hard to track which is in production)
- Environment variables (invisible)
- Hardcoded strings (require code deploy to change)

With a model registry:
- You can roll back to a previous prompt version in seconds (no code deploy)
- You have an audit trail of every change (compliance-friendly)
- Staging lets you test before promotion to production
- You can compare metrics across versions in a UI

### The lifecycle in practice

```
Developer registers new prompt → "Staging"
    → Run A/B test vs current "Production" version
    → Test passes significance threshold
    → Promote to "Production"
    → Old version moves to "Archived"
```

---

## Key Takeaways

- **DSPy**: define what you want (Signature + metric), give it examples (dev set), let it find the prompts for you — don't hand-craft prompts for well-defined tasks
- **BootstrapFewShot**: fast optimiser, start here, needs ~20 examples
- **MIPROv2**: slower but more powerful, use when you need maximum accuracy, needs 50+ examples
- **A/B testing**: split traffic by user hash (deterministic), measure the right metrics, use shadow mode first
- **Chi-square**: tells you if a difference is statistically significant (p < 0.05)
- **Bayesian A/B**: tells you the probability that B is better than A — more intuitive, more useful for decisions
- **MLflow Model Registry**: the source of truth for what is in production — enables rollback, audit trail, and staged promotion
