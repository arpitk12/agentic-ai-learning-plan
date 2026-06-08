# Week 15 Resources — DSPy + A/B Testing

---

## 🏁 Start Here (Read in This Order)

1. **DSPy README** — 5-minute overview of what DSPy is and why:
   https://github.com/stanfordnlp/dspy#readme

2. **DSPy Intro Tutorial** — official "Hello World" in DSPy:
   https://dspy.ai/learn/programming/

3. **Evan Miller's A/B Testing Guide** — the clearest statistical explanation:
   https://www.evanmiller.org/how-not-to-run-an-ab-test.html

---

## DSPy

### Official Resources
- **DSPy Website**:
  https://dspy.ai

- **DSPy GitHub** (source code + examples):
  https://github.com/stanfordnlp/dspy

- **DSPy Documentation** (Signatures, Modules, Optimizers reference):
  https://dspy.ai/api/

- **DSPy Examples** (classification, RAG, agents, multi-hop):
  https://github.com/stanfordnlp/dspy/tree/main/examples

### Foundational Paper
- **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines** (Khattab et al., 2023):
  https://arxiv.org/abs/2310.03714
  *The original paper. Read the intro and Section 3 (the programming model) — both are very readable.*

- **MIPROv2 Paper** (the most powerful DSPy optimizer):
  https://arxiv.org/abs/2406.11695

### Tutorials and Walkthroughs
- **DSPy — Official "Intro to DSPy"** (notebook walkthrough):
  https://dspy.ai/tutorials/

- **"DSPy: Goodbye Prompting, Hello Programming"** (Hamel Husain):
  https://hamel.dev/blog/posts/dspy/

- **"Optimising LLM Pipelines with DSPy"** (thorough beginner guide):
  https://towardsdatascience.com/intro-to-dspy-goodbye-prompting-hello-programming-4ca1c6ce3eb9

- **Weights & Biases — DSPy Course** (video series):
  https://www.wandb.courses/courses/prompting-llms-with-dspy

### Videos
- **Omar Khattab (DSPy creator) — DSPy: Programming, not Prompting**:
  https://youtu.be/CDung1LnLbY

- **Harrison Chase + Omar Khattab — DSPy Overview (talk)**:
  https://youtu.be/im7bCLW2aM4

---

## A/B Testing for LLM Applications

### Core Statistical Concepts
- **Evan Miller — "How Not to Run an A/B Test"** (explains sequential testing pitfalls):
  https://www.evanmiller.org/how-not-to-run-an-ab-test.html

- **Evan Miller — "Bayesian A/B Testing"** (intuitive Bayesian approach):
  https://www.evanmiller.org/bayesian-ab-testing.html

- **Khan Academy — Chi-Square Tests** (free video tutorial):
  https://www.khanacademy.org/math/statistics-probability/inference-categorical-data-chi-square-tests

- **SciPy Stats — chi2_contingency** (what the code uses):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

### LLM-Specific A/B Testing
- **"A/B Testing for LLMs: What's Different?"** (Weights & Biases blog):
  https://wandb.ai/fully-connected/ab-testing-llms

- **"Shadow Testing LLMs in Production"** (Anthropic engineering blog):
  https://www.anthropic.com/research/

- **LangSmith Experiments** — built-in A/B testing for LangChain apps:
  https://docs.smith.langchain.com/how_to_guides/evaluation/run_experiments

- **Braintrust — LLM A/B testing platform**:
  https://www.braintrustdata.com/docs/guides/experiments

### Statistical Tools Used in Exercises
- **SciPy Stats Docs**:
  https://docs.scipy.org/doc/scipy/reference/stats.html

- **statsmodels** — alternative stats library with more options:
  https://www.statsmodels.org/stable/index.html

---

## MLflow Model Registry

### Official Docs
- **MLflow Model Registry Overview**:
  https://mlflow.org/docs/latest/model-registry.html

- **MLflow Tracking** (logging metrics, parameters, artifacts):
  https://mlflow.org/docs/latest/tracking.html

- **MLflow Quickstart**:
  https://mlflow.org/docs/latest/getting-started/intro-quickstart/

### For LLM Applications
- **MLflow LLM Evaluation** (built-in LLM metrics):
  https://mlflow.org/docs/latest/llms/llm-evaluate/index.html

- **MLflow — Logging Prompts as Artifacts**:
  https://mlflow.org/docs/latest/llms/prompt-engineering-ui.html

### Videos
- **MLflow in 20 minutes** (practical intro):
  https://youtu.be/859OxXrt_TI

---

## Related Experiment Tracking Tools

| Tool | Best for | Free tier |
|---|---|---|
| **MLflow** | Self-hosted, maximum control | Yes (open-source) |
| **Weights & Biases** | ML training runs + sweeps | Yes (personal) |
| **Langfuse** | LLM-specific tracing + experiments | Yes |
| **LangSmith** | LangChain apps, evals | Yes (limited) |
| **Braintrust** | LLM prompt A/B testing | Yes (limited) |

---

## Understanding the Beta Distribution (Bayesian A/B)

- **"Beta Distribution Intuition"** (visual explanation):
  https://www.probabilisticworld.com/beta-distribution-intuition/

- **"Bayesian A/B Testing at VWO"** (real-world case study):
  https://vwo.com/downloads/VWO_SmartStats_technical_whitepaper.pdf

- **SciPy — Beta Distribution** (what the code uses):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

---

## Tools Checklist

| Tool | Purpose | Install |
|---|---|---|
| `dspy-ai` | Prompt optimisation framework | `pip install dspy-ai` |
| `mlflow` | Experiment tracking + model registry | `pip install mlflow` |
| `scipy` | Chi-square test, Beta distribution | `pip install scipy` |
| `numpy` | Array operations for stats | `pip install numpy` |
| `langfuse` | LLM tracing + experiment logging | `pip install langfuse` |
