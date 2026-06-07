# Week 10 Resources — Evaluation & Testing

## Evaluation Frameworks

### RAGAS (RAG-specific)
- Docs: https://docs.ragas.io/
- GitHub: https://github.com/explodinggradients/ragas
- Metric reference (faithfulness, relevancy, precision, recall): https://docs.ragas.io/en/latest/concepts/metrics/
- Custom LLM configuration: https://docs.ragas.io/en/latest/howtos/customisations/llms/
- Exercises: **ex3** (basic), **ex9** (advanced chunk comparison)

### DeepEval
- Docs: https://docs.confident-ai.com/
- GitHub: https://github.com/confident-ai/deepeval
- Metrics reference (14+ built-in): https://docs.confident-ai.com/docs/metrics-llm-evals
- Custom BaseMetric guide: https://docs.confident-ai.com/docs/metrics-custom
- pytest integration: https://docs.confident-ai.com/docs/integrations-pytest
- Exercise: **ex10**

### LangSmith Evaluation
- Evaluation guide: https://docs.smith.langchain.com/evaluation
- Dataset management: https://docs.smith.langchain.com/evaluation/concepts#datasets
- Custom evaluators: https://docs.smith.langchain.com/evaluation/how_to_guides/custom_evaluator
- Experiment comparison UI: https://docs.smith.langchain.com/evaluation/tutorials/experiments
- Exercise: **ex11**

### Phoenix / Arize (Open-Source Observability + Eval)
- Phoenix home: https://phoenix.arize.com/
- GitHub: https://github.com/Arize-ai/phoenix
- Eval docs: https://docs.arize.com/phoenix/evaluation/

### BrainTrust
- Home: https://www.braintrustdata.com/
- Eval framework: https://www.braintrustdata.com/docs/guides/evals

---

## Papers

### Foundational
- **LLM-as-a-Judge (MT-Bench 2023)**: https://arxiv.org/abs/2306.05685
  - _"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"_ — the canonical reference for using LLMs to evaluate LLMs
- **RAGAS (2023)**: https://arxiv.org/abs/2309.15217
  - _"RAGAS: Automated Evaluation of Retrieval Augmented Generation"_
- **HELM Benchmark**: https://crfm.stanford.edu/helm/
- **AgentBench**: https://arxiv.org/abs/2308.03688
  - Comprehensive agent evaluation across 8 environments

### Recent (2024)
- **Can LLMs Replace Human Evaluators?** (2024): https://arxiv.org/abs/2404.03622
  - Systematic study of LLM judge reliability and bias
- **LLM Evaluators Recognize and Favor Their Own Generations** (2024): https://arxiv.org/abs/2404.13076
  - Self-preference bias in LLM judges — important for calibration
- **Safety Evaluation of Large Language Models** (2024): https://arxiv.org/abs/2404.01763

---

## Benchmarks to Know

| Benchmark | Domain | URL |
|-----------|--------|-----|
| SWE-bench | Software engineering agents | https://www.swebench.com/ |
| HumanEval | Code generation | https://github.com/openai/human-eval |
| GAIA | General AI assistants | https://huggingface.co/datasets/gaia-benchmark/GAIA |
| BigCodeBench | Realistic code tasks | https://bigcode-bench.github.io/ |
| TruthfulQA | Hallucination detection | https://github.com/sylinrl/TruthfulQA |
| AdvGLUE | Adversarial NLU | https://adversarialglue.github.io/ |

---

## Install

```bash
# Core eval frameworks (all exercises)
pip install ragas datasets deepeval langsmith

# Testing infrastructure
pip install pytest pytest-asyncio

# Analysis + reporting
pip install pandas rich

# LLM provider (project uses LiteLLM)
pip install litellm anthropic

# Optional: open-source observability dashboard
pip install arize-phoenix
```

---

## Key Principles

- **Eval before shipping** — no exceptions; build CI gates that block on eval failures
- **LLM-as-judge bias** — judges favour verbosity and their own generations; calibrate first
- **Golden dataset size** — need ≥50 hand-labelled examples for statistical confidence; 20 is a minimum
- **Track over time** — save eval results (CSV/DB) with timestamp + version; treat score drops as regressions
- **Adversarial cases are mandatory** — prompt injection, PII leak, jailbreak, edge inputs
- **Framework choice** — RAGAS for RAG, DeepEval for typed metrics + pytest, LangSmith for team datasets + experiment versioning
- **P50/P95/P99** — latency percentiles matter; P95 is what most users experience
