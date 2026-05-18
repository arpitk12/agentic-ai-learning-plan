# Week 10 Resources — Evaluation & Testing

## Frameworks
- RAGAS (RAG evaluation): https://docs.ragas.io/
- DeepEval: https://docs.confident-ai.com/
- BrainTrust: https://www.braintrustdata.com/
- LangSmith Evals: https://docs.smith.langchain.com/evaluation

## Papers
- LLM-as-Judge (2023): https://arxiv.org/abs/2306.05685
- HELM Benchmark: https://crfm.stanford.edu/helm/
- AgentBench (agent evaluation): https://arxiv.org/abs/2308.03688

## Benchmarks to Know
- SWE-bench (software engineering): https://www.swebench.com/
- HumanEval (code): https://github.com/openai/human-eval
- GAIA (general AI assistants): https://huggingface.co/datasets/gaia-benchmark/GAIA

## Install
```
pip install ragas deepeval anthropic pydantic rich
```

## Key Principles
- Eval your agent BEFORE shipping — no exceptions
- LLM-as-judge is faster than human eval but biased toward verbosity
- Always have a GOLDEN SET of 20-50 hand-labeled examples
- Track eval scores over time — regression testing matters
- Adversarial cases (prompt injection, edge inputs) are mandatory
