# Week 10 — Evaluation & Testing

## Topics
1. LLM-as-judge evaluation, rubric design
2. Building eval datasets: golden sets, adversarial, edge cases
3. Regression testing for agents (deterministic + LLM checks)
4. Evals frameworks: RAGAS, DeepEval, BrainTrust

## Key Concepts

### LLM-as-Judge
Use a separate LLM call to evaluate outputs:
```python
def llm_judge(question, expected, actual, rubric):
    prompt = f"""
    Question: {question}
    Expected: {expected}
    Actual: {actual}
    Rubric: {rubric}
    
    Score the actual response 1-5 and explain why. JSON only:
    {{"score": N, "reasoning": "..."}}
    """
    # Call LLM, parse response
```

### Dataset Types
| Type | Purpose | Size |
|---|---|---|
| Golden set | Core functionality | 20-50 examples |
| Adversarial | Known failure modes | 10-20 examples |
| Edge cases | Boundary conditions | 10-20 examples |
| Regression | Bugs found in prod | Grows over time |

### RAGAS Metrics for RAG
- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are retrieved chunks relevant?
- **Context Recall**: Did we retrieve all necessary chunks?

### Pytest for Agents
```python
@pytest.mark.parametrize("question,expected_contains", [
    ("What is 2+2?", "4"),
    ("Capital of France?", "Paris"),
])
def test_agent_basic(question, expected_contains):
    result = run_agent(question)
    # Deterministic check
    assert expected_contains.lower() in result.lower()
    # LLM check for open-ended
    score = llm_judge(question, expected_contains, result)
    assert score >= 3
```

## Exercises
- `ex1_golden_dataset.py` — build 50-question eval set
- `ex2_llm_judge.py` — automated scoring pipeline
- `ex3_ragas_eval.py` — evaluate RAG pipeline with RAGAS
- `ex4_pytest_agent.py` — regression test suite

## Checklist
- [ ] 50-question golden dataset created
- [ ] LLM-as-judge pipeline running automatically
- [ ] RAGAS faithfulness score > 0.8
- [ ] pytest suite runs in CI with 0 regressions
