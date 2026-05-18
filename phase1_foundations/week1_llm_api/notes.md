# Week 1 — LLM API Mastery

## Topics
1. Anthropic / OpenAI SDK — chat completions, streaming, role turns
2. Prompt engineering: system prompts, few-shot, chain-of-thought
3. Token limits, context windows, cost estimation
4. Structured outputs (JSON mode, Pydantic parsing)

## Key Concepts

### The Message Format
Every LLM call is a list of role-tagged messages:
- `system` — sets the model's behavior and persona
- `user` — human input
- `assistant` — model response (can be pre-filled)

### Context Window
The model sees ALL messages each call. Manage history carefully:
- Summarize old turns to save tokens
- Never trust the model to "remember" without seeing it

### Structured Output
Force JSON by: (a) saying "reply only in JSON", (b) providing a schema,
(c) parsing with Pydantic and retrying on failure.

## Exercises
See `exercises/` folder for starter code:
- `ex1_multiturn_chatbot.py`
- `ex2_structured_output.py`
- `ex3_streaming.py`
- `ex4_prompt_comparison.py`

## Checklist
- [ ] Built multi-turn CLI chatbot with sliding window memory
- [ ] Forced JSON output and validated with Pydantic
- [ ] Implemented streaming with live token display
- [ ] Compared 3 prompting strategies on a reasoning task
