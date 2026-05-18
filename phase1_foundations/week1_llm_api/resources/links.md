# Week 1 Resources — LLM API Mastery

## 🌟 Start Here
- **Complete Guide: First Project → Production-Ready AI Agents**: https://medium.com/@devkapiltech/a-complete-guide-to-building-production-ready-ai-agents-from-your-first-afternoon-project-to-d5c2f3597565

## Official Docs
- Anthropic Messages API: https://docs.anthropic.com/en/api/messages
- Anthropic Python SDK: https://github.com/anthropics/anthropic-sdk-python
- OpenAI Chat Completions: https://platform.openai.com/docs/guides/chat
- Pydantic v2 Docs: https://docs.pydantic.dev/latest/

## Prompt Engineering
- Prompting Guide (ReAct, CoT, few-shot): https://www.promptingguide.ai
- Anthropic Prompt Library: https://docs.anthropic.com/en/prompt-library
- Anthropic Prompt Engineering Overview: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview

## Courses
- DeepLearning.AI: "ChatGPT Prompt Engineering for Developers" (free): https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/

## Papers
- Chain-of-Thought Prompting (2022): https://arxiv.org/abs/2201.11903
- Large Language Models are Zero-Shot Reasoners (2022): https://arxiv.org/abs/2205.11916

## Cost & Tokens
- Anthropic pricing: https://www.anthropic.com/pricing
- OpenAI Tokenizer: https://platform.openai.com/tokenizer

## Key Takeaways
- Sliding window beats full history for long chats (cost + latency)
- TTFT (Time-To-First-Token) is the streaming UX metric that matters
- Retry with the error in-context — models can self-correct JSON failures
- Lilian Weng — Prompt Engineering: https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/

## Structured Outputs
- Pydantic docs: https://docs.pydantic.dev/latest/
- Instructor library (structured outputs): https://python.useinstructor.com/

## Videos
- Andrej Karpathy — "Intro to Large Language Models": https://youtu.be/zjkBMFhNj_g
- Andrej Karpathy — "Let's build GPT": https://youtu.be/kCc8FmEb1nY

## Cost Estimation
- Anthropic pricing: https://www.anthropic.com/pricing
- Token counter: use `client.count_tokens()` before sending expensive requests
