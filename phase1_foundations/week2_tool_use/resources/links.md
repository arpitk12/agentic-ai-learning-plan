# Week 2 Resources — Tool Use & ReAct

## Official Docs
- Anthropic Tool Use Guide: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Parallel Tool Calls: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#parallel-tool-use
- Tool Results: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#handling-tool-use-and-tool-results

## Papers
- ReAct: Synergizing Reasoning and Acting in LLMs (2022): https://arxiv.org/abs/2210.03629
- Toolformer (2023): https://arxiv.org/abs/2302.04761

## Courses
- DeepLearning.AI: "Functions, Tools and Agents with LangChain": https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/
- DeepLearning.AI: "AI Agents in LangGraph": https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/

## Tutorials
- Anthropic Tool Use Cookbook: https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use
- OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling

## Key Patterns
- Always loop over ALL content blocks — the model may call multiple tools at once
- Return errors as tool_result strings, not Python exceptions
- max_steps guard is non-negotiable for production agents
- Log every tool call for debugging (use structlog in production)
