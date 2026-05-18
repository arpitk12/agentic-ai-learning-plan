# Week 3 Resources — LangGraph & Frameworks

## Official Docs
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- LangGraph Quickstart: https://langchain-ai.github.io/langgraph/tutorials/introduction/
- LangChain LCEL: https://python.langchain.com/docs/expression_language/
- LangSmith (tracing): https://docs.smith.langchain.com/

## Courses
- DeepLearning.AI: "AI Agents in LangGraph" (free): https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/

## Key Repos
- LangGraph Examples: https://github.com/langchain-ai/langgraph/tree/main/examples
- LangGraph ReAct example: https://github.com/langchain-ai/langgraph/blob/main/examples/react_agent.ipynb

## When Framework vs Raw SDK
| Use LangGraph | Use Raw Anthropic SDK |
|---|---|
| Complex branching state machines | Simple linear chains |
| Built-in persistence needed | Full control / minimal deps |
| Multi-agent coordination | Prototyping / learning |
| Human-in-the-loop flows | Latency-critical paths |

## Install
```
pip install langgraph langchain-anthropic langchain-core langsmith
```
