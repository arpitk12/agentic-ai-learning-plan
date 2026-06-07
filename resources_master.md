# Master Resource List — Agentic AI

## 📚 Books
- "Designing Machine Learning Systems" — Chip Huyen (production mindset)
- "Building LLM Applications" — various authors on O'Reilly

## 🎓 Short Courses (DeepLearning.AI — all free)
- AI Agents in LangGraph
- Building Agentic RAG with LlamaIndex
- Multi AI Agent Systems with crewAI
- Evaluating and Debugging Generative AI
- Functions, Tools and Agents with LangChain

## 📄 Must-Read Papers
- ReAct (2022): https://arxiv.org/abs/2210.03629
- Reflexion (2023): https://arxiv.org/abs/2303.11366
- Tree of Thought (2023): https://arxiv.org/abs/2305.10601
- AutoGen (2023): https://arxiv.org/abs/2308.00352
- LATS (2023): https://arxiv.org/abs/2310.04406
- RAGAS (2023): https://arxiv.org/abs/2309.15217
- LLM-as-a-Judge / MT-Bench (2023): https://arxiv.org/abs/2306.05685
- AgentBench (2023): https://arxiv.org/abs/2308.03688
- Can LLMs Replace Human Evaluators? (2024): https://arxiv.org/abs/2404.03622
- Anthropic's "Building Effective Agents": https://www.anthropic.com/research/building-effective-agents

## 🛠 Key Libraries
| Library | Use |
|---|---|
| anthropic | Anthropic SDK |
| litellm | Unified LLM API (OpenAI, Anthropic, Gemini, …) |
| langchain / langgraph | Agent framework |
| chromadb / qdrant | Vector database |
| sentence-transformers | Local embeddings |
| fastapi + uvicorn | API serving |
| celery + redis | Background jobs |
| structlog | Structured logging |
| opentelemetry | Distributed tracing |
| ragas | RAG evaluation (faithfulness, relevancy, precision, recall) |
| datasets | HuggingFace datasets — eval data management |
| deepeval | LLM evaluation with 14+ typed metrics and pytest integration |
| langsmith | LangSmith tracing **and** evaluation datasets + experiments |
| arize-phoenix | Open-source LLM observability + eval dashboard |
| pydantic | Data validation |
| httpx | Async HTTP |
| tavily-python | Web search tool |

## 📰 Newsletters & Blogs
- **Complete Guide to Production-Ready AI Agents** (start here): https://medium.com/@devkapiltech/a-complete-guide-to-building-production-ready-ai-agents-from-your-first-afternoon-project-to-d5c2f3597565
- The Batch (DeepLearning.AI): https://www.deeplearning.ai/the-batch/
- Latent Space: https://www.latent.space/
- Ahead of AI (Sebastian Raschka): https://magazine.sebastianraschka.com/
- Simon Willison's Weblog: https://simonwillison.net/

## 🎥 YouTube Channels
- Andrej Karpathy
- Yannic Kilcher
- AI Explained
- Matt Wolfe

## 🔧 Tools & Platforms
- LangSmith (tracing + eval datasets): https://smith.langchain.com/
- Weights & Biases (experiment tracking): https://wandb.ai/
- Helicone (LLM observability): https://www.helicone.ai/
- Phoenix / Arize (open-source LLM eval dashboard): https://phoenix.arize.com/
- BrainTrust (evals + tracing): https://www.braintrustdata.com/
- Tavily (web search API): https://tavily.com/
- Modal (serverless GPU/CPU): https://modal.com/
- Railway (deploy): https://railway.app/

## 🔐 Security Resources
- OWASP LLM Top 10: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Prompt injection examples: https://github.com/greshake/llm-security
- Lakera AI (guardrails): https://www.lakera.ai/

## 📊 Benchmarks (know what your agent is measured against)
- SWE-bench (software engineering): https://www.swebench.com/
- HumanEval (code generation): https://github.com/openai/human-eval
- GAIA (general AI assistants): https://huggingface.co/datasets/gaia-benchmark/GAIA
- AgentBench (agent evaluation): https://arxiv.org/abs/2308.03688
- BigCodeBench (realistic code tasks): https://bigcode-bench.github.io/
- TruthfulQA (hallucination detection): https://github.com/sylinrl/TruthfulQA
