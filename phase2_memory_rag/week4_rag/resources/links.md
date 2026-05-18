# Week 4 Resources — Memory & RAG

## Official Docs
- ChromaDB: https://docs.trychroma.com/
- Qdrant: https://qdrant.tech/documentation/
- Sentence Transformers: https://www.sbert.net/
- LangChain RAG: https://python.langchain.com/docs/use_cases/question_answering/

## Courses
- DeepLearning.AI: "Building Agentic RAG with LlamaIndex": https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/
- DeepLearning.AI: "Advanced Retrieval for AI with Chroma": https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/

## Papers
- RAG (Lewis et al., 2020): https://arxiv.org/abs/2005.11401
- REALM (2020): https://arxiv.org/abs/2002.08909
- Lost in the Middle (2023) — chunk ordering matters: https://arxiv.org/abs/2307.03172

## Tools
- RAGAS (RAG evaluation framework): https://docs.ragas.io/
- Cohere Reranker API: https://docs.cohere.com/docs/reranking
- rank_bm25: https://github.com/dorianbrown/rank_bm25

## Install
```
pip install chromadb sentence-transformers rank-bm25 pypdf anthropic numpy
```

## Key Insights
- chunk_size=512 with 10% overlap is a solid starting point
- Always rerank after retrieval — initial similarity scores are noisy
- "Lost in the Middle": put most relevant chunks at start/end of context
- Hybrid search (vector + BM25) consistently beats either alone
