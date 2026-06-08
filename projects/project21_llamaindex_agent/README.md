# Project 21 — LlamaIndex: Multi-Document RAG Agent

A **production-grade LlamaIndex application** demonstrating advanced retrieval:
multiple indexes over heterogeneous documents, a SubQuestionQueryEngine that
decomposes complex questions, a RouterQueryEngine that picks the right index,
and a ReActAgent that uses query engines as tools.

---

## 🎯 What You Learn

| Concept | Where |
|---------|-------|
| **VectorStoreIndex** — FAISS-backed index | `src/indexing/vector_index.py` |
| **SummaryIndex** — for summarization queries | `src/indexing/summary_index.py` |
| **SubQuestionQueryEngine** — decompose complex Q | `src/query/sub_question.py` |
| **RouterQueryEngine** — pick right index | `src/query/router.py` |
| **ReActAgent** — agent with tool-use loop | `src/agent/agent.py` |
| **QueryEngineTool** — wrap index as agent tool | `src/agent/tools.py` |
| **Node post-processors** — rerank, filter | `src/query/postprocessors.py` |
| **Custom prompt templates** | `src/query/prompts.py` |
| **Ingestion pipeline** — transformations | `src/indexing/pipeline.py` |
| **IngestionCache** — skip re-embedding | `src/indexing/pipeline.py` |

---

## 🏗 Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║              INGESTION PIPELINE                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  data/docs/ (PDF, MD, TXT)                                       ║
║       │                                                          ║
║  SimpleDirectoryReader      ← loads + extracts metadata         ║
║       │                                                          ║
║  IngestionPipeline                                               ║
║    ├── SentenceSplitter(chunk_size=512, overlap=50)              ║
║    ├── TitleExtractor()     ← LLM-inferred title per chunk      ║
║    ├── QuestionsAnsweredExtractor() ← LLM-generated Q per chunk ║
║    └── HuggingFaceEmbedding ← local embedding (no API cost)     ║
║       │  IngestionCache     ← skip re-embedding unchanged docs  ║
║       │                                                          ║
║  ┌────▼──────────────────────────────────────────────────────┐  ║
║  │  VectorStoreIndex (FAISS)   SummaryIndex (tree)           │  ║
║  │  — for factual Q&A          — for "summarise this doc"    │  ║
║  └───────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║              QUERY LAYER                                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  User question                                                   ║
║       │                                                          ║
║  ReActAgent                                                      ║
║    ├── QueryEngineTool("vector_index") ← factual questions      ║
║    ├── QueryEngineTool("summary_index") ← summarization         ║
║    └── QueryEngineTool("sub_question") ← multi-part questions   ║
║                                                                  ║
║  SubQuestionQueryEngine (for complex multi-part Q):             ║
║       ├── decomposes: "Compare X and Y on A, B, C"             ║
║       ├── sub-Q 1: "What is X's A?" → vector_index             ║
║       ├── sub-Q 2: "What is Y's A?" → vector_index             ║
║       └── combine answers into final response                   ║
║                                                                  ║
║  RouterQueryEngine:                                              ║
║       ├── LLM picks: is this a summary or factual Q?            ║
║       ├── → SummaryIndex   if "summarise / overview / explain"  ║
║       └── → VectorIndex    if "what / who / how / when"        ║
║                                                                  ║
║  Node Post-processors (applied before LLM generation):         ║
║    ├── SentenceTransformerRerank (top-20 → top-5)              ║
║    └── MetadataReplacementPostProcessor (expand context)        ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📁 Folder Structure

```
project21_llamaindex_agent/
├── README.md
├── GUIDE.md
├── starter/
│   ├── requirements.txt
│   ├── .env.example
│   └── src/
│       ├── config.py                   ← given
│       ├── indexing/
│       │   ├── vector_index.py         ← TODO (8 tasks) — FAISS VectorStoreIndex
│       │   ├── summary_index.py        ← TODO (5 tasks) — SummaryIndex
│       │   └── pipeline.py             ← TODO (7 tasks) — IngestionPipeline + cache
│       ├── query/
│       │   ├── sub_question.py         ← TODO (5 tasks) — SubQuestionQueryEngine
│       │   ├── router.py               ← TODO (5 tasks) — RouterQueryEngine
│       │   ├── postprocessors.py       ← TODO (4 tasks) — reranker + metadata
│       │   └── prompts.py              ← TODO (3 tasks) — custom prompt templates
│       ├── agent/
│       │   ├── agent.py                ← TODO (5 tasks) — ReActAgent
│       │   └── tools.py                ← TODO (4 tasks) — QueryEngineTool wrappers
│       └── main.py                     ← TODO (3 tasks)
└── solution/
    └── src/
```

---

## ⚡ Key LlamaIndex Patterns

| Pattern | Code | Why |
|---------|------|-----|
| Build index | `VectorStoreIndex.from_documents(docs, embed_model=embed)` | Embed + store |
| Persist | `index.storage_context.persist("./storage")` | Avoid re-indexing |
| Load | `load_index_from_storage(StorageContext.from_defaults(persist_dir=...))` | Resume |
| Query engine | `index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])` | RAG |
| Sub-question | `SubQuestionQueryEngine.from_defaults(query_engine_tools=[...])` | Decompose |
| Router | `RouterQueryEngine.from_defaults(selector=LLMMultiSelector.from_defaults(), query_engine_tools=[...])` | Route |
| Agent | `ReActAgent.from_tools(tools, llm=llm, verbose=True)` | Tool-use loop |
| Wrap as tool | `QueryEngineTool(query_engine, metadata=ToolMetadata(name=..., description=...))` | Agent-ready |

---

## 🚀 Quick Start

```bash
cd projects/project21_llamaindex_agent/starter
pip install -r requirements.txt
cp .env.example .env

# Ingest documents
python -m src.indexing.pipeline --source data/docs/

# Test sub-question engine
python -m src.query.sub_question "Compare the revenue growth of Apple and Microsoft in 2023"

# Run the agent
python -m src.main "Summarize the key findings and compare them with last year"
```

---

## Milestones

1. **Ingestion pipeline** — `pipeline.py`, verify chunks + metadata
2. **Vector + Summary indexes** — build, persist, load, query each
3. **Sub-question engine** — test with multi-part question
4. **Router engine** — verify correct index selection
5. **Node post-processors** — add reranker, measure quality delta
6. **ReAct agent** — combine all engines as tools
