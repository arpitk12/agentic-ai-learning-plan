# LlamaIndex Multi-Document RAG Agent — Build Guide

## Prerequisites
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # only if using spaCy splitter
```

---

## Phase 1 — Ingestion Pipeline

### 1.1 Load documents
```python
from llama_index.core import SimpleDirectoryReader

docs = SimpleDirectoryReader(
    input_dir="data/docs/",
    required_exts=[".md", ".txt", ".pdf"],
    file_metadata=lambda path: {
        "file_name": Path(path).name,
        "source": str(path),
    },
).load_data()
print(f"Loaded {len(docs)} documents")
```

### 1.2 IngestionPipeline — transformations with caching
```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        TitleExtractor(nodes=3),             # LLM infers title for each chunk group
        QuestionsAnsweredExtractor(questions=3),  # LLM generates 3 Q per chunk
        embed_model,
    ],
    cache=IngestionCache(),      # avoids re-processing unchanged documents
)

nodes = pipeline.run(documents=docs, show_progress=True)
print(f"Created {len(nodes)} nodes")
```

**Why QuestionsAnsweredExtractor?** Each chunk now has 3 questions it can answer stored
in metadata. This dramatically improves retrieval: the retriever can match against the
generated questions, not just the raw text.

### 1.3 Build and persist indexes
```python
from llama_index.core import VectorStoreIndex, SummaryIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Vector index (for factual Q&A)
dimension = 384   # BGE small
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

vector_index = VectorStoreIndex(nodes, storage_context=storage_ctx, embed_model=embed_model)
vector_index.storage_context.persist("storage/vector")

# Summary index (for summarisation)
summary_index = SummaryIndex(nodes)
summary_index.storage_context.persist("storage/summary")
```

**Checkpoint:** `python -m src.indexing.pipeline --source data/docs/ --verbose`

---

## Phase 2 — Query Engines

### 2.1 Basic query engine with reranker
```python
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5,
)

metadata_replacer = MetadataReplacementPostProcessor(
    target_metadata_key="window"  # expand chunk with surrounding sentences
)

vector_qe = vector_index.as_query_engine(
    similarity_top_k=20,         # retrieve 20 candidates
    node_postprocessors=[
        metadata_replacer,       # expand context
        reranker,                # rerank 20 → 5
    ],
    response_mode="compact",     # condense context into fewer LLM calls
)
```

### 2.2 SubQuestionQueryEngine
```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

tools = [
    QueryEngineTool(
        query_engine=vector_qe,
        metadata=ToolMetadata(
            name="knowledge_base",
            description="Answers factual questions about the documents",
        ),
    ),
]

sub_q_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    use_async=True,
    verbose=True,
)

# Automatically decomposes:
# "Compare the revenue and employee count of Apple and Google in 2023"
# → Sub-Q 1: "What is Apple's revenue in 2023?"
# → Sub-Q 2: "What is Google's revenue in 2023?"
# → Sub-Q 3: "How many employees does Apple have?"
# → Sub-Q 4: "How many employees does Google have?"
# → Synthesize all into final answer
response = sub_q_engine.query("Compare Apple and Google on revenue and employees")
```

### 2.3 RouterQueryEngine
```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector

summary_qe = summary_index.as_query_engine(response_mode="tree_summarize")

router_engine = RouterQueryEngine.from_defaults(
    selector=LLMMultiSelector.from_defaults(),   # LLM selects which engine
    query_engine_tools=[
        QueryEngineTool(
            query_engine=vector_qe,
            metadata=ToolMetadata(name="factual", description="Answers specific factual questions"),
        ),
        QueryEngineTool(
            query_engine=summary_qe,
            metadata=ToolMetadata(name="summarizer", description="Summarizes documents or topics"),
        ),
    ],
    verbose=True,
)

# Routes to summary_qe
router_engine.query("Summarize the main themes across all documents")

# Routes to vector_qe
router_engine.query("What is the exact API rate limit mentioned in the docs?")
```

**Checkpoint:** verify the router picks the correct engine for each query type.

---

## Phase 3 — ReAct Agent

### 3.1 Build the agent
```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.litellm import LiteLLM

llm = LiteLLM(model=cfg.model)

agent = ReActAgent.from_tools(
    tools=[
        QueryEngineTool(query_engine=vector_qe,
                        metadata=ToolMetadata(name="factual_search", description="...")),
        QueryEngineTool(query_engine=summary_qe,
                        metadata=ToolMetadata(name="document_summary", description="...")),
        QueryEngineTool(query_engine=sub_q_engine,
                        metadata=ToolMetadata(name="multi_part_research", description="...")),
    ],
    llm=llm,
    verbose=True,
    max_iterations=10,
    context="""You are a research assistant. Use the available tools to answer questions
    accurately. For complex multi-part questions, use multi_part_research.
    For summaries, use document_summary. For specific facts, use factual_search.""",
)

response = agent.chat("What are the key differences between the two approaches described, and which is recommended?")
print(response.response)
```

### 3.2 Streaming chat
```python
streaming_response = agent.stream_chat("Explain the main concepts...")
for token in streaming_response.response_gen:
    print(token, end="", flush=True)
```

---

## Framework Comparison

| | LangChain RAG | LlamaIndex RAG |
|---|---|---|
| Abstraction level | Low (you build the chain) | Higher (engines handle retrieval) |
| Index types | One (vector) | Multiple (vector, summary, tree, keyword) |
| Query decomposition | Manual (custom chain) | Built-in (SubQuestionQueryEngine) |
| Routing | RunnableBranch | RouterQueryEngine |
| Metadata extraction | Custom | Built-in extractors (Title, QA, Summary) |
| When to use | Custom pipelines | Document-centric Q&A |
