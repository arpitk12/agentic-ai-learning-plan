"""LlamaIndex ingestion pipeline — documents → vector + summary indexes."""
from __future__ import annotations

import os
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.extractors import QuestionsAnsweredExtractor, TitleExtractor
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.litellm import LiteLLM


_PERSIST_DIR = os.getenv("PERSIST_DIR", "data/index_storage")


def configure_settings() -> None:
    """Set global LlamaIndex settings (LLM + embeddings)."""
    Settings.llm = LiteLLM(
        model=os.getenv("MODEL", "openai/gpt-4o-mini"),
        api_base=os.getenv("LITELLM_API_BASE"),
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64


def build_pipeline() -> IngestionPipeline:
    """Build an IngestionPipeline with caching, splitting, and metadata extraction."""
    cache = IngestionCache(cache_path=f"{_PERSIST_DIR}/ingest_cache")
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=64),
            TitleExtractor(nodes=5),
            QuestionsAnsweredExtractor(questions=3),
            Settings.embed_model,
        ],
        cache=cache,
    )


def load_or_build_index(docs_path: str) -> VectorStoreIndex:
    """Load existing index from disk or build a new one from documents."""
    configure_settings()
    storage_path = Path(_PERSIST_DIR) / "vector"

    if storage_path.exists() and any(storage_path.iterdir()):
        print(f"Loading existing index from {storage_path}")
        storage_ctx = StorageContext.from_defaults(persist_dir=str(storage_path))
        return load_index_from_storage(storage_ctx)

    print(f"Building new index from {docs_path}")
    documents = SimpleDirectoryReader(docs_path, recursive=True).load_data()
    print(f"Loaded {len(documents)} documents")

    pipeline = build_pipeline()
    nodes = pipeline.run(documents=documents)
    print(f"Generated {len(nodes)} nodes")

    index = VectorStoreIndex(nodes)
    storage_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(storage_path))
    print(f"Index persisted to {storage_path}")
    return index
