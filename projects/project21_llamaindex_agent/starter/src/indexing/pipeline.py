"""Starter stub — Project 21: LlamaIndex ingestion pipeline."""
from __future__ import annotations

from pathlib import Path


def configure_settings() -> None:
    """Configure global LlamaIndex Settings (LLM + embed model).

    Settings is a global singleton — set it once before using any index.
    """
    # TODO 1: from llama_index.core import Settings
    # TODO 2: from llama_index.llms.litellm import LiteLLM
    # TODO 3: from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # TODO 4: Set Settings.llm = LiteLLM(model=..., api_base=...)
    # TODO 5: Set Settings.embed_model = HuggingFaceEmbedding(model_name=...)
    # TODO 6: Set Settings.chunk_size = 512, Settings.chunk_overlap = 64
    raise NotImplementedError


def build_pipeline():
    """Build an IngestionPipeline with caching and metadata extraction.

    Pipeline steps: SentenceSplitter → TitleExtractor → QuestionsAnsweredExtractor → embed_model
    """
    # TODO 7: from llama_index.core.ingestion import IngestionPipeline, IngestionCache
    # TODO 8: from llama_index.core.node_parser import SentenceSplitter
    # TODO 9: from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
    # TODO 10: Return IngestionPipeline(transformations=[...], cache=IngestionCache(...))
    raise NotImplementedError


def load_or_build_index(docs_path: str):
    """Load existing vector index from disk, or build and persist a new one.

    Pattern:
    1. Check if storage_path exists and is non-empty
    2. If yes: load with StorageContext.from_defaults + load_index_from_storage
    3. If no: SimpleDirectoryReader → pipeline.run() → VectorStoreIndex(nodes) → persist
    """
    # TODO 11: configure_settings()
    # TODO 12: Check Path(PERSIST_DIR / "vector").exists()
    # TODO 13: Load existing OR build new index
    # TODO 14: Return the VectorStoreIndex
    raise NotImplementedError
