"""Main entry point for project 18 — LangChain Research Agent."""
from __future__ import annotations

import argparse
import logging

from src.config import cfg

logging.basicConfig(level=cfg.log_level)


def build_faiss_index(docs_path: str) -> None:
    """Ingest documents from a directory into a local FAISS vector store."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"Loading documents from {docs_path}...")
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(cfg.faiss_index_path)
    print(f"FAISS index saved to {cfg.faiss_index_path}")


def run_chat(use_streaming: bool = False) -> None:
    """Launch the interactive research agent REPL."""
    from src.agent.agent import run_interactive, stream_agent_response
    import asyncio

    if use_streaming:
        asyncio.run(stream_agent_response("Tell me about quantum computing"))
    else:
        run_interactive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain Research Agent")
    sub = parser.add_subparsers(dest="cmd")

    ingest_p = sub.add_parser("ingest", help="Build FAISS index from documents")
    ingest_p.add_argument("path", help="Directory containing .txt documents")

    chat_p = sub.add_parser("chat", help="Start interactive research agent")
    chat_p.add_argument("--stream", action="store_true", help="Use streaming output")

    args = parser.parse_args()
    if args.cmd == "ingest":
        build_faiss_index(args.path)
    elif args.cmd == "chat":
        run_chat(args.stream)
    else:
        parser.print_help()
