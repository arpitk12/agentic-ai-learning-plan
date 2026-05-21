"""
SOLUTION — Exercise 1: Basic RAG Pipeline
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv
from llm import chat, get_text

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()


def load_pdf(path: str) -> list[str]:
    reader = PdfReader(path)
    return [page.extract_text() for page in reader.pages if page.extract_text()]


def chunk_text(pages: list[str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    for page in pages:
        start = 0
        while start < len(page):
            end = start + chunk_size
            chunks.append(page[start:end].strip())
            start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]


def build_index(chunks: list[str], collection_name: str = "docs") -> chromadb.Collection:
    collection = chroma.get_or_create_collection(collection_name)
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection


def retrieve(query: str, collection: chromadb.Collection, top_k: int = 3) -> list[str]:
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]


def answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    r = chat(
        [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        system=(
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "If the answer is not in the context, say 'I don't know based on the provided documents.'"
        ),
        max_tokens=1024,
    )
    return get_text(r)


def rag_pipeline(pdf_path: str, question: str) -> str:
    pages = load_pdf(pdf_path)
    chunks = chunk_text(pages)
    print(f"Loaded {len(pages)} pages → {len(chunks)} chunks")

    collection = build_index(chunks)
    relevant = retrieve(question, collection)
    print(f"Retrieved {len(relevant)} chunks for: '{question}'")

    return answer(question, relevant)


if __name__ == "__main__":
    # Create a small sample text file for testing (since we may not have a PDF)
    import tempfile, os
    sample = """
    Artificial Intelligence and Machine Learning

    Machine learning is a type of artificial intelligence that allows computers to learn from data
    without being explicitly programmed. Deep learning uses neural networks with many layers to
    learn complex patterns. Natural language processing (NLP) enables computers to understand and
    generate human language. Large language models like GPT and Claude are trained on vast amounts
    of text data using transformer architectures. Retrieval augmented generation (RAG) combines
    a retrieval system with a language model to answer questions from specific documents.
    """
    # Write to temp file and use it
    print("RAG Pipeline ready. Provide a PDF path to test.")
    print("Example: rag_pipeline('my_document.pdf', 'What is machine learning?')")
