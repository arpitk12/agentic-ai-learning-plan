"""
Exercise 1: Basic RAG Pipeline
Goal: Ingest a PDF, embed it, store in Chroma, and answer questions.

pip install chromadb sentence-transformers pypdf litellm python-dotenv
"""
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from llm import chat, get_text
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()


def load_pdf(path: str) -> list[str]:
    """Extract text from PDF, return list of page texts."""
    reader = PdfReader(path)
    return [page.extract_text() for page in reader.pages if page.extract_text()]


def chunk_text(pages: list[str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """TODO: Split pages into overlapping chunks of ~chunk_size characters."""
    chunks = []
    for page in pages:
        # TODO: implement sliding window chunking
        pass
    return chunks


def build_index(chunks: list[str], collection_name: str = "docs") -> chromadb.Collection:
    """TODO: Embed chunks and upsert them into a Chroma collection."""
    collection = chroma.get_or_create_collection(collection_name)
    # TODO: embed and store
    return collection


def retrieve(query: str, collection: chromadb.Collection, top_k: int = 3) -> list[str]:
    """TODO: Embed the query and return the top_k most similar chunks."""
    return []


def answer(question: str, context_chunks: list[str]) -> str:
    """TODO: Build a context-augmented prompt and call the LLM."""
    context = "\n\n---\n\n".join(context_chunks)
    return "Not implemented"


def rag_pipeline(pdf_path: str, question: str) -> str:
    pages = load_pdf(pdf_path)
    chunks = chunk_text(pages)
    print(f"Loaded {len(pages)} pages → {len(chunks)} chunks")

    collection = build_index(chunks)
    context = retrieve(question, collection)
    return answer(question, context)


if __name__ == "__main__":
    result = rag_pipeline("gpt4_paper.pdf", "What are the key capabilities of GPT-4?")
    print(result)
