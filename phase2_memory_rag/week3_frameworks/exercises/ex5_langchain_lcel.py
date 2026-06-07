"""
Exercise 5: LangChain LCEL Chains & Document Loaders
Guide Section: §2.3 — LangChain: Pipelines & Integrations

Goal: Understand LangChain's pipeline syntax and build a document Q&A chain.

Key Concepts:
- LCEL (LangChain Expression Language): uses | (pipe) to compose chains
- Document Loaders: 300+ connectors for PDF, HTML, CSV, databases, etc.
- Text Splitters: split documents into chunks for embedding
- Retrieval Chain: retrieve → inject context → generate grounded answer

Why LangChain vs raw llm.py?
- Pre-built connectors: load a PDF in 2 lines, not 20
- Pipeline composition: chain components with | operator
- Ecosystem: integrates with every vector DB, embedding model, and tool
- When to prefer raw llm.py: you need fine-grained control, no external deps

pip install langchain langchain-community langchain-core
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from dotenv import load_dotenv
load_dotenv()

import re, json, time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from llm import chat, get_text


# ─── PART 1: The LCEL Pipeline Pattern ────────────────────────────────────────
# LCEL uses the pipe | operator: step1 | step2 | step3
# Each step takes the previous output as its input.
# We simulate this with a simple function-composition pipeline.

def part1_basic_chain():
    """
    Demonstrate a 3-step chain:
    input → expand_topic → format_as_bullets → add_summary
    
    In native LangChain:
        chain = prompt | ChatOpenAI() | StrOutputParser()
        result = chain.invoke({"topic": "RAG"})
    
    We build the same idea with our llm.py wrapper.
    """
    print("\n" + "="*55)
    print("PART 1: LCEL-Style Pipeline")
    print("="*55)

    def step_explain(topic: str) -> str:
        """Step 1: Generate explanation."""
        return get_text(chat([{
            "role": "user",
            "content": f"Explain '{topic}' in 3 concise sentences for a senior engineer."
        }]))

    def step_key_points(explanation: str) -> list[str]:
        """Step 2: Extract key points (receives output of step 1)."""
        raw = get_text(chat([{
            "role": "user",
            "content": (
                f"From this explanation, extract exactly 3 key points as a JSON array:\n"
                f"{explanation}\n\nOutput ONLY valid JSON: [\"point1\", \"point2\", \"point3\"]"
            )
        }]))
        clean = re.sub(r"```json?\s*|\s*```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return [explanation]  # fallback

    def step_format(points: list[str]) -> str:
        """Step 3: Format into readable output (receives output of step 2)."""
        return "\n".join(f"  {i+1}. {p}" for i, p in enumerate(points))

    # Chain: topic → explanation → points → formatted output
    # This is what chain.invoke({"topic": "..."}) does internally
    topics = ["Retrieval-Augmented Generation (RAG)", "LangGraph StateGraph"]
    for topic in topics:
        print(f"\nTopic: {topic}")
        explanation = step_explain(topic)
        points = step_key_points(explanation)
        output = step_format(points)
        print(f"Key Points:\n{output}")


# ─── PART 2: Document Loading ──────────────────────────────────────────────────
# LangChain's biggest value: 300+ document loaders that handle parsing for you.
# We implement the same loaders manually so you see exactly what they do.

def load_text_file(path: str) -> list[dict]:
    """Load a plain text or Markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [{"content": content, "source": path, "type": "text"}]


def load_web_page(url: str) -> list[dict]:
    """
    Load and clean a web page — equivalent to LangChain's WebBaseLoader.
    Removes scripts, nav, ads, and collapses whitespace.
    
    pip install requests beautifulsoup4
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements (same as LangChain's WebBaseLoader)
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return [{"content": text, "source": url, "type": "web"}]
    except Exception as e:
        print(f"  ⚠️  Web load failed: {e}")
        return []


def load_directory(dir_path: str, extensions: tuple = (".txt", ".md")) -> list[dict]:
    """
    Load all matching files from a directory.
    Equivalent to LangChain's DirectoryLoader.
    """
    docs = []
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(extensions):
            full_path = os.path.join(dir_path, filename)
            docs.extend(load_text_file(full_path))
    return docs


# ─── PART 3: Text Splitting ────────────────────────────────────────────────────
# Chunks must be small enough for precise retrieval but large enough for context.
# RecursiveCharacterTextSplitter is the recommended default.

def split_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Split documents into overlapping chunks.
    
    RecursiveCharacterTextSplitter priority:
    \\n\\n (paragraphs) → \\n (lines) → ". " (sentences) → " " (words) → ""
    It tries the highest-level separator first, falls back to smaller ones.
    This preserves semantic boundaries as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        raw_chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(raw_chunks):
            if len(chunk.strip()) > 30:  # skip tiny fragments
                chunks.append({
                    "content": chunk,
                    "source": doc["source"],
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                })

    return chunks


# ─── PART 4: Retrieval Chain ───────────────────────────────────────────────────
# Full chain: question → retrieve context → build prompt → generate answer
# This is what LangChain's create_retrieval_chain() does under the hood.

class DocumentQAChain:
    """
    A RAG chain built with our llm.py wrapper.
    
    Equivalent LangChain LCEL code:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatLiteLLM()
            | StrOutputParser()
        )
        result = chain.invoke("What is RAG?")
    """

    def __init__(self, collection_name: str = "langchain_lcel_demo"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client()
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._doc_count = 0

    def add_documents(self, chunks: list[dict]) -> int:
        """Embed and index document chunks."""
        if not chunks:
            return 0
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        ids = [f"chunk_{self._doc_count + i}" for i in range(len(chunks))]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        self._doc_count += len(chunks)
        return len(chunks)

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        """Step 1 of the chain: embed question → vector search → return chunks."""
        q_vec = self.embedder.encode(question, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        return [
            {
                "content": doc,
                "score": round(1 - dist, 4),
                "source": meta.get("source", "?"),
            }
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]

    def invoke(self, question: str, k: int = 3) -> dict:
        """
        Full chain invocation.
        Equivalent to: chain.invoke(question)
        """
        # Retrieve
        chunks = self.retrieve(question, k=k)
        context = "\n\n---\n\n".join(
            f"[Source {i+1} | score={c['score']}]\n{c['content']}"
            for i, c in enumerate(chunks)
        )

        # Generate (grounded answer with citation rules)
        answer = get_text(chat(
            messages=[{"role": "user", "content": (
                f"Answer using ONLY the sources below. If not found, say so.\n"
                f"Always cite [Source N].\n\nSources:\n{context}\n\nQuestion: {question}"
            )}],
            system="You are a precise assistant. Answer only from provided sources. Always cite [Source N].",
        ))

        return {
            "answer": answer,
            "sources": [c["source"] for c in chunks],
            "scores": [c["score"] for c in chunks],
        }

    def batch(self, questions: list[str]) -> list[dict]:
        """
        Process multiple questions — equivalent to chain.batch([...]).
        Add asyncio.gather() for true parallel processing.
        """
        return [self.invoke(q) for q in questions]

    def stream(self, question: str):
        """
        Streaming — equivalent to chain.stream(question).
        Yields tokens one by one for real-time UI updates.
        """
        import litellm
        chunks = self.retrieve(question, k=3)
        context = "\n\n".join(c["content"] for c in chunks)
        messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]

        # Use LiteLLM streaming
        for chunk in litellm.completion(
            model=os.getenv("MODEL", "gemini/gemini-2.0-flash"),
            messages=messages,
            stream=True,
            max_tokens=500,
        ):
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token


# ─── MAIN ─────────────────────────────────────────────────────────────────────

SAMPLE_KNOWLEDGE_BASE = """
LangChain is a framework for developing applications powered by large language models.
Its core feature is the LangChain Expression Language (LCEL), which uses the pipe
operator | to compose processing steps into chains. Each step receives the output
of the previous step. Example: prompt | model | parser.

LangGraph is a library built on LangChain for creating stateful, multi-actor workflows.
Unlike linear LangChain chains, LangGraph supports cycles and conditional routing using
directed graphs. Each node is a Python function; edges define the flow between nodes.
LangGraph uses TypedDict for explicit state schemas and supports checkpointing.

ChromaDB is an open-source embedding database that runs locally. It stores documents
alongside their vector embeddings and supports semantic search using cosine similarity.
ChromaDB uses HNSW indexing for efficient approximate nearest neighbor search.
It is the recommended choice for development and prototyping of RAG systems.

RAG (Retrieval-Augmented Generation) is a technique that improves LLM accuracy by
retrieving relevant documents from a knowledge base and including them in the prompt.
The retrieval step uses vector similarity search to find semantically related content.
RAG reduces hallucination and allows LLMs to answer questions about private data.

The Transformer architecture introduced in 2017 uses self-attention mechanisms to
process sequences in parallel. Each token attends to all other tokens simultaneously,
enabling the model to capture long-range dependencies. Transformers are the foundation
of GPT, BERT, Claude, Gemini, and virtually every modern LLM.
"""


if __name__ == "__main__":
    # Part 1: Chain composition demo
    part1_basic_chain()

    # Part 2 & 3: Load and split documents
    print("\n" + "="*55)
    print("PART 2-3: Document Loading & Splitting")
    print("="*55)

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_KNOWLEDGE_BASE)
        tmp = f.name

    docs = load_text_file(tmp)
    print(f"Loaded {len(docs)} document(s) ({sum(len(d['content']) for d in docs)} chars)")

    chunks = split_documents(docs, chunk_size=400, chunk_overlap=60)
    print(f"Split into {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {len(c['content'])} chars | {c['content'][:70]}...")

    # Part 4: RAG chain
    print("\n" + "="*55)
    print("PART 4: Retrieval Chain (RAG)")
    print("="*55)

    chain = DocumentQAChain()
    n = chain.add_documents(chunks)
    print(f"Indexed {n} chunks\n")

    questions = [
        "What is LCEL and how does it work?",
        "How does LangGraph differ from a simple LangChain chain?",
        "Why is ChromaDB used for development instead of production?",
    ]

    for q in questions:
        result = chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {result['answer'][:300]}")
        print(f"   Sources: {result['sources'][:2]} | Scores: {result['scores'][:2]}")
        print()

    # Streaming demo
    print("STREAMING DEMO — tokens arrive one by one:")
    print("Q: What is RAG and why does it reduce hallucination?")
    print("A: ", end="", flush=True)
    try:
        for token in chain.stream("What is RAG and why does it reduce hallucination?"):
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"(streaming error: {e})")

    os.unlink(tmp)

    # ─── CHALLENGES ───────────────────────────────────────────────────────────
    # CHALLENGE 1: Load a real PDF using pypdf
    #   pip install pypdf
    #   from pypdf import PdfReader
    #   reader = PdfReader("paper.pdf")
    #   text = "\n".join(p.extract_text() for p in reader.pages)
    #
    # CHALLENGE 2: Load a web page and build a RAG chain over it
    #   docs = load_web_page("https://docs.python.org/3/library/asyncio.html")
    #
    # CHALLENGE 3: Add metadata filtering to the retriever
    #   collection.query(..., where={"source": {"$eq": "specific_file.txt"}})
    #
    # CHALLENGE 4: Compare chunk_size=200 vs 800 vs 2000 on retrieval quality
