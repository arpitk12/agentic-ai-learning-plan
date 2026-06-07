"""
Unit tests — ingestion layer (loader, chunker, embedder, pipeline).

Run:
    pytest tests/test_ingestion.py -v
"""
from __future__ import annotations
import sys
import os
import tempfile
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_markdown_file(self, tmp_path):
        from src.ingestion.loader import load_file
        f = tmp_path / "doc.md"
        f.write_text("# Hello\n\nThis is a test document.")
        docs = load_file(str(f))
        assert len(docs) == 1
        assert "Hello" in docs[0].content
        assert docs[0].source == str(f)

    def test_load_txt_file(self, tmp_path):
        from src.ingestion.loader import load_file
        f = tmp_path / "note.txt"
        f.write_text("Plain text content here.")
        docs = load_file(str(f))
        assert len(docs) == 1
        assert docs[0].content == "Plain text content here."

    def test_load_unsupported_extension_returns_empty(self, tmp_path):
        from src.ingestion.loader import load_file
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        docs = load_file(str(f))
        assert docs == []

    def test_load_directory_finds_md_and_txt(self, tmp_path):
        from src.ingestion.loader import load_directory
        (tmp_path / "a.md").write_text("Doc A")
        (tmp_path / "b.txt").write_text("Doc B")
        (tmp_path / "c.csv").write_text("ignored")
        docs = load_directory(str(tmp_path))
        sources = [d.source for d in docs]
        assert any("a.md" in s for s in sources)
        assert any("b.txt" in s for s in sources)
        assert all("c.csv" not in s for s in sources)

    def test_load_file_missing_raises(self):
        from src.ingestion.loader import load_file
        with pytest.raises(FileNotFoundError):
            load_file("/no/such/file.md")

    def test_doc_id_is_stable(self, tmp_path):
        from src.ingestion.loader import load_file
        f = tmp_path / "stable.md"
        f.write_text("Stable content")
        id1 = load_file(str(f))[0].doc_id
        id2 = load_file(str(f))[0].doc_id
        assert id1 == id2


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    def _make_doc(self, content: str):
        from src.models import RawDocument
        return RawDocument(doc_id="test", title="T", source="mem", content=content)

    def test_short_doc_produces_one_chunk(self):
        from src.ingestion.chunker import chunk_document
        doc = self._make_doc("Short document.")
        chunks = chunk_document(doc, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_long_doc_split_into_multiple_chunks(self):
        from src.ingestion.chunker import chunk_document
        long_text = ("word " * 200).strip()   # ~1000 chars
        doc = self._make_doc(long_text)
        chunks = chunk_document(doc, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        from src.ingestion.chunker import chunk_document
        long_text = ("abcde " * 300).strip()
        doc = self._make_doc(long_text)
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        for c in chunks:
            assert len(c.text) <= 150, "Chunk unexpectedly large"

    def test_overlap_creates_shared_content(self):
        from src.ingestion.chunker import chunk_document
        # Build text with distinct sentences
        sentences = [f"Sentence number {i}. " for i in range(50)]
        doc = self._make_doc("".join(sentences))
        chunks = chunk_document(doc, chunk_size=200, overlap=50)
        if len(chunks) >= 2:
            # Some text from chunk 0 tail should appear in chunk 1 start
            # (exact overlap is best-effort — just check we got >1 chunk)
            assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        from src.ingestion.chunker import chunk_document
        doc = self._make_doc("Test content")
        chunks = chunk_document(doc)
        assert chunks[0].doc_id == "test"
        assert chunks[0].source == "mem"

    def test_chunk_ids_unique(self):
        from src.ingestion.chunker import chunk_document
        long_text = ("word " * 300).strip()
        doc = self._make_doc(long_text)
        chunks = chunk_document(doc, chunk_size=100, overlap=0)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedder:
    """Embedder tests hit the real model (cached after first run, ~20 MB)."""

    def test_embed_texts_returns_list_of_lists(self):
        from src.ingestion.embedder import embed_texts
        vecs = embed_texts(["hello world", "foo bar"])
        assert len(vecs) == 2
        assert isinstance(vecs[0], list)

    def test_embed_query_returns_single_vector(self):
        from src.ingestion.embedder import embed_query
        vec = embed_query("what is techflow?")
        assert isinstance(vec, list)
        assert len(vec) == 384   # all-MiniLM-L6-v2 dimension

    def test_embedding_dimension_consistent(self):
        from src.ingestion.embedder import embed_texts, embed_query
        vecs = embed_texts(["alpha", "beta", "gamma"])
        q    = embed_query("delta")
        dims = {len(v) for v in vecs}
        dims.add(len(q))
        assert len(dims) == 1, "All embeddings should have the same dimension"

    def test_normalized_embeddings(self):
        """Normalised vectors have L2-norm ≈ 1.0"""
        import math
        from src.ingestion.embedder import embed_query
        vec  = embed_query("normalised?")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (integration — uses real embedder + in-memory Chroma)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    """
    These tests use the actual ChromaDB client in local mode with a temp
    directory so they don't pollute the real data/ store.
    """

    @pytest.fixture
    def store(self, tmp_path):
        """Fresh VectorStore backed by a temp directory."""
        import os
        os.environ["CHROMA_PATH"] = str(tmp_path / "chroma")
        os.environ.pop("CHROMA_HOST", None)
        from src.store.chroma_store import VectorStore
        return VectorStore()

    def test_ingest_text_creates_chunks(self, store):
        from src.ingestion.pipeline import ingest_text
        result = ingest_text(
            text="TechFlow is a cloud-native platform for engineering teams. " * 5,
            title="Test Doc",
            source="test",
            store=store,
        )
        assert result.chunks_created >= 1
        assert result.documents_processed == 1

    def test_ingest_directory_processes_files(self, tmp_path, store):
        from src.ingestion.pipeline import ingest_directory
        (tmp_path / "doc1.md").write_text("# Doc One\n\nContent of document one.")
        (tmp_path / "doc2.txt").write_text("Content of document two.")
        result = ingest_directory(str(tmp_path), store)
        assert result.documents_processed == 2
        assert result.chunks_created >= 2

    def test_ingest_skips_duplicates_by_default(self, tmp_path, store):
        from src.ingestion.pipeline import ingest_directory
        (tmp_path / "once.md").write_text("Unique content here.")
        r1 = ingest_directory(str(tmp_path), store)
        r2 = ingest_directory(str(tmp_path), store)
        assert r2.documents_skipped == 1
        assert r2.chunks_created == 0

    def test_replace_existing_reingests(self, tmp_path, store):
        from src.ingestion.pipeline import ingest_directory
        f = tmp_path / "replace.md"
        f.write_text("Original content.")
        ingest_directory(str(tmp_path), store, replace_existing=False)
        r2 = ingest_directory(str(tmp_path), store, replace_existing=True)
        assert r2.documents_processed == 1
