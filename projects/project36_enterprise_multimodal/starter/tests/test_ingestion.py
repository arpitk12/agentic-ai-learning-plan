"""
tests/test_ingestion.py — Unit tests for the ingestion pipeline.
Run with: pytest tests/test_ingestion.py -v
"""
import pytest
import os
import tempfile


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a minimal test PDF using ReportLab or fpdf2."""
    try:
        from reportlab.pdfgen import canvas  # type: ignore
        pdf_path = tmp_path / "test.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "GDPR Data Processing Agreement")
        c.drawString(100, 720, "This DPA is between Acme Corp and AWS.")
        c.drawString(100, 690, "Governed by GDPR Article 28. Value: $500,000.")
        c.save()
        return str(pdf_path)
    except ImportError:
        pytest.skip("reportlab not installed — install to run PDF tests")


# ── TODO: Write these test cases ─────────────────────────────────────────────

class TestPdfExtractor:
    def test_extract_text_chunks_returns_list(self, sample_pdf_path):
        """extract_text_chunks should return a non-empty list."""
        from src.ingestion.pdf_extractor import extract_text_chunks
        # TODO: call extract_text_chunks and assert:
        # - result is a list
        # - each item has "text", "page", "chunk_idx", "source" keys
        # - text is non-empty
        raise NotImplementedError

    def test_chunk_size_respected(self, sample_pdf_path):
        """Chunks should not exceed chunk_size characters."""
        from src.ingestion.pdf_extractor import extract_text_chunks
        # TODO: call with chunk_size=100 and assert all chunks <= 120 chars
        # (some tolerance for word boundary splitting)
        raise NotImplementedError

    def test_extract_images_returns_list(self, sample_pdf_path):
        """extract_images should return a list (empty is fine for text-only PDF)."""
        from src.ingestion.pdf_extractor import extract_images
        # TODO: call extract_images and assert result is a list
        raise NotImplementedError

    def test_extract_images_filters_tiny(self, sample_pdf_path):
        """Images smaller than 50×50 should be filtered out."""
        from src.ingestion.pdf_extractor import extract_images
        # TODO: assert no image in result has width < 50 or height < 50
        raise NotImplementedError


class TestAudioTranscriber:
    def test_chunk_transcript_no_segments(self):
        """chunk_transcript should handle transcripts with no segment info."""
        from src.ingestion.audio_transcriber import chunk_transcript
        transcript = {"text": "This is a long compliance discussion " * 20, "segments": []}
        # TODO: call chunk_transcript and assert:
        # - result is a non-empty list
        # - each item has "text" and "source" keys
        raise NotImplementedError

    def test_chunk_transcript_with_segments(self):
        """chunk_transcript should use segment boundaries."""
        from src.ingestion.audio_transcriber import chunk_transcript
        transcript = {
            "text": "First segment. Second segment. Third segment.",
            "segments": [
                {"text": "First segment.", "start": 0.0, "end": 2.0},
                {"text": "Second segment.", "start": 2.0, "end": 4.5},
                {"text": "Third segment.", "start": 4.5, "end": 7.0},
            ],
            "source": "test.mp3",
        }
        # TODO: call and assert each chunk has start_time and end_time
        raise NotImplementedError
