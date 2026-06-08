"""
solution/src/ingestion/audio_transcriber.py — Full implementation.
"""
from __future__ import annotations
import openai  # type: ignore


def transcribe(audio_path: str, model: str = "whisper-1") -> dict:
    client = openai.OpenAI()
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
        )
    segments = []
    for seg in getattr(result, "segments", []) or []:
        segments.append({
            "text": seg.text if hasattr(seg, "text") else str(seg),
            "start": getattr(seg, "start", 0.0),
            "end": getattr(seg, "end", 0.0),
        })
    return {
        "text": result.text,
        "language": getattr(result, "language", "en"),
        "duration": getattr(result, "duration", 0.0),
        "segments": segments,
        "source": audio_path,
    }


def chunk_transcript(transcript: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    segments = transcript.get("segments", [])
    source = transcript.get("source", "")
    chunks: list[dict] = []

    if segments:
        # Segment-boundary chunking
        buf_texts, buf_start, buf_end = [], 0.0, 0.0
        for seg in segments:
            buf_texts.append(seg["text"])
            if buf_start == 0.0:
                buf_start = seg["start"]
            buf_end = seg["end"]
            combined = " ".join(buf_texts)
            if len(combined) >= chunk_size:
                chunks.append({"text": combined.strip(),
                                "start_time": buf_start, "end_time": buf_end,
                                "source": source})
                # Overlap: keep last segment
                buf_texts = buf_texts[-1:]
                buf_start = segments[segments.index(seg)]["start"] if seg in segments else buf_end
        if buf_texts:
            chunks.append({"text": " ".join(buf_texts).strip(),
                            "start_time": buf_start, "end_time": buf_end,
                            "source": source})
    else:
        # Character-level fallback
        text = transcript.get("text", "")
        start = 0
        while start < len(text):
            chunks.append({"text": text[start:start + chunk_size].strip(),
                            "start_time": 0.0, "end_time": 0.0,
                            "source": source})
            start += chunk_size - overlap

    return [c for c in chunks if c["text"]]
