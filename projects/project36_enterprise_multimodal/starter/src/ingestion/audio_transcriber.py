"""
src/ingestion/audio_transcriber.py
Transcribe audio files using OpenAI Whisper and chunk the result.

TODOs:
  1. implement transcribe() — call Whisper API, return full transcript + segments
  2. implement chunk_transcript() — split long transcript into overlapping text chunks
     suitable for vector indexing
"""
from __future__ import annotations


# ── TODO 1: Transcribe audio file ─────────────────────────────────────────────
def transcribe(audio_path: str, model: str = "whisper-1") -> dict:
    """
    Transcribe an audio file using the OpenAI Whisper API.

    Steps:
      1a. import openai; client = openai.OpenAI()
      1b. with open(audio_path, "rb") as f:
              result = client.audio.transcriptions.create(
                  model=model,
                  file=f,
                  response_format="verbose_json",   # includes word-level timestamps
              )
      1c. Return dict:
          {
            "text": result.text,               # full transcript
            "language": result.language,
            "duration": result.duration,
            "segments": [
                {"text": seg.text, "start": seg.start, "end": seg.end}
                for seg in result.segments
            ],
            "source": audio_path,
          }

    Note: verbose_json gives word-level timestamps. Use "json" if you only need text.
    """
    # import openai
    # ...
    raise NotImplementedError


# ── TODO 2: Chunk transcript for vector indexing ──────────────────────────────
def chunk_transcript(
    transcript: dict,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Split a transcript dict (from transcribe()) into overlapping text chunks.

    Steps:
      2a. Use segment boundaries when possible to avoid cutting mid-sentence
      2b. Accumulate segments until combined text exceeds chunk_size
      2c. When a chunk is full: record {"text", "start_time", "end_time", "source"}
      2d. Include overlap by carrying the last segment(s) into the next chunk
      2e. If no segments (verbose_json failed), fall back to character-level
          sliding window on transcript["text"]

    Returns:
        list[dict] — [{"text": str, "start_time": float, "end_time": float, "source": str}]
    """
    # segments = transcript.get("segments", [])
    # ...
    raise NotImplementedError
