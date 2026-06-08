"""
src/ingestion/vision_analyzer.py
Send extracted images to GPT-4o (or any vision-capable model) for structured analysis.

TODOs:
  1. implement analyze_image() — base64-encode bytes, call litellm vision API,
     parse structured JSON response
  2. implement analyze_images_batch() — run multiple images concurrently with
     asyncio.gather, respecting a concurrency limit
"""
from __future__ import annotations
import asyncio
import base64
import json


# ── TODO 1: Analyze a single image ────────────────────────────────────────────
async def analyze_image(
    image_bytes: bytes,
    ext: str = "png",
    context: str = "",
    model: str = "openai/gpt-4o",
) -> dict:
    """
    Analyze an image with a vision model.

    Steps:
      1a. base64.b64encode(image_bytes).decode() → b64_str
      1b. Build data URL: f"data:image/{ext};base64,{b64_str}"
      1c. litellm.acompletion with:
          messages=[{"role": "user", "content": [
              {"type": "text", "text": prompt},
              {"type": "image_url", "image_url": {"url": data_url}},
          ]}]
          response_format={"type": "json_object"}
      1d. Parse JSON: expected keys = {type, description, key_data, compliance_relevance}
          type: "table" | "chart" | "diagram" | "photo" | "signature" | "other"
          description: one-sentence description
          key_data: extracted text/numbers from tables or charts
          compliance_relevance: why this image matters for compliance

    Returns:
        dict with keys: type, description, key_data, compliance_relevance
    """
    # import litellm
    # prompt = f"Analyze this compliance document image...{context}"
    # ...
    raise NotImplementedError


# ── TODO 2: Batch analyze images concurrently ─────────────────────────────────
async def analyze_images_batch(
    images: list[dict],          # list of {bytes, ext, page, xref, source}
    model: str = "openai/gpt-4o",
    max_concurrent: int = 5,
) -> list[dict]:
    """
    Analyze multiple images concurrently with a semaphore to limit parallelism.

    Steps:
      2a. Create asyncio.Semaphore(max_concurrent)
      2b. Define async helper: async with sem: return await analyze_image(img["bytes"], ...)
      2c. asyncio.gather(*[helper(img) for img in images])
      2d. Merge vision result into each image dict, add "description" key
      2e. Skip images where analyze_image raises (log warning, continue)

    Returns:
        list[dict] — original image dicts with "description", "type",
                     "key_data", "compliance_relevance" added
    """
    # sem = asyncio.Semaphore(max_concurrent)
    # ...
    raise NotImplementedError
