"""
solution/src/ingestion/vision_analyzer.py — Full implementation.
"""
from __future__ import annotations
import asyncio
import base64
import json
import litellm  # type: ignore


async def analyze_image(
    image_bytes: bytes,
    ext: str = "png",
    context: str = "",
    model: str = "openai/gpt-4o",
) -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/{ext};base64,{b64}"
    prompt = (
        f"Analyze this compliance document image.{' Context: ' + context if context else ''}\n"
        "Return JSON with keys:\n"
        '  "type": "table"|"chart"|"diagram"|"photo"|"signature"|"other"\n'
        '  "description": one-sentence description\n'
        '  "key_data": extracted text/numbers from tables or charts (empty string if none)\n'
        '  "compliance_relevance": why this image matters for compliance analysis'
    )
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=400,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"type": "other", "description": f"Analysis failed: {e}",
                "key_data": "", "compliance_relevance": ""}


async def analyze_images_batch(
    images: list[dict],
    model: str = "openai/gpt-4o",
    max_concurrent: int = 5,
) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrent)

    async def _analyze(img: dict) -> dict:
        async with sem:
            try:
                result = await analyze_image(img["bytes"], img.get("ext", "png"), model=model)
                return {**img, **result}
            except Exception:
                return {**img, "type": "other", "description": "", "key_data": "",
                        "compliance_relevance": ""}

    return await asyncio.gather(*[_analyze(img) for img in images])
