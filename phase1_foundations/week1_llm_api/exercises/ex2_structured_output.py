"""
Exercise 2: Structured Output with Pydantic
Goal: Force the LLM to return JSON that matches a schema. Retry on failure.

Uses llm.py — works with Ollama (local) or any cloud model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import json
from pydantic import BaseModel, ValidationError
from llm import chat, get_text


class ResearchSummary(BaseModel):
    topic: str
    key_points: list[str]  # exactly 3 points
    confidence: float       # 0.0 to 1.0
    sources_needed: bool


SYSTEM_PROMPT = """You are a research assistant. 
Always respond with valid JSON matching this schema:
{
  "topic": "string",
  "key_points": ["point1", "point2", "point3"],
  "confidence": 0.0-1.0,
  "sources_needed": true/false
}
No markdown, no preamble. JSON only."""


def get_structured_summary(topic: str, max_retries: int = 3) -> ResearchSummary:
    for attempt in range(max_retries):
        # TODO: Call the API with SYSTEM_PROMPT
        response=chat(
            messages=[{"role": "user", "content": f"Summarize what you know about: {topic}"}],
            system=SYSTEM_PROMPT,
        )   
        # TODO: Parse the response as JSON
        raw=get_text(response=response).strip()
        try:
            data=json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"✗ Attempt {attempt + 1} failed to parse JSON: {e}")
            # Push the bad response and error into conversation so the model can self-correct
            continue

        # TODO: Validate against ResearchSummary, retry with error context on failure
        try:
            summary = ResearchSummary(**data)
            return summary
        except ValidationError as e:
            print(f"✗ Attempt {attempt + 1} failed validation: {e}")
            # Push the validation error into conversation so the model can self-correct
            continue

    raise RuntimeError(f"Failed after {max_retries} attempts")


if __name__ == "__main__":
    result = get_structured_summary("quantum computing")
    print(result.model_dump_json(indent=2))
