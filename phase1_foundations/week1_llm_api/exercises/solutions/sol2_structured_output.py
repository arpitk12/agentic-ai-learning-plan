"""
SOLUTION — Exercise 2: Structured Output with Pydantic + Auto-Retry
"""
import json
from pydantic import BaseModel, ValidationError
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()


class ResearchSummary(BaseModel):
    topic: str
    key_points: list[str]   # exactly 3 points
    confidence: float        # 0.0 to 1.0
    sources_needed: bool


SYSTEM_PROMPT = """You are a research assistant.
Always respond with valid JSON matching this schema:
{
  "topic": "string",
  "key_points": ["point1", "point2", "point3"],
  "confidence": 0.0,
  "sources_needed": true
}
Rules: exactly 3 key_points, confidence between 0.0 and 1.0.
No markdown, no preamble. JSON only."""


def get_structured_summary(topic: str, max_retries: int = 3) -> ResearchSummary:
    messages = [{"role": "user", "content": f"Summarize what you know about: {topic}"}]

    for attempt in range(max_retries):
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        raw = response.content[0].text.strip()

        try:
            data = json.loads(raw)
            result = ResearchSummary(**data)
            print(f"✓ Parsed on attempt {attempt + 1}")
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            # Push the bad response and error into conversation so the model can self-correct
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That response was invalid. Error: {e}\nPlease try again with only valid JSON."
            })

    raise RuntimeError(f"Failed to get valid structured output after {max_retries} attempts")


if __name__ == "__main__":
    result = get_structured_summary("quantum computing")
    print(result.model_dump_json(indent=2))
