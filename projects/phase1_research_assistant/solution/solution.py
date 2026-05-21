"""
Project 1 SOLUTION — Research Assistant CLI
Full working implementation.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
import re
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()


class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_points: list[str]
    sources: list[str]


TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for recent information on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetch and return the text content of a URL (first 2000 chars).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"}
            },
            "required": ["url"]
        }
    }
]

SYSTEM_PROMPT = """You are a research assistant.
1. Use web_search to find the top sources on the topic.
2. Use fetch_url on the 3 most relevant URLs.
3. After reading them, respond ONLY with valid JSON (no markdown):
{
  "topic": "...",
  "summary": "2-3 sentence summary",
  "key_points": ["point1", "point2", "point3", "point4", "point5"],
  "sources": ["url1", "url2", "url3"]
}"""


def web_search(query: str) -> str:
    from tavily import TavilyClient
    tc = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tc.search(query, max_results=5)
    simplified = [{"title": r["title"], "url": r["url"], "snippet": r["content"][:200]}
                  for r in results["results"]]
    return json.dumps(simplified, indent=2)


def fetch_url(url: str) -> str:
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        text = resp.text
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2000]
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


def run_tool(name: str, inputs: dict) -> str:
    print(f"  [TOOL] {name}({json.dumps(inputs)[:80]}...)")
    if name == "web_search":
        result = web_search(inputs["query"])
    elif name == "fetch_url":
        result = fetch_url(inputs["url"])
    else:
        result = f"Unknown tool: {name}"
    print(f"  [RESULT] {result[:100]}...")
    return result


def research(topic: str) -> ResearchReport:
    messages = [{"role": "user", "content": f"Research this topic thoroughly: {topic}"}]
    max_steps = 12

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        response = chat(messages, system=SYSTEM_PROMPT, max_tokens=4096, tools=TOOLS)

        # Append assistant response to history
        messages.append(assistant_message(response))

        if stop_reason(response) == "end_turn":
            # Extract JSON from final message
            text = get_text(response).strip()
            # Strip markdown fences if present
            text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)
            return ResearchReport(**data)

        elif stop_reason(response) == "tool_use":
            for tc in get_tool_calls(response):
                result = run_tool(tc["name"], tc["arguments"])
                messages.append(tool_result_message(tc["id"], result))
        else:
            raise ValueError(f"Unexpected stop_reason: {stop_reason(response)}")

    raise RuntimeError("Max steps reached without completing research")


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "agentic AI systems 2025"
    print(f"Researching: {topic}\n")

    report = research(topic)

    safe = re.sub(r"[^a-z0-9_]", "_", topic[:30].lower())
    filename = f"report_{safe}.json"
    with open(filename, "w") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"\nSaved to {filename}")
    print(report.model_dump_json(indent=2))
