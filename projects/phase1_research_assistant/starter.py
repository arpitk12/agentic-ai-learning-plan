"""
Project 1 Starter — Research Assistant CLI
Fill in the TODOs to complete the project.
"""
import json
import re
import httpx
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()


# --- Output Schema ---
class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_points: list[str]
    sources: list[str]


# --- Tool Schemas ---
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for recent information on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetch and return the text content of a URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"}
            },
            "required": ["url"]
        }
    }
]


# --- Tool Implementations ---
def web_search(query: str) -> str:
    # TODO: Use Tavily or SerpAPI to search
    # from tavily import TavilyClient
    # tc = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    # results = tc.search(query, max_results=3)
    # return json.dumps(results["results"])
    raise NotImplementedError


def fetch_url(url: str) -> str:
    # TODO: Use httpx to fetch URL content, strip HTML tags, return first 2000 chars
    raise NotImplementedError


def run_tool(name: str, inputs: dict) -> str:
    try:
        if name == "web_search":
            return web_search(inputs["query"])
        elif name == "fetch_url":
            return fetch_url(inputs["url"])
        else:
            return f"Error: unknown tool {name}"
    except Exception as e:
        return f"Tool error: {str(e)}"


# --- Agent ---
SYSTEM_PROMPT = """You are a research assistant. 
Given a topic, use web_search to find sources, fetch the top 3 URLs, 
then synthesize a report. When done, output ONLY valid JSON matching:
{"topic": "...", "summary": "...", "key_points": [...], "sources": [...]}"""


def research(topic: str) -> ResearchReport:
    messages = [{"role": "user", "content": f"Research this topic: {topic}"}]
    
    # TODO: Implement the ReAct loop
    # - Call API with TOOLS and SYSTEM_PROMPT
    # - Handle tool_use stops
    # - When end_turn, parse JSON from response
    # - Return ResearchReport(**data)
    
    raise NotImplementedError


# --- Main ---
if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "large language models 2025"
    
    print(f"Researching: {topic}\n")
    report = research(topic)
    
    filename = f"report_{topic[:30].replace(' ', '_')}.json"
    with open(filename, "w") as f:
        f.write(report.model_dump_json(indent=2))
    
    print(f"Saved to {filename}")
    print(report.model_dump_json(indent=2))
