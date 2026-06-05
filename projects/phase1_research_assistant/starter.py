"""
Project 1 Starter — Research Assistant CLI
Fill in the TODOs to complete the project.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import re
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message, normalize_tools

load_dotenv()


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
    from tavily import TavilyClient
    tc = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tc.search(query, max_results=3)
    simplified = [{"title": r["title"], "url": r["url"], "snippet": r["content"][:200]}
                  for r in results["results"]]
    return json.dumps(simplified, indent=2)
    


def fetch_url(url: str) -> str:
    # TODO: Use httpx to fetch URL content, strip HTML tags, return first 2000 chars
    try:
        response=httpx.get(url,timeout=10,follow_redirects=True)
        text=response.text
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2001]
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


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
    max_step=12
    # TODO: Implement the ReAct loop
    for step in range(max_step):

        print(f"\n Step no {step+1}")
        
        response=chat(messages=messages,system=SYSTEM_PROMPT,tools=normalize_tools(TOOLS))

        messages.append(assistant_message(response))

        reason=stop_reason(response)
        if reason=="end_turn":
            text=get_text(response).strip()
            text = re.sub(r"```json|```", "", text).strip()
            data=json.loads(text)
            return ResearchReport(**data)
        elif reason=="tool_use":
            for tc in get_tool_calls(response):
                tool_res=run_tool(tc["name"],tc["arguments"])
                messages.append(tool_result_message(tc["id"],tool_res))  
        else:
            raise ValueError(f"Unexpected stop_reason: {stop_reason(response)}")
        
    raise RuntimeError("Max steps reached without completing research")
    # - Call API with TOOLS and SYSTEM_PROMPT
    # - Handle tool_use stops
    # - When end_turn, parse JSON from response
    # - Return ResearchReport(**data)
    
    


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
