# Project 1 — Research Assistant CLI

## Brief
Build a terminal research assistant that:
1. Accepts a topic from the user
2. Searches the web (Tavily or SerpAPI)
3. Fetches and reads 3 URLs
4. Returns a structured JSON report saved to disk

## Requirements
- [ ] Uses tool calling (search + fetch tools)
- [ ] Multi-step ReAct loop (no LangChain)
- [ ] Output is a valid JSON file with: topic, summary, key_points, sources
- [ ] Graceful failure if a URL is unreachable
- [ ] Logs each step (tool call + result) to terminal

## Setup
```bash
pip install anthropic httpx tavily-python python-dotenv pydantic
```

Set in `.env`:
```
ANTHROPIC_API_KEY=your_key
TAVILY_API_KEY=your_key  # free tier at tavily.com
```

## Usage
```bash
python starter.py "latest advances in quantum computing"
# Saves: report_quantum_computing.json
```

## Hints
- Define tools: `web_search(query)` and `fetch_url(url)`
- The agent should call search first, then fetch each result
- Pydantic model for output: `ResearchReport`
- Max 10 tool call steps before giving up

## Evaluation Criteria
- Does it actually search and retrieve real content?
- Is the JSON valid and well-structured?
- Does it handle a failed URL without crashing?
- Are all steps logged clearly?
