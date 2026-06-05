# Week 2 — Tool Use & The ReAct Loop

## What This Week Is About
Tools transform LLMs from text generators into actors. An LLM with tools can search the web, run code, query databases, call APIs, and modify files. This week covers the mechanics of tool calling and the ReAct pattern — the fundamental control loop of virtually every production agent.

---

## 1. What Is a Tool Call?

A **tool call** is how an LLM says "I need to run this function before I can answer." Instead of generating text, the model outputs a structured JSON object specifying which function to call and with what arguments. Your code executes the function and feeds the result back.

```
User: "What's the weather in Tokyo?"
     ↓
LLM: [decides it needs weather data]
     → tool_call: get_weather(city="Tokyo")
     ↓
Your code: calls get_weather("Tokyo") → "22°C, sunny"
     ↓
LLM: "The weather in Tokyo is currently 22°C and sunny."
```

The LLM never actually calls the function — it just describes the call. You execute it. This separation is critical: it means you can sandbox, validate, and audit every action the agent takes.

---

## 2. Tool Schemas — Describing Functions to the LLM

The LLM decides which tool to call based on your descriptions. Quality descriptions = quality decisions.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the internet for current information. Use for facts, news, prices, and anything that may have changed recently. NOT for mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific. Include dates if relevant."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10). Default 5.",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

**Critical points for good tool schemas:**
- **Description must say WHEN to use the tool** — "Use when the user asks about X, NOT when Y"
- **Parameter descriptions must include format** — "Date in YYYY-MM-DD format", "City name in English"
- **Mark required vs optional** — required params are in `"required": [...]`
- **Include examples in description** — "e.g., 'AAPL stock price today 2025'"

### Using `normalize_tools()` from `llm.py`
```python
from llm import normalize_tools

# Define as plain dicts, normalize_tools converts to provider format
tools = normalize_tools([
    {
        "name": "calculate",
        "description": "Perform mathematical calculations.",
        "parameters": {
            "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2 * (3 + 4)'"}
        },
        "required": ["expression"]
    }
])
```

---

## 3. Registering & Dispatching Tool Calls

```python
import json, math
from llm import chat, get_tool_calls, stop_reason, assistant_message, tool_result_message, MODEL

# Define the actual Python functions
def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def search_web(query: str, num_results: int = 5) -> str:
    # In production, call SerpAPI, Tavily, or similar
    return f"Search results for '{query}': [mock results]"

# Tool registry — maps name → function
TOOLS = {
    "calculate": calculate,
    "search_web": search_web,
}

def dispatch_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        result = TOOLS[tool_name](**arguments)
        return str(result)
    except Exception as e:
        return f"Tool error: {e}"
```

---

## 4. The ReAct Loop — The Core Agent Pattern

**ReAct** = **Re**asoning + **Act**ing. The model alternates between thinking about what to do next and calling tools to gather information, until it has enough to answer.

```
Thought: I need to find the current population of Japan
Action: search_web(query="Japan population 2025")
Observation: Japan population is approximately 123.3 million (2025 estimate)
Thought: Now I can answer the question
Action: [no tool, just respond]
Final Answer: Japan's population in 2025 is approximately 123.3 million.
```

### Full ReAct Implementation

```python
def react_agent(user_query: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": user_query}]
    
    for step in range(max_steps):
        response = chat(messages=messages, tools=tools)
        reason = stop_reason(response)
        
        # Add assistant's response (may contain tool_calls) to history
        messages.append(assistant_message(response))
        
        if reason == "tool_calls":
            # Process all requested tool calls
            tool_calls = get_tool_calls(response)
            for tc in tool_calls:
                result = dispatch_tool(tc["name"], tc["arguments"])
                print(f"[Tool] {tc['name']}({tc['arguments']}) → {result[:100]}")
                messages.append(tool_result_message(tc["id"], result))
            # Continue the loop — model will process the tool results
            
        elif reason == "stop":
            # Model is done — extract and return final answer
            return response.choices[0].message.content
        
        else:
            return f"Stopped with reason: {reason}"
    
    return "Max steps reached without a final answer."
```

### `stop_reason()` Values

| Value | Meaning | What to do |
|-------|---------|------------|
| `"tool_calls"` | Model wants to call function(s) | Execute tools, append results, loop |
| `"stop"` | Model has a final answer | Extract text and return |
| `"length"` | Hit `max_tokens` limit | Increase `max_tokens` or summarize |
| `"content_filter"` | Blocked by provider | Log, return error message |

### `assistant_message()` — Why It's Needed
When the model makes a tool call, the full response (including tool_call metadata) must be in the message history so the model can process the tool result. `assistant_message(response)` extracts this correctly.

```python
# WRONG — loses tool call metadata
messages.append({"role": "assistant", "content": response.choices[0].message.content})

# RIGHT — preserves tool_calls field
messages.append(assistant_message(response))
```

---

## 5. Parallel Tool Calls

Modern LLMs can request multiple tool calls in a single response. Always process ALL of them before continuing.

```python
tool_calls = get_tool_calls(response)
print(f"Model requested {len(tool_calls)} tool calls")

# Process all tool calls (can run in parallel with asyncio if needed)
for tc in tool_calls:
    result = dispatch_tool(tc["name"], tc["arguments"])
    messages.append(tool_result_message(tc["id"], result))

# After ALL tool results are appended, make the next LLM call
```

**Example**: User asks "Compare weather in Tokyo and London" → model requests `get_weather("Tokyo")` AND `get_weather("London")` in one shot. Both results must be appended before the next call.

### Async Parallel Execution (Fast Pattern)
```python
import asyncio

async def run_tools_parallel(tool_calls: list) -> list:
    tasks = [asyncio.to_thread(dispatch_tool, tc["name"], tc["arguments"]) 
             for tc in tool_calls]
    return await asyncio.gather(*tasks)
```

---

## 6. Tool Safety & Sandboxing

**CRITICAL**: Tool calls execute real code. Safety layers are mandatory in production.

### Validation Layer
```python
def safe_dispatch(tool_name: str, arguments: dict) -> str:
    # 1. Whitelist check
    if tool_name not in ALLOWED_TOOLS:
        return f"Error: Tool '{tool_name}' is not permitted"
    
    # 2. Argument validation
    if tool_name == "run_code":
        code = arguments.get("code", "")
        dangerous = ["os.system", "subprocess", "shutil.rmtree", "__import__"]
        for d in dangerous:
            if d in code:
                return f"Error: Blocked dangerous pattern '{d}'"
    
    # 3. Rate limiting (simple)
    if tool_call_count[tool_name] > 10:
        return "Error: Tool call limit reached"
    
    return dispatch_tool(tool_name, arguments)
```

### Code Execution Safety
For agents that run code, always use:
- **Subprocess with timeout**: `subprocess.run(cmd, timeout=10)`
- **Docker sandbox**: Run in isolated container with no network
- **RestrictedPython**: Python library that restricts what code can do
- **E2B.dev**: Cloud sandbox API — perfect for production code execution agents

---

## 7. Error Handling in Tool Loops

Tools fail. Networks time out. APIs return errors. Your agent must handle this gracefully:

```python
def dispatch_tool_with_retry(tool_name: str, arguments: dict, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            result = dispatch_tool(tool_name, arguments)
            if "Error" not in result:  # simple check
                return result
        except Exception as e:
            if attempt == retries:
                return f"Tool {tool_name} failed after {retries+1} attempts: {e}"
            time.sleep(2 ** attempt)  # exponential backoff
    return "Tool failed"
```

**What to return on failure**: Return the error as a string back into the message history. Let the LLM decide how to proceed — it can retry with different arguments, use a different tool, or tell the user it can't complete the request.

---

## 8. When to Use Tools vs. Built-in Knowledge

| Use Tool For | Use LLM Knowledge For |
|-------------|----------------------|
| Current prices, news, weather | Historical facts (pre-cutoff) |
| Your company's specific data | General programming help |
| Calculations (avoid hallucination) | Explanations, summaries |
| File operations | Creative writing |
| Database queries | Reasoning, analysis |
| Sending emails/messages | Language translation |

**Rule**: If the answer could have changed in the last 6 months, or if it's specific to your system — use a tool.

---

## Tools & Libraries Used This Week — Deep Dive

### LiteLLM Tool Support — How It Works

LiteLLM doesn't just translate text — it also translates the **tool calling format** between providers. OpenAI uses `"type": "function"` schema. Anthropic uses different field names. Google Gemini uses its own format. LiteLLM normalizes all of this.

When you call `chat(messages, tools=tools)`:
1. LiteLLM converts your tool schema to the provider's format
2. Makes the API call
3. If the model requests a tool call, converts the response back to OpenAI format
4. `get_tool_calls(response)` extracts: `[{"id": "...", "name": "...", "arguments": {...}}]`

This means your ReAct loop code is **identical** regardless of whether you're using GPT-4o, Gemini, or Claude.

---

### Tavily — The Right Web Search for Agents

**Why not just use Google Search API or Serper?** Those return raw HTML snippets and links — not LLM-friendly text. Tavily is specifically designed for AI agents:
- Returns clean text extracts (no HTML)
- Includes content summaries in the result
- Handles paywalled content better
- Has a dedicated "news" mode for current events
- Designed to be directly injected into LLM context

```python
from tavily import TavilyClient
import os

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# As an agent tool — this is what you put in TOOLS registry
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for current information.
    Use for: news, prices, current events, facts you're uncertain about.
    NOT for: math calculations, historical facts before 2020 that you know.
    """
    try:
        results = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",       # "basic" for speed, "advanced" for quality
        )
        
        if not results["results"]:
            return "No results found for that query."
        
        formatted = []
        for r in results["results"]:
            formatted.append(f"Source: {r['url']}\nContent: {r['content']}")
        
        return "\n\n---\n\n".join(formatted)
    
    except Exception as e:
        return f"Search failed: {e}"

# The model string for the tool definition:
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": """Search the internet for current information. 
Use this when: the question involves recent events, current prices, live data, or facts that may have changed.
Do NOT use this for: math, well-known historical facts, or anything you already know confidently.
Query tips: be specific, include year if asking about current data, use proper names.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific. E.g., 'Python 3.12 release date' not 'Python version'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (1-5). Default 3.",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}
```

---

### E2B — Cloud Code Execution for Agents

**What it is**: E2B (E2B.dev) is a cloud sandbox API for securely executing code inside isolated microVMs. Each agent run gets a fresh, isolated environment.

**Why you need it**: For agents that write and execute code, running code in your main process is extremely dangerous. A malicious prompt could delete files, make network requests, or consume resources. E2B runs code in a completely isolated VM that can't affect your server.

```python
# Install: pip install e2b-code-interpreter
from e2b_code_interpreter import Sandbox

def safe_execute_code(code: str, timeout: int = 30) -> str:
    """Execute Python code safely in an E2B cloud sandbox."""
    with Sandbox() as sandbox:
        execution = sandbox.run_code(code, timeout=timeout)
        
        if execution.error:
            return f"Error:\n{execution.error.name}: {execution.error.value}"
        
        output = ""
        for result in execution.results:
            if result.text:
                output += result.text + "\n"
        
        if execution.logs.stdout:
            output += "\n".join(execution.logs.stdout)
        
        return output.strip() or "(Code executed with no output)"
```

**Cost**: E2B is cheap — ~$0.000002 per execution second. A 10-second code run costs $0.00002.

---

### normalize_tools() — Format Normalization

The `normalize_tools()` function in `llm.py` converts a simplified tool definition format into the full OpenAI schema format. This lets you write tools more concisely:

```python
from llm import normalize_tools

# Simplified format (what you write)
tools = normalize_tools([
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Use when user asks about weather.",
        "parameters": {
            "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
            "units": {"type": "string", "description": "celsius or fahrenheit", "default": "celsius"}
        },
        "required": ["city"]
    }
])

# What normalize_tools() produces (OpenAI format — what LiteLLM needs):
# {
#   "type": "function",
#   "function": {
#     "name": "get_weather",
#     "description": "Get current weather for a city...",
#     "parameters": {
#       "type": "object",
#       "properties": {
#         "city": {"type": "string", "description": "City name..."},
#         "units": {"type": "string", "description": "celsius or fahrenheit", "default": "celsius"}
#       },
#       "required": ["city"]
#     }
#   }
# }
```

---

## The ReAct Loop — Under the Hood

What actually happens inside the ReAct loop at the message level:

```
TURN 1: Initial call
Messages: [{"role": "user", "content": "What's the weather in Tokyo and London?"}]
→ Model response: requests TWO tool calls simultaneously:
  - get_weather(city="Tokyo")
  - get_weather(city="London")
  stop_reason = "tool_calls"

TURN 2: Process tool calls, add results
Messages: [
  {"role": "user", "content": "What's the weather in Tokyo and London?"},
  {"role": "assistant", "tool_calls": [{"id": "tc_1", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}, {"id": "tc_2", ...}]},  ← assistant_message()
  {"role": "tool", "tool_call_id": "tc_1", "content": "Tokyo: 22°C, sunny"},         ← tool_result_message("tc_1", ...)
  {"role": "tool", "tool_call_id": "tc_2", "content": "London: 15°C, cloudy"},       ← tool_result_message("tc_2", ...)
]
→ Model response: "The weather in Tokyo is 22°C and sunny. London is 15°C and cloudy."
  stop_reason = "stop"
```

**The critical insight**: The model sees its OWN tool call request in the history (via `assistant_message()`). This is how it knows which tool results belong to which call. If you skip `assistant_message()`, the model sees orphaned tool results and produces garbage.

---

## Tool Schema Design Patterns

### Pattern 1: The "When NOT To Use" Description
```python
# WEAK description — model doesn't know when to use vs. when not to
"description": "Calculate mathematical expressions"

# STRONG description — tells model exactly when to use AND when not to
"description": """Calculate mathematical expressions using Python operators.
Use when: arithmetic (+,-,*,/), exponents (**), modulo (%), square roots, basic algebra.
Do NOT use for: looking up prices, getting statistics, anything requiring live data.
Examples: '2**10', '(3 + 4) * 7', 'sqrt(144)'"""
```

### Pattern 2: Format Guidance in Parameter Descriptions
```python
# WEAK — model guesses the format
"date": {"type": "string", "description": "The date"}

# STRONG — format is explicit
"date": {
    "type": "string",
    "description": "Date in YYYY-MM-DD format. E.g. '2025-01-15' for January 15, 2025."
}
```

### Pattern 3: Enum for Fixed Options
```python
# Let the model choose from a validated set
"format": {
    "type": "string",
    "enum": ["json", "markdown", "plain_text"],
    "description": "Output format. Use json for structured data, markdown for reports, plain_text for simple answers."
}
```

---

## Error Handling Architecture

In a production agent, tool failures are normal. Design for them:

```python
class ToolResult:
    """Structured tool result with error handling."""
    def __init__(self, success: bool, data: str, error: str = None):
        self.success = success
        self.data = data
        self.error = error
    
    def to_message(self) -> str:
        if self.success:
            return self.data
        return f"Tool failed: {self.error}. You may need to try a different approach."

def robust_dispatch(tool_name: str, arguments: dict) -> str:
    """
    Dispatch with comprehensive error handling.
    Returns structured error message so agent can recover.
    """
    if tool_name not in TOOLS:
        return f"Error: Tool '{tool_name}' doesn't exist. Available tools: {list(TOOLS.keys())}"
    
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Tool '{tool_name}' timed out after 30 seconds")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        result = TOOLS[tool_name](**arguments)
        signal.alarm(0)  # cancel timeout
        
        return str(result)[:5000]  # cap result length
    
    except TimeoutError as e:
        return f"Error: {e}. Try with a more specific query or different tool."
    except TypeError as e:
        return f"Error: Wrong arguments for '{tool_name}': {e}. Check argument names and types."
    except Exception as e:
        return f"Error in '{tool_name}': {type(e).__name__}: {str(e)[:200]}. Try a different approach."
```

---

## Common Pitfalls — Week 2

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forgetting `assistant_message()` | "Tool not found" error or model ignores tool results | Always append `assistant_message(response)` before tool results |
| Processing only first tool call | Agent loops, second tool result missing | Iterate `for tc in get_tool_calls(response):` |
| Returning exception from tool | Agent crashes | Always return error as string: `return f"Error: {e}"` |
| No max_steps limit | Agent loops forever | Add `for step in range(max_steps):` |
| Tool name mismatch | "Unknown tool" error | Tool registry key must exactly match schema `name` |
| JSON parse on `arguments` | AttributeError | `llm.py`'s `get_tool_calls()` returns already-parsed dict |
| All tools as high-risk | Blocks safe tool calls | Classify tools by risk level, auto-approve low-risk |
| Tool description too vague | Model calls wrong tool | Add "Use when X, NOT when Y" to every description |
- `ex2_react_loop.py` — full ReAct loop with web search and calculator
- `ex3_parallel_tools.py` — agent that calls multiple tools simultaneously
- `ex4_safe_code_runner.py` — code execution agent with sandboxing

## Checklist
- [ ] Defined 3 tools with quality descriptions (name, description, parameters, required)
- [ ] Implemented dispatch_tool() registry pattern
- [ ] Built a full ReAct loop: call → tool_calls → execute → append → loop → stop
- [ ] Handled parallel tool calls in a single response
- [ ] Added error handling: tool failures returned as strings, not exceptions
- [ ] Added safety validation (whitelist, argument checks)
