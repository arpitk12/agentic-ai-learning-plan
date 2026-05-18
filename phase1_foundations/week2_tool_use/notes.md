# Week 2 — Tool Use & Function Calling

## Topics
1. Function/tool calling — schema definition, parallel tool calls
2. ReAct loop: Reason → Act → Observe
3. Tool result injection, multi-step reasoning
4. Error handling in tool loops

## Key Concepts

### Tool Schema
Each tool is defined as a JSON schema:
```python
{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
}
```

### The Tool Loop
```
User message
    → LLM decides to call a tool (stop_reason = "tool_use")
    → You execute the tool
    → Inject tool_result back into messages
    → LLM continues (may call more tools or give final answer)
```

### Parallel Tool Calls
The model may call multiple tools in one response. Always loop over
`response.content` and handle ALL tool_use blocks before continuing.

## Exercises
- `ex1_basic_tools.py` — 5 tools, let LLM pick
- `ex2_react_loop.py` — 3-step ReAct from scratch
- `ex3_error_handling.py` — graceful tool failure

## Checklist
- [ ] Defined 5 tools with proper schemas
- [ ] Implemented full ReAct loop with logging
- [ ] Handled tool errors — agent retries or asks user
- [ ] Observed parallel tool calls in action
