"""
Exercise 1: Basic Tool Use — 5 Tools, Let the LLM Pick
Goal: Define 5 tools with proper JSON schemas, run a single-turn tool loop.

Uses llm.py — works with Ollama (local) or any cloud model.
Note: Use qwen2.5:7b for best tool-calling support locally.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import json
import datetime 
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

# --- Tool Schemas (OpenAI/Groq format — works with LiteLLM for all providers) ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. E.g. '2 + 2 * 10'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current UTC time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Count the number of words in a text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reverse_string",
            "description": "Reverse the characters in a string.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]

# --- Tool Implementations ---

weather={
    "tokyo":"sunny with 24 C",
    "new york":"cloudy with 14 C",
    "paris": "Rainy with 18C"
}

def get_weather(city: str) -> str:
    # TODO: Return a fake weather string for the city

    return weather.get(city.lower(),f"Weather not available for {city}")

def calculator(expression: str) -> str:
    # TODO: Safely evaluate the math expression and return the result as a string
    try:
        # Basic safety: only allow numbers and operators
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    

def get_time(timezone: str) -> str:
    # TODO: Return the current UTC time as a string
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

def word_count(text: str) -> str:
    # TODO: Count and return the number of words
    return f"{len(text.split())} words"

def reverse_string(text: str) -> str:
    # TODO: Reverse the input string
    return text[::-1]


TOOL_MAP = {
    "get_weather": get_weather,
    "calculator": calculator,
    "get_time": get_time,
    "word_count": word_count,
    "reverse_string": reverse_string,
}


def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    while True:
        response = chat(messages=messages, tools=TOOLS)
        reason = stop_reason(response)

        if reason == "end_turn":
            return get_text(response)

        if reason == "tool_use":
            tool_calls = get_tool_calls(response)
            messages.append(assistant_message(response))
            for tc in tool_calls:
                fn = TOOL_MAP.get(tc["name"])
                result = fn(**tc["arguments"]) if fn else f"Unknown tool: {tc['name']}"
                print(f"  [Tool: {tc['name']}({tc['arguments']})] → {result}")
                messages.append(tool_result_message(tc["id"], result))



if __name__ == "__main__":
    result = run_agent("What's the weather in Tokyo and what is 144 * 7?")
    print(result)
