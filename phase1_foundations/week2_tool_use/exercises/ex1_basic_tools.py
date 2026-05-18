"""
Exercise 1: Basic Tool Use — 5 Tools, Let the LLM Pick
Goal: Define 5 tools with proper JSON schemas, run a single-turn tool loop.

Uses llm.py — works with Ollama (local) or any cloud model.
Note: Use qwen2.5:7b for best tool-calling support locally.
"""
import json
from llm import chat, get_text, get_tool_calls, stop_reason

# --- Tool Schemas ---
TOOLS = [
    # TODO: Add 5 tool schemas here
]

# --- Tool Implementations ---

def get_weather(city: str) -> str:
    # TODO: Return a fake weather string for the city
    raise NotImplementedError

def calculator(expression: str) -> str:
    # TODO: Safely evaluate the math expression and return the result as a string
    raise NotImplementedError

def get_time(timezone: str) -> str:
    # TODO: Return the current UTC time as a string
    raise NotImplementedError

def word_count(text: str) -> str:
    # TODO: Count and return the number of words
    raise NotImplementedError

def reverse_string(text: str) -> str:
    # TODO: Reverse the input string
    raise NotImplementedError


TOOL_MAP = {
    "get_weather": get_weather,
    "calculator": calculator,
    "get_time": get_time,
    "word_count": word_count,
    "reverse_string": reverse_string,
}


def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    # TODO: Implement the tool loop
    raise NotImplementedError


if __name__ == "__main__":
    result = run_agent("What's the weather in Tokyo and what is 144 * 7?")
    print(result)
