"""
SOLUTION — Exercise 1: Basic Tool Use
"""
import json
import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. E.g. '2 + 2 * 10'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current UTC time.",
        "input_schema": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "Ignored, always UTC"}},
            "required": [],
        },
    },
    {
        "name": "word_count",
        "description": "Count the number of words in a text.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "reverse_string",
        "description": "Reverse the characters in a string.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]

FAKE_WEATHER = {
    "tokyo": "Sunny, 24°C, humidity 65%",
    "paris": "Partly cloudy, 18°C, humidity 72%",
    "new york": "Rainy, 15°C, humidity 88%",
}


def get_weather(city: str) -> str:
    return FAKE_WEATHER.get(city.lower(), f"Weather data unavailable for '{city}'")


def calculator(expression: str) -> str:
    try:
        # Basic safety: only allow numbers and operators
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_time(timezone: str = "UTC") -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def word_count(text: str) -> str:
    count = len(text.split())
    return f"{count} words"


def reverse_string(text: str) -> str:
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
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant's full message to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract text from content list
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn = TOOL_MAP.get(block.name)
                    if fn:
                        result = fn(**block.input)
                        print(f"  [Tool: {block.name}({block.input})] → {result}")
                    else:
                        result = f"Unknown tool: {block.name}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    result = run_agent("What's the weather in Tokyo and Paris, and what is 144 * 7?")
    print("\n" + result)
