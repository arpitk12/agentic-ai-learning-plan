"""
SOLUTION — Exercise 1: Multi-turn CLI Chatbot with Sliding Window Memory
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from dotenv import load_dotenv
from llm import chat, get_text, calc_cost, MODEL

load_dotenv()

WINDOW_SIZE = 10  # keep last N user+assistant pairs


def chat_loop():
    history = []
    system = "You are a helpful, concise assistant. Refer to earlier conversation context when relevant."
    print("Chatbot ready (sliding window = last 10 turns). Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        # Sliding window: keep last WINDOW_SIZE pairs (2 messages per pair)
        trimmed = history[-(WINDOW_SIZE * 2):]

        response = chat(trimmed, system=system, max_tokens=1024)

        reply = get_text(response)
        history.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}")
        print(f"[tokens used: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out]\n")


if __name__ == "__main__":
    chat_loop()
