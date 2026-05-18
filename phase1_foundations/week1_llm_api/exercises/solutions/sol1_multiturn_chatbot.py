"""
SOLUTION — Exercise 1: Multi-turn CLI Chatbot with Sliding Window Memory
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
WINDOW_SIZE = 10  # keep last N user+assistant pairs


def chat():
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

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=system,
            messages=trimmed,
        )

        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}")
        print(f"[tokens used: {response.usage.input_tokens} in / {response.usage.output_tokens} out]\n")


if __name__ == "__main__":
    chat()
