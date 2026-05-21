"""
Exercise 1: Multi-turn CLI Chatbot with Sliding Window Memory
Goal: Build a chatbot that keeps the last 10 turns in context.

Uses llm.py — works with Ollama (local) or any cloud model.
Switch model by changing MODEL in .env. No code changes needed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from llm import chat, get_text, MODEL

WINDOW_SIZE = 10  # keep last N user+assistant pairs


def chatbot():
    history = []
    system="You are a helpful, concise assistant. Refer to earlier conversation context when relevant."
    print(f"Chatbot ready (model: {MODEL}). Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user_input})

        # TODO: Trim history to the last WINDOW_SIZE turns
        history=history[-(WINDOW_SIZE*2):]
        # TODO: Call the API with the trimmed history
        response=chat(history,system=system)
        # TODO: Extract the assistant reply text
        text=get_text(response=response)
        # TODO: Append the reply to history
        history.append({"role":"assistant", "content":text})
        # TODO: Print the reply

        print(f"Assistant: {text}\n")


if __name__ == "__main__":
    chatbot()
