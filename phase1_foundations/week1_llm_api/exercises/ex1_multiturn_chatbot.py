"""
Exercise 1: Multi-turn CLI Chatbot with Sliding Window Memory
Goal: Build a chatbot that keeps the last 10 turns in context.

Uses llm.py — works with Ollama (local) or any cloud model.
Switch model by changing MODEL in .env. No code changes needed.
"""
from llm import chat, get_text, MODEL

WINDOW_SIZE = 10  # keep last N user+assistant pairs


def chatbot():
    history = []
    print(f"Chatbot ready (model: {MODEL}). Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user_input})

        # TODO: Trim history to the last WINDOW_SIZE turns
        # TODO: Call the API with the trimmed history
        # TODO: Extract the assistant reply text
        # TODO: Append the reply to history
        # TODO: Print the reply

        print("Assistant: [your implementation here]\n")


if __name__ == "__main__":
    chatbot()
