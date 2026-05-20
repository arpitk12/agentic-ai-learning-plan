"""
Exercise 4: Prompt Strategy Comparison
Goal: Run the same reasoning task with 3 different prompting strategies
      and compare quality, token usage, and cost.

Strategies to implement:
  1. Zero-shot       — ask the question directly, no guidance
  2. Chain-of-Thought — tell the model to "think step by step"
  3. Few-shot        — provide 2 worked examples before the real question

Task (use the same question for all 3):
  "A bat and a ball cost $1.10 in total.
   The bat costs $1.00 more than the ball.
   How much does the ball cost?"

Tasks:
  1. Implement run_strategy(strategy_name, system_prompt, user_prompt) → dict
     Returns: {"strategy": ..., "answer": ..., "input_tokens": ..., "output_tokens": ...}
  2. Run all 3 strategies and collect results.
  3. Print a comparison table showing answer, tokens used, and cost.
  4. Add a second harder question and see if CoT helps more on that one.

Harder question for step 4:
  "If it takes 5 machines 5 minutes to make 5 widgets,
   how long would it take 100 machines to make 100 widgets?"
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from llm import chat, get_text, calc_cost, MODEL

QUESTION_EASY = (
    "A bat and a ball cost $1.10 in total. "
    "The bat costs $1.00 more than the ball. "
    "How much does the ball cost?"
)

QUESTION_HARD = (
    "If it takes 5 machines 5 minutes to make 5 widgets, "
    "how long would it take 100 machines to make 100 widgets?"
)

# --- Strategy definitions ---
STRATEGIES = {
    "zero_shot": {
        "system": "You are a helpful assistant. Answer the question.",
        "user_template": "{question}",
    },
    "chain_of_thought": {
        "system": "You are a helpful assistant. Think through the problem step by step before giving your final answer.",
        "user_template": "{question}\n\nLet's think step by step.",
    },
    "few_shot": {
        "system": "You are a helpful assistant.",
        "user_template": """Here are two example problems solved correctly:

Problem: If you have 3 apples and give away 1, how many do you have?
Answer: 3 - 1 = 2 apples.

Problem: A dozen eggs costs $3. How much do 2 dozen cost?
Answer: 2 × $3 = $6.

Now solve this:
{question}""",
    },
}


def run_strategy(name: str, question: str) -> dict:
    """
    TODO: Run the given strategy against the given question.
    Return a dict with keys:
      strategy, question_short, answer, input_tokens, output_tokens, cost_usd
    """
    pass


def print_comparison_table(results: list[dict]):
    """
    TODO: Print a formatted table of results.
    Columns: Strategy | Answer (first 60 chars) | Input tok | Output tok | Cost
    """
    pass


if __name__ == "__main__":
    print(f"Model: {MODEL}\n")

    for label, question in [("Easy", QUESTION_EASY), ("Hard", QUESTION_HARD)]:
        print(f"{'='*60}")
        print(f"Question ({label}): {question}\n")

        results = []
        for strategy_name in STRATEGIES:
            result = run_strategy(strategy_name, question)
            results.append(result)

        print_comparison_table(results)
        print()
