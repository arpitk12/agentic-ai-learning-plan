"""
Solution 4: Prompt Strategy Comparison
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

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
    cfg = STRATEGIES[name]
    user_prompt = cfg["user_template"].format(question=question)
    response = chat(
        messages=[{"role": "user", "content": user_prompt}],
        system=cfg["system"],
        max_tokens=512,
    )
    answer = get_text(response)
    input_tok = response.usage.prompt_tokens
    output_tok = response.usage.completion_tokens
    cost = calc_cost(MODEL, input_tok, output_tok)
    return {
        "strategy": name,
        "answer": answer.strip(),
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "cost_usd": cost,
    }


def print_comparison_table(results: list[dict]):
    header = f"{'Strategy':<20} {'Answer (preview)':<55} {'In':>5} {'Out':>5} {'Cost':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        preview = r["answer"].replace("\n", " ")[:54]
        print(
            f"{r['strategy']:<20} {preview:<55} "
            f"{r['input_tokens']:>5} {r['output_tokens']:>5} "
            f"${r['cost_usd']:>7.5f}"
        )


if __name__ == "__main__":
    print(f"Model: {MODEL}\n")

    for label, question in [("Easy", QUESTION_EASY), ("Hard", QUESTION_HARD)]:
        print(f"{'='*60}")
        print(f"Question ({label}): {question}\n")
        results = [run_strategy(name, question) for name in STRATEGIES]
        print_comparison_table(results)
        print()
