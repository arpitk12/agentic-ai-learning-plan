"""
SOLUTION — Project 5: Self-Improving Coding Agent with Reflexion
Reads failing tests → writes code → runs pytest → critiques → iterates.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import json
import subprocess
import re
from pathlib import Path
from dotenv import load_dotenv
from llm import chat, get_text, get_tool_calls, stop_reason, assistant_message, tool_result_message

load_dotenv()


# ── Tools ──────────────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: {path} does not exist"
    return p.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.write_text(content, encoding="utf-8")
    return f"Written {len(content)} chars to {path}"


def run_tests(test_path: str) -> str:
    """Run pytest and return structured output."""
    result = subprocess.run(
        ["python", "-m", "pytest", test_path, "-v", "--tb=short", "--no-header"],
        capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    passed = len(re.findall(r" PASSED", output))
    failed = len(re.findall(r" FAILED", output))
    errors = len(re.findall(r" ERROR", output))
    return (
        f"PASSED: {passed} | FAILED: {failed} | ERRORS: {errors}\n"
        f"Return code: {result.returncode}\n\n"
        f"{output[-3000:]}"  # last 3K of output
    )


def list_files(directory: str) -> str:
    p = Path(directory)
    files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
    return "\n".join(files)


TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run pytest on a test file and return results.",
        "input_schema": {"type": "object", "properties": {"test_path": {"type": "string"}}, "required": ["test_path"]},
    },
    {
        "name": "list_files",
        "description": "List all files in a directory.",
        "input_schema": {"type": "object", "properties": {"directory": {"type": "string"}}, "required": ["directory"]},
    },
]

TOOL_MAP = {"read_file": read_file, "write_file": write_file, "run_tests": run_tests, "list_files": list_files}

SYSTEM = """You are an expert Python programmer. Your job:
1. Read the test file to understand what's needed
2. Write a clean implementation that passes all tests
3. Run the tests to verify
4. If tests fail, analyze errors and fix the code
5. Repeat until all tests pass (max attempts: 5)

Write production-quality code with docstrings. Be concise and correct."""


# ── Reflexion Memory ───────────────────────────────────────────────────────────

def critic_agent(code: str, test_results: str) -> str:
    """Independent critic evaluates code quality."""
    r = chat(
        [{"role": "user", "content": f"Test results:\n{test_results[:500]}\n\nCode:\n{code}"}],
        system="You are a code quality critic. Evaluate this code briefly. Focus on: correctness, edge cases, efficiency, readability.",
        max_tokens=300,
    )
    return get_text(r)


# ── Main Agent Loop ────────────────────────────────────────────────────────────

def solve_problem(problem_dir: str, max_iterations: int = 5) -> dict:
    prob_path = Path(problem_dir)
    test_file = next(prob_path.glob("test_*.py"), None)
    if not test_file:
        return {"error": f"No test_*.py found in {problem_dir}"}

    impl_name = test_file.stem.replace("test_", "") + ".py"
    impl_path = str(prob_path / impl_name)

    print(f"\n🧪 Problem: {prob_path.name}")
    print(f"   Test: {test_file.name} → Impl: {impl_name}")

    failure_log: list[dict] = []
    messages = [{
        "role": "user",
        "content": (
            f"Implement the solution for: {prob_path.name}\n"
            f"Test file: {str(test_file)}\n"
            f"Write your solution to: {impl_path}\n"
            f"Run the tests after writing. Fix until all pass."
        )
    }]

    for iteration in range(1, max_iterations + 1):
        print(f"   Iteration {iteration}/{max_iterations}...")

        # Inject failure log into context for reflexion
        if failure_log:
            reflection = json.dumps(failure_log[-2:], indent=2)
            messages.append({
                "role": "user",
                "content": f"[REFLEXION — learn from these failures]:\n{reflection}\n\nTry again with a different approach."
            })

        while True:
            response = chat(messages, system=SYSTEM, max_tokens=2048, tools=TOOLS)
            messages.append(assistant_message(response))

            if stop_reason(response) == "end_turn":
                break

            if stop_reason(response) == "tool_use":
                last_test_result = ""
                for tc in get_tool_calls(response):
                    result = TOOL_MAP[tc["name"]](**tc["arguments"])
                    print(f"     🔧 {tc['name']}({str(tc['arguments'])[:50]}) → {str(result)[:60]}")
                    if tc["name"] == "run_tests":
                        last_test_result = result
                    messages.append(tool_result_message(tc["id"], str(result)))

                # Check if all tests passed
                if last_test_result and "FAILED: 0" in last_test_result and "ERRORS: 0" in last_test_result:
                    impl_code = read_file(impl_path)
                    critique = critic_agent(impl_code, last_test_result)
                    print(f"   ✅ All tests passed! Critic: {critique[:80]}")
                    return {"status": "passed", "iterations": iteration, "critique": critique}

        # After this iteration, record failure for reflexion
        test_result = run_tests(str(test_file))
        if "FAILED: 0" in test_result and "ERRORS: 0" in test_result:
            impl_code = read_file(impl_path)
            return {"status": "passed", "iterations": iteration,
                    "critique": critic_agent(impl_code, test_result)}

        failure_log.append({
            "iteration": iteration,
            "test_output": test_result[:400],
            "lesson": f"Iteration {iteration} failed. Review the error carefully.",
        })

    return {"status": "failed", "iterations": max_iterations}


def benchmark(problems_dir: str) -> dict:
    """Run agent on all problems, collect stats."""
    problems_path = Path(problems_dir)
    results = []
    for prob in sorted(problems_path.iterdir()):
        if prob.is_dir() and any(prob.glob("test_*.py")):
            r = solve_problem(str(prob))
            results.append({"problem": prob.name, **r})

    passed = sum(1 for r in results if r.get("status") == "passed")
    report = {
        "total": len(results),
        "passed": passed,
        "pass_rate": f"{passed/len(results):.0%}" if results else "0%",
        "avg_iterations": sum(r.get("iterations", 0) for r in results) / len(results) if results else 0,
        "results": results,
    }
    Path("benchmark_results.json").write_text(json.dumps(report, indent=2))
    print(f"\n📊 Benchmark: {passed}/{len(results)} passed ({report['pass_rate']})")
    return report


# ── Create sample problems ─────────────────────────────────────────────────────

def create_sample_problems():
    """Create sample problems for testing."""
    samples = {
        "binary_search": {
            "test": '''def test_binary_search():
    from binary_search import binary_search
    assert binary_search([1, 3, 5, 7, 9], 7) == 3
    assert binary_search([1, 3, 5, 7, 9], 1) == 0
    assert binary_search([1, 3, 5, 7, 9], 10) == -1
    assert binary_search([], 5) == -1
    assert binary_search([5], 5) == 0
''',
        },
        "fizzbuzz": {
            "test": '''def test_fizzbuzz():
    from fizzbuzz import fizzbuzz
    assert fizzbuzz(3) == "Fizz"
    assert fizzbuzz(5) == "Buzz"
    assert fizzbuzz(15) == "FizzBuzz"
    assert fizzbuzz(7) == "7"
    result = [fizzbuzz(i) for i in range(1, 16)]
    assert result[-1] == "FizzBuzz"
''',
        },
    }
    problems_dir = Path("./problems")
    for name, content in samples.items():
        prob_dir = problems_dir / name
        prob_dir.mkdir(parents=True, exist_ok=True)
        (prob_dir / f"test_{name}.py").write_text(content)
    print(f"Created sample problems in ./problems/")
    return str(problems_dir)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        idx = sys.argv.index("--benchmark")
        problems_dir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "./problems"
        if not Path(problems_dir).exists():
            problems_dir = create_sample_problems()
        benchmark(problems_dir)
    elif len(sys.argv) > 1:
        solve_problem(sys.argv[1])
    else:
        # Create and run sample problems
        problems_dir = create_sample_problems()
        solve_problem(f"{problems_dir}/binary_search")
