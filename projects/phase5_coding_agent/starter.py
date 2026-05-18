"""
Project 5 Starter — Self-Improving Coding Agent
Fill in the TODOs to build a reflexion-based coding agent.
"""
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

CODER_SYSTEM = """You are an expert Python developer.
You will be given a test file. Write the implementation that makes all tests pass.
Return ONLY the Python code — no markdown, no explanation."""

CRITIC_SYSTEM = """You are a code quality critic. Review the code and test results.
Return JSON only:
{
  "quality_score": <1-10>,
  "issues": ["issue1", "issue2"],
  "fix_suggestion": "Specific instruction for the next attempt"
}"""


@dataclass
class Attempt:
    iteration: int
    code: str
    test_output: str
    passed: bool
    tests_passed: int
    tests_failed: int
    critic_score: int = 0
    critic_issues: list[str] = field(default_factory=list)
    fix_suggestion: str = ""


def run_pytest(test_file: Path, impl_file: Path) -> tuple[bool, str, int, int]:
    """Run pytest and return (all_passed, output, n_passed, n_failed)."""
    result = subprocess.run(
        ["python", "-m", "pytest", str(test_file), "-v", "--tb=short", "--no-header"],
        capture_output=True, text=True, cwd=impl_file.parent
    )
    output = result.stdout + result.stderr

    # Parse counts
    passed = output.count(" PASSED")
    failed = output.count(" FAILED") + output.count(" ERROR")
    all_passed = result.returncode == 0

    return all_passed, output, passed, failed


def write_code(test_content: str, failure_log: list[Attempt]) -> str:
    """
    TODO: Ask LLM to write implementation.
    Include failure_log in context if it has entries (Reflexion).
    """
    messages = [{"role": "user", "content": f"Test file:\n```python\n{test_content}\n```"}]

    if failure_log:
        reflection = "\n".join([
            f"Attempt {a.iteration}: {a.tests_passed} passed, {a.tests_failed} failed. "
            f"Fix needed: {a.fix_suggestion}"
            for a in failure_log
        ])
        messages[0]["content"] += f"\n\nPrevious failed attempts:\n{reflection}"
        messages[0]["content"] += "\n\nLearn from these failures and write better code."

    # TODO: call client.messages.create with CODER_SYSTEM
    return "# Not implemented\n"


def critique_code(code: str, test_output: str) -> tuple[int, list[str], str]:
    """TODO: Ask critic LLM to evaluate code quality. Return (score, issues, fix_suggestion)."""
    prompt = f"Code:\n```python\n{code}\n```\n\nTest output:\n```\n{test_output[:2000]}\n```"
    # TODO: call client.messages.create with CRITIC_SYSTEM
    # TODO: parse JSON response
    return 5, ["Not implemented"], "Not implemented"


def solve_problem(problem_dir: Path, max_iterations: int = 5) -> dict:
    """Main agent loop for one problem."""
    test_files = list(problem_dir.glob("test_*.py"))
    impl_files = [f for f in problem_dir.glob("*.py") if not f.name.startswith("test_")]

    if not test_files:
        raise FileNotFoundError(f"No test files in {problem_dir}")

    test_file = test_files[0]
    impl_file = impl_files[0] if impl_files else problem_dir / test_file.name.replace("test_", "")
    test_content = test_file.read_text()

    print(f"\n{'='*50}")
    print(f"Problem: {problem_dir.name}")
    print(f"Test: {test_file.name}")

    failure_log: list[Attempt] = []

    for i in range(1, max_iterations + 1):
        print(f"\n--- Iteration {i}/{max_iterations} ---")

        # Generate code
        code = write_code(test_content, failure_log)
        impl_file.write_text(code)
        print(f"Code written ({len(code)} chars)")

        # Run tests
        passed, output, n_pass, n_fail = run_pytest(test_file, impl_file)
        print(f"Tests: {n_pass} passed, {n_fail} failed")

        # Critique
        score, issues, fix = critique_code(code, output)

        attempt = Attempt(
            iteration=i,
            code=code,
            test_output=output,
            passed=passed,
            tests_passed=n_pass,
            tests_failed=n_fail,
            critic_score=score,
            critic_issues=issues,
            fix_suggestion=fix
        )
        failure_log.append(attempt)

        if passed:
            print(f"✓ All tests passed on iteration {i}!")
            break
        else:
            print(f"✗ Failed. Fix suggestion: {fix}")

    final = failure_log[-1]
    return {
        "problem": problem_dir.name,
        "solved": final.passed,
        "iterations": len(failure_log),
        "final_score": final.critic_score
    }


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./problems/example")
    path.mkdir(parents=True, exist_ok=True)

    # Create a tiny example problem if it doesn't exist
    ex_test = path / "test_example.py"
    ex_impl = path / "example.py"
    if not ex_test.exists():
        ex_test.write_text("""from example import add, multiply

def test_add(): assert add(2, 3) == 5
def test_multiply(): assert multiply(4, 5) == 20
def test_add_negative(): assert add(-1, 1) == 0
""")
        ex_impl.write_text("# TODO — agent will fill this in\n")

    result = solve_problem(path)
    print(f"\nResult: {json.dumps(result, indent=2)}")
