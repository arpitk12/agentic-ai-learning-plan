# Project 5 — Self-Improving Coding Agent

## Brief
A coding agent that takes a failing test suite, writes code to fix it,
runs tests, critiques its solution, and iterates until all pass.

## Requirements
- [ ] Accept a directory with failing tests as input
- [ ] Agent reads test file, understands what's needed
- [ ] Writes/edits implementation file
- [ ] Runs `pytest` as a tool, reads results
- [ ] Critic evaluates code quality (not just test pass)
- [ ] Reflexion: failure logs inform next attempt
- [ ] Max 5 iterations per problem
- [ ] Eval harness: benchmark on 20 problems, report pass rate

## Setup
```bash
pip install anthropic python-dotenv pydantic pytest
```

## Usage
```bash
# Single problem
python starter.py ./problems/binary_search/

# Full benchmark
python starter.py --benchmark ./problems/
# Output: benchmark_results.json — pass rate, avg iterations, cost
```

## Problem Format
```
problems/
  binary_search/
    test_binary_search.py   ← provided (read-only)
    binary_search.py        ← agent writes this
  linked_list/
    test_linked_list.py
    linked_list.py
```

## Agent Loop
```
Read test file → Understand requirements
    ↓
Write implementation
    ↓
Run pytest (tool) → Parse results
    ↓
All pass? → Done ✓
    ↓ No
Critic: what's wrong with current code?
    ↓
Reflexion: log failure lesson
    ↓
Loop (max 5 times)
```

## Hints
- Use `subprocess.run(["pytest", test_file, "-v", "--tb=short"])` as a tool
- Parse pytest output: count passed/failed, extract failure messages
- Critic system prompt: "Review this code. Is it correct, readable, efficient?"
- Reflexion memory: include all previous attempts + failures in context
