"""LangGraph node functions — each receives State and returns partial update."""
from __future__ import annotations

import ast
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from src.config import cfg
from src.graph.state import ReviewState

logger = logging.getLogger(__name__)


def _get_llm():
    from langchain_litellm import ChatLiteLLM
    return ChatLiteLLM(model=cfg.model, temperature=0.1)


# ── Node 1: parse_code ────────────────────────────────────────────────────

def parse_code(state: ReviewState) -> dict:
    code = state["code"]
    info: dict[str, Any] = {"lines": len(code.splitlines())}
    try:
        tree = ast.parse(code)
        info["functions"] = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        info["classes"] = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        info["imports"] = [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        status = "parsed"
    except SyntaxError as e:
        info["syntax_error"] = str(e)
        status = "syntax_error"

    return {
        "parsed_info": info,
        "status": status,
        "messages": [{"role": "system", "content": f"Parsed code: {info}"}],
    }


# ── Node 2: analyze_security ──────────────────────────────────────────────

_SECURITY_PATTERNS = [
    ("eval(", "Dangerous eval() call — arbitrary code execution risk"),
    ("exec(", "Dangerous exec() call — arbitrary code execution risk"),
    ("os.system(", "Shell injection risk via os.system()"),
    ("subprocess.call(shell=True", "Shell injection risk in subprocess"),
    ("sql = f\"", "Potential SQL injection via f-string"),
    ("password", "Hardcoded credential? Review carefully"),
    ("secret", "Hardcoded secret? Review carefully"),
    ("pickle.loads(", "Unsafe deserialization via pickle"),
]

def analyze_security(state: ReviewState) -> dict:
    code = state["code"]
    issues = [msg for pattern, msg in _SECURITY_PATTERNS if pattern in code]
    return {
        "security_issues": issues,
        "messages": [{"role": "system", "content": f"Security: {len(issues)} issues found"}],
    }


# ── Node 3: analyze_quality ───────────────────────────────────────────────

def analyze_quality(state: ReviewState) -> dict:
    code = state["code"]
    lines = code.splitlines()
    score = 10.0
    deductions = []

    if len(lines) > 200:
        score -= 1; deductions.append("File >200 lines")
    if not any('"""' in l or "'''" in l for l in lines):
        score -= 2; deductions.append("No docstrings found")
    if not any("def test_" in l for l in lines):
        score -= 1; deductions.append("No test functions")
    if not any(": " in l and "->" in l for l in lines):
        score -= 0.5; deductions.append("Missing type hints")

    return {
        "quality_score": max(0.0, score),
        "messages": [{"role": "system", "content": f"Quality score: {score}/10. Issues: {deductions}"}],
    }


# ── Node 4: generate_review ───────────────────────────────────────────────

_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a senior software engineer doing a code review. Be specific and constructive."),
    ("human", """Review this {language} code:
```{language}
{code}
```

Analysis:
- Security issues: {security_issues}
- Quality score: {quality_score}/10
- Functions: {functions}
{feedback_note}

Write a code review with:
1. **Summary** (2-3 sentences)
2. **Issues** (list with severity: critical/major/minor)
3. **Suggestions** (specific improvements with code examples)
4. **Verdict** (approve / approve with changes / request changes)"""),
])

def generate_review(state: ReviewState) -> dict:
    llm = _get_llm()
    chain = _REVIEW_PROMPT | llm
    feedback_note = f"Human feedback: {state.get('human_feedback')}" if state.get("human_feedback") else ""
    response = chain.invoke({
        "language": state.get("language", "python"),
        "code": state["code"],
        "security_issues": state.get("security_issues", []),
        "quality_score": state.get("quality_score", 0),
        "functions": state.get("parsed_info", {}).get("functions", []),
        "feedback_note": feedback_note,
    })
    return {
        "review": response.content,
        "status": "review_ready",
        "messages": [{"role": "assistant", "content": response.content}],
    }


# ── Node 5: human_approval (with interrupt) ───────────────────────────────

def human_approval(state: ReviewState) -> dict:
    """Pauses graph execution for human review. Resumes via Command(resume=...)."""
    feedback = interrupt({
        "review": state["review"],
        "prompt": "Type 'approve' or 'revise: <your notes>'",
    })
    if str(feedback).lower().startswith("approve"):
        return {"status": "approved", "human_feedback": str(feedback)}
    else:
        notes = str(feedback).replace("revise:", "").strip()
        return {"status": "needs_revision", "human_feedback": notes, "revision_count": state.get("revision_count", 0) + 1}


# ── Node 6: finalize_review ───────────────────────────────────────────────

def finalize_review(state: ReviewState) -> dict:
    return {
        "status": "finalized",
        "messages": [{"role": "system", "content": "Review approved and finalized."}],
    }


# ── Router ────────────────────────────────────────────────────────────────

def route_approval(state: ReviewState) -> str:
    """Conditional edge: returns 'approved' or 'needs_revision'."""
    return state["status"]
