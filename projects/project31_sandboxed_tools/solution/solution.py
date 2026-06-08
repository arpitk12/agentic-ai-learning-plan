"""
Project 31 SOLUTION — Sandboxed Tool Execution
E2B cloud sandbox + Docker local sandbox + reversibility classifier + audit log.
Agent cannot escape the sandbox; all code runs in ephemeral isolated containers.
"""
from __future__ import annotations
import os, re, json, asyncio, hashlib, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import litellm
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SandboxResult:
    stdout: str; stderr: str; exit_code: int; timed_out: bool; latency_ms: float


# ── E2B Sandbox (cloud) ───────────────────────────────────────────────────────

async def e2b_execute(code: str, timeout: int = 30) -> SandboxResult:
    """Run code in E2B cloud sandbox. Requires E2B_API_KEY env var."""
    t0 = time.perf_counter()
    try:
        from e2b_code_interpreter import Sandbox  # type: ignore
        sandbox = Sandbox()
        execution = sandbox.run_code(code, timeout=timeout)
        sandbox.kill()
        latency = (time.perf_counter() - t0) * 1000
        return SandboxResult(
            stdout="\n".join(str(r) for r in (execution.results or [])),
            stderr="\n".join(execution.error.traceback if execution.error else []),
            exit_code=0 if not execution.error else 1,
            timed_out=False, latency_ms=latency,
        )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return SandboxResult(stdout="", stderr=str(e), exit_code=1, timed_out=False, latency_ms=latency)


# ── Docker Sandbox (local) ────────────────────────────────────────────────────

async def docker_execute(code: str, timeout: int = 30) -> SandboxResult:
    """Run code in ephemeral Docker container with resource limits."""
    t0 = time.perf_counter()
    try:
        import docker  # type: ignore
        client = docker.from_env()
        cmd = ["python", "-c", code]
        try:
            container = client.containers.run(
                "python:3.11-slim",
                command=cmd,
                remove=True,
                mem_limit="256m",
                cpu_period=100000,
                cpu_quota=50000,    # 0.5 CPU
                network_disabled=True,   # no network access
                read_only=True,
                timeout=timeout,
                stdout=True, stderr=True,
            )
            out = container.decode() if isinstance(container, bytes) else str(container)
            latency = (time.perf_counter() - t0) * 1000
            return SandboxResult(stdout=out, stderr="", exit_code=0, timed_out=False, latency_ms=latency)
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            stderr = str(e)
            timed_out = "timeout" in stderr.lower()
            return SandboxResult(stdout="", stderr=stderr, exit_code=1, timed_out=timed_out, latency_ms=latency)
    except ImportError:
        return SandboxResult(stdout="", stderr="docker package not installed", exit_code=1, timed_out=False, latency_ms=0.0)


# ── Safe Execute with Timeout ─────────────────────────────────────────────────

async def safe_execute(code: str, backend: Literal["e2b", "docker"] = "e2b", timeout: int = 30) -> SandboxResult:
    executor = e2b_execute if backend == "e2b" else docker_execute
    try:
        return await asyncio.wait_for(executor(code, timeout), timeout=timeout + 2)
    except asyncio.TimeoutError:
        return SandboxResult(stdout="", stderr="Execution timed out", exit_code=1, timed_out=True, latency_ms=timeout * 1000)


# ── Reversibility Classifier ──────────────────────────────────────────────────

_IRREVERSIBLE = [
    (re.compile(r, re.IGNORECASE), label) for r, label in [
        (r'\bopen\s*\(.+,\s*["\']w', "file write"),
        (r'\bos\.remove\b', "file delete"),
        (r'\bshutil\.rmtree\b', "directory delete"),
        (r'\brequests\.post\b', "HTTP POST"),
        (r'\bhttpx\.post\b', "HTTP POST"),
        (r'subprocess.*rm\b', "shell rm"),
        (r'\bsmtplib\b', "email send"),
        (r'\bsqlite3.*execute.*(?:INSERT|UPDATE|DELETE)', "DB write"),
    ]
]

def classify_reversibility(code: str) -> dict:
    found = [label for pattern, label in _IRREVERSIBLE if pattern.search(code)]
    return {
        "reversible": len(found) == 0,
        "irreversible_ops": found,
        "requires_approval": len(found) > 0,
    }


# ── Audit Log ─────────────────────────────────────────────────────────────────

AUDIT_LOG = Path("./sandbox_audit.jsonl")

def audit_log(caller: str, code: str, result: SandboxResult, approved: bool = True):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "caller": caller,
        "code_hash": hashlib.sha256(code.encode()).hexdigest()[:12],
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "latency_ms": round(result.latency_ms, 1),
        "approved": approved,
        "stdout_len": len(result.stdout),
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── LangGraph Data Analyst Agent ──────────────────────────────────────────────

async def data_analyst_agent(question: str, csv_path: str) -> str:
    """Agent that writes Python data analysis code and executes it in a sandbox."""
    SYSTEM = f"""You are a data analyst. The user has a CSV file at: {csv_path}
Write Python code using pandas to answer their question.
Use print() to show results.
Return ONLY the Python code, no explanation, no markdown."""

    # 1. Generate code
    resp = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        temperature=0.0,
    )
    code = resp.choices[0].message.content.strip()
    code = re.sub(r"```(?:python)?", "", code).strip()

    # 2. Classify reversibility
    rev = classify_reversibility(code)
    if not rev["reversible"]:
        return f"Code contains irreversible operations ({rev['irreversible_ops']}). Approval required."

    # 3. Execute in sandbox
    result = await safe_execute(code, backend="e2b", timeout=30)
    audit_log("data_analyst_agent", code, result)

    if result.timed_out:
        return "Execution timed out (30s limit)."
    if result.exit_code != 0:
        # Let agent self-correct
        fix_resp = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Error: {result.stderr[:300]}\nFix the code."},
            ],
        )
        fixed_code = fix_resp.choices[0].message.content.strip()
        fixed_code = re.sub(r"```(?:python)?", "", fixed_code).strip()
        result = await safe_execute(fixed_code, backend="e2b", timeout=30)
        audit_log("data_analyst_agent_retry", fixed_code, result)

    return result.stdout or f"No output. stderr: {result.stderr[:200]}"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 31: Sandboxed Tools SOLUTION ===\n")

    # 1. Test reversibility classifier
    print("1. Reversibility Classifier:")
    safe_code = "import pandas as pd; df = pd.DataFrame({'a':[1,2,3]}); print(df.mean())"
    unsafe_code = "import os; os.remove('important_file.txt')"

    r1 = classify_reversibility(safe_code)
    r2 = classify_reversibility(unsafe_code)
    print(f"  Safe code: reversible={r1['reversible']} ops={r1['irreversible_ops']}")
    print(f"  Unsafe code: reversible={r2['reversible']} ops={r2['irreversible_ops']}\n")

    # 2. Test sandbox execution (uses E2B if API key present, otherwise demo)
    print("2. Sandbox Execution:")
    test_code = "print(sum(range(1, 101)))"  # should print 5050
    if os.getenv("E2B_API_KEY"):
        result = await safe_execute(test_code, backend="e2b")
        audit_log("test", test_code, result)
        print(f"  E2B result: stdout='{result.stdout.strip()}' exit={result.exit_code} latency={result.latency_ms:.0f}ms")
    else:
        print("  E2B_API_KEY not set. Running Docker sandbox...")
        result = await safe_execute(test_code, backend="docker")
        print(f"  Docker result: stdout='{result.stdout.strip()}' exit={result.exit_code}")

    # 3. Data analyst agent demo
    print("\n3. Data Analyst Agent:")
    print("  Creates a sample CSV and asks the agent to analyze it")

    # Create sample CSV
    import csv
    sample_csv = "/tmp/compliance_data.csv"
    with open(sample_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "risk_level", "value"])
        w.writerows([
            ["DOC-001", "high", 750000],
            ["DOC-002", "medium", 50000],
            ["DOC-003", "critical", 1200000],
            ["DOC-004", "low", 10000],
            ["DOC-005", "high", 500000],
        ])

    answer = await data_analyst_agent(
        "What is the total contract value for high and critical risk documents?",
        sample_csv,
    )
    print(f"  Q: Total value for high/critical risk documents?")
    print(f"  A: {answer}")

    # 4. Audit log
    if AUDIT_LOG.exists():
        print(f"\n4. Audit log: {AUDIT_LOG}")
        with open(AUDIT_LOG) as f:
            for line in f:
                entry = json.loads(line)
                print(f"  [{entry['timestamp'][:19]}] {entry['caller']} hash={entry['code_hash']} exit={entry['exit_code']}")

if __name__ == "__main__":
    asyncio.run(main())
