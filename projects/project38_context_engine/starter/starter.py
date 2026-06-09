"""
Project 38 — Context Budget Engine (starter)
=============================================
Build a production context budget engine that allocates a fixed token window
across 5 sources, enforces budgets, manages all 4 memory levels, and reports
before/after savings.

Companion: guide/13_system_design.md §5 — Context Architecture

Fill in every # TODO block. Do NOT look at solution/ until you've tried.
"""
from __future__ import annotations
import asyncio, copy, json, re, string
from dataclasses import dataclass, field
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()
MODEL       = "openai/gpt-4o-mini"
CHEAP_MODEL = "openai/gpt-4o-mini"   # used for history summarisation

# ══════════════════════════════════════════════════════════════════════════════
# 1 — Token Counting
# ══════════════════════════════════════════════════════════════════════════════

def count_tokens(text: str, model: str = MODEL) -> int:
    """Count tokens using tiktoken. Falls back to cl100k_base for unknown models."""
    # TODO 1: import tiktoken; strip provider prefix; encoding_for_model; KeyError → cl100k_base
    raise NotImplementedError


def count_messages_tokens(messages: list[dict], model: str = MODEL) -> int:
    """Count total tokens for a messages array including per-message overhead (4 tokens each + 2 primer)."""
    # TODO 2: total = 2 + sum(4 + count_tokens(m["content"] or "")) for m in messages
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Context Budget Definition
# ══════════════════════════════════════════════════════════════════════════════

class BudgetError(Exception): pass

@dataclass
class ContextBudget:
    total:           int = 8_000
    system_prompt:   int = 400
    long_term_memory:int = 300
    rag_context:     int = 2_000
    history:         int = 2_000
    tool_schemas:    int = 400
    current_message: int = 200
    output_reserve:  int = 2_900  # never used for input — reserved for LLM output

    def compute_available(self) -> int:
        """Sum of all input sources (everything except output_reserve)."""
        # TODO 3: return sum of all budget fields except output_reserve
        raise NotImplementedError

    def validate(self):
        """Raise BudgetError if sources + output_reserve > total."""
        # TODO 4: check compute_available() + output_reserve <= total
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Memory Store (in-memory for this project)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Memory:
    key:         str
    value:       str
    memory_type: str   # "episodic" | "semantic" | "procedural" | "user_profile"
    score:       float = 1.0


class MemoryStore:
    """
    The four memory levels:
      episodic:     "User asked about GDPR DPA last session"
      semantic:     "GDPR Article 28 requires written DPAs"
      procedural:   "Always cite article numbers when answering"
      user_profile: "User is a DPO at a fintech company"
    """
    def __init__(self):
        self._store: list[Memory] = []

    def store(self, key: str, value: str, memory_type: str = "semantic"):
        # TODO 5: append Memory(key, value, memory_type) to self._store
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 5) -> list[Memory]:
        """Return top_k memories most relevant to query (keyword overlap scoring)."""
        # TODO 6:
        # query_words = set of lowercased words in query
        # For each memory, score = len(query_words ∩ words_in_memory.value) / len(query_words)
        # Return top_k by score (descending), only include if score > 0
        raise NotImplementedError

    def get_all(self, memory_type: str) -> list[Memory]:
        # TODO 7: return [m for m in _store if m.memory_type == memory_type]
        raise NotImplementedError

    def format_memories(self, memories: list[Memory]) -> str:
        """Format memories as a compact string for injection into context."""
        # TODO 8: Return "[memory_type] key: value" per line, joined with newlines
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Text Compression (TF-IDF extractive)
# ══════════════════════════════════════════════════════════════════════════════

def compress_text(text: str, target_tokens: int, model: str = MODEL) -> str:
    """Extractive compression: score sentences by word frequency, greedy select."""
    if count_tokens(text, model) <= target_tokens:
        return text

    # TODO 9: Split into sentences; score by word frequency; greedy select top sentences;
    #         re-sort by original position; join and return
    # Hint: use re.split(r"(?<=[.!?])\s+", text) for sentence splitting
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 5 — RAG Greedy Fill
# ══════════════════════════════════════════════════════════════════════════════

def fill_rag(chunks: list[dict], budget: int, model: str = MODEL) -> tuple[list[dict], int]:
    """
    Sort by score descending. Greedily fill until budget exhausted.
    If a chunk is too large but score > 0.7, compress to fit.
    Returns (selected_chunks, tokens_used).
    """
    # TODO 10: sort → greedy fill → compress high-score oversized chunks → return
    raise NotImplementedError


def format_rag(chunks: list[dict]) -> str:
    """Format as compact inline citations: [N] source: text"""
    # TODO 11: "Context:\n" + "\n".join(f"[{i+1}] {c['source']}: {c['text']}" ...)
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 6 — History Management
# ══════════════════════════════════════════════════════════════════════════════

async def summarise_old_turns(old_messages: list[dict]) -> str:
    """LLM-summarise older turns into 3 bullets."""
    # TODO 12: build transcript; call litellm.acompletion with cheap model; return summary
    raise NotImplementedError


async def manage_history(
    messages: list[dict],
    budget: int,
    keep_last_n: int = 4,
    model: str = MODEL,
) -> list[dict]:
    """
    Preserve system messages.
    Keep last keep_last_n turn-pairs.
    If still over budget: summarise older turns → inject as system message.
    Recurse until under budget.
    """
    # TODO 13:
    # 1. If count_messages_tokens(messages) <= budget → return messages
    # 2. Separate system from non-system messages
    # 3. Keep last keep_last_n*2 non-system messages as "recent"
    # 4. Summarise the rest → inject as {"role":"system","content":"[Summary]\n..."}
    # 5. Rebuild: system_msgs + [summary_msg] + recent
    # 6. Recurse if still over budget
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Tool Schema Compaction
# ══════════════════════════════════════════════════════════════════════════════

def compact_tools(tools: list[dict], budget: int, model: str = MODEL) -> tuple[list[dict], int]:
    """
    Strip tool schemas to first sentence (≤80 chars).
    Remove: examples, default, title from properties.
    Abbreviate enums > 5 values.
    Returns (compacted_tools, token_count).
    """
    # TODO 14:
    # Deep copy → for each tool: truncate description → for each prop: remove noise keys
    # → abbreviate long enums → return (compacted, count_tokens(json.dumps(compacted)))
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Context Report
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceReport:
    name:   str
    budget: int
    actual: int

    @property
    def overrun(self) -> int:
        return max(0, self.actual - self.budget)

    @property
    def status(self) -> str:
        return "⚠️  OVERRUN" if self.overrun else "✅ OK"


@dataclass
class ContextReport:
    sources:       list[SourceReport] = field(default_factory=list)
    total_budget:  int = 0
    total_actual:  int = 0

    def warnings(self) -> list[str]:
        # TODO 15: return list of "{name} exceeded budget by {overrun} tokens" for overruns
        raise NotImplementedError

    def render(self):
        # TODO 16: print a formatted table: Source | Budget | Actual | Status
        # After table: print total budget vs actual and headroom %
        raise NotImplementedError

    def diff(self, unmanaged_tokens: int, managed_tokens: int, model: str = MODEL):
        """Print before/after savings in tokens and USD, including monthly projection at 10k calls/day."""
        # TODO 17: calculate savings_pct, cost saved per call, per 10k, per month
        # Use: cost = (tokens / 1000) * 0.0002  (input at gpt-4o-mini rate)
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 9 — Context Engine (main assembler)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AssembledContext:
    messages:      list[dict]
    report:        ContextReport
    rag_chunks:    list[dict]
    compact_tools: list[dict]


class ContextEngine:
    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget()
        self.budget.validate()

    async def assemble(
        self,
        question:      str,
        system_prompt: str,
        history:       list[dict],
        rag_chunks:    list[dict],
        tools:         list[dict],
        memory_store:  MemoryStore | None = None,
    ) -> AssembledContext:
        """
        Assemble context in optimal order (see guide/13_system_design.md §5).
        Order: system prompt → long-term memory → RAG → compressed history → current message.
        Tool schemas are noted in system prompt (not injected as a separate source for simplicity).
        """
        report = ContextReport(total_budget=self.budget.total)
        messages: list[dict] = []

        # TODO 18: Step A — System prompt (measure, warn if over budget)
        # Append {"role":"system","content":system_prompt} to messages
        # Add SourceReport(name="System prompt", budget=..., actual=...)

        # TODO 19: Step B — Long-term memory (retrieve from store, format, budget)
        # Retrieve top memories for question. Format. Compress if needed.
        # Append as {"role":"system","content":"[Memory]\n{formatted}"} if non-empty.
        # Add SourceReport(name="Long-term memory", ...)

        # TODO 20: Step C — RAG context (greedy fill)
        # Call fill_rag(rag_chunks, self.budget.rag_context). Format. Append to messages.
        # Add SourceReport(name="RAG context", ...)

        # TODO 21: Step D — History (sliding window + LLM summary)
        # Call manage_history(history, self.budget.history). Append managed messages.
        # Add SourceReport(name="History", ...)

        # TODO 22: Step E — Tool schemas (compact, budget)
        # Call compact_tools(tools, self.budget.tool_schemas). Add SourceReport.
        # (Return compacted tools in AssembledContext — the caller passes them to litellm)

        # TODO 23: Step F — Current user message (measure, warn if over budget)
        # Append {"role":"user","content":question}
        # Add SourceReport(name="Current message", ...)

        # TODO 24: Set report.total_actual = count_messages_tokens(messages)
        # Return AssembledContext(messages, report, selected_rag_chunks, compacted_tools)
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Sample Data
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_SYSTEM = (
    "You are an enterprise compliance AI assistant. "
    "Answer questions about GDPR, CCPA, HIPAA and other regulations. "
    "Always cite specific article numbers. Be concise."
)

SAMPLE_HISTORY = [
    {"role": "user",      "content": "What is GDPR?"},
    {"role": "assistant", "content": "GDPR stands for General Data Protection Regulation, enacted by the EU in 2018. It governs how organizations collect, store, and process personal data of EU residents."},
    {"role": "user",      "content": "Does it apply to US companies?"},
    {"role": "assistant", "content": "Yes. GDPR applies to any organization that processes personal data of EU residents, regardless of where the organization is located."},
    {"role": "user",      "content": "What are the main rights under GDPR?"},
    {"role": "assistant", "content": "GDPR grants data subjects: right of access (Article 15), right to rectification (Article 16), right to erasure (Article 17), right to data portability (Article 20), and right to object (Article 21)."},
    {"role": "user",      "content": "What about data processing agreements?"},
    {"role": "assistant", "content": "Article 28 GDPR requires a Data Processing Agreement (DPA) between a controller and any processor it engages. The DPA must specify: nature/purpose of processing, data types, obligations and rights."},
]

SAMPLE_CHUNKS = [
    {"text": "GDPR Article 28 requires a Data Processing Agreement between the controller and processor. The processor must only act on documented controller instructions.", "source": "gdpr_art28.pdf", "score": 0.95},
    {"text": "Under Article 28(3), the DPA must specify: subject matter, duration, nature and purpose of processing, type of personal data, categories of data subjects.", "source": "gdpr_art28.pdf", "score": 0.92},
    {"text": "Violation of Article 28 can result in administrative fines up to €10 million or 2% of global annual turnover, whichever is higher.", "source": "gdpr_enforcement.pdf", "score": 0.88},
    {"text": "The CCPA grants California consumers the right to know what personal information is collected, the right to delete, and the right to opt-out of sale.", "source": "ccpa_overview.pdf", "score": 0.45},
    {"text": "HIPAA Privacy Rule requires covered entities to implement administrative, physical, and technical safeguards for protected health information.", "source": "hipaa_overview.pdf", "score": 0.30},
]

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": "This comprehensive search tool allows you to search through our extensive database of regulatory compliance documents including GDPR, CCPA, HIPAA, SOC2, ISO27001 and many more frameworks. Supports full-text, semantic, and metadata-filtered search across all stored regulatory content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query", "examples": ["GDPR Article 28", "data breach notification"]},
                    "regulation": {"type": "string", "enum": ["gdpr", "ccpa", "hipaa", "soc2", "pci-dss", "iso27001"], "title": "Regulation", "default": "gdpr"},
                },
                "required": ["query"],
            },
        },
    }
]


async def main():
    print("=" * 60)
    print("Project 38 — Context Budget Engine")
    print("=" * 60)

    # Set up memory store with sample data (all 4 memory types)
    memory = MemoryStore()
    memory.store("past_session", "User asked about GDPR DPA last session", "episodic")
    memory.store("gdpr_art28", "Article 28 requires written DPA between controller and processor", "semantic")
    memory.store("style", "Always cite specific article numbers in answers", "procedural")
    memory.store("user_role", "User is a Data Protection Officer at a fintech company", "user_profile")

    question = "What are the specific requirements for a Data Processing Agreement under GDPR Article 28?"

    # Unmanaged baseline: no engine, just dump everything
    print("\n── Unmanaged baseline ──")
    unmanaged_msgs = (
        [{"role": "system", "content": SAMPLE_SYSTEM}]
        + SAMPLE_HISTORY
        + [{"role": "user", "content": question}]
        + [{"role": "system", "content": "\n".join(c["text"] for c in SAMPLE_CHUNKS)}]
    )
    unmanaged_tokens = count_messages_tokens(unmanaged_msgs)
    print(f"  Total tokens (unmanaged): {unmanaged_tokens}")

    # Managed: run through ContextEngine
    print("\n── Managed (ContextEngine) ──")
    engine = ContextEngine()
    ctx = await engine.assemble(
        question=question,
        system_prompt=SAMPLE_SYSTEM,
        history=SAMPLE_HISTORY,
        rag_chunks=SAMPLE_CHUNKS,
        tools=SAMPLE_TOOLS,
        memory_store=memory,
    )

    print(f"  Total tokens (managed):   {ctx.report.total_actual}")
    print()
    ctx.report.render()
    print()
    ctx.report.diff(unmanaged_tokens, ctx.report.total_actual)

    warnings = ctx.report.warnings()
    if warnings:
        print("\n⚠️  Budget warnings:")
        for w in warnings:
            print(f"  - {w}")

    print(f"\n  RAG chunks selected: {len(ctx.rag_chunks)}/{len(SAMPLE_CHUNKS)}")
    print(f"  Tool schemas compacted: {len(ctx.compact_tools)} tools")
    print(f"  History messages after management: {sum(1 for m in ctx.messages if m['role'] != 'system' and 'Memory' not in str(m.get('content',''))[:20])}")


if __name__ == "__main__":
    asyncio.run(main())
