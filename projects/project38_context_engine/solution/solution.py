"""
Project 38 — Context Budget Engine (SOLUTION)
==============================================
Full implementation of all TODOs from starter.py.
"""
from __future__ import annotations
import asyncio, copy, json, re, string
from dataclasses import dataclass, field
from typing import Any
import litellm
from dotenv import load_dotenv

load_dotenv()
MODEL       = "openai/gpt-4o-mini"
CHEAP_MODEL = "openai/gpt-4o-mini"


# ──────────────────────────────────────────────────────────────────────────────
# 1 — Token Counting
# ──────────────────────────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = MODEL) -> int:
    """Count tokens using tiktoken. Falls back to cl100k_base for unknown models."""
    import tiktoken
    # Strip provider prefix e.g. "openai/gpt-4o-mini" → "gpt-4o-mini"
    clean_model = model.split("/")[-1] if "/" in model else model
    try:
        enc = tiktoken.encoding_for_model(clean_model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict], model: str = MODEL) -> int:
    """Count total tokens including per-message overhead (4 tokens each + 2 primer)."""
    total = 2  # primer tokens
    for m in messages:
        total += 4  # per-message overhead
        total += count_tokens(m.get("content") or "", model)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# 2 — Context Budget
# ──────────────────────────────────────────────────────────────────────────────

class BudgetError(Exception): pass

@dataclass
class ContextBudget:
    total:            int = 8_000
    system_prompt:    int = 400
    long_term_memory: int = 300
    rag_context:      int = 2_000
    history:          int = 2_000
    tool_schemas:     int = 400
    current_message:  int = 200
    output_reserve:   int = 2_900

    def compute_available(self) -> int:
        return (
            self.system_prompt + self.long_term_memory + self.rag_context
            + self.history + self.tool_schemas + self.current_message
        )

    def validate(self):
        used = self.compute_available() + self.output_reserve
        if used > self.total:
            raise BudgetError(
                f"Budget overcommitted: {used} > {self.total} "
                f"(sources={self.compute_available()}, reserve={self.output_reserve})"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 3 — Memory Store
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Memory:
    key:         str
    value:       str
    memory_type: str
    score:       float = 1.0


class MemoryStore:
    def __init__(self):
        self._store: list[Memory] = []

    def store(self, key: str, value: str, memory_type: str = "semantic"):
        self._store.append(Memory(key, value, memory_type))

    def retrieve(self, query: str, top_k: int = 5) -> list[Memory]:
        query_words = set(query.lower().split())
        scored = []
        for m in self._store:
            value_words = set(m.value.lower().split())
            if query_words:
                score = len(query_words & value_words) / len(query_words)
            else:
                score = 0.0
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def get_all(self, memory_type: str) -> list[Memory]:
        return [m for m in self._store if m.memory_type == memory_type]

    def format_memories(self, memories: list[Memory]) -> str:
        return "\n".join(f"[{m.memory_type}] {m.key}: {m.value}" for m in memories)


# ──────────────────────────────────────────────────────────────────────────────
# 4 — Text Compression (TF-IDF extractive)
# ──────────────────────────────────────────────────────────────────────────────

def compress_text(text: str, target_tokens: int, model: str = MODEL) -> str:
    """Extractive compression: score sentences by word frequency, greedy select."""
    if count_tokens(text, model) <= target_tokens:
        return text

    # Sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        # Hard truncation as last resort
        words = text.split()
        while words and count_tokens(" ".join(words), model) > target_tokens:
            words = words[:-5]
        return " ".join(words) + "..."

    # Word frequency scoring (simple TF)
    stop_words = set(string.punctuation) | {"the","a","an","is","are","was","were","in","of","to","and","or","for","on","at","by","with","that","this","it","its","be","as","from","have","has","had"}
    word_freq: dict[str, int] = {}
    for sent in sentences:
        for word in sent.lower().split():
            clean = word.strip(string.punctuation)
            if clean and clean not in stop_words:
                word_freq[clean] = word_freq.get(clean, 0) + 1

    # Score each sentence
    scored = []
    for i, sent in enumerate(sentences):
        words = [w.strip(string.punctuation).lower() for w in sent.split()]
        score = sum(word_freq.get(w, 0) for w in words if w not in stop_words)
        if words:
            score /= len(words)
        scored.append((score, i, sent))

    # Greedy select top sentences by score
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    tokens_used = 0
    for score, idx, sent in scored:
        t = count_tokens(sent, model)
        if tokens_used + t <= target_tokens:
            selected.append((idx, sent))
            tokens_used += t

    # Re-sort by original position
    selected.sort(key=lambda x: x[0])
    return " ".join(s for _, s in selected)


# ──────────────────────────────────────────────────────────────────────────────
# 5 — RAG Greedy Fill
# ──────────────────────────────────────────────────────────────────────────────

def fill_rag(chunks: list[dict], budget: int, model: str = MODEL) -> tuple[list[dict], int]:
    """Sort by score descending. Greedily fill until budget exhausted."""
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
    selected: list[dict] = []
    tokens_used = 0

    for chunk in sorted_chunks:
        t = count_tokens(chunk["text"], model)
        if tokens_used + t <= budget:
            selected.append(chunk)
            tokens_used += t
        elif chunk.get("score", 0.0) > 0.7:
            # Compress high-relevance chunk to fit remaining space
            remaining = budget - tokens_used
            if remaining > 20:
                compressed_text = compress_text(chunk["text"], remaining, model)
                compressed_chunk = {**chunk, "text": compressed_text}
                ct = count_tokens(compressed_text, model)
                selected.append(compressed_chunk)
                tokens_used += ct

    return selected, tokens_used


def format_rag(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["Context:"]
    for i, c in enumerate(chunks):
        lines.append(f"[{i+1}] {c['source']}: {c['text']}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 6 — History Management
# ──────────────────────────────────────────────────────────────────────────────

async def summarise_old_turns(old_messages: list[dict]) -> str:
    """LLM-summarise older turns into 3 bullets."""
    lines = []
    for m in old_messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m.get('content', '')}")
    transcript = "\n".join(lines)

    response = await litellm.acompletion(
        model=CHEAP_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Summarise the following conversation in 3 bullet points. Be very concise.",
            },
            {"role": "user", "content": transcript},
        ],
        max_tokens=120,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


async def manage_history(
    messages: list[dict],
    budget: int,
    keep_last_n: int = 4,
    model: str = MODEL,
) -> list[dict]:
    """Sliding window + LLM summary compression."""
    if count_messages_tokens(messages, model) <= budget:
        return messages

    # Separate system from non-system
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system  = [m for m in messages if m["role"] != "system"]

    # Keep last keep_last_n turn-pairs (user+assistant = 2 messages each pair)
    recent_n = keep_last_n * 2
    recent   = non_system[-recent_n:] if len(non_system) > recent_n else non_system
    older    = non_system[:-recent_n] if len(non_system) > recent_n else []

    if older:
        summary_text = await summarise_old_turns(older)
        summary_msg  = {"role": "system", "content": f"[Summary of earlier conversation]\n{summary_text}"}
        rebuilt = system_msgs + [summary_msg] + recent
    else:
        rebuilt = system_msgs + recent

    # Recurse if still over budget
    if count_messages_tokens(rebuilt, model) > budget:
        return await manage_history(rebuilt, budget, keep_last_n=max(1, keep_last_n - 1), model=model)

    return rebuilt


# ──────────────────────────────────────────────────────────────────────────────
# 7 — Tool Schema Compaction
# ──────────────────────────────────────────────────────────────────────────────

def compact_tools(tools: list[dict], budget: int, model: str = MODEL) -> tuple[list[dict], int]:
    """Strip tool schemas: first sentence of description, remove noise keys, abbreviate enums."""
    compacted = copy.deepcopy(tools)

    for tool in compacted:
        fn = tool.get("function", {})

        # Truncate description to first sentence (≤80 chars)
        desc = fn.get("description", "")
        sentences = re.split(r"(?<=[.!?])\s+", desc)
        short_desc = sentences[0][:80] if sentences else desc[:80]
        fn["description"] = short_desc

        # Clean up properties
        params = fn.get("parameters", {})
        for prop_name, prop in params.get("properties", {}).items():
            for noise_key in ("examples", "default", "title"):
                prop.pop(noise_key, None)
            # Abbreviate long enums
            if "enum" in prop and len(prop["enum"]) > 5:
                prop["enum"] = prop["enum"][:4] + ["..."]

    token_count = count_tokens(json.dumps(compacted), model)
    return compacted, token_count


# ──────────────────────────────────────────────────────────────────────────────
# 8 — Context Report
# ──────────────────────────────────────────────────────────────────────────────

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
    sources:      list[SourceReport] = field(default_factory=list)
    total_budget: int = 0
    total_actual: int = 0

    def warnings(self) -> list[str]:
        return [
            f"{s.name} exceeded budget by {s.overrun} tokens"
            for s in self.sources if s.overrun > 0
        ]

    def render(self):
        print(f"{'Source':<22} {'Budget':>8} {'Actual':>8}  Status")
        print("─" * 58)
        for s in self.sources:
            print(f"  {s.name:<20} {s.budget:>8} {s.actual:>8}  {s.status}")
        print("─" * 58)
        headroom = self.total_budget - self.total_actual
        pct = (1 - self.total_actual / self.total_budget) * 100 if self.total_budget else 0
        print(f"  {'TOTAL':<20} {self.total_budget:>8} {self.total_actual:>8}  headroom={headroom} ({pct:.1f}%)")

    def diff(self, unmanaged_tokens: int, managed_tokens: int, model: str = MODEL):
        """Print before/after savings."""
        savings = unmanaged_tokens - managed_tokens
        savings_pct = (savings / unmanaged_tokens * 100) if unmanaged_tokens else 0
        cost_per_1k = 0.0002  # gpt-4o-mini input rate $/1k tokens
        saved_per_call = (savings / 1000) * cost_per_1k
        saved_10k_day  = saved_per_call * 10_000
        saved_monthly  = saved_10k_day * 30

        print("── Token Savings Report ──")
        print(f"  Unmanaged : {unmanaged_tokens:>6} tokens")
        print(f"  Managed   : {managed_tokens:>6} tokens")
        print(f"  Saved     : {savings:>6} tokens  ({savings_pct:.1f}%)")
        print(f"  Cost saved: ${saved_per_call:.6f}/call")
        print(f"  At 10k/day: ${saved_10k_day:.2f}/day  →  ${saved_monthly:.2f}/month")


# ──────────────────────────────────────────────────────────────────────────────
# 9 — Context Engine
# ──────────────────────────────────────────────────────────────────────────────

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
        report   = ContextReport(total_budget=self.budget.total)
        messages: list[dict] = []

        # Step A — System prompt
        sys_tokens = count_tokens(system_prompt)
        messages.append({"role": "system", "content": system_prompt})
        report.sources.append(SourceReport("System prompt", self.budget.system_prompt, sys_tokens))

        # Step B — Long-term memory
        mem_text = ""
        if memory_store:
            memories = memory_store.retrieve(question, top_k=5)
            if memories:
                mem_text = memory_store.format_memories(memories)
                mem_tokens = count_tokens(mem_text)
                if mem_tokens > self.budget.long_term_memory:
                    mem_text = compress_text(mem_text, self.budget.long_term_memory)
                    mem_tokens = count_tokens(mem_text)
                messages.append({"role": "system", "content": f"[Memory]\n{mem_text}"})
                report.sources.append(SourceReport("Long-term memory", self.budget.long_term_memory, mem_tokens))
            else:
                report.sources.append(SourceReport("Long-term memory", self.budget.long_term_memory, 0))
        else:
            report.sources.append(SourceReport("Long-term memory", self.budget.long_term_memory, 0))

        # Step C — RAG context
        selected_chunks, rag_tokens = fill_rag(rag_chunks, self.budget.rag_context)
        if selected_chunks:
            rag_text = format_rag(selected_chunks)
            messages.append({"role": "system", "content": rag_text})
        report.sources.append(SourceReport("RAG context", self.budget.rag_context, rag_tokens))

        # Step D — History
        managed_history = await manage_history(history, self.budget.history)
        messages.extend(managed_history)
        hist_tokens = count_messages_tokens(managed_history)
        report.sources.append(SourceReport("History", self.budget.history, hist_tokens))

        # Step E — Tool schemas
        compacted_tools, tool_tokens = compact_tools(tools, self.budget.tool_schemas)
        report.sources.append(SourceReport("Tool schemas", self.budget.tool_schemas, tool_tokens))

        # Step F — Current message
        q_tokens = count_tokens(question)
        messages.append({"role": "user", "content": question})
        report.sources.append(SourceReport("Current message", self.budget.current_message, q_tokens))

        report.total_actual = count_messages_tokens(messages)
        return AssembledContext(messages, report, selected_chunks, compacted_tools)


# ──────────────────────────────────────────────────────────────────────────────
# Sample Data (identical to starter)
# ──────────────────────────────────────────────────────────────────────────────

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
    print("Project 38 — Context Budget Engine  (SOLUTION)")
    print("=" * 60)

    memory = MemoryStore()
    memory.store("past_session", "User asked about GDPR DPA last session", "episodic")
    memory.store("gdpr_art28", "Article 28 requires written DPA between controller and processor", "semantic")
    memory.store("style", "Always cite specific article numbers in answers", "procedural")
    memory.store("user_role", "User is a Data Protection Officer at a fintech company", "user_profile")

    question = "What are the specific requirements for a Data Processing Agreement under GDPR Article 28?"

    print("\n── Unmanaged baseline ──")
    unmanaged_msgs = (
        [{"role": "system", "content": SAMPLE_SYSTEM}]
        + SAMPLE_HISTORY
        + [{"role": "user", "content": question}]
        + [{"role": "system", "content": "\n".join(c["text"] for c in SAMPLE_CHUNKS)}]
    )
    unmanaged_tokens = count_messages_tokens(unmanaged_msgs)
    print(f"  Total tokens (unmanaged): {unmanaged_tokens}")

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
    else:
        print("\n✅  All budgets within limits")

    print(f"\n  RAG chunks selected: {len(ctx.rag_chunks)}/{len(SAMPLE_CHUNKS)}")
    print(f"  Tool schemas compacted: {len(ctx.compact_tools)} tools")


if __name__ == "__main__":
    asyncio.run(main())
