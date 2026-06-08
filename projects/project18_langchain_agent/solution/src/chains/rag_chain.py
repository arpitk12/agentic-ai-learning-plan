"""LangChain LCEL RAG chain and structured output chain."""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

from src.config import cfg

# ── LLM ───────────────────────────────────────────────────────────────────

def get_llm():
    from langchain_litellm import ChatLiteLLM
    return ChatLiteLLM(model=cfg.model, temperature=0.0, max_tokens=2048)


# ── Retriever ─────────────────────────────────────────────────────────────

def get_retriever():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    try:
        vs = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        # Create empty vectorstore if index doesn't exist yet
        vs = FAISS.from_texts(["placeholder"], embeddings)
    return vs.as_retriever(search_kwargs={"k": 5})


# ── RAG Chain (LCEL) ──────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. If the context doesn't contain
the answer, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
)


def build_rag_chain():
    retriever = get_retriever()
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    return (
        RunnableParallel(
            context=(retriever | format_docs),
            question=RunnablePassthrough(),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


# ── Structured Output Chain ───────────────────────────────────────────────

class ResearchSummary(BaseModel):
    title: str = Field(description="Concise title for the research")
    key_points: list[str] = Field(description="3-5 key findings or facts")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in accuracy (0-1)")
    follow_up_questions: list[str] = Field(description="2-3 questions for deeper research")


STRUCT_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the following research findings about {topic}:\n\n{content}"
)


def build_structured_chain():
    llm = get_llm()
    structured_llm = llm.with_structured_output(ResearchSummary)
    return STRUCT_PROMPT | structured_llm


if __name__ == "__main__":
    chain = build_rag_chain()
    result = chain.invoke("What information is available in the knowledge base?")
    print(result)
