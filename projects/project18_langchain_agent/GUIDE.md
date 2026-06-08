# LangChain Research Agent — Build Guide

## Prerequisites
```bash
pip install -r requirements.txt
# Get a free Tavily API key: https://tavily.com
```

---

## Phase 1 — LCEL Chains

### 1.1 What is LCEL?
LangChain Expression Language uses Python's `|` pipe operator to compose `Runnable` objects:
```python
chain = prompt | llm | parser
result = chain.invoke({"question": "..."})
```
Every component is a `Runnable` with `.invoke()`, `.stream()`, `.batch()`, `.ainvoke()`.

### 1.2 Build the RAG chain
Implement `src/chains/rag_chain.py`:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template("""
Answer based ONLY on the context below.
Context: {context}
Question: {question}
""")

llm = ChatLiteLLM(model=cfg.model, temperature=0.0)
parser = StrOutputParser()

# The chain: fetch context + question in parallel, then prompt → llm → parse
rag_chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough(),
    )
    | prompt
    | llm
    | parser
)
```

### 1.3 Structured output chain
```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class ResearchSummary(BaseModel):
    title: str = Field(description="Article title")
    key_points: list[str] = Field(description="3-5 key findings")
    confidence: float = Field(description="0.0-1.0 confidence score")

structured_llm = llm.with_structured_output(ResearchSummary)
chain = prompt | structured_llm
result: ResearchSummary = chain.invoke({"topic": "quantum computing"})
```

**Checkpoint:** `python -m src.chains.rag_chain` → prints answer

---

## Phase 2 — Custom Tools

### 2.1 `@tool` decorator
```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. Input: a math expression like '2 + 2 * 3'."""
    import ast, operator
    # safe eval — only allow numeric operations
    allowed = {ast.Add: operator.add, ast.Sub: operator.sub,
               ast.Mul: operator.mul, ast.Div: operator.truediv}
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body, allowed))
    except Exception as e:
        return f"Error: {e}"
```

### 2.2 Tavily web search
```python
from langchain_community.tools.tavily_search import TavilySearchResults

web_search = TavilySearchResults(max_results=5, api_key=cfg.tavily_api_key)
```

### 2.3 Wikipedia tool
```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
)
```

**Checkpoint:** call each tool directly — `calculator.invoke({"expression": "15 * 23"})`

---

## Phase 3 — ReAct Agent

### 3.1 Create the agent
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Pull the standard ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

tools = [web_search, wikipedia, calculator, rag_search_tool]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,       # prints Thought/Action/Observation
    max_iterations=10,
    handle_parsing_errors=True,
)
```

### 3.2 Invoke with memory
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,   # keep last 5 turns
)

executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Turn 1
executor.invoke({"input": "What is the population of Tokyo?"})

# Turn 2 — agent remembers Tokyo from turn 1
executor.invoke({"input": "How does that compare to London?"})
```

**Checkpoint:** agent uses at least 2 different tools to answer a research question.

---

## Phase 4 — Streaming

### 4.1 `astream_events` — recommended for production
```python
async def stream_agent(question: str):
    async for event in executor.astream_events(
        {"input": question}, version="v2"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            print(token, end="", flush=True)
        elif kind == "on_tool_start":
            print(f"\n[Tool: {event['name']}({event['data'].get('input')})]")
        elif kind == "on_tool_end":
            print(f"[Result: {str(event['data'].get('output'))[:100]}]")
```

### 4.2 Run with asyncio
```python
import asyncio
asyncio.run(stream_agent("Research the top 3 AI breakthroughs in 2025"))
```

**Checkpoint:** tokens stream in real-time, tool calls visible as they happen.

---

## Phase 5 — Custom Callbacks

### 5.1 Implement callback handler
```python
from langchain_core.callbacks import BaseCallbackHandler
import structlog

log = structlog.get_logger()

class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        log.info("llm_start", model=serialized.get("name"))

    def on_tool_start(self, serialized, input_str, **kwargs):
        log.info("tool_start", tool=serialized.get("name"), input=input_str[:100])

    def on_agent_action(self, action, **kwargs):
        log.info("agent_action", tool=action.tool, input=str(action.tool_input)[:100])

    def on_agent_finish(self, finish, **kwargs):
        log.info("agent_finish", output=str(finish.return_values)[:200])
```

### 5.2 Attach to executor
```python
executor = AgentExecutor(
    ...,
    callbacks=[LoggingCallbackHandler()],
)
```

---

## Production Notes

- **Rate limiting**: wrap `AgentExecutor.invoke` in `tenacity.retry` with exponential backoff
- **Max tokens**: set `max_tokens` on ChatLiteLLM to prevent runaway generation
- **Tool timeouts**: wrap tool functions in `asyncio.wait_for(coro, timeout=10)`
- **Observability**: use `langsmith` (LangChain's tracing platform) for full trace capture
