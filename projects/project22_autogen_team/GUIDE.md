# AutoGen Multi-Agent Team — Build Guide

## Prerequisites
```bash
pip install -r requirements.txt
docker pull python:3.11-slim   # for code execution sandbox
```

---

## Phase 1 — Two-Agent Nested Chat (Start Here)

The simplest AutoGen pattern: one assistant + one executor.

### 1.1 Basic coding assistant
```python
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

# Code executor — runs code blocks in a temp directory
executor = LocalCommandLineCodeExecutor(
    work_dir="tmp/code",
    timeout=30,
)

# UserProxyAgent acts as the human/executor
user_proxy = UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER",      # fully automated
    code_execution_config={"executor": executor},
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    max_consecutive_auto_reply=10,
)

# AssistantAgent is the coder
assistant = AssistantAgent(
    name="Coder",
    system_message="""You are an expert Python programmer.
    When asked to implement something:
    1. Write clean, well-commented Python code in a ```python block
    2. Include docstrings and type hints
    3. Write pytest tests in a separate ```python block
    4. Ask the executor to run the tests
    5. Fix any failures and re-run
    6. When all tests pass, say TERMINATE""",
    llm_config={"config_list": [{"model": cfg.model, "api_key": cfg.api_key}]},
)

# Start the conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a binary search implementation with comprehensive tests",
)
```

### 1.2 How code execution works
When AssistantAgent produces a message containing:
````
```python
def binary_search(arr, target):
    ...
```
````
UserProxyAgent automatically:
1. Extracts the code block
2. Writes it to `tmp/code/solution.py`
3. Runs `python tmp/code/solution.py`
4. Returns stdout/stderr back into the conversation

**Checkpoint:** Run the above and watch the agent iterate until tests pass.

---

## Phase 2 — Docker Code Executor (Production Safety)

### 2.1 Why Docker?
Local execution is dangerous — the LLM could generate `import os; os.system("rm -rf /")`
Docker sandboxes execution:

```python
from autogen.coding import DockerCommandLineCodeExecutor

executor = DockerCommandLineCodeExecutor(
    image="python:3.11-slim",
    timeout=30,
    work_dir=Path("tmp/code"),
    bind_dir=Path("tmp/code"),      # mount local dir into container
    auto_remove=True,               # remove container after execution
)
```

### 2.2 Pre-install packages in the container
```python
executor = DockerCommandLineCodeExecutor(
    image="python:3.11-slim",
    # Run pip install before executing user code
    init_command="pip install -q pytest numpy pandas",
    timeout=60,
    work_dir=Path("tmp/code"),
)
```

---

## Phase 3 — Multi-Agent GroupChat

### 3.1 Define specialized agents
```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

llm_config = {"config_list": [{"model": cfg.model, "api_key": cfg.api_key}]}

product_manager = AssistantAgent(
    name="ProductManager",
    system_message="""You are a product manager. Your role:
    1. Clarify requirements from the user's feature request
    2. Write user stories in the format: "As a [user], I want [feature] so that [benefit]"
    3. Define acceptance criteria
    4. Do NOT write code. Delegate to the Architect.""",
    llm_config=llm_config,
)

architect = AssistantAgent(
    name="Architect",
    system_message="""You are a software architect. Your role:
    1. Propose a system design for the feature
    2. Define interfaces and data models
    3. Suggest design patterns (no implementation)
    4. Approve the Developer's code structure before they begin""",
    llm_config=llm_config,
)

developer = AssistantAgent(
    name="Developer",
    system_message="""You are a senior Python developer. Your role:
    1. Implement the feature based on the Architect's design
    2. Write production-quality code with type hints and docstrings
    3. Put ALL code in a single ```python code block
    4. Ask the Executor to run it after writing""",
    llm_config=llm_config,
)

executor = UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER",
    code_execution_config={"executor": DockerCommandLineCodeExecutor(...)},
    is_termination_msg=lambda m: "TERMINATE" in m.get("content", ""),
)

tester = AssistantAgent(
    name="Tester",
    system_message="""You are a QA engineer. Your role:
    1. Write comprehensive pytest tests for the Developer's code
    2. Include unit tests, edge cases, and error cases
    3. Put tests in a ```python code block and ask Executor to run them
    4. If tests fail, tell Developer what to fix""",
    llm_config=llm_config,
)

reviewer = AssistantAgent(
    name="Reviewer",
    system_message="""You are a code reviewer. Your role:
    1. Review the final code for quality, security, and maintainability
    2. Check: naming conventions, error handling, performance, security
    3. Provide specific actionable feedback
    4. When satisfied, say TERMINATE to end the session""",
    llm_config=llm_config,
)
```

### 3.2 Assemble the GroupChat
```python
groupchat = GroupChat(
    agents=[product_manager, architect, developer, executor, tester, reviewer],
    messages=[],
    max_round=30,
    speaker_selection_method="auto",  # LLM decides who speaks next
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    is_termination_msg=lambda m: "TERMINATE" in m.get("content", ""),
)

# Start the conversation — executor initiates on behalf of the "user"
executor.initiate_chat(
    manager,
    message="""Feature request: Build a REST API endpoint for user registration.
    Requirements:
    - POST /api/users/register
    - Validate: email format, password strength (8+ chars, 1 uppercase, 1 number)
    - Return: {user_id, email, created_at} on success
    - Return: {error, field} on validation failure
    """,
)
```

---

## Phase 4 — Custom Speaker Selection

### 4.1 Deterministic turn order
```python
TURN_ORDER = [
    "ProductManager",
    "Architect",
    "Developer",
    "Executor",
    "Tester",
    "Executor",   # run tests
    "Reviewer",
]

def structured_speaker(last_speaker, groupchat):
    messages = groupchat.messages
    if not messages:
        return groupchat.agents[0]  # ProductManager first

    # Find current position in turn order
    last_name = last_speaker.name
    try:
        idx = TURN_ORDER.index(last_name)
        next_name = TURN_ORDER[min(idx + 1, len(TURN_ORDER) - 1)]
    except ValueError:
        next_name = "Developer"

    return next(a for a in groupchat.agents if a.name == next_name)

groupchat = GroupChat(
    ...,
    speaker_selection_method=structured_speaker,
)
```

---

## Framework Comparison

| | AutoGen | CrewAI | LangGraph |
|---|---|---|---|
| Style | Conversation-based | Task-pipeline | Graph-based |
| Code execution | Built-in (Docker/local) | Via tools | Via node function |
| Multi-agent coordination | GroupChat / nesting | Sequential / hierarchical | Explicit edges |
| State | Message history | Task context | TypedDict |
| Human-in-the-loop | `human_input_mode="ALWAYS"` | No built-in | `interrupt()` |
| Best for | Coding tasks, back-and-forth | Content pipelines | Complex workflows |
