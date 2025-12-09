# Agent Module

The `agenticflow.agent` module defines the core agent abstraction - autonomous entities that can think, act, and communicate.

## Overview

Agents are the primary actors in the system. Each agent has:
- A unique identity and role
- Configuration defining its capabilities
- Runtime state tracking its activity
- Access to tools and the event bus

```python
from agenticflow import Agent
from agenticflow.models import ChatModel

model = ChatModel(model="gpt-4o")

agent = Agent(
    name="Researcher",
    model=model,
    tools=[search_tool],
    instructions="You are a research assistant.",
)

result = await agent.run("Find information about quantum computing")
```

## Core Classes

### Agent

The main agent class with multiple construction patterns:

```python
from agenticflow import Agent

# Simplified API (recommended)
agent = Agent(
    name="Writer",
    model=model,
    role="worker",  # or "supervisor", "autonomous", "reviewer"
    tools=[write_tool],
    instructions="You write compelling content.",
)

# Advanced API with AgentConfig
from agenticflow.agent import AgentConfig

config = AgentConfig(
    name="Writer",
    role=AgentRole.WORKER,
    model=model,
    tools=["write_poem", "write_story"],
    resilience_config=ResilienceConfig.aggressive(),
)
agent = Agent(config=config)
```

### Role-Specific Factory Methods

```python
# Supervisor - coordinates work, can finish workflows
supervisor = Agent.as_supervisor(
    name="Manager",
    model=model,
    workers=["analyst", "writer"],
)

# Worker - executes tasks, cannot finish workflows
worker = Agent.as_worker(
    name="Analyst", 
    model=model,
    tools=[search, analyze],
)

# Autonomous - works independently, can finish
autonomous = Agent.as_autonomous(
    name="Assistant",
    model=model,
    tools=[search, write],
)

# Reviewer - approves/rejects, can finish
reviewer = Agent.as_reviewer(
    name="QA",
    model=model,
)
```

## Role System

Roles define capabilities (what an agent CAN do), not personalities:

| Role | can_finish | can_delegate | can_use_tools |
|------|------------|--------------|---------------|
| WORKER | ❌ | ❌ | ✅ |
| SUPERVISOR | ✅ | ✅ | ❌ |
| AUTONOMOUS | ✅ | ❌ | ✅ |
| REVIEWER | ✅ | ❌ | ❌ |

## Memory

Enable conversation memory for multi-turn interactions:

```python
from agenticflow.agent import InMemorySaver

agent = Agent(
    name="Assistant",
    model=model,
    memory=InMemorySaver(),
)

# Run with thread-based memory
response = await agent.run("Hi, I'm Alice", thread_id="conv-1")
response = await agent.run("What's my name?", thread_id="conv-1")  # Remembers!
```

### Memory Backends

- **InMemorySaver**: In-memory storage (for testing)
- **SqliteSaver**: SQLite-based persistence (coming soon)
- **Custom**: Implement `AgentMemory` protocol

## Resilience

Built-in fault tolerance with retries, circuit breakers, and fallbacks:

```python
from agenticflow.agent import ResilienceConfig, RetryPolicy

agent = Agent(
    name="Worker",
    model=model,
    resilience=ResilienceConfig(
        retry_policy=RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            strategy="exponential",
        ),
    ),
)
```

### Resilience Components

- **RetryPolicy**: Configure retry behavior with exponential/linear backoff
- **CircuitBreaker**: Prevent cascading failures
- **FallbackRegistry**: Define fallback behaviors for failures

## Human-in-the-Loop (HITL)

Enable human oversight for sensitive operations:

```python
agent = Agent(
    name="Executor",
    model=model,
    tools=[delete_file, send_email],
    interrupt_on={
        "tools": ["delete_file", "send_email"],  # Require approval
    },
)

try:
    result = await agent.run("Delete temp files")
except InterruptedException as e:
    # Human reviews pending action
    decision = HumanDecision(approved=True)
    result = await agent.resume(e.state, decision)
```

## Reasoning

Enable extended thinking for complex problems:

```python
from agenticflow.agent import ReasoningConfig

agent = Agent(
    name="Analyst",
    model=model,
    reasoning=ReasoningConfig(
        style="chain_of_thought",
        max_thinking_tokens=2000,
    ),
)

result = await agent.run("Analyze this complex problem...")
print(result.thinking_steps)  # Access reasoning trace
```

### Reasoning Styles

- `chain_of_thought`: Step-by-step reasoning
- `tree_of_thought`: Explore multiple paths
- `self_consistency`: Multiple samples for consensus

## Structured Output

Enforce response schemas with validation:

```python
from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: str | None = Field(None, description="Phone number")

agent = Agent(
    name="Extractor",
    model=model,
    output=ContactInfo,  # Enforce schema
)

result = await agent.run("Extract: John Doe, john@acme.com")
print(result.data)  # ContactInfo(name="John Doe", ...)
```

## TaskBoard

Human-like task tracking for complex workflows:

```python
agent = Agent(
    name="Researcher",
    model=model,
    tools=[search, summarize],
    taskboard=True,  # Adds task management tools
)

result = await agent.run("Research Python async patterns")
print(agent.taskboard.summary())
```

## Streaming

Enable token-by-token streaming:

```python
agent = Agent(
    name="Writer",
    model=model,
    stream=True,
)

async for chunk in agent.run("Write a story", stream=True):
    print(chunk.content, end="", flush=True)
```

## Spawning

Dynamic agent creation at runtime:

```python
from agenticflow.agent import SpawningConfig, AgentSpec

agent = Agent(
    name="Coordinator",
    model=model,
    spawning=SpawningConfig(
        allowed_specs=[
            AgentSpec(name="researcher", tools=["search"]),
            AgentSpec(name="writer", tools=["write"]),
        ],
    ),
)

# Agent can spawn sub-agents during execution
result = await agent.run("Research and write about AI")
```

## Observability

Built-in observability for standalone usage:

```python
agent = Agent(
    name="Worker",
    model=model,
    verbose="debug",  # "progress" | "verbose" | "debug" | "trace"
)
```

Verbosity levels:
- `False`: Silent
- `True`/`"progress"`: Show thinking/responding
- `"verbose"`: Include agent outputs
- `"debug"`: Include tool calls
- `"trace"`: Maximum detail

## API Reference

### Agent Methods

| Method | Description |
|--------|-------------|
| `run(task)` | Execute a task and return result |
| `chat(message, thread_id)` | Chat with memory support |
| `think(prompt)` | Single reasoning step |
| `stream_chat(message)` | Streaming chat response |
| `resume(state, decision)` | Resume after HITL interrupt |

### AgentConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Agent name |
| `role` | `AgentRole` | Agent role |
| `model` | `BaseChatModel` | Chat model |
| `tools` | `list[str]` | Tool names |
| `system_prompt` | `str` | System instructions |
| `resilience_config` | `ResilienceConfig` | Fault tolerance |
| `interrupt_on` | `dict` | HITL triggers |
| `stream` | `bool` | Enable streaming |

## Exports

```python
from agenticflow.agent import (
    # Core
    Agent,
    AgentConfig,
    AgentState,
    # Memory
    AgentMemory,
    MemorySnapshot,
    InMemorySaver,
    ThreadConfig,
    # Roles
    RoleBehavior,
    get_role_prompt,
    get_role_behavior,
    # Resilience
    RetryStrategy,
    RetryPolicy,
    CircuitBreaker,
    ResilienceConfig,
    ToolResilience,
    # HITL
    InterruptReason,
    HumanDecision,
    InterruptedException,
    # TaskBoard
    TaskBoard,
    TaskBoardConfig,
    Task,
    TaskStatus,
    # Reasoning
    ReasoningConfig,
    ReasoningStyle,
    ThinkingStep,
    # Output
    ResponseSchema,
    StructuredResult,
    # Spawning
    AgentSpec,
    SpawningConfig,
    SpawnManager,
)
```
