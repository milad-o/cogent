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
    role="worker",  # String: "worker", "supervisor", "autonomous", "reviewer"
    tools=[write_tool],
    instructions="You write compelling content.",
)

# Advanced API with AgentConfig
from agenticflow.agent import AgentConfig
from agenticflow.core.enums import AgentRole

config = AgentConfig(
    name="Writer",
    role=AgentRole.WORKER,
    model=model,
    tools=["write_poem", "write_story"],
    resilience_config=ResilienceConfig.aggressive(),
)
agent = Agent(config=config)
```

### RoleConfig Objects (Recommended)

Use role configuration objects for type-safe, immutable role definitions:

```python
from agenticflow import (
    SupervisorRole,
    WorkerRole,
    ReviewerRole,
    AutonomousRole,
    CustomRole,
)

# Supervisor - coordinates workers
supervisor = Agent(
    name="Manager",
    model=model,
    role=SupervisorRole(workers=["Analyst", "Writer"]),
)

# Worker - executes tasks with tools
worker = Agent(
    name="Analyst",
    model=model,
    role=WorkerRole(specialty="data analysis and visualization"),
    tools=[search, analyze],
)

# Reviewer - evaluates and approves work
reviewer = Agent(
    name="QA",
    model=model,
    role=ReviewerRole(criteria=["accuracy", "clarity", "completeness"]),
)

# Autonomous - independent agent with full capabilities
autonomous = Agent(
    name="Assistant",
    model=model,
    role=AutonomousRole(),
    tools=[search, write],
)

# Custom - hybrid role with explicit capability overrides
custom = Agent(
    name="TechnicalReviewer",
    model=model,
    role=CustomRole(
        base_role=AgentRole.REVIEWER,
        can_use_tools=True,  # Reviewer that can use tools!
    ),
    tools=[code_analyzer, linter],
)
```

**Benefits of RoleConfig objects:**
- Type-safe configuration
- Immutable (frozen dataclasses)
- Built-in prompt enhancement
- Clear, explicit role definitions
- IDE autocomplete and type checking

### Role-Specific Parameters (Backward Compatible)

You can also use string/enum roles with parameters:

```python
# Supervisor - with team members
supervisor = Agent(
    name="Manager",
    model=model,
    role=AgentRole.SUPERVISOR,  # or "supervisor"
    workers=["analyst", "writer"],  # Adds team members to prompt
)

# Worker - with specialty description
worker = Agent(
    name="Analyst", 
    model=model,
    role="worker",
    specialty="data analysis and visualization",  # Adds specialty to prompt
    tools=[search, analyze],
)

# Reviewer - with evaluation criteria
reviewer = Agent(
    name="QA",
    model=model,
    role="reviewer",
    criteria=["accuracy", "clarity", "completeness"],  # Adds criteria to prompt
)

# Autonomous - works independently, can finish
autonomous = Agent(
    name="Assistant",
    model=model,
    role="autonomous",
    tools=[search, write],
)
```

**Note:** While backward compatible, RoleConfig objects are recommended for new code.

### Custom Roles (Capability Overrides)

**Recommended:** Use `CustomRole` for hybrid capabilities:

```python
from agenticflow import CustomRole
from agenticflow.core import AgentRole

# Reviewer that can use tools
hybrid_reviewer = Agent(
    name="TechnicalReviewer",
    model=model,
    role=CustomRole(
        base_role=AgentRole.REVIEWER,
        can_use_tools=True,  # Override!
    ),
    tools=[code_analyzer, linter],
)

# Worker that can finish and delegate
orchestrator = Agent(
    name="Orchestrator",
    model=model,
    role=CustomRole(
        base_role=AgentRole.WORKER,
        can_finish=True,      # Override!
        can_delegate=True,    # Override!
    ),
    tools=[deployment_tool],
)
```

**Backward compatible:** You can also use capability overrides with string/enum roles:

```python
# Hybrid: Reviewer that can use tools
hybrid_reviewer = Agent(
    name="TechnicalReviewer",
    model=model,
    role="reviewer",
    can_use_tools=True,  # Override! Reviewer normally can't use tools
    tools=[code_analyzer, linter],
)

# Custom orchestrator: Worker that can finish and delegate
orchestrator = Agent(
    name="Orchestrator",
    model=model,
    role="worker",
    can_finish=True,      # Override! Worker normally can't finish
    can_delegate=True,   # Override! Worker normally can't delegate
    tools=[deployment_tool],
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

Enable extended thinking for complex problems with AI-controlled reasoning rounds.

### Basic Usage

```python
from agenticflow import Agent
from agenticflow.agent.reasoning import ReasoningConfig

# Simple: Enable with defaults
agent = Agent(
    name="Analyst",
    model=model,
    reasoning=True,  # Default config
)

result = await agent.run("Analyze this complex problem...")
```

### Custom Configuration

```python
# Full control with ReasoningConfig
agent = Agent(
    name="DeepThinker",
    model=model,
    reasoning=ReasoningConfig(
        max_thinking_rounds=15,         # AI decides when ready (up to 15)
        style=ReasoningStyle.CRITICAL,  # Critical reasoning style
        show_thinking=True,             # Include thoughts in output
    ),
)
```

### Per-Call Overrides

Enable or customize reasoning for specific calls:

```python
# Agent without reasoning by default
agent = Agent(name="Helper", model=model, reasoning=False)

# Simple task - no reasoning
result = await agent.run("What time is it?")

# Complex task - enable reasoning
result = await agent.run(
    "Analyze this codebase architecture",
    reasoning=True,  # Enable for this call
)

# Very complex - custom config
result = await agent.run(
    "Debug this complex issue",
    reasoning=ReasoningConfig(
        max_thinking_rounds=10,
        style=ReasoningStyle.ANALYTICAL,
    ),
)
```

### Reasoning Styles

- `ANALYTICAL`: Step-by-step logical breakdown (default)
- `EXPLORATORY`: Consider multiple approaches
- `CRITICAL`: Question assumptions, find flaws
- `CREATIVE`: Generate novel solutions

### AI-Controlled Rounds

The AI signals when reasoning is complete via `<ready>true</ready>` tags. The `max_thinking_rounds` is a safety limit, not a fixed count:

```python
ReasoningConfig.standard()  # max 10 rounds (safety net)
ReasoningConfig.deep()      # max 15 rounds (complex problems)
```

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
