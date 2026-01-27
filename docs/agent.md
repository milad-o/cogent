# Agent Module

The `cogent.agent` module defines the core agent abstraction - autonomous entities that can think, act, and communicate.

## Overview

Agents are the primary actors in the system. Each agent has:
- A unique identity and role
- Configuration defining its capabilities
- Runtime state tracking its activity
- Access to tools and the event bus

```python
from cogent import Agent

# Simple string model (recommended for v1.14.1+)
agent = Agent(
    name="Researcher",
    model="gpt4",  # Auto-resolves to gpt-4o
    tools=[search_tool],
    instructions="You are a research assistant.",
)

# With provider prefix
agent = Agent(
    name="Researcher",
    model="anthropic:claude",  # Explicit provider
    tools=[search_tool],
)

# Medium-level: Factory function
from cogent.models import create_chat
agent = Agent(
    name="Researcher",
    model=create_chat("gpt4"),
    tools=[search_tool],
)

# Low-level: Full control
from cogent.models import OpenAIChat
model = OpenAIChat(model="gpt-4o", temperature=0.7)
agent = Agent(
    name="Researcher",
    model=model,
    tools=[search_tool],
)

result = await agent.run("Find information about quantum computing")
```

## Core Classes

### Agent

The main agent class with multiple construction patterns:

```python
from cogent import Agent

# Simplified API (recommended)
agent = Agent(
    name="Writer",
    model="gpt4",  # String model - auto-resolves to gpt-4o
    role="worker",  # String: "worker", "supervisor", "autonomous", "reviewer"
    tools=[write_tool],
    instructions="You write compelling content.",
)

# With provider prefix for other providers
agent = Agent(
    name="Writer",
    model="anthropic:claude-sonnet-4",
    role="worker",
)

# Advanced API with AgentConfig
from cogent.agent import AgentConfig
from cogent.core.enums import AgentRole
from cogent.models import create_chat

config = AgentConfig(
    name="Writer",
    role=AgentRole.WORKER,
    model=create_chat("gpt4"),
    tools=["write_poem", "write_story"],
    resilience_config=ResilienceConfig.aggressive(),
)
agent = Agent(config=config)
```

### RoleConfig Objects (Recommended)

Use role configuration objects for type-safe, immutable role definitions:

```python
from cogent import (
    SupervisorRole,
    WorkerRole,
    ReviewerRole,
    AutonomousRole,
    CustomRole,
)

# Supervisor - coordinates workers
supervisor = Agent(
    name="Manager",
    model="gpt4",  # String model
    role=SupervisorRole(workers=["Analyst", "Writer"]),
)

# Worker - executes tasks with tools
worker = Agent(
    name="Analyst",
    model="claude",  # Alias for claude-sonnet-4
    role=WorkerRole(specialty="data analysis and visualization"),
    tools=[search, analyze],
)

# Reviewer - evaluates and approves work
reviewer = Agent(
    name="QA",
    model="gemini-pro",  # Alias for gemini-2.5-pro
    role=ReviewerRole(criteria=["accuracy", "clarity", "completeness"]),
)

# Autonomous - independent agent with full capabilities
autonomous = Agent(
    name="Assistant",
    model="anthropic:claude-opus-4",  # Provider prefix
    role=AutonomousRole(),
)    tools=[search, write],
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
from cogent import CustomRole
from cogent.core import AgentRole

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

Roles define **capabilities** (what an agent CAN do) and inject **system prompts** that guide LLM behavior. They don't define personalities - that comes from your `instructions`.

### Role Capabilities

```
┌─────────────┬────────────┬──────────────┬───────────────┐
│ Role        │ can_finish │ can_delegate │ can_use_tools │
├─────────────┼────────────┼──────────────┼───────────────┤
│ WORKER      │     ❌     │      ❌      │      ✅       │
│ SUPERVISOR  │     ✅     │      ✅      │      ❌       │
│ AUTONOMOUS  │     ✅     │      ❌      │      ✅       │
│ REVIEWER    │     ✅     │      ❌      │      ❌       │
└─────────────┴────────────┴──────────────┴───────────────┘
```

**When to Use:**
- **WORKER**: Executes tasks with tools, reports back
- **SUPERVISOR**: Coordinates workers, makes final decisions
- **AUTONOMOUS**: Independent operation, full lifecycle
- **REVIEWER**: Evaluates work, approves/rejects

### How Roles Work

Roles affect agent behavior in two ways:

**1. Capability Controls** - What the agent is allowed to do:
```python
# WORKER can use tools but cannot finish
worker = Agent(name="Analyst", model=model, role="worker", tools=[analyze_tool])
assert worker.can_use_tools == True
assert worker.can_finish == False  # Must report to supervisor

# AUTONOMOUS can use tools AND finish
autonomous = Agent(name="Assistant", model=model, role="autonomous", tools=[search_tool])
assert autonomous.can_use_tools == True
assert autonomous.can_finish == True  # Can conclude independently
```

**2. System Prompt Injection** - How the LLM thinks:

Each role gets a specialized system prompt that guides its behavior:

- **WORKER**: "Execute tasks using tools... You cannot finish the workflow yourself"
- **SUPERVISOR**: "Delegate tasks to workers... Provide FINAL ANSWER when complete"
- **AUTONOMOUS**: "Work independently... Finish when the task is complete"
- **REVIEWER**: "Evaluate work quality... Approve or request revisions"

**Example - See the difference:**
```python
# WORKER won't conclude
worker = Agent(name="Worker", model=model, role="worker")
result = await worker.run("What is Python?")
# Response: "Python is a programming language..." (no conclusion)

# AUTONOMOUS will conclude
autonomous = Agent(name="Assistant", model=model, role="autonomous")
result = await autonomous.run("What is Python?")
# Response: "FINAL ANSWER: Python is a high-level programming language..."
```

### When to Use Each Role

**WORKER** - Task execution:
```python
# ✅ Good: Has tools, reports results
data_analyst = Agent(
    name="DataAnalyst",
    model=model,
    role="worker",
    tools=[load_data, analyze, plot],
    instructions="Analyze datasets and create visualizations",
)

# In a Flow, supervisor coordinates workers
```

**SUPERVISOR** - Team coordination:
```python
# ✅ Good: Delegates to workers, makes final decisions
manager = Agent(
    name="Manager",
    model=model,
    role="supervisor",
    instructions="Coordinate the research team to deliver comprehensive reports",
)

# LLM will try to delegate: "DELEGATE TO researcher: Find information about..."
```

**AUTONOMOUS** - Independent agents:
```python
# ✅ Good: Standalone assistant, full capability
assistant = Agent(
    name="Assistant",
    model=model,
    role="autonomous",
    tools=[search, calculator, send_email],
    instructions="Help users with their requests",
)

# Can use tools AND provide final answers independently
```

**REVIEWER** - Quality control:
```python
# ✅ Good: Evaluates quality, no tool execution
qa = Agent(
    name="QualityAssurance",
    model=model,
    role="reviewer",
    instructions="Review code for quality, security, and best practices",
)

# LLM focuses on judgment: "FINAL ANSWER: Approved" or "REVISION NEEDED: ..."
```

### Capability Overrides

Override role capabilities when needed:
```python
# Hybrid: Reviewer that can use tools
tech_reviewer = Agent(
    name="TechnicalReviewer",
    model=model,
    role="reviewer",
    can_use_tools=True,  # Override! Run automated checks
    tools=[lint_code, run_tests],
)
```

See `examples/basics/role_behavior.py` for real LLM behavior examples.

## Memory

Enable conversation memory for multi-turn interactions:

```python
from cogent.agent import InMemorySaver

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
from cogent.agent import ResilienceConfig, RetryPolicy

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
from cogent import Agent
from cogent.agent.reasoning import ReasoningConfig

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
from cogent.agent import SpawningConfig, AgentSpec

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
from cogent import Agent
from cogent.observability import ObservabilityLevel

# Boolean shorthand
agent = Agent(name="Worker", model=model, verbosity=True)  # Progress level

# String levels
agent = Agent(name="Worker", model=model, verbosity="debug")

# Enum (explicit)
agent = Agent(name="Worker", model=model, verbosity=ObservabilityLevel.DEBUG)

# Integer (0-5)
agent = Agent(name="Worker", model=model, verbosity=4)  # DEBUG

# Advanced: Full control with observer
from cogent.observability import Observer

observer = Observer.debug()
agent = Agent(name="Worker", model=model, observer=observer)
```

**Verbosity levels:**

| Level | Int | String | Description |
|-------|-----|--------|-------------|
| `OFF` | 0 | `"off"` | No output |
| `RESULT` | 1 | `"result"`, `"minimal"` | Only final results |
| `PROGRESS` | 2 | `"progress"`, `"normal"` | Key milestones (default for `True`) |
| `DETAILED` | 3 | `"detailed"`, `"verbose"` | Tool calls, timing |
| `DEBUG` | 4 | `"debug"` | Everything including internal events |
| `TRACE` | 5 | `"trace"` | Maximum detail + execution graph |

**Priority:** `observer` parameter takes precedence over `verbosity`.

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
from cogent.agent import (
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
