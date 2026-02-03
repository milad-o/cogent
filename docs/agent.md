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

# In multi-agent setup, supervisor coordinates workers
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

## TaskBoard

Enable task tracking for complex, multi-step work:

```python
agent = Agent(
    name="ProjectManager",
    model="gpt-4o-mini",
    instructions="You are a helpful project manager.",
    taskboard=True,  # Enables task tracking tools
)

result = await agent.run("Plan a REST API for a todo app")

# Check taskboard after execution
print(agent.taskboard.summary())
```

### TaskBoard Tools

When `taskboard=True`, the agent gets these tools:

| Tool | Description |
|------|-------------|
| `add_task` | Create a new task to track |
| `update_task` | Update task status (pending, in_progress, completed, failed, blocked) |
| `add_note` | Record observations and findings |
| `verify_task` | Verify a task was completed correctly |
| `get_taskboard_status` | See overall progress |

### How It Works

1. **Instructions injected** — Agent receives guidance on when/how to use taskboard
2. **LLM decides** — For complex tasks, the agent breaks them into subtasks
3. **Progress tracked** — Tasks have status, notes, and verification
4. **Summary available** — `agent.taskboard.summary()` shows progress

### TaskBoard Configuration

```python
from cogent.agent.taskboard import TaskBoardConfig

agent = Agent(
    name="Worker",
    model="gpt4",
    taskboard=TaskBoardConfig(
        include_instructions=True,  # Inject usage instructions (default: True)
        max_tasks=50,               # Maximum tasks to track
        track_time=True,            # Track task timing
    ),
)
```

See `examples/advanced/taskboard.py` for a complete example.

## Memory (4-Layer Architecture)

Cogent provides a 4-layer memory architecture:

| Layer | Parameter | Purpose |
|-------|-----------|---------|
| 1 | `conversation=True` | Thread-based message history (default ON) |
| 2 | `acc=True` | Agentic Context Compression - prevents drift |
| 3 | `memory=True` | Long-term memory with remember/recall tools |
| 4 | `cache=True` | Semantic cache for tool outputs |

See [Memory Module](./memory.md) for detailed explanation of how each layer works.

# Layer 3: Long-term memory with tools
agent = Agent(name="Assistant", model="gpt4", memory=True)
# Agent gets remember(), recall(), forget() tools

# Layer 4: Semantic cache for expensive tools
agent = Agent(name="Assistant", model="gpt4", cache=True)

# All layers together
agent = Agent(
    name="SuperAgent",
    model="gpt4",
    acc=True,     # Prevents context drift
    memory=True,  # Long-term facts
    cache=True,   # Cache tool outputs
)
```

### ACC (Agentic Context Compression)

For long conversations (>10 turns), enable ACC to prevent memory drift:

```python
from cogent.memory.acc import AgentCognitiveCompressor

# Simple: Enable with defaults
agent = Agent(name="Assistant", model="gpt4", acc=True)

# Advanced: Custom ACC with specific bounds
acc = AgentCognitiveCompressor(max_constraints=5, max_entities=20)
agent = Agent(name="Assistant", model="gpt4", acc=acc)
```

See [docs/acc.md](acc.md) for detailed ACC documentation.

### Semantic Cache

For expensive tools, enable semantic caching to avoid redundant calls:

```python
from cogent.memory import SemanticCache
from cogent.models import create_embedding

# Simple: Enable with defaults
agent = Agent(name="Assistant", model="gpt4", cache=True)

# Advanced: Custom SemanticCache instance
embed = create_embedding("openai", "text-embedding-3-small")
cache = SemanticCache(
    embedding=embed,
    similarity_threshold=0.90,  # Stricter matching
    max_entries=5000,
    default_ttl=3600,  # 1 hour
)
agent = Agent(name="Assistant", model="gpt4", cache=cache)
```

See [docs/memory.md#semantic-cache](memory.md#semantic-cache) for detailed cache documentation.

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
from typing import Literal, Union
from enum import Enum

# Structured models
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
print(result.content.data)  # ContactInfo(name="John Doe", ...)

# Bare types - return primitive values directly
agent = Agent(name="Reviewer", model=model, output=Literal["PROCEED", "REVISE"])
result = await agent.run("Review this code")
print(result.content.data)  # "PROCEED" (bare string, not wrapped)

# Collections - wrap in models for reliability
class Tags(BaseModel):
    items: list[str]

agent = Agent(name="Tagger", model=model, output=Tags)
result = await agent.run("Extract tags")
print(result.content.data.items)  # ["tag1", "tag2", ...]

# Union types - polymorphic responses
class Success(BaseModel):
    status: Literal["success"] = "success"
    result: str

class Error(BaseModel):
    status: Literal["error"] = "error"
    message: str

agent = Agent(name="Handler", model=model, output=Union[Success, Error])
# Agent chooses which schema based on content

# Enum types
class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"

agent = Agent(name="Prioritizer", model=model, output=Priority)
result = await agent.run("Critical issue!")
print(result.content.data)  # Priority.HIGH

# Dynamic structure - agent decides fields
agent = Agent(name="Analyzer", model=model, output=dict)
result = await agent.run("Analyze user feedback")
print(result.content.data)  # {"sentiment": "positive", "score": 8, ...}

# None type - confirmations
agent = Agent(name="Executor", model=model, output=type(None))
result = await agent.run("Delete temp files")
print(result.content.data)  # None

# Other bare types: str, int, bool, float
agent = Agent(name="Counter", model=model, output=int)
result = await agent.run("How many items?")
print(result.content.data)  # 42 (bare int)
```

Supported schema types:
- **Pydantic models** - Full validation with `BaseModel`
- **Dataclasses** - Standard Python dataclasses
- **TypedDict** - Typed dictionaries
- **Bare primitives** - `str`, `int`, `bool`, `float`
- **Bare Literal** - `Literal["A", "B", ...]` for constrained choices
- **Collections** - `list[T]`, `set[T]`, `tuple[T, ...]` (wrap in models for reliability)
- **Union types** - `Union[A, B]` for polymorphic responses
- **Enum types** - `class MyEnum(str, Enum)` for type-safe choices
- **None type** - `type(None)` for confirmation responses
- **dict** - Agent-decided dynamic structure (any fields)
- **JSON Schema** - Raw JSON Schema dicts

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

## Subagents (Native Delegation)

**New in v0.x.x**: Native subagent support with full metadata preservation.

Delegate tasks to specialist agents while preserving Response metadata (tokens, duration, delegation chain):

```python
from cogent import Agent

# Create specialist agents
data_analyst = Agent(
    name="data_analyst",
    model="gpt-4o-mini",
    instructions="Analyze data and provide statistical insights.",
)

market_researcher = Agent(
    name="market_researcher",
    model="gpt-4o-mini",
    instructions="Research market trends and competitive landscape.",
)

# Create coordinator with subagents
coordinator = Agent(
    name="coordinator",
    model="gpt-4o-mini",
    instructions="""Coordinate research tasks:
- Use data_analyst for numerical analysis
- Use market_researcher for market trends
Synthesize their findings.""",
    # Simply pass the agents - uses their names automatically
    subagents=[data_analyst, market_researcher],
)

# Coordinator delegates automatically
response = await coordinator.run("Analyze Q4 2025 e-commerce growth")

# Full metadata preserved
print(f"Total tokens: {response.metadata.tokens.total_tokens}")  # Includes all subagents
print(f"Subagent calls: {len(response.subagent_responses)}")
for sub_resp in response.subagent_responses:
    print(f"  {sub_resp.metadata.agent}: {sub_resp.metadata.tokens.total_tokens} tokens")
```

**Key Benefits:**
- ✅ Accurate token counting (coordinator + all subagents)
- ✅ Full delegation chain tracking
- ✅ Context propagates automatically
- ✅ Observable with `[subagent-call]`, `[subagent-result]` events
- ✅ Zero LLM behavior changes (uses existing tool calling)

**Migration from `agent.as_tool()`:**

```python
# ❌ Old: Loses metadata
coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    tools=[specialist.as_tool()],  # Response → string
)

# ✅ New: Preserves metadata (list syntax)
coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    subagents=[specialist],  # Uses specialist.name as tool name
)

# ✅ New: Preserves metadata (dict syntax for custom names)
coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    subagents={"custom_name": specialist},  # Override tool name
)
```

See [docs/subagents.md](subagents.md) for complete documentation.

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

observer = Observer(level="debug")
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
| `run(task, context)` | Execute a task with optional context |
| `chat(message, thread_id)` | Chat with memory support |
| `think(prompt)` | Single reasoning step |
| `stream_chat(message)` | Streaming chat response |
| `resume(state, decision)` | Resume after HITL interrupt |
| `as_tool(isolate_context)` | Convert agent to callable tool |

### Agent-as-Tool

Use `agent.as_tool()` to delegate work to specialized agents:

```python
from cogent import Agent, RunContext, tool

# Specialist agent with domain expertise
specialist = Agent(
    name="permission_checker",
    model="gpt4",
    instructions="Check user permissions and return clear authorization status.",
    tools=[check_db, verify_role],
)

# Orchestrator delegates to specialist
orchestrator = Agent(
    name="orchestrator",
    model="gpt4",
    tools=[specialist.as_tool()],  # Context flows automatically
    instructions="Coordinate tasks using available specialist agents.",
)

# Run with context - flows through entire delegation chain
result = await orchestrator.run(
    "Can user 123 delete files?",
    context=UserContext(user_id="123", permissions={"read"}),
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `isolate_context` | `bool` | `False` | If `True`, don't pass RunContext to delegated agent |
| `name` | `str \| None` | `None` | Override tool name (default: agent name) |
| `description` | `str \| None` | `None` | Override tool description (auto-generated if None) |

**Context Propagation:**

By default, `RunContext` flows to delegated agents (like regular tools):

```python
# Context flows automatically (default)
specialist_tool = specialist.as_tool()

# Explicit isolation - creates fresh context boundary
isolated_tool = specialist.as_tool(isolate_context=True)
```

See [docs/context.md](context.md) for context patterns.

### Model-Specific Configuration

Pass model-specific parameters via `model_kwargs`:

```python
from cogent import Agent

# Gemini with thinking enabled
agent = Agent(
    name="Thinker",
    model="gemini-2.5-flash",
    model_kwargs={"thinking_budget": 16384},  # Enable native thinking
)

# OpenAI with specific settings
agent = Agent(
    name="Assistant",
    model="gpt-4o",
    model_kwargs={"seed": 42, "logprobs": True},
)

# Any model-specific parameter
agent = Agent(
    name="Custom",
    model="anthropic:claude-sonnet-4",
    model_kwargs={"top_k": 10},
)
```

**Note:** `model_kwargs` only applies when using string model names. Ignored when passing `ChatModel` instances (configure the instance directly instead).

### AgentConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Agent name |
| `role` | `AgentRole` | Agent role |
| `model` | `str \| BaseChatModel` | Chat model (string or instance) |
| `model_kwargs` | `dict \| None` | Model-specific parameters (for string models) |
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
