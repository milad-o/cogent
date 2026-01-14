# Reactive Module

The `agenticflow.reactive` module provides event-driven orchestration for agents that react to events in real-time.

## Overview

ReactiveFlow enables agents to respond to events as they happen, forming processing chains where one agent's reaction can trigger others:

```python
from agenticflow import Agent
from agenticflow.reactive import ReactiveFlow, react_to, skill

model = get_model()

# Agents react to specific events
classifier = Agent(name="classifier", model=model)
analyst = Agent(name="analyst", model=model)

flow = ReactiveFlow()
flow.register(classifier, [react_to("ticket.created")])
flow.register(analyst, [react_to("classification.done")])

result = await flow.run(
    "Classify and analyze this ticket",
    initial_event="ticket.created",
    initial_data={"ticket_id": "T-123"},
)
```

---

## Triggers

Agents register with triggers that define when they react:

```python
from agenticflow.reactive import react_to

# Simple event name
flow.register(agent, [react_to("order.placed")])

# With condition
flow.register(agent, [
    react_to("order.placed").when(lambda e: e.data.get("value") > 100)
])

# Emit new event after reaction
flow.register(agent, [
    react_to("order.placed").emits("order.processed")
])

# Priority (higher runs first)
flow.register(agent, [
    react_to("order.placed").with_priority(10)
])
```

---

## Skills

Skills are event-triggered behavioral specializations that dynamically modify an agent's context, prompts, and tools based on incoming events.

### Creating Skills

```python
from agenticflow.reactive import skill

python_skill = skill(
    "python_expert",
    on="code.write",
    when=lambda e: e.data.get("language") == "python",
    prompt="""You are a Python expert. Follow these guidelines:
    - Use type hints on all functions
    - Follow PEP 8 conventions
    - Include docstrings with Args/Returns""",
    tools=[run_python, lint_code],
    priority=10,
)
```

### Skill Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique skill identifier |
| `on` | `str \| EventPattern` | Event pattern to match |
| `when` | `Callable[[Event], bool]` | Optional condition filter |
| `prompt` | `str` | Prompt injected into agent context |
| `tools` | `list[Callable]` | Tools temporarily added to agent |
| `context_enricher` | `Callable[[Event, dict], dict]` | Enrich context before execution |
| `priority` | `int` | Higher priority skills apply first |

### Registering Skills

```python
flow = ReactiveFlow()

# Register skills
flow.register_skill(python_skill)
flow.register_skill(debugger_skill)

# Skills apply automatically when events match
result = await flow.run(
    "Write a fibonacci function",
    initial_event="code.write",
    initial_data={"language": "python"},
)
# Agent runs with python_expert prompt and tools injected
```

### Context Enrichers

Add dynamic context based on the triggering event:

```python
def enrich_with_history(event, context):
    ticket_id = event.data.get("ticket_id")
    context["history"] = fetch_ticket_history(ticket_id)
    return context

support_skill = skill(
    "support",
    on="ticket.*",
    prompt="You are a support specialist.",
    context_enricher=enrich_with_history,
)
```

### Skill Lifecycle Events

Skills emit observability events:

```
[coder] [triggered] by code.write
[coder] [skill: python_expert] ← code.write   # Skill activated
[coder] [starting]
[coder] [thinking]
[coder] [completed] (3.2s)
```

---

## Orchestration Patterns

### Chain

Sequential processing:

```python
from agenticflow.reactive import chain

pipeline = chain(agent1, agent2, agent3)
flow.register_pattern(pipeline)
```

### Fanout

Parallel execution:

```python
from agenticflow.reactive import fanout

parallel = fanout(analyst1, analyst2, analyst3)
flow.register_pattern(parallel)
```

### Route

Conditional routing:

```python
from agenticflow.reactive import route

router = route(
    lambda e: e.data.get("type"),
    {"bug": bug_handler, "feature": feature_handler},
    default=general_handler,
)
```

---

## Checkpointing

Enable persistent state for long-running flows with crash recovery support.

### Basic Usage

```python
from agenticflow.reactive import (
    ReactiveFlow, ReactiveFlowConfig,
    MemoryCheckpointer, FileCheckpointer
)

# In-memory (dev/test)
checkpointer = MemoryCheckpointer()

# File-based (simple persistence)
checkpointer = FileCheckpointer("./checkpoints")

# Enable checkpointing every round
config = ReactiveFlowConfig(checkpoint_every=1)
flow = ReactiveFlow(config=config, checkpointer=checkpointer)
```

### Resume from Checkpoint

```python
# After crash, resume from last checkpoint
state = await checkpointer.load_latest("flow_abc123")
if state:
    result = await flow.resume(state)
```

### Checkpointer Implementations

| Class | Description |
|-------|-------------|
| `MemoryCheckpointer` | In-memory storage (lost on restart) |
| `FileCheckpointer` | JSON files in a directory |

### Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `flow_id` | `str \| None` | Auto-generated | Fixed flow ID |
| `checkpoint_every` | `int` | `0` | Checkpoint every N rounds (0 = disabled) |

### FlowState Contents

```python
@dataclass
class FlowState:
    flow_id: str
    checkpoint_id: str
    task: str
    events_processed: int
    pending_events: list[dict]
    context: dict
    last_output: str
    round: int
    timestamp: datetime
```

### Checkpointing vs Memory

It is important to distinguish between **Agent Memory** and **Flow Checkpointing**:

| Feature | Memory | Checkpointing |
|---------|--------|---------------|
| **Scope** | Agent-level | Flow-level |
| **Storage** | Conversation history | Execution state (events, rounds) |
| **Purpose** | Context retention | Crash recovery & persistence |
| **Persistence** | Across runs | Single (resumable) run |

**Recommendation**: Use both together. Use `Memory` for agents to remember user details, and `Checkpointer` to ensure the overall workflow can handle restarts.

---

## ReactiveFlow API


### Constructor

```python
ReactiveFlow(
    observer: Observer | None = None,  # Observability
    max_rounds: int = 10,              # Maximum processing rounds
)
```

### Methods

| Method | Description |
|--------|-------------|
| `register(agent, triggers)` | Register agent with triggers |
| `register_skill(skill)` | Register a skill |
| `unregister_skill(name)` | Remove a skill |
| `run(task, initial_event, initial_data)` | Execute reactive flow |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `skills` | `list[Skill]` | Registered skills |
| `agents` | `list[Agent]` | Registered agents |

---

## Example: Support Ticket System

```python
from agenticflow import Agent
from agenticflow.reactive import ReactiveFlow, react_to, skill

# Skills for specialized handling
urgent_skill = skill(
    "urgent_handler",
    on="ticket.created",
    when=lambda e: e.data.get("priority") == "urgent",
    prompt="This is an URGENT ticket. Prioritize speed and escalation.",
    priority=20,
)

# Agents
classifier = Agent(name="classifier", model=model, instructions="Classify tickets")
responder = Agent(name="responder", model=model, instructions="Draft responses")
escalator = Agent(name="escalator", model=model, instructions="Handle escalations")

# Flow setup
flow = ReactiveFlow()
flow.register_skill(urgent_skill)
flow.register(classifier, [react_to("ticket.created").emits("ticket.classified")])
flow.register(responder, [react_to("ticket.classified").when(lambda e: not e.data.get("escalate"))])
flow.register(escalator, [react_to("ticket.classified").when(lambda e: e.data.get("escalate"))])

# Run
result = await flow.run(
    "Process this customer ticket",
    initial_event="ticket.created",
    initial_data={"ticket_id": "T-456", "priority": "urgent"},
)
```
