# Unified Flow - Event-Driven Orchestration

The **unified Flow** is AgenticFlow's modern orchestration engine for building event-driven multi-agent systems. It provides a single, coherent event-driven model that replaces the previous dual-paradigm approach.

> **Note**: This is the recommended API for new projects. The legacy `Flow` (topology-based) and `ReactiveFlow` remain available for backward compatibility.

## Core Concepts

### Everything is Events

In the unified Flow model:
- **Tasks** become events (`task.created`)
- **Agent outputs** become events (`agent.done`)
- **Errors** become events (`flow.error`)
- **State changes** become events

### Reactors Handle Events

**Reactors** are the universal abstraction for event handlers:

| Reactor Type | Purpose |
|-------------|---------|
| `AgentReactor` | Wraps an Agent for event-driven execution |
| `FunctionReactor` | Wraps a plain function |
| `Aggregator` | Collects multiple events (fan-in) |
| `Router` | Routes events based on conditions |
| `Transform` | Transforms event data |
| `Gateway` | Bridges external systems |

### Patterns Configure Flows

Common orchestration patterns are helper functions that return pre-configured Flow instances:

```python
from agenticflow.flow import pipeline, supervisor, mesh

# Sequential processing
flow = pipeline([agent1, agent2, agent3])

# Coordinator with workers
flow = supervisor(coordinator, workers=[w1, w2, w3])

# Collaborative discussion
flow = mesh([expert1, expert2, expert3], max_rounds=3)
```

## Quick Start

### Basic Flow

```python
from agenticflow import Agent, Flow, FlowConfig
from agenticflow.flow.core import Flow  # Use core.Flow for new API

# Create agents
researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)

# Create flow and wire events
flow = Flow()
flow.register(researcher, on="task.created", emits="research.done")
flow.register(writer, on="research.done", emits="flow.done")

# Run
result = await flow.run("Write about quantum computing")
print(result.output)
```

### Using Patterns

```python
from agenticflow.flow.patterns import pipeline

# Create pipeline: A → B → C
flow = pipeline([researcher, writer, editor])

result = await flow.run("Create a blog post")
```

### Custom Reactors

```python
from agenticflow.flow.core import Flow
from agenticflow.events import Event
from agenticflow.reactors import Aggregator

flow = Flow()

# Register a function reactor
flow.register(
    lambda e: Event(type="processed", source="fn", data={"v": e.data["v"] * 2}),
    on="data.received",
)

# Register an aggregator
flow.register(
    Aggregator(collect=3, emit="all.done"),
    on="worker.*.done",
)

result = await flow.run(data={"v": 21}, initial_event="data.received")
```

### With Middleware

```python
from agenticflow.flow.core import Flow
from agenticflow.middleware import LoggingMiddleware, TimeoutMiddleware, RetryMiddleware

flow = Flow()
flow.use(LoggingMiddleware(level="DEBUG"))
flow.use(TimeoutMiddleware(timeout=30))
flow.use(RetryMiddleware(max_retries=3))

# Register reactors...

result = await flow.run("task")
```

## Flow Configuration

```python
from agenticflow.flow.config import FlowConfig

config = FlowConfig(
    max_rounds=100,           # Max event processing rounds
    max_concurrent=10,        # Max parallel reactor executions
    event_timeout=30.0,       # Timeout for waiting on events
    enable_history=True,      # Record all events
    stop_on_idle=True,        # Stop when no more events
    stop_events=frozenset({"flow.done", "flow.error"}),
    error_policy="fail_fast", # "fail_fast", "continue", or "retry"
)

flow = Flow(config=config)
```

## Event Pattern Matching

Register reactors with wildcard patterns:

```python
# Exact match
flow.register(handler, on="task.created")

# Wildcard match
flow.register(handler, on="task.*")      # task.created, task.updated, etc.
flow.register(handler, on="*.done")       # agent.done, worker.done, etc.
flow.register(handler, on="worker.*.done") # worker.1.done, worker.2.done, etc.
```

## Conditional Registration

```python
# Register with condition
flow.register(
    urgent_handler,
    on="task.created",
    when=lambda e: e.data.get("priority") == "high",
)

# Register with priority (higher executes first)
flow.register(handler1, on="task", priority=10)
flow.register(handler2, on="task", priority=0)
```

## Streaming Events

```python
async for event in flow.stream("Process this"):
    print(f"Event: {event.type}")
    
    if event.type == "agent.chunk":
        print(event.data["content"], end="", flush=True)
```

## Built-in Patterns

### Pipeline

Sequential processing through stages:

```
Task → Stage1 → Stage2 → Stage3 → Done
```

```python
from agenticflow.flow.patterns import pipeline

flow = pipeline([
    researcher,  # Stage 1: Research
    writer,      # Stage 2: Write
    editor,      # Stage 3: Edit
])
```

### Supervisor

One coordinator delegates to workers:

```
        ┌→ Worker A ─┐
Task → Supervisor ──┼→ Worker B ─┼→ Result
        └→ Worker C ─┘
```

```python
from agenticflow.flow.patterns import supervisor

flow = supervisor(
    coordinator=manager,
    workers=[analyst, writer, reviewer],
)
```

### Mesh

All agents collaborate through rounds:

```
    A ←→ B
    ↕   ↕  × N rounds
    C ←→ D
```

```python
from agenticflow.flow.patterns import mesh

flow = mesh(
    [economist, sociologist, technologist],
    max_rounds=3,
)
```

## Built-in Reactors

### Aggregator

Collect multiple events before continuing:

```python
from agenticflow.reactors import Aggregator, FirstWins, WaitAll

# Wait for 3 events
flow.register(
    Aggregator(collect=3, emit="all.done"),
    on="worker.done",
)

# First event wins
flow.register(
    FirstWins(emit="winner"),
    on="response.*",
)

# Wait for specific count
flow.register(
    WaitAll(expected=5, emit="complete"),
    on="result.*",
)
```

### Router

Route events based on conditions:

```python
from agenticflow.reactors import Router, ConditionalRouter

# Route by key value
flow.register(
    Router(
        routes={"high": "priority.high", "low": "priority.low"},
        key="level",
        default="priority.normal",
    ),
    on="task.classified",
)

# Route by condition
flow.register(
    ConditionalRouter(
        conditions=[
            (lambda e: e.data.get("value") > 100, "high.value"),
            (lambda e: e.data.get("value") > 50, "medium.value"),
        ],
        default="low.value",
    ),
    on="data.received",
)
```

### Transform

Transform event data:

```python
from agenticflow.reactors import Transform, MapTransform

# Custom transformation
flow.register(
    Transform(
        transformer=lambda d: {"doubled": d.get("value", 0) * 2},
        emit="transformed",
    ),
    on="input",
)

# Field mapping
flow.register(
    MapTransform(
        mapping={"input_field": "output_field"},
        emit="mapped",
    ),
    on="data.raw",
)
```

## Built-in Middleware

### LoggingMiddleware

```python
from agenticflow.middleware import LoggingMiddleware, VerboseMiddleware

flow.use(LoggingMiddleware(level="INFO", include_timing=True))
flow.use(VerboseMiddleware())  # DEBUG with data
```

### RetryMiddleware

```python
from agenticflow.middleware import RetryMiddleware

flow.use(RetryMiddleware(
    max_retries=3,
    base_delay=1.0,
    exponential=True,  # Exponential backoff
    jitter=True,       # Random jitter
))
```

### TimeoutMiddleware

```python
from agenticflow.middleware import TimeoutMiddleware

flow.use(TimeoutMiddleware(
    timeout=30.0,
    per_reactor={"slow_agent": 120.0},
))
```

### TracingMiddleware

```python
from agenticflow.middleware import TracingMiddleware

tracing = TracingMiddleware(service_name="my-flow")
flow.use(tracing)

# After execution
for span in tracing.spans:
    print(f"{span.name}: {span.duration_ms:.2f}ms")
```

## Event Persistence

Store and replay events:

```python
from agenticflow.events import InMemoryEventStore, FileEventStore

# In-memory store
store = InMemoryEventStore(max_size=1000)

# File-based store (JSONL)
store = FileEventStore("events.jsonl")

# Append events
await store.append(event)

# Query events
events = await store.get_by_type("agent.done")
events = await store.get_since(timestamp)

# Replay
async for event in store.replay():
    process(event)
```

## Migration from Legacy API

### From Old Topology-based Flow

```python
# Old (Topology-based)
from agenticflow import Flow
flow = Flow(name="team", agents=[a1, a2], topology="pipeline")
result = await flow.run("task")

# New (Event-driven)
from agenticflow.flow.patterns import pipeline
flow = pipeline([a1, a2])
result = await flow.run("task")
```

### From ReactiveFlow

```python
# Old (ReactiveFlow)
from agenticflow import ReactiveFlow
from agenticflow.reactive import react_to
flow = ReactiveFlow()
flow.register(agent, [react_to("task.created")])

# New (Unified Flow)
from agenticflow.flow.core import Flow
flow = Flow()
flow.register(agent, on="task.created")
```

## Best Practices

1. **Use patterns for common scenarios** - `pipeline()`, `supervisor()`, `mesh()`
2. **Add middleware early** - Logging and tracing help with debugging
3. **Name your reactors** - Makes logs and traces more readable
4. **Use stop events** - Define clear termination conditions
5. **Handle errors** - Use `error_policy` or middleware for resilience
6. **Stream for UX** - Use `flow.stream()` for real-time feedback
