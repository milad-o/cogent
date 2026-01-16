# Flow - Event-Driven Orchestration

The `agenticflow.flow` module provides event-driven orchestration for building multi-agent systems. Everything in Flow is events — tasks become events, agent outputs become events, and reactors handle events to create complex workflows.

## Overview

Flow uses an event-driven paradigm where **reactors** (agents, functions, aggregators, routers) respond to events:

```python
from agenticflow import Agent, Flow

model = get_model()

# Create agents
researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)

# Event-driven flow
flow = Flow()
flow.register(researcher, on="task.created", emits="research.done")
flow.register(writer, on="research.done", emits="flow.done")

result = await flow.run("Write about quantum computing")
print(result.output)
```

### Using Patterns

Common workflows can be created with built-in patterns:

```python
from agenticflow.flow import pipeline, supervisor, mesh

# Sequential processing
flow = pipeline([researcher, writer, editor])

# Coordinator with workers
flow = supervisor(coordinator, workers=[analyst, developer, tester])

# Collaborative rounds
flow = mesh([expert1, expert2, expert3], max_rounds=3)

result = await flow.run("Create a blog post")
```

---

## Core Concepts

### Everything is Events

In the Flow model:
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

See **[Reactors Documentation](reactors.md)** for complete reactor reference.

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

---

## Quick Start

### Basic Event-Driven Flow

```python
from agenticflow import Agent, Flow
from agenticflow.events import Event
from agenticflow.reactors import Aggregator

model = get_model()

# Create agents
classifier = Agent(name="classifier", model=model)
analyst = Agent(name="analyst", model=model)

# Create flow
flow = Flow()

# Register agents to respond to events
flow.register(classifier, on="task.created", emits="classification.done")
flow.register(analyst, on="classification.done", emits="flow.done")

# Run
result = await flow.run("Classify and analyze this ticket")
print(result.output)
```

### Using Function Reactors

```python
# Register a function reactor
flow.register(
    lambda e: Event(type="processed", source="fn", data={"v": e.data["v"] * 2}),
    on="data.received",
)

result = await flow.run(data={"v": 21}, initial_event="data.received")
```

### Using Built-in Reactors

```python
from agenticflow.reactors import Aggregator, Router, Transform

flow = Flow()

# Fan-out to parallel workers
flow.register(worker1, on="task.created")
flow.register(worker2, on="task.created")
flow.register(worker3, on="task.created")

# Aggregate results
flow.register(
    Aggregator(collect=3, emit="all.done"),
    on="worker.done",
)

# Route based on data
flow.register(
    Router({
        "urgent": "priority.high",
        "normal": "priority.normal"
    }, key="priority"),
    on="task.classified",
)

result = await flow.run("Process task")
```

### With Middleware

```python
from agenticflow.middleware import LoggingMiddleware, TimeoutMiddleware, RetryMiddleware

flow = Flow()
flow.use(LoggingMiddleware(level="DEBUG"))
flow.use(TimeoutMiddleware(timeout=30))
flow.use(RetryMiddleware(max_retries=3))

# Register reactors...

result = await flow.run("task")
```

---

## Agent Registration

### Simple Syntax

```python
# Basic event subscription
flow.register(agent, on="order.placed")

# Event patterns with wildcards
flow.register(monitor, on="task.*")

# Multiple events
flow.register(monitor, on=["order.placed", "order.shipped", "order.delivered"])

# With condition filter
from agenticflow.events import has_data

flow.register(
    agent,
    on="order.placed",
    when=has_data("value", 100),  # value == 100
)

# With priority (higher executes first)
flow.register(urgent_handler, on="alert.*", priority=10)

# Auto-emit event after completion
flow.register(processor, on="order.placed", emits="order.processed")

# Combined features
flow.register(
    agent,
    on="order.placed",
    when=has_data("urgent", True),
    priority=10,
    emits="order.processed",
)

# A2A: Agent handles requests for itself (see A2A docs)
flow.register(specialist, handles=True)
```

### Registration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `on` | `str \| list[str]` | Required | Event type(s) to react to. Supports wildcards like `"task.*"` |
| `handles` | `str \| bool` | `None` | For A2A - if `True`, uses `agent.name`; if string, uses that name |
| `when` | `Callable[[Event], bool]` | `None` | Condition function - only trigger if returns `True` |
| `priority` | `int` | `0` | Higher priority triggers execute first |
| `emits` | `str` | `None` | Event to emit after agent completes |

---

## Event Pattern Matching

Register reactors with wildcard patterns:

```python
# Exact match
flow.register(handler, on="task.created")

# Wildcard - suffix
flow.register(handler, on="task.*")      # task.created, task.updated, etc.

# Wildcard - prefix
flow.register(handler, on="*.done")       # agent.done, worker.done, etc.

# Wildcard - middle
flow.register(handler, on="worker.*.done") # worker.1.done, worker.2.done, etc.

# Multiple patterns
flow.register(handler, on=["task.created", "task.updated", "task.deleted"])

# With condition
from agenticflow.events import has_data

flow.register(
    handler,
    on="task.*",
    when=has_data("priority", "urgent"),
)
```

---

## Conditional Registration

```python
from agenticflow.events import has_data, from_source, all_of, any_of, after

# Register with condition
flow.register(
    urgent_handler,
    on="task.created",
    when=has_data("priority", "high"),
)

# Register with priority (higher executes first)
flow.register(handler1, on="task", priority=10)
flow.register(handler2, on="task", priority=0)

# Combine conditions with AND
flow.register(
    handler,
    on="agent.done",
    when=all_of(
        from_source("researcher"),
        has_data("success", True),
    ),
)

# Combine conditions with OR
flow.register(
    handler,
    on="agent.done",
    when=any_of(
        from_source("researcher"),
        from_source("analyst"),
    ),
)

# Sequential dependency - run after specific source
flow.register(
    writer,
    on="agent.done",
    when=after("researcher"),  # Only after researcher completes
)
```

### Available Condition Helpers

| Helper | Description | Example |
|--------|-------------|----------|
| `has_data(key)` | Check if key exists | `has_data("priority")` |
| `has_data(key, value)` | Check key equals value | `has_data("priority", "high")` |
| `from_source(source)` | Match event source | `from_source("researcher")` |
| `after(source)` | Alias for from_source | `after("researcher")` |
| `all_of(*conditions)` | AND logic | `all_of(cond1, cond2)` |
| `any_of(*conditions)` | OR logic | `any_of(cond1, cond2)` |
| `not_(condition)` | Negate condition | `not_(from_source("system"))` |

### Custom Conditions

For complex logic, use a lambda or define a reusable function:

```python
# Lambda for one-off complex condition
flow.register(
    handler,
    on="data.received",
    when=lambda e: e.data.get("value", 0) > 100 and e.data.get("verified", False),
)

# Reusable condition function
def high_value_and_verified(event: Event) -> bool:
    return event.data.get("value", 0) > 100 and event.data.get("verified", False)

flow.register(handler, on="data.received", when=high_value_and_verified)

# Combine custom with helpers
flow.register(
    handler,
    on="data.received",
    when=all_of(
        from_source("payment_processor"),
        lambda e: e.data.get("amount", 0) > 1000,
    ),
)
```

---

## Built-in Patterns

### Pipeline

Sequential processing through stages:

```
Task → Stage1 → Stage2 → Stage3 → Done
```

```python
from agenticflow.flow import pipeline

flow = pipeline([
    researcher,  # Stage 1: Research
    writer,      # Stage 2: Write
    editor,      # Stage 3: Edit
])

result = await flow.run("Create a blog post")
```

### Supervisor

One coordinator delegates to workers:

```
        ┌→ Worker A ─┐
Task → Supervisor ──┼→ Worker B ─┼→ Result
        └→ Worker C ─┘
```

```python
from agenticflow.flow import supervisor

flow = supervisor(
    coordinator=manager,
    workers=[analyst, writer, reviewer],
)

result = await flow.run("Build a login feature")
```

### Mesh

All agents collaborate through rounds:

```
    A ←→ B
    ↕   ↕  × N rounds
    C ←→ D
```

```python
from agenticflow.flow import mesh

flow = mesh(
    [economist, sociologist, technologist],
    max_rounds=3,
)

result = await flow.run("Evaluate this product proposal")
```

### Chain (Sequential)

Sequential processing:

```python
from agenticflow.flow import chain

# Same as pipeline, alternative name
flow = chain([agent1, agent2, agent3])
```

### Fanout (Parallel)

Parallel execution:

```python
from agenticflow.flow import fanout

flow = fanout([analyst1, analyst2, analyst3])

# All analysts run in parallel, results aggregated
result = await flow.run("Analyze this dataset")
```

### Route (Conditional)

Conditional routing:

```python
from agenticflow.flow import route

flow = route(
    lambda e: e.data.get("type"),
    {"bug": bug_handler, "feature": feature_handler},
    default=general_handler,
)

result = await flow.run("Handle this issue")
```

---

## Flow Configuration

```python
from agenticflow.flow import FlowConfig

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

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_rounds` | `int` | `100` | Maximum event processing rounds |
| `max_concurrent` | `int` | `10` | Max parallel reactor executions |
| `event_timeout` | `float` | `30.0` | Event wait timeout in seconds |
| `enable_history` | `bool` | `True` | Record event history |
| `stop_on_idle` | `bool` | `True` | Stop when no pending events |
| `stop_events` | `frozenset` | `{"flow.done", "flow.error"}` | Events that terminate flow |
| `error_policy` | `str` | `"fail_fast"` | Error handling: `fail_fast`, `continue`, `retry` |
| `checkpoint_every` | `int` | `0` | Checkpoint every N rounds (0 = disabled) |

---

## Running Flows

### Basic Execution

```python
result = await flow.run("Your task description")

print(result.output)       # Final output
print(result.success)      # True if successful
print(result.duration_ms)  # Execution time
```

### With Initial Event

```python
result = await flow.run(
    "Process this data",
    initial_event="data.received",
    data={"value": 42},
)
```

### With Context

```python
result = await flow.run(
    "Execute task",
    context={"user_id": "123", "session": "abc"},
)
```

### Streaming Events

```python
async for event in flow.stream("Process this"):
    print(f"Event: {event.type}")
    
    if event.type == "agent.chunk":
        print(event.data["content"], end="", flush=True)
```

---

## Agent-to-Agent (A2A) Communication

Agents can delegate tasks to each other within Flow:

```python
coordinator = Agent(name="coordinator", model=model)
data_analyst = Agent(name="data_analyst", model=model)
report_writer = Agent(name="report_writer", model=model)

flow = Flow()
flow.register(coordinator, on="task.created")
flow.register(data_analyst, handles=True)  # Handles agent.request for "data_analyst"
flow.register(report_writer, handles=True)

# Coordinator can delegate to specialists
result = await flow.run(
    "Analyze our sales data and create a report",
    initial_event="task.created",
)
```

**See [A2A Documentation](a2a.md) for complete guide on:**
- Agent delegation patterns
- Request/Response tracking
- ExecutionContext API
- Wait vs fire-and-forget
- Common coordination patterns

---

## Skills

Skills are event-triggered behavioral specializations that dynamically modify an agent's context, prompts, and tools based on incoming events.

### Creating Skills

```python
from agenticflow.skills import skill
from agenticflow.events import has_data

python_skill = skill(
    "python_expert",
    on="code.write",
    when=has_data("language", "python"),
    prompt="""You are a Python expert. Follow these guidelines:
    - Use type hints on all functions
    - Follow PEP 8 conventions
    - Include docstrings with Args/Returns""",
    tools=[run_python, lint_code],
    priority=10,
)

debugger_skill = skill(
    "debugger",
    on="code.debug",
    prompt="You are a debugging specialist. Analyze errors systematically.",
    tools=[trace_execution, inspect_variables],
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
flow = Flow()

# Register skills
flow.register_skill(python_skill)
flow.register_skill(debugger_skill)

# Register agents
flow.register(coder, on="code.*")

# Skills apply automatically when events match
result = await flow.run(
    "Write a fibonacci function",
    initial_event="code.write",
    data={"language": "python"},
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

---

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

# Route by condition (requires lambda for complex comparisons)
flow.register(
    ConditionalRouter([
        (lambda e: e.data.get("value", 0) > 100, "high.value"),
        (lambda e: e.data.get("value", 0) > 50, "medium.value"),
    ], default="low.value"),
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
        transform=lambda d: {"doubled": d.get("value", 0) * 2},
        emit="transformed",
    ),
    on="input",
)

# Extract specific fields
flow.register(
    Transform.extract(
        keys=["user_id", "action", "result"],
        emit="analytics.event",
    ),
    on="user.action",
)

# Rename fields
flow.register(
    Transform.rename(
        mapping={"old_key": "new_key", "value": "amount"},
        emit="normalized.event",
    ),
    on="legacy.event",
)
```

### Gateway

Bridge to external systems:

```python
from agenticflow.reactors import Gateway, HttpGateway, CallbackGateway, LogGateway

# HTTP gateway - POST event data to external API
flow.register(
    HttpGateway(
        url="https://api.example.com/webhooks",
        method="POST",
        headers={"Authorization": "Bearer token"},
    ),
    on="order.completed",
)

# Callback gateway - invoke custom function
flow.register(
    CallbackGateway(callback=send_email_notification),
    on="user.registered",
)

# Log gateway - structured logging
flow.register(
    LogGateway(level="info"),
    on="*",  # Log all events
)
```

See **[Reactors Documentation](reactors.md)** for complete reactor reference.

---

## Checkpointing (Crash Recovery)

Enable automatic checkpointing to resume flows after crashes or interruptions.

### Basic Usage

```python
from agenticflow.flow import Flow, FlowConfig
from agenticflow.checkpointing import FileCheckpointer, MemoryCheckpointer

# File-based (simple persistence)
checkpointer = FileCheckpointer("./checkpoints")

# In-memory (dev/test)
checkpointer = MemoryCheckpointer()

# Enable checkpointing every round
config = FlowConfig(checkpoint_every=1)
flow = Flow(config=config, checkpointer=checkpointer)
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
| `PostgresCheckpointer` | PostgreSQL database storage |

### Configuration

```python
config = FlowConfig(
    checkpoint_every=2,  # Checkpoint every 2 rounds (0 = disabled)
    flow_id="my-flow-001",  # Unique flow identifier
)

flow = Flow(config=config, checkpointer=checkpointer)
```

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

### Checkpointing Best Practices

1. **Unique Flow IDs**: Use unique `flow_id` for each flow instance
2. **Checkpoint Frequency**: Balance between safety and performance
   - `checkpoint_every=1`: Maximum safety, slower
   - `checkpoint_every=3`: Good balance for most flows
   - `checkpoint_every=0`: Disabled (no recovery)
3. **Storage Choice**:
   - `FileCheckpointer`: Simple, local development
   - `PostgresCheckpointer`: Production, distributed systems
   - `MemoryCheckpointer`: Testing only

### Example: Production Pipeline with Checkpointing

```python
from agenticflow import Agent, Flow
from agenticflow.flow import FlowConfig
from agenticflow.checkpointing import PostgresCheckpointer

# Production checkpointer
checkpointer = PostgresCheckpointer(
    connection_string=os.getenv("DATABASE_URL"),
)

# Long-running pipeline with crash recovery
flow = Flow()
flow.register(extractor, on="task.created", emits="extracted")
flow.register(transformer, on="extracted", emits="transformed")
flow.register(validator, on="transformed", emits="validated")
flow.register(loader, on="validated", emits="flow.done")

flow.config = FlowConfig(
    checkpoint_every=1,  # Save after each stage
    flow_id=f"data-pipeline-{datetime.now().isoformat()}",
)

flow.checkpointer = checkpointer

try:
    result = await flow.run("Process daily data batch")
except Exception as e:
    # On crash, can resume later
    logger.error(f"Flow crashed: {e}")
    # Resume: flow.resume(checkpoint_id=latest_checkpoint)
```

---

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

---

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

---

## FlowResult

The result returned from flow execution:

```python
result = await flow.run("Task")

# Core properties
result.output          # Final output string
result.success         # True if completed successfully
result.duration_ms     # Total execution time
result.iterations      # Number of iterations/rounds

# Event history
result.events          # List of all events processed

# Metadata
result.metadata        # Additional execution metadata
result.trace_id        # Trace ID for debugging
```

---

## Error Handling

```python
try:
    result = await flow.run("Task")
    if not result.success:
        print(f"Flow failed: {result.error}")
except Exception as e:
    print(f"Flow error: {e}")
```

### Error Policies

```python
# Stop on first error (default)
flow = Flow(config=FlowConfig(error_policy="fail_fast"))

# Continue despite errors
flow = Flow(config=FlowConfig(error_policy="continue"))

# Retry failed reactors
flow = Flow(config=FlowConfig(error_policy="retry"))
```

---

## Observability

### With Observer

```python
from agenticflow.observability import Observer

observer = Observer.debug()

flow = Flow(observer=observer)

result = await flow.run("Execute task")

# Access trace
for trace in observer.traces:
    print(f"{trace.type}: {trace.data}")
```

### Event Bus Integration

```python
from agenticflow.observability import EventBus, ConsoleEventHandler

# Custom event bus
bus = EventBus()
bus.subscribe_all(ConsoleEventHandler())

flow = Flow(event_bus=bus)

# Events will be published to your bus
result = await flow.run("Task")

# Query event history
events = bus.get_history(limit=100)
```

---

## Advanced Features

### Custom Reactors

Create custom reactors by implementing the Reactor protocol:

```python
from agenticflow.reactors import BaseReactor
from agenticflow.events import Event
from agenticflow.flow.context import Context

class EmailReactor(BaseReactor):
    def __init__(self, smtp_config: dict):
        super().__init__(name="email_sender")
        self.smtp = setup_smtp(smtp_config)
    
    async def handle(self, event: Event, ctx: Context) -> Event | None:
        recipient = event.data.get("recipient")
        message = event.data.get("message")
        
        await self.smtp.send(recipient, message)
        
        return Event(
            name="email.sent",
            source=self.name,
            data={"recipient": recipient},
            correlation_id=event.correlation_id,
        )

# Register
flow.register(
    EmailReactor(smtp_config),
    on="notification.email",
)
```

### Manual Event Emission

```python
# Emit events manually during flow execution
await flow.emit("custom.event", data={"key": "value"})
```

### Memory Integration

```python
from agenticflow.memory import Memory

memory = Memory()

result = await flow.run(
    "Create a report",
    memory=memory,
)

# Access shared state after execution
findings = await memory.recall("findings")
```

---

## Examples

### Example: Support Ticket System

```python
from agenticflow import Agent, Flow
from agenticflow.skills import skill
from agenticflow.events import has_data

# Skills for specialized handling
urgent_skill = skill(
    "urgent_handler",
    on="ticket.created",
    when=has_data("priority", "urgent"),
    prompt="This is an URGENT ticket. Prioritize speed and escalation.",
    priority=20,
)

# Agents
classifier = Agent(name="classifier", model=model)
responder = Agent(name="responder", model=model)
escalator = Agent(name="escalator", model=model)

# Flow setup
flow = Flow()
flow.register_skill(urgent_skill)

# Register agents
flow.register(classifier, on="ticket.created", emits="ticket.classified")
flow.register(
    responder,
    on="ticket.classified",
    when=has_data("escalate", False),
    emits="flow.done",
)
flow.register(
    escalator,
    on="ticket.classified",
    when=has_data("escalate", True),
    emits="flow.done",
)

# Run
result = await flow.run(
    "Process this customer ticket",
    initial_event="ticket.created",
    data={"ticket_id": "T-456", "priority": "urgent"},
)
```

### Example: Conditional Workflow

```python
from agenticflow import Agent, Flow
from agenticflow.reactors import Router

# Agents
quick_responder = Agent(
    name="quick_responder",
    model=model,
    system_prompt="Give quick, direct answers.",
)

deep_researcher = Agent(
    name="deep_researcher",
    model=model,
    system_prompt="Do thorough research.",
)

# Flow
flow = Flow()

# Route based on query complexity
flow.register(
    Router(
        routes={"simple": "route.simple", "complex": "route.complex"},
        selector=lambda e: "simple" if len(e.data["task"]) < 50 else "complex",
    ),
    on="task.created",
)

flow.register(quick_responder, on="route.simple", emits="flow.done")
flow.register(deep_researcher, on="route.complex", emits="flow.done")

result = await flow.run("What's 2+2?")
```

### Example: Parallel Processing

```python
from agenticflow import Agent, Flow
from agenticflow.reactors import Aggregator

# Workers
analyst1 = Agent(name="analyst1", model=model)
analyst2 = Agent(name="analyst2", model=model)
analyst3 = Agent(name="analyst3", model=model)

# Synthesizer
synthesizer = Agent(name="synthesizer", model=model)

# Flow
flow = Flow()

# Fan-out to parallel workers
flow.register(analyst1, on="task.created", emits="analyst.done")
flow.register(analyst2, on="task.created", emits="analyst.done")
flow.register(analyst3, on="task.created", emits="analyst.done")

# Aggregate results
flow.register(
    Aggregator(collect=3, emit="analysis.complete"),
    on="analyst.done",
)

# Synthesize final output
flow.register(synthesizer, on="analysis.complete", emits="flow.done")

result = await flow.run("Analyze this dataset")
```

---

## Best Practices

1. **Use patterns for common scenarios** - `pipeline()`, `supervisor()`, `mesh()`
2. **Add middleware early** - Logging and tracing help with debugging
3. **Name your reactors** - Makes logs and traces more readable
4. **Use stop events** - Define clear termination conditions (`flow.done`)
5. **Handle errors** - Use `error_policy` or middleware for resilience
6. **Stream for UX** - Use `flow.stream()` for real-time feedback
7. **Checkpoint long flows** - Enable crash recovery for production
8. **Use skills** - Dynamic behavior based on event context
9. **Leverage A2A** - Let agents delegate to specialists

---

## API Reference

### Flow Class

```python
Flow(
    config: FlowConfig | None = None,
    event_bus: EventBus | None = None,
    observer: Observer | None = None,
    checkpointer: Checkpointer | None = None,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `register(reactor, on, when, priority, emits)` | Register reactor with event pattern |
| `register_skill(skill)` | Register a skill |
| `unregister_skill(name)` | Remove a skill |
| `use(middleware)` | Add middleware |
| `run(task, initial_event, data, context)` | Execute flow |
| `stream(task, initial_event, data, context)` | Execute with streaming |
| `resume(state, context)` | Resume from checkpoint |
| `emit(event_name, data)` | Manually emit an event |

### FlowConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_rounds` | `int` | `100` | Max event processing rounds |
| `max_concurrent` | `int` | `10` | Max parallel executions |
| `event_timeout` | `float` | `30.0` | Event wait timeout |
| `enable_history` | `bool` | `True` | Record events |
| `stop_on_idle` | `bool` | `True` | Stop when idle |
| `stop_events` | `frozenset` | `{"flow.done", "flow.error"}` | Termination events |
| `error_policy` | `str` | `"fail_fast"` | Error handling |
| `checkpoint_every` | `int` | `0` | Checkpoint frequency |

### FlowResult

| Property | Type | Description |
|----------|------|-------------|
| `output` | `str` | Final output |
| `success` | `bool` | Whether flow succeeded |
| `duration_ms` | `float` | Execution time |
| `iterations` | `int` | Number of rounds |
| `events` | `list[Event]` | All events processed |
| `metadata` | `dict` | Additional metadata |
| `trace_id` | `str` | Trace identifier |

---

## Related Documentation

- **[Reactors](reactors.md)** — Complete reactor reference
- **[Events](events.md)** — Event system and patterns
- **[A2A](a2a.md)** — Agent-to-agent delegation
- **[Observability](observability.md)** — Monitoring and tracing

---

## Migration Guide

### From Legacy Topology-based API

The old topology-based API is still supported for backward compatibility:

```python
# Old (Topology-based) - still works
from agenticflow import Flow as LegacyFlow
flow = LegacyFlow(
    name="team",
    agents=[a1, a2, a3],
    topology="pipeline",
)
result = await flow.run("task")

# New (Event-driven) - recommended
from agenticflow.flow import pipeline
flow = pipeline([a1, a2, a3])
result = await flow.run("task")
```

### Topology Equivalents

| Old Topology | New Pattern |
|-------------|-------------|
| `topology="pipeline"` | `pipeline([a1, a2, a3])` |
| `topology="supervisor"` | `supervisor(coord, workers=[w1, w2])` |
| `topology="mesh"` | `mesh([a1, a2, a3], max_rounds=3)` |
| `topology="hierarchical"` | Custom event-driven flow |

### Event-Driven Benefits

The event-driven model provides:
- **Flexibility** — Mix patterns, add reactors dynamically
- **Composability** — Combine agents, functions, aggregators, routers
- **Extensibility** — Custom reactors, middleware, skills
- **Observability** — Every action is an event
- **Resilience** — Checkpointing, error policies, retries
