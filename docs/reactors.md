# Reactors

**Reactors** are the fundamental building blocks of event-driven flows in AgenticFlow. They handle events and optionally emit new events, enabling complex orchestration patterns through simple, composable units.

## Overview

In AgenticFlow's event-driven architecture, everything is events — tasks become events, agent outputs become events, errors become events. Reactors are the universal abstraction for handling these events.

```python
from agenticflow import Flow, Agent
from agenticflow.reactors import Aggregator, Router, Transform

flow = Flow()

# Agents are automatically wrapped as AgentReactor
flow.register(agent, on="task.created")

# Functions are automatically wrapped as FunctionReactor
flow.register(lambda e: process_data(e.data), on="data.ready")

# Built-in reactors for advanced patterns
flow.register(
    Aggregator(collect=3, emit="all.done"),
    on="worker.done",
)
```

---

## Core Concepts

### The Reactor Protocol

All reactors implement a simple protocol:

```python
class Reactor(Protocol):
    @property
    def name(self) -> str:
        """Unique name for this reactor."""
        ...
    
    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | list[Event] | None:
        """Handle an event, optionally emit result event(s)."""
        ...
```

**Key points:**
- Each reactor has a unique name for identification
- `handle()` receives an event and execution context
- Can return a single event, multiple events, or None
- Async by design for non-blocking execution

### Event Flow

```
Event → Reactor.handle() → New Event(s)
  ↓                            ↓
Pattern matching           Emit to flow
  ↓                            ↓
Condition checking         Trigger next reactors
```

---

## Built-in Reactors

### AgentReactor

Wraps an Agent for event-driven execution. Automatically extracts task data from events, runs the agent, and emits the result.

```python
from agenticflow import Agent
from agenticflow.agent.reactor import AgentReactor

agent = Agent(name="researcher", model=model)

# Auto-wrapped when registering
flow.register(agent, on="task.created")

# Or manually wrap
reactor = AgentReactor(
    agent,
    task_key="task",      # Event data key for task
    context_key="context", # Event data key for context
    emit_name="research.done",
)
```

**Configuration:**
- `task_key` — Key in event.data containing the task/prompt (default: `"task"`)
- `context_key` — Key for additional context (default: `"context"`)
- `emit_name` — Override event name (default: `f"{agent.name}.done"`)

**Emitted events:**
```python
Event(
    name="agent.done",  # or custom emit_name
    source=agent.name,
    data={
        "output": "...",     # Agent's response
        "task": "...",       # Original task
        "agent": "...",      # Agent name
    },
)
```

---

### FunctionReactor

Wraps a plain function as a reactor. Supports both sync and async functions.

```python
from agenticflow.reactors import FunctionReactor, function_reactor

# Decorator syntax
@function_reactor
def process_event(event: Event) -> str:
    return f"Processed: {event.data['value']}"

@function_reactor(name="validator", emit_name="validation.done")
async def validate(event: Event, ctx: Context) -> dict:
    result = await async_validate(event.data)
    return {"valid": result}

# Direct instantiation
reactor = FunctionReactor(
    lambda e: e.data["x"] * 2,
    name="doubler",
    emit_name="value.doubled",
)
```

**Function signatures:**
```python
# Event only
def fn(event: Event) -> Any: ...

# Event + Context
def fn(event: Event, ctx: Context) -> Any: ...

# Async variants
async def fn(event: Event) -> Any: ...
async def fn(event: Event, ctx: Context) -> Any: ...
```

**Return values:**
- Scalar values → wrapped in `{"output": value}`
- Dict → used as event data directly
- None → No event emitted

---

### Aggregator

Collects multiple events before proceeding. Implements fan-in patterns where you need to wait for parallel operations to complete.

```python
from agenticflow.reactors import Aggregator, FirstWins, WaitAll

# Wait for 3 events
flow.register(
    Aggregator(
        collect=3,
        emit="all.done",
    ),
    on="worker.done",
)

# Wait for specific workers
flow.register(
    Aggregator(
        collect=3,
        emit="analysis.complete",
        combine_fn=lambda events: {
            "results": [e.data["output"] for e in events],
            "sources": [e.source for e in events],
        },
    ),
    on="agent.done",
    when=lambda e: e.source in ["analyst1", "analyst2", "analyst3"],
)

# First event wins (race condition)
flow.register(
    FirstWins(emit="winner"),
    on="search.*.done",
)

# Wait for all with custom name
flow.register(
    WaitAll(expected=5, emit="batch.complete"),
    on="item.processed",
)
```

**Configuration:**
- `collect` — Number of events to wait for
- `emit` — Event name to emit when complete
- `mode` — Fan-in mode (WAIT_ALL, FIRST_WINS, STREAMING, QUORUM)
- `timeout` — Optional timeout in seconds
- `combine_fn` — Custom function to combine event data

**Fan-in Modes:**

| Mode | Behavior |
|------|----------|
| `WAIT_ALL` | Wait for all expected events |
| `FIRST_WINS` | Emit after first event, ignore rest |
| `STREAMING` | Process each event as it arrives |
| `QUORUM` | Wait for N of M events |

**Emitted event:**
```python
Event(
    name="all.done",
    source="aggregator_name",
    data={
        # Default: outputs by source
        "worker1": "result 1",
        "worker2": "result 2",
        # Or custom combine_fn result
    },
)
```

---

### Router

Routes events to different paths based on conditions. Examines incoming events and re-emits them with a modified name.

```python
from agenticflow.reactors import Router, ConditionalRouter

# Route by data key
flow.register(
    Router(
        routes={
            "billing": "route.billing",
            "technical": "route.technical",
            "general": "route.general",
        },
        key="category",  # Uses event.data["category"]
        default="route.unknown",
    ),
    on="ticket.classified",
)

# Route with custom selector
flow.register(
    Router(
        routes={
            "high": "priority.high",
            "low": "priority.low",
        },
        selector=lambda e: "high" if e.data.get("priority") > 5 else "low",
    ),
    on="task.created",
)

# Conditional routing with predicates
flow.register(
    ConditionalRouter([
        (lambda e: e.data.get("urgent"), "alert.urgent"),
        (lambda e: e.data.get("important"), "alert.important"),
        (lambda e: True, "alert.normal"),  # Default
    ]),
    on="alert.created",
)
```

**Configuration:**
- `routes` — Mapping from route keys to event names
- `key` — Event data key to use for routing
- `selector` — Custom function to determine route
- `default` — Fallback event name if no route matches

**Emitted events:**
```python
Event(
    name="route.billing",  # From routes mapping
    source="router",
    data={
        **original_data,
        "_routed_from": "ticket.classified",
        "_route_key": "billing",
    },
)
```

---

### Transform

Transforms event data before forwarding. Useful for data normalization, enrichment, or extraction.

```python
from agenticflow.reactors import Transform, MapTransform

# Extract fields
flow.register(
    Transform(
        transform=lambda d: {"summary": d.get("output", "")[:100]},
        emit="summary.ready",
    ),
    on="agent.done",
)

# Enrich with metadata
flow.register(
    Transform(
        transform=lambda d: {
            **d,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        },
        emit="enriched.event",
    ),
    on="raw.event",
)

# Helper methods
flow.register(
    Transform.extract(
        keys=["user_id", "action", "result"],
        emit="analytics.event",
    ),
    on="user.action",
)

flow.register(
    Transform.rename(
        mapping={"old_key": "new_key", "value": "amount"},
        emit="normalized.event",
    ),
    on="legacy.event",
)

# Map transform - apply to multiple events
flow.register(
    MapTransform(
        transform=lambda d: {"normalized": d["value"] / 100},
        emit="value.normalized",
    ),
    on="sensor.reading.*",
)
```

**Configuration:**
- `transform` — Function to transform event data
- `emit` — Event name for transformed data

**Common patterns:**
```python
# Projection
Transform(lambda d: {k: d[k] for k in ["id", "name", "email"]}, ...)

# Filtering
Transform(lambda d: {k: v for k, v in d.items() if v is not None}, ...)

# Type conversion
Transform(lambda d: {k: str(v) for k, v in d.items()}, ...)

# Computed fields
Transform(lambda d: {**d, "total": d["price"] * d["quantity"]}, ...)
```

---

### Gateway

Bridges external systems. Base class for integrating with HTTP APIs, webhooks, message queues, etc.

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
    CallbackGateway(
        callback=send_email_notification,
    ),
    on="user.registered",
)

# Log gateway - structured logging
flow.register(
    LogGateway(level="info"),
    on="*",  # Log all events
)

# Custom gateway
class SlackGateway(Gateway):
    async def handle(self, event: Event, ctx: Context) -> None:
        await slack_client.post_message(
            channel="#notifications",
            text=f"Event: {event.name} - {event.data}",
        )
        return None  # Gateways typically don't emit events
```

**Built-in gateways:**
- `HttpGateway` — POST to HTTP endpoints
- `CallbackGateway` — Invoke Python callbacks
- `LogGateway` — Structured logging

---

## Custom Reactors

Create custom reactors by implementing the Reactor protocol or extending BaseReactor.

### Simple Reactor

```python
from agenticflow.reactors import BaseReactor
from agenticflow.events import Event
from agenticflow.flow.context import Context

class EmailReactor(BaseReactor):
    def __init__(self, smtp_config: dict):
        super().__init__(name="email_sender")
        self.smtp = setup_smtp(smtp_config)
    
    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | None:
        # Extract recipient and message
        recipient = event.data.get("recipient")
        message = event.data.get("message")
        
        # Send email
        await self.smtp.send(recipient, message)
        
        # Emit confirmation
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

### Stateful Reactor

```python
class CounterReactor(BaseReactor):
    def __init__(self):
        super().__init__(name="counter")
        self.counts = defaultdict(int)
    
    async def handle(self, event: Event, ctx: Context) -> Event | None:
        event_type = event.name
        self.counts[event_type] += 1
        
        # Emit milestone events
        count = self.counts[event_type]
        if count % 100 == 0:
            return Event(
                name="milestone.reached",
                source=self.name,
                data={
                    "event_type": event_type,
                    "count": count,
                },
            )
        return None
```

### Multi-Event Reactor

```python
class DataProcessorReactor(BaseReactor):
    async def handle(self, event: Event, ctx: Context) -> list[Event]:
        data = event.data["items"]
        
        # Process and emit multiple events
        events = []
        for item in data:
            processed = await process_item(item)
            events.append(Event(
                name="item.processed",
                source=self.name,
                data=processed,
                correlation_id=event.correlation_id,
            ))
        
        return events
```

---

## Reactor Configuration

### ReactorConfig

When registering reactors, use `ReactorConfig` for advanced configuration:

```python
from agenticflow.reactors import ReactorConfig, ErrorPolicy, HandoverStrategy

config = ReactorConfig(
    on="task.created",
    when=lambda e: e.data.get("priority") == "high",
    priority=10,
    emits="task.processed",
    
    # Error handling
    on_error=ErrorPolicy.RETRY,
    fallback=fallback_reactor,
    
    # Handover strategy
    handover=HandoverStrategy.SUMMARY,
    
    # Delegation (A2A)
    can_delegate=["specialist1", "specialist2"],
    can_reply=True,
    
    # Metadata
    role="Primary task processor",
    metadata={"team": "backend", "version": "2.0"},
)

flow.register(reactor, config)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `on` | `str \| list[str]` | Required | Event pattern(s) to react to |
| `when` | `Callable` | `None` | Condition filter function |
| `after` | `str` | `None` | React after events from specific source |
| `priority` | `int` | `0` | Higher priority executes first |
| `emits` | `str` | `None` | Override emitted event name |
| `collect` | `int` | `None` | Events to collect (for Aggregator) |
| `fan_in` | `FanInMode` | `WAIT_ALL` | How to handle multiple events |
| `handover` | `HandoverStrategy` | `FULL_OUTPUT` | Data passing strategy |
| `can_delegate` | `list[str]` | `None` | Agents this reactor can delegate to |
| `can_reply` | `bool` | `False` | Can reply to delegated requests |
| `on_error` | `ErrorPolicy` | `FAIL_FAST` | Error handling strategy |
| `fallback` | `Reactor` | `None` | Fallback reactor on error |
| `role` | `str` | `None` | Role description |
| `metadata` | `dict` | `{}` | Additional metadata |

### Error Policies

```python
from agenticflow.reactors import ErrorPolicy

# Stop flow on first error
config = ReactorConfig(on="task", on_error=ErrorPolicy.FAIL_FAST)

# Skip failed reactor, continue flow
config = ReactorConfig(on="task", on_error=ErrorPolicy.CONTINUE)

# Retry the reactor (uses retry middleware)
config = ReactorConfig(on="task", on_error=ErrorPolicy.RETRY)

# Use fallback reactor
config = ReactorConfig(
    on="task",
    on_error=ErrorPolicy.FALLBACK,
    fallback=simple_fallback_reactor,
)
```

### Handover Strategies

```python
from agenticflow.reactors import HandoverStrategy

# Pass complete output
config = ReactorConfig(on="task", handover=HandoverStrategy.FULL_OUTPUT)

# Summarize before passing (requires summarizer)
config = ReactorConfig(on="task", handover=HandoverStrategy.SUMMARY)

# Extract structured data
config = ReactorConfig(on="task", handover=HandoverStrategy.STRUCTURED)

# Accumulate into growing context
config = ReactorConfig(on="task", handover=HandoverStrategy.ACCUMULATED)
```

---

## Pattern Matching

Reactors can subscribe to events using flexible patterns:

```python
# Exact match
flow.register(handler, on="task.created")

# Wildcard - suffix
flow.register(handler, on="task.*")      # task.created, task.updated, etc.

# Wildcard - prefix
flow.register(handler, on="*.done")      # agent.done, worker.done, etc.

# Wildcard - middle
flow.register(handler, on="worker.*.done")  # worker.1.done, worker.2.done

# Multiple patterns
flow.register(handler, on=["task.created", "task.updated", "task.deleted"])

# With condition
flow.register(
    handler,
    on="task.*",
    when=lambda e: e.data.get("priority") == "urgent",
)
```

---

## Reactor Roles

Reactors serve different architectural roles in event-driven flows:

### Processing Roles

| Role | Purpose | Example Reactors |
|------|---------|------------------|
| **Producer** | Generate events from external sources | `FileWatcherReactor`, `WebhookReactor` |
| **Processor** | Transform or handle events | `AgentReactor`, `FunctionReactor`, `Transform` |
| **Aggregator** | Collect and combine events | `Aggregator`, `WaitAll`, `FirstWins` |
| **Router** | Route events to different paths | `Router`, `ConditionalRouter` |
| **Gateway** | Bridge to external systems | `HttpGateway`, `EmailReactor` |
| **Monitor** | Observe without emitting | `LogGateway`, `MetricsReactor` |

### Coordination Patterns

**Sequential Processing:**
```python
flow.register(step1, on="task.created", emits="step1.done")
flow.register(step2, on="step1.done", emits="step2.done")
flow.register(step3, on="step2.done", emits="complete")
```

**Parallel Processing:**
```python
flow.register(worker1, on="task.created")
flow.register(worker2, on="task.created")
flow.register(worker3, on="task.created")
flow.register(
    Aggregator(collect=3, emit="all.done"),
    on="worker.done",
)
```

**Conditional Branching:**
```python
flow.register(
    Router(
        routes={"urgent": "path.urgent", "normal": "path.normal"},
        key="priority",
    ),
    on="task.classified",
)
flow.register(urgent_handler, on="path.urgent")
flow.register(normal_handler, on="path.normal")
```

**Error Recovery:**
```python
flow.register(
    primary_processor,
    on="task",
    on_error=ErrorPolicy.FALLBACK,
    fallback=backup_processor,
)
```

---

## Best Practices

### Naming

```python
# ✅ Descriptive names
flow.register(AgentReactor(agent), name="email_classifier")
flow.register(Transform(...), name="normalize_dates")

# ❌ Generic names
flow.register(AgentReactor(agent), name="reactor1")
```

### Composition

```python
# ✅ Small, focused reactors
classifier = AgentReactor(classification_agent)
extractor = Transform.extract(keys=["category", "priority"])
router = Router(routes={"urgent": "urgent.path", ...})

flow.register(classifier, on="task.created")
flow.register(extractor, on="classification.done")
flow.register(router, on="extraction.done")

# ❌ Monolithic reactor doing everything
class GiantReactor:  # Don't do this
    async def handle(self, event, ctx):
        # Classify, extract, route, notify...
        pass
```

### Error Handling

```python
# ✅ Define error policies
flow.register(
    critical_processor,
    on="important.task",
    on_error=ErrorPolicy.FAIL_FAST,
)

flow.register(
    optional_notifier,
    on="task.done",
    on_error=ErrorPolicy.CONTINUE,
)

# ✅ Use fallbacks
flow.register(
    primary,
    on="task",
    on_error=ErrorPolicy.FALLBACK,
    fallback=simple_fallback,
)
```

### Testing

```python
# Test reactors in isolation
async def test_email_reactor():
    reactor = EmailReactor(test_config)
    event = Event(
        name="notification.email",
        data={"recipient": "test@example.com", "message": "Test"},
    )
    ctx = Context.create(flow_id="test")
    
    result = await reactor.handle(event, ctx)
    
    assert result is not None
    assert result.name == "email.sent"
    assert result.data["recipient"] == "test@example.com"
```

### Observability

```python
# Use metadata for tracking
flow.register(
    processor,
    on="task",
    metadata={
        "team": "backend",
        "service": "order-processing",
        "version": "2.1.0",
    },
)

# Monitor with log gateway
flow.register(LogGateway(level="info"), on="*")
```

---

## Related Documentation

- **[Flow](flow.md)** — Complete event-driven orchestration guide
- **[Events](events.md)** — Event system and patterns
- **[A2A](a2a.md)** — Agent-to-agent delegation patterns

---

## Examples

See working examples in [`examples/flow/`](../examples/flow/):
- `reactive_flow.py` — Comprehensive reactive patterns
- `conditional_workflow.py` — Conditional routing
- `parallel_processing.py` — Fan-out/fan-in patterns
- `error_handling.py` — Error recovery strategies
