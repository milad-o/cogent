# Observability Module

The `agenticflow.observability` module provides comprehensive monitoring, tracing, metrics, and progress output for understanding system behavior at runtime.

## Overview

The observability module includes:
- **EventBus** - Central pub/sub for all events
- **Observer** - Unified observability for agents and flows
- **Tracer** - Distributed tracing with spans
- **Metrics** - Counters, gauges, histograms
- **Progress** - Rich terminal output and progress tracking
- **Dashboard** - Real-time monitoring UI

```python
from agenticflow import Agent
from agenticflow.observability import Observer

# Simple: verbose output
agent = Agent(name="assistant", model=model, verbose=True)

# Advanced: full observability
observer = Observer.trace()
result = await agent.run("Hello", observer=observer)
```

---

## Observer

Unified observability interface with preset levels:

```python
from agenticflow.observability import Observer

# Preset levels (recommended)
observer = Observer.silent()    # No output
observer = Observer.progress()  # Basic progress
observer = Observer.verbose()   # Show agent outputs
observer = Observer.debug()     # Include tool calls
observer = Observer.trace()     # Maximum detail + graph

# Use with agents
result = await agent.run("Query", observer=observer)

# Use with flows
result = await flow.run("Task", observer=observer)
```

### ObservabilityLevel

```python
from agenticflow.observability import ObservabilityLevel

ObservabilityLevel.SILENT   # No output
ObservabilityLevel.MINIMAL  # Start/complete only
ObservabilityLevel.NORMAL   # Standard progress
ObservabilityLevel.VERBOSE  # Show outputs
ObservabilityLevel.DEBUG    # Tool calls + thinking
ObservabilityLevel.TRACE    # Everything + execution graph
```

### Custom Observer

```python
from agenticflow.observability import Observer, Channel

observer = Observer(
    level=ObservabilityLevel.DEBUG,
    channels=[
        Channel.CONSOLE,    # Terminal output
        Channel.FILE,       # Log to file
        Channel.WEBSOCKET,  # Real-time streaming
    ],
    file_path="agent.log",
)
```

### Modular Channels (Opt-in Observability)

The observability system uses **channels** to let you subscribe to specific event categories. This keeps output clean and focused on what matters to you:

```python
from agenticflow.observability import Observer, Channel

# Subscribe to specific channels
observer = Observer(
    level=ObservabilityLevel.DEBUG,
    channels=[
        Channel.AGENTS,     # Agent lifecycle events
        Channel.TOOLS,      # Tool calls and results
        Channel.TASKS,      # Task execution
    ],
)

# Available channels:
# - Channel.AGENTS: Agent thinking, acting, status
# - Channel.TOOLS: Tool calls, results, errors
# - Channel.MESSAGES: Inter-agent communication
# - Channel.TASKS: Task lifecycle
# - Channel.LLM: Raw LLM request/response (opt-in)
# - Channel.STREAMING: Token-by-token output
# - Channel.MEMORY: Memory operations
# - Channel.RETRIEVAL: RAG retrieval
# - Channel.DOCUMENTS: Document loading/splitting
# - Channel.MCP: Model Context Protocol
# - Channel.REACTIVE: Reactive flow events
# - Channel.SYSTEM: System-level events
# - Channel.RESILIENCE: Retries, circuit breakers
# - Channel.ALL: Everything
```

**LLM Events are Opt-in**: By default, LLM request/response events show **subtle presence** (just that a request/response occurred). To see full details (prompts, responses, content), explicitly subscribe to `Channel.LLM`:

```python
# Default behavior: LLM events show subtle presence only
observer = Observer(
    level=ObservabilityLevel.DEBUG,
    channels=[Channel.AGENTS, Channel.TOOLS],
)
# → LLM request (5 messages, 3 tools)
# ← LLM response 1.2s, 2 tools

# Opt-in for LLM details
observer = Observer(
    level=ObservabilityLevel.DEBUG,
    channels=[Channel.AGENTS, Channel.TOOLS, Channel.LLM],  # Add LLM channel
)
# Now you'll see prompts, system messages, response content at TRACE level

# Or use TRACE level for everything
observer = Observer.trace()  # Includes all channels + full details
```

This modular design ensures:
- ✅ Clean, focused output by default
- ✅ Opt-in to detailed LLM debugging when needed
- ✅ No noise from internal LLM calls unless you want it
- ✅ Easy to filter what you care about

---

## EventBus

Central pub/sub system for all framework events:

```python
from agenticflow.observability import EventBus, Event
from agenticflow.core import EventType

bus = EventBus()

# Subscribe to specific event type
def on_task_complete(event: Event):
    print(f"Task completed: {event.data['task_id']}")

bus.subscribe(EventType.TASK_COMPLETED, on_task_complete)

# Subscribe to multiple types
bus.subscribe_many(
    [EventType.TASK_STARTED, EventType.TASK_COMPLETED],
    log_task_events,
)

# Subscribe to ALL events
bus.subscribe_all(lambda e: print(e))

# Publish events
await bus.publish(Event(
    type=EventType.TASK_STARTED,
    data={"task_id": "123", "agent": "worker"},
))

# Simple publish API
await bus.publish("task.completed", {"task_id": "123"})
```

### Async Handlers

Both sync and async handlers are supported:

```python
# Sync handler
def sync_handler(event: Event):
    print(event.data)

# Async handler
async def async_handler(event: Event):
    await send_notification(event.data)

bus.subscribe(EventType.TASK_COMPLETED, sync_handler)
bus.subscribe(EventType.TASK_COMPLETED, async_handler)
```

### Event History

Query past events:

```python
# Get event history
events = bus.get_history(
    event_type=EventType.TASK_COMPLETED,
    limit=10,
)

# Filter by custom function
task_events = bus.get_history(
    filter_fn=lambda e: e.data.get("task_id") == "123"
)
```

### Global EventBus

```python
from agenticflow.observability import get_event_bus, set_event_bus

# Get the global bus
bus = get_event_bus()

# Set a custom global bus
custom_bus = EventBus(max_history=50000)
set_event_bus(custom_bus)
```

---

## Event

Immutable event records:

```python
from agenticflow.observability import Event

event = Event(
    type=EventType.TASK_COMPLETED,
    data={"task_id": "123", "result": "success"},
    source="agent:researcher",
    correlation_id="req-456",
)

print(event.id)          # Unique event ID
print(event.type)        # EventType enum
print(event.data)        # Event payload
print(event.timestamp)   # When it occurred
print(event.source)      # What emitted it
print(event.correlation_id)  # For tracing
```

---

## Event Handlers

Pre-built handlers for common use cases:

### ConsoleEventHandler

```python
from agenticflow.observability import ConsoleEventHandler

handler = ConsoleEventHandler(
    format="[{timestamp}] {type}: {data}",
    colored=True,
)

bus.subscribe_all(handler)
```

### FileEventHandler

```python
from agenticflow.observability import FileEventHandler

handler = FileEventHandler(
    path="events.jsonl",
    format="json",  # or "text"
    rotate_size_mb=100,
)

bus.subscribe_all(handler)
```

### FilteringEventHandler

```python
from agenticflow.observability import FilteringEventHandler

handler = FilteringEventHandler(
    wrapped=ConsoleEventHandler(),
    include_types=[EventType.TASK_COMPLETED, EventType.TASK_FAILED],
    exclude_data_keys=["sensitive_field"],
)

bus.subscribe_all(handler)
```

### MetricsEventHandler

```python
from agenticflow.observability import MetricsEventHandler

handler = MetricsEventHandler()
bus.subscribe_all(handler)

# Get metrics
print(handler.metrics)
# {
#     "task_completed": 42,
#     "task_failed": 3,
#     "tool_called": 156,
# }
```

---

## Tracer

Distributed tracing with spans:

```python
from agenticflow.observability import Tracer, SpanKind

tracer = Tracer(service_name="my-agent")

async with tracer.start_span("process_request", kind=SpanKind.SERVER) as span:
    span.set_attribute("user_id", "123")
    
    # Nested spans
    async with tracer.start_span("query_database") as db_span:
        db_span.set_attribute("query", "SELECT ...")
        result = await db.query(...)
    
    async with tracer.start_span("call_llm") as llm_span:
        llm_span.set_attribute("model", "gpt-4o")
        response = await llm.invoke(...)
```

### Span Context

```python
from agenticflow.observability import SpanContext

# Get current span context
ctx = tracer.get_current_context()

# Propagate to other services
headers = {"traceparent": ctx.to_header()}

# Create span from incoming context
incoming_ctx = SpanContext.from_header(request.headers["traceparent"])
async with tracer.start_span("handle", context=incoming_ctx) as span:
    ...
```

---

## Execution Tracer

Detailed execution tracing for debugging:

```python
from agenticflow.observability import ExecutionTracer, TraceLevel

tracer = ExecutionTracer(level=TraceLevel.DETAILED)

result = await agent.run("Query", tracer=tracer)

# Access trace
trace = tracer.get_trace()
print(f"Nodes: {len(trace.nodes)}")
print(f"Duration: {trace.duration_ms}ms")

# Export trace
trace.to_json("trace.json")
trace.to_html("trace.html")
```

### TracingObserver

Combine observability with tracing:

```python
from agenticflow.observability import TracingObserver

observer = TracingObserver(
    level=ObservabilityLevel.DEBUG,
    export_on_complete=True,
    export_path="traces/",
)

result = await flow.run("Task", observer=observer)
```

---

## Metrics

Collect and export metrics:

```python
from agenticflow.observability import MetricsCollector, Counter, Gauge, Histogram

collector = MetricsCollector()

# Counter - monotonically increasing
requests = collector.counter("requests_total", "Total requests")
requests.inc()
requests.inc(5)

# Gauge - can go up and down
active = collector.gauge("active_agents", "Currently active agents")
active.set(3)
active.inc()
active.dec()

# Histogram - distribution of values
latency = collector.histogram(
    "request_latency_ms",
    "Request latency",
    buckets=[10, 50, 100, 500, 1000],
)
latency.observe(42.5)

# Timer context manager
timer = collector.timer("operation_duration")
with timer:
    await do_work()
```

### Export Metrics

```python
# Prometheus format
print(collector.export_prometheus())

# JSON format
print(collector.export_json())

# Start Prometheus endpoint
await collector.start_server(port=9090)
```

---

## Progress Output

Rich terminal output for agent execution:

### Quick Start

```python
from agenticflow import Agent

# Simple verbose flag
agent = Agent(name="assistant", model=model, verbose=True)

# Or configure output
from agenticflow.observability import configure_output, Verbosity

configure_output(
    verbosity=Verbosity.DETAILED,
    show_timing=True,
    show_tokens=True,
)
```

### OutputConfig

```python
from agenticflow.observability import OutputConfig, Verbosity, OutputFormat, Theme

config = OutputConfig(
    verbosity=Verbosity.DETAILED,
    format=OutputFormat.RICH,     # or TEXT, JSON, MINIMAL
    theme=Theme.DARK,             # or LIGHT, MONOCHROME
    show_thinking=True,
    show_tool_calls=True,
    show_timing=True,
    show_tokens=True,
)

observer = Observer(output_config=config)
```

### Verbosity Levels

| Level | Shows |
|-------|-------|
| `SILENT` | Nothing |
| `MINIMAL` | Start/complete only |
| `NORMAL` | Progress + results |
| `DETAILED` | + Tool calls |
| `DEBUG` | + Thinking/reasoning |
| `TRACE` | Everything |

### ProgressTracker

```python
from agenticflow.observability import ProgressTracker, ProgressEvent

tracker = ProgressTracker()

# Manual progress updates
tracker.update(ProgressEvent(
    type="task_started",
    agent="researcher",
    message="Starting research...",
))

tracker.update(ProgressEvent(
    type="tool_called",
    agent="researcher",
    tool="search",
    args={"query": "AI news"},
))

tracker.complete(result="Research complete")
```

---

## Dashboard

Real-time monitoring UI:

```python
from agenticflow.observability import Dashboard, DashboardConfig

config = DashboardConfig(
    port=8080,
    refresh_rate_ms=1000,
    show_events=True,
    show_metrics=True,
    show_traces=True,
)

dashboard = Dashboard(config)
await dashboard.start()

# Dashboard available at http://localhost:8080
```

---

## WebSocket Streaming

Real-time event streaming:

```python
from agenticflow.observability import WebSocketServer, start_websocket_server

# Start server
server = await start_websocket_server(port=8765)

# Or use the handler directly with your own server
from agenticflow.observability import websocket_handler

# In your WebSocket endpoint:
async def handle_client(websocket):
    await websocket_handler(websocket)
```

### Client Connection

```javascript
// JavaScript client
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event:', data.type, data.data);
};
```

---

## Inspectors

Inspect system state at runtime:

```python
from agenticflow.observability import SystemInspector, AgentInspector

# System-wide inspection
inspector = SystemInspector()
print(inspector.summary())
print(inspector.active_agents())
print(inspector.recent_events(limit=10))

# Agent-specific inspection
agent_inspector = AgentInspector(agent)
print(agent_inspector.state())
print(agent_inspector.history())
print(agent_inspector.tools())
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Observer` | Unified observability interface |
| `EventBus` | Central pub/sub system |
| `Event` | Immutable event record |
| `Tracer` | Distributed tracing |
| `MetricsCollector` | Metrics collection |
| `ProgressTracker` | Progress output |
| `Dashboard` | Real-time monitoring UI |

### Event Handlers

| Class | Description |
|-------|-------------|
| `ConsoleEventHandler` | Print to terminal |
| `FileEventHandler` | Log to file |
| `FilteringEventHandler` | Filter events |
| `MetricsEventHandler` | Collect metrics |

### Tracing

| Class | Description |
|-------|-------------|
| `Span` | A single traced operation |
| `SpanContext` | Context for distributed tracing |
| `SpanKind` | Type of span (SERVER, CLIENT, etc.) |
| `ExecutionTracer` | Detailed execution tracing |
| `TracingObserver` | Combined observer + tracer |

### Metrics

| Class | Description |
|-------|-------------|
| `Counter` | Monotonically increasing counter |
| `Gauge` | Value that can go up/down |
| `Histogram` | Distribution of values |
| `Timer` | Duration measurement |

### Progress

| Class | Description |
|-------|-------------|
| `OutputConfig` | Output configuration |
| `Verbosity` | Verbosity levels |
| `OutputFormat` | Output formats (RICH, TEXT, JSON) |
| `Theme` | Color themes |
| `ProgressEvent` | Progress update event |
