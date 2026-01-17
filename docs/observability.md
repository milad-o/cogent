# Observability Module

The `agenticflow.observability` module provides comprehensive monitoring, tracing, metrics, and progress output for understanding system behavior at runtime.

## Overview

The observability module includes:
- **TraceBus** - Central pub/sub for all events
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
observer = Observer.debug()     # Include tool calls (excludes raw LLM content)
observer = Observer.trace()     # Maximum detail + graph (excludes raw LLM content)

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

# Opt-in for LLM details (explicit channel subscription required)
observer = Observer(
    level=ObservabilityLevel.DEBUG,
    channels=[Channel.AGENTS, Channel.TOOLS, Channel.LLM],  # Explicitly add LLM channel
)
# Now you'll see prompts, system messages, response content

# Note: Observer.debug() and Observer.trace() do NOT include Channel.LLM by default
# This is intentional - LLM content requires explicit opt-in for privacy
```

This modular design ensures:
- ✅ Clean, focused output by default
- ✅ Opt-in to detailed LLM debugging when needed
- ✅ No noise from internal LLM calls unless you want it
- ✅ Easy to filter what you care about

---

## Observing Event-Driven Flows

The event-driven Flow paradigm is fully integrated with the observability system, providing deep visibility into event processing, reactor activations, and flow execution.

### Understanding the Dual-Bus Architecture

AgenticFlow uses two separate event systems for clean separation of concerns:

#### 1. EventBus (Orchestration)
**Module**: `agenticflow.events.EventBus`  
**Purpose**: Core orchestration and flow control  
**Events**: `task.created`, `agent.done`, `research.complete`, custom events  
**Used by**: Flow, reactors, agent coordination  
**Consumers**: Reactors registered via `flow.register()`

```python
from agenticflow.events import EventBus

# Orchestration bus - handles flow logic
events = EventBus()
await events.publish("task.created", {"id": "123"})
```

#### 2. TraceBus (Observability)
**Module**: `agenticflow.observability.TraceBus`  
**Purpose**: Telemetry, tracing, and monitoring  
**Events**: `TraceType` enum values (REACTIVE_*, AGENT_*, TASK_*)  
**Used by**: Observer, metrics, logging, exporters  
**Consumers**: Observer handlers, dashboards, log files

```python
from agenticflow.observability import TraceBus

# Observability bus - separate from orchestration
traces = TraceBus()
bus.subscribe(TraceType.REACTIVE_FLOW_STARTED, on_flow_start)
```

#### Why Two Buses?

| Reason | Benefit |
|--------|---------|
| **Separation of Concerns** | Orchestration logic ≠ observability logic |
| **Performance** | Observability can be disabled without affecting flow |
| **Flexibility** | Different event schemas and lifecycles |
| **Extensibility** | Each bus can evolve independently |

#### How They Connect

The Flow automatically bridges the two systems:
- Flow publishes orchestration events to **EventBus** (reactors respond)
- Flow's `_observe()` method emits to **TraceBus** (observers see it)
- No direct coupling between buses
- Observer attaches to TraceBus automatically

```python
from agenticflow import Flow, Agent
from agenticflow.observability import Observer

observer = Observer.trace()
flow = Flow(observer=observer)  # Observer attaches to TraceBus

# When you register reactors:
flow.register(agent, on="task.created")  # Listens to EventBus

# When flow runs:
# 1. EventBus: task.created → agent reactor
# 2. TraceBus: REACTIVE_AGENT_TRIGGERED → observer
```

### Flow Trace Events

The Flow emits detailed trace events for every step of execution:

| TraceType | Description | When Emitted | Key Data |
|-----------|-------------|--------------|----------|
| `REACTIVE_FLOW_STARTED` | Flow execution begins | `flow.run()` called | `task`, `agents`, `flow_id` |
| `REACTIVE_EVENT_EMITTED` | Event published to flow | Event enters system | `event_type`, `data`, `source` |
| `REACTIVE_EVENT_PROCESSED` | Event matched and handled | After reactor processes | `event_type`, `reactor`, `duration_ms` |
| `REACTIVE_AGENT_TRIGGERED` | Agent reactor activated | Agent starts processing | `agent`, `event`, `trigger` |
| `REACTIVE_AGENT_COMPLETED` | Agent finished successfully | Agent returns result | `agent`, `output`, `duration_ms` |
| `REACTIVE_AGENT_FAILED` | Agent encountered error | Agent raises exception | `agent`, `error`, `traceback` |
| `REACTIVE_NO_MATCH` | No reactors matched event | Event processed but no match | `event_type`, `available_reactors` |
| `REACTIVE_ROUND_STARTED` | New processing round begins | Start of event loop iteration | `round`, `pending_events` |
| `REACTIVE_ROUND_COMPLETED` | Round finished | All events in round processed | `round`, `events_processed`, `duration_ms` |
| `REACTIVE_FLOW_COMPLETED` | Flow execution finished | Flow terminates successfully | `output`, `total_events`, `total_rounds` |
| `REACTIVE_FLOW_FAILED` | Flow execution failed | Flow terminates with error | `error`, `partial_output`, `events_processed` |

### Observer Levels for Flows

Different observer levels reveal different aspects of flow execution:

#### SILENT
No output whatsoever.

```python
observer = Observer.silent()
flow = Flow(observer=observer)
await flow.run("task")
# → (no output)
```

#### PROGRESS
Basic flow progress only - good for production monitoring.

```python
observer = Observer.progress()
flow = Flow(observer=observer)
await flow.run("task")
```

**Output**:
```
⚡ Flow started (3 agents registered)
⏱️  Round 1...
⏱️  Round 2...
✅ Flow completed in 2.3s
```

#### VERBOSE
Flow progress + agent outputs - shows what's happening.

```python
observer = Observer.verbose()
flow = Flow(observer=observer)
await flow.run("task")
```

**Output**:
```
⚡ Flow started
📤 Event emitted: task.created
🤖 researcher triggered by task.created
📝 researcher: "Based on my research..."
📤 Event emitted: research.done
🤖 writer triggered by research.done
📝 writer: "Here's the article..."
✅ Flow completed
```

#### DEBUG
Detailed execution - includes events, conditions, reactor matching.

```python
observer = Observer.debug()
flow = Flow(observer=observer)
await flow.run("task")
```

**Output**:
```
⚡ REACTIVE_FLOW_STARTED
  task: "Write about quantum computing"
  agents: [researcher, writer]
  
📤 REACTIVE_EVENT_EMITTED: task.created
  data: {type: "research"}
  
🔍 Matching reactors...
  ✓ researcher matches (priority: 0)
  
🤖 REACTIVE_AGENT_TRIGGERED: researcher
  trigger: on="task.created"
  condition: None
  
💬 Agent thinking...

📝 REACTIVE_AGENT_COMPLETED: researcher
  output: "Based on my research..."
  duration: 1.2s
  
📤 REACTIVE_EVENT_EMITTED: research.done
  auto_emit: True
  
⏱️  REACTIVE_ROUND_COMPLETED
  round: 1
  events_processed: 1
  duration: 1.2s
```

#### TRACE
Everything + execution graphs and full event history.

```python
observer = Observer.trace()
flow = Flow(observer=observer)
await flow.run("task")

# Access full trace history
for trace in observer.traces:
    if trace.type.startswith("reactive"):
        print(f"{trace.timestamp}: {trace.type}")
        print(f"  Data: {trace.data}")
```

### Common Observability Patterns

#### 1. Debugging Event Flow

See which events triggered which reactors:

```python
observer = Observer.debug()
flow = Flow(observer=observer)

result = await flow.run("task")

# Filter reactive events
reactive_events = [
    observed.event for observed in observer.events()
    if observed.event.type.value.startswith("reactive")
]

for event in reactive_events:
    print(f"{event.type}: {event.data.get('event_type', 'N/A')}")
```

#### 2. Tracking Performance

Identify slow reactors and bottlenecks:

```python
observer = Observer.trace()
flow = Flow(observer=observer)

result = await flow.run("task")

# Find slow agent executions
slow_agents = [
    observed.event for observed in observer.events()
    if observed.event.type == TraceType.REACTIVE_AGENT_COMPLETED
    and observed.event.data.get("duration_ms", 0) > 1000  # > 1 second
]

for event in slow_agents:
    agent = event.data["agent"]
    duration = event.data["duration_ms"]
    print(f"{agent} took {duration:.0f}ms")
```

#### 3. Understanding Event Chains

See how events flow through the system:

```python
observer = Observer.trace()
flow = Flow(observer=observer)

result = await flow.run("task")

# Build event chain
events = [
    observed.event for observed in observer.events()
    if observed.event.type == TraceType.REACTIVE_EVENT_EMITTED
]

print("Event Chain:")
for i, event in enumerate(events, 1):
    event_type = event.data["event_type"]
    source = event.data.get("source", "system")
    print(f"{i}. {event_type} (from {source})")
```

#### 4. Detecting Issues

Find events that didn't match any reactors:

```python
observer = Observer.debug()
flow = Flow(observer=observer)

result = await flow.run("task")

# Find unmatched events
unmatched = [
    observed.event for observed in observer.events()
    if observed.event.type == TraceType.REACTIVE_NO_MATCH
]

if unmatched:
    print("⚠️  Events with no matching reactors:")
    for event in unmatched:
        event_type = event.data["event_type"]
        available = event.data.get("available_reactors", [])
        print(f"  - {event_type} (available: {available})")
```

#### 5. Exporting Traces

Save flow execution for later analysis:

```python
observer = Observer.trace()
flow = Flow(observer=observer)

result = await flow.run("task")

# Export to JSON
import json
from pathlib import Path

traces_data = [
    {
        "type": observed.event.type.value,
        "timestamp": observed.event.timestamp.isoformat(),
        "data": observed.event.data,
    }
    for observed in observer.events()
    if observed.event.type.value.startswith("reactive")
]

Path("flow_traces.json").write_text(json.dumps(traces_data, indent=2))
print(f"✅ Exported {len(traces_data)} traces")
```

### Working with Multiple Flows

Share an observer across multiple flow executions:

```python
observer = Observer.debug()

# Flow 1
flow1 = Flow(observer=observer)
result1 = await flow1.run("task 1")

# Flow 2
flow2 = Flow(observer=observer)
result2 = await flow2.run("task 2")

# Observer sees both flows
all_flows = [
    observed.event for observed in observer.events()
    if observed.event.type == TraceType.REACTIVE_FLOW_STARTED
]

print(f"Total flows observed: {len(all_flows)}")
```

### Best Practices

1. **Start with PROGRESS**: Use `Observer.progress()` for production
2. **DEBUG for development**: Use `Observer.debug()` during development
3. **TRACE for troubleshooting**: Use `Observer.trace()` when debugging issues
4. **Filter traces**: Don't process all traces - filter by type
5. **Export for analysis**: Save traces to JSON for offline analysis
6. **Monitor performance**: Track `duration_ms` in traces to find bottlenecks
7. **Check for NO_MATCH**: Indicates misconfigured reactors

### Example: Full Flow Observability

```python
from agenticflow import Agent, Flow
from agenticflow.observability import Observer, TraceType

# Setup
model = get_model()
researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)

observer = Observer.debug()
flow = Flow(observer=observer)

flow.register(researcher, on="task.created", emits="research.done")
flow.register(writer, on="research.done", emits="flow.done")

# Execute
result = await flow.run("Write about quantum computing")

# Analyze
print("\n=== Execution Summary ===")
print(f"Output: {result.output[:100]}...")
print(f"Events processed: {len([o for o in observer.events() if 'EVENT' in o.event.type.value])}")
print(f"Agents triggered: {len([o for o in observer.events() if o.event.type == TraceType.REACTIVE_AGENT_TRIGGERED])}")

# Performance analysis
agent_times = {}
for observed in observer.events():
    if observed.event.type == TraceType.REACTIVE_AGENT_COMPLETED:
        agent = observed.event.data["agent"]
        duration = observed.event.data["duration_ms"]
        agent_times[agent] = duration

print("\n=== Performance ===")
for agent, duration in agent_times.items():
    print(f"{agent}: {duration:.0f}ms")

# Event chain
print("\n=== Event Chain ===")
events = [
    observed.event.data["event_type"]
    for observed in observer.events()
    if observed.event.type == TraceType.REACTIVE_EVENT_EMITTED
]
print(" → ".join(events))
```

---

## TraceBus

Central pub/sub system for all framework events:

```python
from agenticflow.observability import TraceBus, Event
from agenticflow.core import TraceType

bus = TraceBus()

# Subscribe to specific event type
def on_task_complete(trace: Trace):
    print(f"Task completed: {event.data['task_id']}")

bus.subscribe(TraceType.TASK_COMPLETED, on_task_complete)

# Subscribe to multiple types
bus.subscribe_many(
    [TraceType.TASK_STARTED, TraceType.TASK_COMPLETED],
    log_task_events,
)

# Subscribe to ALL events
bus.subscribe_all(lambda e: print(e))

# Publish events
await bus.publish(Event(
    type=TraceType.TASK_STARTED,
    data={"task_id": "123", "agent": "worker"},
))

# Simple publish API
await bus.publish("task.completed", {"task_id": "123"})
```

### Async Handlers

Both sync and async handlers are supported:

```python
# Sync handler
def sync_handler(trace: Trace):
    print(event.data)

# Async handler
async def async_handler(trace: Trace):
    await send_notification(event.data)

bus.subscribe(TraceType.TASK_COMPLETED, sync_handler)
bus.subscribe(TraceType.TASK_COMPLETED, async_handler)
```

### Event History

Query past events:

```python
# Get event history
events = bus.get_history(
    event_type=TraceType.TASK_COMPLETED,
    limit=10,
)

# Filter by custom function
task_events = bus.get_history(
    filter_fn=lambda e: e.data.get("task_id") == "123"
)
```

### Global TraceBus

```python
from agenticflow.observability import get_trace_bus, set_trace_bus

# Get the global bus
bus = get_trace_bus()

# Set a custom global bus
custom_bus = TraceBus(max_history=50000)
set_trace_bus(custom_bus)
```

---

## Trace

Immutable event records:

```python
from agenticflow.observability import Trace

event = Event(
    type=TraceType.TASK_COMPLETED,
    data={"task_id": "123", "result": "success"},
    source="agent:researcher",
    correlation_id="req-456",
)

print(event.id)          # Unique event ID
print(event.type)        # TraceType enum
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
    include_types=[TraceType.TASK_COMPLETED, TraceType.TASK_FAILED],
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
| `TraceBus` | Central pub/sub system |
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
