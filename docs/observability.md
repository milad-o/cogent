# Observability Module

The `cogent.observability` module provides real-time visibility into agent execution with a simple, composable API.

## Quick Start

```python
from cogent import Agent
from cogent.observability import Observer

# Create an observer
observer = Observer(level="progress")

# Attach to agent
agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    tools=[my_tool],
    observer=observer,
)

# Run and watch events flow
result = await agent.run("Do something useful")

# See summary
print(observer.summary())
# Events: 8
#   agent: 4
#   tool: 4
```

### Output Levels

| Level | Shows |
|-------|-------|
| `"minimal"` | Only completion events |
| `"progress"` | Starting, thinking, tool calls, completion (default) |
| `"verbose"` | Progress + tool arguments |
| `"debug"` | Verbose + LLM requests/responses |

### Example Output

```
[Assistant] [starting]
[Assistant] [thinking]
[Assistant] [tool-call] 8552e158 calculate
[Assistant] [tool-call] 9c9b41d8 get_weather
[Assistant] [tool-result] 8552e158 calculate
  '100'
[Assistant] [tool-result] 9c9b41d8 get_weather
  '68°F, Clear'
[Assistant] [thinking] (iteration 2)
[Assistant] [completed] (2.0s) • 330 tokens
```

---

## Observer API

### Creating an Observer

```python
from cogent.observability import Observer

# From preset level
observer = Observer(level="progress")
observer = Observer(level="verbose")
observer = Observer(level="debug")

# Disable/enable
observer.enabled = False
observer.level = "debug"  # Change level dynamically
```

### Subscribing to Events

```python
# Subscribe to specific event types
def on_tool_call(event):
    print(f"Tool called: {event.data['tool_name']}")

observer.on("tool.called", on_tool_call)

# Subscribe to patterns (glob-style)
observer.on("tool.*", lambda e: print(f"Tool event: {e.type}"))
observer.on("agent.*", lambda e: print(f"Agent event: {e.type}"))

# Subscribe to all events
observer.on_all(lambda e: print(f"{e.type}: {e.data}"))
```

### Event Types

| Event | Level | Description |
|-------|-------|-------------|
| `agent.invoked` | PROGRESS | Agent started processing |
| `agent.thinking` | PROGRESS | Agent is thinking (per iteration) |
| `agent.responded` | RESULT | Agent completed with response |
| `agent.error` | RESULT | Agent failed with error |
| `tool.called` | PROGRESS | Tool invocation started |
| `tool.result` | PROGRESS | Tool returned result |
| `tool.error` | PROGRESS | Tool failed |
| `llm.request` | DEBUG | LLM API request (verbose) |
| `llm.response` | DEBUG | LLM API response (verbose) |
| `stream.start` | PROGRESS | Streaming started |
| `stream.end` | PROGRESS | Streaming completed |
| `stream.token` | TRACE | Individual token (very verbose) |

### Event Data

Events contain contextual data:

```python
def on_tool_result(event):
    print(f"Agent: {event.data['agent_name']}")
    print(f"Tool: {event.data['tool_name']}")
    print(f"Call ID: {event.data['call_id']}")  # UUID for correlation
    print(f"Result: {event.data['result']}")

observer.on("tool.result", on_tool_result)
```

### Summary

```python
# After running agent
print(observer.summary())
# Events: 10
#   agent: 6
#   tool: 4
```

---

## Tool Call Tracking

Tool calls include UUIDs for correlating calls with results:

```python
[Assistant] [tool-call] 8552e158 calculate
[Assistant] [tool-call] 9c9b41d8 get_weather
[Assistant] [tool-result] 8552e158 calculate  # Same ID
  '100'
[Assistant] [tool-result] 9c9b41d8 get_weather  # Same ID
  '68°F, Clear'
```

This enables tracking parallel tool executions and building execution graphs.

---

## Multiple Agents

A single observer can track multiple agents:

```python
observer = Observer(level="progress")

researcher = Agent(name="Researcher", observer=observer, ...)
writer = Agent(name="Writer", observer=observer, ...)

await researcher.run("Research topic")
await writer.run("Write article")

print(observer.summary())
# Events: 16
#   agent: 9
#   tool: 7
```

Output shows agent names for clarity:
```
[Researcher] [starting]
[Researcher] [thinking]
[Researcher] [tool-call] abc123 search
[Researcher] [completed] (2.1s) • 250 tokens
[Writer] [starting]
[Writer] [thinking]
[Writer] [completed] (1.5s) • 180 tokens
```

---

## Advanced: TraceBus (v1 API)

For advanced use cases, you can use the lower-level TraceBus API directly:

```python
from cogent.observability import TraceBus, TraceType

bus = TraceBus()

# Subscribe to specific trace types
bus.subscribe(TraceType.AGENT_THINKING, lambda e: print(f"Thinking: {e}"))

# Publish events
bus.publish(TraceType.AGENT_INVOKED, {"agent": "assistant"})
```

This is primarily used internally by the Observer. Most users should use the Observer API.

---

## Observing Event-Driven Flows

Different observer levels reveal different aspects of flow execution:

#### SILENT
No output whatsoever.

```python
observer = Observer(level="silent")
flow = Flow(observer=observer)
await flow.run("task")
# → (no output)
```

#### PROGRESS
Basic flow progress only - good for production monitoring.

```python
observer = Observer(level="progress")
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
observer = Observer(level="verbose")
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
observer = Observer(level="debug")
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
observer = Observer(level="trace")
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
observer = Observer(level="debug")
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
observer = Observer(level="trace")
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
observer = Observer(level="trace")
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
observer = Observer(level="debug")
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
observer = Observer(level="trace")
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
observer = Observer(level="debug")

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

1. **Start with PROGRESS**: Use `Observer(level="progress")` for production
2. **DEBUG for development**: Use `Observer(level="debug")` during development
3. **TRACE for troubleshooting**: Use `Observer(level="trace")` when debugging issues
4. **Filter traces**: Don't process all traces - filter by type
5. **Export for analysis**: Save traces to JSON for offline analysis
6. **Monitor performance**: Track `duration_ms` in traces to find bottlenecks
7. **Check for NO_MATCH**: Indicates misconfigured reactors

### Example: Full Flow Observability

```python
from cogent import Agent, Flow
from cogent.observability import Observer, TraceType

# Setup
model = get_model()
researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)

observer = Observer(level="debug")
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
from cogent.observability import TraceBus, Event
from cogent.core import TraceType

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
from cogent.observability import get_trace_bus, set_trace_bus

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
from cogent.observability import Trace

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
from cogent.observability import ConsoleEventHandler

handler = ConsoleEventHandler(
    format="[{timestamp}] {type}: {data}",
    colored=True,
)

bus.subscribe_all(handler)
```

### FileEventHandler

```python
from cogent.observability import FileEventHandler

handler = FileEventHandler(
    path="events.jsonl",
    format="json",  # or "text"
    rotate_size_mb=100,
)

bus.subscribe_all(handler)
```

### FilteringEventHandler

```python
from cogent.observability import FilteringEventHandler

handler = FilteringEventHandler(
    wrapped=ConsoleEventHandler(),
    include_types=[TraceType.TASK_COMPLETED, TraceType.TASK_FAILED],
    exclude_data_keys=["sensitive_field"],
)

bus.subscribe_all(handler)
```

### MetricsEventHandler

```python
from cogent.observability import MetricsEventHandler

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
from cogent.observability import Tracer, SpanKind

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
from cogent.observability import SpanContext

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
from cogent.observability import ExecutionTracer, TraceLevel

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
from cogent.observability import TracingObserver

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
from cogent.observability import MetricsCollector, Counter, Gauge, Histogram

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
from cogent import Agent

# Simple verbose flag
agent = Agent(name="assistant", model=model, verbose=True)

# Or configure output
from cogent.observability import configure_output, Verbosity

configure_output(
    verbosity=Verbosity.DETAILED,
    show_timing=True,
    show_tokens=True,
)
```

### OutputConfig

```python
from cogent.observability import OutputConfig, Verbosity, OutputFormat, Theme

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
from cogent.observability import ProgressTracker, ProgressEvent

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
from cogent.observability import Dashboard, DashboardConfig

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
from cogent.observability import WebSocketServer, start_websocket_server

# Start server
server = await start_websocket_server(port=8765)

# Or use the handler directly with your own server
from cogent.observability import websocket_handler

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
from cogent.observability import SystemInspector, AgentInspector

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
