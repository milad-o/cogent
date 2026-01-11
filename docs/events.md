# Events Module

The events module provides the foundation for event-driven orchestration in AgenticFlow.

## Core Events (`agenticflow.events`)

The core events module powers reactive agent orchestration with:

- **Event**: Immutable event records with name, data, timestamp, and correlation support
- **EventBus**: Lightweight pub/sub for routing events between agents
- **EventSource**: Ingest events from external systems (webhooks, files, queues)
- **EventSink**: Deliver events to external systems (webhooks, databases)

```python
from agenticflow.events import (
    Event,
    EventBus,
    # Sources
    EventSource,
    WebhookSource,
    FileWatcherSource,
    RedisStreamSource,
    # Sinks
    EventSink,
    WebhookSink,
)
```

---

## Event Sources

Event sources inject events from external systems into your reactive flows.

### FileWatcherSource

Monitor directories for file changes:

```python
from agenticflow.reactive.flow import EventFlow
from agenticflow.reactive import react_to
from agenticflow.events import FileWatcherSource

flow = EventFlow()

# Register agents that react to file events
flow.register(json_processor, [react_to("file.created").when(
    lambda e: e.data.get("extension") == ".json"
)])
flow.register(csv_processor, [react_to("file.created").when(
    lambda e: e.data.get("extension") == ".csv"
)])

# Watch directories for new files
flow.source(FileWatcherSource(
    paths=["./incoming", "./uploads"],
    patterns=["*.json", "*.csv", "*.txt"],
    event_prefix="file",  # Emits: file.created, file.modified, file.deleted
))

await flow.run("Process incoming files")
```

**Emitted Events:**
| Event | Data |
|-------|------|
| `file.created` | `{path, filename, extension}` |
| `file.modified` | `{path, filename, extension}` |
| `file.deleted` | `{path, filename, extension}` |

### WebhookSource

Receive HTTP webhooks as events:

```python
from agenticflow.events import WebhookSource

flow.source(WebhookSource(
    path="/api/webhooks",
    port=8080,
    event_type="webhook.received",  # Or extract from request
))

# Handle incoming webhooks
flow.register(webhook_handler, [react_to("webhook.received")])
```

**Requirements:** `starlette`, `uvicorn`

### RedisStreamSource

Consume from Redis Streams with consumer groups:

```python
from agenticflow.events import RedisStreamSource

flow.source(RedisStreamSource(
    redis_url="redis://localhost:6379",
    stream="events",
    group="my-consumer-group",
    consumer="worker-1",
))

flow.register(event_handler, [react_to("redis.*")])
```

**Features:**
- Consumer groups for distributed processing
- Message acknowledgment
- Automatic reconnection

**Requirements:** `redis`

---

## Event Sinks

Event sinks send events to external systems when they occur in your flow.

### WebhookSink

POST events to HTTP endpoints:

```python
from agenticflow.events import WebhookSink

# Send completion events to a webhook
flow.sink(
    WebhookSink(
        url="https://your-service.com/webhooks",
        headers={"Authorization": "Bearer token123"},
    ),
    pattern="*.completed",  # Glob pattern matching
)

# Send error events to Slack
flow.sink(
    WebhookSink(url="https://hooks.slack.com/services/..."),
    pattern="*.error",
)
```

**Pattern Matching:**
| Pattern | Matches |
|---------|---------|
| `*` | All events |
| `*.completed` | `task.completed`, `order.completed`, etc. |
| `order.*` | `order.created`, `order.shipped`, etc. |
| `agent.*.error` | `agent.processor.error`, etc. |

**Requirements:** `httpx`

### Custom Sinks

Create your own sink by extending `EventSink`:

```python
from agenticflow.events import EventSink

class DatabaseSink(EventSink):
    """Store events in a database."""
    
    def __init__(self, connection_string: str):
        self.db = connect(connection_string)
    
    async def send(self, event) -> None:
        await self.db.insert("events", {
            "name": event.name,
            "data": event.data,
            "timestamp": event.timestamp,
        })
    
    async def close(self) -> None:
        await self.db.close()

flow.sink(DatabaseSink("postgresql://..."), pattern="*")
```

---

## Integration with EventFlow

Sources and sinks integrate seamlessly with `EventFlow`:

```python
from agenticflow.reactive.flow import EventFlow
from agenticflow.reactive import react_to, Observer
from agenticflow.events import FileWatcherSource, WebhookSink

# Create flow with observability
flow = EventFlow(observer=Observer.progress())

# External sources → inject events into flow
flow.source(FileWatcherSource(paths=["./incoming"]))

# Agents react to events
flow.register(processor, [react_to("file.created")])
flow.register(notifier, [react_to("*.completed")])

# Event sinks → send events out of flow
flow.sink(WebhookSink(url="https://api.example.com/events"), pattern="*.completed")

# Run the reactive flow
await flow.run("Process incoming events")
```

**Lifecycle:**
1. Sources start when flow runs
2. Sources emit events → agents react
3. Agent completions trigger sinks
4. Sources and sinks stop when flow completes

---

## Observability (TraceBus + Trace)

> **Note:** For observability/telemetry, use `Trace` and `TraceBus` from `agenticflow.observability`.
> The `agenticflow.events.Event` is for core orchestration routing.

```python
# Observability (telemetry, logging)
from agenticflow.observability import Trace, TraceBus, get_trace_bus

# Core orchestration (agent-to-agent events)
from agenticflow.events import Event, EventBus
```

---

## Migration Guide

The observability module has been renamed for clarity:

```python
# Old (deprecated)
from agenticflow.observability import Event, EventBus, get_event_bus

# New (recommended)
from agenticflow.observability import Trace, TraceBus, get_trace_bus
```

---

## EventBus

The central pub/sub system for event distribution:

```python
from agenticflow.events import EventBus
from agenticflow.core import EventType

bus = EventBus(max_history=10000)

# Subscribe to specific event type
def on_task_complete(event):
    print(f"Task completed: {event.data}")

bus.subscribe(EventType.TASK_COMPLETED, on_task_complete)

# Subscribe to multiple event types
bus.subscribe_many(
    [EventType.TASK_STARTED, EventType.TASK_COMPLETED],
    log_task_events,
)

# Subscribe to ALL events
bus.subscribe_all(lambda e: print(e))

# Publish event
await bus.publish(Event(
    type=EventType.TASK_STARTED,
    data={"task_id": "123", "agent": "worker"},
))

# Simple publish API
await bus.publish("task.started", {"task_id": "123"})
```

### Async Handlers

Both sync and async handlers are supported:

```python
# Sync handler
def sync_handler(event):
    print(event.data)

# Async handler
async def async_handler(event):
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

# Get all events for a task
task_events = bus.get_history(
    filter_fn=lambda e: e.data.get("task_id") == "123"
)
```

### Unsubscribing

```python
# Unsubscribe specific handler
bus.unsubscribe(EventType.TASK_COMPLETED, handler)

# Unsubscribe global handler
bus.unsubscribe_all(global_handler)

# Clear all subscriptions
bus.clear_subscriptions()
```

---

## Built-in Handlers

### ConsoleEventHandler

Human-readable console output with icons:

```python
from agenticflow.events import ConsoleEventHandler

handler = ConsoleEventHandler(
    verbose=True,        # Show detailed event data
    show_timestamp=True, # Include timestamps
    show_source=True,    # Show event source
)

bus.subscribe_all(handler)
```

**Output example:**
```
🚀 [10:30:45] SYSTEM_STARTED - flow: my-workflow
📝 [10:30:45] TASK_CREATED - task_id: task_001
🤖 [10:30:45] AGENT_THINKING - agent: researcher
✅ [10:30:47] TASK_COMPLETED - task_id: task_001
```

**Event icons:**
| Category | Events | Icons |
|----------|--------|-------|
| System | STARTED, STOPPED, ERROR | 🚀 🏁 💥 |
| Tasks | CREATED, STARTED, COMPLETED, FAILED | 📝 ▶️ ✅ ❌ |
| Agents | THINKING, ACTING, RESPONDED | 🧠 ⚡ 💬 |
| Tools | CALLED, RESULT, ERROR | 🛠️ 📤 ⚠️ |

---

### FileEventHandler

Persist events to JSON file:

```python
from agenticflow.events import FileEventHandler

handler = FileEventHandler(
    filepath="events.jsonl",
    append=True,           # Append to existing file
    flush_interval=10,     # Flush every 10 events
)

bus.subscribe_all(handler)
```

**Output format (JSON Lines):**
```jsonl
{"type": "task.started", "timestamp": "2024-12-04T10:30:45Z", "data": {...}}
{"type": "task.completed", "timestamp": "2024-12-04T10:30:47Z", "data": {...}}
```

---

### FilteringEventHandler

Filter events before processing:

```python
from agenticflow.events import FilteringEventHandler

# Only handle high-priority events
handler = FilteringEventHandler(
    handler=my_handler,
    event_types=[EventType.TASK_FAILED, EventType.SYSTEM_ERROR],
)

# Or use custom filter
handler = FilteringEventHandler(
    handler=my_handler,
    filter_fn=lambda e: e.data.get("priority") == "high",
)

bus.subscribe_all(handler)
```

---

### MetricsEventHandler

Collect event metrics:

```python
from agenticflow.events import MetricsEventHandler

metrics = MetricsEventHandler()
bus.subscribe_all(metrics)

# ... run workflow ...

# Get collected metrics
print(metrics.event_counts)   # {EventType.TASK_COMPLETED: 10, ...}
print(metrics.total_events)   # 42
print(metrics.events_per_second)  # 2.5
```

---

## WebSocket Streaming

Real-time event streaming to browser/clients:

```python
from agenticflow.events import WebSocketServer, start_websocket_server

# Start WebSocket server
server = await start_websocket_server(
    event_bus=bus,
    host="localhost",
    port=8765,
)

# Events are automatically broadcast to all connected clients
```

### Client Connection

```javascript
// Browser JavaScript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event:', data.type, data.data);
};
```

### websocket_handler Decorator

Create custom WebSocket handlers:

```python
from agenticflow.events import websocket_handler

@websocket_handler
async def handle_client(websocket, event_bus):
    async for event in event_bus.stream():
        await websocket.send(event.to_json())
```

---

## Event Types

All event types are defined in `agenticflow.core.EventType`:

### System Events
| Event | Description |
|-------|-------------|
| `SYSTEM_STARTED` | System/flow started |
| `SYSTEM_STOPPED` | System/flow stopped |
| `SYSTEM_ERROR` | System-level error |

### Task Events
| Event | Description |
|-------|-------------|
| `TASK_CREATED` | New task created |
| `TASK_SCHEDULED` | Task scheduled for execution |
| `TASK_STARTED` | Task execution started |
| `TASK_COMPLETED` | Task finished successfully |
| `TASK_FAILED` | Task failed with error |
| `TASK_CANCELLED` | Task was cancelled |
| `TASK_BLOCKED` | Task waiting on dependencies |
| `TASK_RETRYING` | Task being retried |

### Agent Events
| Event | Description |
|-------|-------------|
| `AGENT_REGISTERED` | Agent registered with system |
| `AGENT_THINKING` | Agent reasoning |
| `AGENT_ACTING` | Agent executing action |
| `AGENT_RESPONDED` | Agent produced response |
| `AGENT_ERROR` | Agent encountered error |
| `AGENT_INTERRUPTED` | Agent paused for HITL |
| `AGENT_RESUMED` | Agent resumed after HITL |

### Tool Events
| Event | Description |
|-------|-------------|
| `TOOL_REGISTERED` | Tool registered |
| `TOOL_CALLED` | Tool invoked |
| `TOOL_RESULT` | Tool returned result |
| `TOOL_ERROR` | Tool execution failed |
| `TOOL_DEFERRED` | Tool returned deferred result |

### LLM Events (Deep Observability)
| Event | Description |
|-------|-------------|
| `LLM_REQUEST` | Request sent to LLM |
| `LLM_RESPONSE` | Response from LLM |
| `LLM_TOOL_DECISION` | LLM decided to call tools |

### Streaming Events
| Event | Description |
|-------|-------------|
| `STREAM_START` | Streaming started |
| `TOKEN_STREAMED` | Token received |
| `STREAM_END` | Streaming completed |
| `STREAM_ERROR` | Streaming error |

---

## Custom Events

Create and publish custom events:

```python
from agenticflow.schemas.event import Event
from agenticflow.core import EventType

# Using custom event type
event = Event(
    type=EventType.CUSTOM,
    data={
        "custom_type": "user_action",
        "action": "button_click",
        "user_id": "user_123",
    },
)

await bus.publish(event)
```

---

## Exports

```python
from agenticflow.events import (
    # Core
    EventBus,
    # Handlers
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
    # WebSocket
    WebSocketServer,
    start_websocket_server,
    websocket_handler,
)
```
