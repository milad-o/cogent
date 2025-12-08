# Events Module

> **Note:** The events module has been consolidated into `agenticflow.observability`.
> Import from `agenticflow.observability` instead of `agenticflow.events`.
> See [Observability Module](observability.md) for the complete documentation.

## Migration Guide

The events functionality is now part of the unified observability module:

```python
# Old (deprecated)
from agenticflow.events import EventBus, ConsoleEventHandler

# New (recommended)
from agenticflow.observability import EventBus, Event, ConsoleEventHandler

bus = EventBus()

# Subscribe to specific events
bus.subscribe(EventType.TASK_COMPLETED, handle_completion)

# Subscribe to all events (for logging)
bus.subscribe_all(ConsoleEventHandler())

# Publish events
await bus.publish(Event(
    type=EventType.TASK_STARTED,
    data={"task_id": "123"},
))
```

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
