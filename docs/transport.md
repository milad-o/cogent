# Distributed Transport

Enable cross-process agent communication with pluggable transport backends.

---

## Overview

The **Transport** system allows agents to communicate across process boundaries using message brokers like Redis. This enables:

- **Multi-process deployments** — Scale agents horizontally across multiple servers
- **Distributed workflows** — Coordinate agents running in different processes
- **Loose coupling** — Agents communicate via events, not direct calls
- **Flexibility** — Swap transport backends (local, Redis, custom) without changing agent code

---

## Quick Start

### Local Transport (In-Memory)

For single-process applications or testing:

```python
from agenticflow.reactive import LocalTransport, EventBus
from agenticflow.events import Event

# Create local transport
transport = LocalTransport()
await transport.connect()

# Use with EventBus
bus = EventBus(transport=transport)

# Subscribe to events
async def handler(event: Event):
    print(f"Received: {event.name} - {event.data}")

await transport.subscribe("task.*", handler)

# Publish events
await transport.publish(Event(name="task.created", data={"id": "123"}))
```

### Redis Transport (Distributed)

For production multi-process deployments:

**Install Redis support**:
```bash
uv add agenticflow[redis]
# or
uv add redis>=5.0.0
```

**Use RedisTransport**:
```python
from agenticflow.reactive import RedisTransport

transport = RedisTransport(redis_url="redis://localhost:6379/0")
await transport.connect()

# Same API as LocalTransport
await transport.subscribe("agent.*.response", handler)
await transport.publish(Event(name="agent.task.response", data={"result": "done"}))

await transport.disconnect()
```

---

## Transport Protocol

All transports implement the `Transport` protocol:

```python
from typing import Protocol

class Transport(Protocol):
    """Abstract event transport for distributed systems."""
    
    async def connect(self) -> None:
        """Establish connection to transport backend."""
        ...
    
    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        ...
    
    async def publish(self, event: Event) -> None:
        """Publish event to all matching subscribers."""
        ...
    
    async def subscribe(self, pattern: str, handler: EventHandler) -> str:
        """Subscribe to events matching pattern. Returns subscription ID."""
        ...
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events. Returns True if successful."""
        ...
```

---

## Pattern Matching

Transports support **wildcard patterns** for flexible subscriptions:

| Pattern | Matches | Examples |
|---------|---------|----------|
| `task.created` | Exact match | `task.created` only |
| `task.*` | Single level wildcard | `task.created`, `task.updated`, `task.deleted` |
| `agent.**` | Multi-level wildcard | `agent.task.created`, `agent.task.subtask.done` |
| `**` | Match all events | Any event |

**Examples**:

```python
# Exact match
await transport.subscribe("user.registered", handle_registration)

# Single level wildcard
await transport.subscribe("task.*", handle_task_events)  # task.created, task.updated

# Multi-level wildcard
await transport.subscribe("agent.**", handle_all_agent_events)  # agent.*, agent.task.*, etc.

# Match all
await transport.subscribe("**", log_all_events)
```

---

## Built-in Transports

### LocalTransport

**Use case**: Single-process applications, testing, development

**Features**:
- In-memory `asyncio.Queue`-based
- No external dependencies
- Fast, low latency
- Automatic cleanup on disconnect

**Configuration**:
```python
transport = LocalTransport()
```

**Limitations**:
- ❌ Events don't cross process boundaries
- ❌ No persistence (events lost on restart)

---

### RedisTransport

**Use case**: Production distributed systems, multi-process deployments

**Features**:
- Redis Pub/Sub for event distribution
- Supports multiple processes/servers
- Built-in pattern matching via Redis channels
- Connection pooling

**Configuration**:
```python
transport = RedisTransport(
    redis_url="redis://localhost:6379/0",
    channel_prefix="agenticflow"  # Optional namespace
)
```

**Requirements**:
- Redis server 5.0+ running
- `redis` Python package installed

**Limitations**:
- ⚠️ Redis Pub/Sub is **at-most-once delivery** (no persistence)
- ⚠️ Requires Redis server availability
- ⚠️ Network latency vs LocalTransport

**Production Setup**:

```bash
# Install Redis
brew install redis  # macOS
sudo apt-get install redis-server  # Ubuntu

# Start Redis
redis-server

# Install Python package
uv add agenticflow[redis]
```

---

## Integration with EventBus

Use transport with `EventBus` for automatic event routing:

```python
from agenticflow.reactive import RedisTransport
from agenticflow.events import EventBus

# Create transport
transport = RedisTransport(redis_url="redis://localhost:6379")
await transport.connect()

# Pass to EventBus
bus = EventBus(transport=transport)

# Subscribe via bus
bus.subscribe("task.*", handle_task_events)

# Publish via bus (delegates to transport)
await bus.publish(Event(name="task.created", data={"id": "123"}))
```

**Benefits**:
- Events automatically routed across processes
- Same API as local-only EventBus
- Easy to swap transports (local → Redis) without code changes

---

## Multi-Process Coordination

Example: Coordinating agents across 2 processes

**Process 1 (Worker Agent)**:
```python
# worker.py
from agenticflow import Agent
from agenticflow.reactive import RedisTransport, EventFlow
from agenticflow.events import Event

transport = RedisTransport(redis_url="redis://localhost:6379")
await transport.connect()

worker = Agent(
    name="worker",
    role="Process tasks from queue",
    model="gpt-4",
)

# Subscribe to task events
async def handle_task(event: Event):
    result = await worker.run(f"Process task: {event.data}")
    await transport.publish(Event(
        name="task.completed",
        data={"task_id": event.data["id"], "result": result.response}
    ))

await transport.subscribe("task.created", handle_task)
```

**Process 2 (Coordinator)**:
```python
# coordinator.py
from agenticflow.reactive import RedisTransport
from agenticflow.events import Event

transport = RedisTransport(redis_url="redis://localhost:6379")
await transport.connect()

# Create task
await transport.publish(Event(
    name="task.created",
    data={"id": "123", "description": "Analyze data"}
))

# Wait for completion
async def handle_completion(event: Event):
    print(f"Task {event.data['task_id']} done: {event.data['result']}")

await transport.subscribe("task.completed", handle_completion)
```

**Run both processes**:
```bash
# Terminal 1
python worker.py

# Terminal 2
python coordinator.py
```

---

## Error Handling

Transports raise specific exceptions:

```python
from agenticflow.reactive import (
    TransportError,
    ConnectionError,
    PublishError,
)

try:
    await transport.connect()
except ConnectionError as e:
    logger.error(f"Failed to connect: {e}")

try:
    await transport.publish(event)
except PublishError as e:
    logger.error(f"Failed to publish: {e}")
except TransportError as e:
    logger.error(f"Transport error: {e}")
```

**Best Practices**:
- Always wrap `connect()` in try/except
- Handle `ConnectionError` on startup
- Retry publish on `PublishError` with exponential backoff
- Call `disconnect()` in finally blocks or use context managers

---

## Testing

**Use LocalTransport for tests** (fast, no external dependencies):

```python
import pytest
from agenticflow.reactive import LocalTransport
from agenticflow.events import Event

@pytest.mark.asyncio
async def test_event_delivery():
    transport = LocalTransport()
    await transport.connect()
    
    received = []
    
    async def handler(event: Event):
        received.append(event)
    
    await transport.subscribe("test.*", handler)
    await transport.publish(Event(name="test.event", data={"value": 123}))
    
    await asyncio.sleep(0.1)  # Let dispatcher process
    
    assert len(received) == 1
    assert received[0].name == "test.event"
    assert received[0].data["value"] == 123
    
    await transport.disconnect()
```

**Integration tests with Redis** (requires Redis server):

```python
@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.asyncio
async def test_redis_transport():
    transport = RedisTransport(redis_url="redis://localhost:6379/15")
    await transport.connect()
    
    # Test event flow
    ...
    
    await transport.disconnect()
```

---

## Advanced Usage

### Multiple Subscribers

Multiple handlers can subscribe to the same pattern:

```python
async def logger(event: Event):
    print(f"LOG: {event.name}")

async def metrics(event: Event):
    print(f"METRIC: {event.name}")

await transport.subscribe("task.*", logger)
await transport.subscribe("task.*", metrics)

# Both handlers receive the event
await transport.publish(Event(name="task.created", data={}))
```

### Subscription Management

Unsubscribe when no longer needed:

```python
# Subscribe
sub_id = await transport.subscribe("task.*", handler)

# Later, unsubscribe
success = await transport.unsubscribe(sub_id)
assert success is True
```

### Context Managers (Future Enhancement)

```python
# Planned for future release
async with RedisTransport(redis_url="...") as transport:
    await transport.subscribe("task.*", handler)
    # Automatically disconnects on exit
```

---

## Custom Transport Backends

Implement the `Transport` protocol for custom backends:

```python
class NATSTransport:
    """NATS JetStream transport (example)."""
    
    async def connect(self) -> None:
        self._client = await nats.connect(self._url)
        self._js = self._client.jetstream()
    
    async def publish(self, event: Event) -> None:
        await self._js.publish(
            subject=event.name,
            payload=event.model_dump_json().encode(),
        )
    
    async def subscribe(self, pattern: str, handler: EventHandler) -> str:
        # Convert pattern to NATS subject
        subject = pattern.replace("**", ">").replace("*", "*")
        
        async def nats_handler(msg):
            event = Event.model_validate_json(msg.data)
            await handler(event)
        
        sub = await self._js.subscribe(subject, cb=nats_handler)
        return str(id(sub))
    
    # ... implement rest of protocol
```

---

## Comparison

| Feature | LocalTransport | RedisTransport | Custom |
|---------|---------------|----------------|--------|
| **Single Process** | ✅ Yes | ⚠️ Works but overkill | Depends |
| **Multi Process** | ❌ No | ✅ Yes | Depends |
| **Persistence** | ❌ No | ❌ No (Pub/Sub) | Depends |
| **Dependencies** | ✅ None | ⚠️ Redis server + package | Depends |
| **Latency** | ⚠️ Lowest | ⚠️ Network overhead | Depends |
| **Pattern Matching** | ✅ Yes | ✅ Yes | Implement |
| **Best For** | Development, testing | Production distributed | Special needs |

---

## Next Steps

- See [examples/reactive/distributed_transport_demo.py](../examples/reactive/distributed_transport_demo.py) for runnable demos
- Explore [Reactive Flows](./reactive.md) for using transport in event-driven workflows
- Check [EventBus documentation](./events.md) for event routing patterns
- Consider [Checkpointing](./checkpointer.md) for fault-tolerant distributed flows

---

## Future Enhancements

Planned features:
- **NATS transport** — JetStream for persistent event streams
- **Context managers** — `async with transport:` syntax
- **Backpressure** — Rate limiting and flow control
- **Dead Letter Queue** — Handle failed event delivery
- **Metrics** — Built-in observability for transport health
- **Encryption** — TLS/mTLS for secure event transmission
