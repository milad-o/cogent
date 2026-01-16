# Core Module

The `agenticflow.core` module provides foundational types, enums, utilities, and dependency injection used throughout the framework.

## Overview

The core module defines:
- Enums for status types, event types, and roles
- Native message types compatible with all LLM providers
- Utility functions for IDs, timestamps, etc.
- **RunContext** for dependency injection (tools and interceptors)
- **Reactive utilities** for event-driven systems (idempotency, retries, delays)

```python
from agenticflow.core import (
    # Enums
    TaskStatus,
    AgentStatus,
    EventType,
    Priority,
    AgentRole,
    # Context
    RunContext,
    EMPTY_CONTEXT,
    # Utilities
    generate_id,
    now_utc,
    # Reactive utilities
    IdempotencyGuard,
    RetryBudget,
    emit_later,
    jittered_delay,
    Stopwatch,
)
```

## Enums

### TaskStatus

Task lifecycle states:

```python
from agenticflow.core import TaskStatus

status = TaskStatus.RUNNING

# Check state categories
status.is_terminal()  # COMPLETED, FAILED, CANCELLED
status.is_active()    # RUNNING, SPAWNING
```

| Status | Description |
|--------|-------------|
| `PENDING` | Task created, not yet scheduled |
| `SCHEDULED` | Task scheduled for execution |
| `BLOCKED` | Task waiting on dependencies |
| `RUNNING` | Task actively executing |
| `SPAWNING` | Task creating subtasks |
| `COMPLETED` | Task finished successfully |
| `FAILED` | Task failed with error |
| `CANCELLED` | Task was cancelled |

---

### AgentStatus

Agent lifecycle states:

```python
from agenticflow.core import AgentStatus

status = AgentStatus.THINKING

# Check state categories
status.is_available()  # Can accept new work (IDLE)
status.is_working()    # Currently working (THINKING, ACTING)
```

| Status | Description |
|--------|-------------|
| `IDLE` | Agent ready for new work |
| `THINKING` | Agent reasoning/planning |
| `ACTING` | Agent executing tools |
| `WAITING` | Agent waiting for response |
| `ERROR` | Agent in error state |
| `OFFLINE` | Agent unavailable |

---

### EventType

System events for observability and coordination:

```python
from agenticflow.core import EventType

event = EventType.TASK_COMPLETED
event.category  # "task"
```

**Categories:**

| Category | Examples |
|----------|----------|
| `system` | SYSTEM_STARTED, SYSTEM_STOPPED, SYSTEM_ERROR |
| `task` | TASK_CREATED, TASK_STARTED, TASK_COMPLETED, TASK_FAILED |
| `subtask` | SUBTASK_SPAWNED, SUBTASK_COMPLETED |
| `agent` | AGENT_THINKING, AGENT_ACTING, AGENT_RESPONDED |
| `tool` | TOOL_CALLED, TOOL_RESULT, TOOL_ERROR |
| `llm` | LLM_REQUEST, LLM_RESPONSE, LLM_TOOL_DECISION |
| `stream` | STREAM_START, TOKEN_STREAMED, STREAM_END |
| `plan` | PLAN_CREATED, PLAN_STEP_STARTED, PLAN_STEP_COMPLETED |
| `message` | MESSAGE_SENT, MESSAGE_RECEIVED |

---

### Priority

Task priority levels (comparable):

```python
from agenticflow.core import Priority

# Priorities are comparable
Priority.HIGH > Priority.NORMAL  # True
Priority.LOW < Priority.CRITICAL  # True
```

| Priority | Value | Description |
|----------|-------|-------------|
| `LOW` | 1 | Background tasks |
| `NORMAL` | 2 | Standard priority |
| `HIGH` | 3 | Important tasks |
| `CRITICAL` | 4 | Urgent tasks |

---

### AgentRole

Agent roles define capabilities:

```python
from agenticflow.core import AgentRole

role = AgentRole.SUPERVISOR
role.can_finish     # True
role.can_delegate   # True
role.can_use_tools  # False
```

| Role | can_finish | can_delegate | can_use_tools |
|------|------------|--------------|---------------|
| `WORKER` | ❌ | ❌ | ✅ |
| `SUPERVISOR` | ✅ | ✅ | ❌ |
| `AUTONOMOUS` | ✅ | ❌ | ✅ |
| `REVIEWER` | ✅ | ❌ | ❌ |

---

## Messages

Native message types compatible with OpenAI/Anthropic/etc. APIs:

### SystemMessage

```python
from agenticflow.core.messages import SystemMessage

msg = SystemMessage("You are a helpful assistant.")
msg.to_dict()  # {"role": "system", "content": "..."}
```

### HumanMessage

```python
from agenticflow.core.messages import HumanMessage

msg = HumanMessage("Hello, how are you?")
msg.to_dict()  # {"role": "user", "content": "..."}
```

### AIMessage

```python
from agenticflow.core.messages import AIMessage

# Text response
msg = AIMessage(content="I'm doing well!")

# Response with tool calls
msg = AIMessage(
    content="",
    tool_calls=[
        {"id": "call_1", "name": "search", "args": {"query": "weather"}}
    ],
)
msg.to_dict()  # Includes properly formatted tool_calls
```

### ToolMessage

```python
from agenticflow.core.messages import ToolMessage

msg = ToolMessage(
    content='{"temperature": 72}',
    tool_call_id="call_1",
)
```

### Helper Functions

```python
from agenticflow.core.messages import (
    messages_to_dict,
    parse_openai_response,
)

# Convert message list for API calls
messages = [SystemMessage("..."), HumanMessage("...")]
api_messages = messages_to_dict(messages)

# Parse OpenAI response into AIMessage
ai_msg = parse_openai_response(openai_response)
```

---

## Context (Dependency Injection)

### RunContext

Base class for invocation-scoped context that provides dependency injection for tools and interceptors.

```python
from dataclasses import dataclass
from agenticflow import Agent, tool
from agenticflow.core import RunContext

@dataclass
class AppContext(RunContext):
    user_id: str
    db: Any  # Your database connection
    api_key: str

@tool
def get_user_data(ctx: RunContext) -> str:
    """Get data for the current user."""
    user = ctx.db.get_user(ctx.user_id)
    return f"User: {user.name}"

agent = Agent(name="assistant", model=model, tools=[get_user_data])

# Pass context at invocation time
result = await agent.run(
    "Get my profile data",
    context=AppContext(user_id="123", db=db, api_key=key),
)
```

**Key Features:**
- Type-safe context data passed to tools and interceptors
- No global state — context scoped to single invocation
- Access via `ctx: RunContext` parameter in tools
- Available in interceptors via `InterceptContext.run_context`

**Methods:**
- `get(key, default)` — Get metadata value by key
- `with_metadata(**kwargs)` — Create new context with additional metadata

---

## Reactive Utilities

Event-driven utilities for building robust reactive systems.

### IdempotencyGuard

In-memory idempotency guard to ensure side-effects execute only once per key:

```python
from agenticflow.core import IdempotencyGuard

guard = IdempotencyGuard()

async def process_event(event_id: str, data: dict):
    if not await guard.claim(event_id):
        return  # Already processed
    
    # Process event exactly once
    await do_work(data)
```

**Methods:**
- `claim(key: str) -> bool` — Atomically claim a key (returns True if first time)
- `run_once(key: str, fn: Callable) -> tuple[bool, Any]` — Run coroutine exactly once per key

**Note:** Process-local memory. For distributed systems, back with Redis/database.

### RetryBudget

Bounded retry tracker for exponential backoff and retry policies:

```python
from agenticflow.core import RetryBudget

budget = RetryBudget.in_memory(max_attempts=3)

async def handle_with_retries(task_id: str):
    attempt = budget.next_attempt(task_id)
    
    if attempt >= 3:
        # Escalate to error handler
        await escalate_error(task_id)
        return
    
    # Try again
    await retry_task(task_id)
```

**Methods:**
- `in_memory(max_attempts: int) -> RetryBudget` — Create in-memory tracker
- `next_attempt(key: str) -> int` — Increment and return attempt count (0-based)
- `can_retry(key: str) -> bool` — Check if more retries available

### emit_later

Schedule delayed event emission:

```python
from agenticflow.core import emit_later

# In a ReactiveFlow
async def handle_timeout(event, ctx):
    # Schedule a timeout event
    ctx.flow.spawn(
        emit_later(
            flow=ctx.flow,
            delay_seconds=30.0,
            event_name="task.timeout",
            data={"task_id": event.data["task_id"]},
        )
    )
```

### jittered_delay

Calculate exponential backoff with jitter:

```python
from agenticflow.core import jittered_delay
import random

attempt = 2
base_delay = 2 ** attempt  # 4 seconds
jitter = random.uniform(-1, 1)

delay = jittered_delay(
    base_seconds=base_delay,
    jitter_seconds=jitter,
    min_seconds=1.0,
    max_seconds=60.0,
)

await asyncio.sleep(delay)
```

### Stopwatch

Performance timing helper:

```python
from agenticflow.core import Stopwatch

stopwatch = Stopwatch()

await do_work()

elapsed = stopwatch.elapsed_s
print(f"Work completed in {elapsed:.2f}s")
```

---

## Utilities

### generate_id

Generate unique identifiers:

```python
from agenticflow.core import generate_id

task_id = generate_id()  # "task_a1b2c3d4"
agent_id = generate_id(prefix="agent")  # "agent_e5f6g7h8"
```

### Timestamps

```python
from agenticflow.core import now_utc, now_local, to_local, format_timestamp

# Get current time
utc_now = now_utc()      # datetime in UTC
local_now = now_local()  # datetime in local timezone

# Convert UTC to local
local_time = to_local(utc_now)

# Format for display
formatted = format_timestamp(utc_now)  # "2024-12-04 10:30:45 UTC"
```

---

## Exports

```python
from agenticflow.core import (
    # Enums
    TaskStatus,
    AgentStatus,
    EventType,
    Priority,
    AgentRole,
    # Context
    RunContext,
    EMPTY_CONTEXT,
    # Utilities
    generate_id,
    now_utc,
    now_local,
    to_local,
    format_timestamp,
    # Reactive utilities
    IdempotencyGuard,
    RetryBudget,
    emit_later,
    jittered_delay,
    Stopwatch,
)

from agenticflow.core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    messages_to_dict,
    parse_openai_response,
)
```
