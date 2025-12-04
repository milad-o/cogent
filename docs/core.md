# Core Module

The `agenticflow.core` module provides foundational types, enums, and utilities used throughout the framework.

## Overview

The core module defines:
- Enums for status types, event types, and roles
- Native message types compatible with all LLM providers
- Utility functions for IDs, timestamps, etc.

```python
from agenticflow.core import (
    TaskStatus,
    AgentStatus,
    EventType,
    Priority,
    AgentRole,
    generate_id,
    now_utc,
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
    # Utilities
    generate_id,
    now_utc,
    now_local,
    to_local,
    format_timestamp,
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
