# Tools Module

The `agenticflow.tools` module provides tool creation, registration, and deferred execution for agent capabilities.

## Overview

Tools are functions that agents can call to interact with the world:

```python
from agenticflow import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = Agent(
    name="assistant",
    model=model,
    tools=[search],
)

result = await agent.run("Search for AI news")
```

---

## Creating Tools

### @tool Decorator

The simplest way to create tools:

```python
from agenticflow import tool

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")
    
    Returns:
        The result of the calculation
    """
    return eval(expression)

# Docstring becomes tool description
# Type hints become parameter schema
# Return type is included in description for LLM visibility
```

### Return Type Information

The `@tool` decorator automatically extracts return type information and includes it in the tool description. This helps the LLM understand what output to expect:

```python
@tool
def get_weather(city: str) -> dict[str, int]:
    """Get weather data for a city.
    
    Args:
        city: City name to query.
    
    Returns:
        A dictionary with temp, humidity, and wind_speed.
    """
    return {"temp": 75, "humidity": 45, "wind_speed": 10}

# LLM sees this description:
# "Get weather data for a city. Returns: dict[str, int] - A dictionary with temp, humidity, and wind_speed."

# Access the return info directly:
print(get_weather.return_info)
# Output: "dict[str, int] - A dictionary with temp, humidity, and wind_speed."
```

**What gets extracted:**

| Source | Example | Result |
|--------|---------|--------|
| Return type annotation | `-> str` | `"str"` |
| Generic types | `-> dict[str, int]` | `"dict[str, int]"` |
| Optional types | `-> str \| None` | `"str \| None"` |
| Docstring Returns section | `Returns: The result.` | `"The result."` |
| Both combined | Type + docstring | `"dict[str, int] - A dictionary with..."` |

> [!TIP]
> Always include a `Returns:` section in your docstrings to give the LLM context about the output format.

### With Options

```python
@tool(
    name="web_search",           # Override function name
    description="Search the web",  # Override docstring
    return_direct=True,          # Return result directly to user
)
def search(query: str, max_results: int = 10) -> str:
    """Search implementation."""
    return f"Found {max_results} results for: {query}"
```

### Async Tools

```python
@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

### Tools with Context

Access run context in tools:

```python
from agenticflow import tool, RunContext

@tool
def get_user_data(ctx: RunContext) -> str:
    """Get data for the current user."""
    user_id = ctx.metadata.get("user_id")
    return f"Data for user: {user_id}"

# Pass context when running agent
result = await agent.run(
    "Get my data",
    context=RunContext(metadata={"user_id": "123"}),
)
```

---

## Tool Registry

Manage collections of tools:

```python
from agenticflow.tools import ToolRegistry

registry = ToolRegistry()

# Register tools
registry.register(search_tool)
registry.register(calculate_tool)
registry.register(fetch_tool)

# Get all tools
all_tools = registry.get_all()

# Get by name
search = registry.get("search")

# Check existence
has_search = registry.has("search")

# List names
names = registry.list_names()  # ["search", "calculate", "fetch"]
```

### From Functions

```python
from agenticflow.tools import create_tool_from_function

def my_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

tool = create_tool_from_function(my_function)
registry.register(tool)
```

### Categories

Organize tools by category:

```python
# Register with category
registry.register(search_tool, category="web")
registry.register(fetch_tool, category="web")
registry.register(calculate_tool, category="math")

# Get by category
web_tools = registry.get_by_category("web")
```

---

## Deferred Tools

For operations requiring human approval or async completion:

### DeferredResult

```python
from agenticflow.tools import DeferredResult, DeferredStatus

@tool
def send_email(to: str, subject: str, body: str) -> DeferredResult:
    """Send an email (requires approval)."""
    return DeferredResult(
        status=DeferredStatus.PENDING,
        message="Email pending approval",
        data={"to": to, "subject": subject},
    )
```

### DeferredManager

Manage deferred operations:

```python
from agenticflow.tools import DeferredManager

manager = DeferredManager()

# Register deferred result
result_id = await manager.register(deferred_result)

# Check status
status = await manager.status(result_id)

# Approve/reject
await manager.approve(result_id, approver="admin")
await manager.reject(result_id, reason="Not allowed")

# Get result after approval
final_result = await manager.get_result(result_id)
```

### DeferredStatus

```python
from agenticflow.tools import DeferredStatus

DeferredStatus.PENDING    # Waiting for action
DeferredStatus.APPROVED   # Approved, ready to execute
DeferredStatus.REJECTED   # Rejected
DeferredStatus.COMPLETED  # Execution completed
DeferredStatus.FAILED     # Execution failed
```

### DeferredWaiter

Wait for deferred completion:

```python
from agenticflow.tools import DeferredWaiter

waiter = DeferredWaiter(manager)

# Wait for result (with timeout)
result = await waiter.wait(result_id, timeout=300)

# Wait for multiple
results = await waiter.wait_all([id1, id2, id3])
```

### DeferredRetry

Auto-retry failed deferred operations:

```python
from agenticflow.tools import DeferredRetry

retry = DeferredRetry(
    manager=manager,
    max_attempts=3,
    backoff="exponential",
)

result = await retry.execute(result_id)
```

### is_deferred Helper

```python
from agenticflow.tools import is_deferred

result = tool_call()

if is_deferred(result):
    # Handle deferred result
    result_id = await manager.register(result)
else:
    # Handle immediate result
    print(result)
```

---

## Tool Schemas

Tools automatically generate JSON schemas from type hints:

```python
@tool
def create_event(
    title: str,
    date: str,
    attendees: list[str] | None = None,
    priority: int = 1,
) -> str:
    """Create a calendar event."""
    ...

# Generated schema:
# {
#     "name": "create_event",
#     "description": "Create a calendar event.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "title": {"type": "string"},
#             "date": {"type": "string"},
#             "attendees": {"type": "array", "items": {"type": "string"}},
#             "priority": {"type": "integer", "default": 1}
#         },
#         "required": ["title", "date"]
#     }
# }
```

---

## Complex Types

### Pydantic Models

```python
from pydantic import BaseModel

class EmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    cc: list[str] = []

@tool
def send_email(request: EmailRequest) -> str:
    """Send an email."""
    return f"Sent to {request.to}"
```

### Enum Parameters

```python
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@tool
def create_task(title: str, priority: Priority = Priority.MEDIUM) -> str:
    """Create a task with priority."""
    return f"Created: {title} ({priority.value})"
```

---

## Agent Tool Integration

### Adding Tools to Agents

```python
from agenticflow import Agent

agent = Agent(
    name="assistant",
    model=model,
    tools=[search, calculate, fetch],  # List of tools
)
```

### Dynamic Tool Access

```python
# Agent can see which tools are available
for tool in agent.tools:
    print(f"{tool.name}: {tool.description}")
```

### Tool Results in Responses

When agents use tools, results are available in the response:

```python
result = await agent.run("Calculate 2 + 2")

# Access tool calls made
for call in result.tool_calls:
    print(f"Tool: {call.name}")
    print(f"Args: {call.args}")
    print(f"Result: {call.result}")
```

---

## Error Handling

### Tool Errors

```python
@tool
def risky_operation(data: str) -> str:
    """Perform risky operation."""
    if not data:
        raise ValueError("Data required")
    return process(data)

# Errors are caught and returned to agent for handling
```

### Graceful Degradation

```python
@tool
def search_with_fallback(query: str) -> str:
    """Search with fallback."""
    try:
        return primary_search(query)
    except Exception:
        return fallback_search(query)
```

---

## API Reference

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@tool` | Create a tool from a function |

### Core Classes

| Class | Description |
|-------|-------------|
| `ToolRegistry` | Manage tool collections |
| `BaseTool` | Base class for tools |

### Deferred Execution

| Class | Description |
|-------|-------------|
| `DeferredResult` | Result requiring async completion |
| `DeferredStatus` | Status of deferred operation |
| `DeferredManager` | Manage deferred results |
| `DeferredWaiter` | Wait for deferred completion |
| `DeferredRetry` | Retry failed operations |

### Utility Functions

| Function | Description |
|----------|-------------|
| `create_tool_from_function(fn)` | Create tool from function |
| `is_deferred(result)` | Check if result is deferred |
