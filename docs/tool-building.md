# Building Custom Tools

**Create production-ready tools for cogent agents.**

## Overview

Tools are functions that agents can call to interact with the world — APIs, databases, files, web searches, and more. Cogent provides two ways to add tools:

1. **@tool decorator** — Create custom tools from any function (this guide)
2. **Capabilities** — Use pre-built tool classes like WebSearch, FileSystem, CodeSandbox

**Capabilities are classes that provide tools.** They handle setup, state management, and expose multiple related tools:

```python
from cogent import Agent
from cogent.capabilities import WebSearch, FileSystem, CodeSandbox

agent = Agent(
    model="gpt-4o-mini",
    capabilities=[
        WebSearch(),                          # Provides: search, fetch_url
        FileSystem(allowed_paths=["./data"]), # Provides: read, write, list
        CodeSandbox(),                        # Provides: execute_python
    ],
)
```

> **See [Capabilities](capabilities.md) for 12+ production-ready capabilities** — KnowledgeGraph, Browser, PDF, Shell, MCP, and more.

For custom logic, use the `@tool` decorator:

## Quick Start

```python
from cogent import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: The city name to get weather for.
    
    Returns:
        Weather information as a string.
    """
    # Your implementation
    return f"Weather in {city}: 72°F, sunny"
```

**That's it!** The `@tool` decorator automatically:
- Extracts function signature → JSON schema
- Parses docstring → tool description
- Handles sync/async execution
- Provides error handling

## The @tool Decorator

```python
from cogent import tool

@tool
def my_tool(param1: str, param2: int = 10) -> str:
    """Tool description goes here.
    
    Args:
        param1: Description of param1.
        param2: Description of param2 (optional).
    
    Returns:
        Description of return value.
    """
    return f"Result: {param1} x {param2}"
```

### What Gets Generated

```json
{
  "type": "function",
  "function": {
    "name": "my_tool",
    "description": "Tool description goes here.",
    "parameters": {
      "type": "object",
      "properties": {
        "param1": {
          "type": "string",
          "description": "Description of param1."
        },
        "param2": {
          "type": "integer",
          "description": "Description of param2 (optional).",
          "default": 10
        }
      },
      "required": ["param1"]
    }
  }
}
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

### Decorator Options

```python
@tool(
    name="web_search",           # Override function name
    description="Search the web",  # Override docstring
    return_direct=True,          # Return result directly to user
    cache=True,                  # Enable semantic caching (requires agent.cache)
)
def search(query: str, max_results: int = 10) -> str:
    """Search implementation."""
    return f"Found {max_results} results for: {query}"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | `str` | Function name | Override the tool name |
| `description` | `str` | Docstring | Override the tool description |
| `return_direct` | `bool` | `False` | Return result directly to user without LLM processing |
| `cache` | `bool` | `False` | Enable automatic semantic caching (see [Semantic Caching](#semantic-caching)) |

## Async Tools

Most production tools are async (API calls, database queries, file I/O):

```python
from cogent import tool
import httpx

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL.
    
    Args:
        url: The URL to fetch.
    
    Returns:
        The response text.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

**The executor handles both sync and async tools automatically.**

## Type Hints

Use type hints for automatic schema generation:

```python
from cogent import tool
from typing import Literal

@tool
def search(
    query: str,
    engine: Literal["google", "bing", "duckduckgo"] = "duckduckgo",
    max_results: int = 10,
) -> list[dict[str, str]]:
    """Search the web.
    
    Args:
        query: Search query string.
        engine: Search engine to use.
        max_results: Maximum number of results.
    
    Returns:
        List of search results with title and URL.
    """
    # Implementation
    return [{"title": "Result 1", "url": "https://..."}]
```

**Supported types:**
- `str`, `int`, `float`, `bool`
- `list[T]`, `dict[K, V]`
- `Literal["option1", "option2"]`
- `Optional[T]` or `T | None`
- Pydantic models

## Error Handling

Always handle errors gracefully:

```python
from cogent import tool
import httpx

@tool
async def safe_fetch(url: str) -> str:
    """Safely fetch URL with error handling."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.TimeoutException:
        return f"Error: Request to {url} timed out after 10 seconds"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} from {url}"
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"
```

**Best practices:**
- Catch specific exceptions first
- Provide helpful error messages
- Return error as string (don't raise in tools)
- Log errors for debugging

## Tool Composition

Combine related tools into a capability class:

```python
from cogent import tool

class Calculator:
    """Calculator with memory."""
    
    def __init__(self):
        self.memory: float = 0.0
    
    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.memory = result
        return result
    
    @tool
    def recall(self) -> float:
        """Recall last result from memory."""
        return self.memory
```

**Pattern:** Related tools in a class share state and configuration. This is exactly how capabilities work — see [Capabilities](capabilities.md).

## Context Injection

Access agent context in tools:

```python
from cogent import tool
from cogent.core.context import RunContext

@tool
async def get_user_data(
    user_id: int,
    ctx: RunContext,  # Auto-injected
) -> dict:
    """Get user data with context."""
    # Access agent context
    session_id = ctx.session_id
    user = ctx.user_id
    
    # Use context for logging, auth, etc
    logger.info(f"User {user} requesting data for {user_id} in session {session_id}")
    
    return await fetch_user(user_id)
```

**Context fields:**
- `session_id` — Current session identifier
- `user_id` — User making the request
- `metadata` — Custom metadata dict
- `agent` — Reference to the agent instance

## Registering Tools with Agents

```python
from cogent import Agent
from cogent.capabilities import WebSearch, FileSystem

# Custom tools + capabilities together
agent = Agent(
    model="gpt-4o-mini",
    tools=[get_weather, my_custom_tool],  # Your @tool functions
    capabilities=[WebSearch(), FileSystem()],  # Pre-built tool providers
)
```

All tools (custom + from capabilities) are available to the agent.

## Testing Tools

Use pytest for tool testing:

```python
import pytest
from cogent import tool

@tool
async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@pytest.mark.asyncio
async def test_divide_success():
    result = await divide(10, 2)
    assert result == 5.0

@pytest.mark.asyncio
async def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        await divide(10, 0)
```

See comprehensive testing guide: [docs/testing.md](testing.md)

## Tool Patterns

### Pattern 1: Retry with Exponential Backoff

```python
import asyncio
from cogent import tool

@tool
async def resilient_api_call(url: str, max_retries: int = 3) -> str:
    """API call with retries."""
    for attempt in range(max_retries):
        try:
            return await fetch(url)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Failed after {max_retries} attempts: {e}"
            await asyncio.sleep(2 ** attempt)
```

### Pattern 2: Rate Limiting

```python
import asyncio
from cogent import tool

class RateLimitedAPI:
    def __init__(self, calls_per_second: int = 10):
        self.delay = 1.0 / calls_per_second
        self.last_call = 0.0
    
    @tool
    async def call_api(self, endpoint: str) -> str:
        """Call API with rate limiting."""
        now = asyncio.get_event_loop().time()
        wait_time = max(0, self.delay - (now - self.last_call))
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        self.last_call = asyncio.get_event_loop().time()
        return await self._make_request(endpoint)
```

### Pattern 3: Caching

#### Semantic Caching (Recommended)

Use `@tool(cache=True)` for automatic semantic caching. Similar queries return cached results:

```python
from cogent import Agent, tool

@tool(cache=True)
async def search_products(query: str) -> str:
    """Search products in the catalog.
    
    Args:
        query: Search query for products.
    
    Returns:
        Product search results.
    """
    # Expensive API call - cached semantically
    return await product_api.search(query)

# Agent must have cache enabled
agent = Agent(
    model="gpt-4o-mini",
    tools=[search_products],
    cache=True,  # Required for @tool(cache=True)
)

# First call executes the tool
await agent.run("Find running shoes")

# Similar query hits cache (semantic match)
await agent.run("Show me running sneakers")  # Cache hit!
```

**How it works:**

1. Tool input is embedded using the agent's embedding model
2. Cache checks for semantically similar previous calls
3. If similarity exceeds threshold, cached result is returned
4. Otherwise, tool executes and result is stored

**Requirements:**

- Agent must have `cache=True` enabled
- An embedding model must be configured (or uses default)
- Tool must have `cache=True` in decorator

#### Simple LRU Caching

For exact-match caching (same input = same output):

```python
from cogent import tool
from functools import lru_cache

@tool
@lru_cache(maxsize=100)
def cached_computation(input: str) -> str:
    """Expensive computation with exact-match caching."""
    # Cached result for identical input only
    return expensive_operation(input)
```

> [!TIP]
> Use `@tool(cache=True)` for semantic similarity matching, `@lru_cache` for exact string matching.

### Pattern 4: Validation

```python
from cogent import tool
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(ge=1, le=100)

@tool
async def validated_search(params: SearchParams) -> list:
    """Search with validated parameters."""
    # Pydantic ensures constraints
    return await search(params.query, params.max_results)
```

## Tool Execution

When an agent has multiple tools, they execute in **parallel by default** via NativeExecutor:

```python
from cogent import Agent

agent = Agent(
    model="gpt-4o-mini",
    tools=[fetch_weather, fetch_news, fetch_stock]
)

# If LLM requests multiple tools in one turn, they run concurrently
result = await agent.run("Get weather, news, and stock data")
# 3 tools × 0.5s each = ~0.5s total (parallel via asyncio.gather)
```

**Execution behavior:**
- **Parallel**: When LLM requests multiple tools in one turn
- **Sequential**: When LLM naturally calls tools one at a time across turns
- **LLM decides**: Based on task requirements and your prompt

**Configuration:**

```python
from cogent.executors import NativeExecutor

executor = NativeExecutor(
    agent,
    max_tool_calls_per_turn=50,  # Max tools per LLM response
    max_concurrent_tools=20,      # Tune for external API rate limits
    resilience=True               # Auto-retry on LLM rate limits
)
```

### Standalone Execution

For quick tasks without creating an Agent:

```python
from cogent.executors import run

result = await run(
    "Search for Python tutorials",
    tools=[search],
    model="gpt-4o-mini",
)
```

## Best Practices

1. **Use type hints** — Enable automatic schema generation
2. **Write clear docstrings** — Becomes tool description for LLM
3. **Handle errors gracefully** — Return error strings, don't raise
4. **Make tools atomic** — One clear purpose per tool
5. **Use async for I/O** — All network/DB/file operations
6. **Validate inputs** — Pydantic models for complex inputs
7. **Add retries for resilience** — External calls can fail
8. **Log for observability** — Track tool usage and errors
9. **Test thoroughly** — Unit tests for all tools
10. **Document return types** — Help LLM use results correctly

## Common Pitfalls

| Issue | Problem | Solution |
|-------|---------|----------|
| Missing type hints | No schema generated | Add type hints to all params |
| Unclear docstring | LLM misuses tool | Write clear, specific descriptions |
| Raising exceptions | Agent execution halts | Return error strings instead |
| Blocking I/O | Poor performance | Use async for all I/O operations |
| No error handling | Crashes on failures | Wrap in try/except |
| Too complex | LLM struggles to use | Split into multiple simpler tools |

## Tool Registry

Manage collections of tools:

```python
from cogent.tools import ToolRegistry

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
from cogent.tools import create_tool_from_function

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

## Deferred Tools

For operations requiring human approval or async completion:

### DeferredResult

```python
from cogent.tools import DeferredResult, DeferredStatus

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
from cogent.tools import DeferredManager

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
from cogent.tools import DeferredStatus

DeferredStatus.PENDING    # Waiting for action
DeferredStatus.APPROVED   # Approved, ready to execute
DeferredStatus.REJECTED   # Rejected
DeferredStatus.COMPLETED  # Execution completed
DeferredStatus.FAILED     # Execution failed
```

### DeferredWaiter

Wait for deferred completion:

```python
from cogent.tools import DeferredWaiter

waiter = DeferredWaiter(manager)

# Wait for result (with timeout)
result = await waiter.wait(result_id, timeout=300)

# Wait for multiple
results = await waiter.wait_all([id1, id2, id3])
```

### DeferredRetry

Auto-retry failed deferred operations:

```python
from cogent.tools import DeferredRetry

retry = DeferredRetry(
    manager=manager,
    max_attempts=3,
    backoff="exponential",
)

result = await retry.execute(result_id)
```

### is_deferred Helper

```python
from cogent.tools import is_deferred

result = tool_call()

if is_deferred(result):
    # Handle deferred result
    result_id = await manager.register(result)
else:
    # Handle immediate result
    print(result)
```

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

## Further Reading

- [Capabilities](capabilities.md) — 12+ production-ready tool classes
- [Agent Configuration](agent.md) — Using tools with agents
- [Resilience](resilience.md) — Error handling and retry policies
