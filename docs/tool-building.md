# Building Custom Tools

**Create production-ready tools for cogent agents.**

## Overview

Tools extend agent capabilities by providing access to external systems, APIs, databases, and custom logic. Cogent makes tool creation simple with the `@tool` decorator while supporting advanced patterns for production use.

**This guide uses cogent's 8+ production capabilities as examples.**

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

## Example: HTTPClient Capability

Full-featured HTTP client from Phase 4.4:

```python
from cogent import tool
import httpx
from typing import Any

class HTTPClient:
    """HTTP client with retries, timeouts, and streaming."""
    
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
    
    @tool
    async def http_get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP GET request.
        
        Args:
            url: URL to request.
            headers: Optional HTTP headers.
        
        Returns:
            Response with status, headers, and body.
        """
        full_url = f"{self.base_url}{url}" if self.base_url else url
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(full_url, headers=headers)
                    return {
                        "status": response.status_code,
                        "headers": dict(response.headers),
                        "body": response.text,
                    }
                except httpx.TimeoutException:
                    if attempt == self.max_retries - 1:
                        return {"error": "Request timed out after retries"}
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    return {"error": str(e)}
    
    @tool
    async def http_post(
        self,
        url: str,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP POST request with data or JSON."""
        # Similar implementation
        pass
```

**Key features:**
- Configurable timeouts and retries
- Exponential backoff
- Multiple HTTP methods
- Structured responses

See full implementation: [src/cogent/capabilities/http_client.py](../src/cogent/capabilities/http_client.py)

## Example: Database Capability

Async SQL database access:

```python
from cogent import tool
import aiosqlite
from typing import Any

class Database:
    """Async SQLite database wrapper."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._pool: aiosqlite.Connection | None = None
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if not self._pool:
            self._pool = await aiosqlite.connect(self.db_path)
        return self._pool
    
    @tool
    async def execute_query(
        self,
        query: str,
        params: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute SQL query and return results.
        
        Args:
            query: SQL query to execute.
            params: Optional query parameters for safety.
        
        Returns:
            List of rows as dictionaries.
        """
        conn = await self._get_connection()
        conn.row_factory = aiosqlite.Row
        
        try:
            async with conn.execute(query, params or []) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            return [{"error": f"Query failed: {str(e)}"}]
    
    async def close(self):
        """Close database connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None
```

**Key features:**
- Connection pooling
- Row factory for dict results
- Parameterized queries (SQL injection prevention)
- Proper resource cleanup

See full implementation: [src/cogent/capabilities/database.py](../src/cogent/capabilities/database.py)

## Example: DataValidator Capability

Schema validation with Pydantic:

```python
from cogent import tool
from pydantic import BaseModel, ValidationError
from typing import Any

class DataValidator:
    """Validate data against Pydantic schemas."""
    
    @tool
    def validate_data(
        self,
        data: dict[str, Any],
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        """Validate data against a Pydantic schema.
        
        Args:
            data: Data to validate.
            schema: Pydantic model class.
        
        Returns:
            Validation result with errors if any.
        """
        try:
            validated = schema(**data)
            return {
                "valid": True,
                "data": validated.model_dump(),
            }
        except ValidationError as e:
            return {
                "valid": False,
                "errors": e.errors(),
            }
```

**Key features:**
- Type-safe validation
- Clear error reporting
- Integration with Pydantic ecosystem

See full implementation: [src/cogent/capabilities/data_validator.py](../src/cogent/capabilities/data_validator.py)

## Tool Composition

Combine multiple tools into a capability class:

```python
from cogent import tool

class APITester:
    """Test HTTP APIs with assertions."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results: list[dict] = []
    
    @tool
    async def test_endpoint(
        self,
        path: str,
        expected_status: int = 200,
    ) -> dict:
        """Test API endpoint."""
        # Make request
        response = await self._request(path)
        
        # Assert status
        passed = response.status == expected_status
        
        # Record result
        result = {
            "endpoint": path,
            "expected": expected_status,
            "actual": response.status,
            "passed": passed,
        }
        self.results.append(result)
        
        return result
    
    @tool
    def get_results(self) -> dict:
        """Get all test results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "results": self.results,
        }
```

**Pattern:** Related tools in a class share state and configuration.

See full implementation: [src/cogent/capabilities/api_tester.py](../src/cogent/capabilities/api_tester.py)

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
from cogent.capabilities import HTTPClient, Database, WebSearch

# Method 1: Pass tool functions directly
agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    tools=[get_weather, search_web, fetch_url],
)

# Method 2: Use capability classes
agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    capabilities=[
        HTTPClient(),
        Database("data.db"),
        WebSearch(),
    ],
)

# Method 3: Mix both
agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    tools=[custom_tool],
    capabilities=[HTTPClient(), Database("data.db")],
)
```

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

```python
from cogent import tool
from functools import lru_cache

@tool
@lru_cache(maxsize=100)
def cached_computation(input: str) -> str:
    """Expensive computation with caching."""
    # Cached result for same input
    return expensive_operation(input)
```

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

## Production Capabilities Reference

Study these for production-ready patterns:

| Capability | Key Features | File |
|------------|--------------|------|
| **HTTPClient** | Retries, timeouts, streaming | [http_client.py](../src/cogent/capabilities/http_client.py) |
| **Database** | Connection pooling, safety | [database.py](../src/cogent/capabilities/database.py) |
| **APITester** | Assertions, test suites | [api_tester.py](../src/cogent/capabilities/api_tester.py) |
| **DataValidator** | Schema validation | [data_validator.py](../src/cogent/capabilities/data_validator.py) |
| **WebSearch** | Semantic caching, news search | [web_search.py](../src/cogent/capabilities/web_search.py) |
| **Browser** | Playwright automation | [browser.py](../src/cogent/capabilities/browser.py) |
| **Shell** | Command injection protection | [shell.py](../src/cogent/capabilities/shell.py) |
| **FileSystem** | Sandboxed file operations | [filesystem.py](../src/cogent/capabilities/filesystem.py) |

## Further Reading

- [Tool Composition](tool-composition.md) — Patterns for combining tools
- [Testing](testing.md) — Testing framework for tools
- [Capabilities Overview](capabilities.md) — All built-in capabilities
- [Agent Configuration](agent.md) — Using tools with agents
