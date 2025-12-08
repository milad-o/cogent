# RunContext Module

The `agenticflow.context` module provides invocation-scoped context for dependency injection into tools and interceptors.

## Overview

RunContext enables passing typed data to tools and interceptors at invocation time without global state:

```python
from dataclasses import dataclass
from agenticflow import Agent, RunContext, tool

@dataclass
class AppContext(RunContext):
    user_id: str
    db: Database
    api_key: str

@tool
def get_user_data(ctx: RunContext) -> str:
    """Get data for the current user."""
    user = ctx.db.get_user(ctx.user_id)
    return f"User: {user.name}"

agent = Agent(name="assistant", model=model, tools=[get_user_data])

result = await agent.run(
    "Get my profile data",
    context=AppContext(user_id="123", db=db, api_key=key),
)
```

---

## Creating Custom Contexts

### Basic Context

```python
from dataclasses import dataclass
from agenticflow import RunContext

@dataclass
class MyContext(RunContext):
    user_id: str
    session_id: str
    permissions: list[str] = field(default_factory=list)
    
    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions
```

### With Services

```python
@dataclass
class ServiceContext(RunContext):
    db: Database
    cache: Redis
    api_client: APIClient
    logger: Logger
    
    async def get_user(self, user_id: str) -> User:
        """Get user with caching."""
        cached = await self.cache.get(f"user:{user_id}")
        if cached:
            return User.from_json(cached)
        user = await self.db.get_user(user_id)
        await self.cache.set(f"user:{user_id}", user.to_json())
        return user
```

---

## Using Context in Tools

### Accessing Context

```python
from agenticflow import tool, RunContext

@tool
def get_user_orders(ctx: RunContext) -> str:
    """Get orders for the current user."""
    # Access context properties
    user_id = ctx.user_id
    orders = ctx.db.get_orders(user_id)
    return f"Found {len(orders)} orders"

@tool
def admin_action(action: str, ctx: RunContext) -> str:
    """Perform admin action (requires admin permission)."""
    if not ctx.has_permission("admin"):
        return "Permission denied"
    return f"Performed: {action}"
```

### Context with Other Parameters

```python
@tool
def search_with_context(
    query: str,
    limit: int = 10,
    ctx: RunContext = None,  # Context is optional
) -> str:
    """Search with user context."""
    if ctx and ctx.user_id:
        # Personalized search
        return personalized_search(query, ctx.user_id, limit)
    return generic_search(query, limit)
```

---

## Using Context in Interceptors

```python
from agenticflow.interceptors import Interceptor, InterceptContext, InterceptResult

class PermissionInterceptor(Interceptor):
    async def intercept(
        self,
        phase: Phase,
        context: InterceptContext,
    ) -> InterceptResult:
        run_ctx = context.run_context
        
        # Check permissions
        if not run_ctx.has_permission("use_agent"):
            return InterceptResult.stop("Permission denied")
        
        return InterceptResult.continue_()
```

---

## Context Metadata

The base `RunContext` includes a metadata dict for extension:

```python
from agenticflow import RunContext

ctx = RunContext(metadata={
    "request_id": "req-123",
    "correlation_id": "corr-456",
    "trace_id": "trace-789",
})

# Access metadata
request_id = ctx.get("request_id")
request_id = ctx.metadata.get("request_id")

# Create new context with additional metadata
new_ctx = ctx.with_metadata(
    timestamp=datetime.now(),
    version="1.0",
)
```

---

## Passing Context to Agents

```python
from agenticflow import Agent

agent = Agent(name="assistant", model=model, tools=[...])

# Pass context at run time
result = await agent.run(
    "Perform action",
    context=MyContext(
        user_id="user-123",
        session_id="sess-456",
        permissions=["read", "write"],
    ),
)
```

---

## Context Immutability

Context should be treated as immutable during execution:

```python
@dataclass(frozen=True)  # Enforce immutability
class ImmutableContext(RunContext):
    user_id: str
    tenant_id: str

# To "modify", create a new instance
new_ctx = ctx.with_metadata(extra="value")
```

---

## Request-Scoped Context

Common pattern for web applications:

```python
from fastapi import FastAPI, Depends
from agenticflow import Agent, RunContext

app = FastAPI()

@dataclass
class RequestContext(RunContext):
    user_id: str
    tenant_id: str
    trace_id: str
    db: AsyncSession

async def get_context(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> RequestContext:
    return RequestContext(
        user_id=request.state.user_id,
        tenant_id=request.state.tenant_id,
        trace_id=request.headers.get("X-Trace-ID"),
        db=db,
    )

@app.post("/chat")
async def chat(
    message: str,
    context: RequestContext = Depends(get_context),
):
    agent = get_agent()
    result = await agent.run(message, context=context)
    return {"response": result.output}
```

---

## Multi-Tenant Context

```python
@dataclass
class TenantContext(RunContext):
    tenant_id: str
    tenant_config: dict
    allowed_tools: list[str]
    rate_limit: int
    
    def can_use_tool(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools

# Use in agent
result = await agent.run(
    "Query",
    context=TenantContext(
        tenant_id="acme-corp",
        tenant_config={"max_tokens": 4000},
        allowed_tools=["search", "calculate"],
        rate_limit=100,
    ),
)
```

---

## Testing with Context

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def test_context():
    return MyContext(
        user_id="test-user",
        db=Mock(),
        cache=Mock(),
    )

async def test_tool_with_context(test_context):
    result = await get_user_orders.__wrapped__(ctx=test_context)
    assert "orders" in result
```

---

## API Reference

### RunContext

```python
@dataclass
class RunContext:
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        
    def with_metadata(self, **kwargs) -> RunContext:
        """Create new context with additional metadata."""
```

### Usage Patterns

| Pattern | Description |
|---------|-------------|
| Tool injection | `def my_tool(ctx: RunContext)` |
| Optional context | `def my_tool(arg: str, ctx: RunContext = None)` |
| Interceptor access | `context.run_context` |
| Metadata access | `ctx.get("key")` or `ctx.metadata["key"]` |
| Extend context | `ctx.with_metadata(key="value")` |
