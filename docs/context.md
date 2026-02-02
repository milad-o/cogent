# RunContext Module

The `cogent.context` module provides invocation-scoped context for dependency injection into tools and interceptors.

## Overview

RunContext enables passing typed data to tools and interceptors at invocation time without global state. Context automatically includes the original user query for task lineage tracking.

```python
from dataclasses import dataclass
from cogent import Agent, RunContext, tool

@dataclass
class AppContext(RunContext):
    user_id: str
    db: Database
    api_key: str

@tool
def get_user_data(ctx: RunContext) -> str:
    """Get data for the current user."""
    user = ctx.db.get_user(ctx.user_id)
    # Access original query
    print(f"Original query: {ctx.query}")
    return f"User: {user.name}"

agent = Agent(name="assistant", model=model, tools=[get_user_data])

result = await agent.run(
    "Get my profile data",
    context=AppContext(user_id="123", db=db, api_key=key),
)
```

---

## Built-in Context Fields

The base `RunContext` provides framework-managed fields:

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Original user query that initiated execution. Auto-populated on first `agent.run()`. |
| `metadata` | `dict` | Extension dict for untyped data. |
| `agent` | `Agent \| None` | Internal reference to executing agent. |

### Query Field

The `query` field tracks the original user request through agent delegations:

```python
# Top-level agent
result = await orchestrator.run("Can I delete files?", context=ctx)
# ctx.query = "Can I delete files?"

# Delegated sub-agent (via agent.as_tool())
# Still has: ctx.query = "Can I delete files?"
# But receives different task parameter: "check user permissions"
```

This enables sub-agents to understand the broader context while working on their specific subtask.

---

## Creating Custom Contexts

### Basic Context

```python
from dataclasses import dataclass
from cogent import RunContext

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
from cogent import tool, RunContext

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
from cogent.interceptors import Interceptor, InterceptContext, InterceptResult

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
from cogent import RunContext

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

## Agent-as-Tool Context Propagation

When using `agent.as_tool()`, context flows automatically to delegated agents (like regular tools):

```python
from cogent import Agent, tool
from dataclasses import dataclass

@dataclass
class UserContext(RunContext):
    user_id: str
    permissions: set[str]

@tool
def check_permissions(ctx: RunContext) -> str:
    """Check if user has required permissions."""
    print(f"Original query: {ctx.query}")
    if "admin" in ctx.permissions:
        return "Permission granted"
    return "Permission denied"

# Specialist agent receives context automatically
specialist = Agent(
    name="permission_checker",
    model=model,
    tools=[check_permissions],
)

# Orchestrator delegates to specialist
orchestrator = Agent(
    name="orchestrator",
    model=model,
    tools=[specialist.as_tool()],  # Context flows automatically
)

# Context passes through entire delegation chain
result = await orchestrator.run(
    "Can I delete files?",
    context=UserContext(user_id="123", permissions={"read"}),
)
```

### Isolating Context

Use `isolate_context=True` to create explicit context boundaries:

```python
# Delegate without context (fresh boundary)
specialist_tool = specialist.as_tool(isolate_context=True)

orchestrator = Agent(
    name="orchestrator",
    model=model,
    tools=[specialist_tool],
)

# Specialist runs without access to UserContext
result = await orchestrator.run(
    "Process data",
    context=UserContext(...),
)
```

Use isolation when:
- Sub-agent operates on different data domains
- Security boundaries required
- Independent execution needed

---

## Passing Context to Agents

```python
from cogent import Agent

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
from cogent import Agent, RunContext

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

## Common Patterns

### Tracking Delegation Depth

Prevent infinite delegation loops and adapt behavior based on depth:

```python
from dataclasses import dataclass, field
from cogent import Agent, RunContext, tool

@dataclass
class DelegationContext(RunContext):
    depth: int = 0
    max_depth: int = 3

@tool
def process_task(ctx: DelegationContext) -> str:
    """Process a task with depth awareness."""
    if ctx.depth >= ctx.max_depth:
        return "Maximum delegation depth reached - handling directly"
    
    # Different strategy based on depth
    if ctx.depth == 0:
        return "Top-level processing"
    else:
        return f"Sub-task processing at depth {ctx.depth}"

# When delegating, increment depth
specialist = Agent(name="specialist", model=model, tools=[process_task])

# In orchestrator, increment depth when delegating
@tool
def delegate_task(task: str, ctx: DelegationContext) -> str:
    """Delegate to specialist with incremented depth."""
    new_ctx = DelegationContext(
        query=ctx.query,
        depth=ctx.depth + 1,
        max_depth=ctx.max_depth,
        metadata=ctx.metadata,
    )
    return specialist.run(task, context=new_ctx).output
```

### Retry Tracking

Track retry attempts to implement fallback strategies:

```python
@dataclass
class RetryContext(RunContext):
    retry_count: int = 0
    max_retries: int = 3
    is_retry: bool = False

@tool
def fetch_data(url: str, ctx: RetryContext) -> str:
    """Fetch data with retry awareness."""
    if ctx.is_retry:
        # Use more conservative approach on retry
        timeout = 10 + (ctx.retry_count * 5)  # Increase timeout
        print(f"Retry #{ctx.retry_count} with timeout={timeout}s")
    else:
        timeout = 5
    
    # Fetch logic...
    return f"Data from {url}"

# On retry, create new context
if failed and retry_count < max_retries:
    retry_ctx = RetryContext(
        query=ctx.query,
        retry_count=retry_count + 1,
        is_retry=True,
        metadata=ctx.metadata,
    )
    result = agent.run(task, context=retry_ctx)
```

### Task Lineage Tracking

Track parent-child task relationships:

```python
from cogent.core import generate_id

@dataclass
class TaskContext(RunContext):
    task_id: str = field(default_factory=generate_id)
    parent_task_id: str | None = None
    task_chain: list[str] = field(default_factory=list)
    
    @property
    def is_root_task(self) -> bool:
        return self.parent_task_id is None
    
    @property
    def depth(self) -> int:
        return len(self.task_chain)

@tool
def analyze_lineage(ctx: TaskContext) -> str:
    """Show task lineage information."""
    if ctx.is_root_task:
        return f"Root task: {ctx.task_id}"
    else:
        chain = " → ".join(ctx.task_chain + [ctx.task_id])
        return f"Task chain ({ctx.depth} deep): {chain}"

# When delegating, build chain
def delegate_with_lineage(task: str, ctx: TaskContext) -> str:
    child_ctx = TaskContext(
        query=ctx.query,
        task_id=generate_id(),
        parent_task_id=ctx.task_id,
        task_chain=ctx.task_chain + [ctx.task_id],
        metadata=ctx.metadata,
    )
    return specialist.run(task, context=child_ctx).output
```

### Execution Timing

Track execution duration and deadlines:

```python
from datetime import datetime, timedelta
from cogent.core import now_utc

@dataclass
class TimedContext(RunContext):
    started_at: datetime = field(default_factory=now_utc)
    deadline: datetime | None = None
    
    @property
    def elapsed_seconds(self) -> float:
        return (now_utc() - self.started_at).total_seconds()
    
    @property
    def is_overdue(self) -> bool:
        if self.deadline is None:
            return False
        return now_utc() > self.deadline
    
    @property
    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return (self.deadline - now_utc()).total_seconds()

@tool
def long_running_task(ctx: TimedContext) -> str:
    """Task that respects deadlines."""
    if ctx.is_overdue:
        return "Task deadline exceeded - aborting"
    
    remaining = ctx.remaining_seconds
    if remaining and remaining < 5:
        return "Running in fast mode - deadline approaching"
    
    return f"Normal execution (elapsed: {ctx.elapsed_seconds:.1f}s)"

# Create context with deadline
ctx = TimedContext(
    deadline=now_utc() + timedelta(seconds=30)
)
result = agent.run("Perform task", context=ctx)
```

### Combining Patterns

Compose multiple patterns for comprehensive tracking:

```python
@dataclass
class ExecutionContext(RunContext):
    """Rich execution context combining common patterns."""
    
    # Task identity
    task_id: str = field(default_factory=generate_id)
    parent_task_id: str | None = None
    
    # Delegation tracking
    depth: int = 0
    max_depth: int = 5
    
    # Retry tracking
    retry_count: int = 0
    is_retry: bool = False
    
    # Timing
    started_at: datetime = field(default_factory=now_utc)
    deadline: datetime | None = None
    
    # Helper methods
    @property
    def can_delegate(self) -> bool:
        return self.depth < self.max_depth
    
    @property
    def elapsed_seconds(self) -> float:
        return (now_utc() - self.started_at).total_seconds()
    
    def create_child_context(self) -> "ExecutionContext":
        """Create context for delegated task."""
        return ExecutionContext(
            query=self.query,
            task_id=generate_id(),
            parent_task_id=self.task_id,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            metadata=self.metadata,
            started_at=now_utc(),
            deadline=self.deadline,  # Propagate deadline
        )

@tool
def smart_task(ctx: ExecutionContext) -> str:
    """Task that uses comprehensive context."""
    if not ctx.can_delegate:
        return "Max delegation depth reached"
    
    if ctx.is_retry:
        return f"Retry attempt #{ctx.retry_count}"
    
    return f"Task {ctx.task_id} at depth {ctx.depth} (elapsed: {ctx.elapsed_seconds:.1f}s)"
```

---

## API Reference

### RunContext

```python
@dataclass
class RunContext:
    query: str = field(default="", repr=False)
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
| Subclass context | `class MyContext(RunContext)` |
