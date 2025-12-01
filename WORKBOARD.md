# AgenticFlow Context Layer

## Concept
Extend the interceptor system with a **Context Layer** - a runtime injection system that provides tools, interceptors, and agents with access to invocation-scoped data. This enables dependency injection, dynamic behavior, and cross-cutting concerns without global state.

## Naming (aflow-style)
| Concept | Name | Description |
|---------|------|-------------|
| Invocation context | `RunContext` | Typed context passed to `agent.run()` |
| Tool access | `ctx: RunContext` | Tools receive context as parameter |
| Dynamic tools | `ToolGate` | Interceptor that filters tools per-call |
| Model failover | `Failover` | Interceptor for automatic model fallback |
| Tool retry | `ToolGuard` | Interceptor for tool retry with backoff |
| Circuit breaker | `CircuitBreaker` | Interceptor to prevent cascading failures |
| Dynamic prompt | `PromptAdapter` | Interceptor to modify system prompt |

## Architecture

```
agent.run(task, context=RunContext(...))
       │
       ▼
┌─────────────────────────────────────┐
│  RunContext (invocation-scoped)     │
│  • user_id, session_id              │
│  • db connections, API keys         │
│  • custom typed data                │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌────────┐          ┌─────────────┐
│ Tools  │          │ Interceptors│
│ ctx.x  │          │ ctx.x       │
└────────┘          └─────────────┘
```

## Features

### 1. RunContext - Invocation Context
Pass typed context data when running an agent. Available to tools and interceptors.

```python
from dataclasses import dataclass
from agenticflow import Agent, RunContext

@dataclass
class AppContext(RunContext):
    user_id: str
    db: Database
    api_key: str

result = await agent.run(
    "Process user request",
    context=AppContext(user_id="123", db=db, api_key=key),
)
```

### 2. ToolGate - Dynamic Tool Selection
Filter available tools based on context (permissions, conversation stage, etc.)

```python
from agenticflow import Agent, ToolGate

class PermissionGate(ToolGate):
    async def filter(self, tools: list[BaseTool], ctx: InterceptContext) -> list[BaseTool]:
        if ctx.run_context.user_role == "admin":
            return tools  # All tools
        return [t for t in tools if not t.name.startswith("admin_")]

agent = Agent(
    name="assistant",
    model=model,
    tools=[read_data, write_data, admin_delete],
    intercept=[PermissionGate()],
)
```

### 3. Failover - Model Fallback
Automatic fallback to backup models on failure.

```python
from agenticflow import Agent, Failover
from agenticflow.models import ChatModel

agent = Agent(
    name="assistant",
    model=ChatModel(model="gpt-4o"),
    intercept=[
        Failover(
            fallbacks=["gpt-4o-mini", "claude-sonnet-4-20250514"],
            on=["rate_limit", "timeout", "error"],
        ),
    ],
)
```

### 4. ToolGuard - Tool Retry
Automatic retry for transient tool failures with exponential backoff.

```python
from agenticflow import Agent, ToolGuard

agent = Agent(
    name="assistant",
    model=model,
    tools=[search, database_query],
    intercept=[
        ToolGuard(
            max_retries=3,
            backoff=2.0,  # Exponential backoff multiplier
            retry_on=[TimeoutError, ConnectionError],
        ),
    ],
)
```

### 5. CircuitBreaker - Prevent Cascading Failures
Open circuit after too many failures, prevent repeated calls to failing tools.

```python
from agenticflow import Agent, CircuitBreaker

agent = Agent(
    name="assistant",
    model=model,
    tools=[external_api],
    intercept=[
        CircuitBreaker(
            failure_threshold=5,
            reset_timeout=30.0,
        ),
    ],
)
```

### 6. PromptAdapter - Dynamic System Prompt
Modify system prompt based on context.

```python
from agenticflow import Agent, LambdaPrompt

agent = Agent(
    name="assistant",
    model=model,
    system_prompt="You are a helpful assistant.",
    intercept=[
        LambdaPrompt(lambda p, ctx: f"{p}\n\nUser: {ctx.run_context.user_name}"),
    ],
)
```

## Implementation Plan

### Phase 1: RunContext ✅
- [x] Define `RunContext` base class (context.py)
- [x] Add `context=` parameter to `agent.run()`
- [x] Pass context to `InterceptContext`
- [x] Enable tool access via `ctx` parameter
- [x] Tests and example

### Phase 2: ToolGate ✅
- [x] Create `ToolGate` interceptor base
- [x] Implement tool filtering in executor
- [x] Built-in: `PermissionGate`, `ConversationGate`
- [x] Tests

### Phase 3: Failover ✅
- [x] Create `Failover` interceptor
- [x] Implement model switching logic
- [x] Handle rate limits, timeouts, errors
- [x] Tests

### Phase 4: ToolGuard ✅
- [x] Create `ToolGuard` interceptor
- [x] Implement retry with exponential backoff
- [x] Configurable retry conditions
- [x] Create `CircuitBreaker` interceptor
- [x] Tests

### Phase 5: PromptAdapter ✅
- [x] Create `PromptAdapter` interceptor base
- [x] Built-in: `ContextPrompt`, `ConversationPrompt`, `LambdaPrompt`
- [x] Implement prompt modification in executor
- [x] Tests

## Files Created/Modified
- `src/agenticflow/interceptors/gates.py` - ToolGate, PermissionGate, ConversationGate
- `src/agenticflow/interceptors/failover.py` - Failover interceptor
- `src/agenticflow/interceptors/guards.py` - ToolGuard, CircuitBreaker
- `src/agenticflow/interceptors/prompt.py` - PromptAdapter, ContextPrompt, ConversationPrompt, LambdaPrompt
- `src/agenticflow/interceptors/base.py` - Enhanced with modified_tools, modified_model, modified_prompt
- `src/agenticflow/interceptors/__init__.py` - Updated exports
- `src/agenticflow/__init__.py` - Added RunContext and new interceptor exports
- `tests/test_interceptors.py` - 45 new tests (125 total)

## Status: ✅ Complete

All context layer interceptors implemented with 125 tests passing.
