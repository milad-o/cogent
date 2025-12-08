# Interceptors Module

The `agenticflow.interceptors` module provides composable units that intercept agent execution for cross-cutting concerns like cost control, security, context management, and observability.

## Overview

Interceptors are middleware that wrap agent execution at specific phases:
- **Before LLM call** - Modify context, filter tools, check budgets
- **After LLM call** - Validate responses, mask PII, audit
- **Before tool call** - Gate access, rate limit, retry logic
- **After tool call** - Post-process results, aggregate data

```python
from agenticflow import Agent
from agenticflow.interceptors import BudgetGuard, PIIShield

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        BudgetGuard(max_model_calls=10, max_tool_calls=50),
        PIIShield(patterns=["email", "ssn"]),
    ],
)
```

---

## Core Concepts

### Interceptor Lifecycle

Interceptors run at specific phases:

```
User Query
    ↓
[BEFORE_LLM] ← Context modification, validation
    ↓
LLM Call
    ↓
[AFTER_LLM] ← Response validation, PII masking
    ↓
[BEFORE_TOOL] ← Tool gating, rate limiting
    ↓
Tool Execution
    ↓
[AFTER_TOOL] ← Result post-processing
    ↓
Response to User
```

### Phase Enum

```python
from agenticflow.interceptors import Phase

Phase.BEFORE_LLM    # Before sending to LLM
Phase.AFTER_LLM     # After receiving LLM response
Phase.BEFORE_TOOL   # Before tool execution
Phase.AFTER_TOOL    # After tool execution
```

### InterceptResult

Interceptors return a result that can:
- **Continue** - Proceed to next interceptor/phase
- **Modify** - Change the data and continue
- **Stop** - Halt execution with a response

```python
from agenticflow.interceptors import InterceptResult

# Continue unchanged
return InterceptResult.continue_()

# Modify and continue
return InterceptResult.modify(new_messages=modified_messages)

# Stop execution
return InterceptResult.stop(response="Cannot proceed: budget exceeded")
```

---

## Built-in Interceptors

### BudgetGuard

Control costs by limiting LLM and tool calls:

```python
from agenticflow.interceptors import BudgetGuard

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        BudgetGuard(
            max_model_calls=10,     # Max LLM invocations
            max_tool_calls=50,      # Max tool executions
            max_tokens=100000,      # Max token usage
            on_exceeded="stop",     # "stop" or "warn"
        ),
    ],
)

# Check budget status
guard = agent.interceptors[0]
print(f"Calls: {guard.model_calls}/{guard.max_model_calls}")
print(f"Tokens: {guard.tokens_used}/{guard.max_tokens}")
```

### PIIShield

Detect and handle PII in inputs/outputs:

```python
from agenticflow.interceptors import PIIShield, PIIAction

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        PIIShield(
            patterns=["email", "phone", "ssn", "credit_card"],
            action=PIIAction.MASK,  # MASK, REDACT, or BLOCK
        ),
    ],
)

# Input: "Contact john@email.com"
# Masked: "Contact [EMAIL]"
```

**Actions:**

| Action | Behavior |
|--------|----------|
| `PIIAction.MASK` | Replace with `[TYPE]` placeholder |
| `PIIAction.REDACT` | Remove entirely |
| `PIIAction.BLOCK` | Stop execution with error |

### ContentFilter

Filter harmful or inappropriate content:

```python
from agenticflow.interceptors import ContentFilter

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        ContentFilter(
            block_patterns=["password", "secret key"],
            allow_patterns=["public api"],  # Whitelist
        ),
    ],
)
```

### TokenLimiter

Limit context size to fit model constraints:

```python
from agenticflow.interceptors import TokenLimiter

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        TokenLimiter(
            max_tokens=8000,        # Max context tokens
            strategy="truncate",   # "truncate" or "summarize"
            keep_system=True,      # Always keep system message
            keep_last_n=5,         # Keep last N messages
        ),
    ],
)
```

### ContextCompressor

Compress context to reduce token usage:

```python
from agenticflow.interceptors import ContextCompressor

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        ContextCompressor(
            model=model,           # LLM for summarization
            trigger_tokens=6000,   # Compress when above this
            target_tokens=3000,    # Target after compression
        ),
    ],
)
```

---

## Tool Control

### ToolGate

Control which tools are available:

```python
from agenticflow.interceptors import ToolGate

agent = Agent(
    name="assistant",
    model=model,
    tools=[search, write_file, delete_file],
    intercept=[
        ToolGate(
            allow=["search"],           # Only these tools
            # Or: deny=["delete_file"], # Block these tools
        ),
    ],
)
```

**Dynamic gating:**

```python
def gate_by_user(ctx: InterceptContext) -> list[str]:
    if ctx.run_context.get("is_admin"):
        return ["*"]  # All tools
    return ["search", "read_file"]

agent = Agent(
    intercept=[ToolGate(allow=gate_by_user)],
)
```

### PermissionGate

Role-based tool permissions:

```python
from agenticflow.interceptors import PermissionGate

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        PermissionGate(
            permissions={
                "admin": ["*"],
                "user": ["search", "read"],
                "guest": ["search"],
            },
            get_role=lambda ctx: ctx.run_context.get("role", "guest"),
        ),
    ],
)
```

### ConversationGate

Enable tools based on conversation state:

```python
from agenticflow.interceptors import ConversationGate

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        ConversationGate(
            # Unlock tools after specific messages
            unlock_on={
                "confirmed": ["execute_order"],
                "authenticated": ["view_account", "transfer"],
            },
        ),
    ],
)
```

---

## Resilience

### RateLimiter

Limit request rates:

```python
from agenticflow.interceptors import RateLimiter

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        RateLimiter(
            max_requests=10,       # Max requests
            window_seconds=60,     # Per time window
            on_exceeded="wait",    # "wait" or "error"
        ),
    ],
)
```

### Failover

Automatic model failover:

```python
from agenticflow.interceptors import Failover, FailoverTrigger
from agenticflow.models import ChatModel

agent = Agent(
    name="assistant",
    model=primary_model,
    intercept=[
        Failover(
            fallback_models=[
                ChatModel(model="gpt-4o-mini"),
                ChatModel(model="gpt-3.5-turbo"),
            ],
            triggers=[
                FailoverTrigger.RATE_LIMIT,
                FailoverTrigger.TIMEOUT,
                FailoverTrigger.ERROR,
            ],
            max_retries=2,
        ),
    ],
)
```

### CircuitBreaker

Prevent cascade failures:

```python
from agenticflow.interceptors import CircuitBreaker

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        CircuitBreaker(
            failure_threshold=5,   # Failures before opening
            recovery_timeout=60,   # Seconds before retry
            half_open_requests=2,  # Test requests when recovering
        ),
    ],
)
```

### ToolGuard

Per-tool retry and circuit breaker:

```python
from agenticflow.interceptors import ToolGuard

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        ToolGuard(
            tool_configs={
                "search": {
                    "max_retries": 3,
                    "backoff": "exponential",
                    "circuit_breaker": True,
                },
                "database": {
                    "max_retries": 1,
                    "timeout": 30,
                },
            },
        ),
    ],
)
```

---

## Auditing

### Auditor

Log all agent activity:

```python
from agenticflow.interceptors import Auditor, AuditEventType

async def log_event(event):
    print(f"[{event.type}] {event.agent}: {event.data}")

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        Auditor(
            handler=log_event,
            events=[
                AuditEventType.LLM_REQUEST,
                AuditEventType.LLM_RESPONSE,
                AuditEventType.TOOL_CALL,
                AuditEventType.TOOL_RESULT,
            ],
            include_content=True,  # Log message content
        ),
    ],
)
```

---

## Prompt Adapters

### ContextPrompt

Inject dynamic context into system prompt:

```python
from agenticflow.interceptors import ContextPrompt

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        ContextPrompt(
            template="""Current time: {time}
User timezone: {timezone}
User preferences: {preferences}""",
            get_context=lambda ctx: {
                "time": datetime.now().isoformat(),
                "timezone": ctx.run_context.get("timezone", "UTC"),
                "preferences": ctx.run_context.get("preferences", {}),
            },
        ),
    ],
)
```

### ConversationPrompt

Add conversation-aware context:

```python
from agenticflow.interceptors import ConversationPrompt

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        ConversationPrompt(
            summary_threshold=20,  # Summarize after N messages
            include_summary=True,
            model=model,
        ),
    ],
)
```

### LambdaPrompt

Custom prompt modification:

```python
from agenticflow.interceptors import LambdaPrompt

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        LambdaPrompt(
            modifier=lambda messages, ctx: [
                {**m, "content": m["content"].upper()}
                if m["role"] == "user" else m
                for m in messages
            ],
        ),
    ],
)
```

---

## Custom Interceptors

### Basic Structure

```python
from agenticflow.interceptors import Interceptor, Phase, InterceptContext, InterceptResult

class MyInterceptor(Interceptor):
    """Custom interceptor example."""
    
    phases = [Phase.BEFORE_LLM, Phase.AFTER_LLM]
    
    async def intercept(
        self,
        phase: Phase,
        context: InterceptContext,
    ) -> InterceptResult:
        if phase == Phase.BEFORE_LLM:
            # Modify messages before LLM
            messages = context.messages
            messages.append({"role": "system", "content": "Be concise."})
            return InterceptResult.modify(new_messages=messages)
        
        elif phase == Phase.AFTER_LLM:
            # Log response
            print(f"Response: {context.response.content}")
            return InterceptResult.continue_()
```

### InterceptContext

Available context in interceptors:

```python
@dataclass
class InterceptContext:
    agent: Agent                  # Current agent
    phase: Phase                  # Current phase
    messages: list[dict]          # Current messages
    response: AIMessage | None    # LLM response (after phases)
    tool_call: dict | None        # Tool call info (tool phases)
    tool_result: Any | None       # Tool result (AFTER_TOOL)
    run_context: RunContext       # User-provided context
    metadata: dict                # Additional data
```

### Stateful Interceptors

```python
class ConversationTracker(Interceptor):
    """Track conversation statistics."""
    
    phases = [Phase.AFTER_LLM]
    
    def __init__(self):
        self.message_count = 0
        self.total_tokens = 0
    
    async def intercept(
        self,
        phase: Phase,
        context: InterceptContext,
    ) -> InterceptResult:
        self.message_count += 1
        if context.response and context.response.usage:
            self.total_tokens += context.response.usage.get("total_tokens", 0)
        return InterceptResult.continue_()
    
    def stats(self) -> dict:
        return {
            "messages": self.message_count,
            "tokens": self.total_tokens,
        }
```

---

## Combining Interceptors

Interceptors execute in order. Use `StopExecution` to halt the chain:

```python
from agenticflow.interceptors import StopExecution

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        # Order matters - these run sequentially
        PIIShield(patterns=["ssn"]),       # First: mask PII
        BudgetGuard(max_model_calls=10),   # Second: check budget
        ToolGate(allow=["search"]),        # Third: filter tools
        Auditor(handler=log),              # Last: audit all activity
    ],
)

# If BudgetGuard exceeds limit, it raises StopExecution
# and Auditor never runs for that call
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Interceptor` | Base class for all interceptors |
| `InterceptContext` | Context passed to interceptors |
| `InterceptResult` | Return type from intercept method |
| `Phase` | Enum of interception phases |
| `StopExecution` | Exception to halt execution |

### Built-in Interceptors

| Category | Interceptors |
|----------|--------------|
| **Budget** | `BudgetGuard` |
| **Security** | `PIIShield`, `ContentFilter` |
| **Context** | `TokenLimiter`, `ContextCompressor` |
| **Gates** | `ToolGate`, `PermissionGate`, `ConversationGate` |
| **Rate Limit** | `RateLimiter`, `ThrottleInterceptor` |
| **Resilience** | `Failover`, `CircuitBreaker`, `ToolGuard` |
| **Audit** | `Auditor` |
| **Prompts** | `ContextPrompt`, `ConversationPrompt`, `LambdaPrompt` |
