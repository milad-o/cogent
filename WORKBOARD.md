# AgenticFlow Interceptors

## Concept
Interceptors are composable units that intercept and modify agent execution at key phases. They enable cross-cutting concerns like cost control, context management, security, and observability without cluttering core agent logic.

## Naming
- **Interceptor**: Base class for all interceptors
- **Parameter**: `intercept=[...]` on Agent
- **Phases**: `PRE_RUN`, `PRE_THINK`, `POST_THINK`, `PRE_ACT`, `POST_ACT`, `POST_RUN`

## Built-in Interceptors

| Interceptor | Purpose | Status |
|-------------|---------|--------|
| `BudgetGuard` | Limit model/tool calls for cost control | ✅ Done |
| `ContextCompressor` | Summarize when approaching token limits | ✅ Done |
| `TokenLimiter` | Hard stop when context exceeds limit | ✅ Done |
| `PIIShield` | Detect and mask/block sensitive data | ✅ Done |
| `ContentFilter` | Block specific words/patterns | ✅ Done |
| `RateLimiter` | Rate limit tool calls | 🔜 Planned |
| `Auditor` | Log all actions for compliance | 🔜 Planned |

## Implementation Tasks

- [x] Core: `Interceptor` base class and `InterceptContext`
- [x] Core: `Phase` enum for execution phases
- [x] Core: Agent integration (`intercept=` parameter)
- [x] Core: Executor integration (call interceptors at phases)
- [x] Built-in: `BudgetGuard` (model/tool call limits)
- [x] Built-in: `ContextCompressor` (summarization)
- [x] Built-in: `TokenLimiter` (hard limit)
- [x] Built-in: `PIIShield` (PII detection/masking)
- [x] Built-in: `ContentFilter` (word/pattern blocking)
- [x] Tests: All interceptor tests (59 tests)
- [x] Example: `25_interceptors.py` (9 examples)
- [x] Export to main package `__init__.py`

## API Design

```python
from agenticflow import Agent, BudgetGuard, PIIShield, PIIAction

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        BudgetGuard(max_model_calls=10, max_tool_calls=50),
        PIIShield(patterns=["email", "ssn"], action=PIIAction.MASK),
    ],
)
```

## Files

| File | Purpose |
|------|---------|
| `src/agenticflow/interceptors/__init__.py` | Module exports |
| `src/agenticflow/interceptors/base.py` | Core: Interceptor, Phase, InterceptContext, InterceptResult |
| `src/agenticflow/interceptors/budget.py` | BudgetGuard implementation |
| `src/agenticflow/interceptors/context.py` | ContextCompressor, TokenLimiter |
| `src/agenticflow/interceptors/security.py` | PIIShield, ContentFilter |
| `tests/test_interceptors.py` | Unit tests (59 tests) |
| `examples/25_interceptors.py` | Usage examples (9 demos) |
