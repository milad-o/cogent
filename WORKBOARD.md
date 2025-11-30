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
| `ContextCompressor` | Summarize when approaching token limits | 🔜 Planned |
| `PIIShield` | Detect and mask/block sensitive data | 🔜 Planned |
| `ToolSelector` | Filter relevant tools for large toolsets | 🔜 Planned |
| `RateLimiter` | Rate limit tool calls | 🔜 Planned |
| `Auditor` | Log all actions for compliance | 🔜 Planned |

## Implementation Tasks

- [x] Core: `Interceptor` base class and `InterceptContext`
- [x] Core: `Phase` enum for execution phases
- [x] Core: Agent integration (`intercept=` parameter)
- [x] Core: Executor integration (call interceptors at phases)
- [x] Built-in: `BudgetGuard` (model/tool call limits)
- [ ] Built-in: `ContextCompressor` (summarization)
- [ ] Built-in: `PIIShield` (PII detection)
- [x] Tests: Core interceptor tests (32 tests)
- [x] Example: `25_interceptors.py`
- [x] Export to main package `__init__.py`

## API Design

```python
from agenticflow import Agent, BudgetGuard

agent = Agent(
    name="assistant",
    model=model,
    intercept=[
        BudgetGuard(max_model_calls=10, max_tool_calls=50),
    ],
)
```

## Files

| File | Purpose |
|------|---------|
| `src/agenticflow/interceptors/__init__.py` | Module exports |
| `src/agenticflow/interceptors/base.py` | Core: Interceptor, Phase, InterceptContext, InterceptResult |
| `src/agenticflow/interceptors/budget.py` | BudgetGuard implementation |
| `tests/test_interceptors.py` | Unit tests (32 tests) |
| `examples/25_interceptors.py` | Usage examples |
