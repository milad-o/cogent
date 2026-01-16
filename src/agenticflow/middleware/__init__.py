"""Middleware for Flow execution.

Middleware provides cross-cutting concerns for Flow execution:

- **LoggingMiddleware**: Log event processing and timing
- **RetryMiddleware**: Retry failed reactor executions
- **TimeoutMiddleware**: Enforce execution timeouts
- **TracingMiddleware**: Distributed tracing support

Example:
    ```python
    from agenticflow import Flow
    from agenticflow.middleware import (
        LoggingMiddleware,
        RetryMiddleware,
        TimeoutMiddleware,
    )

    flow = Flow()
    flow.use(LoggingMiddleware(level="DEBUG"))
    flow.use(RetryMiddleware(max_retries=3))
    flow.use(TimeoutMiddleware(timeout=30))

    result = await flow.run("task")
    ```
"""

from agenticflow.middleware.base import (
    BaseMiddleware,
    Middleware,
    MiddlewareChain,
)
from agenticflow.middleware.logging import (
    LoggingMiddleware,
    VerboseMiddleware,
)
from agenticflow.middleware.retry import (
    RetryMiddleware,
    SimpleRetryMiddleware,
)
from agenticflow.middleware.timeout import (
    AggressiveTimeoutMiddleware,
    TimeoutError,
    TimeoutMiddleware,
)
from agenticflow.middleware.tracing import (
    SimpleTracingMiddleware,
    Span,
    TracingMiddleware,
)

__all__ = [
    # Base
    "Middleware",
    "BaseMiddleware",
    "MiddlewareChain",
    # Logging
    "LoggingMiddleware",
    "VerboseMiddleware",
    # Retry
    "RetryMiddleware",
    "SimpleRetryMiddleware",
    # Timeout
    "TimeoutMiddleware",
    "AggressiveTimeoutMiddleware",
    "TimeoutError",
    # Tracing
    "TracingMiddleware",
    "SimpleTracingMiddleware",
    "Span",
]
