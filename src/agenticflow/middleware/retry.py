"""Retry middleware for Flow execution.

Provides automatic retry logic for failed reactor executions.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agenticflow.middleware.base import BaseMiddleware

if TYPE_CHECKING:
    from agenticflow.events import Event
    from agenticflow.reactors.base import Reactor


@dataclass
class RetryMiddleware(BaseMiddleware):
    """Retry failed reactor executions with configurable backoff.

    Implements retry logic with exponential backoff and jitter.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential: Use exponential backoff
        jitter: Add random jitter to delays
        retry_on: Optional predicate to determine if error is retryable

    Example:
        ```python
        from agenticflow.middleware import RetryMiddleware

        flow = Flow()
        flow.use(RetryMiddleware(max_retries=3, base_delay=1.0))
        ```

    Example with custom retry logic:
        ```python
        def should_retry(error: Exception) -> bool:
            # Only retry on transient errors
            return isinstance(error, (TimeoutError, ConnectionError))

        flow.use(RetryMiddleware(max_retries=3, retry_on=should_retry))
        ```
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential: bool = True
    jitter: bool = True
    retry_on: Callable[[Exception], bool] | None = None
    _retry_counts: dict[str, int] = field(default_factory=dict, init=False)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        delay = self.base_delay * 2 ** attempt if self.exponential else self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def _should_retry(self, error: Exception) -> bool:
        """Determine if the error should trigger a retry."""
        if self.retry_on:
            return self.retry_on(error)
        # Default: retry on most errors except explicit stops
        return not isinstance(error, (KeyboardInterrupt, SystemExit))

    async def wrap_execution(
        self,
        reactor: Reactor,
        event: Event,
        execute: Callable[[], Event | list[Event] | None],
    ) -> Event | list[Event] | None:
        """Execute with retry logic.

        This method should be called by the Flow to wrap reactor execution.

        Args:
            reactor: The reactor being executed
            event: The event being processed
            execute: The execution function to retry

        Returns:
            Reactor result or raises after max retries
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await execute()
            except Exception as e:
                last_error = e

                if attempt >= self.max_retries or not self._should_retry(e):
                    raise

                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)

        if last_error:
            raise last_error
        return None


@dataclass
class SimpleRetryMiddleware(RetryMiddleware):
    """Simple retry with fixed delay (no exponential backoff).

    Example:
        ```python
        flow.use(SimpleRetryMiddleware(max_retries=2, base_delay=0.5))
        ```
    """

    exponential: bool = False
    jitter: bool = False
