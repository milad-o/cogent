"""Logging middleware for Flow execution.

Provides structured logging of event processing and reactor execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.middleware.base import BaseMiddleware

if TYPE_CHECKING:
    from agenticflow.events import Event
    from agenticflow.reactors.base import Reactor


logger = logging.getLogger("agenticflow.middleware")


@dataclass
class LoggingMiddleware(BaseMiddleware):
    """Log event processing and reactor execution.

    Logs events before and after reactor processing with configurable
    detail levels.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        include_data: Whether to log event data
        include_timing: Whether to log execution time
        logger: Custom logger instance

    Example:
        ```python
        from agenticflow.middleware import LoggingMiddleware

        flow = Flow()
        flow.use(LoggingMiddleware(level="DEBUG", include_timing=True))
        ```
    """

    level: str = "INFO"
    include_data: bool = False
    include_timing: bool = True
    custom_logger: Any = None
    _start_times: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._log = self.custom_logger or logger
        self._level = getattr(logging, self.level.upper(), logging.INFO)

    async def before(
        self,
        event: Event,
        reactor: Reactor,
    ) -> Event | None:
        """Log before reactor execution."""
        reactor_name = getattr(reactor, "name", type(reactor).__name__)

        msg = f"[{reactor_name}] Processing event: {event.type}"
        if self.include_data:
            msg += f" data={event.data}"

        self._log.log(self._level, msg)

        if self.include_timing:
            self._start_times[event.id] = time.perf_counter()

        return None

    async def after(
        self,
        result: Event | list[Event] | None,
        event: Event,
        reactor: Reactor,
    ) -> Event | list[Event] | None:
        """Log after reactor execution."""
        reactor_name = getattr(reactor, "name", type(reactor).__name__)

        # Calculate timing
        elapsed = ""
        if self.include_timing:
            start = self._start_times.pop(event.id, None)
            if start:
                elapsed = f" ({(time.perf_counter() - start) * 1000:.2f}ms)"

        # Build message
        if result is None:
            msg = f"[{reactor_name}] Completed{elapsed}: no output"
        elif isinstance(result, list):
            types = [e.type for e in result]
            msg = f"[{reactor_name}] Completed{elapsed}: emitted {types}"
        else:
            msg = f"[{reactor_name}] Completed{elapsed}: emitted {result.type}"

        self._log.log(self._level, msg)

        return None


@dataclass
class VerboseMiddleware(LoggingMiddleware):
    """Verbose logging middleware that includes event data.

    Example:
        ```python
        flow = Flow()
        flow.use(VerboseMiddleware())
        ```
    """

    level: str = "DEBUG"
    include_data: bool = True
    include_timing: bool = True
