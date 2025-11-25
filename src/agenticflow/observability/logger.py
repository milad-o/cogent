"""Structured logging for observability.

Provides structured logging with levels, context,
and integration with the event system.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, TextIO
from collections.abc import Callable

from agenticflow.core import now_utc, generate_id


class LogLevel(IntEnum):
    """Log severity levels."""

    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEntry:
    """A structured log entry.

    Attributes:
        level: Severity level.
        message: Log message.
        timestamp: When the log was created.
        logger_name: Name of the logger.
        context: Additional context data.
        trace_id: Optional trace ID for correlation.
        span_id: Optional span ID for correlation.
    """

    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=now_utc)
    logger_name: str = "root"
    context: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None
    entry_id: str = field(default_factory=generate_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.entry_id,
            "level": self.level.name,
            "level_value": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "logger": self.logger_name,
            "context": self.context,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def format(self, fmt: str = "text") -> str:
        """Format log entry as string.

        Args:
            fmt: Format type ("text" or "json").

        Returns:
            Formatted log string.
        """
        if fmt == "json":
            return self.to_json()

        # Text format
        parts = [
            self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"[{self.level.name:8}]",
            f"[{self.logger_name}]",
        ]

        if self.trace_id:
            parts.append(f"[trace:{self.trace_id[:8]}]")

        parts.append(self.message)

        if self.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"| {ctx_str}")

        return " ".join(parts)


class ObservabilityLogger:
    """Structured logger with context propagation.

    Example:
        >>> logger = ObservabilityLogger("agent.executor")
        >>> logger.info("Task started", task_id="123", agent="researcher")
        >>> with logger.context(trace_id="abc"):
        ...     logger.debug("Processing step 1")
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        output: TextIO | None = None,
        format: str = "text",
        on_log: Callable[[LogEntry], None] | None = None,
    ) -> None:
        """Initialize logger.

        Args:
            name: Logger name (e.g., "agent.executor").
            level: Minimum level to log.
            output: Output stream (default: stderr).
            format: Output format ("text" or "json").
            on_log: Optional callback for each log entry.
        """
        self.name = name
        self.level = level
        self.output = output or sys.stderr
        self.format = format
        self.on_log = on_log
        self._context: dict[str, Any] = {}
        self._trace_id: str | None = None
        self._span_id: str | None = None
        self._entries: list[LogEntry] = []
        self._max_entries: int = 10000

    def _should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged."""
        return level >= self.level

    def _log(
        self,
        level: LogLevel,
        message: str,
        **kwargs: Any,
    ) -> LogEntry | None:
        """Internal log method.

        Args:
            level: Log level.
            message: Log message.
            **kwargs: Additional context.

        Returns:
            The log entry if logged, None if filtered.
        """
        if not self._should_log(level):
            return None

        # Merge context
        context = {**self._context, **kwargs}

        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            context=context,
            trace_id=self._trace_id,
            span_id=self._span_id,
        )

        # Store entry
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Output
        self.output.write(entry.format(self.format) + "\n")
        self.output.flush()

        # Callback
        if self.on_log:
            self.on_log(entry)

        return entry

    def trace(self, message: str, **kwargs: Any) -> LogEntry | None:
        """Log at TRACE level."""
        return self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> LogEntry | None:
        """Log at DEBUG level."""
        return self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> LogEntry | None:
        """Log at INFO level."""
        return self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> LogEntry | None:
        """Log at WARNING level."""
        return self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> LogEntry | None:
        """Log at ERROR level."""
        return self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> LogEntry | None:
        """Log at CRITICAL level."""
        return self._log(LogLevel.CRITICAL, message, **kwargs)

    def exception(self, message: str, exc: Exception, **kwargs: Any) -> LogEntry | None:
        """Log exception with traceback info.

        Args:
            message: Log message.
            exc: The exception.
            **kwargs: Additional context.

        Returns:
            The log entry.
        """
        return self._log(
            LogLevel.ERROR,
            message,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            **kwargs,
        )

    def set_context(self, **kwargs: Any) -> None:
        """Set persistent context for all future logs.

        Args:
            **kwargs: Context key-value pairs.
        """
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all persistent context."""
        self._context.clear()

    def set_trace(self, trace_id: str, span_id: str | None = None) -> None:
        """Set trace correlation IDs.

        Args:
            trace_id: Trace ID for correlation.
            span_id: Optional span ID.
        """
        self._trace_id = trace_id
        self._span_id = span_id

    def clear_trace(self) -> None:
        """Clear trace correlation."""
        self._trace_id = None
        self._span_id = None

    def child(self, name: str) -> "ObservabilityLogger":
        """Create a child logger.

        Args:
            name: Child logger name suffix.

        Returns:
            New logger with inherited settings.
        """
        child_name = f"{self.name}.{name}"
        child = ObservabilityLogger(
            name=child_name,
            level=self.level,
            output=self.output,
            format=self.format,
            on_log=self.on_log,
        )
        child._context = self._context.copy()
        child._trace_id = self._trace_id
        child._span_id = self._span_id
        return child

    @property
    def entries(self) -> list[LogEntry]:
        """Get stored log entries."""
        return self._entries.copy()

    def get_entries(
        self,
        level: LogLevel | None = None,
        since: datetime | None = None,
        trace_id: str | None = None,
    ) -> list[LogEntry]:
        """Query stored log entries.

        Args:
            level: Filter by minimum level.
            since: Filter by timestamp.
            trace_id: Filter by trace ID.

        Returns:
            Matching log entries.
        """
        entries = self._entries

        if level:
            entries = [e for e in entries if e.level >= level]

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        if trace_id:
            entries = [e for e in entries if e.trace_id == trace_id]

        return entries
