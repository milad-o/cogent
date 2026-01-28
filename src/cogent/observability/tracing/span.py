"""
Span - Represents a unit of work in a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    pass


class SpanStatus(StrEnum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """
    Context for distributed tracing.

    Contains IDs that can be propagated across service boundaries.
    """

    trace_id: str
    """Unique ID for the entire trace."""

    span_id: str
    """Unique ID for this span."""

    parent_span_id: str | None = None
    """ID of the parent span, if any."""

    @classmethod
    def create(cls, parent: SpanContext | None = None) -> SpanContext:
        """Create a new span context, optionally as child of parent."""
        trace_id = parent.trace_id if parent else uuid4().hex[:16]
        return cls(
            trace_id=trace_id,
            span_id=uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else None,
        )


@dataclass
class Span:
    """
    A span represents a single unit of work.

    Spans can be nested to form a trace tree. Use as a context
    manager for automatic timing and status handling.

    Example:
        ```python
        with Span(name="process_request") as span:
            span.set_attribute("user_id", 123)
            result = do_work()
            span.set_attribute("result_size", len(result))
        # Span automatically ends with OK status
        ```
    """

    name: str
    """Name of the span (e.g., "agent.run", "tool.execute")."""

    context: SpanContext = field(default_factory=lambda: SpanContext.create())
    """Tracing context with IDs."""

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the span started."""

    end_time: datetime | None = None
    """When the span ended (None if still running)."""

    status: SpanStatus = SpanStatus.UNSET
    """Current status of the span."""

    attributes: dict[str, object] = field(default_factory=dict)
    """Key-value attributes attached to the span."""

    events: list[dict[str, object]] = field(default_factory=list)
    """Events that occurred during the span."""

    error: str | None = None
    """Error message if status is ERROR."""

    # === Mutators ===

    def set_attribute(self, key: str, value: object) -> Span:
        """
        Set an attribute on the span.

        Args:
            key: Attribute name
            value: Attribute value

        Returns:
            Self for chaining
        """
        self.attributes[key] = value
        return self

    def set_attributes(self, **attrs: object) -> Span:
        """
        Set multiple attributes.

        Args:
            **attrs: Attributes as keyword arguments

        Returns:
            Self for chaining
        """
        self.attributes.update(attrs)
        return self

    def add_event(self, name: str, **attributes: object) -> Span:
        """
        Record an event during the span.

        Args:
            name: Event name
            **attributes: Event attributes

        Returns:
            Self for chaining
        """
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(UTC).isoformat(),
                **attributes,
            }
        )
        return self

    def set_status(self, status: SpanStatus, error: str | None = None) -> Span:
        """
        Set the span status.

        Args:
            status: New status
            error: Error message (for ERROR status)

        Returns:
            Self for chaining
        """
        self.status = status
        if error:
            self.error = error
        return self

    def end(self, status: SpanStatus | None = None) -> Span:
        """
        End the span.

        Args:
            status: Optional status to set

        Returns:
            Self for chaining
        """
        self.end_time = datetime.now(UTC)
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
        return self

    # === Properties ===

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds, or None if not ended."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def is_recording(self) -> bool:
        """Whether the span is still recording (not ended)."""
        return self.end_time is None

    @property
    def trace_id(self) -> str:
        """Shortcut to context.trace_id."""
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        """Shortcut to context.span_id."""
        return self.context.span_id

    @property
    def parent_span_id(self) -> str | None:
        """Shortcut to context.parent_span_id."""
        return self.context.parent_span_id

    # === Context Manager ===

    def __enter__(self) -> Span:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, auto-ending span."""
        if exc_val is not None:
            self.set_status(SpanStatus.ERROR, str(exc_val))
        self.end()

    # === Factory ===

    @classmethod
    def create(
        cls,
        name: str,
        parent: Span | SpanContext | None = None,
        **attributes: object,
    ) -> Span:
        """
        Create a new span.

        Args:
            name: Span name
            parent: Parent span or context
            **attributes: Initial attributes

        Returns:
            New span
        """
        if isinstance(parent, Span):
            context = SpanContext.create(parent.context)
        elif isinstance(parent, SpanContext):
            context = SpanContext.create(parent)
        else:
            context = SpanContext.create()

        return cls(
            name=name,
            context=context,
            attributes=dict(attributes),
        )
