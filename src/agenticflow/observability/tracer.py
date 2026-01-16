"""Distributed tracing for multi-agent systems.

Trace execution flow across agents, tasks, and tool calls
with spans that can be visualized and analyzed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticflow.core import generate_id


class SpanKind(Enum):
    """Type of operation being traced."""

    AGENT = "agent"
    TASK = "task"
    TOOL = "tool"
    LLM = "llm"
    EVENT = "event"
    TOPOLOGY = "topology"
    MEMORY = "memory"
    INTERNAL = "internal"


@dataclass
class SpanContext:
    """Context propagated through span hierarchy.

    Allows correlation of related spans across the system.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def child_context(self, new_span_id: str) -> SpanContext:
        """Create child context for nested span.

        Args:
            new_span_id: ID for the child span.

        Returns:
            New context with this span as parent.
        """
        return SpanContext(
            trace_id=self.trace_id,
            span_id=new_span_id,
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
        )


@dataclass
class Span:
    """A single traced operation.

    Spans track timing, status, and attributes of operations
    within the system.
    """

    name: str
    kind: SpanKind
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def is_finished(self) -> bool:
        """Check if span has been finished."""
        return self.end_time is not None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute.

        Args:
            key: Attribute name.
            value: Attribute value.
        """
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the span.

        Args:
            name: Event name.
            attributes: Optional event attributes.
        """
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def finish(self, status: str = "ok", error: str | None = None) -> None:
        """Finish the span.

        Args:
            status: Final status ("ok", "error", "cancelled").
            error: Error message if status is "error".
        """
        self.end_time = time.time()
        self.status = status
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """Distributed tracer for the agent system.

    Creates and manages spans to track execution flow
    across components.

    Example:
        >>> tracer = Tracer("my-service")
        >>> with tracer.span("process_task", SpanKind.TASK) as span:
        ...     span.set_attribute("task_id", "123")
        ...     # do work
        ...     span.add_event("checkpoint_reached")
    """

    def __init__(
        self,
        service_name: str,
        on_span_finish: Callable[[Span], None] | None = None,
    ) -> None:
        """Initialize tracer.

        Args:
            service_name: Name of the service being traced.
            on_span_finish: Optional callback when spans finish.
        """
        self.service_name = service_name
        self.on_span_finish = on_span_finish
        self._active_spans: dict[str, Span] = {}
        self._finished_spans: list[Span] = []
        self._current_context: SpanContext | None = None

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_context: SpanContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span]:
        """Create and manage a span context.

        Args:
            name: Span name.
            kind: Type of operation.
            parent_context: Optional parent context.
            attributes: Initial attributes.

        Yields:
            The active span.
        """
        # Determine parent
        parent = parent_context or self._current_context

        # Create context
        span_id = generate_id()
        if parent:
            context = parent.child_context(span_id)
        else:
            context = SpanContext(
                trace_id=generate_id(12),
                span_id=span_id,
            )

        # Create span
        span = Span(
            name=name,
            kind=kind,
            context=context,
            attributes=attributes or {},
        )
        span.set_attribute("service", self.service_name)

        # Track active span
        self._active_spans[span_id] = span
        previous_context = self._current_context
        self._current_context = context

        try:
            yield span
            if not span.is_finished:
                span.finish("ok")
        except Exception as e:
            span.finish("error", str(e))
            raise
        finally:
            # Cleanup
            self._active_spans.pop(span_id, None)
            self._current_context = previous_context
            self._finished_spans.append(span)

            # Callback
            if self.on_span_finish:
                self.on_span_finish(span)

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_context: SpanContext | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a span manually (must be finished explicitly).

        Args:
            name: Span name.
            kind: Type of operation.
            parent_context: Optional parent context.
            attributes: Initial attributes.

        Returns:
            The new span.
        """
        parent = parent_context or self._current_context

        span_id = generate_id()
        if parent:
            context = parent.child_context(span_id)
        else:
            context = SpanContext(
                trace_id=generate_id(12),
                span_id=span_id,
            )

        span = Span(
            name=name,
            kind=kind,
            context=context,
            attributes=attributes or {},
        )
        span.set_attribute("service", self.service_name)

        self._active_spans[span_id] = span
        return span

    def finish_span(self, span: Span, status: str = "ok", error: str | None = None) -> None:
        """Finish a manually started span.

        Args:
            span: The span to finish.
            status: Final status.
            error: Error message if any.
        """
        span.finish(status, error)
        self._active_spans.pop(span.context.span_id, None)
        self._finished_spans.append(span)

        if self.on_span_finish:
            self.on_span_finish(span)

    @property
    def current_context(self) -> SpanContext | None:
        """Get current active span context."""
        return self._current_context

    @property
    def active_spans(self) -> list[Span]:
        """Get list of currently active spans."""
        return list(self._active_spans.values())

    @property
    def finished_spans(self) -> list[Span]:
        """Get list of finished spans."""
        return self._finished_spans.copy()

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace.

        Args:
            trace_id: The trace ID.

        Returns:
            List of spans in the trace.
        """
        return [
            s for s in self._finished_spans
            if s.context.trace_id == trace_id
        ]

    def clear_finished(self) -> None:
        """Clear finished spans from memory."""
        self._finished_spans.clear()

    def export(self) -> list[dict[str, Any]]:
        """Export all finished spans as dictionaries.

        Returns:
            List of span dictionaries.
        """
        return [s.to_dict() for s in self._finished_spans]
