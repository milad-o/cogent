"""
Tracer - Creates and manages spans.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from cogent.observability.tracing.span import Span, SpanContext, SpanStatus

if TYPE_CHECKING:
    from cogent.observability.observer import Observer


# Context variable to track current span
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


class Tracer:
    """
    Creates and manages spans for tracing.

    Maintains a stack of active spans and integrates with
    Observer for event emission.

    Example:
        ```python
        tracer = Tracer(observer=observer)

        with tracer.span("process_request") as span:
            span.set_attribute("path", "/api/users")

            with tracer.span("db_query") as child:
                result = db.query(...)
                child.set_attribute("row_count", len(result))
        ```
    """

    def __init__(
        self,
        observer: Observer | None = None,
        *,
        service_name: str = "cogent",
    ) -> None:
        """
        Initialize tracer.

        Args:
            observer: Observer for emitting span events
            service_name: Name of the service (for span attributes)
        """
        self._observer = observer
        self._service_name = service_name

    @contextmanager
    def span(
        self,
        name: str,
        **attributes: object,
    ) -> Generator[Span]:
        """
        Create a span as a context manager.

        Automatically:
        - Sets parent from current context
        - Emits span.start and span.end events
        - Sets status based on exceptions

        Args:
            name: Span name
            **attributes: Initial attributes

        Yields:
            The created span
        """
        parent = _current_span.get()

        span = Span.create(
            name=name,
            parent=parent,
            service=self._service_name,
            **attributes,
        )

        # Set as current span
        token = _current_span.set(span)

        # Emit start event
        if self._observer:
            self._observer.emit(
                "span.start",
                span_name=name,
                trace_id=span.trace_id,
                span_id=span.span_id,
                parent_span_id=span.parent_span_id,
                attributes=span.attributes,
            )

        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            span.end()

            # Emit end event
            if self._observer:
                self._observer.emit(
                    "span.end",
                    span_name=name,
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    duration_ms=span.duration_ms,
                    status=span.status,
                    error=span.error,
                    attributes=span.attributes,
                )

            # Restore previous span
            _current_span.reset(token)

    def start_span(
        self,
        name: str,
        **attributes: object,
    ) -> Span:
        """
        Start a span without context manager.

        Remember to call span.end() when done.

        Args:
            name: Span name
            **attributes: Initial attributes

        Returns:
            The created span
        """
        parent = _current_span.get()

        span = Span.create(
            name=name,
            parent=parent,
            service=self._service_name,
            **attributes,
        )

        _current_span.set(span)

        if self._observer:
            self._observer.emit(
                "span.start",
                span_name=name,
                trace_id=span.trace_id,
                span_id=span.span_id,
                parent_span_id=span.parent_span_id,
            )

        return span

    def end_span(self, span: Span) -> None:
        """
        End a span started with start_span.

        Args:
            span: Span to end
        """
        span.end()

        if self._observer:
            self._observer.emit(
                "span.end",
                span_name=span.name,
                trace_id=span.trace_id,
                span_id=span.span_id,
                duration_ms=span.duration_ms,
                status=span.status,
                error=span.error,
            )

        # Try to restore parent span
        if span.parent_span_id:
            # This is simplified - in production, maintain a proper span stack
            _current_span.set(None)

    @property
    def current_span(self) -> Span | None:
        """Get the current active span."""
        return _current_span.get()

    @property
    def current_context(self) -> SpanContext | None:
        """Get the context of the current span."""
        span = _current_span.get()
        return span.context if span else None


# === Module-level API ===


_default_tracer: Tracer | None = None


def get_tracer(observer: Observer | None = None) -> Tracer:
    """
    Get or create the default tracer.

    Args:
        observer: Observer to use (only used on first call)

    Returns:
        Default tracer instance
    """
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer(observer=observer)
    return _default_tracer


def current_span() -> Span | None:
    """Get the current active span."""
    return _current_span.get()


def current_trace_id() -> str | None:
    """Get the current trace ID."""
    span = _current_span.get()
    return span.trace_id if span else None
