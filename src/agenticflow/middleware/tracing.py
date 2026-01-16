"""Tracing middleware for Flow execution.

Provides distributed tracing support for reactor execution.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.middleware.base import BaseMiddleware

if TYPE_CHECKING:
    from agenticflow.events import Event
    from agenticflow.reactors.base import Reactor


@dataclass
class Span:
    """A trace span representing a unit of work.

    Attributes:
        trace_id: Unique identifier for the trace
        span_id: Unique identifier for this span
        parent_id: Parent span ID (if any)
        name: Operation name
        start_time: Start timestamp (unix epoch)
        end_time: End timestamp (unix epoch)
        attributes: Additional span attributes
        status: Span status (ok, error)
    """

    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def finish(self, status: str = "ok") -> None:
        """Mark the span as finished."""
        self.end_time = time.time()
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
        }


@dataclass
class TracingMiddleware(BaseMiddleware):
    """Add distributed tracing to reactor execution.

    Creates spans for each reactor execution and propagates trace context
    through events.

    Attributes:
        service_name: Name of the service for traces
        span_exporter: Optional callback to export spans
        include_data: Whether to include event data in spans

    Example:
        ```python
        from agenticflow.middleware import TracingMiddleware

        flow = Flow()
        flow.use(TracingMiddleware(service_name="my-agent-flow"))
        ```

    Example with span exporter:
        ```python
        def export_to_jaeger(span: Span):
            # Send span to Jaeger/Zipkin/etc
            pass

        flow.use(TracingMiddleware(
            service_name="my-flow",
            span_exporter=export_to_jaeger,
        ))
        ```
    """

    service_name: str = "agenticflow"
    span_exporter: Any = None  # Callable[[Span], None]
    include_data: bool = False
    _active_spans: dict[str, Span] = field(default_factory=dict, init=False)
    _all_spans: list[Span] = field(default_factory=list, init=False)

    def _generate_id(self) -> str:
        """Generate a unique span/trace ID."""
        return uuid.uuid4().hex[:16]

    async def before(
        self,
        event: Event,
        reactor: Reactor,
    ) -> Event | None:
        """Start a span before reactor execution."""
        reactor_name = getattr(reactor, "name", type(reactor).__name__)

        # Get or create trace ID
        trace_id = event.data.get("_trace_id") or self._generate_id()
        parent_id = event.data.get("_span_id")
        span_id = self._generate_id()

        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=f"{self.service_name}.{reactor_name}",
            start_time=time.time(),
            attributes={
                "reactor.name": reactor_name,
                "event.type": event.type,
                "event.source": event.source,
            },
        )

        if self.include_data:
            span.attributes["event.data"] = event.data

        self._active_spans[event.id] = span

        # Propagate trace context in event
        event.data["_trace_id"] = trace_id
        event.data["_span_id"] = span_id

        return event

    async def after(
        self,
        result: Event | list[Event] | None,
        event: Event,
        reactor: Reactor,
    ) -> Event | list[Event] | None:
        """Finish span after reactor execution."""
        span = self._active_spans.pop(event.id, None)

        if span:
            span.finish("ok")
            self._all_spans.append(span)

            if self.span_exporter:
                self.span_exporter(span)

        # Propagate trace context to result events
        if result:
            trace_id = event.data.get("_trace_id")
            span_id = event.data.get("_span_id")

            if isinstance(result, list):
                for e in result:
                    e.data["_trace_id"] = trace_id
                    e.data["_parent_span_id"] = span_id
            else:
                result.data["_trace_id"] = trace_id
                result.data["_parent_span_id"] = span_id

        return result

    @property
    def spans(self) -> list[Span]:
        """Get all completed spans."""
        return list(self._all_spans)

    def clear_spans(self) -> None:
        """Clear collected spans."""
        self._all_spans.clear()


@dataclass
class SimpleTracingMiddleware(TracingMiddleware):
    """Simple tracing that just collects spans locally.

    Useful for debugging and testing.

    Example:
        ```python
        tracing = SimpleTracingMiddleware()
        flow.use(tracing)

        await flow.run("task")

        for span in tracing.spans:
            print(f"{span.name}: {span.duration_ms:.2f}ms")
        ```
    """

    include_data: bool = True
