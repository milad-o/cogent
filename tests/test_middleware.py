"""Tests for middleware implementations."""

import pytest
import time
from datetime import datetime
from unittest.mock import MagicMock

from agenticflow.events import Event
from agenticflow.middleware import (
    BaseMiddleware,
    MiddlewareChain,
    LoggingMiddleware,
    VerboseMiddleware,
    RetryMiddleware,
    TimeoutMiddleware,
    TracingMiddleware,
    Span,
)


class TestBaseMiddleware:
    """Tests for BaseMiddleware."""

    @pytest.mark.asyncio
    async def test_default_before_returns_none(self) -> None:
        """Default before() returns None."""

        class TestMiddleware(BaseMiddleware):
            pass

        middleware = TestMiddleware()
        event = Event(name="test", source="test")
        reactor = MagicMock()

        result = await middleware.before(event, reactor)
        assert result is None

    @pytest.mark.asyncio
    async def test_default_after_returns_none(self) -> None:
        """Default after() returns None."""

        class TestMiddleware(BaseMiddleware):
            pass

        middleware = TestMiddleware()
        event = Event(name="test", source="test")
        reactor = MagicMock()

        result = await middleware.after(None, event, reactor)
        assert result is None


class TestMiddlewareChain:
    """Tests for MiddlewareChain."""

    def test_chain_creation(self) -> None:
        """MiddlewareChain can be created."""
        chain = MiddlewareChain()
        assert chain is not None

    def test_chain_add(self) -> None:
        """MiddlewareChain.add() adds middleware."""
        chain = MiddlewareChain()

        class TestMiddleware(BaseMiddleware):
            pass

        chain.add(TestMiddleware())
        assert len(chain) == 1

    def test_chain_multiple(self) -> None:
        """MiddlewareChain can hold multiple middleware."""
        chain = MiddlewareChain()

        class M1(BaseMiddleware):
            pass

        class M2(BaseMiddleware):
            pass

        chain.add(M1())
        chain.add(M2())
        assert len(chain) == 2


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    def test_creation(self) -> None:
        """LoggingMiddleware can be created."""
        middleware = LoggingMiddleware()
        assert middleware is not None

    def test_creation_with_level(self) -> None:
        """LoggingMiddleware accepts level parameter."""
        middleware = LoggingMiddleware(level="DEBUG")
        assert middleware is not None

    def test_creation_with_timing(self) -> None:
        """LoggingMiddleware accepts include_timing parameter."""
        middleware = LoggingMiddleware(include_timing=True)
        assert middleware is not None


class TestVerboseMiddleware:
    """Tests for VerboseMiddleware."""

    def test_creation(self) -> None:
        """VerboseMiddleware can be created."""
        middleware = VerboseMiddleware()
        assert middleware is not None


class TestRetryMiddleware:
    """Tests for RetryMiddleware."""

    def test_creation(self) -> None:
        """RetryMiddleware can be created."""
        middleware = RetryMiddleware()
        assert middleware is not None

    def test_creation_with_max_retries(self) -> None:
        """RetryMiddleware accepts max_retries parameter."""
        middleware = RetryMiddleware(max_retries=5)
        assert middleware is not None

    def test_creation_with_delay(self) -> None:
        """RetryMiddleware accepts delay parameters."""
        middleware = RetryMiddleware(
            base_delay=1.0,
            max_delay=30.0,
        )
        assert middleware is not None

    def test_creation_with_exponential(self) -> None:
        """RetryMiddleware accepts exponential parameter."""
        middleware = RetryMiddleware(exponential=True)
        assert middleware is not None


class TestTimeoutMiddleware:
    """Tests for TimeoutMiddleware."""

    def test_creation(self) -> None:
        """TimeoutMiddleware can be created."""
        middleware = TimeoutMiddleware()
        assert middleware is not None

    def test_creation_with_timeout(self) -> None:
        """TimeoutMiddleware accepts timeout parameter."""
        middleware = TimeoutMiddleware(timeout=60.0)
        assert middleware is not None

    def test_creation_with_per_reactor(self) -> None:
        """TimeoutMiddleware accepts per_reactor parameter."""
        middleware = TimeoutMiddleware(
            timeout=30.0,
            per_reactor={"slow_agent": 120.0},
        )
        assert middleware is not None


class TestTracingMiddleware:
    """Tests for TracingMiddleware."""

    def test_creation(self) -> None:
        """TracingMiddleware can be created."""
        middleware = TracingMiddleware()
        assert middleware is not None

    def test_creation_with_service_name(self) -> None:
        """TracingMiddleware accepts service_name parameter."""
        middleware = TracingMiddleware(service_name="my-flow")
        assert middleware.service_name == "my-flow"

    def test_spans_starts_empty(self) -> None:
        """TracingMiddleware.spans starts empty."""
        middleware = TracingMiddleware()
        assert middleware.spans == []

    def test_clear_spans(self) -> None:
        """TracingMiddleware.clear_spans() clears spans."""
        middleware = TracingMiddleware()
        middleware.clear_spans()
        assert middleware.spans == []


class TestSpan:
    """Tests for Span dataclass."""

    def test_creation(self) -> None:
        """Span can be created."""
        span = Span(
            name="test-span",
            trace_id="trace-123",
            span_id="span-456",
            parent_id=None,
            start_time=time.time(),
        )
        assert span.name == "test-span"
        assert span.trace_id == "trace-123"
        assert span.span_id == "span-456"

    def test_duration_unfinished(self) -> None:
        """Span.duration_ms is None when not finished."""
        span = Span(
            name="test",
            trace_id="t",
            span_id="s",
            parent_id=None,
            start_time=time.time(),
        )
        assert span.duration_ms is None

    def test_finish(self) -> None:
        """Span.finish() sets end_time."""
        span = Span(
            name="test",
            trace_id="t",
            span_id="s",
            parent_id=None,
            start_time=time.time(),
        )
        span.finish()
        assert span.end_time is not None

    def test_duration_after_finish(self) -> None:
        """Span.duration_ms works after finish."""
        span = Span(
            name="test",
            trace_id="t",
            span_id="s",
            parent_id=None,
            start_time=time.time(),
        )
        span.finish()
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_to_dict(self) -> None:
        """Span.to_dict() returns dictionary."""
        span = Span(
            name="test",
            trace_id="t",
            span_id="s",
            parent_id=None,
            start_time=time.time(),
        )
        d = span.to_dict()

        assert d["name"] == "test"
        assert d["trace_id"] == "t"
        assert d["span_id"] == "s"
