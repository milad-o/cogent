"""
Tests for TraceBus.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from agenticflow.observability.trace_record import TraceType
from agenticflow.observability.bus import TraceBus, get_trace_bus, set_trace_bus
from agenticflow.observability.trace_record import Trace


class TestTraceBus:
    """Tests for TraceBus."""

    @pytest.fixture
    def trace_bus(self) -> TraceBus:
        return TraceBus()

    @pytest.fixture
    def sample_trace(self) -> Trace:
        return Trace(
            type=TraceType.TASK_CREATED,
            data={"name": "test"},
            correlation_id="corr-123",
        )

    async def test_subscribe_and_publish(self, trace_bus: TraceBus) -> None:
        handler = Mock()
        trace_bus.subscribe(TraceType.TASK_CREATED, handler)

        event = Trace(type=TraceType.TASK_CREATED)
        await trace_bus.publish(event)

        handler.assert_called_once_with(event)

    async def test_subscribe_all(self, trace_bus: TraceBus) -> None:
        handler = Mock()
        trace_bus.subscribe_all(handler)

        event1 = Trace(type=TraceType.TASK_CREATED)
        event2 = Trace(type=TraceType.AGENT_INVOKED)

        await trace_bus.publish(event1)
        await trace_bus.publish(event2)

        assert handler.call_count == 2

    async def test_async_handler(self, trace_bus: TraceBus) -> None:
        handler = AsyncMock()
        trace_bus.subscribe(TraceType.TASK_STARTED, handler)

        event = Trace(type=TraceType.TASK_STARTED)
        await trace_bus.publish(event)

        handler.assert_awaited_once_with(event)

    async def test_unsubscribe(self, trace_bus: TraceBus) -> None:
        handler = Mock()
        trace_bus.subscribe(TraceType.TASK_CREATED, handler)
        trace_bus.unsubscribe(TraceType.TASK_CREATED, handler)

        await trace_bus.publish(Trace(type=TraceType.TASK_CREATED))

        handler.assert_not_called()

    async def test_subscribe_many(self, trace_bus: TraceBus) -> None:
        handler = Mock()
        trace_bus.subscribe_many(
            [TraceType.TASK_CREATED, TraceType.TASK_COMPLETED],
            handler,
        )

        await trace_bus.publish(Trace(type=TraceType.TASK_CREATED))
        await trace_bus.publish(Trace(type=TraceType.TASK_COMPLETED))
        await trace_bus.publish(Trace(type=TraceType.AGENT_INVOKED))

        assert handler.call_count == 2

    async def test_event_history(
        self, trace_bus: TraceBus, sample_trace: Trace
    ) -> None:
        await trace_bus.publish(sample_trace)

        history = trace_bus.get_history()
        assert len(history) == 1
        assert history[0].id == sample_trace.id

    async def test_history_filter_by_type(self, trace_bus: TraceBus) -> None:
        await trace_bus.publish(Trace(type=TraceType.TASK_CREATED))
        await trace_bus.publish(Trace(type=TraceType.TASK_COMPLETED))
        await trace_bus.publish(Trace(type=TraceType.TASK_CREATED))

        history = trace_bus.get_history(event_type=TraceType.TASK_CREATED)
        assert len(history) == 2

    async def test_history_filter_by_correlation_id(
        self, trace_bus: TraceBus
    ) -> None:
        await trace_bus.publish(
            Trace(type=TraceType.TASK_CREATED, correlation_id="a")
        )
        await trace_bus.publish(
            Trace(type=TraceType.TASK_CREATED, correlation_id="b")
        )
        await trace_bus.publish(
            Trace(type=TraceType.TASK_COMPLETED, correlation_id="a")
        )

        history = trace_bus.get_history(correlation_id="a")
        assert len(history) == 2

    async def test_history_limit(self, trace_bus: TraceBus) -> None:
        for i in range(10):
            await trace_bus.publish(Trace(type=TraceType.TASK_CREATED))

        history = trace_bus.get_history(limit=5)
        assert len(history) == 5

    async def test_history_max_size(self) -> None:
        bus = TraceBus(max_history=5)

        for i in range(10):
            await bus.publish(Trace(type=TraceType.TASK_CREATED))

        assert bus.history_size == 5

    def test_clear_history(self, trace_bus: TraceBus) -> None:
        trace_bus._event_history.append(Trace(type=TraceType.TASK_CREATED))
        trace_bus.clear_history()
        assert trace_bus.history_size == 0

    def test_clear_subscriptions(self, trace_bus: TraceBus) -> None:
        handler = Mock()
        trace_bus.subscribe(TraceType.TASK_CREATED, handler)
        trace_bus.subscribe_all(handler)

        trace_bus.clear_subscriptions()

        assert len(trace_bus._handlers) == 0
        assert len(trace_bus._global_handlers) == 0

    async def test_get_stats(self, trace_bus: TraceBus) -> None:
        await trace_bus.publish(Trace(type=TraceType.TASK_CREATED))
        await trace_bus.publish(Trace(type=TraceType.TASK_CREATED))
        await trace_bus.publish(Trace(type=TraceType.TASK_COMPLETED))

        stats = trace_bus.get_stats()
        assert stats["history_size"] == 3
        assert stats["event_type_counts"]["task.created"] == 2
        assert stats["event_type_counts"]["task.completed"] == 1

    async def test_handler_error_doesnt_break_bus(
        self, trace_bus: TraceBus
    ) -> None:
        def bad_handler(trace: Trace) -> None:
            raise RuntimeError("Handler failed!")

        good_handler = Mock()

        trace_bus.subscribe_all(bad_handler)
        trace_bus.subscribe_all(good_handler)

        # Should not raise
        await trace_bus.publish(Trace(type=TraceType.TASK_CREATED))

        # Good handler should still be called
        good_handler.assert_called_once()


class TestGlobalTraceBus:
    """Tests for global event bus functions."""

    def test_get_trace_bus_creates_singleton(self) -> None:
        bus1 = get_trace_bus()
        bus2 = get_trace_bus()
        assert bus1 is bus2

    def test_set_trace_bus(self) -> None:
        custom_bus = TraceBus()
        set_trace_bus(custom_bus)
        assert get_trace_bus() is custom_bus
