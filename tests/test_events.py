"""
Tests for EventBus.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from agenticflow.core.enums import EventType
from agenticflow.observability.bus import EventBus, get_event_bus, set_event_bus
from agenticflow.observability.event import Event


class TestEventBus:
    """Tests for EventBus."""

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def sample_event(self) -> Event:
        return Event(
            type=EventType.TASK_CREATED,
            data={"name": "test"},
            correlation_id="corr-123",
        )

    async def test_subscribe_and_publish(self, event_bus: EventBus) -> None:
        handler = Mock()
        event_bus.subscribe(EventType.TASK_CREATED, handler)

        event = Event(type=EventType.TASK_CREATED)
        await event_bus.publish(event)

        handler.assert_called_once_with(event)

    async def test_subscribe_all(self, event_bus: EventBus) -> None:
        handler = Mock()
        event_bus.subscribe_all(handler)

        event1 = Event(type=EventType.TASK_CREATED)
        event2 = Event(type=EventType.AGENT_INVOKED)

        await event_bus.publish(event1)
        await event_bus.publish(event2)

        assert handler.call_count == 2

    async def test_async_handler(self, event_bus: EventBus) -> None:
        handler = AsyncMock()
        event_bus.subscribe(EventType.TASK_STARTED, handler)

        event = Event(type=EventType.TASK_STARTED)
        await event_bus.publish(event)

        handler.assert_awaited_once_with(event)

    async def test_unsubscribe(self, event_bus: EventBus) -> None:
        handler = Mock()
        event_bus.subscribe(EventType.TASK_CREATED, handler)
        event_bus.unsubscribe(EventType.TASK_CREATED, handler)

        await event_bus.publish(Event(type=EventType.TASK_CREATED))

        handler.assert_not_called()

    async def test_subscribe_many(self, event_bus: EventBus) -> None:
        handler = Mock()
        event_bus.subscribe_many(
            [EventType.TASK_CREATED, EventType.TASK_COMPLETED],
            handler,
        )

        await event_bus.publish(Event(type=EventType.TASK_CREATED))
        await event_bus.publish(Event(type=EventType.TASK_COMPLETED))
        await event_bus.publish(Event(type=EventType.AGENT_INVOKED))

        assert handler.call_count == 2

    async def test_event_history(
        self, event_bus: EventBus, sample_event: Event
    ) -> None:
        await event_bus.publish(sample_event)

        history = event_bus.get_history()
        assert len(history) == 1
        assert history[0].id == sample_event.id

    async def test_history_filter_by_type(self, event_bus: EventBus) -> None:
        await event_bus.publish(Event(type=EventType.TASK_CREATED))
        await event_bus.publish(Event(type=EventType.TASK_COMPLETED))
        await event_bus.publish(Event(type=EventType.TASK_CREATED))

        history = event_bus.get_history(event_type=EventType.TASK_CREATED)
        assert len(history) == 2

    async def test_history_filter_by_correlation_id(
        self, event_bus: EventBus
    ) -> None:
        await event_bus.publish(
            Event(type=EventType.TASK_CREATED, correlation_id="a")
        )
        await event_bus.publish(
            Event(type=EventType.TASK_CREATED, correlation_id="b")
        )
        await event_bus.publish(
            Event(type=EventType.TASK_COMPLETED, correlation_id="a")
        )

        history = event_bus.get_history(correlation_id="a")
        assert len(history) == 2

    async def test_history_limit(self, event_bus: EventBus) -> None:
        for i in range(10):
            await event_bus.publish(Event(type=EventType.TASK_CREATED))

        history = event_bus.get_history(limit=5)
        assert len(history) == 5

    async def test_history_max_size(self) -> None:
        bus = EventBus(max_history=5)

        for i in range(10):
            await bus.publish(Event(type=EventType.TASK_CREATED))

        assert bus.history_size == 5

    def test_clear_history(self, event_bus: EventBus) -> None:
        event_bus._event_history.append(Event(type=EventType.TASK_CREATED))
        event_bus.clear_history()
        assert event_bus.history_size == 0

    def test_clear_subscriptions(self, event_bus: EventBus) -> None:
        handler = Mock()
        event_bus.subscribe(EventType.TASK_CREATED, handler)
        event_bus.subscribe_all(handler)

        event_bus.clear_subscriptions()

        assert len(event_bus._handlers) == 0
        assert len(event_bus._global_handlers) == 0

    async def test_get_stats(self, event_bus: EventBus) -> None:
        await event_bus.publish(Event(type=EventType.TASK_CREATED))
        await event_bus.publish(Event(type=EventType.TASK_CREATED))
        await event_bus.publish(Event(type=EventType.TASK_COMPLETED))

        stats = event_bus.get_stats()
        assert stats["history_size"] == 3
        assert stats["event_type_counts"]["task.created"] == 2
        assert stats["event_type_counts"]["task.completed"] == 1

    async def test_handler_error_doesnt_break_bus(
        self, event_bus: EventBus
    ) -> None:
        def bad_handler(event: Event) -> None:
            raise RuntimeError("Handler failed!")

        good_handler = Mock()

        event_bus.subscribe_all(bad_handler)
        event_bus.subscribe_all(good_handler)

        # Should not raise
        await event_bus.publish(Event(type=EventType.TASK_CREATED))

        # Good handler should still be called
        good_handler.assert_called_once()


class TestGlobalEventBus:
    """Tests for global event bus functions."""

    def test_get_event_bus_creates_singleton(self) -> None:
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_set_event_bus(self) -> None:
        custom_bus = EventBus()
        set_event_bus(custom_bus)
        assert get_event_bus() is custom_bus
