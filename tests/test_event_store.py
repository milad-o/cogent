"""Tests for event store implementations."""

import pytest
import tempfile
from pathlib import Path

from agenticflow.events import Event
from agenticflow.events.store import InMemoryEventStore, FileEventStore


class TestInMemoryEventStore:
    """Tests for InMemoryEventStore."""

    def test_creation(self) -> None:
        """InMemoryEventStore can be created."""
        store = InMemoryEventStore()
        assert store is not None

    def test_creation_with_max_events(self) -> None:
        """InMemoryEventStore accepts max_events_per_flow."""
        store = InMemoryEventStore(max_events_per_flow=100)
        assert store is not None

    @pytest.mark.asyncio
    async def test_append_event(self) -> None:
        """Events can be appended."""
        store = InMemoryEventStore()
        event = Event(name="test", source="test", data={"key": "value"})

        await store.append(event, flow_id="flow-123")

        events = await store.get_events(flow_id="flow-123")
        assert len(events) == 1
        assert events[0].name == "test"

    @pytest.mark.asyncio
    async def test_get_events_empty(self) -> None:
        """get_events returns empty list for unknown flow."""
        store = InMemoryEventStore()
        events = await store.get_events(flow_id="unknown")
        assert events == []

    @pytest.mark.asyncio
    async def test_get_events_multiple(self) -> None:
        """Multiple events can be retrieved."""
        store = InMemoryEventStore()

        await store.append(Event(name="e1", source="test"), flow_id="flow-123")
        await store.append(Event(name="e2", source="test"), flow_id="flow-123")
        await store.append(Event(name="e3", source="test"), flow_id="flow-123")

        events = await store.get_events(flow_id="flow-123")
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_get_events_with_limit(self) -> None:
        """get_events respects limit."""
        store = InMemoryEventStore()

        for i in range(10):
            await store.append(Event(name=f"e{i}", source="test"), flow_id="flow-123")

        events = await store.get_events(flow_id="flow-123", limit=5)
        assert len(events) == 5

    @pytest.mark.asyncio
    async def test_replay(self) -> None:
        """replay returns events for a flow."""
        store = InMemoryEventStore()

        e1 = Event(name="e1", source="test")
        e2 = Event(name="e2", source="test")
        e3 = Event(name="e3", source="test")

        await store.append(e1, flow_id="flow-123")
        await store.append(e2, flow_id="flow-123")
        await store.append(e3, flow_id="flow-123")

        events = await store.replay(flow_id="flow-123")
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_replay_to_event(self) -> None:
        """replay can stop at specific event."""
        store = InMemoryEventStore()

        e1 = Event(name="e1", source="test")
        e2 = Event(name="e2", source="test")
        e3 = Event(name="e3", source="test")

        await store.append(e1, flow_id="flow-123")
        await store.append(e2, flow_id="flow-123")
        await store.append(e3, flow_id="flow-123")

        events = await store.replay(flow_id="flow-123", to_event_id=e2.id)
        assert len(events) == 2
        assert events[-1].id == e2.id

    @pytest.mark.asyncio
    async def test_get_flow_ids(self) -> None:
        """get_flow_ids returns known flow IDs."""
        store = InMemoryEventStore()

        await store.append(Event(name="e1", source="test"), flow_id="flow-1")
        await store.append(Event(name="e2", source="test"), flow_id="flow-2")
        await store.append(Event(name="e3", source="test"), flow_id="flow-3")

        flow_ids = await store.get_flow_ids()
        assert "flow-1" in flow_ids
        assert "flow-2" in flow_ids
        assert "flow-3" in flow_ids

    def test_clear(self) -> None:
        """clear removes all events."""
        store = InMemoryEventStore()
        store.clear()
        # Should not raise

    @pytest.mark.asyncio
    async def test_isolation_between_flows(self) -> None:
        """Events are isolated by flow_id."""
        store = InMemoryEventStore()

        await store.append(Event(name="e1", source="test"), flow_id="flow-A")
        await store.append(Event(name="e2", source="test"), flow_id="flow-B")

        events_a = await store.get_events(flow_id="flow-A")
        events_b = await store.get_events(flow_id="flow-B")

        assert len(events_a) == 1
        assert len(events_b) == 1
        assert events_a[0].name == "e1"
        assert events_b[0].name == "e2"


class TestFileEventStore:
    """Tests for FileEventStore."""

    def test_creation(self) -> None:
        """FileEventStore can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileEventStore(base_dir=tmpdir)
            assert store is not None

    @pytest.mark.asyncio
    async def test_append_and_get(self) -> None:
        """Events can be appended and retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileEventStore(base_dir=tmpdir)

            event = Event(name="test", source="test", data={"key": "value"})
            await store.append(event, flow_id="flow-123")

            events = await store.get_events(flow_id="flow-123")
            assert len(events) == 1
            assert events[0].name == "test"

    @pytest.mark.asyncio
    async def test_persistence(self) -> None:
        """Events persist across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First store
            store1 = FileEventStore(base_dir=tmpdir)
            await store1.append(
                Event(name="e1", source="test"),
                flow_id="flow-123",
            )

            # Second store (same directory)
            store2 = FileEventStore(base_dir=tmpdir)
            events = await store2.get_events(flow_id="flow-123")

            assert len(events) == 1
            assert events[0].name == "e1"
