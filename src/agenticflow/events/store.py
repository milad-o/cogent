"""Event store for persistence and replay.

Provides event sourcing capabilities: store events, replay flows,
rebuild state from event history.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from agenticflow.events.event import Event


class EventStore(ABC):
    """Abstract base for event persistence.

    Implementations can store events in memory, files, databases, etc.
    Enables event sourcing patterns: replay, audit, state reconstruction.

    Example:
        ```python
        store = InMemoryEventStore()

        # Append events
        await store.append(event, flow_id="flow-123")

        # Retrieve events
        events = await store.get_events(flow_id="flow-123")

        # Replay up to a point
        events = await store.replay(flow_id="flow-123", to_event_id="evt-456")
        ```
    """

    @abstractmethod
    async def append(self, event: Event, flow_id: str) -> None:
        """Append an event to the store.

        Args:
            event: The event to store
            flow_id: The flow execution this event belongs to
        """
        ...

    @abstractmethod
    async def get_events(
        self,
        flow_id: str,
        *,
        after: str | None = None,
        before: str | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Get events for a flow.

        Args:
            flow_id: The flow execution ID
            after: Return events after this event ID
            before: Return events before this event ID
            limit: Maximum number of events to return

        Returns:
            List of events in chronological order
        """
        ...

    @abstractmethod
    async def replay(
        self,
        flow_id: str,
        to_event_id: str | None = None,
    ) -> list[Event]:
        """Replay events for a flow up to a specific point.

        Args:
            flow_id: The flow execution ID
            to_event_id: Stop at this event ID (inclusive), or None for all

        Returns:
            List of events in chronological order
        """
        ...

    @abstractmethod
    async def get_flow_ids(self, *, limit: int = 100) -> list[str]:
        """Get recent flow IDs.

        Args:
            limit: Maximum number of flow IDs to return

        Returns:
            List of flow IDs, most recent first
        """
        ...

    async def stream_events(
        self,
        flow_id: str,
    ) -> AsyncIterator[Event]:
        """Stream events for a flow.

        Default implementation fetches all and yields.
        Subclasses can override for true streaming.
        """
        events = await self.get_events(flow_id)
        for event in events:
            yield event


class InMemoryEventStore(EventStore):
    """In-memory event store for development and testing.

    Events are lost when the process exits.
    """

    def __init__(self, max_events_per_flow: int = 10_000) -> None:
        self._events: dict[str, list[Event]] = defaultdict(list)
        self._flow_order: list[str] = []  # Track flow creation order
        self._max_events = max_events_per_flow

    async def append(self, event: Event, flow_id: str) -> None:
        if flow_id not in self._events:
            self._flow_order.append(flow_id)

        self._events[flow_id].append(event)

        # Trim if needed
        if len(self._events[flow_id]) > self._max_events:
            self._events[flow_id] = self._events[flow_id][-self._max_events:]

    async def get_events(
        self,
        flow_id: str,
        *,
        after: str | None = None,
        before: str | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        events = self._events.get(flow_id, [])

        # Filter by after
        if after:
            found = False
            filtered = []
            for e in events:
                if found:
                    filtered.append(e)
                elif e.id == after:
                    found = True
            events = filtered

        # Filter by before
        if before:
            filtered = []
            for e in events:
                if e.id == before:
                    break
                filtered.append(e)
            events = filtered

        # Apply limit
        if limit:
            events = events[:limit]

        return events

    async def replay(
        self,
        flow_id: str,
        to_event_id: str | None = None,
    ) -> list[Event]:
        events = self._events.get(flow_id, [])

        if to_event_id is None:
            return list(events)

        # Replay up to (inclusive) the target event
        result = []
        for e in events:
            result.append(e)
            if e.id == to_event_id:
                break

        return result

    async def get_flow_ids(self, *, limit: int = 100) -> list[str]:
        # Return most recent first
        return list(reversed(self._flow_order))[:limit]

    def clear(self) -> None:
        """Clear all stored events."""
        self._events.clear()
        self._flow_order.clear()


class FileEventStore(EventStore):
    """File-based event store using JSONL format.

    Each flow gets its own file: {base_dir}/{flow_id}.jsonl
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _flow_path(self, flow_id: str) -> Path:
        # Sanitize flow_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in flow_id)
        return self._base_dir / f"{safe_id}.jsonl"

    async def append(self, event: Event, flow_id: str) -> None:
        path = self._flow_path(flow_id)
        with open(path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    async def get_events(
        self,
        flow_id: str,
        *,
        after: str | None = None,
        before: str | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        path = self._flow_path(flow_id)
        if not path.exists():
            return []

        events = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    events.append(Event.from_dict(json.loads(line)))

        # Apply filters (same logic as InMemoryEventStore)
        if after:
            found = False
            filtered = []
            for e in events:
                if found:
                    filtered.append(e)
                elif e.id == after:
                    found = True
            events = filtered

        if before:
            filtered = []
            for e in events:
                if e.id == before:
                    break
                filtered.append(e)
            events = filtered

        if limit:
            events = events[:limit]

        return events

    async def replay(
        self,
        flow_id: str,
        to_event_id: str | None = None,
    ) -> list[Event]:
        events = await self.get_events(flow_id)

        if to_event_id is None:
            return events

        result = []
        for e in events:
            result.append(e)
            if e.id == to_event_id:
                break

        return result

    async def get_flow_ids(self, *, limit: int = 100) -> list[str]:
        flows = []
        for path in self._base_dir.glob("*.jsonl"):
            # Get modification time for sorting
            mtime = path.stat().st_mtime
            flows.append((mtime, path.stem))

        # Sort by modification time, most recent first
        flows.sort(reverse=True)
        return [flow_id for _, flow_id in flows[:limit]]


@dataclass
class EventStoreConfig:
    """Configuration for event stores."""

    type: str = "memory"
    """Store type: 'memory', 'file', 'redis', 'postgres'"""

    path: str | None = None
    """File path for file-based stores"""

    url: str | None = None
    """Connection URL for database stores"""

    max_events_per_flow: int = 10_000
    """Maximum events to keep per flow"""

    retention_days: int | None = None
    """Auto-delete events older than this (None = keep forever)"""


def create_event_store(config: EventStoreConfig | None = None) -> EventStore:
    """Factory function to create an event store.

    Args:
        config: Store configuration

    Returns:
        Configured EventStore instance
    """
    if config is None:
        return InMemoryEventStore()

    if config.type == "memory":
        return InMemoryEventStore(max_events_per_flow=config.max_events_per_flow)

    if config.type == "file":
        if not config.path:
            raise ValueError("FileEventStore requires 'path' configuration")
        return FileEventStore(base_dir=config.path)

    # Future: redis, postgres, etc.
    raise ValueError(f"Unknown event store type: {config.type}")
