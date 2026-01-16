"""Core eventing (non-observability).

This module is the foundation for orchestration features (e.g., reactive flows).
It is intentionally separate from `agenticflow.observability`, which is reserved
for tracing/telemetry and developer-facing instrumentation.

Includes:
- Event: Immutable event records
- EventBus: Core pub/sub for orchestration events
- EventStore: Persistent event storage for sourcing and replay
- Pattern matching: Utilities for event filtering
- Sources: External event sources (webhooks, file watchers, queues)
- Sinks: Outbound event sinks (webhooks, queues)
"""

from agenticflow.events.bus import EventBus
from agenticflow.events.event import Event
from agenticflow.events.patterns import (
    EventCondition,
    EventMatcher,
    EventPattern,
    after,
    all_of,
    any_of,
    from_source,
    has_data,
    matches,
    matches_event,
    not_,
)
from agenticflow.events.sinks import (
    EventSink,
    WebhookSink,
)
from agenticflow.events.sources import (
    EventSource,
    FileWatcherSource,
    RedisStreamSource,
    WebhookSource,
)
from agenticflow.events.store import (
    EventStore,
    EventStoreConfig,
    FileEventStore,
    InMemoryEventStore,
    create_event_store,
)

__all__ = [
    # Core
    "Event",
    "EventBus",
    # Store
    "EventStore",
    "InMemoryEventStore",
    "FileEventStore",
    "EventStoreConfig",
    "create_event_store",
    # Patterns
    "EventPattern",
    "EventCondition",
    "EventMatcher",
    "matches",
    "matches_event",
    "from_source",
    "after",
    "has_data",
    "all_of",
    "any_of",
    "not_",
    # Sources
    "EventSource",
    "WebhookSource",
    "FileWatcherSource",
    "RedisStreamSource",
    # Sinks
    "EventSink",
    "WebhookSink",
]
