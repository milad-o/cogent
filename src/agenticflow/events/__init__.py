"""Core eventing (non-observability).

This module is the foundation for orchestration features (e.g., event-driven flows).
It is intentionally separate from `agenticflow.observability`, which is reserved
for tracing/telemetry and developer-facing instrumentation.

Includes:
- Event: Immutable event records
- EventBus: Core pub/sub for orchestration events
- EventStore: Persistent event storage for sourcing and replay
- Pattern matching: Utilities for event filtering
- Pattern parsing: Parse event@source syntax
- Standards: Semantic event name conventions
- Sources: External event sources (webhooks, file watchers, queues)
- Sinks: Outbound event sinks (webhooks, queues)
"""

from agenticflow.events.bus import EventBus
from agenticflow.events.event import Event
from agenticflow.events.patterns import (
    EventCondition,
    EventMatcher,
    EventPattern,
    SourceFilter,
    after,
    all_of,
    any_of,
    any_source,
    from_source,
    has_data,
    matching_sources,
    matches,
    matches_event,
    not_,
    not_from_source,
)
from agenticflow.events.standards import (
    AgentEvents,
    BatchEvents,
    DeploymentEvents,
    FlowEvents,
    IncidentEvents,
    ReviewEvents,
    TaskEvents,
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
from agenticflow.flow.parser import ParsedPattern, parse_pattern

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
    "not_from_source",
    "any_source",
    "matching_sources",
    "SourceFilter",
    "after",
    "has_data",
    "all_of",
    "any_of",
    "not_",
    # Pattern parsing
    "parse_pattern",
    "ParsedPattern",
    # Sources
    "EventSource",
    "WebhookSource",
    "FileWatcherSource",
    "RedisStreamSource",
    # Sinks
    "EventSink",
    "WebhookSink",
]
