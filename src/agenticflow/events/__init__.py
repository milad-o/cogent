"""Core eventing (non-observability).

This module is the foundation for orchestration features (e.g., reactive flows).
It is intentionally separate from `agenticflow.observability`, which is reserved
for tracing/telemetry and developer-facing instrumentation.

Includes:
- EventBus: Core pub/sub for orchestration events
- Event: Immutable event records
- sources: External event sources (webhooks, file watchers, queues)
- sinks: Outbound event sinks (webhooks, queues)
"""

from agenticflow.events.bus import EventBus
from agenticflow.events.event import Event
from agenticflow.events.sources import (
    EventSource,
    WebhookSource,
    FileWatcherSource,
    RedisStreamSource,
)
from agenticflow.events.sinks import (
    EventSink,
    WebhookSink,
)

__all__ = [
    "Event",
    "EventBus",
    # Sources
    "EventSource",
    "WebhookSource",
    "FileWatcherSource",
    "RedisStreamSource",
    # Sinks
    "EventSink",
    "WebhookSink",
]
