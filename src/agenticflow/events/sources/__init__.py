"""External event sources for event-driven orchestration.

This package provides adapters for ingesting events from external systems
into Flow for event-driven processing.

Available sources:
- WebhookSource: Receive events via HTTP webhooks
- FileWatcherSource: Watch directories for file changes
- RedisStreamSource: Consume events from Redis Streams

Example:
    ```python
    from agenticflow.flow import Flow
    from agenticflow.events.sources import WebhookSource, FileWatcherSource

    flow = Flow()
    flow.source(WebhookSource(path="/events", port=8080))
    flow.source(FileWatcherSource(paths=["./incoming"], patterns=["*.json"]))

    # Flow will now react to external events
    await flow.run("Process incoming events")
    ```
"""

from agenticflow.events.sources.base import EventSource
from agenticflow.events.sources.file_watcher import FileWatcherSource
from agenticflow.events.sources.redis_stream import RedisStreamSource
from agenticflow.events.sources.webhook import WebhookSource

__all__ = [
    "EventSource",
    "WebhookSource",
    "FileWatcherSource",
    "RedisStreamSource",
]
