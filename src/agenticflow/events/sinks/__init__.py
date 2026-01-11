"""Event sinks for outbound event delivery.

This package provides adapters for sending events from EventFlow
to external systems (webhooks, message queues, etc.).

Available sinks:
- WebhookSink: Send events to HTTP endpoints

Example:
    ```python
    from agenticflow.reactive import EventFlow
    from agenticflow.events.sinks import WebhookSink

    flow = EventFlow()
    flow.sink(WebhookSink(url="https://example.com/callback"), pattern="*.completed")

    # When any .completed event occurs, it's sent to the webhook
    await flow.run("Process task")
    ```
"""

from agenticflow.events.sinks.base import EventSink
from agenticflow.events.sinks.webhook import WebhookSink

__all__ = [
    "EventSink",
    "WebhookSink",
]
