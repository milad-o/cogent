"""HTTP Webhook event sink.

Sends events to HTTP endpoints via POST requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agenticflow.events.event import Event
from agenticflow.events.sinks.base import EventSink


@dataclass
class WebhookSink(EventSink):
    """Send events to HTTP endpoints via POST.

    Delivers events to external systems by making HTTP POST requests
    with the event data as JSON body.

    Attributes:
        url: Target URL to POST events to
        headers: Additional HTTP headers (e.g., Authorization)
        timeout: Request timeout in seconds (default: 30)
        include_headers: Event headers to include in request headers

    Example:
        ```python
        sink = WebhookSink(
            url="https://example.com/events",
            headers={"Authorization": "Bearer token123"},
        )
        flow.sink(sink, pattern="order.*")

        # When order.created event occurs, POST to example.com/events:
        # {"name": "order.created", "data": {...}, "id": "...", ...}
        ```

    Request Format:
        POST /events
        Content-Type: application/json
        X-Event-Name: order.created
        X-Event-Id: evt_abc123

        {
            "name": "order.created",
            "data": {"order_id": "123", ...},
            "id": "evt_abc123",
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "..."
        }
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    include_event_headers: bool = True

    _client: Any = field(default=None, repr=False)

    async def _get_client(self) -> Any:
        """Get or create the HTTP client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "WebhookSink requires 'httpx'. "
                    "Install with: uv add httpx"
                ) from e
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def send(self, event: Event) -> None:
        """Send event to the webhook URL.

        Args:
            event: The event to send.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        client = await self._get_client()

        # Build headers
        headers = {
            "Content-Type": "application/json",
            **self.headers,
        }
        if self.include_event_headers:
            headers["X-Event-Name"] = event.name
            headers["X-Event-Id"] = event.id
            if event.correlation_id:
                headers["X-Correlation-Id"] = event.correlation_id

        # Build payload
        payload = {
            "name": event.name,
            "data": event.data,
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "correlation_id": event.correlation_id,
        }

        response = await client.post(
            self.url,
            content=json.dumps(payload, default=str),
            headers=headers,
        )
        response.raise_for_status()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def name(self) -> str:
        """Human-readable name for this sink."""
        return f"WebhookSink({self.url})"
