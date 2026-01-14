"""HTTP Webhook event source.

Receives events via HTTP POST requests and emits them into EventFlow.
Uses Starlette for ASGI compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agenticflow.events.event import Event
from agenticflow.events.sources.base import EventSource, EmitCallback


@dataclass
class WebhookSource(EventSource):
    """Receive events via HTTP webhooks.

    Starts an HTTP server that listens for POST requests and converts
    them into events for the flow.

    Attributes:
        path: URL path to listen on (default: "/events")
        host: Host to bind to (default: "0.0.0.0")
        port: Port to listen on (default: 8080)
        event_name: Event name to emit (default: "webhook.received")
        event_name_header: HTTP header to read event name from (optional)
        secret: Shared secret for request validation (optional)

    Example:
        ```python
        source = WebhookSource(
            path="/api/events",
            port=8080,
            event_name="external.webhook",
        )
        flow.source(source)

        # External systems can POST to http://host:8080/api/events
        # with JSON body, which becomes event.data
        ```

    Request Format:
        POST /events
        Content-Type: application/json
        X-Event-Name: custom.event  # Optional, overrides event_name

        {"key": "value", ...}
    """

    path: str = "/events"
    host: str = "0.0.0.0"
    port: int = 8080
    event_name: str = "webhook.received"
    event_name_header: str = "X-Event-Name"
    secret: str | None = None

    _server: Any = field(default=None, repr=False)
    _emit: EmitCallback | None = field(default=None, repr=False)

    async def start(self, emit: EmitCallback) -> None:
        """Start the HTTP server.

        Args:
            emit: Callback to emit events into the flow.
        """
        self._emit = emit

        try:
            from starlette.applications import Starlette
            from starlette.routing import Route
            from starlette.responses import JSONResponse
            from starlette.requests import Request
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "WebhookSource requires 'starlette' and 'uvicorn'. "
                "Install with: uv add starlette uvicorn"
            ) from e

        async def handle_webhook(request: Request) -> JSONResponse:
            """Handle incoming webhook requests."""
            # Validate secret if configured
            if self.secret:
                auth = request.headers.get("Authorization", "")
                if auth != f"Bearer {self.secret}":
                    return JSONResponse(
                        {"error": "Unauthorized"},
                        status_code=401,
                    )

            # Parse request body
            try:
                data = await request.json()
            except json.JSONDecodeError:
                data = {"raw_body": (await request.body()).decode("utf-8", errors="replace")}

            # Determine event name
            name = request.headers.get(self.event_name_header, self.event_name)

            # Create and emit event
            event = Event(
                name=name,
                data=data,
                source=f"webhook:{self.path}",
            )

            if self._emit:
                await self._emit(event)

            return JSONResponse({"status": "accepted", "event_id": event.id})

        # Create Starlette app
        app = Starlette(
            routes=[Route(self.path, handle_webhook, methods=["POST"])],
        )

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        # Run server (blocks until stop() is called)
        await self._server.serve()

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.should_exit = True

    @property
    def name(self) -> str:
        """Human-readable name for this source."""
        return f"WebhookSource({self.host}:{self.port}{self.path})"
