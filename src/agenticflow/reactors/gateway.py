"""Gateway reactor - bridges external systems."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from agenticflow.events import Event
from agenticflow.reactors.base import BaseReactor

if TYPE_CHECKING:
    from agenticflow.flow.context import Context


class Gateway(BaseReactor):
    """Bridges external systems to the event flow.

    Gateways handle communication with external APIs, databases,
    message queues, and other systems.

    Example:
        ```python
        class WebhookGateway(Gateway):
            async def send(self, event: Event, ctx: Context) -> dict:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url, json=event.data) as resp:
                        return await resp.json()

        flow.register(
            WebhookGateway("https://api.example.com/webhook"),
            on="notification.send",
        )
        ```
    """

    def __init__(
        self,
        name: str | None = None,
        emit: str | None = None,
    ) -> None:
        """Initialize gateway.

        Args:
            name: Reactor name
            emit: Event name to emit on success (None = no emit)
        """
        super().__init__(name or "gateway")
        self._emit = emit

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | None:
        """Handle event by sending to external system."""
        try:
            result = await self.send(event, ctx)

            if self._emit:
                return Event(
                    name=self._emit,
                    source=self.name,
                    data={"result": result, "success": True},
                    correlation_id=event.correlation_id,
                )
            return None

        except Exception as e:
            if self._emit:
                return Event(
                    name=f"{self._emit}.error",
                    source=self.name,
                    data={"error": str(e), "success": False},
                    correlation_id=event.correlation_id,
                )
            raise

    @abstractmethod
    async def send(self, event: Event, ctx: Context) -> Any:
        """Send event to external system.

        Args:
            event: The event to send
            ctx: Flow execution context

        Returns:
            Response from external system
        """
        ...


class HttpGateway(Gateway):
    """HTTP gateway for REST API calls.

    Example:
        ```python
        flow.register(
            HttpGateway(
                url="https://api.example.com/process",
                method="POST",
                emit="api.response",
            ),
            on="data.ready",
        )
        ```
    """

    def __init__(
        self,
        url: str,
        *,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        name: str | None = None,
        emit: str | None = None,
    ) -> None:
        """Initialize HTTP gateway.

        Args:
            url: Target URL
            method: HTTP method
            headers: Request headers
            name: Reactor name
            emit: Event name to emit on success
        """
        super().__init__(name or "http_gateway", emit)
        self._url = url
        self._method = method
        self._headers = headers or {}

    async def send(self, event: Event, ctx: Context) -> Any:
        """Send HTTP request."""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp is required for HttpGateway")

        async with aiohttp.ClientSession() as session, session.request(
            self._method,
            self._url,
            json=event.data,
            headers=self._headers,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()


class LogGateway(Gateway):
    """Gateway that logs events (useful for debugging).

    Example:
        ```python
        flow.register(LogGateway(), on="*")  # Log all events
        ```
    """

    def __init__(
        self,
        *,
        logger: Any = None,
        level: str = "info",
        name: str | None = None,
    ) -> None:
        """Initialize log gateway.

        Args:
            logger: Logger instance (default: print)
            level: Log level
            name: Reactor name
        """
        super().__init__(name or "log_gateway", emit=None)
        self._logger = logger
        self._level = level

    async def send(self, event: Event, ctx: Context) -> None:
        """Log the event."""
        msg = f"[{event.name}] source={event.source} data={event.data}"

        if self._logger:
            log_fn = getattr(self._logger, self._level, self._logger.info)
            log_fn(msg)
        else:
            print(msg)


class CallbackGateway(Gateway):
    """Gateway that calls a callback function.

    Example:
        ```python
        async def on_complete(event, ctx):
            await notify_user(event.data)

        flow.register(
            CallbackGateway(on_complete),
            on="task.complete",
        )
        ```
    """

    def __init__(
        self,
        callback: Any,  # Callable[[Event, Context], Any]
        *,
        name: str | None = None,
        emit: str | None = None,
    ) -> None:
        """Initialize callback gateway.

        Args:
            callback: Async function to call
            name: Reactor name
            emit: Event name to emit on success
        """
        super().__init__(name or "callback_gateway", emit)
        self._callback = callback

    async def send(self, event: Event, ctx: Context) -> Any:
        """Call the callback."""
        import asyncio

        if asyncio.iscoroutinefunction(self._callback):
            return await self._callback(event, ctx)
        return self._callback(event, ctx)
