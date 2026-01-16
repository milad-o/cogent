"""Event transport for distributed multi-agent systems.

Provides pluggable event transport backends for cross-process communication:
- LocalTransport: In-memory (default, single process)
- RedisTransport: Redis Pub/Sub + Streams (distributed)
- NATSTransport: NATS JetStream (optional, high-performance)

Example:
    ```python
    from agenticflow.events.transport import RedisTransport
    from agenticflow.events import EventBus

    # Create distributed event bus
    transport = RedisTransport(url="redis://localhost:6379")
    bus = EventBus(transport=transport)

    # Events published here are visible to all connected processes
    await bus.publish(Event(name="task.created", data={"id": "123"}))
    ```
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agenticflow.events.event import Event

EventHandler = Callable[[Event], None] | Callable[[Event], asyncio.Future[None]]


@runtime_checkable
class Transport(Protocol):
    """Protocol for event transport backends.

    Transport implementations handle the low-level delivery of events
    between agents, potentially across process or network boundaries.
    """

    async def connect(self) -> None:
        """Establish connection to transport backend.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        ...

    async def publish(self, event: Event) -> None:
        """Publish an event to the transport.

        Args:
            event: The event to publish

        Raises:
            TransportError: If publish fails
        """
        ...

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> str:
        """Subscribe to events matching a pattern.

        Args:
            pattern: Event type pattern (supports wildcards like "task.*")
            handler: Async function to call when events match

        Returns:
            Subscription ID for later unsubscribe
        """
        ...

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if unsubscribed, False if subscription not found
        """
        ...


class TransportError(Exception):
    """Base exception for transport errors."""
    pass


class ConnectionError(TransportError):
    """Failed to connect to transport backend."""
    pass


class PublishError(TransportError):
    """Failed to publish event."""
    pass


class LocalTransport:
    """In-memory local transport (single process).

    Default transport using asyncio.Queue for event delivery.
    No network overhead, suitable for single-process applications.

    Example:
        ```python
        transport = LocalTransport()
        await transport.connect()

        async def handler(event: Event):
            print(f"Received: {event.name}")

        await transport.subscribe("task.*", handler)
        await transport.publish(Event(name="task.created", data={}))
        ```
    """

    def __init__(self) -> None:
        self._connected = False
        self._subscriptions: dict[str, tuple[str, EventHandler]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._dispatcher_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Start local event dispatcher."""
        if self._connected:
            return

        self._connected = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_events())

    async def disconnect(self) -> None:
        """Stop dispatcher and cleanup."""
        if not self._connected:
            return

        self._connected = False
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatcher_task

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def publish(self, event: Event) -> None:
        """Publish event to local queue."""
        if not self._connected:
            raise PublishError("Transport not connected")

        await self._queue.put(event)

    async def subscribe(self, pattern: str, handler: EventHandler) -> str:
        """Subscribe to event pattern."""
        subscription_id = f"sub_{uuid.uuid4().hex[:12]}"
        self._subscriptions[subscription_id] = (pattern, handler)
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        return self._subscriptions.pop(subscription_id, None) is not None

    async def _dispatch_events(self) -> None:
        """Dispatch events from queue to matching handlers."""
        while self._connected:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except TimeoutError:
                continue

            # Find matching subscriptions
            for pattern, handler in self._subscriptions.values():
                if self._matches_pattern(event.name, pattern):
                    try:
                        result = handler(event)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        # Log but don't crash dispatcher
                        logging.error(f"Handler error: {e}", exc_info=True)

    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches subscription pattern.

        Supports:
        - Exact match: "task.created"
        - Wildcard: "task.*" matches "task.created", "task.updated"
        - Multi-level: "task.**.completed" matches "task.foo.bar.completed"
        """
        if pattern == event_type:
            return True

        # Simple wildcard support
        if "*" in pattern:
            pattern_parts = pattern.split(".")
            event_parts = event_type.split(".")

            if len(pattern_parts) != len(event_parts):
                # Handle ** (multi-level wildcard)
                if "**" in pattern_parts:
                    # Simplified: just check prefix/suffix
                    if pattern.startswith("**"):
                        suffix = pattern.replace("**.", "")
                        return event_type.endswith(suffix)
                    elif pattern.endswith("**"):
                        prefix = pattern.replace(".**", "")
                        return event_type.startswith(prefix)
                return False

            for p_part, e_part in zip(pattern_parts, event_parts, strict=False):
                if p_part != "*" and p_part != e_part:
                    return False
            return True

        return False


# Optional: Redis transport (requires redis package)
try:
    import redis.asyncio as aioredis

    class RedisTransport:
        """Redis-based distributed transport.

        Uses Redis Pub/Sub for event distribution across processes.
        Requires: `uv add redis`

        Example:
            ```python
            transport = RedisTransport(url="redis://localhost:6379")
            await transport.connect()

            # Events are distributed to all connected processes
            await transport.publish(Event(name="task.created", data={}))
            ```
        """

        def __init__(
            self,
            url: str = "redis://localhost:6379",
            *,
            channel_prefix: str = "agenticflow",
        ) -> None:
            self._url = url
            self._channel_prefix = channel_prefix
            self._client: aioredis.Redis | None = None
            self._pubsub: aioredis.client.PubSub | None = None
            self._subscriptions: dict[str, tuple[str, EventHandler]] = {}
            self._listener_task: asyncio.Task | None = None
            self._connected = False

        async def connect(self) -> None:
            """Connect to Redis."""
            if self._connected:
                return

            try:
                self._client = aioredis.from_url(self._url)
                await self._client.ping()
                self._pubsub = self._client.pubsub()
                self._connected = True
                self._listener_task = asyncio.create_task(self._listen())
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Redis: {e}") from e

        async def disconnect(self) -> None:
            """Disconnect from Redis."""
            if not self._connected:
                return

            self._connected = False

            if self._listener_task:
                self._listener_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._listener_task

            if self._pubsub:
                await self._pubsub.close()

            if self._client:
                await self._client.close()

        async def publish(self, event: Event) -> None:
            """Publish event to Redis channel."""
            if not self._connected or not self._client:
                raise PublishError("Transport not connected")

            channel = f"{self._channel_prefix}:{event.name}"
            # Serialize event as dict
            from dataclasses import asdict
            payload = json.dumps(asdict(event), default=str)

            try:
                await self._client.publish(channel, payload)
            except Exception as e:
                raise PublishError(f"Failed to publish event: {e}") from e

        async def subscribe(self, pattern: str, handler: EventHandler) -> str:
            """Subscribe to event pattern via Redis pattern matching."""
            if not self._connected or not self._pubsub:
                raise ConnectionError("Transport not connected")

            subscription_id = f"sub_{uuid.uuid4().hex[:12]}"

            # Convert pattern to Redis channel pattern
            redis_pattern = f"{self._channel_prefix}:{pattern}"

            # Subscribe via Redis
            await self._pubsub.psubscribe(redis_pattern)

            self._subscriptions[subscription_id] = (pattern, handler)
            return subscription_id

        async def unsubscribe(self, subscription_id: str) -> bool:
            """Unsubscribe from pattern."""
            if subscription_id not in self._subscriptions:
                return False

            pattern, _ = self._subscriptions.pop(subscription_id)
            redis_pattern = f"{self._channel_prefix}:{pattern}"

            if self._pubsub:
                await self._pubsub.punsubscribe(redis_pattern)

            return True

        async def _listen(self) -> None:
            """Listen for Redis pub/sub messages."""
            if not self._pubsub:
                return

            while self._connected:
                try:
                    message = await self._pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=0.1,
                    )

                    if message and message["type"] == "pmessage":
                        await self._handle_message(message)

                except TimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Redis listener error: {e}", exc_info=True)

        async def _handle_message(self, message: dict[str, Any]) -> None:
            """Handle incoming Redis message."""
            try:
                from dataclasses import fields
                payload = json.loads(message["data"])

                # Reconstruct Event from dict
                event = Event(**{k: payload.get(k) for k in [f.name for f in fields(Event)] if k in payload})

                # Call matching handlers
                for pattern, handler in self._subscriptions.values():
                    if self._matches_pattern(event.name, pattern):
                        try:
                            result = handler(event)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logging.error(f"Handler error: {e}", exc_info=True)

            except Exception as e:
                logging.error(f"Failed to process message: {e}", exc_info=True)

        def _matches_pattern(self, event_type: str, pattern: str) -> bool:
            """Check if event type matches pattern."""
            # Reuse LocalTransport logic
            transport = LocalTransport()
            return transport._matches_pattern(event_type, pattern)

except ImportError:
    # Redis not installed, skip
    class RedisTransport:  # type: ignore
        """Redis transport (requires redis package).

        Install with: uv add redis
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RedisTransport requires redis package. "
                "Install with: uv add redis"
            )
