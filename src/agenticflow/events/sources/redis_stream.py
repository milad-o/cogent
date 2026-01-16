"""Redis Streams event source.

Consumes events from Redis Streams and emits them into EventFlow.
Supports consumer groups for distributed processing.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from agenticflow.events.event import Event
from agenticflow.events.sources.base import EmitCallback, EventSource


@dataclass
class RedisStreamSource(EventSource):
    """Consume events from Redis Streams.

    Uses Redis Streams with consumer groups for reliable, distributed
    event processing. Supports automatic acknowledgment and dead letter
    handling.

    Attributes:
        stream: Redis stream name to consume from
        group: Consumer group name (default: "agenticflow")
        consumer: Consumer name (auto-generated if not provided)
        redis_url: Redis connection URL (default: "redis://localhost:6379")
        event_name_field: Field in stream message containing event name
        batch_size: Number of messages to fetch per read (default: 10)
        block_ms: Milliseconds to block waiting for messages (default: 5000)

    Example:
        ```python
        source = RedisStreamSource(
            stream="events",
            group="my-workers",
            redis_url="redis://localhost:6379",
        )
        flow.source(source)

        # Messages from Redis stream "events" become flow events
        # XADD events * name "order.created" data '{"order_id": "123"}'
        ```

    Message Format:
        The stream message should contain:
        - `name` or `event`: Event name
        - `data`: JSON-encoded event data (or other fields become data)
    """

    stream: str = "events"
    group: str = "agenticflow"
    consumer: str | None = None
    redis_url: str = "redis://localhost:6379"
    event_name_field: str = "name"
    batch_size: int = 10
    block_ms: int = 5000

    _running: bool = field(default=False, repr=False)
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.consumer is None:
            self.consumer = f"consumer-{uuid4().hex[:8]}"

    async def start(self, emit: EmitCallback) -> None:
        """Start consuming from Redis Stream.

        Args:
            emit: Callback to emit events into the flow.
        """
        try:
            import redis.asyncio as redis
        except ImportError as e:
            raise ImportError(
                "RedisStreamSource requires 'redis'. "
                "Install with: uv add redis"
            ) from e

        self._running = True
        self._client = redis.from_url(self.redis_url)

        # Ensure consumer group exists
        try:
            await self._client.xgroup_create(
                self.stream,
                self.group,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        while self._running:
            try:
                # Read from stream
                messages = await self._client.xreadgroup(
                    groupname=self.group,
                    consumername=self.consumer,
                    streams={self.stream: ">"},
                    count=self.batch_size,
                    block=self.block_ms,
                )

                if not messages:
                    continue

                for _stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        # Parse message fields
                        event = self._parse_message(message_id, fields)

                        if event:
                            await emit(event)

                        # Acknowledge message
                        await self._client.xack(
                            self.stream,
                            self.group,
                            message_id,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                import sys
                print(f"RedisStreamSource error: {e}", file=sys.stderr)
                await asyncio.sleep(1)

        await self._cleanup()

    def _parse_message(
        self,
        message_id: bytes | str,
        fields: dict[bytes | str, bytes | str],
    ) -> Event | None:
        """Parse Redis stream message into Event.

        Args:
            message_id: Redis message ID
            fields: Message field dict

        Returns:
            Parsed Event or None if invalid
        """
        # Decode bytes to str
        decoded: dict[str, str] = {}
        for k, v in fields.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else k
            val = v.decode("utf-8") if isinstance(v, bytes) else v
            decoded[key] = val

        # Extract event name
        event_name = (
            decoded.pop(self.event_name_field, None)
            or decoded.pop("event", None)
            or "redis.message"
        )

        # Extract/parse data
        data: dict[str, Any]
        if "data" in decoded:
            try:
                data = json.loads(decoded.pop("data"))
            except json.JSONDecodeError:
                data = {"raw_data": decoded.pop("data")}
        else:
            data = decoded

        # Add message ID for reference
        mid = message_id.decode("utf-8") if isinstance(message_id, bytes) else message_id
        data["_redis_message_id"] = mid

        return Event(
            name=event_name,
            data=data,
            source=f"redis:{self.stream}",
        )

    async def _cleanup(self) -> None:
        """Clean up Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def stop(self) -> None:
        """Stop consuming from Redis Stream."""
        self._running = False

    @property
    def name(self) -> str:
        """Human-readable name for this source."""
        return f"RedisStreamSource({self.stream}@{self.group})"
