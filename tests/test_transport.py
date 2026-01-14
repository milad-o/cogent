"""Tests for distributed transport."""

import asyncio

import pytest

from agenticflow.events import Event
from agenticflow.reactive.transport import (
    LocalTransport,
    Transport,
    PublishError,
)


# =============================================================================
# LocalTransport Tests
# =============================================================================


class TestLocalTransport:
    """Test LocalTransport (in-memory)."""
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connection lifecycle."""
        transport = LocalTransport()
        
        assert not transport._connected
        
        await transport.connect()
        assert transport._connected
        
        await transport.disconnect()
        assert not transport._connected
    
    @pytest.mark.asyncio
    async def test_publish_requires_connection(self):
        """Test publish fails if not connected."""
        transport = LocalTransport()
        event = Event(name="test.event", data={})
        
        with pytest.raises(PublishError, match="not connected"):
            await transport.publish(event)
    
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Test basic pub/sub."""
        transport = LocalTransport()
        await transport.connect()
        
        received = []
        
        async def handler(event: Event):
            received.append(event)
        
        await transport.subscribe("test.*", handler)
        
        # Publish matching event
        event1 = Event(name="test.created", data={"id": "1"})
        await transport.publish(event1)
        
        # Give dispatcher time to process
        await asyncio.sleep(0.1)
        
        assert len(received) == 1
        assert received[0].name == "test.created"
        
        await transport.disconnect()
    
    @pytest.mark.asyncio
    async def test_pattern_matching(self):
        """Test wildcard pattern matching."""
        transport = LocalTransport()
        await transport.connect()
        
        received = []
        
        async def handler(event: Event):
            received.append(event.name)
        
        await transport.subscribe("task.*", handler)
        
        # Matching events
        await transport.publish(Event(name="task.created", data={}))
        await transport.publish(Event(name="task.updated", data={}))
        await transport.publish(Event(name="task.completed", data={}))
        
        # Non-matching event
        await transport.publish(Event(name="user.login", data={}))
        
        await asyncio.sleep(0.1)
        
        assert len(received) == 3
        assert "task.created" in received
        assert "task.updated" in received
        assert "task.completed" in received
        assert "user.login" not in received
        
        await transport.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple handlers for same pattern."""
        transport = LocalTransport()
        await transport.connect()
        
        received1 = []
        received2 = []
        
        async def handler1(event: Event):
            received1.append(event)
        
        async def handler2(event: Event):
            received2.append(event)
        
        await transport.subscribe("test.*", handler1)
        await transport.subscribe("test.*", handler2)
        
        event = Event(name="test.event", data={})
        await transport.publish(event)
        
        await asyncio.sleep(0.1)
        
        assert len(received1) == 1
        assert len(received2) == 1
        
        await transport.disconnect()
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribe."""
        transport = LocalTransport()
        await transport.connect()
        
        received = []
        
        async def handler(event: Event):
            received.append(event)
        
        sub_id = await transport.subscribe("test.*", handler)
        
        # Publish before unsubscribe
        await transport.publish(Event(name="test.before", data={}))
        await asyncio.sleep(0.1)
        
        # Unsubscribe
        result = await transport.unsubscribe(sub_id)
        assert result is True
        
        # Publish after unsubscribe
        await transport.publish(Event(name="test.after", data={}))
        await asyncio.sleep(0.1)
        
        # Should only receive first event
        assert len(received) == 1
        assert received[0].name == "test.before"
        
        await transport.disconnect()
    
    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact event type matching."""
        transport = LocalTransport()
        await transport.connect()
        
        received = []
        
        async def handler(event: Event):
            received.append(event.name)
        
        # Subscribe to exact type (no wildcards)
        await transport.subscribe("task.created", handler)
        
        await transport.publish(Event(name="task.created", data={}))
        await transport.publish(Event(name="task.updated", data={}))
        
        await asyncio.sleep(0.1)
        
        assert len(received) == 1
        assert received[0] == "task.created"
        
        await transport.disconnect()


# =============================================================================
# RedisTransport Tests (requires Redis)
# =============================================================================


@pytest.mark.skipif(
    True,  # Skip by default - requires Redis server
    reason="Requires Redis server running on localhost:6379"
)
class TestRedisTransport:
    """Test RedisTransport (requires Redis server)."""
    
    @pytest.mark.asyncio
    async def test_redis_connect(self):
        """Test Redis connection."""
        from agenticflow.reactive.transport import RedisTransport
        
        transport = RedisTransport(url="redis://localhost:6379")
        
        await transport.connect()
        assert transport._connected
        
        await transport.disconnect()
        assert not transport._connected
    
    @pytest.mark.asyncio
    async def test_redis_pub_sub(self):
        """Test Redis pub/sub."""
        from agenticflow.reactive.transport import RedisTransport
        
        transport = RedisTransport(url="redis://localhost:6379")
        await transport.connect()
        
        received = []
        
        async def handler(event: Event):
            received.append(event)
        
        await transport.subscribe("test.*", handler)
        
        # Publish event
        event = Event(name="test.redis", data={"msg": "hello"})
        await transport.publish(event)
        
        # Wait for Redis pub/sub
        await asyncio.sleep(0.5)
        
        assert len(received) > 0
        assert received[0].type == "test.redis"
        assert received[0].data["msg"] == "hello"
        
        await transport.disconnect()


# =============================================================================
# Transport Protocol Tests
# =============================================================================


class TestTransportProtocol:
    """Test Transport protocol compliance."""
    
    def test_local_transport_implements_protocol(self):
        """Test LocalTransport implements Transport protocol."""
        transport = LocalTransport()
        assert isinstance(transport, Transport)
    
    @pytest.mark.skipif(
        True,
        reason="Requires Redis"
    )
    def test_redis_transport_implements_protocol(self):
        """Test RedisTransport implements Transport protocol."""
        from agenticflow.reactive.transport import RedisTransport
        
        transport = RedisTransport(url="redis://localhost:6379")
        assert isinstance(transport, Transport)
