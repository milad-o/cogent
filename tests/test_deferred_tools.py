"""
Tests for deferred/event-driven tool completion.

Tests the DeferredResult pattern where tools can return immediately
and the agent waits for an event to signal completion.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenticflow.observability.event import EventType
from agenticflow.tools.deferred import (
    DeferredManager,
    DeferredResult,
    DeferredRetry,
    DeferredStatus,
    DeferredWaiter,
    is_deferred,
)


# ==============================================================================
# DeferredResult Tests
# ==============================================================================


class TestDeferredResult:
    """Tests for DeferredResult dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        deferred = DeferredResult()
        
        assert deferred.job_id is not None
        assert len(deferred.job_id) > 0
        assert deferred.wait_for is None
        assert deferred.match == {}
        assert deferred.timeout == 300.0
        assert deferred.on_timeout == "error"
        assert deferred.poll_url is None
        assert deferred.poll_interval == 5.0
        assert deferred.complete_when is None
        assert deferred.metadata == {}
        assert deferred.status == DeferredStatus.PENDING
        assert deferred.is_pending

    def test_custom_values(self) -> None:
        """Test custom field values."""
        deferred = DeferredResult(
            job_id="custom-123",
            wait_for="webhook.complete",
            match={"request_id": "abc"},
            timeout=60.0,
            on_timeout="default_value",
            metadata={"user": "test"},
        )
        
        assert deferred.job_id == "custom-123"
        assert deferred.wait_for == "webhook.complete"
        assert deferred.match == {"request_id": "abc"}
        assert deferred.timeout == 60.0
        assert deferred.on_timeout == "default_value"
        assert deferred.metadata == {"user": "test"}

    def test_complete(self) -> None:
        """Test completing a deferred result."""
        deferred = DeferredResult(job_id="test-1")
        
        assert deferred.is_pending
        assert not deferred.is_completed
        
        deferred.complete({"data": "result"})
        
        assert not deferred.is_pending
        assert deferred.is_completed
        assert deferred.result == {"data": "result"}
        assert deferred._completed_at is not None

    def test_fail(self) -> None:
        """Test failing a deferred result."""
        deferred = DeferredResult(job_id="test-2")
        
        deferred.fail("Something went wrong")
        
        assert deferred.status == DeferredStatus.FAILED
        assert deferred.error == "Something went wrong"
        assert deferred._completed_at is not None

    def test_timeout_reached(self) -> None:
        """Test timeout status."""
        deferred = DeferredResult(job_id="test-3")
        
        deferred.timeout_reached()
        
        assert deferred.status == DeferredStatus.TIMEOUT
        assert deferred._completed_at is not None

    def test_cancel(self) -> None:
        """Test cancellation."""
        deferred = DeferredResult(job_id="test-4")
        
        deferred.cancel()
        
        assert deferred.status == DeferredStatus.CANCELLED
        assert deferred._completed_at is not None

    def test_elapsed_seconds(self) -> None:
        """Test elapsed time calculation."""
        deferred = DeferredResult()
        
        # Should be very small (just created)
        assert deferred.elapsed_seconds < 1.0

    def test_is_timed_out(self) -> None:
        """Test timeout detection."""
        import time
        
        # Short timeout - wait a tiny bit to ensure it's expired
        deferred = DeferredResult(timeout=0.001)
        time.sleep(0.01)  # Wait 10ms to ensure timeout
        assert deferred.is_timed_out
        
        # Long timeout
        deferred2 = DeferredResult(timeout=3600.0)
        assert not deferred2.is_timed_out

    def test_with_event_type(self) -> None:
        """Test using EventType for wait_for."""
        deferred = DeferredResult(
            wait_for=EventType.CUSTOM,
            match={"event_name": "my_event"},
        )
        
        assert deferred.wait_for == EventType.CUSTOM


class TestDeferredRetry:
    """Tests for DeferredRetry marker."""

    def test_new_deferred(self) -> None:
        """Test creating new deferred from retry."""
        original = DeferredResult(
            job_id="original-1",
            wait_for="test.event",
            match={"id": 123},
            timeout=120.0,
            metadata={"attempt": 1},
        )
        
        retry = DeferredRetry(original)
        new = retry.new_deferred()
        
        # Should have new job_id
        assert new.job_id != original.job_id
        
        # Should preserve other fields
        assert new.wait_for == original.wait_for
        assert new.match == original.match
        assert new.timeout == original.timeout
        assert new.metadata == original.metadata
        
        # Should be fresh (pending)
        assert new.is_pending


# ==============================================================================
# is_deferred Helper Tests
# ==============================================================================


class TestIsDeferred:
    """Tests for is_deferred helper function."""

    def test_deferred_result(self) -> None:
        """Test with DeferredResult."""
        result = DeferredResult()
        assert is_deferred(result)

    def test_regular_string(self) -> None:
        """Test with regular string."""
        assert not is_deferred("hello")

    def test_regular_dict(self) -> None:
        """Test with regular dict."""
        assert not is_deferred({"result": "data"})

    def test_none(self) -> None:
        """Test with None."""
        assert not is_deferred(None)

    def test_number(self) -> None:
        """Test with number."""
        assert not is_deferred(42)


# ==============================================================================
# DeferredManager Tests
# ==============================================================================


class TestDeferredManager:
    """Tests for DeferredManager."""

    @pytest.fixture
    def mock_event_bus(self) -> MagicMock:
        """Create a mock event bus."""
        bus = MagicMock()
        bus.subscribe = MagicMock()
        bus.unsubscribe = MagicMock()
        bus.subscribe_all = MagicMock()
        bus.unsubscribe_all = MagicMock()
        return bus

    def test_init(self, mock_event_bus: MagicMock) -> None:
        """Test manager initialization."""
        manager = DeferredManager(event_bus=mock_event_bus)
        
        assert manager.pending_count == 0
        assert manager.pending_jobs == []

    def test_register(self, mock_event_bus: MagicMock) -> None:
        """Test registering deferred results."""
        manager = DeferredManager(event_bus=mock_event_bus)
        
        deferred1 = DeferredResult(job_id="job-1")
        deferred2 = DeferredResult(job_id="job-2")
        
        manager.register(deferred1)
        manager.register(deferred2)
        
        assert manager.pending_count == 2
        assert "job-1" in manager.pending_jobs
        assert "job-2" in manager.pending_jobs

    def test_get(self, mock_event_bus: MagicMock) -> None:
        """Test getting deferred by ID."""
        manager = DeferredManager(event_bus=mock_event_bus)
        
        deferred = DeferredResult(job_id="job-1", metadata={"test": True})
        manager.register(deferred)
        
        result = manager.get("job-1")
        assert result is deferred
        
        result = manager.get("nonexistent")
        assert result is None

    def test_cancel(self, mock_event_bus: MagicMock) -> None:
        """Test cancelling a deferred."""
        manager = DeferredManager(event_bus=mock_event_bus)
        
        deferred = DeferredResult(job_id="job-1")
        manager.register(deferred)
        
        result = manager.cancel("job-1")
        
        assert result is True
        assert deferred.status == DeferredStatus.CANCELLED
        assert manager.pending_count == 0

    def test_cancel_nonexistent(self, mock_event_bus: MagicMock) -> None:
        """Test cancelling nonexistent deferred."""
        manager = DeferredManager(event_bus=mock_event_bus)
        
        result = manager.cancel("nonexistent")
        assert result is False

    def test_cancel_all(self, mock_event_bus: MagicMock) -> None:
        """Test cancelling all pending."""
        manager = DeferredManager(event_bus=mock_event_bus)
        
        for i in range(5):
            manager.register(DeferredResult(job_id=f"job-{i}"))
        
        count = manager.cancel_all()
        
        assert count == 5
        assert manager.pending_count == 0

    def test_get_summary(self, mock_event_bus: MagicMock) -> None:
        """Test getting summary."""
        manager = DeferredManager(event_bus=mock_event_bus)
        
        # Register some
        for i in range(3):
            manager.register(DeferredResult(job_id=f"job-{i}"))
        
        summary = manager.get_summary()
        
        assert summary["pending"] == 3
        assert len(summary["pending_jobs"]) == 3


# ==============================================================================
# DeferredWaiter Tests
# ==============================================================================


class TestDeferredWaiter:
    """Tests for DeferredWaiter."""

    @pytest.fixture
    def mock_event_bus(self) -> MagicMock:
        """Create a mock event bus."""
        bus = MagicMock()
        bus.subscribe = MagicMock()
        bus.unsubscribe = MagicMock()
        bus.subscribe_all = MagicMock()
        bus.unsubscribe_all = MagicMock()
        return bus

    async def test_timeout_error(self, mock_event_bus: MagicMock) -> None:
        """Test timeout raises error."""
        deferred = DeferredResult(
            job_id="test-1",
            wait_for=EventType.CUSTOM,
            timeout=0.1,  # Very short
            on_timeout="error",
        )
        
        waiter = DeferredWaiter(deferred=deferred, event_bus=mock_event_bus)
        
        with pytest.raises(TimeoutError):
            await waiter.wait()

    async def test_timeout_default_value(self, mock_event_bus: MagicMock) -> None:
        """Test timeout returns default value."""
        deferred = DeferredResult(
            job_id="test-2",
            wait_for=EventType.CUSTOM,
            timeout=0.1,
            on_timeout={"default": True},
        )
        
        waiter = DeferredWaiter(deferred=deferred, event_bus=mock_event_bus)
        result = await waiter.wait()
        
        assert result == {"default": True}

    async def test_timeout_retry(self, mock_event_bus: MagicMock) -> None:
        """Test timeout returns retry marker."""
        deferred = DeferredResult(
            job_id="test-3",
            wait_for=EventType.CUSTOM,
            timeout=0.1,
            on_timeout="retry",
        )
        
        waiter = DeferredWaiter(deferred=deferred, event_bus=mock_event_bus)
        result = await waiter.wait()
        
        assert isinstance(result, DeferredRetry)
        assert result.original is deferred

    async def test_event_completion(self, mock_event_bus: MagicMock) -> None:
        """Test event-based completion."""
        deferred = DeferredResult(
            job_id="test-4",
            wait_for=EventType.CUSTOM,
            match={"job_id": "test-4"},
            timeout=5.0,
        )
        
        waiter = DeferredWaiter(deferred=deferred, event_bus=mock_event_bus)
        
        # Simulate event arrival in background
        async def trigger_event():
            await asyncio.sleep(0.05)
            # Get the callback that was registered
            callback = mock_event_bus.subscribe.call_args[0][1]
            # Create mock event
            mock_event = MagicMock()
            mock_event.data = {"job_id": "test-4", "result": "success!"}
            callback(mock_event)
        
        asyncio.create_task(trigger_event())
        result = await waiter.wait()
        
        assert result == "success!"

    async def test_event_match_filter(self, mock_event_bus: MagicMock) -> None:
        """Test event filtering by match criteria."""
        deferred = DeferredResult(
            job_id="test-5",
            wait_for=EventType.CUSTOM,
            match={"request_id": "abc123", "type": "complete"},
            timeout=5.0,
        )
        
        waiter = DeferredWaiter(deferred=deferred, event_bus=mock_event_bus)
        
        async def trigger_events():
            await asyncio.sleep(0.05)
            callback = mock_event_bus.subscribe.call_args[0][1]
            
            # Non-matching event (wrong request_id)
            wrong_event = MagicMock()
            wrong_event.data = {"request_id": "xyz", "type": "complete", "result": "wrong"}
            callback(wrong_event)
            
            await asyncio.sleep(0.02)
            
            # Matching event
            correct_event = MagicMock()
            correct_event.data = {"request_id": "abc123", "type": "complete", "result": "correct!"}
            callback(correct_event)
        
        asyncio.create_task(trigger_events())
        result = await waiter.wait()
        
        assert result == "correct!"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestDeferredToolIntegration:
    """Integration tests for deferred tools with Agent."""

    async def test_deferred_result_detection(self) -> None:
        """Test that DeferredResult is properly detected."""
        # Simulate a tool that returns DeferredResult
        async def webhook_tool(url: str) -> DeferredResult:
            return DeferredResult(
                job_id="webhook-123",
                wait_for="webhook.complete",
                match={"webhook_id": "webhook-123"},
                timeout=60.0,
            )
        
        result = await webhook_tool("https://example.com/api")
        
        assert is_deferred(result)
        assert result.job_id == "webhook-123"
        assert result.wait_for == "webhook.complete"

    async def test_deferred_with_manager(self) -> None:
        """Test full flow with DeferredManager."""
        mock_bus = MagicMock()
        mock_bus.subscribe = MagicMock()
        mock_bus.unsubscribe = MagicMock()
        
        manager = DeferredManager(event_bus=mock_bus)
        
        # Create deferred with very short timeout
        deferred = DeferredResult(
            job_id="flow-test",
            wait_for=EventType.CUSTOM,
            timeout=0.2,
            on_timeout={"status": "timed_out", "default": True},
        )
        
        # Wait for it (will timeout and return default)
        result = await manager.wait_for(deferred)
        
        assert result == {"status": "timed_out", "default": True}
        assert manager.pending_count == 0  # Should be moved to completed

    async def test_cancel_during_wait(self) -> None:
        """Test cancelling a deferred while waiting."""
        mock_bus = MagicMock()
        mock_bus.subscribe = MagicMock()
        mock_bus.unsubscribe = MagicMock()
        
        manager = DeferredManager(event_bus=mock_bus)
        
        deferred = DeferredResult(
            job_id="cancel-test",
            wait_for=EventType.CUSTOM,
            timeout=10.0,  # Long timeout
        )
        
        # Start waiting in background
        async def wait_task():
            try:
                await manager.wait_for(deferred)
            except Exception:
                pass
        
        task = asyncio.create_task(wait_task())
        
        # Give it a moment to start
        await asyncio.sleep(0.05)
        
        # Cancel it
        manager.cancel("cancel-test")
        
        # Wait for task to finish
        await asyncio.sleep(0.1)
        
        assert deferred.status == DeferredStatus.CANCELLED


# ==============================================================================
# Event Type Tests
# ==============================================================================


class TestDeferredEventTypes:
    """Test that deferred event types are properly defined."""

    def test_tool_deferred_exists(self) -> None:
        """Test TOOL_DEFERRED event type exists."""
        assert hasattr(EventType, "TOOL_DEFERRED")
        assert EventType.TOOL_DEFERRED.value == "tool.deferred"

    def test_tool_deferred_waiting_exists(self) -> None:
        """Test TOOL_DEFERRED_WAITING event type exists."""
        assert hasattr(EventType, "TOOL_DEFERRED_WAITING")
        assert EventType.TOOL_DEFERRED_WAITING.value == "tool.deferred.waiting"

    def test_tool_deferred_completed_exists(self) -> None:
        """Test TOOL_DEFERRED_COMPLETED event type exists."""
        assert hasattr(EventType, "TOOL_DEFERRED_COMPLETED")
        assert EventType.TOOL_DEFERRED_COMPLETED.value == "tool.deferred.completed"

    def test_tool_deferred_timeout_exists(self) -> None:
        """Test TOOL_DEFERRED_TIMEOUT event type exists."""
        assert hasattr(EventType, "TOOL_DEFERRED_TIMEOUT")
        assert EventType.TOOL_DEFERRED_TIMEOUT.value == "tool.deferred.timeout"
