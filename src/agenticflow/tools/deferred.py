"""
Deferred Tools - Event-driven tool completion for async workflows.

This module enables tools that don't complete immediately, instead returning
a DeferredResult that the agent waits for. Perfect for:
- Webhook callbacks from external APIs
- Long-running job processing
- Human-in-the-loop approvals
- External system integrations

Example - Webhook Callback:
    ```python
    from agenticflow.tools import tool, DeferredResult
    
    @tool
    async def process_video(video_url: str) -> DeferredResult:
        '''Submit video for processing, wait for webhook callback.'''
        job_id = await video_api.submit(video_url)
        
        return DeferredResult(
            job_id=job_id,
            wait_for="webhook.video_complete",
            match={"job_id": job_id},
            timeout=300,
        )
    
    # When webhook arrives at your server:
    await event_bus.publish("webhook.video_complete", {
        "job_id": "abc123",
        "result": {"url": "https://..."},
    })
    # Agent automatically resumes with the result!
    ```

Example - Human Approval:
    ```python
    @tool
    async def request_approval(request: str) -> DeferredResult:
        '''Request human approval before proceeding.'''
        approval_id = str(uuid4())
        await notify_slack(f"Approval needed: {request}")
        
        return DeferredResult(
            job_id=approval_id,
            wait_for="human.approval",
            match={"approval_id": approval_id},
            timeout=3600,  # 1 hour
            on_timeout="reject",  # Default action if timeout
        )
    ```

Example - Polling Pattern:
    ```python
    @tool
    async def run_analysis(data: str) -> DeferredResult:
        '''Run long analysis with polling.'''
        job = await analytics.start(data)
        
        return DeferredResult(
            job_id=job.id,
            poll_url=job.status_url,
            poll_interval=5,  # Check every 5 seconds
            complete_when=lambda status: status["state"] == "done",
            timeout=600,
        )
    ```
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from agenticflow.core.enums import EventType
from agenticflow.core.utils import generate_id, now_utc

if TYPE_CHECKING:
    from agenticflow.events.bus import EventBus


class DeferredStatus(Enum):
    """Status of a deferred result."""
    
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class DeferredResult:
    """
    A deferred result that the agent will wait for.
    
    When a tool returns a DeferredResult instead of a direct value,
    the agent loop will suspend and wait for the completion event.
    
    Attributes:
        job_id: Unique identifier for this deferred operation.
        wait_for: Event type to wait for (string or EventType).
        match: Dict of fields that must match in the event data.
        timeout: Maximum seconds to wait (default 300 = 5 minutes).
        on_timeout: Action on timeout: "error", "retry", or default value.
        poll_url: Optional URL to poll for status.
        poll_interval: Seconds between polls (default 5).
        complete_when: Optional predicate for poll completion.
        metadata: Additional data to pass through.
    """
    
    # Required
    job_id: str = field(default_factory=lambda: generate_id())
    
    # Event-based completion
    wait_for: str | EventType | None = None
    match: dict[str, Any] = field(default_factory=dict)
    
    # Timeout handling
    timeout: float = 300.0  # 5 minutes default
    on_timeout: str | Any = "error"  # "error", "retry", or a default value
    
    # Polling-based completion (alternative to events)
    poll_url: str | None = None
    poll_interval: float = 5.0
    complete_when: Callable[[dict], bool] | None = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)
    
    # Internal state
    _status: DeferredStatus = field(default=DeferredStatus.PENDING, repr=False)
    _result: Any = field(default=None, repr=False)
    _error: str | None = field(default=None, repr=False)
    _completed_at: datetime | None = field(default=None, repr=False)
    
    @property
    def status(self) -> DeferredStatus:
        """Current status of the deferred result."""
        return self._status
    
    @property
    def is_pending(self) -> bool:
        """Check if still waiting."""
        return self._status == DeferredStatus.PENDING
    
    @property
    def is_completed(self) -> bool:
        """Check if successfully completed."""
        return self._status == DeferredStatus.COMPLETED
    
    @property
    def result(self) -> Any:
        """Get the result (only valid if completed)."""
        return self._result
    
    @property
    def error(self) -> str | None:
        """Get error message if failed."""
        return self._error
    
    def complete(self, result: Any) -> None:
        """Mark as completed with result."""
        self._status = DeferredStatus.COMPLETED
        self._result = result
        self._completed_at = now_utc()
    
    def fail(self, error: str) -> None:
        """Mark as failed with error."""
        self._status = DeferredStatus.FAILED
        self._error = error
        self._completed_at = now_utc()
    
    def timeout_reached(self) -> None:
        """Mark as timed out."""
        self._status = DeferredStatus.TIMEOUT
        self._completed_at = now_utc()
    
    def cancel(self) -> None:
        """Mark as cancelled."""
        self._status = DeferredStatus.CANCELLED
        self._completed_at = now_utc()
    
    @property
    def elapsed_seconds(self) -> float:
        """Seconds since creation."""
        return (now_utc() - self.created_at).total_seconds()
    
    @property
    def is_timed_out(self) -> bool:
        """Check if timeout has been exceeded."""
        return self.elapsed_seconds > self.timeout


@dataclass
class DeferredWaiter:
    """
    Manages waiting for deferred results.
    
    Handles both event-based and polling-based completion patterns.
    """
    
    deferred: DeferredResult
    event_bus: "EventBus"
    _completion_event: asyncio.Event = field(default_factory=asyncio.Event)
    _subscription_id: str | None = field(default=None)
    
    async def wait(self) -> Any:
        """
        Wait for the deferred result to complete.
        
        Returns:
            The result value when completed.
            
        Raises:
            TimeoutError: If timeout exceeded.
            RuntimeError: If the operation failed.
        """
        if self.deferred.poll_url:
            return await self._wait_with_polling()
        else:
            return await self._wait_for_event()
    
    async def _wait_for_event(self) -> Any:
        """Wait for completion via event bus."""
        # Subscribe to the completion event
        event_type = self.deferred.wait_for
        if isinstance(event_type, str):
            # Custom event string - subscribe to CUSTOM type
            self.event_bus.subscribe_all(self._handle_event)
        else:
            self.event_bus.subscribe(event_type, self._handle_event)
        
        try:
            # Wait with timeout
            remaining = self.deferred.timeout - self.deferred.elapsed_seconds
            if remaining <= 0:
                return self._handle_timeout()
            
            try:
                await asyncio.wait_for(
                    self._completion_event.wait(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return self._handle_timeout()
            
            # Check result
            if self.deferred.is_completed:
                return self.deferred.result
            elif self.deferred._status == DeferredStatus.FAILED:
                raise RuntimeError(f"Deferred operation failed: {self.deferred.error}")
            else:
                raise RuntimeError(f"Unexpected status: {self.deferred.status}")
        
        finally:
            # Unsubscribe
            if isinstance(event_type, str):
                self.event_bus.unsubscribe_all(self._handle_event)
            else:
                self.event_bus.unsubscribe(event_type, self._handle_event)
    
    def _handle_event(self, event) -> None:
        """Handle incoming event and check if it matches."""
        # Check event type match
        event_type = self.deferred.wait_for
        if isinstance(event_type, str):
            # Custom event - check the event_name in data
            if event.data.get("event_name") != event_type:
                return
        
        # Check field matches
        for key, expected_value in self.deferred.match.items():
            if event.data.get(key) != expected_value:
                return
        
        # Match found! Extract result
        result = event.data.get("result", event.data)
        error = event.data.get("error")
        
        if error:
            self.deferred.fail(str(error))
        else:
            self.deferred.complete(result)
        
        self._completion_event.set()
    
    async def _wait_with_polling(self) -> Any:
        """Wait for completion via polling."""
        import httpx
        
        while not self.deferred.is_timed_out:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.deferred.poll_url)
                    response.raise_for_status()
                    status = response.json()
                
                # Check completion predicate
                if self.deferred.complete_when:
                    if self.deferred.complete_when(status):
                        self.deferred.complete(status)
                        return status
                else:
                    # Default: check for "complete", "done", "finished" states
                    state = status.get("state", status.get("status", ""))
                    if state.lower() in ("complete", "completed", "done", "finished", "success"):
                        self.deferred.complete(status)
                        return status
                    elif state.lower() in ("failed", "error"):
                        error = status.get("error", "Operation failed")
                        self.deferred.fail(error)
                        raise RuntimeError(error)
                
                # Wait before next poll
                await asyncio.sleep(self.deferred.poll_interval)
                
            except httpx.HTTPError as e:
                # Log but continue polling
                await asyncio.sleep(self.deferred.poll_interval)
        
        return self._handle_timeout()
    
    def _handle_timeout(self) -> Any:
        """Handle timeout based on on_timeout setting."""
        self.deferred.timeout_reached()
        
        on_timeout = self.deferred.on_timeout
        if on_timeout == "error":
            raise TimeoutError(
                f"Deferred operation timed out after {self.deferred.timeout}s "
                f"(job_id: {self.deferred.job_id})"
            )
        elif on_timeout == "retry":
            # Return special marker for retry
            return DeferredRetry(self.deferred)
        else:
            # Return the default value
            return on_timeout


@dataclass
class DeferredRetry:
    """Marker indicating the deferred operation should be retried."""
    
    original: DeferredResult
    
    def new_deferred(self) -> DeferredResult:
        """Create a fresh DeferredResult for retry."""
        return DeferredResult(
            job_id=generate_id(),
            wait_for=self.original.wait_for,
            match=self.original.match,
            timeout=self.original.timeout,
            on_timeout=self.original.on_timeout,
            poll_url=self.original.poll_url,
            poll_interval=self.original.poll_interval,
            complete_when=self.original.complete_when,
            metadata=self.original.metadata,
        )


class DeferredManager:
    """
    Manages all pending deferred operations for an agent.
    
    Tracks pending deferreds, handles completion events,
    and provides query capabilities.
    """
    
    __slots__ = ("_pending", "_completed", "_event_bus", "_max_history")
    
    def __init__(
        self,
        event_bus: "EventBus",
        max_history: int = 100,
    ) -> None:
        """
        Initialize the deferred manager.
        
        Args:
            event_bus: Event bus for receiving completion events.
            max_history: Maximum completed deferreds to keep.
        """
        self._pending: dict[str, DeferredResult] = {}
        self._completed: list[DeferredResult] = []
        self._event_bus = event_bus
        self._max_history = max_history
    
    @property
    def pending_count(self) -> int:
        """Number of pending deferred operations."""
        return len(self._pending)
    
    @property
    def pending_jobs(self) -> list[str]:
        """List of pending job IDs."""
        return list(self._pending.keys())
    
    def register(self, deferred: DeferredResult) -> None:
        """Register a new deferred result."""
        self._pending[deferred.job_id] = deferred
    
    def get(self, job_id: str) -> DeferredResult | None:
        """Get a deferred by job ID."""
        return self._pending.get(job_id)
    
    async def wait_for(self, deferred: DeferredResult) -> Any:
        """
        Wait for a deferred result to complete.
        
        Args:
            deferred: The deferred result to wait for.
            
        Returns:
            The result value.
        """
        self.register(deferred)
        
        try:
            waiter = DeferredWaiter(
                deferred=deferred,
                event_bus=self._event_bus,
            )
            result = await waiter.wait()
            
            # Handle retry
            if isinstance(result, DeferredRetry):
                new_deferred = result.new_deferred()
                return await self.wait_for(new_deferred)
            
            return result
            
        finally:
            # Move to completed
            if deferred.job_id in self._pending:
                del self._pending[deferred.job_id]
            self._completed.append(deferred)
            
            # Trim history
            if len(self._completed) > self._max_history:
                self._completed = self._completed[-self._max_history:]
    
    def cancel(self, job_id: str) -> bool:
        """
        Cancel a pending deferred operation.
        
        Args:
            job_id: The job ID to cancel.
            
        Returns:
            True if cancelled, False if not found.
        """
        deferred = self._pending.get(job_id)
        if deferred:
            deferred.cancel()
            del self._pending[job_id]
            self._completed.append(deferred)
            return True
        return False
    
    def cancel_all(self) -> int:
        """
        Cancel all pending operations.
        
        Returns:
            Number of operations cancelled.
        """
        count = len(self._pending)
        for deferred in self._pending.values():
            deferred.cancel()
            self._completed.append(deferred)
        self._pending.clear()
        return count
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of deferred operations."""
        return {
            "pending": self.pending_count,
            "completed": len([d for d in self._completed if d.is_completed]),
            "failed": len([d for d in self._completed if d._status == DeferredStatus.FAILED]),
            "timed_out": len([d for d in self._completed if d._status == DeferredStatus.TIMEOUT]),
            "cancelled": len([d for d in self._completed if d._status == DeferredStatus.CANCELLED]),
            "pending_jobs": self.pending_jobs,
        }


def is_deferred(result: Any) -> bool:
    """Check if a result is a DeferredResult."""
    return isinstance(result, DeferredResult)
