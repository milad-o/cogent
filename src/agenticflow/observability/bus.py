"""
EventBus - central pub/sub system for event distribution.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from typing import Callable, Awaitable

from agenticflow.observability.event import EventType
from agenticflow.observability.event import Event


# Type alias for event handlers
EventHandler = Callable[[Event], None] | Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Central event bus with pub/sub pattern.
    
    The EventBus is the backbone of the event-driven architecture.
    It supports:
    - Type-specific subscriptions
    - Global subscriptions (for logging, metrics)
    - Sync and async handlers
    - Event history with querying
    - WebSocket client broadcasting
    
    Attributes:
        max_history: Maximum events to keep in history
        
    Example:
        ```python
        bus = EventBus()
        
        # Subscribe to specific events
        bus.subscribe(EventType.TASK_COMPLETED, handle_completion)
        
        # Subscribe to all events
        bus.subscribe_all(log_event)
        
        # Publish an event
        await bus.publish(Event(
            type=EventType.TASK_STARTED,
            data={"task_id": "123"},
        ))
        
        # Query history
        events = bus.get_history(
            event_type=EventType.TASK_COMPLETED,
            limit=10,
        )
        ```
    """

    def __init__(self, max_history: int = 10000) -> None:
        """
        Initialize the EventBus.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []
        self._event_history: list[Event] = []
        self._websocket_clients: set = set()
        self._lock = asyncio.Lock()
        self._max_history = max_history
        self._loop: asyncio.AbstractEventLoop | None = None  # Store loop reference

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: The type of events to subscribe to
            handler: Callback function (sync or async)
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def subscribe_many(
        self,
        event_types: list[EventType],
        handler: EventHandler,
    ) -> None:
        """
        Subscribe to multiple event types.
        
        Args:
            event_types: List of event types to subscribe to
            handler: Callback function (sync or async)
        """
        for event_type in event_types:
            self.subscribe(event_type, handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe to ALL events.
        
        Useful for logging, metrics, debugging, or WebSocket streaming.
        
        Args:
            handler: Callback function (sync or async)
        """
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: The event type to unsubscribe from
            handler: The handler to remove
        """
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def unsubscribe_all(self, handler: EventHandler) -> None:
        """
        Unsubscribe a global handler.
        
        Args:
            handler: The handler to remove
        """
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)

    def clear_subscriptions(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
        self._global_handlers.clear()

    async def publish(self, event: Event | str, data: dict | None = None) -> None:
        """
        Publish an event to all subscribers.
        
        Can be called two ways:
        1. publish(event) - with an Event object
        2. publish("event_type", {"key": "value"}) - with string and dict
        
        Args:
            event: The Event object, or event type string
            data: Event data (only used if event is a string)
        """
        # Capture event loop reference on first publish (for publish_sync to use)
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        
        # Handle simple string/dict API
        if isinstance(event, str):
            from agenticflow.observability.event import EventType
            from agenticflow.observability.event import Event as EventClass
            
            # Try to parse as EventType enum, otherwise use custom type
            try:
                event_type = EventType(event)
            except ValueError:
                # Custom event type - use a generic type
                event_type = EventType.CUSTOM
            
            event = EventClass(
                type=event_type,
                data={"event_name": event if event_type == EventType.CUSTOM else None, **(data or {})},
            )
        
        # Add to history
        async with self._lock:
            self._event_history.append(event)
            # Trim history if needed
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # Call specific handlers
        for handler in self._handlers[event.type]:
            await self._call_handler(handler, event)

        # Call global handlers
        for handler in self._global_handlers:
            await self._call_handler(handler, event)

        # Broadcast to WebSocket clients
        await self._broadcast_to_websockets(event)

    def publish_sync(self, event: Event | str, data: dict | None = None) -> None:
        """
        Publish an event synchronously (fire and forget).
        
        This method is designed for use in sync contexts (like tool functions)
        where you need to emit events but can't await. It uses the event loop
        reference captured during EventBus initialization to schedule events
        from threads.
        
        **Use only when:**
        - Called from sync code that can't be made async
        - There's a running event loop (e.g., within async application)
        - You don't need to wait for event delivery
        
        Args:
            event: The Event object, or event type string
            data: Event data (only used if event is a string)
        
        Example:
            ```python
            # From a sync tool function:
            def my_tool():
                bus.publish_sync("tool.started", {"tool": "my_tool"})
                result = do_work()
                bus.publish_sync("tool.completed", {"result": result})
                return result
            ```
        """
        # First, try get_running_loop() - works if called from async context
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event, data))
            return
        except RuntimeError:
            pass
        
        # Second, try using stored loop reference (for thread-safe calls)
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.publish(event, data), self._loop)
            return
        
        # No event loop available - silently skip
        # This can happen in pure sync contexts or during testing
        pass

    async def publish_many(self, events: list[Event]) -> None:
        """
        Publish multiple events.
        
        Args:
            events: List of events to publish
        """
        for event in events:
            await self.publish(event)

    async def _call_handler(self, handler: EventHandler, event: Event) -> None:
        """
        Call a handler (sync or async).
        
        Args:
            handler: The handler to call
            event: The event to pass to the handler
        """
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            # Log but don't propagate handler errors
            print(f"⚠️ Event handler error: {e}")

    async def _broadcast_to_websockets(self, event: Event) -> None:
        """
        Send event to all connected WebSocket clients.
        
        Args:
            event: The event to broadcast
        """
        if not self._websocket_clients:
            return

        message = event.to_json()
        disconnected = set()

        for ws in self._websocket_clients:
            try:
                await ws.send(message)
            except Exception:
                disconnected.add(ws)

        # Clean up disconnected clients
        self._websocket_clients -= disconnected

    def add_websocket(self, ws) -> None:
        """
        Register a WebSocket client for event streaming.
        
        Args:
            ws: WebSocket connection
        """
        self._websocket_clients.add(ws)

    def remove_websocket(self, ws) -> None:
        """
        Unregister a WebSocket client.
        
        Args:
            ws: WebSocket connection to remove
        """
        self._websocket_clients.discard(ws)

    @property
    def websocket_count(self) -> int:
        """Number of connected WebSocket clients."""
        return len(self._websocket_clients)

    def get_history(
        self,
        event_type: EventType | None = None,
        correlation_id: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Query event history.
        
        Args:
            event_type: Filter by event type
            correlation_id: Filter by correlation ID
            source: Filter by source
            limit: Maximum events to return
            
        Returns:
            List of matching events (most recent last)
        """
        events = self._event_history

        if event_type:
            events = [e for e in events if e.type == event_type]

        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]

        if source:
            events = [e for e in events if e.source == source]

        return events[-limit:]

    def get_history_by_category(
        self,
        category: str,
        limit: int = 100,
    ) -> list[Event]:
        """
        Query event history by category.
        
        Args:
            category: Event category (e.g., "task", "agent")
            limit: Maximum events to return
            
        Returns:
            List of matching events
        """
        events = [e for e in self._event_history if e.category == category]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    @property
    def history_size(self) -> int:
        """Number of events in history."""
        return len(self._event_history)

    def get_stats(self) -> dict:
        """
        Get event bus statistics.
        
        Returns:
            Dictionary with statistics
        """
        type_counts: dict[str, int] = defaultdict(int)
        for event in self._event_history:
            type_counts[event.type.value] += 1

        return {
            "history_size": len(self._event_history),
            "max_history": self._max_history,
            "handler_count": sum(len(h) for h in self._handlers.values()),
            "global_handler_count": len(self._global_handlers),
            "websocket_clients": len(self._websocket_clients),
            "event_type_counts": dict(type_counts),
        }


# Global singleton instance (optional)
_default_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Get the default global EventBus instance.
    
    Creates one if it doesn't exist.
    
    Returns:
        The global EventBus instance
    """
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def set_event_bus(bus: EventBus) -> None:
    """
    Set the global EventBus instance.
    
    Args:
        bus: The EventBus to use as global default
    """
    global _default_bus
    _default_bus = bus
