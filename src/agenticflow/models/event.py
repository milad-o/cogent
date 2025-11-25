"""
Event - immutable record of something that happened in the system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from agenticflow.core.enums import EventType
from agenticflow.core.utils import generate_id, now_utc


@dataclass(frozen=False)  # Not frozen to allow default_factory
class Event:
    """
    An immutable event representing something that happened in the system.
    
    Events are the foundation of the event-driven architecture. They provide:
    - Full audit trail of system activity
    - Decoupled communication between components
    - Replay capability for debugging and recovery
    
    Attributes:
        type: The type of event (from EventType enum)
        data: Event-specific payload data
        id: Unique identifier for this event
        timestamp: When the event occurred (UTC)
        source: Identifier of the component that created this event
        parent_event_id: ID of the event that triggered this one (if any)
        correlation_id: ID linking related events across the system
        
    Example:
        ```python
        event = Event(
            type=EventType.TASK_COMPLETED,
            data={"task_id": "abc123", "result": "success"},
            source="agent:writer",
            correlation_id="request-456",
        )
        ```
    """

    type: EventType
    data: dict = field(default_factory=dict)
    id: str = field(default_factory=generate_id)
    timestamp: datetime = field(default_factory=now_utc)
    source: str = "system"
    parent_event_id: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "parent_event_id": self.parent_event_id,
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        """
        Convert to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> Event:
        """
        Create an Event from a dictionary.
        
        Args:
            data: Dictionary with event data
            
        Returns:
            New Event instance
        """
        return cls(
            id=data.get("id", generate_id()),
            type=EventType(data["type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
            source=data.get("source", "system"),
            parent_event_id=data.get("parent_event_id"),
            correlation_id=data.get("correlation_id"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Event:
        """
        Create an Event from a JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            New Event instance
        """
        return cls.from_dict(json.loads(json_str))

    def with_correlation(self, correlation_id: str) -> Event:
        """
        Create a copy of this event with a correlation ID.
        
        Args:
            correlation_id: The correlation ID to set
            
        Returns:
            New Event with the correlation ID
        """
        return Event(
            type=self.type,
            data=self.data,
            id=self.id,
            timestamp=self.timestamp,
            source=self.source,
            parent_event_id=self.parent_event_id,
            correlation_id=correlation_id,
        )

    def child_event(
        self,
        event_type: EventType,
        data: dict | None = None,
        source: str | None = None,
    ) -> Event:
        """
        Create a child event linked to this one.
        
        Args:
            event_type: Type of the child event
            data: Event data (default empty)
            source: Event source (defaults to this event's source)
            
        Returns:
            New Event linked to this one
        """
        return Event(
            type=event_type,
            data=data or {},
            source=source or self.source,
            parent_event_id=self.id,
            correlation_id=self.correlation_id,
        )

    @property
    def category(self) -> str:
        """Get the event category (e.g., 'task', 'agent')."""
        return self.type.category

    def __repr__(self) -> str:
        return f"Event(type={self.type.value}, id={self.id}, source={self.source})"
