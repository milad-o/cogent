"""
Message - communication between agents or components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agenticflow.core.utils import generate_id, now_utc


class MessageType(Enum):
    """Types of messages that can be exchanged."""

    TEXT = "text"  # Plain text message
    COMMAND = "command"  # Instruction to perform an action
    QUERY = "query"  # Request for information
    RESPONSE = "response"  # Response to a query
    NOTIFICATION = "notification"  # Informational update
    ERROR = "error"  # Error notification
    SYSTEM = "system"  # System-level message


@dataclass
class Message:
    """
    A message exchanged between agents or components.
    
    Messages enable agent-to-agent communication and maintain conversation context.
    They support reply threading and metadata for routing/filtering.
    
    Attributes:
        content: The message content
        sender_id: ID of the sending agent/component
        receiver_id: ID of the receiving agent (None for broadcast)
        id: Unique message identifier
        timestamp: When the message was created (UTC)
        message_type: Type of message (text, command, query, etc.)
        metadata: Additional routing/filtering metadata
        reply_to: ID of message being replied to (for threading)
        
    Example:
        ```python
        message = Message(
            content="Please analyze this data",
            sender_id="orchestrator",
            receiver_id="analyst_agent",
            message_type=MessageType.COMMAND,
            metadata={"priority": "high"},
        )
        ```
    """

    content: str
    sender_id: str
    receiver_id: str | None = None  # None = broadcast
    id: str = field(default_factory=generate_id)
    timestamp: datetime = field(default_factory=now_utc)
    message_type: MessageType = MessageType.TEXT
    metadata: dict = field(default_factory=dict)
    reply_to: str | None = None  # ID of message being replied to

    def to_dict(self) -> dict:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "id": self.id,
            "content": self.content,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type.value,
            "metadata": self.metadata,
            "reply_to": self.reply_to,
        }

    def to_json(self) -> str:
        """
        Convert to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        """
        Create a Message from a dictionary.
        
        Args:
            data: Dictionary with message data
            
        Returns:
            New Message instance
        """
        return cls(
            id=data.get("id", generate_id()),
            content=data["content"],
            sender_id=data["sender_id"],
            receiver_id=data.get("receiver_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_type=MessageType(data.get("message_type", "text")),
            metadata=data.get("metadata", {}),
            reply_to=data.get("reply_to"),
        )

    def reply(
        self,
        content: str,
        sender_id: str,
        message_type: MessageType = MessageType.RESPONSE,
    ) -> Message:
        """
        Create a reply to this message.
        
        Args:
            content: Reply content
            sender_id: ID of the replying agent
            message_type: Type of reply (default RESPONSE)
            
        Returns:
            New Message that is a reply to this one
        """
        return Message(
            content=content,
            sender_id=sender_id,
            receiver_id=self.sender_id,
            message_type=message_type,
            reply_to=self.id,
        )

    @property
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.receiver_id is None

    @property
    def is_reply(self) -> bool:
        """Check if this is a reply to another message."""
        return self.reply_to is not None

    def __repr__(self) -> str:
        target = self.receiver_id or "broadcast"
        return f"Message(from={self.sender_id}, to={target}, type={self.message_type.value})"
