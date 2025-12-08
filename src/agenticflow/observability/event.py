"""
Event types and Event class - the foundation of observability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agenticflow.core.utils import generate_id, now_utc


class EventType(Enum):
    """All event types in the system."""

    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"

    # User interaction events
    USER_INPUT = "user.input"  # User provided input/prompt
    USER_FEEDBACK = "user.feedback"  # User provided feedback (HITL)
    
    # Output events
    OUTPUT_GENERATED = "output.generated"  # Final output ready for user
    OUTPUT_STREAMED = "output.streamed"  # Output token streamed to user

    # Task lifecycle events
    TASK_CREATED = "task.created"
    TASK_SCHEDULED = "task.scheduled"
    TASK_STARTED = "task.started"
    TASK_BLOCKED = "task.blocked"
    TASK_UNBLOCKED = "task.unblocked"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_RETRYING = "task.retrying"

    # Subtask events
    SUBTASK_SPAWNED = "subtask.spawned"
    SUBTASK_COMPLETED = "subtask.completed"
    SUBTASKS_AGGREGATED = "subtasks.aggregated"

    # Agent events
    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_INVOKED = "agent.invoked"
    AGENT_THINKING = "agent.thinking"
    AGENT_REASONING = "agent.reasoning"  # Extended thinking/chain-of-thought
    AGENT_ACTING = "agent.acting"
    AGENT_RESPONDED = "agent.responded"
    AGENT_ERROR = "agent.error"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_INTERRUPTED = "agent.interrupted"  # HITL: agent paused for human input
    AGENT_RESUMED = "agent.resumed"  # HITL: agent resumed after human decision
    
    # Agent spawning events (dynamic agent creation)
    AGENT_SPAWNED = "agent.spawned"  # New agent spawned by parent
    AGENT_SPAWN_COMPLETED = "agent.spawn_completed"  # Spawned agent finished task
    AGENT_SPAWN_FAILED = "agent.spawn_failed"  # Spawned agent failed
    AGENT_DESPAWNED = "agent.despawned"  # Ephemeral agent cleaned up
    
    # LLM Request/Response events (deep observability)
    LLM_REQUEST = "llm.request"  # Full request being sent to LLM
    LLM_RESPONSE = "llm.response"  # Full response from LLM (before parsing)
    LLM_TOOL_DECISION = "llm.tool_decision"  # LLM decided to call tool(s)

    # Streaming events (token-by-token LLM output)
    STREAM_START = "stream.start"  # Streaming has started
    TOKEN_STREAMED = "stream.token"  # A token was streamed from the LLM
    STREAM_TOOL_CALL = "stream.tool_call"  # Tool call detected during streaming
    STREAM_END = "stream.end"  # Streaming has completed
    STREAM_ERROR = "stream.error"  # Error during streaming

    # Tool events
    TOOL_REGISTERED = "tool.registered"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    
    # Deferred tool events (async/event-driven completion)
    TOOL_DEFERRED = "tool.deferred"  # Tool returned DeferredResult
    TOOL_DEFERRED_WAITING = "tool.deferred.waiting"  # Waiting for completion
    TOOL_DEFERRED_COMPLETED = "tool.deferred.completed"  # Deferred completed
    TOOL_DEFERRED_TIMEOUT = "tool.deferred.timeout"  # Deferred timed out
    TOOL_DEFERRED_CANCELLED = "tool.deferred.cancelled"  # Deferred cancelled
    
    # Webhook/external events
    WEBHOOK_RECEIVED = "webhook.received"  # External webhook callback

    # Planning events
    PLAN_CREATED = "plan.created"
    PLAN_STEP_STARTED = "plan.step.started"
    PLAN_STEP_COMPLETED = "plan.step.completed"
    PLAN_FAILED = "plan.failed"

    # Message events
    MESSAGE_SENT = "message.sent"
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_BROADCAST = "message.broadcast"

    # WebSocket/Client events
    CLIENT_CONNECTED = "client.connected"
    CLIENT_DISCONNECTED = "client.disconnected"
    CLIENT_MESSAGE = "client.message"

    # Memory events (conversation history, semantic memory)
    MEMORY_READ = "memory.read"  # Memory retrieved
    MEMORY_WRITE = "memory.write"  # Memory stored
    MEMORY_SEARCH = "memory.search"  # Semantic search in memory
    MEMORY_DELETE = "memory.delete"  # Memory entry deleted
    MEMORY_CLEAR = "memory.clear"  # Memory cleared
    THREAD_CREATED = "memory.thread.created"  # New conversation thread
    THREAD_MESSAGE_ADDED = "memory.thread.message"  # Message added to thread

    # Retrieval events (RAG retrieval pipeline)
    RETRIEVAL_START = "retrieval.start"  # Retrieval query started
    RETRIEVAL_COMPLETE = "retrieval.complete"  # Retrieval finished with results
    RETRIEVAL_ERROR = "retrieval.error"  # Retrieval failed
    RERANK_START = "retrieval.rerank.start"  # Reranking started
    RERANK_COMPLETE = "retrieval.rerank.complete"  # Reranking finished
    FUSION_APPLIED = "retrieval.fusion"  # Ensemble fusion applied

    # MCP (Model Context Protocol) events
    MCP_SERVER_CONNECTING = "mcp.server.connecting"  # Connecting to MCP server
    MCP_SERVER_CONNECTED = "mcp.server.connected"  # Server connection established
    MCP_SERVER_DISCONNECTED = "mcp.server.disconnected"  # Server disconnected
    MCP_SERVER_ERROR = "mcp.server.error"  # Server connection/communication error
    MCP_TOOLS_DISCOVERED = "mcp.tools.discovered"  # Tools discovered from server
    MCP_RESOURCES_DISCOVERED = "mcp.resources.discovered"  # Resources discovered
    MCP_PROMPTS_DISCOVERED = "mcp.prompts.discovered"  # Prompts discovered
    MCP_TOOL_CALLED = "mcp.tool.called"  # MCP tool invocation started
    MCP_TOOL_RESULT = "mcp.tool.result"  # MCP tool returned result
    MCP_TOOL_ERROR = "mcp.tool.error"  # MCP tool execution failed

    # VectorStore events
    VECTORSTORE_ADD = "vectorstore.add"  # Documents added to store
    VECTORSTORE_SEARCH = "vectorstore.search"  # Similarity search performed
    VECTORSTORE_DELETE = "vectorstore.delete"  # Documents deleted

    # Document events (loading, splitting)
    DOCUMENT_LOADED = "document.loaded"  # Document loaded from source
    DOCUMENT_SPLIT = "document.split"  # Document split into chunks
    DOCUMENT_ENRICHED = "document.enriched"  # Metadata added to document

    CUSTOM = "custom"

    @property
    def category(self) -> str:
        """Get the category of this event type."""
        return self.value.split(".")[0]


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
