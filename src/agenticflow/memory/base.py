"""
Base memory abstractions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from agenticflow.core.utils import generate_id, now_utc


class MemoryType(Enum):
    """Types of memory."""

    SHORT_TERM = "short_term"  # Thread-scoped, conversation history
    LONG_TERM = "long_term"  # Cross-thread, persistent
    SHARED = "shared"  # Multi-agent collaboration
    SEMANTIC = "semantic"  # Vector-based retrieval
    EPISODIC = "episodic"  # Task/session-based


@dataclass
class MemoryEntry:
    """
    A single memory entry.
    
    Attributes:
        content: The memory content (can be any JSON-serializable type)
        metadata: Additional metadata for filtering/retrieval
        id: Unique identifier
        timestamp: When the memory was created
        namespace: Namespace for scoping (agent_id, thread_id, etc.)
        relevance_score: Optional relevance score for semantic search
        expires_at: Optional expiration time
    """

    content: Any
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=generate_id)
    timestamp: datetime = field(default_factory=now_utc)
    namespace: str = "default"
    relevance_score: float | None = None
    expires_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "namespace": self.namespace,
            "relevance_score": self.relevance_score,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        """Create from dictionary."""
        return cls(
            id=data.get("id", generate_id()),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else now_utc(),
            namespace=data.get("namespace", "default"),
            relevance_score=data.get("relevance_score"),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
        )

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return now_utc() > self.expires_at


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""

    memory_type: MemoryType = MemoryType.SHORT_TERM
    max_entries: int = 1000
    ttl_seconds: int | None = None  # Time-to-live for entries
    namespace: str = "default"
    # For semantic memory
    embedding_model: str | None = None
    similarity_threshold: float = 0.7


T = TypeVar("T")


class Memory(ABC, Generic[T]):
    """
    Abstract base class for memory systems.
    
    All memory implementations must provide:
    - add(): Store a memory entry
    - get(): Retrieve a specific entry
    - search(): Find relevant entries
    - delete(): Remove an entry
    - clear(): Remove all entries
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """Initialize memory with config."""
        self.config = config or MemoryConfig()
        self.id = generate_id()

    @property
    def memory_type(self) -> MemoryType:
        """Get the memory type."""
        return self.config.memory_type

    @abstractmethod
    async def add(
        self,
        content: T,
        metadata: dict | None = None,
        namespace: str | None = None,
    ) -> MemoryEntry:
        """
        Add a memory entry.
        
        Args:
            content: The content to store
            metadata: Optional metadata for filtering
            namespace: Optional namespace override
            
        Returns:
            The created MemoryEntry
        """
        pass

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None:
        """
        Get a specific memory entry.
        
        Args:
            entry_id: The entry ID
            
        Returns:
            The MemoryEntry or None if not found
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Search for memory entries.
        
        Args:
            query: Optional search query (for semantic search)
            namespace: Filter by namespace
            metadata_filter: Filter by metadata
            limit: Maximum entries to return
            
        Returns:
            List of matching MemoryEntry objects
        """
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The entry ID
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def clear(self, namespace: str | None = None) -> int:
        """
        Clear memory entries.
        
        Args:
            namespace: Optional namespace to clear (None = all)
            
        Returns:
            Number of entries cleared
        """
        pass

    async def get_recent(
        self,
        limit: int = 10,
        namespace: str | None = None,
    ) -> list[MemoryEntry]:
        """
        Get most recent entries.
        
        Args:
            limit: Maximum entries to return
            namespace: Filter by namespace
            
        Returns:
            List of recent MemoryEntry objects
        """
        entries = await self.search(namespace=namespace, limit=limit)
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    async def count(self, namespace: str | None = None) -> int:
        """
        Count entries.
        
        Args:
            namespace: Filter by namespace
            
        Returns:
            Number of entries
        """
        entries = await self.search(namespace=namespace, limit=999999)
        return len(entries)
