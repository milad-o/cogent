"""Base protocols and types for the memory system.

This module defines the core abstractions for memory:
- MemoryBackend: Protocol for storage backends
- MemoryScope: Enum for scoping (thread, user, execution)
- MemoryConfig: Configuration dataclass
- BaseMemory: Abstract base for all memory types
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agenticflow.core.messages import Message


class MemoryScope(str, Enum):
    """Scope for memory isolation."""

    THREAD = "thread"  # Per conversation thread
    USER = "user"  # Per user (across threads)
    TEAM = "team"  # Per topology/team (shared between agents)
    EXECUTION = "execution"  # Per flow execution (multi-agent)
    GLOBAL = "global"  # Shared across all


@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""

    key: str
    value: Any
    scope: MemoryScope
    scope_id: str  # thread_id, user_id, or execution_id
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    ttl_seconds: int | None = None  # Time-to-live, None = forever

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(UTC) - self.created_at).total_seconds()
        return age > self.ttl_seconds


@dataclass
class ConversationEntry:
    """A conversation message entry."""

    message: Message
    thread_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""

    # Conversation memory settings
    max_messages: int = 50
    summarize_after: int | None = None  # Auto-summarize after N messages

    # User memory settings
    max_facts: int = 1000

    # Working memory settings
    auto_cleanup: bool = True

    # Backend settings
    persist: bool = False
    db_path: str | None = None

    # Scoping
    default_scope: MemoryScope = MemoryScope.THREAD


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for memory storage backends.

    Backends handle the actual storage and retrieval of memory entries.
    They should be stateless regarding business logic - just store/retrieve.
    """

    async def get(self, key: str, scope: MemoryScope, scope_id: str) -> Any | None:
        """Get a value by key within a scope."""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        scope: MemoryScope,
        scope_id: str,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set a value by key within a scope."""
        ...

    async def delete(self, key: str, scope: MemoryScope, scope_id: str) -> bool:
        """Delete a value by key. Returns True if deleted."""
        ...

    async def list_keys(self, scope: MemoryScope, scope_id: str) -> list[str]:
        """List all keys within a scope."""
        ...

    async def clear(self, scope: MemoryScope, scope_id: str) -> None:
        """Clear all entries within a scope."""
        ...

    async def search(
        self,
        query: str,
        scope: MemoryScope,
        scope_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search for entries matching a query (simple text search)."""
        ...


@runtime_checkable
class ConversationBackend(Protocol):
    """Protocol for conversation history backends.

    Specialized backend for message storage with ordering.
    """

    async def add_message(
        self,
        thread_id: str,
        message: Message,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to conversation history."""
        ...

    async def get_messages(
        self,
        thread_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Get messages from conversation history."""
        ...

    async def clear_thread(self, thread_id: str) -> None:
        """Clear all messages in a thread."""
        ...

    async def get_message_count(self, thread_id: str) -> int:
        """Get the number of messages in a thread."""
        ...

    async def list_threads(self) -> list[str]:
        """List all thread IDs."""
        ...


class BaseMemory(ABC):
    """Abstract base class for all memory types.

    Provides common functionality and enforces the interface.
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize memory with optional backend and config.

        Args:
            backend: Storage backend. If None, uses InMemoryBackend.
            config: Memory configuration. If None, uses defaults.
        """
        # Import here to avoid circular imports
        from agenticflow.memory.backends.inmemory import InMemoryBackend

        self._backend = backend or InMemoryBackend()
        self._config = config or MemoryConfig()

    @property
    def backend(self) -> MemoryBackend:
        """Get the storage backend."""
        return self._backend

    @property
    def config(self) -> MemoryConfig:
        """Get the configuration."""
        return self._config

    @abstractmethod
    async def save(self, key: str, value: Any, **kwargs: Any) -> None:
        """Save a value to memory."""
        ...

    @abstractmethod
    async def load(self, key: str, **kwargs: Any) -> Any | None:
        """Load a value from memory."""
        ...

    @abstractmethod
    async def clear(self, **kwargs: Any) -> None:
        """Clear memory."""
        ...

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())


def create_default_memory() -> BaseMemory:
    """Create a default memory instance (ConversationMemory with InMemoryBackend).

    This is what's used when memory=True is passed to Agent.
    """
    from agenticflow.memory.conversation import ConversationMemory

    return ConversationMemory()
