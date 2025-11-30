"""
Agent Memory - Short-term and Long-term memory for agents.

This module provides memory capabilities for agents:

- **Short-term memory**: Track conversation history within a thread
- **Long-term memory**: Persist information across threads and sessions
- **Save/Restore**: Persist agent state across sessions

Memory uses native message types and provides pluggable backends:
- `MemorySaver` / `InMemorySaver` - For testing and development
- SQL backends can be added for production persistence

Example:
    ```python
    from agenticflow import Agent
    
    # Simple: just pass True for in-memory
    agent = Agent(name="Assistant", model=model, memory=True)
    
    # Chat within a thread (maintains history)
    response1 = await agent.chat("Hi, I'm Alice", thread_id="conv-1")
    response2 = await agent.chat("What's my name?", thread_id="conv-1")  # Remembers!
    
    # Different thread = fresh context
    response3 = await agent.chat("What's my name?", thread_id="conv-2")  # Doesn't know
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agenticflow.core.utils import generate_id, now_utc

if TYPE_CHECKING:
    from agenticflow.core.messages import BaseMessage


# ============================================================
# Protocols for Memory Backends
# ============================================================

@runtime_checkable
class MemorySaver(Protocol):
    """Protocol for persistence backend interface.
    
    This allows agenticflow to accept any compatible saver:
    - InMemorySaver (in-memory for testing)
    - SQL-based backends for production
    - Custom implementations
    """
    
    def get_tuple(self, config: dict) -> Any:
        """Get state tuple for a config."""
        ...
    
    def put(
        self,
        config: dict,
        state: dict,
        metadata: dict,
        new_versions: dict,
    ) -> dict:
        """Store state."""
        ...
    
    async def aget_tuple(self, config: dict) -> Any:
        """Async get state tuple."""
        ...
    
    async def aput(
        self,
        config: dict,
        state: dict,
        metadata: dict,
        new_versions: dict,
    ) -> dict:
        """Async store state."""
        ...


@runtime_checkable  
class MemoryStore(Protocol):
    """Protocol for long-term memory store interface.
    
    Stores support key-value operations across namespaces:
    - mget: Get multiple values
    - mset: Set multiple values
    - mdelete: Delete multiple values
    - yield_keys: Iterate keys
    """
    
    def mget(self, keys: list[str]) -> list[Any | None]:
        """Get multiple values by key."""
        ...
    
    def mset(self, key_value_pairs: list[tuple[str, Any]]) -> None:
        """Set multiple key-value pairs."""
        ...
    
    def mdelete(self, keys: list[str]) -> None:
        """Delete multiple keys."""
        ...


# ============================================================
# Memory State Types
# ============================================================

@dataclass
class MemorySnapshot:
    """Snapshot of agent memory state."""
    
    id: str = field(default_factory=generate_id)
    thread_id: str | None = None
    version: str | None = None
    
    # Core state
    messages: list[BaseMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Agent-specific state
    agent_name: str | None = None
    agent_role: str | None = None
    current_task: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    
    # Timing
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    
    def to_checkpoint_format(self) -> dict:
        """Convert to checkpoint format for persistence."""
        return {
            "v": 1,
            "id": self.version or self.id,
            "ts": self.updated_at.isoformat(),
            "channel_values": {
                "messages": self.messages,
                "__metadata__": self.metadata,
                "__context__": self.context,
            },
            "channel_versions": {},
            "versions_seen": {},
        }
    
    @classmethod
    def from_checkpoint_format(cls, data: dict, thread_id: str | None = None) -> MemorySnapshot:
        """Create from checkpoint format."""
        channel_values = data.get("channel_values", {})
        return cls(
            id=data.get("id", generate_id()),
            thread_id=thread_id,
            version=data.get("id"),
            messages=channel_values.get("messages", []),
            metadata=channel_values.get("__metadata__", {}),
            context=channel_values.get("__context__", {}),
        )


# Alias for backward compatibility
MemoryCheckpoint = MemorySnapshot


@dataclass
class ThreadConfig:
    """Configuration for a conversation thread."""
    
    thread_id: str
    version: str | None = None
    namespace: str = ""
    
    def to_config(self) -> dict:
        """Convert to config dict format."""
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "checkpoint_id": self.version,
                "checkpoint_ns": self.namespace,
            }
        }
    
    @classmethod
    def from_config(cls, config: dict) -> ThreadConfig:
        """Create from config dict format."""
        configurable = config.get("configurable", config)
        return cls(
            thread_id=configurable.get("thread_id", generate_id()),
            version=configurable.get("checkpoint_id"),
            namespace=configurable.get("checkpoint_ns", ""),
        )


# ============================================================
# Memory Manager
# ============================================================

class AgentMemory:
    """Memory manager for agents.
    
    Handles both short-term (conversation) and long-term memory,
    with pluggable persistence backends.
    
    Args:
        backend: Optional persistence backend.
            - InMemorySaver: For testing/development
            - SQL backends: For production persistence  
        store: Optional store for long-term memory.
        max_history: Maximum messages to keep in short-term memory.
        
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.agent.memory import InMemorySaver
        
        # In-memory for development/testing
        memory = AgentMemory(backend=InMemorySaver())
        
        # Or just pass True to Agent
        agent = Agent(name="Assistant", model=model, memory=True)
        
        # Save state
        await memory.save(
            thread_id="conv-1",
            messages=[...],
            metadata={"user": "alice"},
        )
        
        # Load state
        snapshot = await memory.load(thread_id="conv-1")
        ```
    """
    
    def __init__(
        self,
        backend: MemorySaver | None = None,
        store: MemoryStore | None = None,
        max_history: int = 100,
        # Legacy parameter name support
        checkpointer: MemorySaver | None = None,
    ):
        # Support both 'backend' and legacy 'checkpointer' parameter
        self._backend = backend or checkpointer
        self._store = store
        self.max_history = max_history
        
        # In-memory cache for active threads
        self._cache: dict[str, MemorySnapshot] = {}
        self._version_counters: dict[str, int] = {}
        # In-memory fallback for long-term storage
        self._long_term: dict[str, Any] = {}
    
    @property
    def backend(self) -> MemorySaver | None:
        """The persistence backend."""
        return self._backend
    
    # Legacy property alias
    @property
    def checkpointer(self) -> MemorySaver | None:
        """Legacy alias for backend."""
        return self._backend
    
    @property
    def store(self) -> MemoryStore | None:
        """The long-term memory store."""
        return self._store
    
    @property
    def has_persistence(self) -> bool:
        """Whether a persistence backend is configured."""
        return self._backend is not None
    
    # Legacy alias
    @property
    def has_checkpointer(self) -> bool:
        """Legacy alias for has_persistence."""
        return self.has_persistence
    
    @property
    def has_store(self) -> bool:
        """Whether a long-term store is configured."""
        return self._store is not None
    
    def _get_next_version(self, thread_id: str) -> str:
        """Get next version ID for a thread."""
        counter = self._version_counters.get(thread_id, 0) + 1
        self._version_counters[thread_id] = counter
        return f"v{counter}"
    
    # ========================================
    # Short-term Memory (Thread-scoped)
    # ========================================
    
    async def save(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        agent_name: str | None = None,
        agent_role: str | None = None,
    ) -> MemorySnapshot:
        """Save memory state for a thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread.
            messages: List of messages in the conversation.
            metadata: Optional metadata to store.
            context: Optional context data.
            agent_name: Name of the agent.
            agent_role: Role of the agent.
            
        Returns:
            The saved snapshot.
        """
        # Trim to max history
        if len(messages) > self.max_history:
            messages = messages[-self.max_history:]
        
        snapshot = MemorySnapshot(
            thread_id=thread_id,
            version=self._get_next_version(thread_id),
            messages=messages,
            metadata=metadata or {},
            context=context or {},
            agent_name=agent_name,
            agent_role=agent_role,
            updated_at=now_utc(),
        )
        
        # Update cache
        self._cache[thread_id] = snapshot
        
        # Persist if backend configured
        if self._backend:
            config = ThreadConfig(
                thread_id=thread_id, 
                version=snapshot.version,
            ).to_config()
            
            try:
                await self._backend.aput(
                    config,
                    snapshot.to_checkpoint_format(),
                    {"source": "agenticflow", "agent": agent_name},
                    {},
                )
            except AttributeError:
                # Fall back to sync method if async not available
                self._backend.put(
                    config,
                    snapshot.to_checkpoint_format(),
                    {"source": "agenticflow", "agent": agent_name},
                    {},
                )
        
        return snapshot
    
    # Alias for backward compatibility
    async def save_checkpoint(self, *args, **kwargs) -> MemorySnapshot:
        """Legacy alias for save()."""
        return await self.save(*args, **kwargs)
    
    async def load(
        self,
        thread_id: str,
        version: str | None = None,
    ) -> MemorySnapshot | None:
        """Load memory state for a thread.
        
        Args:
            thread_id: Thread identifier.
            version: Specific version (None for latest).
            
        Returns:
            The snapshot if found, None otherwise.
        """
        # Check cache first
        if version is None and thread_id in self._cache:
            return self._cache[thread_id]
        
        # Try to load from backend
        if self._backend:
            config = ThreadConfig(
                thread_id=thread_id,
                version=version,
            ).to_config()
            
            try:
                result = await self._backend.aget_tuple(config)
            except AttributeError:
                result = self._backend.get_tuple(config)
            
            if result and result.checkpoint:
                snapshot = MemorySnapshot.from_checkpoint_format(
                    result.checkpoint, thread_id
                )
                self._cache[thread_id] = snapshot
                return snapshot
        
        return self._cache.get(thread_id)
    
    # Alias for backward compatibility
    async def load_checkpoint(self, *args, **kwargs) -> MemorySnapshot | None:
        """Legacy alias for load()."""
        return await self.load(*args, **kwargs)
    
    async def get_messages(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> list[BaseMessage]:
        """Get messages for a thread.
        
        Args:
            thread_id: Thread identifier.
            limit: Maximum number of recent messages to return.
            
        Returns:
            List of messages (empty if thread not found).
        """
        snapshot = await self.load(thread_id)
        if not snapshot:
            return []
        
        messages = snapshot.messages
        if limit:
            messages = messages[-limit:]
        return messages
    
    async def add_messages(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        metadata: dict[str, Any] | None = None,
    ) -> MemorySnapshot:
        """Add messages to a thread.
        
        Args:
            thread_id: Thread identifier.
            messages: Messages to add.
            metadata: Optional metadata to update.
            
        Returns:
            The updated snapshot.
        """
        # Load existing
        snapshot = await self.load(thread_id)
        existing_messages = snapshot.messages if snapshot else []
        existing_metadata = snapshot.metadata if snapshot else {}
        existing_context = snapshot.context if snapshot else {}
        
        # Merge
        all_messages = existing_messages + messages
        all_metadata = {**existing_metadata, **(metadata or {})}
        
        return await self.save(
            thread_id=thread_id,
            messages=all_messages,
            metadata=all_metadata,
            context=existing_context,
        )
    
    async def clear_thread(self, thread_id: str) -> None:
        """Clear all memory for a thread.
        
        Args:
            thread_id: Thread to clear.
        """
        self._cache.pop(thread_id, None)
        self._version_counters.pop(thread_id, None)
        
        # Clear from backend if supported
        if self._backend and hasattr(self._backend, 'adelete_thread'):
            try:
                await self._backend.adelete_thread(thread_id)
            except Exception:
                pass  # Not all backends support deletion
    
    # ========================================
    # Long-term Memory (Cross-thread)
    # ========================================
    
    async def remember(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
    ) -> None:
        """Store a long-term memory.
        
        Args:
            key: Memory key.
            value: Value to store.
            namespace: Namespace for organization.
        """
        full_key = f"{namespace}:{key}"
        
        if self._store:
            self._store.mset([(full_key, value)])
        else:
            self._long_term[full_key] = value
    
    async def recall(
        self,
        key: str,
        namespace: str = "default",
    ) -> Any | None:
        """Retrieve a long-term memory.
        
        Args:
            key: Memory key.
            namespace: Namespace.
            
        Returns:
            The stored value or None.
        """
        full_key = f"{namespace}:{key}"
        
        if self._store:
            results = self._store.mget([full_key])
            return results[0] if results else None
        else:
            return self._long_term.get(full_key)
    
    async def forget(
        self,
        key: str,
        namespace: str = "default",
    ) -> None:
        """Remove a long-term memory.
        
        Args:
            key: Memory key.
            namespace: Namespace.
        """
        full_key = f"{namespace}:{key}"
        
        if self._store:
            self._store.mdelete([full_key])
        else:
            self._long_term.pop(full_key, None)
    
    # ========================================
    # Context Management
    # ========================================
    
    async def set_context(
        self,
        thread_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Set a context value for a thread.
        
        Context is useful for storing thread-specific data that isn't
        part of the message history (e.g., user preferences, extracted entities).
        
        Args:
            thread_id: Thread identifier.
            key: Context key.
            value: Value to store.
        """
        snapshot = await self.load(thread_id)
        if snapshot:
            snapshot.context[key] = value
            snapshot.updated_at = now_utc()
            await self.save(
                thread_id=thread_id,
                messages=snapshot.messages,
                metadata=snapshot.metadata,
                context=snapshot.context,
            )
        else:
            await self.save(
                thread_id=thread_id,
                messages=[],
                context={key: value},
            )
    
    async def get_context(
        self,
        thread_id: str,
        key: str | None = None,
    ) -> Any:
        """Get context for a thread.
        
        Args:
            thread_id: Thread identifier.
            key: Specific key to get (None for all context).
            
        Returns:
            The context value(s).
        """
        snapshot = await self.load(thread_id)
        if not snapshot:
            return None if key else {}
        
        if key:
            return snapshot.context.get(key)
        return snapshot.context
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def get_active_threads(self) -> list[str]:
        """Get list of active thread IDs in cache."""
        return list(self._cache.keys())
    
    def summary(self) -> dict[str, Any]:
        """Get memory summary."""
        return {
            "has_persistence": self.has_persistence,
            "has_store": self.has_store,
            "max_history": self.max_history,
            "active_threads": len(self._cache),
            "thread_ids": list(self._cache.keys()),
        }


# ============================================================
# Built-in In-Memory Backend (for testing/dev)
# ============================================================

class InMemorySaver:
    """Simple in-memory persistence backend for testing.
    
    This is a self-contained memory backend for development and testing.
    For production, consider using a persistent backend like Redis or PostgreSQL.
    
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.agent.memory import InMemorySaver
        
        # Use our built-in saver
        agent = Agent(name="Test", model=model, memory=InMemorySaver())
        
        # Or just use memory=True for convenience
        agent = Agent(name="Test", model=model, memory=True)
        ```
    """
    
    def __init__(self):
        self._storage: dict[str, dict] = {}
        self._metadata: dict[str, dict] = {}
    
    def _make_key(self, config: dict) -> str:
        """Create storage key from config."""
        configurable = config.get("configurable", config)
        thread_id = configurable.get("thread_id", "")
        namespace = configurable.get("checkpoint_ns", "")
        return f"{thread_id}:{namespace}"
    
    def get_tuple(self, config: dict) -> Any:
        """Get state tuple."""
        key = self._make_key(config)
        data = self._storage.get(key)
        if not data:
            return None
        
        @dataclass
        class StateTuple:
            config: dict
            checkpoint: dict
            metadata: dict
        
        return StateTuple(
            config=config,
            checkpoint=data,
            metadata=self._metadata.get(key, {}),
        )
    
    async def aget_tuple(self, config: dict) -> Any:
        """Async get state tuple."""
        return self.get_tuple(config)
    
    def put(
        self,
        config: dict,
        state: dict,
        metadata: dict,
        new_versions: dict,
    ) -> dict:
        """Store state."""
        key = self._make_key(config)
        self._storage[key] = state
        self._metadata[key] = metadata
        return config
    
    async def aput(
        self,
        config: dict,
        state: dict,
        metadata: dict,
        new_versions: dict,
    ) -> dict:
        """Async store state."""
        return self.put(config, state, metadata, new_versions)
    
    def list(self, config: dict, *, limit: int | None = None) -> list[Any]:
        """List states."""
        result = self.get_tuple(config)
        return [result] if result else []
    
    async def alist(self, config: dict, *, limit: int | None = None) -> list[Any]:
        """Async list states."""
        return self.list(config, limit=limit)
    
    async def adelete_thread(self, thread_id: str) -> None:
        """Delete a thread."""
        keys_to_delete = [k for k in self._storage if k.startswith(f"{thread_id}:")]
        for key in keys_to_delete:
            self._storage.pop(key, None)
            self._metadata.pop(key, None)


# Backward compatibility aliases
InMemoryCheckpointer = InMemorySaver
