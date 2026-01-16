"""Memory-first architecture for AgenticFlow.

Core principle: Memory is a first-class citizen.
Wire it to any entity (Agent, Team, Flow) and it changes behavior.

Example:
    ```python
    from agenticflow.memory import Memory
    from agenticflow.vectorstore import VectorStore

    # Basic - in-memory, no persistence
    memory = Memory()

    # With persistence
    memory = Memory(store=SQLiteStore("./data.db"))

    # With semantic search (reuses existing VectorStore!)
    memory = Memory(
        store=SQLiteStore("./data.db"),
        vectorstore=VectorStore(),
    )

    # Scoped views (replaces UserMemory, TeamMemory, etc.)
    user_mem = memory.scoped("user:alice")
    team_mem = memory.scoped("team:research")
    conv_mem = memory.scoped("conv:thread-123")

    # Share between entities
    agent_a = Agent(memory=memory)
    agent_b = Agent(memory=memory)  # Same memory = shared knowledge
    ```
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agenticflow.core.messages import BaseMessage as Message
    from agenticflow.observability.bus import TraceBus
    from agenticflow.vectorstore import VectorStore


# =============================================================================
# STORE PROTOCOL - Pluggable Persistence
# =============================================================================


@runtime_checkable
class Store(Protocol):
    """Protocol for persistence stores.

    Implement this to add new storage backends.
    All operations are async for consistency.

    Built-in implementations:
    - InMemoryStore: Fast, no persistence (testing/dev)
    - SQLiteStore: Local file persistence
    - PostgresStore: Production database
    - RedisStore: Distributed cache
    """

    async def get(self, key: str) -> Any | None:
        """Get value by key. Returns None if not found."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value by key. Optional TTL in seconds."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete by key. Returns True if deleted."""
        ...

    async def keys(self, prefix: str = "") -> list[str]:
        """List keys matching prefix."""
        ...

    async def clear(self, prefix: str = "") -> None:
        """Clear keys matching prefix. Empty prefix = all."""
        ...


# =============================================================================
# MEMORY - The Core Abstraction
# =============================================================================


class Memory:
    """Universal memory. Wire to any entity.

    Memory is the central abstraction for sharing knowledge between
    agents, teams, and flows. It provides:

    - Key-value storage via `remember()` and `recall()`
    - Semantic search via `search()` (requires VectorStore)
    - Message history via `add_message()` and `get_messages()`
    - Namespace isolation via `scoped()`

    Example:
        ```python
        # Basic usage
        memory = Memory()
        await memory.remember("user_preference", "dark mode")
        pref = await memory.recall("user_preference")

        # With persistence
        memory = Memory(store=SQLiteStore("./data.db"))

        # With semantic search
        from agenticflow.vectorstore import VectorStore
        memory = Memory(vectorstore=VectorStore())
        await memory.add_document("Python is dynamically typed")
        results = await memory.search("type systems")

        # Namespace isolation (replaces TeamMemory, UserMemory, etc.)
        team_memory = memory.scoped("team:research")
        agent_memory = memory.scoped("agent:analyst")
        # Both use same store, but isolated keyspaces
        ```
    """

    def __init__(
        self,
        store: Store | None = None,
        vectorstore: VectorStore | None = None,
        namespace: str = "",
        event_bus: TraceBus | None = None,
        *,
        agentic: bool = False,
    ) -> None:
        """Initialize Memory.

        Args:
            store: Persistence store. Defaults to InMemoryStore.
            vectorstore: VectorStore for semantic search. Optional.
            namespace: Key prefix for isolation.
            event_bus: TraceBus for observability. Optional.
            agentic: If True, expose memory tools to agents automatically.
                When an Agent receives an agentic Memory, tools are auto-added.
        """
        self._store = store or InMemoryStore()
        self._vectorstore = vectorstore
        self._namespace = namespace
        self._event_bus = event_bus
        self._agentic = agentic
        self._lock = asyncio.Lock()
        self._tools_cache: list[Any] | None = None

    @property
    def store(self) -> Store:
        """Get the underlying store."""
        return self._store

    @property
    def vectorstore(self) -> VectorStore | None:
        """Get the VectorStore (if any)."""
        return self._vectorstore

    @property
    def namespace(self) -> str:
        """Get the namespace prefix."""
        return self._namespace

    @property
    def event_bus(self) -> TraceBus | None:
        """Get the EventBus (if any)."""
        return self._event_bus

    @property
    def agentic(self) -> bool:
        """Whether this memory exposes tools to agents."""
        return self._agentic

    @property
    def tools(self) -> list[Any]:
        """Get memory tools (for agentic mode).

        Returns:
            List of memory tools (remember, recall, forget, search_memories).
            Returns empty list if not in agentic mode.

        Example:
            ```python
            # Agentic memory - tools auto-added to agent
            memory = Memory(agentic=True)
            agent = Agent(memory=memory)  # Tools added automatically

            # Non-agentic - no tools, just storage
            memory = Memory()
            agent = Agent(memory=memory)  # No tools, memory for code use only

            # Manual tool access (for custom setups)
            memory = Memory(agentic=True)
            tools = memory.tools  # [remember, recall, forget, search_memories]
            ```
        """
        if not self._agentic:
            return []

        if self._tools_cache is None:
            from agenticflow.memory.tools import create_memory_tools
            self._tools_cache = create_memory_tools(self)

        return self._tools_cache

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event if event_bus is configured."""
        if self._event_bus:
            from agenticflow.observability.trace_record import TraceType
            await self._event_bus.publish(TraceType(event_type), {
                "namespace": self._namespace,
                **data,
            })

    def _key(self, key: str) -> str:
        """Prefix key with namespace."""
        if self._namespace:
            return f"{self._namespace}:{key}"
        return key

    # -------------------------------------------------------------------------
    # Core Key-Value Operations
    # -------------------------------------------------------------------------

    async def remember(
        self,
        key: str,
        value: Any,
        *,
        ttl: int | None = None,
    ) -> None:
        """Store a value in memory.

        Args:
            key: Storage key.
            value: Value to store (must be JSON-serializable).
            ttl: Time-to-live in seconds. None = forever.
        """
        start = time.perf_counter()
        await self._store.set(self._key(key), value, ttl=ttl)
        await self._emit("memory.write", {
            "key": key,
            "ttl": ttl,
            "duration_ms": (time.perf_counter() - start) * 1000,
        })

    async def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from memory.

        Args:
            key: Storage key.
            default: Value to return if key not found.

        Returns:
            Stored value or default.
        """
        start = time.perf_counter()
        result = await self._store.get(self._key(key))
        found = result is not None
        await self._emit("memory.read", {
            "key": key,
            "found": found,
            "duration_ms": (time.perf_counter() - start) * 1000,
        })
        return result if found else default

    async def forget(self, key: str) -> bool:
        """Remove a value from memory.

        Args:
            key: Storage key.

        Returns:
            True if key was deleted.
        """
        deleted = await self._store.delete(self._key(key))
        await self._emit("memory.delete", {
            "key": key,
            "deleted": deleted,
        })
        return deleted

    async def clear(self) -> None:
        """Clear all memory in this namespace."""
        prefix = self._namespace + ":" if self._namespace else ""
        await self._store.clear(prefix)
        await self._emit("memory.clear", {})

    async def keys(self) -> list[str]:
        """List all keys in this namespace."""
        prefix = self._namespace + ":" if self._namespace else ""
        all_keys = await self._store.keys(prefix)

        # Strip namespace prefix from returned keys
        if self._namespace:
            prefix_len = len(self._namespace) + 1
            return [k[prefix_len:] for k in all_keys]
        return all_keys

    async def list(self, prefix: str = "") -> list[str]:
        """List all keys in this namespace (alias for keys()).

        Args:
            prefix: Optional prefix filter within namespace.

        Returns:
            List of keys.
        """
        all_keys = await self.keys()
        if prefix:
            return [k for k in all_keys if k.startswith(prefix)]
        return all_keys

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    async def remember_many(
        self,
        items: dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> None:
        """Store multiple values in memory.

        Args:
            items: Dict of key -> value pairs.
            ttl: Time-to-live in seconds. Applied to all items.
        """
        # Build namespaced items
        namespaced = {self._key(k): v for k, v in items.items()}

        # Use batch operation if store supports it
        if hasattr(self._store, "set_many"):
            await self._store.set_many(namespaced, ttl=ttl)
        else:
            # Fallback to individual sets
            for key, value in namespaced.items():
                await self._store.set(key, value, ttl=ttl)

    async def recall_many(
        self,
        keys: list[str],
        default: Any = None,
    ) -> dict[str, Any]:
        """Retrieve multiple values from memory.

        Args:
            keys: List of storage keys.
            default: Value to use for missing keys.

        Returns:
            Dict of key -> value (or default).
        """
        # Build namespaced keys
        namespaced_keys = [self._key(k) for k in keys]

        # Use batch operation if store supports it
        if hasattr(self._store, "get_many"):
            results = await self._store.get_many(namespaced_keys)
            # Un-namespace the keys in result
            return {
                k: (results.get(self._key(k)) if results.get(self._key(k)) is not None else default)
                for k in keys
            }
        else:
            # Fallback to individual gets
            return {
                k: (await self._store.get(self._key(k)) or default)
                for k in keys
            }

    # -------------------------------------------------------------------------
    # Semantic Search (VectorStore integration)
    # -------------------------------------------------------------------------

    async def add_document(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add a document for semantic search.

        Requires a VectorStore to be configured.

        Args:
            text: Document text.
            metadata: Optional metadata.
            doc_id: Optional document ID. Auto-generated if not provided.

        Returns:
            Document ID.

        Raises:
            RuntimeError: If no VectorStore is configured.
        """
        if not self._vectorstore:
            raise RuntimeError("No vectorstore configured. Pass vectorstore= to Memory().")

        doc_id = doc_id or str(uuid.uuid4())
        full_metadata = {"namespace": self._namespace, **(metadata or {})}

        await self._vectorstore.add_texts([text], metadatas=[full_metadata], ids=[doc_id])
        return doc_id

    async def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Any]:  # Returns list of SearchResult from vectorstore
        """Semantic search across documents.

        Requires a VectorStore to be configured.

        Args:
            query: Search query.
            k: Maximum results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            List of SearchResult from vectorstore.

        Raises:
            RuntimeError: If no VectorStore is configured.
        """
        if not self._vectorstore:
            raise RuntimeError("No vectorstore configured. Pass vectorstore= to Memory().")

        start = time.perf_counter()

        # Filter by namespace
        ns_filter = {"namespace": self._namespace} if self._namespace else {}
        combined_filter = {**ns_filter, **(filter_metadata or {})}

        results = await self._vectorstore.search(
            query,
            k=k,
            filter=combined_filter if combined_filter else None,
        )

        await self._emit("memory.search", {
            "query": query[:100],  # Truncate for logging
            "k": k,
            "results_count": len(results),
            "duration_ms": (time.perf_counter() - start) * 1000,
        })

        return results

    # -------------------------------------------------------------------------
    # Message History (Conversation)
    # -------------------------------------------------------------------------

    async def add_message(self, message: Message) -> None:
        """Add a message to conversation history.

        Args:
            message: Message to add.
        """
        messages_key = "_messages"
        async with self._lock:
            messages = await self.recall(messages_key, [])
            messages.append(_message_to_dict(message))
            await self.remember(messages_key, messages)

    async def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get conversation history.

        Args:
            limit: Maximum messages to return. None = all.

        Returns:
            List of messages, oldest first.
        """
        messages_key = "_messages"
        raw = await self.recall(messages_key, [])

        messages = [_dict_to_message(m) for m in raw]

        if limit:
            messages = messages[-limit:]

        return messages

    async def clear_messages(self) -> None:
        """Clear conversation history."""
        await self.forget("_messages")

    # -------------------------------------------------------------------------
    # Thread-Aware Methods (Agent Integration)
    # -------------------------------------------------------------------------

    def thread(self, thread_id: str) -> Memory:
        """Get a scoped view for a specific thread.

        Convenience method that creates a scoped view with thread prefix.
        """
        return self.scoped(f"thread:{thread_id}")

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Get messages for a specific thread.

        Args:
            thread_id: Thread identifier.
            limit: Maximum messages to return.

        Returns:
            List of messages for the thread.
        """
        return await self.thread(thread_id).get_messages(limit)

    async def add_thread_message(
        self,
        thread_id: str,
        message: Message,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to a specific thread.

        Args:
            thread_id: Thread identifier.
            message: Message to add.
            metadata: Optional metadata (stored separately).
        """
        thread = self.thread(thread_id)
        await thread.add_message(message)
        if metadata:
            await thread.merge("_metadata", metadata)

    async def add_thread_messages(
        self,
        thread_id: str,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add multiple messages to a specific thread.

        Args:
            thread_id: Thread identifier.
            messages: Messages to add.
            metadata: Optional metadata.
        """
        thread = self.thread(thread_id)
        for msg in messages:
            await thread.add_message(msg)
        if metadata:
            await thread.merge("_metadata", metadata)

    async def get_context_for_prompt(
        self,
        thread_id: str,
        user_id: str | None = None,
    ) -> str:
        """Get memory context to inject into agent prompts.

        Args:
            thread_id: Current thread ID.
            user_id: Optional user ID for user-specific facts.

        Returns:
            Formatted context string (empty if no relevant memory).
        """
        parts = []

        # Get all global facts (not thread or internal)
        all_keys = await self.keys()
        global_facts = []
        for key in all_keys:
            # Skip internal keys (threads, metadata, etc.)
            if key.startswith(("thread:", "_", "user:")):
                continue
            value = await self.recall(key)
            if value is not None:
                global_facts.append(f"- {key}: {value}")

        if global_facts:
            parts.append("Known facts:\n" + "\n".join(global_facts[:15]))  # Limit

        # Get user-specific facts if user_id provided
        if user_id:
            user_mem = self.scoped(f"user:{user_id}:facts")
            user_keys = await user_mem.keys()
            if user_keys:
                facts = []
                for key in user_keys[:10]:  # Limit to 10 facts
                    value = await user_mem.recall(key)
                    facts.append(f"- {key}: {value}")
                if facts:
                    parts.append("User facts:\n" + "\n".join(facts))

        return "\n\n".join(parts)

    @property
    def has_long_term(self) -> bool:
        """Whether this memory supports long-term storage.

        Returns True if a persistent store is configured.
        """
        # InMemoryStore is not persistent
        return not isinstance(self._store, InMemoryStore)

    # -------------------------------------------------------------------------
    # Scoped Views
    # -------------------------------------------------------------------------

    def scoped(self, namespace: str) -> Memory:
        """Create a scoped view of this memory.

        The scoped memory shares the same store and vectorstore,
        but uses a different namespace for isolation.

        Args:
            namespace: Namespace for the scoped view.

        Returns:
            New Memory instance with scoped namespace.

        Example:
            ```python
            memory = Memory()
            team_mem = memory.scoped("team:research")
            agent_mem = memory.scoped("agent:analyst")

            # These are isolated:
            await team_mem.remember("status", "active")
            await agent_mem.remember("status", "thinking")

            # team:research:status != agent:analyst:status
            ```
        """
        # Combine namespaces
        full_ns = f"{self._namespace}:{namespace}" if self._namespace else namespace

        return Memory(
            store=self._store,
            vectorstore=self._vectorstore,
            namespace=full_ns,
        )

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def append(self, key: str, value: Any) -> None:
        """Append a value to a list at key.

        Creates the list if it doesn't exist.
        """
        async with self._lock:
            existing = await self.recall(key, [])
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(value)
            await self.remember(key, existing)

    async def merge(self, key: str, data: dict[str, Any]) -> None:
        """Merge a dict with existing dict at key.

        Creates the dict if it doesn't exist.
        """
        async with self._lock:
            existing = await self.recall(key, {})
            if not isinstance(existing, dict):
                raise TypeError(f"Cannot merge dict with {type(existing)}")
            existing.update(data)
            await self.remember(key, existing)

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter at key.

        Creates the counter (starting at 0) if it doesn't exist.
        """
        async with self._lock:
            existing = await self.recall(key, 0)
            new_value = existing + amount
            await self.remember(key, new_value)
            return new_value


# =============================================================================
# INMEMORY STORE - Default Implementation
# =============================================================================


class InMemoryStore:
    """In-memory store for testing and development.

    Fast, zero-config, but no persistence across restarts.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float | None]] = {}  # key -> (value, expiry_time)

    async def get(self, key: str, default: Any = None) -> Any | None:
        """Get value, checking expiry."""
        if key not in self._data:
            return default

        value, expiry = self._data[key]

        # Check expiry
        if expiry is not None and datetime.now(UTC).timestamp() > expiry:
            del self._data[key]
            return default

        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value with optional TTL."""
        expiry = None
        if ttl is not None:
            expiry = datetime.now(UTC).timestamp() + ttl

        self._data[key] = (value, expiry)

    async def delete(self, key: str) -> bool:
        """Delete by key."""
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def keys(self, prefix: str = "") -> list[str]:
        """List keys matching prefix."""
        # Clean expired entries first
        now = datetime.now(UTC).timestamp()
        expired = [k for k, (_, exp) in self._data.items() if exp and now > exp]
        for k in expired:
            del self._data[k]

        if not prefix:
            return list(self._data.keys())
        return [k for k in self._data if k.startswith(prefix)]

    async def clear(self, prefix: str = "") -> None:
        """Clear keys matching prefix."""
        if not prefix:
            self._data.clear()
        else:
            to_delete = [k for k in self._data if k.startswith(prefix)]
            for k in to_delete:
                del self._data[k]

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values by keys."""
        return {key: await self.get(key) for key in keys}

    async def set_many(
        self, items: dict[str, Any], ttl: int | None = None
    ) -> None:
        """Set multiple values."""
        for key, value in items.items():
            await self.set(key, value, ttl=ttl)


# =============================================================================
# MESSAGE SERIALIZATION HELPERS
# =============================================================================


def _message_to_dict(message: Message) -> dict[str, Any]:
    """Convert a Message to a serializable dict."""
    return {
        "role": message.role,
        "content": message.content,
        "name": getattr(message, "name", None),
        "tool_calls": getattr(message, "tool_calls", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
    }


def _dict_to_message(data: dict[str, Any]) -> Message:
    """Convert a dict back to a Message."""
    from agenticflow.core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    role = data["role"]
    content = data["content"]

    if role == "user":
        return HumanMessage(content)
    elif role == "assistant":
        return AIMessage(content, tool_calls=data.get("tool_calls"))
    elif role == "tool":
        return ToolMessage(content, tool_call_id=data.get("tool_call_id", ""))
    elif role == "system":
        return SystemMessage(content)
    else:
        # Fallback to HumanMessage for unknown roles
        return HumanMessage(content)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Memory",
    "Store",
    "InMemoryStore",
]
