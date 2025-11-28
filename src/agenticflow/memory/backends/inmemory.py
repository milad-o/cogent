"""In-memory storage backend for memory system.

Fast, zero-config backend that stores everything in memory.
Data is lost when the process exits.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agenticflow.memory.base import MemoryEntry, MemoryScope

if TYPE_CHECKING:
    from agenticflow.core.messages import Message


@dataclass
class InMemoryBackend:
    """In-memory implementation of MemoryBackend.

    Stores data in nested dictionaries: scope -> scope_id -> key -> entry.
    Thread-safe for single-threaded async use.

    Example:
        backend = InMemoryBackend()
        await backend.set("name", "Alice", MemoryScope.USER, "user-123")
        name = await backend.get("name", MemoryScope.USER, "user-123")
    """

    _store: dict[MemoryScope, dict[str, dict[str, MemoryEntry]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )

    async def get(self, key: str, scope: MemoryScope, scope_id: str) -> Any | None:
        """Get a value by key within a scope.

        Args:
            key: The key to look up.
            scope: Memory scope (THREAD, USER, EXECUTION, GLOBAL).
            scope_id: ID within the scope (thread_id, user_id, etc.).

        Returns:
            The stored value, or None if not found or expired.
        """
        entry = self._store.get(scope, {}).get(scope_id, {}).get(key)
        if entry is None:
            return None
        if entry.is_expired:
            # Clean up expired entry
            del self._store[scope][scope_id][key]
            return None
        return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        scope: MemoryScope,
        scope_id: str,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set a value by key within a scope.

        Args:
            key: The key to store under.
            value: The value to store.
            scope: Memory scope.
            scope_id: ID within the scope.
            ttl_seconds: Optional time-to-live in seconds.
            metadata: Optional metadata dict.
        """
        now = datetime.now(UTC)
        existing = self._store[scope][scope_id].get(key)

        entry = MemoryEntry(
            key=key,
            value=value,
            scope=scope,
            scope_id=scope_id,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds,
        )
        self._store[scope][scope_id][key] = entry

    async def delete(self, key: str, scope: MemoryScope, scope_id: str) -> bool:
        """Delete a value by key.

        Args:
            key: The key to delete.
            scope: Memory scope.
            scope_id: ID within the scope.

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        if key in self._store.get(scope, {}).get(scope_id, {}):
            del self._store[scope][scope_id][key]
            return True
        return False

    async def list_keys(self, scope: MemoryScope, scope_id: str) -> list[str]:
        """List all keys within a scope.

        Args:
            scope: Memory scope.
            scope_id: ID within the scope.

        Returns:
            List of keys (excluding expired entries).
        """
        entries = self._store.get(scope, {}).get(scope_id, {})
        # Filter out expired entries
        valid_keys = []
        expired_keys = []
        for key, entry in entries.items():
            if entry.is_expired:
                expired_keys.append(key)
            else:
                valid_keys.append(key)
        # Clean up expired
        for key in expired_keys:
            del self._store[scope][scope_id][key]
        return valid_keys

    async def clear(self, scope: MemoryScope, scope_id: str) -> None:
        """Clear all entries within a scope.

        Args:
            scope: Memory scope.
            scope_id: ID within the scope.
        """
        if scope in self._store and scope_id in self._store[scope]:
            self._store[scope][scope_id].clear()

    async def search(
        self,
        query: str,
        scope: MemoryScope,
        scope_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search for entries matching a query (simple text search).

        Performs case-insensitive substring matching on keys and string values.

        Args:
            query: Search query string.
            scope: Memory scope.
            scope_id: ID within the scope.
            limit: Maximum number of results.

        Returns:
            List of matching MemoryEntry objects.
        """
        query_lower = query.lower()
        results: list[MemoryEntry] = []

        entries = self._store.get(scope, {}).get(scope_id, {})
        for entry in entries.values():
            if entry.is_expired:
                continue
            # Search in key
            if query_lower in entry.key.lower():
                results.append(entry)
                continue
            # Search in string values
            if isinstance(entry.value, str) and query_lower in entry.value.lower():
                results.append(entry)
                continue
            if len(results) >= limit:
                break

        return results[:limit]

    async def clear_all(self) -> None:
        """Clear all data across all scopes. Use with caution."""
        self._store.clear()


@dataclass
class InMemoryConversationBackend:
    """In-memory implementation of ConversationBackend.

    Optimized for conversation history with ordering preserved.

    Example:
        backend = InMemoryConversationBackend()
        await backend.add_message("thread-1", message)
        messages = await backend.get_messages("thread-1", limit=10)
    """

    _threads: dict[str, list[tuple[datetime, Message, dict[str, Any]]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    async def add_message(
        self,
        thread_id: str,
        message: Message,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to conversation history.

        Args:
            thread_id: The conversation thread ID.
            message: The message to add.
            metadata: Optional metadata.
        """
        self._threads[thread_id].append(
            (datetime.now(UTC), message, metadata or {})
        )

    async def get_messages(
        self,
        thread_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Get messages from conversation history.

        Args:
            thread_id: The conversation thread ID.
            limit: Maximum number of messages to return.
            offset: Number of messages to skip from the start.

        Returns:
            List of messages, oldest first.
        """
        entries = self._threads.get(thread_id, [])
        if offset:
            entries = entries[offset:]
        if limit is not None:
            entries = entries[:limit]
        return [msg for _, msg, _ in entries]

    async def get_recent_messages(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[Message]:
        """Get the most recent messages from a thread.

        Args:
            thread_id: The conversation thread ID.
            limit: Maximum number of messages to return.

        Returns:
            List of messages, oldest first (but only the last N).
        """
        entries = self._threads.get(thread_id, [])
        recent = entries[-limit:] if limit else entries
        return [msg for _, msg, _ in recent]

    async def clear_thread(self, thread_id: str) -> None:
        """Clear all messages in a thread.

        Args:
            thread_id: The conversation thread ID.
        """
        if thread_id in self._threads:
            self._threads[thread_id].clear()

    async def get_message_count(self, thread_id: str) -> int:
        """Get the number of messages in a thread.

        Args:
            thread_id: The conversation thread ID.

        Returns:
            Number of messages.
        """
        return len(self._threads.get(thread_id, []))

    async def list_threads(self) -> list[str]:
        """List all thread IDs.

        Returns:
            List of thread IDs that have messages.
        """
        return [tid for tid, msgs in self._threads.items() if msgs]

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread entirely.

        Args:
            thread_id: The conversation thread ID.

        Returns:
            True if the thread existed and was deleted.
        """
        if thread_id in self._threads:
            del self._threads[thread_id]
            return True
        return False

    async def clear_all(self) -> None:
        """Clear all threads. Use with caution."""
        self._threads.clear()
