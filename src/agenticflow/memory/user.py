"""User memory for long-term user facts and preferences.

Stores key-value facts per user with search capability.
"""

from __future__ import annotations

from typing import Any

from agenticflow.memory.backends.inmemory import InMemoryBackend
from agenticflow.memory.base import (
    BaseMemory,
    MemoryBackend,
    MemoryConfig,
    MemoryEntry,
    MemoryScope,
)


class UserMemory(BaseMemory):
    """Memory for user-specific facts and preferences.

    Stores key-value pairs scoped to user_id for long-term memory.
    Supports simple text search across stored facts.

    Example:
        memory = UserMemory()

        # Store user facts
        await memory.save("preference", "dark mode", user_id="user-123")
        await memory.save("language", "Python", user_id="user-123")

        # Retrieve facts
        pref = await memory.load("preference", user_id="user-123")

        # Search facts
        results = await memory.search("mode", user_id="user-123")

        # List all facts for a user
        facts = await memory.list_facts(user_id="user-123")
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize user memory.

        Args:
            backend: Storage backend. If None, uses InMemoryBackend.
            config: Additional configuration.
        """
        super().__init__(backend=backend or InMemoryBackend(), config=config)

    async def save(
        self,
        key: str,
        value: Any,
        *,
        user_id: str,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Save a fact for a user.

        Args:
            key: Fact key (e.g., "preference", "name", "favorite_language").
            value: Fact value.
            user_id: User identifier.
            ttl_seconds: Optional time-to-live in seconds.
            metadata: Optional metadata.
        """
        await self._backend.set(
            key=key,
            value=value,
            scope=MemoryScope.USER,
            scope_id=user_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )

    async def load(self, key: str, *, user_id: str, **kwargs: Any) -> Any | None:
        """Load a fact for a user.

        Args:
            key: Fact key.
            user_id: User identifier.

        Returns:
            The stored value, or None if not found.
        """
        return await self._backend.get(key, MemoryScope.USER, user_id)

    async def delete(self, key: str, *, user_id: str) -> bool:
        """Delete a fact for a user.

        Args:
            key: Fact key.
            user_id: User identifier.

        Returns:
            True if deleted, False if not found.
        """
        return await self._backend.delete(key, MemoryScope.USER, user_id)

    async def list_facts(self, user_id: str) -> list[str]:
        """List all fact keys for a user.

        Args:
            user_id: User identifier.

        Returns:
            List of fact keys.
        """
        return await self._backend.list_keys(MemoryScope.USER, user_id)

    async def get_all_facts(self, user_id: str) -> dict[str, Any]:
        """Get all facts for a user as a dictionary.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary of key -> value.
        """
        keys = await self.list_facts(user_id)
        facts = {}
        for key in keys:
            value = await self.load(key, user_id=user_id)
            if value is not None:
                facts[key] = value
        return facts

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search for facts matching a query.

        Performs case-insensitive substring matching.

        Args:
            query: Search query.
            user_id: User identifier.
            limit: Maximum results.

        Returns:
            List of matching MemoryEntry objects.
        """
        return await self._backend.search(query, MemoryScope.USER, user_id, limit)

    async def clear(self, *, user_id: str | None = None, **kwargs: Any) -> None:
        """Clear facts for a user.

        Args:
            user_id: User identifier. If None, raises ValueError.
        """
        if user_id is None:
            raise ValueError("user_id is required to clear user memory")
        await self._backend.clear(MemoryScope.USER, user_id)

    async def has_fact(self, key: str, *, user_id: str) -> bool:
        """Check if a fact exists for a user.

        Args:
            key: Fact key.
            user_id: User identifier.

        Returns:
            True if the fact exists.
        """
        value = await self.load(key, user_id=user_id)
        return value is not None

    async def update(
        self,
        key: str,
        value: Any,
        *,
        user_id: str,
        merge: bool = False,
    ) -> None:
        """Update a fact, optionally merging with existing value.

        Args:
            key: Fact key.
            value: New value.
            user_id: User identifier.
            merge: If True and both values are dicts, merge them.
        """
        if merge:
            existing = await self.load(key, user_id=user_id)
            if isinstance(existing, dict) and isinstance(value, dict):
                value = {**existing, **value}
        await self.save(key, value, user_id=user_id)
