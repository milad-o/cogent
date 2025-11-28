"""Working memory for current execution scratchpad.

Temporary storage for intermediate results during a single execution.
"""

from __future__ import annotations

from typing import Any, TypeVar, overload

from agenticflow.memory.backends.inmemory import InMemoryBackend
from agenticflow.memory.base import (
    BaseMemory,
    MemoryBackend,
    MemoryConfig,
    MemoryScope,
)

T = TypeVar("T")


class WorkingMemory(BaseMemory):
    """Scratchpad memory for current execution.

    Provides a simple dict-like interface for storing intermediate results
    during a single task execution. Auto-cleans up after execution.

    Unlike TeamMemory (persistent across topology executions), WorkingMemory
    is scoped to a single run and cleared afterward.

    Example:
        working = WorkingMemory()

        # Store intermediate results
        await working.save("draft", "First draft of the article...")
        await working.save("research", {"sources": [...], "summary": "..."})

        # Retrieve
        draft = await working.load("draft")

        # Check contents
        keys = await working.list_keys()

        # Clear after execution
        await working.clear()
    """

    def __init__(
        self,
        execution_id: str | None = None,
        backend: MemoryBackend | None = None,
        config: MemoryConfig | None = None,
        auto_cleanup: bool = True,
    ) -> None:
        """Initialize working memory.

        Args:
            execution_id: Execution identifier. Auto-generated if None.
            backend: Storage backend. If None, uses InMemoryBackend.
            config: Additional configuration.
            auto_cleanup: If True, clear on __aexit__ (when used as context manager).
        """
        super().__init__(backend=backend or InMemoryBackend(), config=config)
        self._execution_id = execution_id or self._generate_id()
        self._auto_cleanup = auto_cleanup

    @property
    def execution_id(self) -> str:
        """Get the execution identifier."""
        return self._execution_id

    async def __aenter__(self) -> WorkingMemory:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager, optionally cleaning up."""
        if self._auto_cleanup:
            await self.clear()

    async def save(
        self,
        key: str,
        value: Any,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Save a value to working memory.

        Args:
            key: Key to store under.
            value: Value to store.
            metadata: Optional metadata.
        """
        await self._backend.set(
            key=key,
            value=value,
            scope=MemoryScope.EXECUTION,
            scope_id=self._execution_id,
            metadata=metadata,
        )

    async def load(self, key: str, **kwargs: Any) -> Any | None:
        """Load a value from working memory.

        Args:
            key: Key to retrieve.

        Returns:
            The stored value, or None if not found.
        """
        return await self._backend.get(key, MemoryScope.EXECUTION, self._execution_id)

    @overload
    async def get(self, key: str) -> Any | None: ...

    @overload
    async def get(self, key: str, default: T) -> Any | T: ...

    async def get(self, key: str, default: Any = None) -> Any:
        """Load a value with a default fallback.

        Args:
            key: Key to retrieve.
            default: Default value if key not found.

        Returns:
            The stored value, or default if not found.
        """
        value = await self.load(key)
        return value if value is not None else default

    async def delete(self, key: str) -> bool:
        """Delete a value from working memory.

        Args:
            key: Key to delete.

        Returns:
            True if deleted, False if not found.
        """
        return await self._backend.delete(
            key, MemoryScope.EXECUTION, self._execution_id
        )

    async def list_keys(self) -> list[str]:
        """List all keys in working memory.

        Returns:
            List of keys.
        """
        return await self._backend.list_keys(MemoryScope.EXECUTION, self._execution_id)

    async def get_all(self) -> dict[str, Any]:
        """Get all key-value pairs in working memory.

        Returns:
            Dictionary of all stored data.
        """
        keys = await self.list_keys()
        data = {}
        for key in keys:
            value = await self.load(key)
            if value is not None:
                data[key] = value
        return data

    async def clear(self, **kwargs: Any) -> None:
        """Clear all working memory."""
        await self._backend.clear(MemoryScope.EXECUTION, self._execution_id)

    async def has_key(self, key: str) -> bool:
        """Check if a key exists in working memory.

        Args:
            key: Key to check.

        Returns:
            True if the key exists.
        """
        value = await self.load(key)
        return value is not None

    # Convenience methods

    async def append(self, key: str, value: Any) -> None:
        """Append a value to a list at key.

        Creates the list if it doesn't exist.

        Args:
            key: Key of the list.
            value: Value to append.
        """
        existing = await self.load(key)
        if existing is None:
            existing = []
        elif not isinstance(existing, list):
            existing = [existing]
        existing.append(value)
        await self.save(key, existing)

    async def update(self, key: str, data: dict[str, Any]) -> None:
        """Update a dict at key with new data.

        Creates the dict if it doesn't exist.

        Args:
            key: Key of the dict.
            data: Data to merge.
        """
        existing = await self.load(key)
        if existing is None:
            existing = {}
        elif not isinstance(existing, dict):
            raise TypeError(f"Cannot update non-dict type: {type(existing)}")
        existing.update(data)
        await self.save(key, existing)

    async def set_if_absent(self, key: str, value: Any) -> bool:
        """Set a value only if the key doesn't exist.

        Args:
            key: Key to set.
            value: Value to set.

        Returns:
            True if value was set, False if key already existed.
        """
        if await self.has_key(key):
            return False
        await self.save(key, value)
        return True

    # Type-safe getters for common types

    async def get_str(self, key: str, default: str = "") -> str:
        """Get a string value.

        Args:
            key: Key to retrieve.
            default: Default if not found or wrong type.

        Returns:
            String value.
        """
        value = await self.load(key)
        return str(value) if value is not None else default

    async def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer value.

        Args:
            key: Key to retrieve.
            default: Default if not found or wrong type.

        Returns:
            Integer value.
        """
        value = await self.load(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    async def get_list(self, key: str) -> list[Any]:
        """Get a list value.

        Args:
            key: Key to retrieve.

        Returns:
            List value, or empty list if not found.
        """
        value = await self.load(key)
        if value is None:
            return []
        return list(value) if not isinstance(value, list) else value

    async def get_dict(self, key: str) -> dict[str, Any]:
        """Get a dict value.

        Args:
            key: Key to retrieve.

        Returns:
            Dict value, or empty dict if not found.
        """
        value = await self.load(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value)}")
        return value
