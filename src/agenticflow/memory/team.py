"""Team memory for shared state between agents in a topology.

Enables agents to share information during multi-agent execution.
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


class TeamMemory(BaseMemory):
    """Shared memory for agents working together in a topology.

    Provides a shared key-value store that all agents in a team can read/write.
    Useful for coordination, sharing research results, passing context, etc.

    Example:
        # Create shared team memory
        team_memory = TeamMemory(team_id="research-team")

        # Agent 1 (Researcher) stores findings
        await team_memory.save("findings", {"topic": "AI", "summary": "..."})

        # Agent 2 (Writer) reads findings
        findings = await team_memory.load("findings")

        # Store coordination info
        await team_memory.save("status", {"researcher": "done", "writer": "in_progress"})

        # Pass in topology
        topology = Supervisor(
            coordinator=manager,
            workers=[researcher, writer],
            team_memory=team_memory,
        )
    """

    def __init__(
        self,
        team_id: str | None = None,
        backend: MemoryBackend | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize team memory.

        Args:
            team_id: Team/topology identifier. Auto-generated if None.
            backend: Storage backend. If None, uses InMemoryBackend.
            config: Additional configuration.
        """
        super().__init__(backend=backend or InMemoryBackend(), config=config)
        self._team_id = team_id or self._generate_id()

    @property
    def team_id(self) -> str:
        """Get the team identifier."""
        return self._team_id

    async def save(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Save a value to team memory.

        Args:
            key: Key to store under.
            value: Value to store.
            ttl_seconds: Optional time-to-live in seconds.
            metadata: Optional metadata.
        """
        await self._backend.set(
            key=key,
            value=value,
            scope=MemoryScope.TEAM,
            scope_id=self._team_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )

    async def load(self, key: str, **kwargs: Any) -> Any | None:
        """Load a value from team memory.

        Args:
            key: Key to retrieve.

        Returns:
            The stored value, or None if not found.
        """
        return await self._backend.get(key, MemoryScope.TEAM, self._team_id)

    async def delete(self, key: str) -> bool:
        """Delete a value from team memory.

        Args:
            key: Key to delete.

        Returns:
            True if deleted, False if not found.
        """
        return await self._backend.delete(key, MemoryScope.TEAM, self._team_id)

    async def list_keys(self) -> list[str]:
        """List all keys in team memory.

        Returns:
            List of keys.
        """
        return await self._backend.list_keys(MemoryScope.TEAM, self._team_id)

    async def get_all(self) -> dict[str, Any]:
        """Get all key-value pairs in team memory.

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

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search for entries matching a query.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching MemoryEntry objects.
        """
        return await self._backend.search(
            query, MemoryScope.TEAM, self._team_id, limit
        )

    async def clear(self, **kwargs: Any) -> None:
        """Clear all team memory."""
        await self._backend.clear(MemoryScope.TEAM, self._team_id)

    async def has_key(self, key: str) -> bool:
        """Check if a key exists in team memory.

        Args:
            key: Key to check.

        Returns:
            True if the key exists.
        """
        value = await self.load(key)
        return value is not None

    # Convenience methods for common patterns

    async def append_to_list(self, key: str, value: Any) -> None:
        """Append a value to a list stored at key.

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

    async def merge_dict(self, key: str, data: dict[str, Any]) -> None:
        """Merge a dict with existing dict at key.

        Creates the dict if it doesn't exist.

        Args:
            key: Key of the dict.
            data: Data to merge.
        """
        existing = await self.load(key)
        if existing is None:
            existing = {}
        elif not isinstance(existing, dict):
            raise TypeError(f"Cannot merge dict with {type(existing)}")
        existing.update(data)
        await self.save(key, existing)

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter stored at key.

        Creates the counter (starting at 0) if it doesn't exist.

        Args:
            key: Key of the counter.
            amount: Amount to increment by.

        Returns:
            The new counter value.
        """
        existing = await self.load(key)
        if existing is None:
            existing = 0
        elif not isinstance(existing, (int, float)):
            raise TypeError(f"Cannot increment {type(existing)}")
        new_value = existing + amount
        await self.save(key, new_value)
        return new_value

    # Agent-specific convenience methods

    async def report_status(self, agent_name: str, status: str) -> None:
        """Report an agent's status to team memory.

        Args:
            agent_name: Name of the agent.
            status: Status string (e.g., "working", "done", "error").
        """
        await self.merge_dict("agent_statuses", {agent_name: status})

    async def get_agent_statuses(self) -> dict[str, str]:
        """Get all agent statuses.

        Returns:
            Dict mapping agent name to status.
        """
        statuses = await self.load("agent_statuses")
        return statuses if isinstance(statuses, dict) else {}

    async def share_result(self, agent_name: str, result: Any) -> None:
        """Share an agent's result with the team.

        Args:
            agent_name: Name of the agent.
            result: Result to share.
        """
        await self.merge_dict("agent_results", {agent_name: result})

    async def get_agent_results(self) -> dict[str, Any]:
        """Get all shared agent results.

        Returns:
            Dict mapping agent name to result.
        """
        results = await self.load("agent_results")
        return results if isinstance(results, dict) else {}
