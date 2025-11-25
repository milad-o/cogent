"""
Shared memory - multi-agent collaboration memory.

Enables agents to share information, coordinate, and collaborate.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from agenticflow.memory.base import (
    Memory,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
)
from agenticflow.core.utils import generate_id, now_utc


class SharedMemory(Memory[Any]):
    """
    Shared memory for multi-agent collaboration.
    
    Enables agents to:
    - Share information with other agents
    - Coordinate on tasks
    - Build shared knowledge bases
    - Pass intermediate results
    
    Attributes:
        workspace_id: Shared workspace identifier
        
    Example:
        ```python
        # Create shared workspace
        shared = SharedMemory(workspace_id="project-alpha")
        
        # Agent A publishes findings
        await shared.publish(
            "market_analysis",
            {"trend": "up", "confidence": 0.85},
            publisher_id="analyst_agent",
        )
        
        # Agent B subscribes to updates
        data = await shared.get_latest("market_analysis")
        ```
    """

    # Class-level shared storage (singleton pattern for shared state)
    _workspaces: dict[str, dict[str, dict[str, MemoryEntry]]] = {}
    _subscribers: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    _lock = asyncio.Lock()

    def __init__(
        self,
        workspace_id: str | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """
        Initialize shared memory.
        
        Args:
            workspace_id: Shared workspace ID
            config: Memory configuration
        """
        cfg = config or MemoryConfig(memory_type=MemoryType.SHARED)
        super().__init__(cfg)
        self.workspace_id = workspace_id or "default"
        
        # Initialize workspace if needed
        if self.workspace_id not in self._workspaces:
            self._workspaces[self.workspace_id] = defaultdict(dict)

    @property
    def _storage(self) -> dict[str, dict[str, MemoryEntry]]:
        """Get the storage for current workspace."""
        return self._workspaces[self.workspace_id]

    async def add(
        self,
        content: Any,
        metadata: dict | None = None,
        namespace: str | None = None,
    ) -> MemoryEntry:
        """Add a memory entry to shared storage."""
        ns = namespace or "shared"
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
            namespace=ns,
        )
        
        async with self._lock:
            self._storage[ns][entry.id] = entry

            # Notify subscribers
            await self._notify_subscribers(ns, entry)

            # Enforce max entries per namespace
            if len(self._storage[ns]) > self.config.max_entries:
                oldest = min(
                    self._storage[ns].values(),
                    key=lambda e: e.timestamp,
                )
                del self._storage[ns][oldest.id]

        return entry

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry."""
        for ns_entries in self._storage.values():
            if entry_id in ns_entries:
                return ns_entries[entry_id]
        return None

    async def search(
        self,
        query: str | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search shared entries."""
        if namespace:
            entries = list(self._storage.get(namespace, {}).values())
        else:
            entries = []
            for ns_entries in self._storage.values():
                entries.extend(ns_entries.values())

        # Apply metadata filter
        if metadata_filter:
            filtered = []
            for entry in entries:
                match = all(
                    entry.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                )
                if match:
                    filtered.append(entry)
            entries = filtered

        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries[:limit]

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        async with self._lock:
            for ns_entries in self._storage.values():
                if entry_id in ns_entries:
                    del ns_entries[entry_id]
                    return True
        return False

    async def clear(self, namespace: str | None = None) -> int:
        """Clear entries."""
        async with self._lock:
            if namespace:
                count = len(self._storage.get(namespace, {}))
                self._storage[namespace] = {}
            else:
                count = sum(len(e) for e in self._storage.values())
                self._storage.clear()
        return count

    # ========================================================================
    # Pub/Sub methods for agent coordination
    # ========================================================================

    async def publish(
        self,
        topic: str,
        content: Any,
        publisher_id: str | None = None,
    ) -> MemoryEntry:
        """
        Publish content to a topic.
        
        Args:
            topic: The topic/channel name
            content: Content to publish
            publisher_id: ID of publishing agent
            
        Returns:
            The created MemoryEntry
        """
        return await self.add(
            content=content,
            metadata={
                "type": "publication",
                "topic": topic,
                "publisher_id": publisher_id,
            },
            namespace=f"topic:{topic}",
        )

    async def subscribe(
        self,
        topic: str,
        callback: Any,
        subscriber_id: str | None = None,
    ) -> None:
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to
            callback: Async callback function(entry: MemoryEntry)
            subscriber_id: ID of subscribing agent
        """
        key = f"topic:{topic}"
        self._subscribers[self.workspace_id][key].append({
            "callback": callback,
            "subscriber_id": subscriber_id,
        })

    async def unsubscribe(
        self,
        topic: str,
        subscriber_id: str,
    ) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: The topic
            subscriber_id: The subscriber ID
            
        Returns:
            True if unsubscribed
        """
        key = f"topic:{topic}"
        subs = self._subscribers[self.workspace_id][key]
        original_len = len(subs)
        self._subscribers[self.workspace_id][key] = [
            s for s in subs if s["subscriber_id"] != subscriber_id
        ]
        return len(self._subscribers[self.workspace_id][key]) < original_len

    async def _notify_subscribers(self, namespace: str, entry: MemoryEntry) -> None:
        """Notify subscribers of new entry."""
        subs = self._subscribers[self.workspace_id].get(namespace, [])
        for sub in subs:
            try:
                callback = sub["callback"]
                if asyncio.iscoroutinefunction(callback):
                    await callback(entry)
                else:
                    callback(entry)
            except Exception:
                pass  # Don't let subscriber errors break publishing

    async def get_latest(self, topic: str) -> Any | None:
        """
        Get the latest published content for a topic.
        
        Args:
            topic: The topic name
            
        Returns:
            Latest content or None
        """
        entries = await self.search(
            namespace=f"topic:{topic}",
            limit=1,
        )
        return entries[0].content if entries else None

    async def get_topic_history(
        self,
        topic: str,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """
        Get history for a topic.
        
        Args:
            topic: The topic name
            limit: Maximum entries
            
        Returns:
            List of entries in chronological order
        """
        entries = await self.search(namespace=f"topic:{topic}", limit=limit)
        return sorted(entries, key=lambda e: e.timestamp)

    # ========================================================================
    # Coordination primitives
    # ========================================================================

    async def set_flag(self, flag_name: str, value: bool = True) -> MemoryEntry:
        """
        Set a coordination flag.
        
        Args:
            flag_name: The flag name
            value: Flag value
            
        Returns:
            The created entry
        """
        return await self.add(
            content=value,
            metadata={"type": "flag", "flag_name": flag_name},
            namespace="flags",
        )

    async def get_flag(self, flag_name: str) -> bool:
        """
        Get a coordination flag.
        
        Args:
            flag_name: The flag name
            
        Returns:
            Flag value (False if not set)
        """
        entries = await self.search(
            namespace="flags",
            metadata_filter={"flag_name": flag_name},
            limit=1,
        )
        return bool(entries[0].content) if entries else False

    async def set_shared_state(self, key: str, value: Any) -> MemoryEntry:
        """
        Set a shared state value.
        
        Args:
            key: State key
            value: State value
            
        Returns:
            The created entry
        """
        # Remove old value if exists
        old_entries = await self.search(
            namespace="state",
            metadata_filter={"key": key},
            limit=1,
        )
        for entry in old_entries:
            await self.delete(entry.id)

        return await self.add(
            content=value,
            metadata={"type": "state", "key": key},
            namespace="state",
        )

    async def get_shared_state(self, key: str) -> Any | None:
        """
        Get a shared state value.
        
        Args:
            key: State key
            
        Returns:
            State value or None
        """
        entries = await self.search(
            namespace="state",
            metadata_filter={"key": key},
            limit=1,
        )
        return entries[0].content if entries else None

    async def get_all_state(self) -> dict[str, Any]:
        """Get all shared state as a dictionary."""
        entries = await self.search(
            namespace="state",
            metadata_filter={"type": "state"},
            limit=1000,
        )
        return {e.metadata["key"]: e.content for e in entries if "key" in e.metadata}
