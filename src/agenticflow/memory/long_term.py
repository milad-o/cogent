"""
Long-term memory - persistent cross-thread storage.

Uses LangGraph stores for persistent memory across threads.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from agenticflow.memory.base import (
    Memory,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
)
from agenticflow.core.utils import generate_id, now_utc


class LongTermMemory(Memory[Any]):
    """
    Long-term memory for persistent storage across threads.
    
    Stores information that should persist across conversations:
    - User preferences
    - Learned facts
    - Important decisions
    - Task outcomes
    
    Attributes:
        user_id: User identifier for scoping memories
        persist_path: Optional path for file persistence
        
    Example:
        ```python
        memory = LongTermMemory(user_id="user-123")
        
        # Store a preference
        await memory.add(
            {"preference": "dark_mode", "value": True},
            metadata={"category": "settings"},
        )
        
        # Retrieve preferences
        prefs = await memory.search(metadata_filter={"category": "settings"})
        ```
    """

    def __init__(
        self,
        user_id: str | None = None,
        persist_path: str | Path | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """
        Initialize long-term memory.
        
        Args:
            user_id: User identifier
            persist_path: Path for file persistence
            config: Memory configuration
        """
        cfg = config or MemoryConfig(memory_type=MemoryType.LONG_TERM)
        super().__init__(cfg)
        self.user_id = user_id or "default"
        self.persist_path = Path(persist_path) if persist_path else None
        self._entries: dict[str, dict[str, MemoryEntry]] = defaultdict(dict)
        
        # Load from file if exists
        if self.persist_path and self.persist_path.exists():
            self._load_from_file()

    def _load_from_file(self) -> None:
        """Load memories from file."""
        if not self.persist_path:
            return
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            for ns, entries in data.items():
                for entry_data in entries.values():
                    entry = MemoryEntry.from_dict(entry_data)
                    if not entry.is_expired():
                        self._entries[ns][entry.id] = entry
        except Exception:
            pass  # File doesn't exist or is corrupted

    def _save_to_file(self) -> None:
        """Save memories to file."""
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                ns: {eid: e.to_dict() for eid, e in entries.items()}
                for ns, entries in self._entries.items()
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    async def add(
        self,
        content: Any,
        metadata: dict | None = None,
        namespace: str | None = None,
    ) -> MemoryEntry:
        """Add a memory entry."""
        ns = namespace or self.user_id
        
        # Calculate expiration if TTL is set
        expires_at = None
        if self.config.ttl_seconds:
            expires_at = now_utc() + timedelta(seconds=self.config.ttl_seconds)
        
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
            namespace=ns,
            expires_at=expires_at,
        )
        self._entries[ns][entry.id] = entry

        # Enforce max entries
        if len(self._entries[ns]) > self.config.max_entries:
            oldest = min(
                self._entries[ns].values(),
                key=lambda e: e.timestamp,
            )
            del self._entries[ns][oldest.id]

        self._save_to_file()
        return entry

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry."""
        for ns_entries in self._entries.values():
            if entry_id in ns_entries:
                entry = ns_entries[entry_id]
                if entry.is_expired():
                    del ns_entries[entry_id]
                    return None
                return entry
        return None

    async def search(
        self,
        query: str | None = None,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search entries."""
        ns = namespace or self.user_id
        
        if ns in self._entries:
            entries = list(self._entries[ns].values())
        else:
            # Search all namespaces if not specified
            entries = []
            for ns_entries in self._entries.values():
                entries.extend(ns_entries.values())

        # Filter expired
        entries = [e for e in entries if not e.is_expired()]

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

        # Simple text search in content
        if query:
            scored = []
            query_lower = query.lower()
            for entry in entries:
                content_str = str(entry.content).lower()
                if query_lower in content_str:
                    # Simple relevance: count occurrences
                    score = content_str.count(query_lower)
                    entry.relevance_score = score
                    scored.append(entry)
            entries = sorted(scored, key=lambda e: e.relevance_score or 0, reverse=True)

        # Sort by timestamp (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries[:limit]

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        for ns_entries in self._entries.values():
            if entry_id in ns_entries:
                del ns_entries[entry_id]
                self._save_to_file()
                return True
        return False

    async def clear(self, namespace: str | None = None) -> int:
        """Clear entries."""
        ns = namespace or self.user_id
        
        if ns:
            count = len(self._entries.get(ns, {}))
            self._entries[ns] = {}
        else:
            count = sum(len(e) for e in self._entries.values())
            self._entries.clear()
        
        self._save_to_file()
        return count

    async def remember(
        self,
        key: str,
        value: Any,
        category: str = "general",
    ) -> MemoryEntry:
        """
        Store a key-value fact.
        
        Args:
            key: The fact key
            value: The fact value
            category: Category for organization
            
        Returns:
            The created MemoryEntry
        """
        # Check if key exists and update
        existing = await self.search(
            metadata_filter={"key": key, "category": category},
            limit=1,
        )
        if existing:
            await self.delete(existing[0].id)

        return await self.add(
            content={"key": key, "value": value},
            metadata={"key": key, "category": category, "type": "fact"},
        )

    async def recall(self, key: str, category: str = "general") -> Any | None:
        """
        Recall a stored fact.
        
        Args:
            key: The fact key
            category: The category
            
        Returns:
            The fact value or None
        """
        entries = await self.search(
            metadata_filter={"key": key, "category": category},
            limit=1,
        )
        if entries:
            content = entries[0].content
            if isinstance(content, dict):
                return content.get("value")
            return content
        return None

    async def get_facts(self, category: str | None = None) -> dict[str, Any]:
        """
        Get all facts, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary of key-value facts
        """
        filter_dict = {"type": "fact"}
        if category:
            filter_dict["category"] = category
            
        entries = await self.search(metadata_filter=filter_dict, limit=1000)
        facts = {}
        for entry in entries:
            if isinstance(entry.content, dict):
                key = entry.content.get("key")
                value = entry.content.get("value")
                if key:
                    facts[key] = value
        return facts
