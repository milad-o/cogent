"""
Short-term memory - thread-scoped conversation history.

Uses LangGraph's InMemorySaver for checkpointing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.messages import BaseMessage

from agenticflow.memory.base import (
    Memory,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
)
from agenticflow.core.utils import generate_id, now_utc


class ShortTermMemory(Memory[Any]):
    """
    Short-term memory for conversation history.
    
    Thread-scoped memory that maintains conversation context within
    a single session/thread. Integrates with LangGraph checkpointers.
    
    Attributes:
        thread_id: The current thread ID
        
    Example:
        ```python
        memory = ShortTermMemory(thread_id="conversation-123")
        
        # Add messages
        await memory.add(HumanMessage(content="Hello"))
        await memory.add(AIMessage(content="Hi there!"))
        
        # Get recent history
        history = await memory.get_messages(limit=10)
        ```
    """

    def __init__(
        self,
        thread_id: str | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """
        Initialize short-term memory.
        
        Args:
            thread_id: Thread/conversation ID
            config: Memory configuration
        """
        cfg = config or MemoryConfig(memory_type=MemoryType.SHORT_TERM)
        super().__init__(cfg)
        self.thread_id = thread_id or generate_id()
        self._entries: dict[str, dict[str, MemoryEntry]] = defaultdict(dict)

    async def add(
        self,
        content: Any,
        metadata: dict | None = None,
        namespace: str | None = None,
    ) -> MemoryEntry:
        """Add a memory entry."""
        ns = namespace or self.thread_id
        entry = MemoryEntry(
            content=content,
            metadata=metadata or {},
            namespace=ns,
        )
        self._entries[ns][entry.id] = entry

        # Enforce max entries
        if len(self._entries[ns]) > self.config.max_entries:
            # Remove oldest
            oldest = min(
                self._entries[ns].values(),
                key=lambda e: e.timestamp,
            )
            del self._entries[ns][oldest.id]

        return entry

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry."""
        for ns_entries in self._entries.values():
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
        """Search entries."""
        ns = namespace or self.thread_id
        
        if ns in self._entries:
            entries = list(self._entries[ns].values())
        else:
            entries = []

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

        # Sort by timestamp (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries[:limit]

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        for ns_entries in self._entries.values():
            if entry_id in ns_entries:
                del ns_entries[entry_id]
                return True
        return False

    async def clear(self, namespace: str | None = None) -> int:
        """Clear entries."""
        ns = namespace or self.thread_id
        
        if ns:
            count = len(self._entries.get(ns, {}))
            self._entries[ns] = {}
            return count
        else:
            count = sum(len(e) for e in self._entries.values())
            self._entries.clear()
            return count

    async def add_message(self, message: BaseMessage) -> MemoryEntry:
        """
        Add a LangChain message to memory.
        
        Args:
            message: The message to add
            
        Returns:
            The created MemoryEntry
        """
        return await self.add(
            content=message,
            metadata={
                "type": "message",
                "role": getattr(message, "type", "unknown"),
            },
        )

    async def get_messages(self, limit: int = 50) -> list[BaseMessage]:
        """
        Get recent messages.
        
        Args:
            limit: Maximum messages to return
            
        Returns:
            List of messages in chronological order
        """
        entries = await self.search(
            metadata_filter={"type": "message"},
            limit=limit,
        )
        # Return in chronological order
        entries.sort(key=lambda e: e.timestamp)
        return [e.content for e in entries if isinstance(e.content, BaseMessage)]

    def switch_thread(self, thread_id: str) -> None:
        """
        Switch to a different thread.
        
        Args:
            thread_id: The new thread ID
        """
        self.thread_id = thread_id

    def get_thread_ids(self) -> list[str]:
        """Get all thread IDs with entries."""
        return list(self._entries.keys())

    def to_langchain_messages(self) -> list[BaseMessage]:
        """Get all messages for current thread as LangChain messages."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.get_messages())
