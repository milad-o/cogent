"""
MemoryManager - unified interface for all memory types.

Provides a single entry point for managing agent memory with
automatic routing to appropriate memory systems.
"""

from __future__ import annotations

from typing import Any

from agenticflow.memory.base import Memory, MemoryConfig, MemoryEntry, MemoryType
from agenticflow.memory.short_term import ShortTermMemory
from agenticflow.memory.long_term import LongTermMemory
from agenticflow.memory.shared import SharedMemory
from agenticflow.core.utils import generate_id


class MemoryManager:
    """
    Unified manager for all memory types.
    
    Provides a single interface for agents to interact with
    different memory systems based on context and need.
    
    Attributes:
        agent_id: The owning agent's ID
        user_id: User ID for long-term memory scoping
        workspace_id: Workspace ID for shared memory
        
    Example:
        ```python
        memory = MemoryManager(
            agent_id="agent-123",
            user_id="user-456",
            workspace_id="project-alpha",
        )
        
        # Short-term: conversation context
        await memory.remember_conversation(message)
        history = await memory.get_conversation_history()
        
        # Long-term: persistent facts
        await memory.remember_fact("user_name", "Alice")
        name = await memory.recall_fact("user_name")
        
        # Shared: multi-agent coordination
        await memory.share("analysis_result", result)
        data = await memory.get_shared("analysis_result")
        ```
    """

    def __init__(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        thread_id: str | None = None,
        persist_path: str | None = None,
    ) -> None:
        """
        Initialize the memory manager.
        
        Args:
            agent_id: ID of the owning agent
            user_id: User ID for long-term memory
            workspace_id: Workspace ID for shared memory
            thread_id: Initial thread ID for short-term memory
            persist_path: Path for long-term memory persistence
        """
        self.agent_id = agent_id or generate_id()
        self.user_id = user_id or "default"
        self.workspace_id = workspace_id or "default"

        # Initialize memory systems
        self.short_term = ShortTermMemory(
            thread_id=thread_id or generate_id(),
            config=MemoryConfig(
                memory_type=MemoryType.SHORT_TERM,
                max_entries=500,
            ),
        )

        self.long_term = LongTermMemory(
            user_id=user_id,
            persist_path=persist_path,
            config=MemoryConfig(
                memory_type=MemoryType.LONG_TERM,
                max_entries=10000,
            ),
        )

        self.shared = SharedMemory(
            workspace_id=workspace_id,
            config=MemoryConfig(
                memory_type=MemoryType.SHARED,
                max_entries=5000,
            ),
        )

    # ========================================================================
    # Conversation Memory (Short-term)
    # ========================================================================

    async def remember_conversation(
        self,
        content: Any,
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """
        Add to conversation memory.
        
        Args:
            content: Message or content to remember
            metadata: Optional metadata
            
        Returns:
            The created MemoryEntry
        """
        meta = {"agent_id": self.agent_id, **(metadata or {})}
        return await self.short_term.add(content, metadata=meta)

    async def get_conversation_history(self, limit: int = 50) -> list[MemoryEntry]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum entries
            
        Returns:
            List of memory entries
        """
        return await self.short_term.search(limit=limit)

    def switch_thread(self, thread_id: str) -> None:
        """Switch conversation thread."""
        self.short_term.switch_thread(thread_id)

    @property
    def thread_id(self) -> str:
        """Current thread ID."""
        return self.short_term.thread_id

    # ========================================================================
    # Persistent Memory (Long-term)
    # ========================================================================

    async def remember_fact(
        self,
        key: str,
        value: Any,
        category: str = "general",
    ) -> MemoryEntry:
        """
        Store a persistent fact.
        
        Args:
            key: Fact key
            value: Fact value
            category: Category for organization
            
        Returns:
            The created MemoryEntry
        """
        return await self.long_term.remember(key, value, category)

    async def recall_fact(
        self,
        key: str,
        category: str = "general",
    ) -> Any | None:
        """
        Recall a stored fact.
        
        Args:
            key: Fact key
            category: Category
            
        Returns:
            Fact value or None
        """
        return await self.long_term.recall(key, category)

    async def get_all_facts(self, category: str | None = None) -> dict[str, Any]:
        """Get all stored facts."""
        return await self.long_term.get_facts(category)

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Search long-term memories.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching entries
        """
        return await self.long_term.search(query=query, limit=limit)

    # ========================================================================
    # Shared Memory (Multi-agent)
    # ========================================================================

    async def share(
        self,
        topic: str,
        content: Any,
    ) -> MemoryEntry:
        """
        Share content with other agents.
        
        Args:
            topic: Topic/channel name
            content: Content to share
            
        Returns:
            The created MemoryEntry
        """
        return await self.shared.publish(
            topic=topic,
            content=content,
            publisher_id=self.agent_id,
        )

    async def get_shared(self, topic: str) -> Any | None:
        """
        Get latest shared content.
        
        Args:
            topic: Topic name
            
        Returns:
            Latest content or None
        """
        return await self.shared.get_latest(topic)

    async def subscribe_to(
        self,
        topic: str,
        callback: Any,
    ) -> None:
        """
        Subscribe to shared topic updates.
        
        Args:
            topic: Topic to subscribe to
            callback: Async callback function
        """
        await self.shared.subscribe(
            topic=topic,
            callback=callback,
            subscriber_id=self.agent_id,
        )

    async def set_shared_state(self, key: str, value: Any) -> MemoryEntry:
        """Set shared state value."""
        return await self.shared.set_shared_state(key, value)

    async def get_shared_state(self, key: str) -> Any | None:
        """Get shared state value."""
        return await self.shared.get_shared_state(key)

    # ========================================================================
    # Unified operations
    # ========================================================================

    async def remember(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        **kwargs: Any,
    ) -> MemoryEntry:
        """
        Universal remember function.
        
        Args:
            content: Content to remember
            memory_type: Type of memory to use
            **kwargs: Additional arguments for specific memory type
            
        Returns:
            The created MemoryEntry
        """
        if memory_type == MemoryType.SHORT_TERM:
            return await self.short_term.add(content, **kwargs)
        elif memory_type == MemoryType.LONG_TERM:
            return await self.long_term.add(content, **kwargs)
        elif memory_type == MemoryType.SHARED:
            return await self.shared.add(content, **kwargs)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    async def recall(
        self,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Universal recall function.
        
        Args:
            query: Search query
            memory_type: Type of memory to search (None = all)
            limit: Maximum results
            
        Returns:
            List of matching entries
        """
        results = []

        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            results.extend(await self.short_term.search(query=query, limit=limit))

        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            results.extend(await self.long_term.search(query=query, limit=limit))

        if memory_type is None or memory_type == MemoryType.SHARED:
            results.extend(await self.shared.search(query=query, limit=limit))

        # Sort by timestamp and limit
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    async def clear_all(self) -> dict[str, int]:
        """
        Clear all memory.
        
        Returns:
            Dict with counts cleared per memory type
        """
        return {
            "short_term": await self.short_term.clear(),
            "long_term": await self.long_term.clear(),
            "shared": await self.shared.clear(),
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dict with memory stats
        """
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "thread_id": self.thread_id,
            "short_term_threads": len(self.short_term.get_thread_ids()),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "thread_id": self.thread_id,
        }
