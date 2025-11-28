"""Memory Manager - Unified memory configuration for agents.

This module provides a unified interface for configuring and using memory:
- Short-term: Conversation history (per thread)
- Long-term: User facts and preferences (per user, persists across threads)
- Team: Shared state between agents in multi-agent flows
- Working: Scratchpad for current execution

The MemoryManager handles:
1. Memory configuration and setup
2. Context injection into prompts
3. Memory tools for agents to save/recall information
4. Proper scoping (thread_id, user_id, team_id)

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.memory import MemoryManager, SQLiteBackend
    
    # Simple: just enable memory
    agent = Agent(
        name="Assistant",
        model="gpt-4o",
        memory=True,  # Uses defaults
    )
    
    # Configured: customize everything
    agent = Agent(
        name="Assistant",
        model="gpt-4o",
        memory=MemoryManager(
            short_term=True,  # Conversation history
            long_term=True,   # User facts
            backend=SQLiteBackend("memory.db"),  # Persistent
            max_messages=100,
            auto_summarize=True,
        ),
    )
    
    # Chat with memory
    response = await agent.chat(
        "Remember I prefer dark mode",
        thread_id="conv-1",
        user_id="user-123",
    )
    
    # Different thread, same user - remembers preferences
    response = await agent.chat(
        "What are my preferences?",
        thread_id="conv-2",
        user_id="user-123",
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.memory.base import MemoryBackend, MemoryScope
from agenticflow.memory.backends.inmemory import (
    InMemoryBackend,
    InMemoryConversationBackend,
)
from agenticflow.memory.conversation import ConversationMemory
from agenticflow.memory.team import TeamMemory
from agenticflow.memory.user import UserMemory
from agenticflow.memory.working import WorkingMemory

if TYPE_CHECKING:
    from agenticflow.core.messages import Message
    from agenticflow.memory.base import ConversationBackend


@dataclass
class MemoryConfig:
    """Configuration for agent memory.

    Controls what types of memory are enabled and how they behave.

    Attributes:
        short_term: Enable conversation memory (per thread).
        long_term: Enable user memory (per user, persists across threads).
        team: Enable team memory (shared between agents).
        working: Enable working memory (per execution scratchpad).
        max_messages: Max messages in conversation history.
        auto_summarize: Summarize old messages (not yet implemented).
        inject_context: Inject memory context into prompts.
        provide_tools: Give agent tools to save/recall memories.
        backend: Storage backend (InMemory, SQLite, etc.).
    """

    # Which memory types to enable
    short_term: bool = True
    long_term: bool = False
    team: bool = False
    working: bool = False

    # Short-term (conversation) settings
    max_messages: int = 50
    auto_summarize: bool = False  # Future: summarize old messages

    # Behavior settings
    inject_context: bool = True  # Inject memory into system prompt
    provide_tools: bool = True  # Give agent memory tools

    # Backend
    backend: MemoryBackend | None = None
    conversation_backend: ConversationBackend | None = None

    def __post_init__(self) -> None:
        """Set up default backends if not provided."""
        if self.backend is None:
            self.backend = InMemoryBackend()
        if self.conversation_backend is None:
            self.conversation_backend = InMemoryConversationBackend()


class MemoryManager:
    """Unified memory manager for agents.

    Combines short-term, long-term, team, and working memory with:
    - Proper configuration and defaults
    - Context injection for prompts
    - Memory tools for agents
    - Clean API for chat/run methods

    Example:
        ```python
        # Create with defaults (conversation memory only)
        manager = MemoryManager()

        # Create with all features
        manager = MemoryManager(
            short_term=True,
            long_term=True,
            team=True,
            backend=SQLiteBackend("memory.db"),
        )

        # Or from config
        config = MemoryConfig(
            short_term=True,
            long_term=True,
            max_messages=100,
        )
        manager = MemoryManager.from_config(config)
        ```
    """

    def __init__(
        self,
        *,
        short_term: bool = True,
        long_term: bool = False,
        team: bool = False,
        working: bool = False,
        max_messages: int = 50,
        auto_summarize: bool = False,
        inject_context: bool = True,
        provide_tools: bool = True,
        backend: MemoryBackend | None = None,
        conversation_backend: ConversationBackend | None = None,
    ) -> None:
        """Initialize memory manager.

        Args:
            short_term: Enable conversation memory.
            long_term: Enable user fact memory.
            team: Enable team shared memory.
            working: Enable working scratchpad.
            max_messages: Max messages in conversation.
            auto_summarize: Auto-summarize old messages.
            inject_context: Inject memory into prompts.
            provide_tools: Provide memory tools to agent.
            backend: Storage backend for key-value memory.
            conversation_backend: Storage backend for conversations.
        """
        self._config = MemoryConfig(
            short_term=short_term,
            long_term=long_term,
            team=team,
            working=working,
            max_messages=max_messages,
            auto_summarize=auto_summarize,
            inject_context=inject_context,
            provide_tools=provide_tools,
            backend=backend,
            conversation_backend=conversation_backend,
        )

        # Initialize memory instances
        self._conversation: ConversationMemory | None = None
        self._user: UserMemory | None = None
        self._team: TeamMemory | None = None
        self._working: WorkingMemory | None = None

        if self._config.short_term:
            self._conversation = ConversationMemory(
                backend=self._config.conversation_backend,
                max_messages=self._config.max_messages,
            )

        if self._config.long_term:
            self._user = UserMemory(backend=self._config.backend)

        if self._config.team:
            self._team = TeamMemory(backend=self._config.backend)

        if self._config.working:
            self._working = WorkingMemory(backend=self._config.backend)

    @classmethod
    def from_config(cls, config: MemoryConfig) -> MemoryManager:
        """Create MemoryManager from a MemoryConfig."""
        return cls(
            short_term=config.short_term,
            long_term=config.long_term,
            team=config.team,
            working=config.working,
            max_messages=config.max_messages,
            auto_summarize=config.auto_summarize,
            inject_context=config.inject_context,
            provide_tools=config.provide_tools,
            backend=config.backend,
            conversation_backend=config.conversation_backend,
        )

    @classmethod
    def default(cls) -> MemoryManager:
        """Create a default memory manager (conversation only)."""
        return cls(short_term=True)

    @classmethod
    def full(cls, backend: MemoryBackend | None = None) -> MemoryManager:
        """Create a full-featured memory manager."""
        return cls(
            short_term=True,
            long_term=True,
            team=True,
            working=True,
            backend=backend,
        )

    @property
    def config(self) -> MemoryConfig:
        """Get the memory configuration."""
        return self._config

    @property
    def conversation(self) -> ConversationMemory | None:
        """Get conversation memory (short-term)."""
        return self._conversation

    @property
    def user(self) -> UserMemory | None:
        """Get user memory (long-term)."""
        return self._user

    @property
    def team(self) -> TeamMemory | None:
        """Get team memory (shared)."""
        return self._team

    @property
    def working(self) -> WorkingMemory | None:
        """Get working memory (scratchpad)."""
        return self._working

    @property
    def has_short_term(self) -> bool:
        """Whether short-term memory is enabled."""
        return self._conversation is not None

    @property
    def has_long_term(self) -> bool:
        """Whether long-term memory is enabled."""
        return self._user is not None

    @property
    def has_team(self) -> bool:
        """Whether team memory is enabled."""
        return self._team is not None

    @property
    def has_working(self) -> bool:
        """Whether working memory is enabled."""
        return self._working is not None

    # ========================================
    # Conversation Memory (Short-term)
    # ========================================

    async def add_message(
        self,
        thread_id: str,
        message: Message,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to conversation history.

        Args:
            thread_id: Conversation thread ID.
            message: Message to add.
            metadata: Optional metadata.
        """
        if self._conversation:
            await self._conversation.add_message(thread_id, message, metadata)

    async def add_messages(
        self,
        thread_id: str,
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add multiple messages to conversation history.

        Args:
            thread_id: Conversation thread ID.
            messages: Messages to add.
            metadata: Optional metadata.
        """
        for message in messages:
            await self.add_message(thread_id, message, metadata)

    async def get_messages(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Get messages from conversation history.

        Args:
            thread_id: Conversation thread ID.
            limit: Max messages to return.

        Returns:
            List of messages.
        """
        if self._conversation:
            return await self._conversation.get_messages(thread_id, limit)
        return []

    async def get_recent_messages(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[Message]:
        """Get recent messages from conversation.

        Args:
            thread_id: Conversation thread ID.
            limit: Number of recent messages.

        Returns:
            List of recent messages.
        """
        if self._conversation:
            return await self._conversation.get_recent_messages(thread_id, limit)
        return []

    async def clear_conversation(self, thread_id: str) -> None:
        """Clear conversation history for a thread.

        Args:
            thread_id: Conversation thread ID.
        """
        if self._conversation:
            await self._conversation.clear(thread_id=thread_id)

    # ========================================
    # User Memory (Long-term)
    # ========================================

    async def remember(
        self,
        key: str,
        value: Any,
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Remember a fact about a user.

        Args:
            key: Fact key (e.g., "preference", "name").
            value: Fact value.
            user_id: User identifier.
            metadata: Optional metadata.
        """
        if self._user:
            await self._user.save(key, value, user_id=user_id, metadata=metadata)

    async def recall(
        self,
        key: str,
        *,
        user_id: str,
    ) -> Any | None:
        """Recall a fact about a user.

        Args:
            key: Fact key.
            user_id: User identifier.

        Returns:
            The fact value, or None if not found.
        """
        if self._user:
            return await self._user.load(key, user_id=user_id)
        return None

    async def forget(
        self,
        key: str,
        *,
        user_id: str,
    ) -> bool:
        """Forget a fact about a user.

        Args:
            key: Fact key.
            user_id: User identifier.

        Returns:
            True if forgotten, False if not found.
        """
        if self._user:
            return await self._user.delete(key, user_id=user_id)
        return False

    async def get_user_facts(self, user_id: str) -> dict[str, Any]:
        """Get all facts about a user.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary of facts.
        """
        if self._user:
            return await self._user.get_all_facts(user_id)
        return {}

    async def search_user_memory(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 10,
    ) -> list[Any]:
        """Search user memory for matching facts.

        Args:
            query: Search query.
            user_id: User identifier.
            limit: Max results.

        Returns:
            List of matching memory entries.
        """
        if self._user:
            return await self._user.search(query, user_id=user_id, limit=limit)
        return []

    # ========================================
    # Context Generation for Prompts
    # ========================================

    async def get_context_for_prompt(
        self,
        *,
        thread_id: str | None = None,
        user_id: str | None = None,
        include_conversation: bool = True,
        include_user_facts: bool = True,
        max_facts: int = 10,
    ) -> str:
        """Generate memory context to inject into prompts.

        This creates a formatted string with relevant memory context
        that can be added to the system prompt.

        Args:
            thread_id: Conversation thread (for conversation context).
            user_id: User ID (for user facts).
            include_conversation: Include conversation summary.
            include_user_facts: Include user facts.
            max_facts: Max user facts to include.

        Returns:
            Formatted context string for prompt injection.
        """
        sections: list[str] = []

        # Add user facts if available
        if include_user_facts and user_id and self._user:
            facts = await self.get_user_facts(user_id)
            if facts:
                facts_str = "\n".join(f"- {k}: {v}" for k, v in list(facts.items())[:max_facts])
                sections.append(f"## Known facts about this user:\n{facts_str}")

        # Add conversation summary if available
        if include_conversation and thread_id and self._conversation:
            count = await self._conversation.get_message_count(thread_id)
            if count > 0:
                sections.append(
                    f"## Conversation context:\n"
                    f"This conversation has {count} previous messages. "
                    f"You have access to the full history."
                )

        if not sections:
            return ""

        return "\n\n".join(sections)

    # ========================================
    # Memory Tools for Agent
    # ========================================

    def get_memory_tools(self, user_id: str | None = None) -> list[Any]:
        """Get memory tools for the agent.

        Returns tools that allow the agent to:
        - Save facts to long-term memory
        - Recall facts from long-term memory
        - Search memory

        Args:
            user_id: Default user ID for memory operations.

        Returns:
            List of tool functions.
        """
        from agenticflow.memory.tools import create_memory_tools

        return create_memory_tools(self, default_user_id=user_id)

    # ========================================
    # State Management
    # ========================================

    async def clear_all(
        self,
        *,
        thread_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Clear all memory.

        Args:
            thread_id: Clear conversation for this thread.
            user_id: Clear facts for this user.
        """
        if thread_id and self._conversation:
            await self._conversation.clear(thread_id=thread_id)

        if user_id and self._user:
            await self._user.clear(user_id=user_id)

        if self._working:
            await self._working.clear()


# ============================================================
# Factory Functions
# ============================================================


def create_memory(
    memory_input: bool | MemoryManager | MemoryConfig | dict[str, Any] | None,
) -> MemoryManager | None:
    """Create a MemoryManager from various input types.

    This is the main entry point for memory configuration.

    Args:
        memory_input: Memory configuration:
            - None/False: No memory
            - True: Default memory (conversation only)
            - MemoryManager: Use as-is
            - MemoryConfig: Create from config
            - dict: Create from kwargs

    Returns:
        MemoryManager instance or None.

    Example:
        ```python
        # These all work:
        memory = create_memory(True)
        memory = create_memory({"short_term": True, "long_term": True})
        memory = create_memory(MemoryConfig(max_messages=100))
        memory = create_memory(MemoryManager(...))
        ```
    """
    if memory_input is None or memory_input is False:
        return None

    if memory_input is True:
        return MemoryManager.default()

    if isinstance(memory_input, MemoryManager):
        return memory_input

    if isinstance(memory_input, MemoryConfig):
        return MemoryManager.from_config(memory_input)

    if isinstance(memory_input, dict):
        return MemoryManager(**memory_input)

    raise TypeError(
        f"Invalid memory type: {type(memory_input)}. "
        "Expected bool, MemoryManager, MemoryConfig, or dict."
    )
