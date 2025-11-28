"""Conversation memory for chat history.

Stores conversation messages per thread with sliding window support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.memory.backends.inmemory import InMemoryConversationBackend
from agenticflow.memory.base import BaseMemory, ConversationBackend, MemoryConfig

if TYPE_CHECKING:
    from agenticflow.core.messages import Message


class ConversationMemory(BaseMemory):
    """Memory for conversation history with sliding window.

    Stores messages per thread_id with configurable max messages.
    Oldest messages are dropped when the limit is reached.

    Example:
        memory = ConversationMemory(max_messages=50)

        # Add messages
        await memory.add_message("thread-1", user_message)
        await memory.add_message("thread-1", assistant_message)

        # Get recent history
        messages = await memory.get_messages("thread-1")

        # Clear a thread
        await memory.clear(thread_id="thread-1")
    """

    def __init__(
        self,
        backend: ConversationBackend | None = None,
        max_messages: int = 50,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize conversation memory.

        Args:
            backend: Storage backend. If None, uses InMemoryConversationBackend.
            max_messages: Maximum messages to keep per thread.
            config: Additional configuration.
        """
        # Use conversation-specific backend
        self._conversation_backend = backend or InMemoryConversationBackend()
        self._max_messages = max_messages

        # Initialize parent with a dummy backend (we use _conversation_backend)
        super().__init__(config=config)

    @property
    def max_messages(self) -> int:
        """Maximum messages kept per thread."""
        return self._max_messages

    async def add_message(
        self,
        thread_id: str,
        message: Message,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to the conversation history.

        If the thread exceeds max_messages, oldest messages are trimmed.

        Args:
            thread_id: Conversation thread identifier.
            message: Message to add.
            metadata: Optional metadata.
        """
        await self._conversation_backend.add_message(thread_id, message, metadata)

        # Trim if over limit (keep most recent)
        count = await self._conversation_backend.get_message_count(thread_id)
        if count > self._max_messages:
            # Get all messages and keep only the most recent
            all_messages = await self._conversation_backend.get_messages(thread_id)
            await self._conversation_backend.clear_thread(thread_id)
            for msg in all_messages[-self._max_messages :]:
                await self._conversation_backend.add_message(thread_id, msg)

    async def get_messages(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Get messages from a thread.

        Args:
            thread_id: Conversation thread identifier.
            limit: Optional limit on messages returned.

        Returns:
            List of messages, oldest first.
        """
        if limit is None:
            return await self._conversation_backend.get_messages(thread_id)
        return await self._conversation_backend.get_messages(thread_id, limit=limit)

    async def get_recent_messages(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[Message]:
        """Get the most recent messages from a thread.

        Args:
            thread_id: Conversation thread identifier.
            limit: Number of recent messages to return.

        Returns:
            List of recent messages, oldest first.
        """
        if hasattr(self._conversation_backend, "get_recent_messages"):
            return await self._conversation_backend.get_recent_messages(
                thread_id, limit
            )
        # Fallback for backends without get_recent_messages
        all_messages = await self._conversation_backend.get_messages(thread_id)
        return all_messages[-limit:] if limit else all_messages

    async def get_message_count(self, thread_id: str) -> int:
        """Get the number of messages in a thread.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Number of messages.
        """
        return await self._conversation_backend.get_message_count(thread_id)

    async def list_threads(self) -> list[str]:
        """List all thread IDs with messages.

        Returns:
            List of thread IDs.
        """
        return await self._conversation_backend.list_threads()

    # BaseMemory interface implementation

    async def save(self, key: str, value: Any, **kwargs: Any) -> None:
        """Save a message (key is thread_id, value is message).

        This method exists for BaseMemory compatibility.
        Prefer using add_message() directly.
        """
        thread_id = key
        if hasattr(value, "content"):  # It's a Message
            await self.add_message(thread_id, value, kwargs.get("metadata"))
        else:
            raise TypeError(f"Expected Message, got {type(value)}")

    async def load(self, key: str, **kwargs: Any) -> list[Message]:
        """Load messages for a thread (key is thread_id).

        This method exists for BaseMemory compatibility.
        Prefer using get_messages() directly.
        """
        limit = kwargs.get("limit")
        return await self.get_messages(key, limit=limit)

    async def clear(self, **kwargs: Any) -> None:
        """Clear a thread or all threads.

        Args:
            thread_id: If provided, clear only this thread.
                      If None, clear all threads.
        """
        thread_id = kwargs.get("thread_id")
        if thread_id:
            await self._conversation_backend.clear_thread(thread_id)
        elif hasattr(self._conversation_backend, "clear_all"):
            await self._conversation_backend.clear_all()
        else:
            # Fallback: clear each thread
            threads = await self._conversation_backend.list_threads()
            for tid in threads:
                await self._conversation_backend.clear_thread(tid)
