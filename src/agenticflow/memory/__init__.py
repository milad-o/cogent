"""Memory system for agenticflow.

Provides memory capabilities for agents, teams, and flows:
- ConversationMemory: Per-thread chat history with sliding window
- UserMemory: Per-user long-term facts and preferences
- TeamMemory: Shared state between agents in a topology
- WorkingMemory: Scratchpad for current execution
- MemoryManager: Unified configuration for all memory types

Backends:
- InMemoryBackend: Fast, zero-config (default)
- SQLiteBackend: Local file persistence (coming soon)

Example:
    ```python
    # Simple: just enable memory
    agent = Agent(
        name="Assistant",
        model="gpt-4o",
        memory=True,  # Uses ConversationMemory with defaults
    )

    # Configured: enable all features
    from agenticflow.memory import MemoryManager
    
    agent = Agent(
        name="Assistant",
        model="gpt-4o",
        memory=MemoryManager(
            short_term=True,   # Conversation history
            long_term=True,    # User facts (persists across threads)
            max_messages=100,
        ),
    )
    
    # Chat with memory
    response = await agent.chat(
        "Remember I prefer dark mode",
        thread_id="conv-1",
        user_id="user-123",
    )

    # User memory for facts
    from agenticflow.memory import UserMemory

    memory = UserMemory()
    await memory.save("preference", "dark mode", user_id="user-123")

    # Team memory for multi-agent
    from agenticflow.memory import TeamMemory

    team_memory = TeamMemory(team_id="research-team")
    await team_memory.save("findings", researcher_output)

    # Working memory for execution
    from agenticflow.memory import WorkingMemory

    async with WorkingMemory() as working:
        await working.save("draft", initial_draft)
        # ... execution ...
        # Auto-cleared on exit
    ```
"""

from agenticflow.memory.base import (
    BaseMemory,
    ConversationBackend,
    ConversationEntry,
    MemoryBackend,
    MemoryConfig as BaseMemoryConfig,
    MemoryEntry,
    MemoryScope,
    create_default_memory,
)
from agenticflow.memory.backends import InMemoryBackend, InMemoryConversationBackend
from agenticflow.memory.conversation import ConversationMemory
from agenticflow.memory.manager import (
    MemoryConfig,
    MemoryManager,
    create_memory,
)
from agenticflow.memory.team import TeamMemory
from agenticflow.memory.tools import (
    create_memory_tools,
    format_memory_context,
    get_memory_prompt_addition,
)
from agenticflow.memory.user import UserMemory
from agenticflow.memory.working import WorkingMemory

__all__ = [
    # Main entry points
    "MemoryManager",
    "MemoryConfig",
    "create_memory",
    # Memory types
    "ConversationMemory",
    "UserMemory",
    "TeamMemory",
    "WorkingMemory",
    # Base classes and protocols
    "BaseMemory",
    "MemoryBackend",
    "ConversationBackend",
    "BaseMemoryConfig",
    "MemoryScope",
    "MemoryEntry",
    "ConversationEntry",
    # Backends
    "InMemoryBackend",
    "InMemoryConversationBackend",
    # Tools and prompts
    "create_memory_tools",
    "get_memory_prompt_addition",
    "format_memory_context",
    # Utilities
    "create_default_memory",
]
