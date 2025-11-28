"""Memory tools for agents.

Provides tools that allow agents to manage long-term memory:
- remember: Save important facts
- recall: Get a specific fact  
- forget: Remove a fact
- search_memories: Find relevant memories

These tools are automatically added to agents when long_term memory is enabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.tools.base import BaseTool

if TYPE_CHECKING:
    from agenticflow.memory.manager import MemoryManager


def _create_remember_tool(manager: MemoryManager, default_user_id: str | None) -> BaseTool:
    """Create a remember tool bound to a manager and user.
    
    Args:
        manager: The MemoryManager instance.
        default_user_id: Default user ID for operations.
    
    Returns:
        BaseTool for remembering facts.
    """
    async def remember(key: str, value: str) -> str:
        """Save an important fact to long-term memory.
        
        Args:
            key: Short descriptive key (e.g., "name", "preferred_language", "project")
            value: The fact to remember
        """
        if not default_user_id:
            return "Cannot save: no user_id provided."
        
        await manager.remember(key, value, user_id=default_user_id)
        return f"Remembered: {key}"
    
    return BaseTool(
        name="remember",
        description="""Save an important fact to long-term memory.

Use this to remember:
- User preferences (language, style, formatting)
- User information (name, role, project context)
- Important corrections or clarifications
- Key decisions or agreements

Do NOT save:
- Temporary task details
- Information already in the conversation
- Obvious or trivial facts

Args:
    key: Short descriptive key (e.g., "name", "preferred_language", "project")
    value: The fact to remember""",
        func=remember,
        args_schema={
            "key": {"type": "string", "description": "Short descriptive key for the fact"},
            "value": {"type": "string", "description": "The fact to remember"},
        },
    )


def _create_recall_tool(manager: MemoryManager, default_user_id: str | None) -> BaseTool:
    """Create a recall tool bound to a manager and user.
    
    Args:
        manager: The MemoryManager instance.
        default_user_id: Default user ID for operations.
    
    Returns:
        BaseTool for recalling facts.
    """
    async def recall(key: str) -> str:
        """Recall a specific fact from long-term memory.
        
        Args:
            key: The key of the fact to recall
        """
        if not default_user_id:
            return "Cannot recall: no user_id provided."
        
        value = await manager.recall(key, user_id=default_user_id)
        if value is None:
            return f"No memory found for: {key}"
        return str(value)
    
    return BaseTool(
        name="recall",
        description="""Recall a specific fact from long-term memory.

Use this to retrieve a previously saved fact by its key.

Args:
    key: The key of the fact to recall

Returns:
    The saved value, or indication if not found""",
        func=recall,
        args_schema={
            "key": {"type": "string", "description": "The key of the fact to recall"},
        },
    )


def _create_forget_tool(manager: MemoryManager, default_user_id: str | None) -> BaseTool:
    """Create a forget tool bound to a manager and user.
    
    Args:
        manager: The MemoryManager instance.
        default_user_id: Default user ID for operations.
    
    Returns:
        BaseTool for forgetting facts.
    """
    async def forget(key: str) -> str:
        """Remove a fact from long-term memory.
        
        Args:
            key: The key of the fact to forget
        """
        if not default_user_id:
            return "Cannot forget: no user_id provided."
        
        success = await manager.forget(key, user_id=default_user_id)
        if success:
            return f"Forgot: {key}"
        return f"No memory found for: {key}"
    
    return BaseTool(
        name="forget",
        description="""Remove a fact from long-term memory.

Only use this when:
- User explicitly asks to forget something
- Information is confirmed to be incorrect
- User wants to clear their data

Args:
    key: The key of the fact to forget""",
        func=forget,
        args_schema={
            "key": {"type": "string", "description": "The key of the fact to forget"},
        },
    )


def _create_search_memories_tool(manager: MemoryManager, default_user_id: str | None) -> BaseTool:
    """Create a search_memories tool bound to a manager and user.
    
    Args:
        manager: The MemoryManager instance.
        default_user_id: Default user ID for operations.
    
    Returns:
        BaseTool for searching memories.
    """
    async def search_memories(query: str) -> str:
        """Search long-term memory for relevant facts.
        
        Args:
            query: Search term (matches keys and values)
        """
        if not default_user_id:
            return "Cannot search: no user_id provided."
        
        results = await manager.search_user_memory(query, user_id=default_user_id, limit=5)
        if not results:
            return f"No memories found for: {query}"
        
        lines = [f"- {entry.key}: {entry.value}" for entry in results]
        return "Found:\n" + "\n".join(lines)
    
    return BaseTool(
        name="search_memories",
        description="""Search long-term memory for relevant facts.

Use this when you need to find information but don't know the exact key.

Args:
    query: Search term (matches keys and values)

Returns:
    Matching memories or indication if none found""",
        func=search_memories,
        args_schema={
            "query": {"type": "string", "description": "Search term to find relevant memories"},
        },
    )


def create_memory_tools(
    manager: MemoryManager,
    user_id: str | None = None,
) -> list[BaseTool]:
    """Create memory tools bound to a MemoryManager.

    Only creates tools if long-term memory is enabled.

    Args:
        manager: The MemoryManager instance.
        user_id: Default user ID for operations.

    Returns:
        List of memory tools (empty if long_term not enabled).
    """
    if not manager.has_long_term:
        return []

    return [
        _create_remember_tool(manager, user_id),
        _create_recall_tool(manager, user_id),
        _create_forget_tool(manager, user_id),
        _create_search_memories_tool(manager, user_id),
    ]


# ============================================================
# Memory Prompt Templates
# ============================================================

MEMORY_SYSTEM_PROMPT = """
## Long-Term Memory

You can remember important facts about users across conversations:
- `remember(key, value)` - Save a fact
- `recall(key)` - Get a specific fact  
- `forget(key)` - Remove a fact (only when asked)
- `search_memories(query)` - Find relevant memories

**When to remember:**
- User shares preferences: "I prefer Python" → remember(key="preferred_language", value="Python")
- User introduces themselves: "I'm Alice" → remember(key="name", value="Alice")
- Important context: "I'm building a healthcare app" → remember(key="current_project", value="healthcare app")
- Corrections: "Actually it's spelled Milad" → remember(key="name", value="Milad")

**When NOT to remember:**
- Temporary tasks or questions
- Information already in current conversation
- Obvious facts that don't need persistence
"""


def get_memory_prompt_addition(has_tools: bool = True) -> str:
    """Get the memory system prompt addition.

    Args:
        has_tools: Whether memory tools are available.

    Returns:
        Prompt text to add to system prompt.
    """
    if has_tools:
        return MEMORY_SYSTEM_PROMPT
    return ""


def format_memory_context(facts: dict[str, Any] | str) -> str:
    """Format user facts for prompt injection.

    Args:
        facts: Dict of key-value facts or pre-formatted string.

    Returns:
        Formatted context string.
    """
    if not facts:
        return ""

    if isinstance(facts, str):
        # Already formatted string
        context = facts
    else:
        # Dict of facts
        lines = [f"- {k}: {v}" for k, v in facts.items()]
        context = "\n".join(lines)

    return f"""## User Context (from memory)

{context}

Use this context to personalize responses. Update with `remember()` if you learn new facts.
"""
