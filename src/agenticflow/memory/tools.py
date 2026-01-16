"""Memory tools for agents.

Provides tools that allow agents to manage long-term memory:
- remember: Save important facts
- recall: Get a specific fact
- forget: Remove a fact
- search_memories: Find relevant memories

These tools are automatically added to agents with memory enabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.tools.base import BaseTool

if TYPE_CHECKING:
    from agenticflow.memory.core import Memory


def _create_remember_tool(memory: Memory) -> BaseTool:
    """Create a remember tool bound to a Memory instance."""
    async def remember(key: str, value: str) -> str:
        """Save an important fact to long-term memory.

        Args:
            key: Short descriptive key (e.g., "name", "preferred_language", "project")
            value: The fact to remember
        """
        await memory.remember(key, value)
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


def _create_recall_tool(memory: Memory) -> BaseTool:
    """Create a recall tool bound to a Memory instance."""
    async def recall(key: str) -> str:
        """Recall a specific fact from long-term memory.

        Args:
            key: The key of the fact to recall
        """
        value = await memory.recall(key)
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


def _create_forget_tool(memory: Memory) -> BaseTool:
    """Create a forget tool bound to a Memory instance."""
    async def forget(key: str) -> str:
        """Remove a fact from long-term memory.

        Args:
            key: The key of the fact to forget
        """
        success = await memory.forget(key)
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


def _create_search_memories_tool(memory: Memory) -> BaseTool:
    """Create a search_memories tool bound to a Memory instance."""
    async def search_memories(query: str) -> str:
        """Search long-term memory for relevant facts.

        Args:
            query: Search term (matches keys and values)
        """
        # Get all keys and search through them
        keys = await memory.keys()
        if not keys:
            return "No memories stored."

        query_lower = query.lower()
        matches: list[tuple[str, Any]] = []

        for key in keys:
            if key.startswith("_"):  # Skip internal keys like _messages
                continue
            value = await memory.recall(key)
            if value is not None:
                # Check if query matches key or value
                if query_lower in key.lower() or query_lower in str(value).lower():
                    matches.append((key, value))

        if not matches:
            return f"No memories found matching: {query}"

        lines = [f"- {k}: {v}" for k, v in matches[:5]]
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


def create_memory_tools(memory: Memory) -> list[BaseTool]:
    """Create memory tools bound to a Memory instance.

    Args:
        memory: The Memory instance.

    Returns:
        List of memory tools.
    """
    return [
        _create_remember_tool(memory),
        _create_recall_tool(memory),
        _create_forget_tool(memory),
        _create_search_memories_tool(memory),
    ]


# ============================================================
# Memory Prompt Templates
# ============================================================

MEMORY_SYSTEM_PROMPT = """
## Long-Term Memory

You have long-term memory tools. USE THEM to persist important facts.

**IMPORTANT WORKFLOW:**
1. **At start of conversation** - Call `search_memories("user")` to recall what you know
2. **When user shares info** - IMMEDIATELY call `remember()` to save it
3. **When asked about user** - Call `recall()` or `search_memories()` first

**ALWAYS call `remember(key, value)` when user shares:**
- Their name → remember("name", "Alice")
- Preferences → remember("favorite_color", "blue")
- Personal info → remember("birth_year", "1990")
- Their work/projects → remember("occupation", "developer")

**Available tools:**
- `remember(key, value)` - Save a fact (CALL THIS when user shares info!)
- `recall(key)` - Get a specific saved fact
- `forget(key)` - Remove a fact (only when user asks)
- `search_memories(query)` - Find relevant memories

**Critical:** When answering questions about user preferences or info, ALWAYS check memory first with `search_memories()` or `recall()` before saying "I don't know".
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
        context = facts
    else:
        lines = [f"- {k}: {v}" for k, v in facts.items()]
        context = "\n".join(lines)

    return f"""## IMPORTANT: Known User Information

The following facts are stored in your memory about this user. USE this information to answer their questions:

{context}

When the user asks about themselves (name, preferences, etc.), REFER to this information directly. Do NOT say "I don't know" if the answer is here.
"""
