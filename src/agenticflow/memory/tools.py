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
            query: Search term (semantic or keyword-based)
        """
        # Try semantic search first if vectorstore is available
        if memory.vectorstore:
            try:
                results = await memory.search(query, k=5)
                if results:
                    lines = []
                    for result in results:
                        content = result.content[:200]  # Truncate long content
                        score = f"(relevance: {result.score:.2f})" if hasattr(result, 'score') else ""
                        lines.append(f"- {content} {score}")
                    return "Found (semantic search):\n" + "\n".join(lines)
            except Exception as e:
                # Fall back to keyword search if semantic fails
                pass
        
        # Fallback: keyword search over long-term facts
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
        return "Found (keyword search):\n" + "\n".join(lines)

    return BaseTool(
        name="search_memories",
        description="""Search long-term memory for relevant facts.

Uses semantic search when available, falls back to keyword matching.
Use this when you need to find information but don't know the exact key.

Args:
    query: Search term (will find semantically related content)

Returns:
    Matching memories or indication if none found""",
        func=search_memories,
        args_schema={
            "query": {"type": "string", "description": "Search term to find relevant memories"},
        },
    )


def _create_search_conversation_tool(memory: Memory) -> BaseTool:
    """Create a tool to search conversation history semantically."""
    async def search_conversation(query: str, max_results: int = 5) -> str:
        """Search through past conversation messages for relevant context.
        
        Use this for:
        - Finding what was discussed earlier in long conversations
        - Retrieving context that may not have been saved as a fact
        - When conversation exceeds context window limits
        
        Args:
            query: What to search for in the conversation history
            max_results: Maximum number of messages to return (default: 5)
        """
        # Get conversation messages
        messages = await memory.get_messages()
        
        if not messages:
            return "No conversation history found."
        
        # If vectorstore available, use semantic search
        if memory.vectorstore:
            try:
                # Search through message content
                query_lower = query.lower()
                relevant_messages = []
                
                for msg in messages:
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    # Simple relevance check (could be enhanced with embeddings)
                    if query_lower in content.lower():
                        relevant_messages.append(content)
                
                if relevant_messages:
                    results = relevant_messages[:max_results]
                    return "Relevant messages from conversation:\n" + "\n---\n".join(results)
                else:
                    return f"No relevant conversation found for: {query}"
            except Exception:
                pass
        
        # Fallback: return recent messages
        recent = messages[-max_results:]
        lines = []
        for msg in recent:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            role = getattr(msg, 'role', 'unknown')
            lines.append(f"[{role}] {content[:150]}")
        
        return "Recent conversation context:\n" + "\n".join(lines)
    
    return BaseTool(
        name="search_conversation",
        description="""Search through conversation history for relevant context.

Critical for long conversations that exceed context window limits.
Use when you need to recall what was discussed earlier.

Args:
    query: What to search for in past messages
    max_results: How many results to return (default: 5)

Returns:
    Relevant messages from the conversation""",
        func=search_conversation,
        args_schema={
            "query": {"type": "string", "description": "What to search for in conversation history"},
            "max_results": {"type": "integer", "description": "Maximum results (default: 5)", "default": 5},
        },
    )


def create_memory_tools(memory: Memory) -> list[BaseTool]:
    """Create memory tools bound to a Memory instance.

    Args:
        memory: The Memory instance.

    Returns:
        List of memory tools.
    """
    tools = [
        _create_remember_tool(memory),
        _create_recall_tool(memory),
        _create_forget_tool(memory),
        _create_search_memories_tool(memory),
        _create_search_conversation_tool(memory),  # New: search conversation history
    ]
    return tools


# ============================================================
# Memory Prompt Templates
# ============================================================

MEMORY_SYSTEM_PROMPT = """
## Long-Term Memory & Conversation Search

You have memory tools to persist facts AND search long conversations.

**CRITICAL WORKFLOWS:**

1. **At start of conversation** - Call `search_memories("user")` to recall what you know
2. **When user shares info** - IMMEDIATELY call `remember()` to save it  
3. **When asked about something** - ALWAYS search first before saying "I don't know":
   - For facts → `search_memories(query)` or `recall(key)`
   - For past conversation → `search_conversation(query)`
4. **In long conversations** - Use `search_conversation()` to find earlier context

**ALWAYS call `remember(key, value)` when user shares:**
- Their name → remember("name", "Alice")
- Preferences → remember("favorite_color", "blue")
- Personal info → remember("birth_year", "1990")
- Their work/projects → remember("occupation", "developer")

**Available tools:**
- `remember(key, value)` - Save a fact to long-term memory (CALL THIS when user shares info!)
- `recall(key)` - Get a specific saved fact
- `forget(key)` - Remove a fact (only when user asks)
- `search_memories(query)` - Search long-term facts (semantic when available)
- `search_conversation(query)` - Search conversation history (for long conversations!)

**When to use search_conversation:**
- Conversation is very long (10+ messages)
- User asks "what did I say earlier about..."
- You need context from earlier in the conversation
- Important details may have been discussed but not saved as facts

**Critical:** NEVER say "I don't know" without searching first!
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
