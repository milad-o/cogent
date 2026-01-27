"""Memory tools for agents.

Provides tools that allow agents to manage long-term memory:
- remember: Save important facts
- recall: Get a specific fact
- forget: Remove a fact
- search_memories: Find relevant memories

These tools are automatically added to agents with memory enabled.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from cogent.tools.base import BaseTool

# Optional: fuzzy matching for fast key search
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

if TYPE_CHECKING:
    from cogent.memory.core import Memory


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
            "key": {
                "type": "string",
                "description": "Short descriptive key for the fact",
            },
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


def _normalize_key(key: str) -> str:
    """Normalize a key for better fuzzy matching.
    
    Transformations:
    - Lowercase
    - Replace underscores/hyphens with spaces
    - Remove special characters
    - Collapse multiple spaces
    
    Args:
        key: Original key
        
    Returns:
        Normalized key for matching
    """
    # Lowercase
    key = key.lower()
    # Replace separators with spaces
    key = key.replace("_", " ").replace("-", " ")
    # Remove special characters (keep alphanumeric and spaces)
    key = re.sub(r"[^a-z0-9\s]", "", key)
    # Collapse multiple spaces
    key = re.sub(r"\s+", " ", key).strip()
    return key


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
        """Search long-term memory for relevant facts using fuzzy matching.

        Uses fuzzy string matching (rapidfuzz) for fast, offline key matching.
        Falls back to semantic search if vectorstore is available and fuzzy fails.

        Args:
            query: Search term (fuzzy matched against keys)
        """
        # Get all keys (excluding internal ones)
        all_keys = await memory.keys()
        user_keys = [
            key for key in all_keys
            if not key.startswith("thread:") and not key.startswith("_")
        ]

        if not user_keys:
            return "No memories stored yet."

        # Method 1: Fuzzy matching (fast, free, offline)
        if HAS_RAPIDFUZZ:
            # Normalize query for better matching
            normalized_query = _normalize_key(query)
            
            # Create mapping of normalized keys to original keys
            key_map = {_normalize_key(key): key for key in user_keys}
            normalized_keys = list(key_map.keys())
            
            # Fuzzy search with token_sort_ratio (handles word order)
            matches = process.extract(
                normalized_query,
                normalized_keys,
                scorer=fuzz.token_sort_ratio,
                limit=5,
                score_cutoff=40  # Minimum 40% similarity
            )
            
            if matches:
                lines = []
                for norm_key, score, _ in matches:
                    original_key = key_map[norm_key]
                    value = await memory.recall(original_key)
                    lines.append(f"- {original_key}: {value}")
                return "Found (fuzzy match):\\n" + "\\n".join(lines)
        
        # Method 2: Semantic search (if vectorstore available and fuzzy failed/unavailable)
        if memory.vectorstore:
            try:
                results = await memory.search(query, k=5)
                if results:
                    lines = []
                    seen_keys = set()
                    for result in results:
                        if (
                            "memory_key" in result.document.metadata.custom
                            and result.document.metadata.custom["memory_key"]
                        ):
                            key = result.document.metadata.custom["memory_key"]
                            if key.startswith("thread:") or key.startswith("_"):
                                continue
                            if key not in seen_keys:
                                seen_keys.add(key)
                                value = await memory.recall(key)
                                lines.append(f"- {key}: {value}")
                    if lines:
                        return "Found (semantic search):\\n" + "\\n".join(lines)
            except Exception:
                pass

        # Method 3: Simple keyword search (fallback)
        query_lower = query.lower()
        matching_keys = [
            key for key in user_keys
            if query_lower in key.lower()
        ]

        if not matching_keys:
            return f"No memories found matching: {query}"

        lines = []
        for key in matching_keys[:5]:
            value = await memory.recall(key)
            lines.append(f"- {key}: {value}")

        return "Found (keyword search):\\n" + "\\n".join(lines)
        if not keys:
            return "No memories stored."

        query_lower = query.lower()
        matches: list[tuple[str, Any, float]] = []  # (key, value, score)

        for key in keys:
            if key.startswith("_"):  # Skip internal keys like _messages
                continue
            value = await memory.recall(key)
            if value is not None:
                score = 0.0
                key_lower = key.lower()
                value_str = str(value).lower()
                
                # Exact match on key (highest score)
                if query_lower == key_lower:
                    score = 100.0
                # Substring match on key (high score)
                elif query_lower in key_lower:
                    score = 80.0
                # Key contains substring of query (good score)
                elif any(word in key_lower for word in query_lower.split() if len(word) > 2):
                    score = 60.0
                # Substring match in value
                elif query_lower in value_str:
                    score = 40.0
                # Value contains substring of query
                elif any(word in value_str for word in query_lower.split() if len(word) > 2):
                    score = 20.0
                
                if score > 0:
                    matches.append((key, value, score))

        if not matches:
            return f"No memories found matching: {query}"

        # Sort by score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        lines = [f"- {k}: {v}" for k, v, _ in matches[:5]]
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
            "query": {
                "type": "string",
                "description": "Search term to find relevant memories",
            },
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
        # Get current thread_id from memory context
        thread_id = getattr(memory, "_current_thread_id", None)

        # Get conversation messages
        messages = await memory.get_messages(thread_id=thread_id)

        if not messages:
            return "No conversation history found."

        # If vectorstore available, use semantic search
        if memory.vectorstore:
            try:
                # Search through message content
                query_lower = query.lower()
                relevant_messages = []

                for msg in messages:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    # Simple relevance check (could be enhanced with embeddings)
                    if query_lower in content.lower():
                        relevant_messages.append(content)

                if relevant_messages:
                    results = relevant_messages[:max_results]
                    return "Relevant messages from conversation:\n" + "\n---\n".join(
                        results
                    )
                else:
                    return f"No relevant conversation found for: {query}"
            except Exception:
                pass  # Vector search failed, fall back to recent messages

        # Fallback: return recent messages
        recent = messages[-max_results:]
        lines = []
        for msg in recent:
            content = msg.content if hasattr(msg, "content") else str(msg)
            role = getattr(msg, "role", "unknown")
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
            "query": {
                "type": "string",
                "description": "What to search for in conversation history",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results (default: 5)",
                "default": 5,
            },
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
