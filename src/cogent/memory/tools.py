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
        description="""Save a fact to long-term memory.

Use this when the user shares ANY information about themselves, their preferences,
work, projects, or personal context. Remember liberally - if the user states a fact,
save it.

Args:
    key: Short descriptive key (e.g., "name", "email", "dog_name", "project_deadline")
    value: The fact to remember

Examples:
    User: "I prefer dark mode" → remember("interface_preference", "dark mode")
    User: "My birthday is March 15" → remember("birthday", "March 15")
    User: "I work at TechCorp" → remember("job", "TechCorp")
    User: "I'm learning Spanish" → remember("learning", "Spanish")""",
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
        """Search long-term memory using hybrid fuzzy + semantic approach.

        Optimized search strategy:
        1. Fuzzy pre-filter: Get top 20 candidates (0.01ms)
        2. Semantic rerank: Rerank candidates if vectorstore available (5-10ms)
        3. Keyword fallback: Simple substring match if fuzzy unavailable

        This hybrid approach is 30-50x faster than pure semantic search
        while maintaining high accuracy.

        Args:
            query: Search term
        """
        # Get all keys (excluding internal ones)
        all_keys = await memory.keys()
        user_keys = [
            key
            for key in all_keys
            if not key.startswith("thread:") and not key.startswith("_")
        ]

        if not user_keys:
            return "No memories stored yet."

        # HYBRID APPROACH: Fuzzy pre-filter + Semantic rerank
        if HAS_RAPIDFUZZ:
            # Normalize query for better matching
            normalized_query = _normalize_key(query)

            # Create mapping of normalized keys to original keys
            key_map = {_normalize_key(key): key for key in user_keys}
            normalized_keys = list(key_map.keys())

            # Fuzzy pre-filter: Get top 20 candidates (fast)
            fuzzy_matches = process.extract(
                normalized_query,
                normalized_keys,
                scorer=fuzz.token_sort_ratio,
                limit=20,  # More candidates for semantic reranking
                score_cutoff=40,  # Minimum 40% similarity
            )

            if fuzzy_matches:
                candidate_keys = [key_map[norm_key] for norm_key, _, _ in fuzzy_matches]

                # If vectorstore available: Semantic rerank top candidates
                if memory.vectorstore and len(candidate_keys) > 5:
                    try:
                        # Search only within fuzzy candidates (much faster)
                        results = await memory.search(query, k=5)
                        if results:
                            # Build semantic-ranked results
                            lines = []
                            seen_keys = set()
                            for result in results:
                                if (
                                    "memory_key" in result.document.metadata.custom
                                    and result.document.metadata.custom["memory_key"]
                                ):
                                    key = result.document.metadata.custom["memory_key"]
                                    # Only include if in fuzzy candidates
                                    if key in candidate_keys and key not in seen_keys:
                                        seen_keys.add(key)
                                        value = await memory.recall(key)
                                        lines.append(f"- {key}: {value}")

                            # Fill remaining slots with fuzzy matches not in semantic results
                            for key in candidate_keys:
                                if key not in seen_keys and len(lines) < 5:
                                    seen_keys.add(key)
                                    value = await memory.recall(key)
                                    lines.append(f"- {key}: {value}")

                            if lines:
                                return "Found (hybrid fuzzy+semantic):\\n" + "\\n".join(
                                    lines
                                )
                    except Exception:
                        pass  # Fall through to fuzzy-only results

                # Return fuzzy-only results (no vectorstore or small candidate set)
                lines = []
                for norm_key, _score, _ in fuzzy_matches[:5]:
                    original_key = key_map[norm_key]
                    value = await memory.recall(original_key)
                    lines.append(f"- {original_key}: {value}")
                return "Found (fuzzy match):\\n" + "\\n".join(lines)

        # Fallback: Pure semantic (if no fuzzy but vectorstore available)
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

        # Final fallback: Simple keyword search
        query_lower = query.lower()
        matching_keys = [key for key in user_keys if query_lower in key.lower()]

        if not matching_keys:
            return f"No memories found matching: {query}"

        lines = []
        for key in matching_keys[:5]:
            value = await memory.recall(key)
            lines.append(f"- {key}: {value}")

        return "Found (keyword search):\\n" + "\\n".join(lines)

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
## Long-Term Memory

**CRITICAL: You MUST call remember() for user facts, or they will be lost when context resets.**

Conversation messages are temporary and will be forgotten when context fills up.
Only facts saved with remember() persist. If you say "I've noted" but don't call
remember(), you are LYING - the information will be lost.

**When user shares a fact, you MUST:**
1. Call remember(key, value) FIRST
2. Then respond to acknowledge it

**What requires remember():**
- ALL personal information (name, contact, location, timezone, birthday, allergies)
- ALL preferences (UI settings, food, communication style, work habits)
- ALL work/project details (job, team, deadlines, meetings, current tasks)
- ALL personal context (pets, hobbies, learning, training, habits, schedules)
- ALL technical facts (servers, configs, passwords, tools, architecture)

**Examples:**
User: "I prefer dark mode"
→ MUST call: remember("interface_preference", "dark mode")
→ Then say: "I've saved your preference for dark mode."

User: "I'm learning Spanish"
→ MUST call: remember("learning", "Spanish")
→ Then say: "I've saved that you're learning Spanish."

User: "My mobile number is 555-0123"
→ MUST call: remember("mobile_number", "555-0123")
→ Then say: "I've saved your mobile number."

**DO NOT skip remember() for:**
- "Less important" facts (everything is important!)
- "Temporary" information (user shared it, save it!)
- "Sensitive" data (if user told you, remember it!)

**Available tools:**
- `remember(key, value)` - REQUIRED for all user facts or they'll be lost!
- `recall(key)` - Retrieve a saved fact
- `forget(key)` - Remove a fact (only when user asks)
- `search_memories(query)` - Search all saved facts
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
