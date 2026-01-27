"""
Demo: Memory-First Architecture with LLM

Demonstrates the unified Memory system:

**Memory is always agentic** - automatically provides memory tools:
- remember(): Store long-term facts
- recall(): Retrieve stored facts (requires exact key)
- forget(): Remove facts
- search_memories(): Search and discover keys (fuzzy matching)
- search_conversation(): Search conversation history

**Key Discovery Pattern:**
Agents use search_memories() to discover what keys exist, then recall() for direct access.

Features:
- Key discovery via search_memories()
- Thread-based conversation memory
- Cross-thread memory sharing via scoped namespaces
- Persistence with SQLAlchemyStore
- Automatic long-term fact management

Usage:
    uv run python examples/basics/memory.py
"""

import asyncio
import tempfile
from pathlib import Path

from cogent import Agent
from cogent.memory import Memory


async def demo_memory_discovery():
    """Show how agents discover memory keys via search."""
    print("\n--- Memory Key Discovery ---")

    memory = Memory()

    # Pre-populate with some facts using varied key names
    print("\n  📝 Pre-populating memory with facts...")
    await memory.remember("user_name", "Alice")
    await memory.remember("favorite_language", "Python")
    await memory.remember("job_title", "Senior Developer")
    await memory.remember("team", "AI Research")
    await memory.remember("email", "alice@example.com")

    print("  Stored keys:")
    keys = await memory.keys()
    for key in keys:
        if not key.startswith("thread:") and not key.startswith("_"):
            value = await memory.recall(key)
            print(f"    • {key}: {value}")

    agent = Agent(
        name="Assistant",
        model="gpt4",
        instructions="Always use search_memories() to discover what you know. Be concise.",
        memory=memory,
    )

    # Test 1: Agent searches for 'user'
    print("\n  [Test: What do you know about the user?]")
    response1 = await agent.run(
        "What do you know about the user?",
        thread_id="discovery-demo",
    )
    print(f"  Agent: {response1.content}")
    if response1.tool_calls:
        print(f"  Tools used: {[(tc.tool_name, tc.arguments) for tc in response1.tool_calls]}")

    # Test 2: Specific query that requires discovery
    print("\n  [Test: What programming language does the user prefer?]")
    response2 = await agent.run(
        "What programming language does the user prefer?",
        thread_id="discovery-demo",
    )
    print(f"  Agent: {response2.content}")
    if response2.tool_calls:
        print(f"  Tools used: {[(tc.tool_name, tc.arguments) for tc in response2.tool_calls]}")

    # Show direct search vs recall
    print("\n  💡 Key Discovery Mechanism:")
    search_tool = next(t for t in memory.tools if t.name == "search_memories")

    # Search finds keys even with fuzzy queries
    result = await search_tool.func(query="language")
    print(f"    search_memories('language') → {result}")

    result = await search_tool.func(query="work")
    print(f"    search_memories('work') → {result}")

    # Recall needs exact key
    recall_tool = next(t for t in memory.tools if t.name == "recall")
    result = await recall_tool.func(key="favorite_language")
    print(f"    recall('favorite_language') → {result}")

    result = await recall_tool.func(key="occupation")  # Wrong key
    print(f"    recall('occupation') → {result} ❌")


async def demo_conversation_memory():
    """Show conversation memory with automatic fact storage."""
    print("\n--- Conversation Memory (Thread-Based) ---")


    # Memory is always agentic - tools are automatically available
    memory = Memory()

    agent = Agent(
        name="Assistant",
        model="gpt4",
        instructions="You are a helpful assistant. Be concise (1-2 sentences max).",
        memory=memory,  # Tools: remember, recall, forget, search_memories, search_conversation
    )

    # Show what tools were added
    tool_names = [t.name for t in agent._direct_tools]
    print(f"  🔧 Memory tools auto-added: {tool_names}\n")

    # Chat in a thread - agent automatically remembers!
    thread_id = "user-alice-123"

    print(f"\n[Thread: {thread_id}]")

    # Turn 1: User introduces themselves
    response1 = await agent.run(
        "Hi! My name is Alice and I'm a Python developer.",
        thread_id=thread_id,
    )
    print("  User: Hi! My name is Alice and I'm a Python developer.")
    print(f"  Assistant: {response1}")

    # Turn 2: Ask something that requires memory
    response2 = await agent.run(
        "What's my name and what do I do?",
        thread_id=thread_id,  # Same thread - remembers!
    )
    print("\n  User: What's my name and what do I do?")
    print(f"  Assistant: {response2.content}")
    if response2.tool_calls:
        print(f"  Tools used: {[(tc.tool_name, tc.arguments) for tc in response2.tool_calls]}")

    # Show what's stored in long-term memory
    print("\n  📝 Long-term facts (stored via remember() tool):")
    keys = await memory.keys()
    for key in keys:
        if not key.startswith("thread:") and not key.startswith("_"):
            value = await memory.recall(key)
            print(f"    • {key}: {value}")

    # Different thread - agent can recall from long-term memory!
    print("\n[Different thread - recalls from long-term memory]")
    response3 = await agent.run(
        "What's my name and occupation?",
        thread_id="different-thread",
    )
    print("  User: What's my name and occupation?")
    print(f"  Assistant: {response3.content}")
    if response3.tool_calls:
        print(f"  Tools used: {[(tc.tool_name, tc.arguments) for tc in response3.tool_calls]}")


async def demo_persistent_memory():
    """Show memory persisting across agent restarts."""
    print("\n--- Persistent Memory (SQLite) ---")

    try:
        from cogent.memory.stores import SQLAlchemyStore
    except ImportError:
        print("  ⚠ SQLAlchemy not installed. Run: uv add sqlalchemy aiosqlite")
        return


    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "agent_memory.db"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        print(f"  📁 Database: {db_path}")

        # --- Session 1: Have a conversation ---
        print("\n  [Session 1]")
        store = SQLAlchemyStore(db_url)
        await store.initialize()
        memory = Memory(store=store)

        agent = Agent(
            name="PersistentAgent",
            model="gpt4",
            instructions="Be very concise (one sentence max).",
            memory=memory,
        )

        response = await agent.run(
            "My favorite programming language is Rust.",
            thread_id="session-demo",
        )
        print("  User: My favorite programming language is Rust.")
        print(f"  Assistant: {response.content}")

        await store.close()
        print("  ✓ Session 1 ended, database closed")

        # --- Session 2: New agent, same database ---
        print("\n  [Session 2] (simulating app restart)")
        store2 = SQLAlchemyStore(db_url)
        await store2.initialize()
        memory2 = Memory(store=store2)

        agent2 = Agent(
            name="PersistentAgent",
            model="gpt4",
            instructions="Be very concise (one sentence max).",
            memory=memory2,
        )

        response2 = await agent2.run(
            "What's my favorite programming language?",
            thread_id="session-demo",  # Same thread!
        )
        print("  User: What's my favorite programming language?")
        print(f"  Assistant: {response2.content}")

        await store2.close()


async def demo_memory_tools():
    """Show agent automatically using memory tools."""
    print("\n--- Agent with Memory Tools ---")

    memory = Memory()  # Always agentic

    # Just pass memory - tools are auto-added!
    agent = Agent(
        name="MemoryAgent",
        model="gpt4",
        instructions="You are a helpful assistant. Be concise.",
        memory=memory,  # Tools: remember, recall, forget, search_memories
    )

    # Show what tools the agent has
    tool_names = [t.name for t in agent._direct_tools]
    print(f"  🔧 Auto-added tools: {tool_names}")

    # Conversation
    print("\n  User: My favorite color is blue and I was born in 1990.")
    response1 = await agent.run(
        "My favorite color is blue and I was born in 1990.",
        thread_id="facts-demo",
    )
    print(f"  Agent: {response1}")

    print("\n  User: What do you remember about me?")
    response2 = await agent.run(
        "What do you remember about me?",
        thread_id="facts-demo",
    )
    print(f"  Agent: {response2}")

    # Show what's been stored
    print("\n  📦 Long-term facts stored in memory:")
    keys = await memory.keys()
    for key in keys:
        if not key.startswith("thread:") and not key.startswith("_"):
            value = await memory.recall(key)
            print(f"    {key}: {value}")


async def demo_shared_memory():
    """Show multiple agents sharing memory."""
    print("\n--- Shared Memory (Multi-Agent) ---")


    # Create shared memory
    shared_memory = Memory()

    # Research scope for the team
    research = shared_memory.scoped("team:research")

    # Two agents sharing memory
    researcher = Agent(
        name="Researcher",
        model="gpt4",
        instructions="You are a researcher. When asked to research, provide 2-3 bullet points. Be concise.",
        memory=shared_memory,
    )

    writer = Agent(
        name="Writer",
        model="gpt4",
        instructions="You are a writer. Summarize findings into one paragraph.",
        memory=shared_memory,
    )

    # Researcher stores findings
    print("\n  [Researcher working...]")
    findings = await researcher.run(
        "Research: What are 3 benefits of Python for AI?",
        thread_id="research-session",
    )
    print(f"  Researcher: {findings}")

    # Store in shared memory
    await research.remember("python_ai_benefits", str(findings))
    print("  ✓ Findings saved to team memory")

    # Writer retrieves and uses findings
    print("\n  [Writer working...]")
    stored = await research.recall("python_ai_benefits")

    summary = await writer.run(
        f"Summarize these findings in one paragraph: {stored}",
        thread_id="writing-session",
    )
    print(f"  Writer: {summary}")


async def main():
    print("\n" + "=" * 60)
    print("  Memory-First Architecture Demo (with LLM)")
    print("  Memory is always agentic - tools auto-added")
    print("=" * 60)

    await demo_memory_discovery()     # How agents discover keys
    await demo_conversation_memory()  # Conversation + long-term memory
    await demo_persistent_memory()    # SQLite persistence
    await demo_memory_tools()         # Detailed tool usage
    await demo_shared_memory()        # Multi-agent sharing

    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
