"""
Demo: Memory Key Search with Fuzzy Matching

Shows how memory uses intelligent key search with automatic fallback:
1. Fuzzy matching (default) - Fast, free, offline (0.1ms)
2. Semantic search (optional) - Slower but smarter if vectorstore available
3. Keyword search (fallback) - Simple substring matching

Fuzzy matching allows flexible key discovery:
- Query: "preferences" → Finds: "preferred_mode", "preferred_language"
- Query: "contact" → Finds: "email", "phone_number"
- Query: "settings" → Finds: "notification_settings"

Usage:
    uv run python examples/basics/memory_semantic_search.py
"""

import asyncio

from cogent import Agent
from cogent.memory import Memory
from cogent.observability import Observer
from cogent.vectorstore import VectorStore


async def demo_semantic_memory():
    """Show fuzzy matching on memory keys (default behavior)."""
    print("\n" + "=" * 60)
    print("MEMORY KEY SEARCH DEMO (Fuzzy Matching)")
    print("=" * 60)

    # Create Memory with VectorStore as optional fallback
    memory = Memory(vectorstore=VectorStore())

    # Create agent with memory
    observer = Observer.trace()
    agent = Agent(
        name="Assistant",
        model="gpt-4o",
        instructions=(
            "You are a helpful assistant. "
            "Use search_memories() to find relevant information. "
            "Be concise."
        ),
        memory=memory,  # Fuzzy matching by default, semantic as fallback
        observer=observer,
    )

    # Save some preferences with varied key names
    print("\n📝 Saving user information...")
    await agent.run(
        "My name is Alice, email is alice@example.com, "
        "I prefer dark mode, and my language is Python.",
        thread_id="demo",
    )

    # Show saved keys
    print("\n💾 Stored keys:")
    keys = await memory.keys()
    for key in sorted(keys):
        if not key.startswith("thread:") and not key.startswith("_"):
            value = await memory.recall(key)
            print(f"   • {key} = {value}")

    print("\n" + "-" * 60)
    print("KEY SEARCH TESTS (Fuzzy Matching)")
    print("-" * 60)

    # Test 1: Query "preferences" should find "preferred_mode"
    print("\n[Test 1] Query: 'preferences'")
    print("         Expected: Find 'preferred_mode' key via fuzzy match")
    response = await agent.run("What are my preferences?", thread_id="demo")
    print(f"         Result: {response.content}")

    # Test 2: Query "contact" should find "email"
    print("\n[Test 2] Query: 'contact'")
    print("         Expected: Find 'email' key via fuzzy match")
    response = await agent.run("What's my contact info?", thread_id="demo")
    print(f"         Result: {response.content}")

    # Test 3: Query "user info" should find multiple keys
    print("\n[Test 3] Query: 'user info'")
    print("         Expected: Find 'name', 'email', etc.")
    response = await agent.run("Tell me about the user", thread_id="demo")
    print(f"         Result: {response.content}")

    print("\n" + "=" * 60)
    print("✅ Fuzzy matching enables fast, free key retrieval!")
    print("   Speed: ~0.1ms | Cost: Free | Offline: Yes")
    print("=" * 60)


async def demo_without_vectorstore():
    """Show behavior without vectorstore (pure fuzzy/keyword matching)."""
    print("\n" + "=" * 60)
    print("WITHOUT VECTORSTORE (Fuzzy + Keyword Only)")
    print("=" * 60)

    # Create Memory WITHOUT vectorstore
    memory = Memory()  # No vectorstore - fuzzy + keyword only

    observer = Observer.trace()
    agent = Agent(
        name="Basic",
        model="gpt-4o",
        instructions="Use search_memories() to find info. Be concise.",
        memory=memory,  # Fuzzy matching, no semantic fallback
        observer=observer,
    )

    print("\n📝 Saving: preferred_mode=dark")
    await memory.remember("preferred_mode", "dark")

    print("\n[Query] 'preferences'")
    print("        (Fuzzy matching finds it!)")
    response = await agent.run("What are my preferences?", thread_id="basic")
    print(f"        Result: {response.content}")

    print("\n[Query] 'mode'")
    print("        (Keyword match also works)")
    response = await agent.run("What mode do I prefer?", thread_id="basic")
    print(f"        Result: {response.content}")

    print("\n✅ Fuzzy matching works great without vectorstore!")
    print("   No API costs, instant results")


async def main():
    """Run all demos."""
    await demo_semantic_memory()
    print("\n")
    await demo_without_vectorstore()


if __name__ == "__main__":
    asyncio.run(main())
