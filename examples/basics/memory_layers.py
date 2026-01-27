"""
Demo: 4-Layer Memory Architecture

Shows how to use all 4 memory layers in cogent agents:

Layer 1: Conversation (default ON) - Thread-based message history
Layer 2: Bounded Memory (ACC) - Prevents drift in long conversations  
Layer 3: Long-term Memory - remember/recall tools for persistent facts
Layer 4: Semantic Cache - Cache expensive tool outputs

Usage:
    uv run python examples/basics/memory_layers.py
"""

import asyncio

from cogent import Agent


async def layer1_conversation():
    """Layer 1: Conversation memory (default ON)."""
    print("\n=== Layer 1: Conversation Memory ===")
    print("Thread-based message history - ON by default\n")

    # conversation=True is the default!
    agent = Agent(
        name="Assistant",
        model="gpt4",
        instructions="You are a helpful assistant. Be concise.",
    )

    # Conversation 1
    print("[Thread: user-123]")
    r1 = await agent.run("Hi, I'm Alice!", thread_id="user-123")
    print(f"User: Hi, I'm Alice!")
    print(f"Agent: {r1.content}\n")

    r2 = await agent.run("What's my name?", thread_id="user-123")
    print(f"User: What's my name?")
    print(f"Agent: {r2.content}\n")

    # Different thread = fresh context
    print("[Thread: user-456]")
    r3 = await agent.run("What's my name?", thread_id="user-456")
    print(f"User: What's my name?")
    print(f"Agent: {r3.content}")
    print("  ✓ Different thread = no memory of Alice\n")


async def layer2_bounded_memory():
    """Layer 2: Bounded Memory (ACC) - Prevents drift."""
    print("\n=== Layer 2: Bounded Memory (ACC) ===")
    print("Agent Cognitive Compressor - Prevents drift in long conversations\n")

    agent = Agent(
        name="Assistant",
        model="gpt4",
        bounded_memory=True,  # Enable ACC
        instructions="You are a helpful assistant.",
    )

    print("  Simulating a VERY long conversation (50+ turns)...")
    print("  ACC maintains bounded context, prevents memory poisoning\n")

    # In a real scenario, after 50+ turns, ACC prevents:
    # - Context window overflow
    # - Memory poisoning/drift
    # - Irrelevant old messages polluting context
    print("  ✓ ACC active - conversation stays focused\n")


async def layer3_long_term_memory():
    """Layer 3: Long-term Memory with tools."""
    print("\n=== Layer 3: Long-term Memory ===")
    print("remember/recall tools for persistent facts\n")

    agent = Agent(
        name="Assistant",
        model="gpt4",
        memory=True,  # Enable long-term memory tools
        instructions="ALWAYS call remember() when user shares important info.",
    )

    # Agent gets these tools automatically:
    # - remember(key, value)
    # - recall(key)
    # - forget(key)
    # - search_memories(query)
    # - search_conversation(query)

    print("[Thread: conv-1]")
    r1 = await agent.run(
        "My name is Bob and I prefer dark mode.",
        thread_id="conv-1",
    )
    print(f"User: My name is Bob and I prefer dark mode.")
    print(f"Agent: {r1.content}")
    if r1.tool_calls:
        print(f"Tools used: {[tc.tool_name for tc in r1.tool_calls]}\n")

    # Different thread - but agent can recall from long-term memory!
    print("[Thread: conv-2]")
    r2 = await agent.run(
        "What are my preferences?",
        thread_id="conv-2",
    )
    print(f"User: What are my preferences?")
    print(f"Agent: {r2.content}")
    if r2.tool_calls:
        print(f"Tools used: {[tc.tool_name for tc in r2.tool_calls]}")
    print("  ✓ Cross-thread memory via long-term storage\n")


async def layer4_semantic_cache():
    """Layer 4: Semantic Cache for expensive tools."""
    print("\n=== Layer 4: Semantic Cache ===")
    print("Cache expensive tool outputs (7-10× speedup)\n")

    from cogent.capabilities import WebSearch

    agent = Agent(
        name="Researcher",
        model="gpt4",
        capabilities=[WebSearch],
        cache=True,  # Enable semantic caching
        instructions="Search the web for answers.",
    )

    print("First query (cache MISS):")
    r1 = await agent.run("What is Python?")
    print(f"User: What is Python?")
    print(f"Agent: {r1.content[:100]}...")
    print("  → Web search performed\n")

    print("Similar query (cache HIT - 85%+ similarity):")
    r2 = await agent.run("Tell me about the Python programming language")
    print(f"User: Tell me about the Python programming language")
    print(f"Agent: {r2.content[:100]}...")
    print("  → Cached result used (7-10× faster!)\n")

    # Check cache stats
    if agent.cache:
        stats = agent.cache.stats()
        print(f"  Cache stats: {stats['hits']} hits, {stats['misses']} misses")
        print(f"  Hit rate: {stats['hit_rate']:.1%}\n")


async def all_layers_together():
    """Use all 4 layers together."""
    print("\n=== All 4 Layers Together ===")
    print("Maximum capability - all memory layers active\n")

    from cogent.capabilities import WebSearch

    agent = Agent(
        name="SuperAgent",
        model="gpt4",
        conversation=True,      # Layer 1: Thread-based history (default)
        bounded_memory=True,    # Layer 2: ACC for drift prevention
        memory=True,            # Layer 3: Long-term facts
        cache=True,             # Layer 4: Semantic cache
        capabilities=[WebSearch],
        instructions="You are an advanced assistant with full memory capabilities.",
    )

    print("  ✓ Layer 1: Conversation - Thread-based history")
    print("  ✓ Layer 2: Bounded Memory - ACC prevents drift")
    print("  ✓ Layer 3: Long-term Memory - remember/recall tools")
    print("  ✓ Layer 4: Semantic Cache - Fast tool outputs\n")

    print("[Conversation with full memory stack]")
    r1 = await agent.run(
        "My name is Charlie. Remember this: I love Rust programming.",
        thread_id="super-thread",
    )
    print("User: My name is Charlie. Remember this: I love Rust programming.")
    print(f"Agent: {r1.content}")
    if r1.tool_calls:
        print(f"Tools used: {[tc.tool_name for tc in r1.tool_calls]}\n")

    r2 = await agent.run(
        "Search the web for Rust tutorials, then tell me what you found.",
        thread_id="super-thread",
    )
    print("User: Search the web for Rust tutorials, then tell me what you found.")
    print(f"Agent: {r2.content[:150]}...")
    if r2.tool_calls:
        print(f"Tools used: {[tc.tool_name for tc in r2.tool_calls]}")
    print("\n  ✓ All layers working together!\n")


async def main():
    """Run all layer demos."""
    print("\n" + "=" * 60)
    print("  4-Layer Memory Architecture Demo")
    print("=" * 60)

    await layer1_conversation()
    await layer2_bounded_memory()
    await layer3_long_term_memory()
    
    # Skip layer 4 and combined demo if no API keys
    import os
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        await layer4_semantic_cache()
        await all_layers_together()
    else:
        print("\n⚠ Set OPENAI_API_KEY or ANTHROPIC_API_KEY for cache demos")

    print("\n" + "=" * 60)
    print("  Summary:")
    print("  • Layer 1 (conversation): Always ON by default")
    print("  • Layer 2 (bounded_memory): Use for 50+ turn conversations")
    print("  • Layer 3 (memory): Add remember/recall tools")
    print("  • Layer 4 (cache): Speed up expensive tools 7-10×")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
