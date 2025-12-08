"""
Demo: Memory-First Architecture with LLM

Demonstrates the new unified Memory system with actual LLM interactions.

Features:
- Thread-based conversation memory (agent remembers within a thread)
- Cross-thread memory sharing via scoped namespaces  
- Persistence with SQLAlchemyStore
- Memory tools for long-term facts

Usage:
    uv run python examples/06_memory.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import tempfile
from pathlib import Path

from config import get_model, settings

from agenticflow import Agent
from agenticflow.memory import Memory


async def demo_conversation_memory():
    """Show LLM remembering conversation context via chat()."""
    print("\n--- Conversation Memory (Thread-Based) ---")
    
    model = get_model()
    memory = Memory(agentic=True)  # agentic=True enables memory tools
    
    # Create agent with the new Memory class
    agent = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant. Be concise (1-2 sentences max).",
        memory=memory,  # Tools auto-added when agentic=True!
    )
    
    # Chat in a thread - agent automatically remembers!
    thread_id = "user-alice-123"
    
    print(f"\n[Thread: {thread_id}]")
    
    # Turn 1: User introduces themselves
    response1 = await agent.chat(
        "Hi! My name is Alice and I'm a Python developer.",
        thread_id=thread_id,
    )
    print(f"  User: Hi! My name is Alice and I'm a Python developer.")
    print(f"  Assistant: {response1}")
    
    # Turn 2: Ask something that requires memory
    response2 = await agent.chat(
        "What's my name and what do I do?",
        thread_id=thread_id,  # Same thread - remembers!
    )
    print(f"\n  User: What's my name and what do I do?")
    print(f"  Assistant: {response2}")
    
    # Show what's stored in long-term memory (via remember() tool)
    print(f"\n  📝 Long-term facts (stored via remember() tool):")
    keys = await memory.keys()
    for key in keys:
        if not key.startswith("thread:") and not key.startswith("_"):
            value = await memory.recall(key)
            print(f"    • {key}: {value}")
    
    # Different thread - should know from context injection!
    print(f"\n[Thread: different-thread] (fresh thread, but knows from memory)")
    response3 = await agent.chat(
        "What's my name and occupation?",
        thread_id="different-thread",
    )
    print(f"  User: What's my name and occupation?")
    print(f"  Assistant: {response3}")


async def demo_persistent_memory():
    """Show memory persisting across agent restarts."""
    print("\n--- Persistent Memory (SQLite) ---")
    
    try:
        from agenticflow.memory.stores import SQLAlchemyStore
    except ImportError:
        print("  ⚠ SQLAlchemy not installed. Run: uv add sqlalchemy aiosqlite")
        return
    
    model = get_model()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "agent_memory.db"
        db_url = f"sqlite+aiosqlite:///{db_path}"
        
        print(f"  📁 Database: {db_path}")
        
        # --- Session 1: Have a conversation ---
        print("\n  [Session 1]")
        store = SQLAlchemyStore(db_url)
        await store.initialize()
        memory = Memory(store=store, agentic=True)
        
        agent = Agent(
            name="PersistentAgent",
            model=model,
            instructions="Be very concise (one sentence max).",
            memory=memory,
        )
        
        response = await agent.chat(
            "My favorite programming language is Rust.",
            thread_id="session-demo",
        )
        print(f"  User: My favorite programming language is Rust.")
        print(f"  Assistant: {response}")
        
        await store.close()
        print("  ✓ Session 1 ended, database closed")
        
        # --- Session 2: New agent, same database ---
        print("\n  [Session 2] (simulating app restart)")
        store2 = SQLAlchemyStore(db_url)
        await store2.initialize()
        memory2 = Memory(store=store2, agentic=True)
        
        agent2 = Agent(
            name="PersistentAgent",
            model=model,
            instructions="Be very concise (one sentence max).",
            memory=memory2,
        )
        
        response2 = await agent2.chat(
            "What's my favorite programming language?",
            thread_id="session-demo",  # Same thread!
        )
        print(f"  User: What's my favorite programming language?")
        print(f"  Assistant: {response2}")
        
        await store2.close()


async def demo_memory_tools():
    """Show agent automatically using memory tools with agentic=True."""
    print("\n--- Agent with Memory Tools (agentic=True) ---")
    
    model = get_model()
    memory = Memory(agentic=True)  # agentic=True enables tools
    
    # Just pass memory - tools are auto-added when agentic=True!
    agent = Agent(
        name="MemoryAgent",
        model=model,
        instructions="You are a helpful assistant. Be concise.",
        memory=memory,  # Tools: remember, recall, forget, search_memories
    )
    
    # Show what tools the agent has
    tool_names = [t.name for t in agent._direct_tools]
    print(f"  🔧 Auto-added tools: {tool_names}")
    
    # Conversation
    print("\n  User: My favorite color is blue and I was born in 1990.")
    response1 = await agent.chat(
        "My favorite color is blue and I was born in 1990.",
        thread_id="facts-demo",
    )
    print(f"  Agent: {response1}")
    
    print("\n  User: What do you remember about me?")
    response2 = await agent.chat(
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
    
    model = get_model()
    
    # Create shared memory with agentic tools
    shared_memory = Memory(agentic=True)
    
    # Research scope for the team
    research = shared_memory.scoped("team:research")
    
    # Two agents sharing memory
    researcher = Agent(
        name="Researcher",
        model=model,
        instructions="You are a researcher. When asked to research, provide 2-3 bullet points. Be concise.",
        memory=shared_memory,
    )
    
    writer = Agent(
        name="Writer",
        model=model,
        instructions="You are a writer. Summarize findings into one paragraph.",
        memory=shared_memory,
    )
    
    # Researcher stores findings
    print("\n  [Researcher working...]")
    findings = await researcher.chat(
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
    
    summary = await writer.chat(
        f"Summarize these findings in one paragraph: {stored}",
        thread_id="writing-session",
    )
    print(f"  Writer: {summary}")


async def main():
    print("\n" + "=" * 60)
    print("  Memory-First Architecture Demo (with LLM)")
    print(f"  Provider: {settings.llm_provider}")
    print("=" * 60)
    
    await demo_conversation_memory()
    await demo_persistent_memory()
    await demo_memory_tools()
    await demo_shared_memory()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
