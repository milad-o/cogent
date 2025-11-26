"""
Demo: Agent Memory

Demonstrates memory capabilities for conversation persistence.

Features:
- Short-term memory: Thread-based conversation history
- Long-term memory: Cross-thread key-value storage
- LangGraph compatibility: Use MemorySaver, SqliteSaver, etc.

Usage:
    uv run python examples/06_memory.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agenticflow import Agent, InMemorySaver

load_dotenv()


async def demo_conversation_memory():
    """Show thread-based conversation memory."""
    print("\n--- Thread-Based Memory ---")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    # Create agent with memory
    assistant = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant. Remember user details.",
        memory=InMemorySaver(),  # Enable memory
    )
    
    # Chat in thread 1
    print("\n[Thread: user-123]")
    response1 = await assistant.chat(
        "Hi! My name is Alice and I love Python.",
        thread_id="user-123",
    )
    print(f"  User: Hi! My name is Alice and I love Python.")
    print(f"  Assistant: {response1[:100]}...")
    
    response2 = await assistant.chat(
        "What's my name and what do I love?",
        thread_id="user-123",  # Same thread - remembers!
    )
    print(f"  User: What's my name and what do I love?")
    print(f"  Assistant: {response2[:100]}...")
    
    # Different thread - fresh context
    print("\n[Thread: user-456] (different thread)")
    response3 = await assistant.chat(
        "What's my name?",
        thread_id="user-456",  # Different thread
    )
    print(f"  User: What's my name?")
    print(f"  Assistant: {response3[:100]}...")


async def demo_long_term_memory():
    """Show cross-thread long-term memory."""
    print("\n--- Long-Term Memory ---")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    assistant = Agent(
        name="Assistant",
        model=model,
        memory=True,  # Shorthand for InMemorySaver()
    )
    
    # Store facts
    await assistant.memory.remember("user_name", "Bob", namespace="user_facts")
    await assistant.memory.remember("favorite_color", "blue", namespace="user_facts")
    print("  Stored: user_name=Bob, favorite_color=blue")
    
    # Recall from anywhere
    name = await assistant.memory.recall("user_name", namespace="user_facts")
    color = await assistant.memory.recall("favorite_color", namespace="user_facts")
    print(f"  Recalled: user_name={name}, favorite_color={color}")
    
    # Forget
    await assistant.memory.forget("favorite_color", namespace="user_facts")
    color_after = await assistant.memory.recall("favorite_color", namespace="user_facts")
    print(f"  After forget: favorite_color={color_after}")


async def demo_memory_operations():
    """Show direct memory operations."""
    print("\n--- Memory Operations ---")
    
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    assistant = Agent(
        name="Assistant",
        model=model,
        memory=InMemorySaver(),
    )
    
    # Save state directly
    await assistant.memory.save(
        thread_id="demo-thread",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        metadata={"user_id": "user-789"},
    )
    print("  Saved state to demo-thread")
    
    # Load state
    snapshot = await assistant.memory.load("demo-thread")
    print(f"  Loaded: {len(snapshot.messages)} messages")
    print(f"  Metadata: {snapshot.metadata}")
    
    # Get messages
    messages = await assistant.memory.get_messages("demo-thread")
    print(f"  Messages: {messages}")
    
    # Set context
    await assistant.memory.set_context("demo-thread", "mood", "happy")
    mood = await assistant.memory.get_context("demo-thread", "mood")
    print(f"  Context mood={mood}")
    
    # Summary
    summary = assistant.memory.summary()
    print(f"  Summary: {summary}")


async def main():
    print("\n" + "=" * 50)
    print("  Agent Memory Demo")
    print("=" * 50)
    
    await demo_conversation_memory()
    await demo_long_term_memory()
    await demo_memory_operations()
    
    print("\n" + "=" * 50)
    print("✅ Done!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
