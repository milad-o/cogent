"""ACC Demo - Bounded Memory vs Transcript Replay.

Shows the REAL difference: ACC maintains fixed-size state while
transcript replay grows unbounded.

Run: uv run python examples/advanced/acc_comparison.py
"""

import asyncio

from dotenv import load_dotenv

from cogent import Agent
from cogent.memory import Memory
from cogent.memory.acc import AgentCognitiveCompressor

load_dotenv()

# Simulate a long conversation
CONVERSATION = [
    "My name is Alice and I'm a software engineer",
    "I prefer dark mode and vim keybindings",
    "My favorite language is Python",
    "I work at TechCorp on the payments team",
    "Our main product is a payment gateway API",
    "We use PostgreSQL and Redis for storage",
    "I need help refactoring our auth module",
    "The current code uses JWT tokens",
    "We want to add OAuth2 support",
    "Security is critical - we handle credit cards",
]


async def run_with_transcript_replay():
    """Memory stores FULL transcript - grows with every turn."""
    print("\n" + "=" * 60)
    print("📜 TRANSCRIPT REPLAY (Memory without ACC)")
    print("=" * 60)

    memory = Memory()
    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. Be very brief.",
        memory=memory,
    )

    # Run conversation
    for i, msg in enumerate(CONVERSATION, 1):
        await agent.run(msg, thread_id="transcript")
        history = await memory.get_messages("transcript")
        print(f"Turn {i:2d}: {len(history):3d} messages stored")

    # Final question
    result = await agent.run("What's my name, company, and what am I working on?", thread_id="transcript")
    history = await memory.get_messages("transcript")

    print(f"\n📊 Final: {len(history)} messages (grows forever)")
    print(f"💬 Answer: {result.content}")
    return len(history)


async def run_with_acc():
    """ACC stores bounded state - never exceeds limits."""
    print("\n" + "=" * 60)
    print("🧠 ACC (Bounded Compression)")
    print("=" * 60)

    acc = AgentCognitiveCompressor(
        max_constraints=10,
        max_entities=20,
        max_actions=10,
        max_context=10,
        extraction_mode="model",
        model="gpt-4o-mini",
    )
    memory = Memory(acc=acc)
    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. Be very brief.",
        memory=memory,
    )

    # Run conversation
    for i, msg in enumerate(CONVERSATION, 1):
        await agent.run(msg, thread_id="acc")
        total = acc.state.total_items
        print(f"Turn {i:2d}: {total:3d} items stored (max: 50)")

    # Final question
    result = await agent.run("What's my name, company, and what am I working on?", thread_id="acc")

    print(f"\n📊 Final: {acc.state.total_items} items (bounded at 50 max)")
    print(f"   Entities: {[e.content[:50] for e in acc.state.entities[:3]]}")
    print(f"💬 Answer: {result.content}")
    return acc.state.total_items


async def main():
    print("🔬 ACC vs Transcript Replay - Actual Comparison")
    print("=" * 60)
    print(f"Running {len(CONVERSATION)} turns of conversation...\n")

    transcript_size = await run_with_transcript_replay()
    acc_size = await run_with_acc()

    print("\n" + "=" * 60)
    print("📊 RESULT")
    print("=" * 60)
    print(f"""
After {len(CONVERSATION)} turns:
  • Transcript: {transcript_size} messages (and growing)
  • ACC:        {acc_size} items (bounded at 50)

After 100 turns:
  • Transcript: ~200+ messages → may hit token limits
  • ACC:        ~50 items max → always fits
""")


if __name__ == "__main__":
    asyncio.run(main())
