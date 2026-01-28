"""ACC Comparison - Heuristic vs Model Extraction.

Demonstrates the importance of ACC (Agentic Context Compression) and
compares heuristic vs model-based extraction modes.

Run: uv run python examples/advanced/acc_comparison.py
"""

import asyncio

from dotenv import load_dotenv

from cogent import Agent
from cogent.memory import Memory
from cogent.memory.acc import AgentCognitiveCompressor

load_dotenv()


async def run_without_acc():
    """Run agent WITHOUT ACC - no context between turns."""
    print("\n" + "=" * 60)
    print("❌ WITHOUT ACC (No memory between turns)")
    print("=" * 60)

    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. Be concise.",
    )

    # Turn 1
    r1 = await agent.run("My name is Alice and I prefer dark mode")
    print(f"\nTurn 1: {r1}")

    # Turn 2 - Agent has NO memory of turn 1!
    r2 = await agent.run("What's my name and what do I prefer?")
    print(f"Turn 2: {r2}")


async def run_with_acc_heuristic():
    """Run agent WITH ACC (heuristic mode) - fast, rule-based extraction."""
    print("\n" + "=" * 60)
    print("✅ WITH ACC - Heuristic Mode (Fast, Rule-based)")
    print("=" * 60)

    acc = AgentCognitiveCompressor(
        max_constraints=5,
        max_entities=10,
        max_actions=5,
        max_context=5,
        extraction_mode="heuristic",  # Fast, rule-based
    )
    memory = Memory(acc=acc)

    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. Be concise.",
        memory=memory,
    )

    # Turn 1
    r1 = await agent.run("My name is Alice and I prefer dark mode", thread_id="session-h")
    print(f"\nTurn 1: {r1}")
    print(f"   ACC State: {len(acc.state.entities)} entities, {len(acc.state.context)} context")

    # Turn 2 - ACC remembers context!
    r2 = await agent.run("What's my name and what do I prefer?", thread_id="session-h")
    print(f"Turn 2: {r2}")
    print(f"   ACC State: {len(acc.state.entities)} entities, {len(acc.state.context)} context")

    # Show what was extracted
    print("\n📋 Extracted Entities (heuristic):")
    for item in acc.state.entities[:3]:
        print(f"   • {item.content[:80]}...")

    return acc


async def run_with_acc_model():
    """Run agent WITH ACC (model mode) - LLM-based semantic extraction."""
    print("\n" + "=" * 60)
    print("✅ WITH ACC - Model Mode (LLM-based Semantic Extraction)")
    print("=" * 60)

    acc = AgentCognitiveCompressor(
        max_constraints=5,
        max_entities=10,
        max_actions=5,
        max_context=5,
        extraction_mode="model",      # LLM-based extraction
        model="gpt-4o-mini",          # Efficient model
    )
    memory = Memory(acc=acc)

    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. Be concise.",
        memory=memory,
    )

    # Turn 1
    r1 = await agent.run("My name is Alice and I prefer dark mode", thread_id="session-m")
    print(f"\nTurn 1: {r1}")
    print(f"   ACC State: {len(acc.state.entities)} entities, {len(acc.state.context)} context")

    # Turn 2 - ACC remembers context!
    r2 = await agent.run("What's my name and what do I prefer?", thread_id="session-m")
    print(f"Turn 2: {r2}")
    print(f"   ACC State: {len(acc.state.entities)} entities, {len(acc.state.context)} context")

    # Show what was extracted
    print("\n📋 Extracted Entities (model):")
    for item in acc.state.entities[:3]:
        print(f"   • {item.content[:80]}")

    return acc


async def main():
    print("🧠 ACC (Agentic Context Compression) Comparison")
    print("=" * 60)
    print("""
ACC maintains bounded internal state instead of unbounded transcript replay.
This prevents context drift, memory poisoning, and context overflow.

Extraction Modes:
  • heuristic - Fast, rule-based (keyword matching, capitalization)
  • model - LLM-based semantic extraction (better quality, slower)
""")

    # 1. Without ACC - agent forgets between turns
    await run_without_acc()

    # 2. With ACC (heuristic) - fast extraction
    acc_h = await run_with_acc_heuristic()

    # 3. With ACC (model) - semantic extraction
    acc_m = await run_with_acc_model()

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"""
| Mode       | Entities | Context | Quality | Speed  |
|------------|----------|---------|---------|--------|
| No ACC     | 0        | 0       | N/A     | N/A    |
| Heuristic  | {len(acc_h.state.entities):<8} | {len(acc_h.state.context):<7} | Good    | ⚡ Fast |
| Model      | {len(acc_m.state.entities):<8} | {len(acc_m.state.context):<7} | Better  | Slower |

Key Benefits of ACC:
  ✅ Bounded memory - ~110 items max regardless of conversation length
  ✅ Prevents context drift - constraints persist across turns
  ✅ Prevents memory poisoning - verified artifacts only
  ✅ Semantic compression - keeps what matters, forgets noise
""")


if __name__ == "__main__":
    asyncio.run(main())
