"""Agentic Context Compression (ACC) - Bounded memory for long conversations.

Based on arXiv:2601.11653 - Prevents memory poisoning, drift, and context overflow.

ACC maintains bounded internal state instead of unbounded transcript replay:
- Constraints (10): Goals, rules, requirements
- Entities (50): Facts, knowledge, data
- Actions (30): What worked/failed
- Context (20): Relevant snippets

Total: ~110 items regardless of conversation length.
"""

import asyncio

from cogent import Agent
from cogent.memory.acc import AgentCognitiveCompressor
from cogent.observability import Observer


async def demo_basic_acc():
    """Basic ACC usage with an agent."""
    print("\n--- Basic ACC with Agent ---")

    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        acc=True,  # Enable ACC
        observer=Observer(level="trace"),
    )

    # Simulate conversation that would normally cause drift
    await agent.run("My name is Alice and I work at TechCorp.", thread_id="conv-1")
    await agent.run("I prefer Python for backend development.", thread_id="conv-1")
    await agent.run("What's my name and where do I work?", thread_id="conv-1")

    print("✓ ACC maintains bounded context across turns")


async def demo_custom_acc():
    """Pass a custom ACC instance with specific bounds."""
    print("\n--- Custom ACC Instance ---")

    # Create custom ACC with smaller bounds (state is internal)
    acc = AgentCognitiveCompressor(
        max_constraints=5,
        max_entities=20,
        max_actions=10,
        max_context=10,
    )

    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        acc=acc,  # Pass custom ACC instance
        observer=Observer(level="trace"),
    )

    await agent.run("Remember: Always respond in JSON format.", thread_id="custom")
    await agent.run("What format should responses be in?", thread_id="custom")

    print(
        f"ACC state - Constraints: {len(acc.state.constraints)}, Entities: {len(acc.state.entities)}"
    )
    print("✓ Custom ACC bounds applied")


async def demo_drift_prevention():
    """Show how ACC prevents drift in long conversations."""
    print("\n--- Drift Prevention ---")

    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        acc=True,
        observer=Observer(level="trace"),
    )

    # Initial constraint
    await agent.run(
        "Important: All code must be Python 3.12+ with type hints.",
        thread_id="long-conv",
    )

    # Many turns later... ACC keeps the constraint
    for i in range(3):
        await agent.run(
            f"Question {i + 1}: How do I handle errors?", thread_id="long-conv"
        )

    # Verify constraint still present
    result = await agent.run(
        "Write a function to read a file. Remember my requirements!",
        thread_id="long-conv",
    )

    print(f"\nResult (should have type hints):\n{str(result)[:300]}...")


async def main():
    print("=" * 50)
    print("AGENTIC CONTEXT COMPRESSION (ACC)")
    print("=" * 50)

    await demo_basic_acc()
    await demo_custom_acc()
    await demo_drift_prevention()

    print("\n✓ Done")


if __name__ == "__main__":
    asyncio.run(main())
