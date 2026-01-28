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
from cogent.memory.acc import AgentCognitiveCompressor, BoundedMemoryState
from cogent.observability import Observer


async def demo_basic_acc():
    """Basic ACC usage with an agent."""
    print("\n--- Basic ACC with Agent ---")
    
    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        bounded_memory=True,  # Enable ACC
        observer=Observer.trace(),
    )

    # Simulate conversation that would normally cause drift
    await agent.run("My name is Alice and I work at TechCorp.", thread_id="conv-1")
    await agent.run("I prefer Python for backend development.", thread_id="conv-1")
    await agent.run("What's my name and where do I work?", thread_id="conv-1")
    
    print("✓ ACC maintains bounded context across turns")


async def demo_acc_internals():
    """Show how ACC works internally."""
    print("\n--- ACC Internals ---")
    
    # Create bounded state manually
    state = BoundedMemoryState(
        max_constraints=5,
        max_entities=10,
        max_actions=5,
        max_context=5,
    )
    
    # Create compressor
    acc = AgentCognitiveCompressor(state=state)
    
    # Update from conversation turns
    await acc.update_from_turn(
        user_message="Remember: I need responses in JSON format",
        assistant_message="I'll format all responses as JSON.",
        tool_calls=[],
        current_task="API development help",
    )
    
    await acc.update_from_turn(
        user_message="My API endpoint is /users",
        assistant_message="Got it, working with /users endpoint.",
        tool_calls=[],
        current_task="API development help",
    )
    
    # Check state
    print(f"Constraints: {len(state.constraints)}")
    print(f"Entities:    {len(state.entities)}")
    print(f"Actions:     {len(state.actions)}")
    print(f"Utilization: {state.utilization:.0%}")
    
    # Format for LLM prompt
    context = acc.format_for_prompt("Create a GET /users endpoint")
    print(f"\nFormatted context ({len(context)} chars):")
    print(context[:200] + "..." if len(context) > 200 else context)


async def demo_drift_prevention():
    """Show how ACC prevents drift in long conversations."""
    print("\n--- Drift Prevention ---")
    
    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        bounded_memory=True,
        observer=Observer.trace(),
    )

    # Initial constraint
    await agent.run(
        "Important: All code must be Python 3.12+ with type hints.",
        thread_id="long-conv",
    )
    
    # Many turns later... ACC keeps the constraint
    for i in range(5):
        await agent.run(f"Question {i+1}: How do I handle errors?", thread_id="long-conv")
    
    # Verify constraint still present
    result = await agent.run(
        "Write a function to read a file. Remember my requirements!",
        thread_id="long-conv",
    )
    
    print(f"\nResult (should have Python 3.12+ type hints):\n{result}")


async def main():
    print("=" * 50)
    print("AGENTIC CONTEXT COMPRESSION (ACC)")
    print("=" * 50)
    
    await demo_basic_acc()
    await demo_acc_internals()
    await demo_drift_prevention()
    
    print("\n✓ Done")


if __name__ == "__main__":
    asyncio.run(main())
