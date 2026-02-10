"""Simple subagent delegation example.

Demonstrates the basics of delegating tasks to specialist agents.

Run:
    uv run python examples/basics/simple_delegation.py
"""

import asyncio

from cogent import Agent, Observer


async def main():
    """Simple coordinator with one specialist subagent."""

    observer = Observer(level="progress")

    # Create a specialist agent
    math_expert = Agent(
        name="math_expert",
        model="gemini:gemini-2.5-flash",
        instructions="You are a mathematics expert. Solve math problems step-by-step.",
    )

    # Create coordinator that can delegate to the specialist
    assistant = Agent(
        name="assistant",
        model="gemini:gemini-2.5-flash",
        instructions="""You are a helpful assistant.
For math problems, delegate to the math_expert.
For other questions, answer directly.""",
        # Simply pass the agent - uses its name as the tool name
        subagents=[math_expert],
        observer=observer,
    )

    # Test: Math problem (should delegate)
    print("=" * 60)
    print("TASK: Solve a math problem")
    print("=" * 60)

    response = await assistant.run(
        "If I have 15 apples and give away 40% of them, how many do I have left?"
    )

    print(f"\nResponse: {response.content}")
    print(f"\nTokens used: {response.metadata.tokens.total_tokens}")

    if response.subagent_responses:
        print(f"Delegated to: {response.subagent_responses[0].metadata.agent}")
        print(
            f"Subagent tokens: {response.subagent_responses[0].metadata.tokens.total_tokens}"
        )
    else:
        print("No delegation (answered directly)")

    # Test: Simple question (should answer directly)
    print("\n" + "=" * 60)
    print("TASK: Simple question")
    print("=" * 60)

    response = await assistant.run("What is the capital of France?")

    print(f"\nResponse: {response.content}")
    print(f"\nTokens used: {response.metadata.tokens.total_tokens}")

    if response.subagent_responses:
        print("Delegated (unexpected!)")
    else:
        print("Answered directly (as expected)")


if __name__ == "__main__":
    asyncio.run(main())
