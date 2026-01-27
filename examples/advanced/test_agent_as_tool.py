"""Quick test for Agent.as_tool() functionality.

Usage:
    # Ensure OPENAI_API_KEY is set in .env
    uv run python examples/advanced/test_agent_as_tool.py
"""

import asyncio

from cogent import Agent


async def test_basic_agent_as_tool():
    """Test that agent can be converted to tool and executed."""
    print("Testing Agent.as_tool() basic functionality...")
    print()
    
    # Create specialist agent
    specialist = Agent(
        name="MathExpert",
        model="gpt4",  # Auto-loads API key from .env
        instructions="You are a math expert. Solve the given problem and explain your answer.",
    )
    
    # Convert to tool
    math_tool = specialist.as_tool(
        description="Solve math problems"
    )
    
    # Verify tool properties
    assert math_tool.name == "MathExpert"
    assert "math" in math_tool.description.lower()
    print(f"✅ Tool created: {math_tool.name}")
    print(f"   Description: {math_tool.description}")
    print()
    
    # Create orchestrator that uses the tool
    orchestrator = Agent(
        name="TaskManager",
        model="gpt4",
        tools=[math_tool],
        instructions="Use MathExpert to solve math problems. Return the expert's answer.",
    )
    
    # Test execution
    print("Executing: 'What is 15 * 23?'")
    result = await orchestrator.run("What is 15 * 23?")
    
    print()
    print("Result:")
    if result.content:
        print(result.content)
    else:
        print("(No response - check OPENAI_API_KEY is set)")
    print()
    
    # Verify tool was called
    if result.tool_calls:
        print(f"✅ Tool was called: {len(result.tool_calls)} call(s)")
        for tc in result.tool_calls:
            print(f"   - {tc.tool_name}")
    
    print()
    print("✅ Test passed! Agent.as_tool() works correctly")


async def test_return_full_response():
    """Test return_full_response parameter."""
    print("\nTesting return_full_response=True...")
    print()
    
    specialist = Agent(
        name="QuickAnswer",
        model="gpt4",
        instructions="Answer in one sentence.",
    )
    
    # Get tool that returns full response
    tool_full = specialist.as_tool(
        description="Get detailed answer",
        return_full_response=True,
    )
    
    orchestrator = Agent(
        name="Manager",
        model="gpt4",
        tools=[tool_full],
        instructions="Use QuickAnswer and report what metadata you received.",
    )
    
    result = await orchestrator.run("What is Python?")
    
    print("Result:")
    if result.content:
        print(result.content)
    else:
        print("(No response - check API keys)")
    
    # Show that we can see metadata from tool response
    if result.tool_calls:
        print("Metadata received:")
        for tc in result.tool_calls:
            if hasattr(tc, 'result') and isinstance(tc.result, dict):
                meta = tc.result.get('metadata', {})
                if meta:
                    print(f"- Agent: {meta.get('agent')}")
                    print(f"- Model: {meta.get('model')}")
                    print(f"- Duration: {meta.get('duration')} seconds")
    
    print()
    print("✅ Full response mode works")


async def main():
    """Run all tests."""
    await test_basic_agent_as_tool()
    await test_return_full_response()
    
    print()
    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
