"""
Demo: Response Metadata in Observer

Shows how Observer automatically displays Response[T] metadata:
- Token usage
- Tool call tracking  
- Duration

Usage:
    uv run python examples/observability/response_metadata.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent, Observer, tool

@tool
def calculator(expression: str) -> float:
    """Calculate a mathematical expression.
    
    Args:
        expression: A Python math expression like '2+2' or '15*8'
        
    Returns:
        The result of the calculation
    """
    return eval(expression)

@tool
def get_weather(city: str) -> str:
    """Get weather information for a city.
    
    Args:
        city: Name of the city
        
    Returns:
        Weather description
    """
    # Mock weather data
    return f"Sunny and 72°F in {city}"


async def demo_basic_response():
    """Basic response with metadata."""
    print("\n=== Basic Response ===")
    
    model = get_model()
    agent = Agent(
        name="Assistant",
        model=model,
        verbose=True,  # Observer shows metadata
    )
    
    response = await agent.run("What is 2+2?")
    
    print(f"\nContent: {response.content or '(empty response from model)'}")
    if response.metadata.tokens:
        print(f"Tokens: {response.metadata.tokens.total_tokens}")


async def demo_tool_tracking():
    """Response with tool calls."""
    print("\n=== Tool Call Tracking ===")
    
    model = get_model()
    agent = Agent(
        name="MathBot",
        model=model,
        tools=[calculator],
        verbose=True,
    )
    
    response = await agent.run("Calculate 123 * 456")
    
    print(f"\nTool calls: {len(response.tool_calls)}")
    for tc in response.tool_calls:
        print(f"  - {tc.tool_name} ({tc.duration:.3f}s)")


async def demo_observer_integration():
    """Observer with Response metadata."""
    print("\n=== Observer Integration ===")
    
    model = get_model()
    observer = Observer.verbose()
    
    agent = Agent(
        name="Worker",
        model=model,
        tools=[calculator],
        observer=observer,
    )
    
    response = await agent.run("Calculate 999 * 888")
    
    print(f"\nTotal events tracked: {len(observer.events())}")


async def main():
    """Run demos."""
    await demo_basic_response()
    await demo_tool_tracking()
    await demo_observer_integration()


if __name__ == "__main__":
    asyncio.run(main())
