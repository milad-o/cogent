"""
Demo: Observer v2 - See Your Agents in Action

Observer lets you watch what your agents are doing in real-time:
- See when agents start thinking
- Watch tool calls and results
- Track completion with timing and token counts

Usage:
    uv run python examples/observability/observer.py
"""

import asyncio

from cogent import Agent
from cogent.observability import Observer
from cogent.tools import tool


# Simple tools for the demo
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Fake weather data
    weather = {
        "new york": "72°F, Sunny",
        "london": "58°F, Cloudy",
        "tokyo": "68°F, Clear",
    }
    return weather.get(city.lower(), f"Weather data not available for {city}")


async def demo_progress():
    """Progress level - see agent activity."""
    print("\n" + "=" * 60)
    print("1. PROGRESS LEVEL - See Agent Activity")
    print("=" * 60)

    observer = Observer(level="progress")

    agent = Agent(
        name="Assistant",
        model="gpt-4o-mini",
        tools=[calculate, get_weather],
        observer=observer,
    )

    await agent.run("What's 25 * 4? And what's the weather in Tokyo?")

    print("\n" + observer.summary())


async def demo_verbose():
    """Verbose level - see more details."""
    print("\n" + "=" * 60)
    print("2. VERBOSE LEVEL - More Details")
    print("=" * 60)

    observer = Observer(level="verbose")

    agent = Agent(
        name="Analyst",
        model="gpt-4o-mini",
        tools=[calculate],
        observer=observer,
    )

    await agent.run("Calculate 100 / 4 + 50")

    print("\n" + observer.summary())


async def demo_minimal():
    """Minimal level - just results."""
    print("\n" + "=" * 60)
    print("3. MINIMAL LEVEL - Just Results")
    print("=" * 60)

    observer = Observer(level="minimal")

    agent = Agent(
        name="Helper",
        model="gpt-4o-mini",
        observer=observer,
    )

    await agent.run("Say hello in 3 words.")

    print("\n" + observer.summary())


async def demo_multi_agent():
    """Watch multiple agents work."""
    print("\n" + "=" * 60)
    print("4. MULTIPLE AGENTS")
    print("=" * 60)

    observer = Observer(level="progress")

    researcher = Agent(
        name="Researcher",
        model="gpt-4o-mini",
        tools=[get_weather],
        observer=observer,
    )

    writer = Agent(
        name="Writer",
        model="gpt-4o-mini",
        observer=observer,
    )

    # Run sequentially
    weather_info = await researcher.run("Get weather for London and New York")
    await writer.run(f"Write a short comparison: {weather_info.content}")

    print("\n" + observer.summary())


async def main():
    print("\n" + "=" * 60)
    print("  🔭 Observer v2 - Watch Your Agents Work")
    print("=" * 60)

    await demo_progress()
    await demo_verbose()
    await demo_minimal()
    await demo_multi_agent()

    print("\n" + "=" * 60)
    print("✅ Observer: Simple, Real-time Agent Visibility")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
