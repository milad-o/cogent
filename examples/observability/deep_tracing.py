#!/usr/bin/env python3
"""
Deep Observability - Full LLM transparency.

Shows all observability features: LLM requests/responses, tool calls, timing.

Run:
    uv run python examples/observability/deep_tracing.py
"""

import asyncio

from cogent import Agent
from cogent.observability import Channel, ObservabilityLevel, Observer
from cogent.tools import tool


# Define some simple tools
@tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    """Get weather for a city."""
    # Mock weather data
    weather_data = {
        "london": "Cloudy, 15°C, 60% humidity",
        "paris": "Sunny, 22°C, 45% humidity",
        "tokyo": "Rainy, 18°C, 85% humidity",
        "new york": "Clear, 25°C, 40% humidity",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool(description="Convert temperature between Celsius and Fahrenheit")
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between units."""
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = (value * 9 / 5) + 32
        return f"{value}°C = {result:.1f}°F"
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.1f}°C"
    else:
        return f"Cannot convert from {from_unit} to {to_unit}"


async def main():
    # Create observer with DEBUG level and LLM channel for full transparency
    observer = Observer(
        level=ObservabilityLevel.DEBUG,
        channels=[Channel.AGENTS, Channel.TOOLS, Channel.TASKS, Channel.LLM],
        show_timestamps=True,
        truncate=300,
    )

    # Create agent with tools
    agent = Agent(
        name="WeatherAssistant",
        model="gpt4",
        system_prompt="You are a helpful weather assistant. Use tools to help with weather questions.",
        tools=[get_weather, convert_temperature],
        observer=observer,
    )

    print("=" * 60)
    print("Deep Observability Demo")
    print("=" * 60)
    print(f"\nObserver Level: {observer.config.level.name}")
    print("\nYou'll see LLM request/response, tool decisions, and executions")
    print("-" * 60)

    # Run query that uses tools
    print("\n🔍 Query: 'What's the weather in London and convert it to Fahrenheit?'\n")

    result = await agent.run("What's the weather in London? Convert the temperature to Fahrenheit.")

    print("-" * 60)
    print(f"\n✅ Final Result: {result.unwrap()}")

    # Show observer summary
    print("\n" + "=" * 60)
    print("Observer Summary:")
    print(observer.summary())


if __name__ == "__main__":
    asyncio.run(main())
