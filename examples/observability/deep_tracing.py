#!/usr/bin/env python3
"""
Example 27: Deep Observability
==============================

This example demonstrates the enhanced observability features that provide
full transparency into LLM interactions - no more black box!

What you'll see:
- LLM_REQUEST: Full prompt, messages, and available tools sent to the LLM
- LLM_RESPONSE: Raw LLM response with timing and token usage
- LLM_TOOL_DECISION: When LLM decides to use tools and which ones

Run with different observability levels:
- PROGRESS: Basic milestones (thinking, responding)
- DETAILED: + Tool calls and tool decisions
- DEBUG: + Full LLM request/response payloads
- TRACE: Everything including internal events
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model, settings

from agenticflow import Agent, Flow
from agenticflow.observability import Observer, ObservabilityLevel, Channel
from agenticflow.tools import tool


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
    # Get model from config
    model = get_model()
    print(f"Using LLM provider: {settings.llm_provider}")

    # Create an observer with DEBUG level AND explicit LLM channel opt-in
    # Note: Channel.LLM must be explicitly added to see LLM request/response details
    observer = Observer(
        level=ObservabilityLevel.DEBUG,
        channels=[Channel.AGENTS, Channel.TOOLS, Channel.TASKS, Channel.LLM],  # Opt-in to LLM content
        show_timestamps=True,
        truncate=300,  # Show more content
    )

    # Create an agent with tools
    agent = Agent(
        name="WeatherAssistant",
        model=model,
        system_prompt="You are a helpful weather assistant. Use the available tools to help users with weather-related questions.",
        tools=[get_weather, convert_temperature],
    )

    # Create flow with observer
    flow = Flow(name="deep_observability_demo", agents=[agent], observer=observer)

    print("=" * 60)
    print("Deep Observability Demo")
    print("=" * 60)
    print(f"\nObserver Level: {observer.config.level.name}")
    print("\nYou'll see:")
    print("  📤 LLM_REQUEST  - What's being sent to the LLM")
    print("  📥 LLM_RESPONSE - What the LLM returns")
    print("  🎯 TOOL_DECISION - When LLM decides to use tools")
    print("  🔧 TOOL_CALLED  - Actual tool executions")
    print("-" * 60)

    # Run a query that will use tools
    print("\n🔍 Query: 'What's the weather in London and convert it to Fahrenheit?'\n")

    result = await flow.run("What's the weather in London? Please convert the temperature to Fahrenheit.")

    print("-" * 60)
    print(f"\n✅ Final Result: {result}")

    # Show observer summary
    print("\n" + "=" * 60)
    print("Observer Summary:")
    print(observer.summary())


if __name__ == "__main__":
    asyncio.run(main())
