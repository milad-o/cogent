#!/usr/bin/env python3
"""
Streaming Responses
===================

Demonstrates real-time token-by-token streaming from agents.

With streaming enabled, tokens arrive as they're generated, providing:
- Real-time feedback during agent processing
- Better UX with progressive output
- Lower perceived latency

Key Features:
1. Basic streaming - see tokens as they arrive
2. Streaming with tools - watch tool calls and responses
3. Streaming with conversation memory

Prerequisites:
    - Streaming-capable model (OpenAI, Anthropic, Grok, etc.)
    - Set API key: export OPENAI_API_KEY=your-key

Run:
    uv run python examples/basics/streaming.py
"""

import asyncio

from cogent import Agent
from cogent.tools import tool


# =============================================================================
# Demo 1: Basic Streaming
# =============================================================================

async def basic_streaming():
    """Demo 1: Basic streaming from an agent."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic Streaming")
    print("=" * 70)

    agent = Agent(
        name="Assistant",
        model="gpt4",
        instructions="You are a helpful assistant. Be concise.",
    )

    print("\n📝 Task: Explain streaming in 2 sentences")
    print("-" * 70)
    print()

    # Stream execution - tokens arrive in real-time
    async for chunk in agent.run("Explain streaming in 2 sentences", stream=True):
        print(chunk.content, end="", flush=True)

    print("\n")
    print("✅ Streaming complete!")


# =============================================================================
# Demo 2: Streaming with Tools
# =============================================================================

async def streaming_with_tools():
    """Demo 2: Streaming while using tools."""
    print("\n" + "=" * 70)
    print("Demo 2: Streaming with Tools")
    print("=" * 70)

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        # Simulated weather data
        weather_data = {
            "london": "Cloudy, 12°C",
            "tokyo": "Sunny, 22°C",
            "new york": "Rainy, 15°C",
        }
        return weather_data.get(city.lower(), f"Weather data unavailable for {city}")

    @tool
    def get_time(city: str) -> str:
        """Get the current time in a city."""
        from datetime import datetime, timezone
        # Simplified - just return current UTC time
        return datetime.now(timezone.utc).strftime("%H:%M UTC")

    agent = Agent(
        name="WeatherBot",
        model="gpt4",
        tools=[get_weather, get_time],
        instructions="You help with weather and time queries. Be brief.",
    )

    print("\n📝 Task: What's the weather and time in Tokyo?")
    print("-" * 70)
    print()

    async for chunk in agent.run("What's the weather and time in Tokyo?", stream=True):
        print(chunk.content, end="", flush=True)

    print("\n")
    print("✅ Tool streaming complete!")


# =============================================================================
# Demo 3: Streaming with Conversation
# =============================================================================

async def streaming_with_conversation():
    """Demo 3: Streaming with conversation memory."""
    print("\n" + "=" * 70)
    print("Demo 3: Streaming with Conversation Memory")
    print("=" * 70)

    agent = Agent(
        name="ChatBot",
        model="gpt4",
        instructions="You are a friendly assistant. Remember what the user tells you.",
    )

    thread_id = "demo-conversation"

    # First message
    print("\n📝 User: My name is Alice and I love Python.")
    print("-" * 70)
    print()

    async for chunk in agent.run(
        "My name is Alice and I love Python.",
        stream=True,
        thread_id=thread_id,
    ):
        print(chunk.content, end="", flush=True)

    print("\n")

    # Second message - agent should remember
    print("📝 User: What's my name and what do I like?")
    print("-" * 70)
    print()

    async for chunk in agent.run(
        "What's my name and what do I like?",
        stream=True,
        thread_id=thread_id,
    ):
        print(chunk.content, end="", flush=True)

    print("\n")
    print("✅ Conversation streaming complete!")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all streaming demos."""
    print("=" * 70)
    print("        COGENT STREAMING EXAMPLES")
    print("=" * 70)

    await basic_streaming()
    await streaming_with_tools()
    await streaming_with_conversation()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

