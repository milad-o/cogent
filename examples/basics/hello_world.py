"""
Demo: Basic Usage

Single agent with instructions performing a task.

Usage:
    # Ensure OPENAI_API_KEY is set in .env file at project root
    uv run python examples/basics/hello_world.py
"""

import asyncio

from cogent import Agent


async def main():
    # Simple string model - API key loaded automatically from .env
    assistant = Agent(
        name="Assistant",
        model="xai:grok-4-1-fast-non-reasoning",
        instructions="You are a helpful assistant. Be concise.",
    )

    # Simple agent call
    result = await assistant.run(
        "Name three European countries and a short description for each.",
    )
    print(f"\n{result.content}")


if __name__ == "__main__":
    asyncio.run(main())
