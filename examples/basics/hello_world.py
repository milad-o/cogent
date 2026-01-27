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
        model="gpt4",  # or "gemini", "claude", etc.
        instructions="You are a helpful assistant. Be concise.",
    )

    # Simple agent call
    result = await assistant.run(
        "Name three European countries with their capitals.",
    )
    print(f"\n{result.content}")


if __name__ == "__main__":
    asyncio.run(main())
