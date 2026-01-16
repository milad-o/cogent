"""
Demo: Basic Usage

Single agent with instructions performing a task.

Usage:
    uv run python examples/basics/hello_world.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model, settings

from agenticflow import Agent, Observer, pipeline


async def main():
    model = get_model()  # Automatically uses first available provider

    assistant = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant. Be concise.",
    )

    # Simple flow with pipeline pattern
    # Pipeline runs agents sequentially
    flow = pipeline([assistant])

    result = await flow.run("What is 2 + 2? Explain briefly.")
    print(f"\nResult: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
