"""
Demo: Basic Usage

Single agent with instructions performing a task.

Usage:
    uv run python examples/01_basic_usage.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model, settings

from agenticflow import Agent, Flow


async def main():
    model = get_model()  # Automatically uses first available provider

    assistant = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant. Be concise.",
    )

    # Simple observability with verbose parameter
    # Options: True (progress), "verbose" (outputs), "debug" (tools), "trace" (all)
    flow = Flow(
        name="basic",
        agents=[assistant],
        topology="pipeline",
        verbose=settings.verbose_level,
    )

    result = await flow.run("What is 2 + 2? Explain briefly.")
    print(f"\nResult: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
