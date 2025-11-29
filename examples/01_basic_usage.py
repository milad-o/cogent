"""
Demo: Basic Usage

Single agent with instructions performing a task.

Usage:
    uv run python examples/01_basic_usage.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow
from agenticflow.models.gemini import GeminiChat

load_dotenv()


async def main():
    model = GeminiChat(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

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
        verbose=True,  # Simple progress output
    )

    result = await flow.run("What is 2 + 2? Explain briefly.")
    print(f"\nResult: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
