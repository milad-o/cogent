"""
Demo: Basic Usage

Single agent with instructions performing a task.

Usage:
    uv run python examples/01_basic_usage.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow, FlowObserver
from agenticflow.models import ChatModel

load_dotenv()


async def main():
    model = ChatModel(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    assistant = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant. Be concise.",
    )

    flow = Flow(
        name="basic",
        agents=[assistant],
        topology="pipeline",
        observer=FlowObserver.verbose(),
    )

    result = await flow.run("What is 2 + 2? Explain briefly.")
    print(f"\nResult: {result.results[-1]['thought']}")


if __name__ == "__main__":
    asyncio.run(main())
