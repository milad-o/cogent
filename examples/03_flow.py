"""
Demo: Streaming Progress

Watch agents work in real-time using flow.stream().

Usage:
    uv run python examples/03_flow.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow
from agenticflow.models import ChatModel

load_dotenv()


async def main():
    model = ChatModel(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    researcher = Agent(name="Researcher", model=model, instructions="Research the topic.")
    writer = Agent(name="Writer", model=model, instructions="Write a summary.")
    reviewer = Agent(name="Reviewer", model=model, instructions="Review and finalize.")

    flow = Flow(
        name="content",
        agents=[researcher, writer, reviewer],
        topology="pipeline",
    )

    print("Streaming progress:")
    async for state in flow.stream("Topic: AI in healthcare"):
        agent = state.get("current_agent", "starting")
        print(f"  [{agent}] completed")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
