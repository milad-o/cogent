"""
Demo: Pipeline Topology

Sequential workflow: Analyst → Writer → Editor

Usage:
    uv run python examples/02_topologies.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow, FlowObserver
from agenticflow.models import ChatModel

load_dotenv()


async def main():
    model = ChatModel(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    analyst = Agent(name="Analyst", model=model, instructions="Analyze the topic briefly.")
    writer = Agent(name="Writer", model=model, instructions="Write a short summary.")
    editor = Agent(name="Editor", model=model, instructions="Polish and finalize.")

    flow = Flow(
        name="pipeline",
        agents=[analyst, writer, editor],
        topology="pipeline",
        observer=FlowObserver.verbose(),
    )

    result = await flow.run("Topic: Benefits of exercise")
    print(f"\nFinal: {result.results[-1]['thought']}")


if __name__ == "__main__":
    asyncio.run(main())
