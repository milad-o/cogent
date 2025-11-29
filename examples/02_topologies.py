"""
Demo: Pipeline Topology

Sequential workflow: Analyst → Writer → Editor
With detailed observability showing agent thoughts.

Usage:
    uv run python examples/02_topologies.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow, FlowObserver
from agenticflow.models.gemini import GeminiChat

load_dotenv()


async def main():
    model = GeminiChat(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

    analyst = Agent(name="Analyst", model=model, instructions="Analyze the topic briefly. Keep response under 50 words.")
    writer = Agent(name="Writer", model=model, instructions="Write a short summary. Keep response under 50 words.")
    editor = Agent(name="Editor", model=model, instructions="Polish and finalize. Keep response under 50 words.")

    flow = Flow(
        name="pipeline",
        agents=[analyst, writer, editor],
        topology="pipeline",
        observer=FlowObserver.verbose(),
    )

    result = await flow.run("Topic: Benefits of exercise")
    print(f"\n{'='*60}")
    print("FINAL OUTPUT:")
    print("="*60)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
