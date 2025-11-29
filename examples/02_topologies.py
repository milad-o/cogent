"""
Demo: Pipeline Topology

Sequential workflow: Analyst → Writer → Editor
With observability showing agent thoughts.

Usage:
    uv run python examples/02_topologies.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow
from agenticflow.models.gemini import GeminiChat

load_dotenv()


async def main():
    model = GeminiChat(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

    analyst = Agent(name="Analyst", model=model, instructions="Analyze the topic briefly. Keep response under 50 words.")
    writer = Agent(name="Writer", model=model, instructions="Write a short summary. Keep response under 50 words.")
    editor = Agent(name="Editor", model=model, instructions="Polish and finalize. Keep response under 50 words.")

    # Simple observability: verbose="verbose" shows agent outputs
    # Options: True (progress), "verbose" (outputs), "debug" (tools), "trace" (all)
    flow = Flow(
        name="pipeline",
        agents=[analyst, writer, editor],
        topology="pipeline",
        verbose="verbose",  # Simple! Shows agent thoughts with timing
    )

    result = await flow.run("Topic: Benefits of exercise")
    print(f"\n{'='*60}")
    print("FINAL OUTPUT:")
    print("="*60)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
