"""
Demo: Content Pipeline

Research → Write → Edit sequential workflow.

Usage:
    uv run python examples/02_topologies.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from agenticflow import Agent, Flow, TopologyPattern

load_dotenv()


@tool
def research(topic: str) -> str:
    """Gather information on a topic."""
    return f"Research findings on {topic}: Key facts and recent developments compiled."


@tool
def write_draft(research: str) -> str:
    """Write initial draft from research."""
    return f"Draft article based on: {research[:50]}..."


@tool
def edit(draft: str) -> str:
    """Edit and polish the draft."""
    return f"Edited version: {draft[:50]}... [Grammar fixed, flow improved]"


async def main():
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    researcher = Agent(name="Researcher", model=model, tools=[research])
    writer = Agent(name="Writer", model=model, tools=[write_draft])
    editor = Agent(name="Editor", model=model, tools=[edit])

    flow = Flow(
        name="content-pipeline",
        agents=[researcher, writer, editor],
        topology=TopologyPattern.PIPELINE,
    )

    async for state in flow.stream("Write an article about sustainable energy"):
        agent = state.get("current_agent", "starting")
        print(f"[{agent}] working...")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
