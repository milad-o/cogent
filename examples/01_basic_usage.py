"""
Demo: Research Assistant

A simple agent that researches topics and provides summaries.

Usage:
    uv run python examples/01_basic_usage.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from agenticflow import Agent, Flow, TopologyPattern

load_dotenv()


@tool
def search(query: str) -> str:
    """Search for information on a topic."""
    # In production, connect to a real search API
    return f"Found: {query} is a rapidly evolving field with recent advances in efficiency and accessibility."


@tool
def summarize(text: str) -> str:
    """Summarize text into key points."""
    return f"Summary: {text[:100]}... [Key takeaways extracted]"


async def main():
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    assistant = Agent(
        name="ResearchAssistant",
        model=model,
        tools=[search, summarize],
        instructions="You are a research assistant. Search for information, then summarize the key points concisely.",
    )

    flow = Flow(
        name="research",
        agents=[assistant],
        topology=TopologyPattern.PIPELINE,
    )

    result = await flow.run("What are the latest developments in quantum computing?")
    print(result["results"][-1]["thought"])


if __name__ == "__main__":
    asyncio.run(main())
