"""
Demo: Document Processor with Logging

Pipeline with event hooks for observability.

Usage:
    uv run python examples/04_events.py
"""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from agenticflow import Agent, Flow, TopologyPattern, EventHooks, pipeline_topology

load_dotenv()


@tool
def extract_entities(text: str) -> str:
    """Extract key entities from text."""
    return "Entities: [Company: Acme Corp, Amount: $1.2M, Date: 2025-01-15]"


@tool
def classify_document(entities: str) -> str:
    """Classify the document type."""
    return "Type: Financial Report - Q4 Earnings"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


async def main():
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))

    extractor = Agent(name="Extractor", model=model, tools=[extract_entities])
    classifier = Agent(name="Classifier", model=model, tools=[classify_document])

    # Create topology with hooks for logging
    pipeline_topology(
        name="doc-processor",
        stages=[extractor, classifier],
        hooks=EventHooks(
            on_handoff=lambda f, t, d: log(f"Handoff: {f} → {t}"),
            on_complete=lambda a, r: log(f"Complete: {a}"),
            on_error=lambda a, e: log(f"Error in {a}: {e}"),
        ),
    )

    flow = Flow(
        name="doc-processor",
        agents=[extractor, classifier],
        topology=TopologyPattern.PIPELINE,
    )

    result = await flow.run("Process this quarterly financial report from Acme Corp")
    print(f"\nResult: {result['results'][-1]['thought']}")


if __name__ == "__main__":
    asyncio.run(main())
