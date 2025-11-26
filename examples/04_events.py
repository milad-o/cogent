"""
Demo: Custom Callbacks

Use FlowObserver callbacks to track events programmatically.

Usage:
    uv run python examples/04_events.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agenticflow import Agent, Flow, FlowObserver, ObservabilityLevel

load_dotenv()


async def main():
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    planner = Agent(name="Planner", model=model, instructions="Create a plan.")
    executor = Agent(name="Executor", model=model, instructions="Execute the plan.")

    # Track events via callbacks
    events = []
    
    observer = FlowObserver(
        level=ObservabilityLevel.OFF,  # Silent - we handle output
        on_event=lambda e: events.append(e.type.value),
    )

    flow = Flow(
        name="task",
        agents=[planner, executor],
        topology="pipeline",
        observer=observer,
    )

    print("Running with custom event tracking...")
    result = await flow.run("Plan and execute: organize a meeting")
    
    print(f"\nCollected {len(events)} events:")
    for event in events[:10]:
        print(f"  - {event}")
    
    print(f"\nMetrics: {observer.metrics().get('by_channel', {})}")
    print(f"\nResult: {result.results[-1]['thought'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
