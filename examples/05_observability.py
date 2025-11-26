"""
Demo: FlowObserver - Integrated Observability

The FlowObserver plugs directly into Flow for fine-grained event visibility.

Usage:
    uv run python examples/05_observability.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agenticflow import Agent, Flow, FlowObserver, Channel, ObservabilityLevel

load_dotenv()


async def demo_levels():
    """Show different verbosity levels."""
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    analyst = Agent(name="Analyst", model=model)
    writer = Agent(name="Writer", model=model)
    
    # Normal - key milestones
    print("\n--- Level: normal() ---")
    flow = Flow(
        name="demo",
        agents=[analyst, writer],
        topology="pipeline",
        observer=FlowObserver.normal(),
    )
    await flow.run("Analyze: What's 2+2? Then summarize.")
    
    # Debug - everything
    print("\n--- Level: debug() ---")
    flow = Flow(
        name="demo",
        agents=[analyst, writer],
        topology="pipeline",
        observer=FlowObserver.debug(),
    )
    await flow.run("Analyze: What's 2+2? Then summarize.")


async def demo_callbacks():
    """Custom callbacks with silent display."""
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    analyst = Agent(name="Analyst", model=model)
    writer = Agent(name="Writer", model=model)
    
    # Collect events via callbacks
    events = []
    
    observer = FlowObserver(
        level=ObservabilityLevel.OFF,  # No console output
        on_event=lambda e: events.append(e.type.value),
    )
    
    print("\n--- Custom callbacks (silent) ---")
    flow = Flow(
        name="demo",
        agents=[analyst, writer],
        topology="pipeline",
        observer=observer,
    )
    await flow.run("Quick analysis")
    
    print(f"  Collected {len(events)} events: {events[:5]}...")
    
    # Check metrics
    metrics = observer.metrics()
    print(f"  Total: {metrics.get('total_events', 0)}")
    print(f"  By channel: {metrics.get('by_channel', {})}")


async def main():
    print("\n" + "="*50)
    print("  FlowObserver Demo")
    print("="*50)
    
    await demo_levels()
    await demo_callbacks()
    
    print("\n" + "="*50)
    print("✅ Done!")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
