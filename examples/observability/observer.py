"""
Demo: Observer - Unified Observability System

Observer is your single entry point for ALL observability needs:
- Live console output (see agent thoughts in real-time)
- Deep execution tracing (graph, timeline, spans)
- Metrics and statistics
- Export to JSON, Mermaid

Usage:
    uv run python examples/observability/observer.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent, Flow, Observer, Channel, ObservabilityLevel


async def demo_levels():
    """Show different verbosity levels."""
    model = get_model()
    
    analyst = Agent(name="Analyst", model=model)
    writer = Agent(name="Writer", model=model)
    
    # Verbose - see what agents are thinking
    print("\n--- Level: verbose() - See Agent Thoughts ---")
    flow = Flow(
        name="demo",
        agents=[analyst, writer],
        topology="pipeline",
        observer=Observer.verbose(),
    )
    await flow.run("Analyze: What's 2+2? Then summarize.")
    
    # JSON - structured, readable output
    print("\n--- Level: json() - Structured Output ---")
    flow = Flow(
        name="demo",
        agents=[analyst, writer],
        topology="pipeline",
        observer=Observer.json(),
    )
    await flow.run("Explain why the sky is blue.")


async def demo_trace():
    """Show deep tracing with execution graph."""
    model = get_model()
    
    researcher = Agent(name="Researcher", model=model)
    analyst = Agent(name="Analyst", model=model)
    writer = Agent(name="Writer", model=model)
    
    # Trace - maximum observability
    print("\n--- Level: trace() - Deep Execution Tracing ---")
    observer = Observer.trace()
    
    flow = Flow(
        name="research-pipeline",
        agents=[researcher, analyst, writer],
        topology="pipeline",
        observer=observer,
    )
    await flow.run("Research the benefits of exercise, analyze the data, and write a summary.")
    
    # After execution, get insights:
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print(observer.summary())
    
    print("\n" + "="*60)
    print("TIMELINE (detailed)")
    print(observer.timeline(detailed=True))
    
    print("\n" + "="*60)
    print("EXECUTION GRAPH (Mermaid)")
    print(observer.graph())


async def demo_callbacks():
    """Custom callbacks with silent display."""
    model = get_model()
    
    analyst = Agent(name="Analyst", model=model)
    writer = Agent(name="Writer", model=model)
    
    # Collect events via callbacks
    events = []
    agent_actions = []
    
    observer = Observer(
        level=ObservabilityLevel.OFF,  # No console output
        on_event=lambda e: events.append(e.type.value),
        on_agent=lambda name, action, data: agent_actions.append(f"{name}:{action}"),
    )
    
    print("\n--- Custom Callbacks (silent collection) ---")
    flow = Flow(
        name="demo",
        agents=[analyst, writer],
        topology="pipeline",
        observer=observer,
    )
    await flow.run("Quick analysis of Python vs JavaScript")
    
    print(f"  Collected {len(events)} events")
    print(f"  Agent actions: {agent_actions}")
    
    # Get structured trace data (for export to external systems)
    trace_data = observer.execution_trace()
    print(f"  Trace ID: {trace_data['trace_id']}")
    print(f"  Nodes: {len(trace_data['nodes'])}")
    print(f"  Edges: {len(trace_data['edges'])}")


async def main():
    print("\n" + "="*60)
    print("  Observer - Unified Observability Demo")
    print("="*60)
    
    await demo_levels()
    await demo_trace()
    await demo_callbacks()
    
    print("\n" + "="*60)
    print("✅ Done! Observer provides complete observability.")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
