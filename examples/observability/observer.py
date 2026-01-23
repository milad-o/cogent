"""
Demo: Observer - Unified Observability System

Observer is your single entry point for ALL observability needs:
- Live console output (see agent thoughts in real-time)
- Response metadata (tokens, tool calls, duration) - NEW in v1.13.0
- Deep execution tracing (graph, timeline, spans)
- Metrics and statistics
- Export to JSON, Mermaid

The Observer automatically displays Response[T] metadata:
  [AgentName] [completed] (Xs) • N tokens • M tools

Usage:
    uv run python examples/observability/observer.py
    
See also:
    examples/observability/response_metadata.py - Detailed Response metadata demo
"""

import asyncio

from agenticflow import Agent, ObservabilityLevel, Observer, pipeline


async def demo_levels():
    """Show different verbosity levels."""
    model = "gpt4"

    analyst = Agent(name="Analyst", model="gpt4")
    writer = Agent(name="Writer", model="gpt4")

    # Verbose - see what agents are thinking
    print("\n--- Level: verbose() - See Agent Thoughts ---")
    flow = pipeline([analyst, writer], observer=Observer.verbose())
    await flow.run("Analyze: What's 2+2? Then summarize.")

    # JSON - structured, readable output
    print("\n--- Level: json() - Structured Output ---")
    flow = pipeline([analyst, writer], observer=Observer.json())
    await flow.run("Explain why the sky is blue.")


async def demo_trace():
    """Show deep tracing with execution graph."""
    model = "gpt4"

    researcher = Agent(name="Researcher", model="gpt4")
    analyst = Agent(name="Analyst", model="gpt4")
    writer = Agent(name="Writer", model="gpt4")

    # Trace - maximum observability
    print("\n--- Level: trace() - Deep Execution Tracing ---")
    observer = Observer.trace()

    flow = pipeline([researcher, analyst, writer], observer=observer)
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
    model = "gpt4"

    analyst = Agent(name="Analyst", model="gpt4")
    writer = Agent(name="Writer", model="gpt4")

    # Collect events via callbacks
    events = []
    agent_actions = []

    observer = Observer(
        level=ObservabilityLevel.OFF,  # No console output
        on_event=lambda e: events.append(e.type.value),
        on_agent=lambda name, action, data: agent_actions.append(f"{name}:{action}"),
    )

    print("\n--- Custom Callbacks (silent collection) ---")
    flow = pipeline([analyst, writer], observer=observer)
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
