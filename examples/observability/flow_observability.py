"""Flow Observability Demo.

Demonstrates how to observe and debug event-driven flows using the Observer system.
Shows different observer levels, trace filtering, performance analysis, and export.

Run with:
    uv run python examples/observability/flow_observability.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add examples to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model

from agenticflow import Agent, Flow
from agenticflow.observability import Observer, TraceType


# =============================================================================
# Demo 1: Observer Levels
# =============================================================================


async def demo_observer_levels():
    """Show how different observer levels reveal different information."""
    print("=" * 80)
    print("DEMO 1: Observer Levels")
    print("=" * 80)

    model = get_model()
    researcher = Agent(
        name="researcher",
        model=model,
        system_prompt="You are a research specialist. Provide concise, factual information.",
    )
    writer = Agent(
        name="writer",
        model=model,
        system_prompt="You are a writer. Create clear, engaging content.",
    )

    # Test all observer levels
    levels = [
        ("PROGRESS", Observer.progress()),
        ("VERBOSE", Observer.verbose()),
        ("DEBUG", Observer.debug()),
    ]

    for level_name, observer in levels:
        print(f"\n{'-' * 80}")
        print(f"Level: {level_name}")
        print(f"{'-' * 80}\n")

        flow = Flow(observer=observer)
        flow.register(researcher, on="task.created", emits="research.done")
        flow.register(writer, on="research.done", emits="flow.done")

        result = await flow.run(
            "Write a brief explanation of quantum entanglement",
            initial_event="task.created",
        )

        print(f"\n✅ Flow completed")
        print(f"Output length: {len(result.output)} chars")


# =============================================================================
# Demo 2: Trace Filtering and Analysis
# =============================================================================


async def demo_trace_filtering():
    """Demonstrate how to filter and analyze traces."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Trace Filtering and Analysis")
    print("=" * 80)

    model = get_model()
    classifier = Agent(name="classifier", model=model, system_prompt="Classify the task type.")
    processor = Agent(name="processor", model=model, system_prompt="Process the task.")

    observer = Observer.trace()
    flow = Flow(observer=observer)

    flow.register(classifier, on="task.created", emits="task.classified")
    flow.register(processor, on="task.classified", emits="flow.done")

    result = await flow.run(
        "Analyze this customer feedback: 'Great product!'",
        initial_event="task.created",
    )

    print("\n" + "-" * 80)
    print("Analysis Results")
    print("-" * 80)

    # 1. Event chain
    print("\n📊 Event Chain:")
    events = [
        observed.event.data["event_type"]
        for observed in observer.events()
        if observed.event.type == TraceType.REACTIVE_EVENT_EMITTED
    ]
    print("   " + " → ".join(events))

    # 2. Agent performance
    print("\n⏱️  Agent Performance:")
    for observed in observer.events():
        if observed.event.type == TraceType.REACTIVE_AGENT_COMPLETED:
            agent = observed.event.data["agent"]
            duration = observed.event.data.get("duration_ms", 0)
            print(f"   {agent}: {duration:.0f}ms")

    # 3. Event processing stats
    print("\n📈 Processing Stats:")
    rounds = [
        observed for observed in observer.events()
        if observed.event.type == TraceType.REACTIVE_ROUND_COMPLETED
    ]
    print(f"   Total rounds: {len(rounds)}")
    
    total_events = len([
        observed for observed in observer.events()
        if observed.event.type == TraceType.REACTIVE_EVENT_PROCESSED
    ])
    print(f"   Events processed: {total_events}")

    # 4. Check for issues
    print("\n🔍 Issue Detection:")
    no_matches = [
        observed for observed in observer.events()
        if observed.event.type == TraceType.REACTIVE_NO_MATCH
    ]
    if no_matches:
        print(f"   ⚠️  {len(no_matches)} events had no matching reactors")
        for observed in no_matches:
            event_type = observed.event.data.get("event_type", "unknown")
            print(f"      - {event_type}")
    else:
        print("   ✅ All events matched reactors")


# =============================================================================
# Demo 3: Performance Analysis
# =============================================================================


async def demo_performance_analysis():
    """Show how to identify performance bottlenecks."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Performance Analysis")
    print("=" * 80)

    model = get_model()
    
    # Create a multi-stage pipeline
    agents = [
        Agent(name=f"stage_{i}", model=model, system_prompt=f"Stage {i} processor")
        for i in range(1, 4)
    ]

    observer = Observer.trace()
    flow = Flow(observer=observer)

    # Register in sequence
    flow.register(agents[0], on="task.created", emits="stage_1.done")
    flow.register(agents[1], on="stage_1.done", emits="stage_2.done")
    flow.register(agents[2], on="stage_2.done", emits="flow.done")

    result = await flow.run(
        "Process this data through all stages",
        initial_event="task.created",
    )

    print("\n" + "-" * 80)
    print("Performance Report")
    print("-" * 80)

    # Calculate performance metrics
    agent_times = {}
    for observed in observer.events():
        if observed.event.type == TraceType.REACTIVE_AGENT_COMPLETED:
            agent = observed.event.data["agent"]
            duration = observed.event.data.get("duration_ms", 0)
            agent_times[agent] = duration

    total_time = sum(agent_times.values())
    print(f"\n⏱️  Total Agent Time: {total_time:.0f}ms")
    print("\n📊 Breakdown:")
    
    for agent, duration in sorted(agent_times.items(), key=lambda x: x[1], reverse=True):
        percent = (duration / total_time * 100) if total_time > 0 else 0
        bar = "█" * int(percent / 5)
        print(f"   {agent:12s} {duration:6.0f}ms [{percent:5.1f}%] {bar}")

    # Identify bottleneck
    slowest = max(agent_times.items(), key=lambda x: x[1]) if agent_times else None
    if slowest:
        print(f"\n🐌 Bottleneck: {slowest[0]} ({slowest[1]:.0f}ms)")


# =============================================================================
# Demo 4: Debugging Unmatched Events
# =============================================================================


async def demo_debugging_unmatched():
    """Show how to debug events that don't match any reactors."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Debugging Unmatched Events")
    print("=" * 80)

    model = get_model()
    agent = Agent(name="worker", model=model)

    observer = Observer.debug()
    flow = Flow(observer=observer)

    # Intentionally register with wrong event pattern
    flow.register(agent, on="task.started", emits="flow.done")  # Wrong pattern!

    print("\n⚠️  Attempting to run with mismatched event pattern...")
    
    result = await flow.run(
        "Do some work",
        initial_event="task.created",  # This won't match!
    )

    print("\n" + "-" * 80)
    print("Debug Analysis")
    print("-" * 80)

    # Check for unmatched events
    no_matches = [
        observed for observed in observer.events()
        if observed.event.type == TraceType.REACTIVE_NO_MATCH
    ]

    if no_matches:
        print("\n🔍 Unmatched Events Detected:")
        for observed in no_matches:
            event_type = observed.event.data.get("event_type", "unknown")
            available = observed.event.data.get("available_reactors", [])
            print(f"\n   Event: {event_type}")
            print(f"   Available reactors: {available}")
            print(f"   ⚠️  This event didn't trigger any reactors!")
            print(f"   💡 Fix: Register a reactor for '{event_type}'")


# =============================================================================
# Demo 5: Exporting Traces
# =============================================================================


async def demo_export_traces():
    """Show how to export traces for offline analysis."""
    print("\n\n" + "=" * 80)
    print("DEMO 5: Exporting Traces")
    print("=" * 80)

    model = get_model()
    agent1 = Agent(name="analyzer", model=model)
    agent2 = Agent(name="reporter", model=model)

    observer = Observer.trace()
    flow = Flow(observer=observer)

    flow.register(agent1, on="task.created", emits="analysis.done")
    flow.register(agent2, on="analysis.done", emits="flow.done")

    result = await flow.run(
        "Analyze and report on the data",
        initial_event="task.created",
    )

    print("\n" + "-" * 80)
    print("Exporting Traces")
    print("-" * 80)

    # Export reactive flow traces to JSON
    flow_traces = [
        {
            "type": observed.event.type.value,
            "timestamp": observed.event.timestamp.isoformat(),
            "data": observed.event.data,
        }
        for observed in observer.events()
        if observed.event.type.value.startswith("reactive")
    ]

    output_file = Path("flow_traces_demo.json")
    output_file.write_text(json.dumps(flow_traces, indent=2))

    print(f"\n✅ Exported {len(flow_traces)} traces to {output_file}")
    print(f"\nTrace types included:")
    
    trace_types = {}
    for trace in flow_traces:
        trace_type = trace["type"]
        trace_types[trace_type] = trace_types.get(trace_type, 0) + 1
    
    for trace_type, count in sorted(trace_types.items()):
        print(f"   {trace_type}: {count}")

    print(f"\n💡 You can now analyze these traces offline or share them for debugging")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all demos."""
    await demo_observer_levels()
    await demo_trace_filtering()
    await demo_performance_analysis()
    await demo_debugging_unmatched()
    await demo_export_traces()

    print("\n\n" + "=" * 80)
    print("✅ All observability demos completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • Use Observer.progress() for production")
    print("  • Use Observer.debug() for development")
    print("  • Use Observer.trace() for troubleshooting")
    print("  • Filter traces by type for specific analysis")
    print("  • Export traces for offline analysis")
    print("  • Monitor REACTIVE_NO_MATCH to catch misconfigurations")
    print("\n📚 See docs/observability.md for complete guide")


if __name__ == "__main__":
    asyncio.run(main())
