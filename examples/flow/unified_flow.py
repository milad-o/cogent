"""Unified Flow Example - Event-Driven Multi-Agent Orchestration.

This example demonstrates the new unified Flow API that replaces both
the old imperative topologies and reactive flows with a single,
event-driven model.

The unified approach:
- Everything is events
- Agents become Reactors
- Patterns (pipeline, supervisor, mesh) are just Flow configurations
- Middleware provides cross-cutting concerns

Usage:
    uv run python examples/flow/unified_flow.py
"""

import asyncio

from agenticflow import (
    Agent,
    Aggregator,
    Event,
    Flow,
    FlowConfig,
    Observer,
)
from agenticflow.flow import pipeline, supervisor


def get_observer():
    """Get Observer based on settings.verbose_level."""
    # Force trace level for debugging
    return Observer.trace()


async def basic_flow_example():
    """Basic Flow with explicit event wiring."""
    observer = get_observer()

    # Create agents with clean API
    researcher = Agent(
        name="researcher",
        model="gpt4",
        instructions="You research topics thoroughly. Return key findings in 2-3 sentences.",
        observer=observer,
    )

    writer = Agent(
        name="writer",
        model="gpt4",
        instructions="You write engaging content based on research. Keep it brief.",
        observer=observer,
    )

    # Create flow with explicit event wiring
    flow = Flow(config=FlowConfig(max_rounds=10), observer=observer)

    # Register agents with event patterns
    flow.register(researcher, on="task.created", emits="research.done")
    flow.register(writer, on="research.done", emits="flow.done")

    # Run the flow
    result = await flow.run(
        task="Write a brief summary about quantum computing",
        initial_event="task.created",
    )

    return result


async def pipeline_pattern_example():
    """Pipeline pattern - sequential processing."""
    observer = get_observer()

    # Create stage agents
    stage1 = Agent(
        name="extractor",
        model="gpt4",
        instructions="Extract key points from the input. Be concise.",
        observer=observer,
    )

    stage2 = Agent(
        name="organizer",
        model="gpt4",
        instructions="Organize and structure the key points logically.",
        observer=observer,
    )

    stage3 = Agent(
        name="polisher",
        model="gpt4",
        instructions="Polish and finalize the content. Make it professional.",
        observer=observer,
    )

    # Create pipeline using pattern helper
    flow = pipeline([stage1, stage2, stage3])

    result = await flow.run("The quick brown fox jumps over the lazy dog. This is a test sentence for processing.")

    return result


async def supervisor_pattern_example():
    """Supervisor pattern - coordinator with workers."""
    observer = get_observer()

    # Coordinator agent
    coordinator = Agent(
        name="coordinator",
        model="gpt4",
        role="supervisor",
        instructions="You coordinate work and delegate to specialists. Provide a final summary.",
        observer=observer,
    )

    # Worker agents
    analyst = Agent(
        name="analyst",
        model="gpt4",
        instructions="You analyze data and provide insights. Be analytical.",
        observer=observer,
    )

    writer = Agent(
        name="writer",
        model="gpt4",
        instructions="You write reports based on analysis. Be clear and concise.",
        observer=observer,
    )

    # Create supervisor flow
    flow = supervisor(
        coordinator=coordinator,
        workers=[analyst, writer],
    )

    result = await flow.run("Analyze the benefits of remote work and write a brief report")

    return result


async def custom_reactors_example():
    """Custom reactors - pure functions without LLM."""
    observer = get_observer()

    # Create flow
    flow = Flow(observer=observer)

    # Register function reactors (no LLM needed)
    @flow.register
    def process_data(event: Event) -> Event:
        """Process incoming data - doubles the value."""
        value = event.data.get("value", 0)
        return Event(
            name="data.processed",
            source="processor",
            data={"result": value * 2},
        )

    flow.register(
        lambda e: Event(
            name="data.validated",
            source="validator",
            data={"valid": True, **e.data},
        ),
        on="data.processed",
    )

    # Add aggregator to collect results
    flow.register(
        Aggregator(collect=2, emit="all.done"),
        on="data.validated",
    )

    result = await flow.run(
        data={"value": 42},
        initial_event="data.received",
    )

    return result


async def streaming_example():
    """Streaming events as they occur."""
    observer = get_observer()

    agent = Agent(
        name="joker",
        model="gpt4",
        instructions="Tell a short, clean joke. Keep it brief.",
        stream=True,
        observer=observer,
    )

    flow = Flow(observer=observer)
    flow.register(agent, on="task.created", emits="flow.done")

    async for _event in flow.stream("Tell me a joke"):
        # Events are logged by observer, no manual printing needed
        pass

    return None


async def main():
    """Run all examples."""
    print("\nStarting Unified Flow Examples\n")

    # Basic flow
    print("═══ Basic Flow Example ═══")
    result = await basic_flow_example()
    print(f"Result: success={result.success}, events={result.events_processed}")
    if result.output:
        output_str = str(result.output)
        print(f"Output: {output_str[:200]}..." if len(output_str) > 200 else f"Output: {output_str}")

    # Pipeline pattern
    print("\n═══ Pipeline Pattern Example ═══")
    result = await pipeline_pattern_example()
    print(f"Result: success={result.success}, stages={result.events_processed}")

    # Supervisor pattern
    print("\n═══ Supervisor Pattern Example ═══")
    result = await supervisor_pattern_example()
    print(f"Result: success={result.success}")

    # Custom reactors (no LLM)
    print("\n═══ Custom Reactors Example ═══")
    result = await custom_reactors_example()
    print(f"Result: success={result.success}")
    if result.event_history:
        print(f"Events: {[e.name for e in result.event_history]}")

    # Streaming
    print("\n═══ Streaming Example ═══")
    await streaming_example()

    print("\n✓ All examples completed")


if __name__ == "__main__":
    asyncio.run(main())
