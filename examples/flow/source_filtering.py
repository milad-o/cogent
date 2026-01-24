"""
Source-Based Filtering

React to events from specific sources using AgenticFlow's source filtering API.
Demonstrates: after parameter, wildcards, helper functions, composition.

Usage: uv run python examples/flow/source_filtering.py
"""

import asyncio

from agenticflow import Agent, Flow
from agenticflow.events import Event, any_source, from_source, not_from_source

model = "gpt4"


async def demo_basic_after():
    """Basic source filtering with after parameter."""
    print("\n" + "=" * 60)
    print("1. BASIC after PARAMETER")
    print("=" * 60)

    flow = Flow()

    researcher = Agent(
        name="researcher",
        model=model,
        instructions="Provide brief 2-sentence analysis."
    )
    reviewer = Agent(
        name="reviewer",
        model=model,
        instructions="Review the analysis in 2 sentences."
    )

    flow.register(researcher, on="task.created", emits="analysis.done")
    flow.register(
        reviewer,
        on="analysis.done",
        after="researcher",  # Only react to researcher
        emits="flow.done"
    )

    print("\nRunning flow...")
    result = await flow.run("Impact of AI on software development")

    print(f"Success: {result.success}")
    print(f"Output: {str(result.output)[:150]}...")


async def demo_wildcards():
    """Wildcard patterns for matching multiple sources."""
    print("\n" + "=" * 60)
    print("2. WILDCARD PATTERNS (analyst_*)")
    print("=" * 60)

    flow = Flow()

    analyst1 = Agent(
        name="analyst_1",
        model=model,
        instructions="Brief 1-sentence analysis."
    )
    analyst2 = Agent(
        name="analyst_2",
        model=model,
        instructions="Brief 1-sentence analysis."
    )
    supervisor = Agent(
        name="supervisor",
        model=model,
        instructions="Review reports in 2 sentences."
    )

    for analyst in [analyst1, analyst2]:
        flow.register(analyst, on="task.created", emits="analysis.done")

    flow.register(
        supervisor,
        on="analysis.done",
        after="analyst_*",  # Matches analyst_1, analyst_2, etc.
        emits="flow.done"
    )

    print("\nRunning flow...")
    result = await flow.run("Market trends")

    print(f"Success: {result.success}")
    print(f"Output: {str(result.output)[:150]}...")


async def demo_from_source():
    """Using from_source() helper for explicit filtering."""
    print("\n" + "=" * 60)
    print("3. from_source() HELPER")
    print("=" * 60)

    flow = Flow()

    researcher = Agent(
        name="researcher",
        model=model,
        instructions="Brief 2-sentence analysis."
    )
    reviewer = Agent(
        name="reviewer",
        model=model,
        instructions="Review in 2 sentences."
    )

    flow.register(researcher, on="task.created", emits="research.done")
    flow.register(
        reviewer,
        on="research.done",
        when=from_source("researcher"),  # Explicit filtering
        emits="flow.done"
    )

    print("\nRunning flow...")
    result = await flow.run("Quantum computing applications")

    print(f"Success: {result.success}")
    print(f"Output: {str(result.output)[:150]}...")


async def demo_not_from_source():
    """Exclude specific sources with not_from_source()."""
    print("\n" + "=" * 60)
    print("4. not_from_source() - EXCLUSION")
    print("=" * 60)

    flow = Flow()
    event_log = []

    def event_logger(event: Event, ctx):
        event_log.append(f"{event.name} from {event.source}")
        print(f"  Logged: {event.name} from {event.source}")
        return None

    flow.register(
        event_logger,
        on="*",
        when=not_from_source("system")  # Exclude system events
    )

    await flow.emit(Event(name="user.action", source="ui"))
    await flow.emit(Event(name="data.processed", source="processor"))
    await flow.emit(Event(name="cleanup", source="system"))  # Excluded
    await flow.emit(Event(name="api.call", source="gateway"))

    print(f"\nLogged {len(event_log)} events (excluded 1 system event)")


async def demo_composition():
    """Compose filters with boolean operators."""
    print("\n" + "=" * 60)
    print("5. FILTER COMPOSITION (OR)")
    print("=" * 60)

    flow = Flow()

    analyst_a = Agent(
        name="analyst_a",
        model=model,
        instructions="Brief 1-sentence analysis."
    )
    analyst_b = Agent(
        name="analyst_b",
        model=model,
        instructions="Brief 1-sentence analysis."
    )
    aggregator = Agent(
        name="aggregator",
        model=model,
        instructions="Combine reports in 2-3 sentences."
    )

    flow.register(analyst_a, on="task.created", emits="report.done")
    flow.register(analyst_b, on="task.created", emits="report.done")
    flow.register(
        aggregator,
        on="report.done",
        when=from_source("analyst_a") | from_source("analyst_b"),
        emits="flow.done"
    )

    print("\nRunning flow...")
    result = await flow.run("Customer feedback analysis")

    print(f"Success: {result.success}")
    print(f"Output: {str(result.output)[:150]}...")


async def demo_any_source():
    """Using any_source() convenience function."""
    print("\n" + "=" * 60)
    print("6. any_source() CONVENIENCE")
    print("=" * 60)

    flow = Flow()

    sources = ["analyst_1", "analyst_2"]

    for source in sources:
        agent = Agent(
            name=source,
            model=model,
            instructions="Brief 1-sentence analysis."
        )
        flow.register(agent, on="task.created", emits="analysis.done")

    aggregator = Agent(
        name="aggregator",
        model=model,
        instructions="Combine analyses in 2-3 sentences."
    )

    flow.register(
        aggregator,
        on="analysis.done",
        when=any_source(sources),
        emits="flow.done"
    )

    print("\nRunning flow...")
    result = await flow.run("Market data analysis")

    print(f"Success: {result.success}")
    print(f"Output: {str(result.output)[:150]}...")


async def demo_research_workflow():
    """Real-world: Research workflow with specialized reviewers."""
    print("\n" + "=" * 60)
    print("7. RESEARCH WORKFLOW (Multi-stage)")
    print("=" * 60)

    flow = Flow()

    researcher = Agent(
        name="researcher",
        model=model,
        instructions="Brief 2-sentence analysis."
    )
    technical_reviewer = Agent(
        name="technical_reviewer",
        model=model,
        instructions="Review technical accuracy in 1-2 sentences."
    )
    style_reviewer = Agent(
        name="style_reviewer",
        model=model,
        instructions="Review style in 1-2 sentences."
    )
    final_editor = Agent(
        name="final_editor",
        model=model,
        instructions="Combine reviews in 2-3 sentences."
    )

    flow.register(researcher, on="task.created", emits="research.draft")
    flow.register(
        technical_reviewer,
        on="research.draft",
        after="researcher",
        emits="review.technical"
    )
    flow.register(
        style_reviewer,
        on="research.draft",
        after="researcher",
        emits="review.style"
    )
    flow.register(
        final_editor,
        on="review.*",
        after=["technical_reviewer", "style_reviewer"],
        emits="flow.done"
    )

    print("\nWorkflow: researcher → reviewers → final_editor")
    print("Running flow...")
    result = await flow.run("ML in healthcare diagnostics")

    print(f"Success: {result.success}")
    print(f"Output: {str(result.output)[:150]}...")


async def main():
    """Run all source filtering demos."""
    print("\n" + "=" * 60)
    print("SOURCE-BASED FILTERING EXAMPLES")
    print("=" * 60)

    await demo_basic_after()
    await demo_wildcards()
    await demo_from_source()
    await demo_not_from_source()
    await demo_composition()
    await demo_any_source()
    await demo_research_workflow()

    print("\n" + "=" * 60)
    print("✓ All demos completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
