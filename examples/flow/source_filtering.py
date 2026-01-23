"""Source-Based Filtering - React to events from specific sources.

This example demonstrates how to filter events by their source using AgenticFlow's
multi-level source filtering API. This enables precise control over which reactors
respond to which events based on who emitted them.

The example shows:
1. Beginner API: Simple `after` parameter
2. Intermediate API: Helper functions and composition
3. Real-world scenarios: Research workflow with specialized reviewers
"""

import asyncio

from agenticflow import Agent, Flow
from agenticflow.events import Event, any_source, from_source, not_from_source

# Initialize model
model = "gpt4"


# =============================================================================
# Quick Demo - No LLM Calls (Fast!)
# =============================================================================


async def quick_demo_no_llm():
    """Quick demonstration without LLM calls - shows core filtering logic."""
    print("\n" + "=" * 70)
    print("QUICK DEMO: Source Filtering (No LLM Calls)")
    print("=" * 70)

    # Track which reactors were called
    called = []

    def make_handler(name, emit_done=False):
        """Create a simple handler that tracks calls."""
        def handler(event: Event, ctx):
            called.append(f"{name} received {event.name} from {event.source}")
            print(f"  ✓ {name} → received '{event.name}' from '{event.source}'")
            # Emit done event to trigger next iteration
            if emit_done:
                return Event(name="flow.done", source=name)
            return None
        return handler

    print("\n📌 Testing: handler_a only reacts to 'researcher'")
    print("=" * 70)

    flow1 = Flow()
    flow1.register(make_handler("handler_a", emit_done=True), on="task.created", after="researcher")

    print("\n📨 Event from 'researcher':")
    await flow1.run(initial_event=Event(name="task.created", source="researcher"))

    print("\n📨 Event from 'other':")
    flow2 = Flow()
    flow2.register(make_handler("handler_a_2", emit_done=True), on="task.created", after="researcher")
    await flow2.run(initial_event=Event(name="task.created", source="other"))

    print("\n" + "=" * 70)
    print("📌 Testing: handler_b reacts to analyst1 OR analyst2")
    print("=" * 70)

    flow3 = Flow()
    flow3.register(make_handler("handler_b", emit_done=True), on="task.created", after=["analyst1", "analyst2"])

    print("\n📨 Event from 'analyst1':")
    await flow3.run(initial_event=Event(name="task.created", source="analyst1"))

    print("\n📨 Event from 'analyst2':")
    flow4 = Flow()
    flow4.register(make_handler("handler_b_2", emit_done=True), on="task.created", after=["analyst1", "analyst2"])
    await flow4.run(initial_event=Event(name="task.created", source="analyst2"))

    print("\n📨 Event from 'other':")
    flow5 = Flow()
    flow5.register(make_handler("handler_b_3", emit_done=True), on="task.created", after=["analyst1", "analyst2"])
    await flow5.run(initial_event=Event(name="task.created", source="other"))

    print("\n" + "=" * 70)
    print("📌 Testing: handler_c reacts to 'analyst*' (wildcard)")
    print("=" * 70)

    flow6 = Flow()
    flow6.register(make_handler("handler_c", emit_done=True), on="task.created", after="analyst*")

    print("\n📨 Event from 'analyst1':")
    await flow6.run(initial_event=Event(name="task.created", source="analyst1"))

    print("\n📨 Event from 'analyst_xyz':")
    flow7 = Flow()
    flow7.register(make_handler("handler_c_2", emit_done=True), on="task.created", after="analyst*")
    await flow7.run(initial_event=Event(name="task.created", source="analyst_xyz"))

    print("\n📨 Event from 'other':")
    flow8 = Flow()
    flow8.register(make_handler("handler_c_3", emit_done=True), on="task.created", after="analyst*")
    await flow8.run(initial_event=Event(name="task.created", source="other"))

    print("\n" + "=" * 70)
    print(f"✅ Total handler calls: {len(called)}")
    print("\n✅ Source filtering working correctly!")


# =============================================================================
# Level 1: Beginner API - Using `after` parameter
# =============================================================================


async def level1_basic_after():
    """Level 1: Simple source filtering with after parameter."""
    print("\n" + "=" * 70)
    print("LEVEL 1: Basic After Parameter")
    print("=" * 70)

    flow = Flow()

    # Create research agent
    researcher = Agent(
        name="researcher",
        model="gpt4",
        instructions="You are a research analyst. Provide a brief 2-sentence analysis."
    )

    # Create reviewer agent
    reviewer = Agent(
        name="reviewer",
        model="gpt4",
        instructions="You are a quality reviewer. Provide brief feedback in 2 sentences."
    )

    # Register researcher - reacts to initial task
    flow.register(researcher, on="task.created", emits="research.done")

    # Register reviewer - ONLY reacts to research from the researcher
    # This is the key: after="researcher" filters events by source
    flow.register(reviewer, on="research.done", after="researcher", emits="flow.done")

    print("\n📌 Setup:")
    print("  - Researcher reacts to 'task.created'")
    print("  - Reviewer reacts to 'research.done' BUT only from 'researcher'")

    print("\n🚀 Running flow...")
    # Run the flow
    result = await flow.run("Benefits of async programming in Python")

    print(f"\n✅ Success: {result.success}")
    print(f"📊 Events processed: {result.events_processed}")
    print("\n📝 Final output (first 200 chars):")
    print(f"{str(result.output)[:200]}...")


async def level1_multiple_sources():
    """Level 1: React to events from multiple sources (OR logic)."""
    print("\n" + "=" * 70)
    print("LEVEL 1: Multiple Sources (OR Logic)")
    print("=" * 70)

    flow = Flow()

    # Create multiple analyst agents
    tech_analyst = Agent(
        name="tech_analyst",
        model="gpt4",
        instructions="You analyze technical aspects. Keep it brief (1-2 sentences)."
    )

    business_analyst = Agent(
        name="business_analyst",
        model="gpt4",
        instructions="You analyze business aspects. Keep it brief (1-2 sentences)."
    )

    # Create aggregator that combines reports from both analysts
    aggregator = Agent(
        name="aggregator",
        model="gpt4",
        instructions="Combine the analyses into a brief summary (2-3 sentences)."
    )

    # Register analysts
    flow.register(tech_analyst, on="task.created", emits="analysis.done")
    flow.register(business_analyst, on="task.created", emits="analysis.done")

    # Aggregator reacts to analysis from EITHER analyst
    # Pass a list for OR logic
    flow.register(
        aggregator,
        on="analysis.done",
        after=["tech_analyst", "business_analyst"],
        emits="flow.done"
    )

    print("\n📌 Setup:")
    print("  - tech_analyst and business_analyst both analyze the task")
    print("  - aggregator reacts to 'analysis.done' from EITHER analyst")

    print("\n🚀 Running flow...")
    result = await flow.run("Impact of AI on software development")

    print(f"\n✅ Success: {result.success}")
    print(f"📊 Events processed: {result.events_processed}")
    print("\n📝 Final output (first 200 chars):")
    print(f"{str(result.output)[:200]}...")


async def level1_wildcards():
    """Level 1: Wildcard patterns in after parameter."""
    print("\n" + "=" * 70)
    print("LEVEL 1: Wildcard Patterns")
    print("=" * 70)

    flow = Flow()

    # Create multiple similar agents
    analyst1 = Agent(
        name="analyst_1",
        model="gpt4",
        instructions="Provide a brief 1-sentence analysis."
    )
    analyst2 = Agent(
        name="analyst_2",
        model="gpt4",
        instructions="Provide a brief 1-sentence analysis."
    )

    # Create supervisor that reacts to ALL analysts
    supervisor = Agent(
        name="supervisor",
        model="gpt4",
        instructions="Review the analyst reports and provide brief feedback (2 sentences)."
    )

    # Register analysts
    for analyst in [analyst1, analyst2]:
        flow.register(analyst, on="task.created", emits="analysis.done")

    # Supervisor reacts to ANY agent matching pattern
    # Use * wildcard to match all analysts
    flow.register(
        supervisor,
        on="analysis.done",
        after="analyst_*",  # Matches analyst_1, analyst_2, etc.
        emits="flow.done"
    )

    print("\n📌 Setup:")
    print("  - 2 analysts (analyst_1, analyst_2)")
    print("  - supervisor reacts to 'analysis.done' from 'analyst_*' (wildcard)")

    print("\n🚀 Running flow...")
    result = await flow.run("Market trends analysis")

    print(f"\n✅ Success: {result.success}")
    print(f"📊 Events processed: {result.events_processed}")
    print("\n📝 Final output (first 200 chars):")
    print(f"{str(result.output)[:200]}...")


# =============================================================================
# Level 2: Intermediate API - Helper Functions and Composition
# =============================================================================


async def level2_from_source():
    """Level 2: Using from_source() for explicit filtering."""
    print("\n" + "=" * 70)
    print("LEVEL 2: from_source() Helper")
    print("=" * 70)

    flow = Flow()

    researcher = Agent(
        name="researcher",
        model="gpt4",
        instructions="Provide a brief 2-sentence analysis."
    )
    reviewer = Agent(
        name="reviewer",
        model="gpt4",
        instructions="Review the analysis briefly (2 sentences)."
    )

    flow.register(researcher, on="task.created", emits="research.done")

    # Explicit source filtering with from_source()
    # This is equivalent to after="researcher" but more explicit
    flow.register(
        reviewer,
        on="research.done",
        when=from_source("researcher"),  # Same as after="researcher"
        emits="flow.done"
    )

    print("\n📌 Setup:")
    print("  - Using when=from_source('researcher')")
    print("  - Equivalent to after='researcher' but more explicit")

    print("\n🚀 Running flow...")
    result = await flow.run("Quantum computing applications")

    print(f"\n✅ Success: {result.success}")
    print(f"📊 Events processed: {result.events_processed}")
    print("\n📝 Final output (first 200 chars):")
    print(f"{str(result.output)[:200]}...")


async def level2_not_from_source():
    """Level 2: Exclude specific sources with not_from_source()."""
    print("\n" + "=" * 70)
    print("LEVEL 2: not_from_source() - Exclusion")
    print("=" * 70)

    flow = Flow()

    # Track all events
    event_log = []

    def event_logger(event: Event, ctx):
        """Log all events EXCEPT from internal system."""
        event_log.append(f"{event.name} from {event.source}")
        print(f"  📝 Logged: {event.name} from {event.source}")
        return None

    # Log all events EXCEPT from 'system'
    flow.register(
        event_logger,
        on="*",  # Listen to all events
        when=not_from_source("system")  # Exclude 'system' events
    )

    print("\n📌 Setup:")
    print("  - Logger reacts to ALL events ('*')")
    print("  - BUT excludes events from 'system' source")

    # Emit various events
    await flow.emit(Event(name="user.action", source="user_interface"))
    await flow.emit(Event(name="data.processed", source="processor"))
    await flow.emit(Event(name="internal.cleanup", source="system"))  # Excluded
    await flow.emit(Event(name="api.call", source="api_gateway"))

    print(f"\n✅ Logged {len(event_log)} events (excluded 1 system event)")


async def level2_composition():
    """Level 2: Composing filters with boolean operators."""
    print("\n" + "=" * 70)
    print("LEVEL 2: Filter Composition with & | ~")
    print("=" * 70)

    flow = Flow()

    # Multiple analysts
    analyst_a = Agent(
        name="analyst_a",
        model="gpt4",
        instructions="Provide brief 1-sentence analysis."
    )
    analyst_b = Agent(
        name="analyst_b",
        model="gpt4",
        instructions="Provide brief 1-sentence analysis."
    )

    # Aggregator that combines reports from A or B
    aggregator = Agent(
        name="aggregator",
        model="gpt4",
        instructions="Combine the analyst reports into 2-3 sentences."
    )

    flow.register(analyst_a, on="task.created", emits="report.done")
    flow.register(analyst_b, on="task.created", emits="report.done")

    # Compose filters: from A OR from B
    flow.register(
        aggregator,
        on="report.done",
        when=from_source("analyst_a") | from_source("analyst_b"),
        emits="flow.done"
    )

    print("\n📌 Setup:")
    print("  - Aggregator filter: analyst_a OR analyst_b")
    print("  - Demonstrates | (OR) composition")

    print("\n🚀 Running flow...")
    result = await flow.run("Customer feedback analysis")

    print(f"\n✅ Success: {result.success}")
    print(f"📊 Events processed: {result.events_processed}")
    print("\n📝 Final output (first 200 chars):")
    print(f"{str(result.output)[:200]}...")


async def level2_any_source():
    """Level 2: Using any_source() convenience function."""
    print("\n" + "=" * 70)
    print("LEVEL 2: any_source() Convenience")
    print("=" * 70)

    flow = Flow()

    sources = ["analyst_1", "analyst_2"]

    # Create multiple analysts
    for source in sources:
        agent = Agent(
            name=source,
            model="gpt4",
            instructions="Provide a brief 1-sentence analysis."
        )
        flow.register(agent, on="task.created", emits="analysis.done")

    # Aggregator using any_source() - cleaner than long OR chain
    aggregator = Agent(
        name="aggregator",
        model="gpt4",
        instructions="Combine analyses into 2-3 sentences."
    )

    flow.register(
        aggregator,
        on="analysis.done",
        when=any_source(sources),  # Convenience function
        emits="flow.done"
    )

    print("\n📌 Setup:")
    print(f"  - Using any_source({sources})")
    print("  - Cleaner than: after=['analyst_1', 'analyst_2', ...]")

    print("\n🚀 Running flow...")
    result = await flow.run("Market data analysis")

    print(f"\n✅ Success: {result.success}")
    print(f"📊 Events processed: {result.events_processed}")
    print("\n📝 Final output (first 200 chars):")
    print(f"{str(result.output)[:200]}...")

    print(f"\n✅ Result: {result.output[:200]}...")


# =============================================================================
# Real-World Scenario: Research Workflow with Specialized Reviewers
# =============================================================================


async def real_world_research_workflow():
    """Real-world: Research workflow with specialized reviewers."""
    print("\n" + "=" * 70)
    print("REAL WORLD: Research Workflow with Specialized Review")
    print("=" * 70)

    flow = Flow()

    # Create agents
    researcher = Agent(
        name="researcher",
        model="gpt4",
        instructions="You are a research analyst. Provide a brief 2-sentence analysis."
    )

    technical_reviewer = Agent(
        name="technical_reviewer",
        model="gpt4",
        instructions="Review technical accuracy in 1-2 sentences."
    )

    style_reviewer = Agent(
        name="style_reviewer",
        model="gpt4",
        instructions="Review writing style and clarity in 1-2 sentences."
    )

    final_editor = Agent(
        name="final_editor",
        model="gpt4",
        instructions="Combine the reviews into a brief final summary (2-3 sentences)."
    )

    # Build workflow
    # 1. Researcher produces initial research
    flow.register(researcher, on="task.created", emits="research.draft")

    # 2. Both reviewers react to researcher's draft
    flow.register(
        technical_reviewer,
        on="research.draft",
        after="researcher",  # Only review researcher's work
        emits="review.technical"
    )

    flow.register(
        style_reviewer,
        on="research.draft",
        after="researcher",  # Only review researcher's work
        emits="review.style"
    )

    # 3. Final editor combines both reviews
    flow.register(
        final_editor,
        on="review.*",  # Listen to all review events
        after=["technical_reviewer", "style_reviewer"],  # Only from reviewers
        emits="flow.done"
    )

    print("\n📌 Workflow:")
    print("  1. researcher → research.draft")
    print("  2. technical_reviewer (after researcher) → review.technical")
    print("  3. style_reviewer (after researcher) → review.style")
    print("  4. final_editor (after reviewers) → flow.done")

    print("\n🚀 Running flow...")
    result = await flow.run("Machine learning in healthcare diagnostics")

    print(f"\n✅ Success: {result.success}")
    print(f"📊 Events processed: {result.events_processed}")
    print("\n📝 Final output (first 200 chars):")
    print(f"{str(result.output)[:200]}...")


async def real_world_high_priority_alerts():
    """Real-world: Only process high-priority events from API."""
    print("\n" + "=" * 70)
    print("REAL WORLD: High-Priority Alerts from Specific Source")
    print("=" * 70)

    flow = Flow()

    # Alert handler that only processes high-priority API events
    alert_handler = Agent(
        name="alert_handler",
        model="gpt4",
        instructions="Handle urgent alerts and take immediate action."
    )

    # Combine source filter with data filter
    flow.register(
        alert_handler,
        on="alert.created",
        when=from_source("api_gateway") & (lambda e: e.data.get("priority") == "high"),
        emits="alert.handled"
    )

    print("\n📌 Setup:")
    print("  - Only react to alerts from 'api_gateway' with priority='high'")
    print("  - Demonstrates combining source filter with data filter")

    # Test events
    events = [
        Event(name="alert.created", source="api_gateway", data={"priority": "high", "message": "Critical error"}),
        Event(name="alert.created", source="api_gateway", data={"priority": "low", "message": "Minor issue"}),
        Event(name="alert.created", source="internal_system", data={"priority": "high", "message": "Internal check"}),
    ]

    for event in events:
        print(f"\n  📨 Emitting: {event.name} from {event.source} (priority={event.data.get('priority')})")
        await flow.emit(event)

    print("\n✅ Only high-priority API alerts were processed")


# =============================================================================
# Main - Run all examples
# =============================================================================


async def main():
    """Run source filtering examples interactively."""
    print("\n" + "=" * 70)
    print("SOURCE-BASED FILTERING EXAMPLES")
    print("=" * 70)
    print("\nChoose which examples to run:")
    print("  0. Quick demo - No LLM calls (fastest!)")
    print("  1. Level 1: Basic after parameter")
    print("  2. Level 1: Multiple sources")
    print("  3. Level 1: Wildcards")
    print("  4. Level 2: from_source helper")
    print("  5. Level 2: not_from_source (no LLM calls)")
    print("  6. Level 2: Filter composition")
    print("  7. Level 2: any_source convenience")
    print("  8. Real-world: Research workflow (slow)")
    print("  9. Real-world: High-priority alerts (no LLM calls)")
    print("  a. Run ALL examples (very slow!)")

    choice = input("\nEnter your choice (0-9, a) [default: 0]: ").strip().lower() or "0"

    examples = {
        "0": quick_demo_no_llm,
        "1": level1_basic_after,
        "2": level1_multiple_sources,
        "3": level1_wildcards,
        "4": level2_from_source,
        "5": level2_not_from_source,
        "6": level2_composition,
        "7": level2_any_source,
        "8": real_world_research_workflow,
        "9": real_world_high_priority_alerts,
    }

    if choice == "a":
        # Run all
        for example in examples.values():
            await example()
    elif choice in examples:
        await examples[choice]()
    else:
        print(f"Invalid choice: {choice}")
        return

    print("\n" + "=" * 70)
    print("✅ EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
