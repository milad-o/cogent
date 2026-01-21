"""Pattern Syntax Examples - Event@Source Filtering

Demonstrates the new pattern syntax for filtering events by source:
- event@source - @ separator for concise event-source filtering
- Wildcards in both event and source parts
- Multiple patterns with OR logic
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent, Flow
from agenticflow.events import Event


# ============================================================================
# Example 1: Basic @ Separator
# ============================================================================

async def example_1_basic_at_separator():
    """Most common pattern: event@source"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic @ Separator")
    print("=" * 70)
    
    flow = Flow()
    
    # Create agents
    researcher = Agent(
        name="researcher",
        model=get_model(),
        instructions="You are a research agent. Analyze the topic briefly in 1-2 sentences.",
    )
    
    reviewer = Agent(
        name="reviewer",
        model=get_model(),
        instructions="You are a reviewer. Review the research output briefly. Just say 'approved' or suggest 1 improvement.",
    )
    
    # Register with pattern syntax - reviewer only processes researcher's output
    flow.register(researcher, on="task.created")
    flow.register(reviewer, on="agent.done@researcher")  # ✨ Pattern syntax!
    
    print("\n📌 Pattern: 'agent.done@researcher'")
    print("   - Event: agent.done")
    print("   - Source filter: researcher")
    print("   - Reviewer only runs after researcher completes\n")
    
    # Run
    result = await flow.run(
        initial_event=Event(name="task.created", data={"task": "Analyze quantum computing"})
    )
    
    print(f"\n✅ Success: {result.success}")
    if result.output:
        output_str = str(result.output)
        print(f"📝 Final output: {output_str[:150]}..." if len(output_str) > 150 else f"📝 Final output: {output_str}")


# ============================================================================
# Example 2: Wildcard in Event
# ============================================================================

async def example_2_wildcard_event():
    """Wildcard in event part - match any .done events"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Wildcard in Event (*.done@researcher)")
    print("=" * 70)
    
    flow = Flow()
    
    researcher = Agent(
        name="researcher",
        model=get_model(),
        instructions="Research briefly in 1-2 sentences.",
    )
    
    # Monitor catches ALL .done events from researcher
    def monitor(event: Event):
        print(f"🔍 Monitor detected: {event.name} from {event.source}")
        return f"Logged: {event.name}"
    
    flow.register(researcher, on="task.created")
    flow.register(monitor, on="*.done@researcher")  # ✨ Wildcard in event!
    
    print("\n📌 Pattern: '*.done@researcher'")
    print("   - Matches: agent.done, task.done, analysis.done, etc.")
    print("   - From source: researcher only\n")
    
    result = await flow.run(
        initial_event=Event(name="task.created", data={"task": "Quick research"})
    )
    
    print(f"\n✅ Success: {result.success}")


# ============================================================================
# Example 3: Wildcard in Source
# ============================================================================

async def example_3_wildcard_source():
    """Wildcard in source part - match multiple sources"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Wildcard in Source (agent.done@worker*)")
    print("=" * 70)
    
    flow = Flow()
    
    worker1 = Agent(name="worker1", model=get_model(), instructions="Work briefly in 1 sentence.")
    worker2 = Agent(name="worker2", model=get_model(), instructions="Work briefly in 1 sentence.")
    
    # Aggregator catches agent.done from any worker
    def aggregator(event: Event):
        print(f"📊 Aggregating results from {event.source}")
        return f"Aggregated: {event.source}"
    
    flow.register(worker1, on="task.assigned")
    flow.register(worker2, on="task.assigned")
    flow.register(aggregator, on="agent.done@worker*")  # ✨ Wildcard in source!
    
    print("\n📌 Pattern: 'agent.done@worker*'")
    print("   - Event: agent.done (exact)")
    print("   - Sources: worker1, worker2, worker3, etc.\n")
    
    # Run with initial event
    await flow.run(initial_event=Event(name="task.assigned", data={"task": "Process data"}))
    
    print(f"\n✅ Completed")


# ============================================================================
# Example 4: Wildcards in Both
# ============================================================================

async def example_4_wildcard_both():
    """Wildcards in both event and source"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Wildcards in Both (*.error@agent*)")
    print("=" * 70)
    
    flow = Flow()
    
    # Error handler catches ALL .error events from ANY agent
    def error_handler(event: Event):
        print(f"🚨 ERROR: {event.name} from {event.source}")
        print(f"   Details: {event.data}")
        return "Error logged"
    
    flow.register(error_handler, on="*.error@agent*")  # ✨ Double wildcards!
    
    print("\n📌 Pattern: '*.error@agent*'")
    print("   - Events: auth.error, db.error, api.error, etc.")
    print("   - Sources: agent1, agent2, agent_worker, etc.\n")
    
    # Simulate an error
    await flow.run(
        initial_event=Event(
            name="auth.error", 
            source="agent1",
            data={"message": "Invalid credentials"}
        )
    )
    
    print(f"\n✅ All errors handled")


# ============================================================================
# Example 5: Multiple Patterns
# ============================================================================

async def example_5_multiple_patterns():
    """Multiple patterns for OR logic across sources"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multiple Patterns (OR Logic)")
    print("=" * 70)
    
    flow = Flow()
    
    analyst1 = Agent(name="analyst1", model=get_model(), instructions="Analyze briefly in 1 sentence.")
    analyst2 = Agent(name="analyst2", model=get_model(), instructions="Analyze briefly in 1 sentence.")
    
    # Aggregator processes results from EITHER analyst
    def aggregator(event: Event):
        print(f"📊 Aggregating from {event.source}")
        return f"Combined analysis from {event.source}"
    
    flow.register(analyst1, on="data.received")
    flow.register(analyst2, on="data.received")
    flow.register(
        aggregator,
        on=["agent.done@analyst1", "agent.done@analyst2"]  # ✨ Multiple patterns!
    )
    
    print("\n📌 Patterns: ['agent.done@analyst1', 'agent.done@analyst2']")
    print("   - OR logic: Triggers if from analyst1 OR analyst2")
    print("   - Equivalent to: on='agent.done', after=['analyst1', 'analyst2']\n")
    
    # Run with initial event
    await flow.run(initial_event=Event(name="data.received", data={"data": "Sample data"}))
    
    print(f"\n✅ Aggregation complete")


# ============================================================================
# Example 6: Comparison with After Parameter
# ============================================================================

async def example_6_comparison():
    """Compare pattern syntax with traditional after parameter"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Pattern Syntax vs After Parameter")
    print("=" * 70)
    
    flow = Flow()
    
    researcher = Agent(name="researcher", model=get_model(), instructions="Research briefly in 1-2 sentences.")
    
    # Method 1: Pattern syntax (concise)
    def reviewer1(event: Event):
        return "Review via pattern syntax"
    
    # Method 2: After parameter (explicit)
    def reviewer2(event: Event):
        return "Review via after parameter"
    
    flow.register(researcher, on="task.created")
    flow.register(reviewer1, on="agent.done@researcher")  # Pattern syntax
    flow.register(reviewer2, on="agent.done", after="researcher")  # After param
    
    print("\n📌 Two equivalent approaches:")
    print("   1. Pattern syntax:   on='agent.done@researcher'")
    print("   2. After parameter:  on='agent.done', after='researcher'")
    print("\n   ✅ Both work identically - choose based on preference")
    print("   ✅ Pattern syntax is more concise")
    print("   ✅ After parameter is more explicit\n")
    
    result = await flow.run(
        initial_event=Event(name="task.created", data={"task": "Compare approaches"})
    )
    
    print(f"\n✅ Both reviewers ran successfully")


# ============================================================================
# Main Menu
# ============================================================================

async def main():
    """Run examples interactively."""
    while True:
        print("\n" + "=" * 70)
        print("Pattern Syntax Examples (@)")
        print("=" * 70)
        print("\n1. Basic @ separator")
        print("2. Wildcard in event (*.done@researcher)")
        print("3. Wildcard in source (agent.done@worker*)")
        print("4. Wildcards in both (*.error@agent*)")
        print("5. Multiple patterns (OR logic)")
        print("6. Comparison with after parameter")
        print("\na. Run all examples")
        print("q. Quit")
        
        choice = input("\nSelect example (1-6, a, q): ").strip().lower()
        
        if choice == "q":
            break
        elif choice == "1":
            await example_1_basic_at_separator()
        elif choice == "2":
            await example_2_wildcard_event()
        elif choice == "3":
            await example_3_wildcard_source()
        elif choice == "4":
            await example_4_wildcard_both()
        elif choice == "5":
            await example_5_multiple_patterns()
        elif choice == "6":
            await example_6_comparison()
        elif choice == "a":
            await example_1_basic_at_separator()
            await example_2_wildcard_event()
            await example_3_wildcard_source()
            await example_4_wildcard_both()
            await example_5_multiple_patterns()
            await example_6_comparison()
        else:
            print("❌ Invalid choice")
    
    print("\n👋 Thanks for exploring pattern syntax!")


if __name__ == "__main__":
    asyncio.run(main())
