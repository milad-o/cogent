"""Source Groups Examples - Named groups for cleaner multi-source filtering.

This example demonstrates:
1. Defining source groups with add_source_group()
2. Using :group references in 'after' parameter
3. Using :group references in pattern@source syntax
4. Built-in :agents and :system groups
5. Method chaining with source groups

Run: uv run python examples/flow/source_groups.py
"""

import asyncio
import sys
from pathlib import Path



from agenticflow import Agent, Flow


# -----------------------------------------------------------------------------
# Example 1: Basic Source Groups
# -----------------------------------------------------------------------------


async def example_basic_groups():
    """Define and use a basic source group."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Source Groups")
    print("=" * 60)

    flow = Flow()
    model = "gpt4"

    # Create analyst agents
    analyst1 = Agent(name="analyst1", model="gpt4", instructions="You are analyst 1")
    analyst2 = Agent(name="analyst2", model="gpt4", instructions="You are analyst 2")
    analyst3 = Agent(name="analyst3", model="gpt4", instructions="You are analyst 3")

    # Define a group of analysts
    flow.add_source_group("analysts", ["analyst1", "analyst2", "analyst3"])

    # Register analysts
    flow.register(analyst1, on="task.created", emits="analysis.done")
    flow.register(analyst2, on="task.created", emits="analysis.done")
    flow.register(analyst3, on="task.created", emits="analysis.done")

    # Aggregator waits for all analysts using :group reference
    flow.register(
        lambda e: print(f"✓ Aggregator received from: {e.source}"),
        on="analysis.done",
        after=":analysts",  # Only from analysts group!
    )

    print("\nFlow setup:")
    print(f"  Analysts group: {flow.get_source_group('analysts')}")
    print("  Aggregator waits for: after=':analysts'")

    print("\nRunning flow...")
    result = await flow.run("Analyze this task", initial_event="task.created")

    print(f"\nResult: {result.output}")


# -----------------------------------------------------------------------------
# Example 2: Pattern Syntax with Groups
# -----------------------------------------------------------------------------


async def example_pattern_syntax_with_groups():
    """Use :group in pattern@source syntax."""
    print("\n" + "=" * 60)
    print("Example 2: Pattern Syntax with Groups")
    print("=" * 60)

    flow = Flow()
    model = "gpt4"

    # Create worker agents
    worker1 = Agent(name="worker1", model="gpt4", instructions="You are worker 1")
    worker2 = Agent(name="worker2", model="gpt4", instructions="You are worker 2")

    # Define workers group
    flow.add_source_group("workers", ["worker1", "worker2"])

    # Register workers
    flow.register(worker1, on="job.assigned", emits="job.done")
    flow.register(worker2, on="job.assigned", emits="job.done")

    # Reviewer only reacts to jobs from workers group
    flow.register(
        lambda e: print(f"✓ Reviewer checking work from: {e.source}"),
        on="job.done@:workers",  # Pattern syntax with :group!
    )

    print("\nFlow setup:")
    print(f"  Workers group: {flow.get_source_group('workers')}")
    print("  Reviewer listens to: on='job.done@:workers'")

    print("\nRunning flow...")
    result = await flow.run("Complete this job", initial_event="job.assigned")

    print(f"\nResult: {result.output}")


# -----------------------------------------------------------------------------
# Example 3: Built-in :agents Group
# -----------------------------------------------------------------------------


async def example_builtin_agents_group():
    """Use built-in :agents group (auto-populated)."""
    print("\n" + "=" * 60)
    print("Example 3: Built-in :agents Group")
    print("=" * 60)

    flow = Flow()
    model = "gpt4"

    # Create multiple agents
    researcher = Agent(name="researcher", model="gpt4", instructions="Research topics")
    writer = Agent(name="writer", model="gpt4", instructions="Write content")
    editor = Agent(name="editor", model="gpt4", instructions="Edit content")

    # Register agents (they're automatically added to :agents group)
    flow.register(researcher, on="task.created", emits="research.done")
    flow.register(writer, on="research.done", emits="draft.done")
    flow.register(editor, on="draft.done", emits="flow.done")

    # Monitor all agent completions using :agents
    flow.register(
        lambda e: print(f"✓ Monitor: {e.source} completed {e.name}"),
        on="*.done@:agents",  # Monitors all agents!
    )

    print("\nFlow setup:")
    print(f"  :agents group (auto): {flow.get_source_group('agents')}")
    print("  Monitor listens to: on='*.done@:agents'")

    print("\nRunning flow...")
    result = await flow.run("Write an article about AI", initial_event="task.created")

    print(f"\nResult: {result.output}")


# -----------------------------------------------------------------------------
# Example 4: Built-in :system Group
# -----------------------------------------------------------------------------


async def example_builtin_system_group():
    """Use built-in :system group for system events."""
    print("\n" + "=" * 60)
    print("Example 4: Built-in :system Group")
    print("=" * 60)

    flow = Flow()

    print("\nFlow setup:")
    print(f"  :system group: {flow.get_source_group('system')}")

    # Monitor system events
    flow.register(
        lambda e: print(f"✓ System event: {e.name} from {e.source}"),
        on="*",
        after=":system",
    )

    print("  Monitor listens to: on='*', after=':system'")

    print("\n(System events would be emitted during flow execution)")


# -----------------------------------------------------------------------------
# Example 5: Method Chaining with Multiple Groups
# -----------------------------------------------------------------------------


async def example_chaining_groups():
    """Chain multiple group definitions."""
    print("\n" + "=" * 60)
    print("Example 5: Method Chaining with Multiple Groups")
    print("=" * 60)

    flow = Flow()
    model = "gpt4"

    # Chain group definitions
    flow.add_source_group("writers", ["w1", "w2", "w3"]) \
        .add_source_group("reviewers", ["r1", "r2"]) \
        .add_source_group("approvers", ["a1"])

    print("\nDefined groups:")
    print(f"  writers: {flow.get_source_group('writers')}")
    print(f"  reviewers: {flow.get_source_group('reviewers')}")
    print(f"  approvers: {flow.get_source_group('approvers')}")

    # Create workflow agents
    writer = Agent(name="w1", model="gpt4", instructions="Write content")
    reviewer = Agent(name="r1", model="gpt4", instructions="Review content")
    approver = Agent(name="a1", model="gpt4", instructions="Approve content")

    # Register with group-based routing
    flow.register(writer, on="task.created", emits="draft.ready")
    flow.register(reviewer, on="draft.ready@:writers", emits="review.done")
    flow.register(approver, on="review.done@:reviewers", emits="flow.done")

    print("\nWorkflow:")
    print("  writer → draft.ready")
    print("  reviewer (after :writers) → review.done")
    print("  approver (after :reviewers) → flow.done")

    print("\nRunning flow...")
    result = await flow.run("Create a document", initial_event="task.created")

    print(f"\nResult: {result.output}")


# -----------------------------------------------------------------------------
# Interactive Menu
# -----------------------------------------------------------------------------


async def main():
    """Run examples interactively."""
    examples = {
        "1": ("Basic Source Groups", example_basic_groups),
        "2": ("Pattern Syntax with Groups", example_pattern_syntax_with_groups),
        "3": ("Built-in :agents Group", example_builtin_agents_group),
        "4": ("Built-in :system Group", example_builtin_system_group),
        "5": ("Method Chaining", example_chaining_groups),
        "all": ("Run All Examples", None),
    }

    print("\n" + "=" * 60)
    print("Source Groups Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    choice = input("\nSelect example (1-5, all, or 'q' to quit): ").strip()

    if choice.lower() == "q":
        print("Goodbye!")
        return

    if choice == "all":
        for key in ["1", "2", "3", "4", "5"]:
            await examples[key][1]()
            if key != "5":
                input("\nPress Enter to continue to next example...")
    elif choice in examples and examples[choice][1] is not None:
        await examples[choice][1]()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    asyncio.run(main())
