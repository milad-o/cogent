"""
Demo: Clean 4-Role System

Demonstrates the simplified role system with only 4 distinct roles.

See docs/agent.md for the complete role system reference table and guide.

Usage:
    uv run python examples/basics/roles.py
"""

import asyncio

from agenticflow import Agent, Observer
from agenticflow.agent import get_role_behavior, get_role_prompt, parse_delegation
from agenticflow.core.enums import AgentRole


async def demo_role_system():
    """Show the clean 4-role system."""
    print("\n--- The 4 Roles ---")

    for role in AgentRole:
        behavior = get_role_behavior(role)
        print(f"\n  {role.value.upper()}:")
        print(f"    can_finish:    {'✅' if behavior.can_finish else '❌'}")
        print(f"    can_delegate:  {'✅' if behavior.can_delegate else '❌'}")
        print(f"    can_use_tools: {'✅' if behavior.can_use_tools else '❌'}")


async def demo_role_factories():
    """Show role configuration objects and backwards compatibility."""
    print("\n--- Role Configuration API ---")

    # New API: Role configuration objects (recommended)
    from agenticflow import ReviewerRole, SupervisorRole, WorkerRole

    supervisor_role = SupervisorRole(workers=["Analyst", "Writer"])
    supervisor = Agent(
        name="Manager",
        model="gpt4",
        role=supervisor_role,
    )
    print(f"\n  SupervisorRole(workers=[...]) → {supervisor.role.value}")
    print(f"    Can finish: {supervisor.can_finish}, Can delegate: {supervisor.can_delegate}")

    worker_role = WorkerRole(specialty="data analysis")
    worker = Agent(
        name="Analyst",
        model="gpt4",
        role=worker_role,
    )
    print(f"\n  WorkerRole(specialty='...') → {worker.role.value}")
    print(f"    Can finish: {worker.can_finish}, Can use tools: {worker.can_use_tools}")

    reviewer_role = ReviewerRole(criteria=["accuracy", "clarity"])
    reviewer = Agent(
        name="QA",
        model="gpt4",
        role=reviewer_role,
    )
    print(f"\n  ReviewerRole(criteria=[...]) → {reviewer.role.value}")
    print(f"    Can finish: {reviewer.can_finish}, Can use tools: {reviewer.can_use_tools}")

    # Backwards compatible: String/enum still works
    print("\n--- Backwards Compatible (String/Enum) ---")

    simple = Agent(
        name="Simple",
        model="gpt4",
        role="worker",  # String still works
    )
    print(f"\n  role='worker' (string) → {simple.role.value}")


async def demo_role_prompts():
    """Show built-in role prompts."""
    print("\n--- Built-in Role Prompts ---")

    for role in AgentRole:
        prompt = get_role_prompt(role)
        # Show first 80 chars
        preview = prompt.strip()[:80].replace("\n", " ") + "..."
        print(f"\n  {role.value}:")
        print(f"    {preview}")


async def demo_delegation():
    """Show delegation parsing."""
    print("\n--- Delegation Parsing ---")

    # Supervisor delegation format
    responses = [
        "DELEGATE TO Analyst: Research market trends for AI",
        "FINAL ANSWER: The analysis is complete.",
        "ROUTE TO Writer: Please summarize these findings",
        "I'm thinking about this problem...",  # No command
    ]

    for response in responses:
        delegation = parse_delegation(response)
        print(f"\n  Input: \"{response[:50]}{'...' if len(response) > 50 else ''}\"")
        if delegation:
            print(f"    ✅ {delegation.action.upper()}", end="")
            if delegation.target:
                print(f" → {delegation.target}", end="")
            print(f": {delegation.task[:40]}...")
        else:
            print("    ❌ No command detected")


async def demo_supervisor_flow():
    """Show supervisor coordinating workers."""
    print("\n--- Supervisor Flow Pattern ---")
    print("  Pattern: SUPERVISOR ↔ [WORKER, WORKER]")
    print("  SUPERVISOR delegates and finishes; WORKERs do tool work")

    model = "gpt4"

    from agenticflow import SupervisorRole, WorkerRole

    # Create team
    supervisor = Agent(
        name="Manager",
        model="gpt4",
        role=SupervisorRole(workers=["Researcher", "Writer"]),
        instructions="Coordinate the team to analyze the topic. Delegate to workers, then synthesize.",
    )

    researcher = Agent(
        name="Researcher",
        model="gpt4",
        role=WorkerRole(),
        instructions="Research and provide key facts.",
    )

    writer = Agent(
        name="Writer",
        model="gpt4",
        role=WorkerRole(),
        instructions="Write clear summaries.",
    )

    # Run with supervisor pattern
    from agenticflow import supervisor as supervisor_pattern

    flow = supervisor_pattern(
        coordinator=supervisor,
        workers=[researcher, writer],
        observer=Observer.normal(),
    )

    result = await flow.run("Benefits of remote work")
    out = result.output
    preview = (
        out.content if hasattr(out, "content") else out
    )
    text = str(preview)
    print(f"\n  Result: {text[:200]}...")


async def demo_review_flow():
    """Show reviewer checking work."""
    print("\n--- Pipeline with Review Pattern ---")
    print("  Pattern: WORKER → REVIEWER")
    print("  WORKER does work; REVIEWER approves/rejects and finishes")

    model = "gpt4"

    from agenticflow import ReviewerRole, WorkerRole

    writer = Agent(
        name="Writer",
        model="gpt4",
        role=WorkerRole(),
        instructions="Write a short paragraph about the topic.",
    )

    reviewer = Agent(
        name="QA",
        model="gpt4",
        role=ReviewerRole(criteria=["clarity"]),
        instructions="Review for clarity. Give FINAL ANSWER when approved.",
    )

    # Pipeline: Writer → Reviewer
    from agenticflow import pipeline

    flow = pipeline(
        [writer, reviewer],
        observer=Observer.normal(),
    )

    result = await flow.run("Cloud computing benefits")
    out = result.output
    preview = (
        out.content if hasattr(out, "content") else out
    )
    text = str(preview)
    print(f"\n  Final output: {text[:200]}...")


async def demo_autonomous():
    """Show autonomous agent working independently."""
    print("\n--- Autonomous Pattern ---")
    print("  Pattern: Single AUTONOMOUS agent")
    print("  Can use tools AND finish (perfect for solo tasks)")

    model = "gpt4"
    from agenticflow import AutonomousRole

    assistant = Agent(
        name="Assistant",
        model="gpt4",
        role=AutonomousRole(),
        instructions="Help the user with their request. Answer directly.",
    )

    # For single agent, just run directly (no Flow needed)
    response = await assistant.run("What's 2+2?")
    text = response.unwrap()
    print(f"\n  Answer: {str(text)[:200]}...")


async def main():
    print("\n" + "=" * 60)
    print("  Clean 4-Role System Demo")
    print("=" * 60)

    await demo_role_system()
    await demo_role_factories()
    await demo_role_prompts()
    await demo_delegation()
    await demo_supervisor_flow()
    await demo_review_flow()
    await demo_autonomous()

    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
