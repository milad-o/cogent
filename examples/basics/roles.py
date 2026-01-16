"""
Demo: Clean 4-Role System

Demonstrates the simplified role system with only 4 distinct roles.

The Role System:
┌─────────────┬────────────┬──────────────┬───────────────┐
│ Role        │ can_finish │ can_delegate │ can_use_tools │
├─────────────┼────────────┼──────────────┼───────────────┤
│ WORKER      │     ❌     │      ❌      │      ✅       │
│ SUPERVISOR  │     ✅     │      ✅      │      ❌       │
│ AUTONOMOUS  │     ✅     │      ❌      │      ✅       │
│ REVIEWER    │     ✅     │      ❌      │      ❌       │
└─────────────┴────────────┴──────────────┴───────────────┘

Usage:
    uv run python examples/basics/roles.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent, Flow, Observer
from agenticflow.agent import parse_delegation, get_role_prompt, get_role_behavior
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
    
    model = get_model()
    
    # New API: Role configuration objects (recommended)
    from agenticflow import SupervisorRole, WorkerRole, ReviewerRole
    
    supervisor_role = SupervisorRole(workers=["Analyst", "Writer"])
    supervisor = Agent(
        name="Manager",
        model=model,
        role=supervisor_role,
    )
    print(f"\n  SupervisorRole(workers=[...]) → {supervisor.role.value}")
    print(f"    Can finish: {supervisor.can_finish}, Can delegate: {supervisor.can_delegate}")
    
    worker_role = WorkerRole(specialty="data analysis")
    worker = Agent(
        name="Analyst",
        model=model,
        role=worker_role,
    )
    print(f"\n  WorkerRole(specialty='...') → {worker.role.value}")
    print(f"    Can finish: {worker.can_finish}, Can use tools: {worker.can_use_tools}")
    
    reviewer_role = ReviewerRole(criteria=["accuracy", "clarity"])
    reviewer = Agent(
        name="QA",
        model=model,
        role=reviewer_role,
    )
    print(f"\n  ReviewerRole(criteria=[...]) → {reviewer.role.value}")
    print(f"    Can finish: {reviewer.can_finish}, Can use tools: {reviewer.can_use_tools}")
    
    # Backwards compatible: String/enum still works
    print("\n--- Backwards Compatible (String/Enum) ---")
    
    simple = Agent(
        name="Simple",
        model=model,
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
            print(f"    ❌ No command detected")


async def demo_supervisor_flow():
    """Show supervisor coordinating workers."""
    print("\n--- Supervisor Flow Pattern ---")
    print("  Pattern: SUPERVISOR ↔ [WORKER, WORKER]")
    print("  SUPERVISOR delegates and finishes; WORKERs do tool work")
    
    model = get_model()
    
    from agenticflow import SupervisorRole, WorkerRole
    
    # Create team
    supervisor = Agent(
        name="Manager",
        model=model,
        role=SupervisorRole(workers=["Researcher", "Writer"]),
        instructions="Coordinate the team to analyze the topic. Delegate to workers, then synthesize.",
    )
    
    researcher = Agent(
        name="Researcher",
        model=model,
        role=WorkerRole(),
        instructions="Research and provide key facts.",
    )
    
    writer = Agent(
        name="Writer",
        model=model,
        role=WorkerRole(),
        instructions="Write clear summaries.",
    )
    
    # Run with supervisor pattern
    from agenticflow import supervisor as supervisor_pattern
    
    flow = supervisor_pattern(
        supervisor=supervisor,
        workers=[researcher, writer],
        observer=Observer.normal(),
    )
    
    result = await flow.run("Benefits of remote work")
    print(f"\n  Result: {result.output[:200]}...")


async def demo_review_flow():
    """Show reviewer checking work."""
    print("\n--- Pipeline with Review Pattern ---")
    print("  Pattern: WORKER → REVIEWER")
    print("  WORKER does work; REVIEWER approves/rejects and finishes")
    
    model = get_model()
    
    from agenticflow import WorkerRole, ReviewerRole
    
    writer = Agent(
        name="Writer",
        model=model,
        role=WorkerRole(),
        instructions="Write a short paragraph about the topic.",
    )
    
    reviewer = Agent(
        name="QA",
        model=model,
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
    print(f"\n  Final output: {result.output[:200]}...")


async def demo_autonomous():
    """Show autonomous agent working independently."""
    print("\n--- Autonomous Pattern ---")
    print("  Pattern: Single AUTONOMOUS agent")
    print("  Can use tools AND finish (perfect for solo tasks)")
    
    model = get_model()
    from agenticflow import AutonomousRole
    
    assistant = Agent(
        name="Assistant",
        model=model,
        role=AutonomousRole(),
        instructions="Help the user with their request. Answer directly.",
    )
    
    # For single agent, just run directly (no Flow needed)
    result = await assistant.run("What's 2+2?")
    print(f"\n  Answer: {result[:200]}...")


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
