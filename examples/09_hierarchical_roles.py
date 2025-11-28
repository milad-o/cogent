"""
Demo: Hierarchical Topology with Role-Based Levels

Uses the clean 4-role system for auto-inferred hierarchy:
- SUPERVISOR at top: Can finish, can delegate (Directors/Managers)
- WORKER at bottom: Can use tools, cannot finish (Analysts/Writers)

Role System:
┌─────────────┬────────────┬──────────────┬───────────────┐
│ Role        │ can_finish │ can_delegate │ can_use_tools │
├─────────────┼────────────┼──────────────┼───────────────┤
│ SUPERVISOR  │     ✅     │      ✅      │      ❌       │
│ WORKER      │     ❌     │      ❌      │      ✅       │
└─────────────┴────────────┴──────────────┴───────────────┘

Hierarchy is auto-inferred:
  Level 0: SUPERVISOR (top, can finish)
  Level 1: WORKER (bottom, does work)

Usage:
    uv run python examples/09_hierarchical_roles.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow, FlowObserver
from agenticflow.core.enums import AgentRole
from agenticflow.models import ChatModel

load_dotenv()


async def main() -> None:
    print("=" * 60)
    print("  Hierarchical Topology: Clean Role-Based System")
    print("=" * 60)

    model = ChatModel(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Define agents with roles - hierarchy is auto-inferred!
    
    # Top level: SUPERVISOR (can finish, can delegate)
    director = Agent(
        name="Director",
        model=model,
        role=AgentRole.SUPERVISOR,
        instructions="""You are the Director overseeing this project.

You manage two teams: Research (Analyst) and Content (Writer).
Delegate tasks to the right worker, then synthesize results.

When all work is complete, output "FINAL ANSWER:" with the final result.""",
    )

    # Bottom level: WORKERS (can use tools, cannot finish)
    analyst = Agent(
        name="Analyst",
        model=model,
        role=AgentRole.WORKER,
        instructions="""Research and analyze data.
Provide factual findings to the Director.
You cannot finish - pass your results back.""",
    )

    writer = Agent(
        name="Writer",
        model=model,
        role=AgentRole.WORKER,
        instructions="""Write content based on research.
Create clear, engaging text for the Director.
You cannot finish - pass your results back.""",
    )

    # Create flow - levels auto-inferred from roles!
    flow = Flow(
        name="org-hierarchy",
        agents=[director, analyst, writer],
        topology="hierarchical",  # Auto-infers levels from roles
        observer=FlowObserver.verbose(),
    )

    # Show the inferred levels
    print("\n📊 Auto-inferred hierarchy from roles:")
    print(f"   Level 0 (Top):    Director [SUPERVISOR] - can finish, can delegate")
    print(f"   Level 1 (Bottom): Analyst, Writer [WORKER] - do work, cannot finish")
    
    print("\n💡 Key Insight:")
    print("   Only SUPERVISOR can finish → forces delegation back up")

    task = "Create a brief analysis of AI trends in 2024"
    print(f"\n📋 Task: {task}")
    print("-" * 60)

    result = await flow.run(task)

    print("-" * 60)
    print("\n✅ Final Result:")
    
    final = result.results[-1]["thought"] if result.results else "No result"
    if "FINAL ANSWER:" in final:
        final = final.split("FINAL ANSWER:", 1)[1].strip()
    print(final[:500] + "..." if len(final) > 500 else final)
    
    print(f"\n   Completed in {result.iteration} iterations")
    print(f"   Agents involved: {[r['agent'] for r in result.results]}")


if __name__ == "__main__":
    asyncio.run(main())
