"""
Demo: Mesh Topology - Collaborative Writers

Three writers with different specialties collaborate in a mesh topology.
- Workers: Technical and Creative writers contribute their expertise
- Supervisor: SEO writer leads, reviews and decides when done

The SUPERVISOR role is the only one who can finish the flow.
Workers can use tools but cannot finish.

Role System:
┌─────────────┬────────────┬──────────────┬───────────────┐
│ Role        │ can_finish │ can_delegate │ can_use_tools │
├─────────────┼────────────┼──────────────┼───────────────┤
│ WORKER      │     ❌     │      ❌      │      ✅       │
│ SUPERVISOR  │     ✅     │      ✅      │      ❌       │
└─────────────┴────────────┴──────────────┴───────────────┘

Usage:
    uv run python examples/08_mesh_writers.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agenticflow import Agent, Flow, FlowObserver
from agenticflow.core.enums import AgentRole
from agenticflow.flow import FlowConfig

load_dotenv()


async def main() -> None:
    print("=" * 60)
    print("  Mesh Topology: Collaborative Blog Writing")
    print("=" * 60)

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Workers - they add their expertise but can't finish
    technical = Agent(
        name="Technical",
        model=model,
        role=AgentRole.WORKER,  # Can contribute, cannot finish
        instructions="Add technical accuracy and facts. Pass to Creative or SEO.",
    )

    creative = Agent(
        name="Creative",
        model=model,
        role=AgentRole.WORKER,  # Can contribute, cannot finish
        instructions="Add engaging storytelling. Pass to Technical or SEO.",
    )

    # Supervisor - leads the team AND decides when done
    seo = Agent(
        name="SEO",
        model=model,
        role=AgentRole.SUPERVISOR,  # Can finish the flow
        instructions="""Optimize structure and formatting. You are the team lead.

IMPORTANT: Before finishing, ensure BOTH Technical AND Creative have contributed.
Check "Previous contributions" - if either is missing, pass to them first.

When ALL contributors have added their input and the post is polished,
output "FINAL ANSWER:" followed by the complete blog post.""",
    )

    # Create mesh topology - all writers can talk to each other
    flow = Flow(
        name="writer-collaboration",
        agents=[technical, creative, seo],
        topology="mesh",
        observer=FlowObserver.verbose(),
        config=FlowConfig(max_iterations=10),  # Allow multiple rounds of collaboration
    )

    task = """Write a short blog post (200-300 words) about "5 Tips for Better Sleep".
Each contributor should add their expertise, then SEO (lead) finalizes."""

    print(f"\nTask: {task}\n")
    print("-" * 60)

    result = await flow.run(task)

    print("-" * 60)
    print("\n📝 FINAL BLOG POST:")
    print("=" * 60)
    
    # Get the final result
    final = result.results[-1]["thought"] if result.results else "No result"
    
    # Extract just the final answer if present
    if "FINAL ANSWER:" in final:
        final = final.split("FINAL ANSWER:", 1)[1].strip()
    
    print(final)
    
    print("=" * 60)
    print(f"\n✅ Completed in {result.iteration} iterations")
    print(f"   Writers involved: {[r['agent'] for r in result.results]}")


if __name__ == "__main__":
    asyncio.run(main())
