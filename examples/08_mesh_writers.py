"""
Demo: Mesh Topology - Collaborative Writers

Mesh topology is for COMPLEX tasks requiring multiple perspectives and iterations.
For simple tasks, use pipeline or single agent instead.

This demo shows mesh with max_rounds=1 (minimal collaboration).
For max collaboration, increase max_rounds (but expect more LLM calls).

Usage:
    uv run python examples/08_mesh_writers.py
"""

import asyncio
import os

from dotenv import load_dotenv

from agenticflow import Agent, Flow, FlowObserver, Channel, ObservabilityLevel
from agenticflow.core.enums import AgentRole
from agenticflow.models.gemini import GeminiChat

load_dotenv()


async def main() -> None:
    print("=" * 60)
    print("  Mesh Topology: Collaborative Blog Writing")
    print("=" * 60)

    model = GeminiChat(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))

    # Three specialists collaborate
    technical = Agent(
        name="Technical",
        model=model,
        role=AgentRole.WORKER,
        instructions="Add technical accuracy and scientific facts.",
    )

    creative = Agent(
        name="Creative", 
        model=model,
        role=AgentRole.WORKER,
        instructions="Add engaging storytelling and emotional hooks.",
    )

    seo = Agent(
        name="SEO",
        model=model,
        role=AgentRole.SUPERVISOR,
        instructions="Optimize structure, add keywords, finalize the post.",
    )

    # Mesh with max_rounds=2: agents see each other's work and refine
    # Round 1: 3 parallel calls (initial contributions)
    # Round 2: 3 parallel calls (agents read & refine based on others' work)
    # Synthesis: 1 call
    # Total: ~7 LLM calls
    
    # Full observability - see everything with no truncation
    observer = FlowObserver(
        level=ObservabilityLevel.TRACE,
        channels=[Channel.ALL],
        truncate=0,           # No truncation - show full content
        show_timestamps=True,
        show_duration=True,
        show_trace_ids=True,  # Show correlation IDs
    )
    
    flow = Flow(
        name="writer-collaboration",
        agents=[technical, creative, seo],
        topology="mesh",
        max_rounds=2,  # Two rounds so agents can see each other's contributions
        observer=observer,
    )

    task = "Write a 100-word blog post about '5 Tips for Better Sleep'."

    print(f"\nTask: {task}")
    print(f"Topology: mesh (3 agents × 2 rounds + synthesis = ~7 LLM calls)")
    print(f"Communication: Agents share outputs after each round")
    print("-" * 60)

    result = await flow.run(task)

    print("-" * 60)
    print("\n📝 FINAL OUTPUT:")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
