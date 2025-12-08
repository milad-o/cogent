"""
Demo: Hierarchical Organization with Role-Based Delegation

Demonstrates a supervisor-worker pattern where:
- SUPERVISOR (Director): Coordinates and delegates work, can finish
- WORKERS (Analyst, Writer): Do specialized work, report back

This example shows:
1. Using supervisor topology for manager-worker coordination
2. Supervisor delegates to appropriate workers (in parallel)
3. Workers complete tasks and report back
4. Supervisor synthesizes results into final answer

Note: The supervisor topology makes multiple LLM calls:
  1. Director plans and assigns work (~2-5s)
  2. Workers execute in parallel (~2-5s each)
  3. Director synthesizes results (~2-5s)

For fastest results, use Groq (GROQ_API_KEY) or gpt-4o-mini.

Usage:
    uv run python examples/09_hierarchical_roles.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent, Flow, Observer
from agenticflow.core.enums import AgentRole


async def main() -> None:
    print("=" * 60)
    print("  Hierarchical Org: Supervisor-Worker Pattern")
    print("=" * 60)

    model = get_model()

    # SUPERVISOR: Coordinates work
    director = Agent(
        name="Director",
        model=model,
        role=AgentRole.SUPERVISOR,
        instructions="You coordinate research projects. Delegate to Analyst (research) and Writer (content). Synthesize their work into final deliverables.",
    )

    # WORKERS: Do specialized work
    analyst = Agent(
        name="Analyst",
        model=model,
        role=AgentRole.WORKER,
        instructions="Research analyst. Find facts, trends, and data. Be concise.",
    )

    writer = Agent(
        name="Writer",
        model=model,
        role=AgentRole.WORKER,
        instructions="Content writer. Create clear, engaging text. Be concise.",
    )

    # Supervisor topology: Director coordinates workers
    flow = Flow(
        name="org-hierarchy",
        agents=[director, analyst, writer],
        topology="supervisor",
        supervisor=director,
        parallel=True,  # Workers run in parallel
        observer=Observer.verbose(),
    )

    print("\n📊 Organization Structure:")
    print("   Director [SUPERVISOR] - coordinates, delegates, synthesizes")
    print("   ├── Analyst [WORKER] - research and analysis")
    print("   └── Writer [WORKER] - content creation")

    task = "List 3 key AI trends in 2024 with brief explanations"
    print(f"\n📋 Task: {task}")
    print("-" * 60)

    result = await flow.run(task)

    print("-" * 60)
    print("\n✅ Final Result:")
    print(result.output[:1000] + "..." if len(result.output) > 1000 else result.output)
    
    print(f"\n   Agents involved: {list(result.agent_outputs.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
