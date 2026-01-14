"""
Demo: Context Management Strategies for Mesh Topology

Shows how to use different context strategies to prevent context explosion
in multi-round collaboration.

Strategies demonstrated:
1. SlidingWindowStrategy - Keep only last N rounds (default)
2. SummarizationStrategy - LLM-compress older rounds
3. StructuredHandoffStrategy - Extract decisions/findings as JSON

Usage:
    uv run python examples/topologies/context_strategies.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model, settings

from agenticflow import Agent
from agenticflow.core.enums import AgentRole
from agenticflow.topologies import (
    Mesh,
    AgentConfig,
    SlidingWindowStrategy,
    SummarizationStrategy,
    StructuredHandoffStrategy,
)


async def demo_sliding_window() -> None:
    """Demo: SlidingWindowStrategy - simplest, no LLM calls."""
    print("\n" + "=" * 60)
    print("  Strategy 1: Sliding Window (keeps last N rounds)")
    print("=" * 60)
    
    model = get_model()
    
    # Set up observability
    from agenticflow.observability.bus import TraceBus
    trace_bus = TraceBus()
    
    # Subscribe to context events
    async def on_context_event(event):
        print(f"  📊 TRACE: {event.type.value}")
        for k, v in event.data.items():
            print(f"      {k}: {v}")
    
    trace_bus.subscribe_all(on_context_event)
    
    analyst1 = Agent(
        name="Business",
        model=model,
        role=AgentRole.WORKER,
        instructions="Analyze from a business perspective.",
    )
    analyst2 = Agent(
        name="Technical",
        model=model,
        role=AgentRole.WORKER,
        instructions="Analyze from a technical perspective.",
    )
    
    # With 3 rounds but only keeping last 2: prevents context explosion
    # Pass trace_bus to get observability events
    mesh = Mesh(
        agents=[
            AgentConfig(agent=analyst1, role="business analyst"),
            AgentConfig(agent=analyst2, role="technical analyst"),
        ],
        max_rounds=3,
        context_strategy=SlidingWindowStrategy(max_rounds=2, trace_bus=trace_bus),
    )
    
    print(f"\nConfig: max_rounds=3, but only last 2 rounds of context passed")
    print(f"Result: Context stays bounded even with many rounds")
    print(f"\n🔍 Watch for TRACE events showing context management:\n")
    
    result = await mesh.run("Should we build or buy a CRM system?")
    
    print("-" * 60)
    print("📝 OUTPUT:")
    print(result.output[:500] + "..." if len(result.output) > 500 else result.output)


async def demo_summarization() -> None:
    """Demo: SummarizationStrategy - LLM compresses older rounds."""
    print("\n" + "=" * 60)
    print("  Strategy 2: Summarization (LLM-compress older rounds)")
    print("=" * 60)
    
    model = get_model()
    
    analyst1 = Agent(
        name="Optimist",
        model=model,
        role=AgentRole.WORKER,
        instructions="See the opportunities and upsides.",
    )
    analyst2 = Agent(
        name="Skeptic",
        model=model,
        role=AgentRole.WORKER,
        instructions="Identify risks and potential issues.",
    )
    
    # Summarize older rounds, keep last 1 round in full
    mesh = Mesh(
        agents=[
            AgentConfig(agent=analyst1, role="optimistic analyst"),
            AgentConfig(agent=analyst2, role="skeptical analyst"),
        ],
        max_rounds=3,
        context_strategy=SummarizationStrategy(
            model=model,
            keep_full_rounds=1,  # Keep only last round in full
            max_summary_tokens=200,
        ),
    )
    
    print(f"\nConfig: LLM summarizes rounds 1-2, round 3 stays full")
    print(f"Result: Rich context without token explosion\n")
    
    result = await mesh.run("Evaluate the risks of migrating to cloud.")
    
    print("-" * 60)
    print("📝 OUTPUT:")
    print(result.output[:500] + "..." if len(result.output) > 500 else result.output)


async def demo_structured_handoff() -> None:
    """Demo: StructuredHandoffStrategy - extract key decisions/findings."""
    print("\n" + "=" * 60)
    print("  Strategy 3: Structured Handoff (extract decisions/findings)")
    print("=" * 60)
    
    model = get_model()
    
    analyst1 = Agent(
        name="Researcher",
        model=model,
        role=AgentRole.WORKER,
        instructions="Research facts and data.",
    )
    analyst2 = Agent(
        name="Strategist",
        model=model,
        role=AgentRole.WORKER,
        instructions="Develop strategic recommendations.",
    )
    
    # Extract structured data instead of passing full text
    mesh = Mesh(
        agents=[
            AgentConfig(agent=analyst1, role="market researcher"),
            AgentConfig(agent=analyst2, role="strategy consultant"),
        ],
        max_rounds=2,
        context_strategy=StructuredHandoffStrategy(
            model=model,
            max_items_per_category=5,
        ),
    )
    
    print(f"\nConfig: Extract decisions, findings, questions as structured data")
    print(f"Result: Clean, focused context - no rambling text\n")
    
    result = await mesh.run("What should our pricing strategy be for the EU market?")
    
    print("-" * 60)
    print("📝 OUTPUT:")
    print(result.output[:500] + "..." if len(result.output) > 500 else result.output)


async def main() -> None:
    print("=" * 60)
    print("  Context Management Strategies Demo")
    print("=" * 60)
    print("\nThese strategies prevent context explosion in multi-round topologies.")
    print("Without them, 5 agents × 10 rounds would pass 25,000+ tokens of history!\n")
    
    # Run all demos
    await demo_sliding_window()
    await demo_summarization()
    await demo_structured_handoff()
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print("""
| Strategy              | Best For                        | LLM Calls |
|-----------------------|--------------------------------|-----------|
| SlidingWindowStrategy | Simple tasks, recent context   | None      |
| SummarizationStrategy | Long collaborations, nuance    | Yes       |
| StructuredHandoff     | Decision-making, clarity       | Yes       |
| RetrievalStrategy     | Very long history, search      | None      |
| BlackboardStrategy    | Shared team state              | None      |
""")


if __name__ == "__main__":
    asyncio.run(main())
