#!/usr/bin/env python3
"""
Parallel Execution Example
==========================

Demonstrates async/parallel processing capabilities:

1. Parallel Tool Execution - Agent executes multiple tools simultaneously
2. Parallel Agent Execution - Multiple agents process the same task concurrently
3. Fan-Out Pattern - Supervisor delegates to workers in parallel

This example uses real OpenAI API calls to show actual performance benefits.
"""

import asyncio
import time
from dotenv import load_dotenv

load_dotenv()

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    ToolRegistry,
)
from agenticflow.topologies import (
    MeshTopology,
    SupervisorTopology,
    TopologyConfig,
    TopologyPolicy,
    ExecutionMode,
)


def create_agent(name: str, role: AgentRole, system_prompt: str) -> Agent:
    """Create an agent with the given configuration."""
    return Agent(
        config=AgentConfig(
            name=name,
            role=role,
            model="gpt-4o-mini",
            system_prompt=system_prompt,
            temperature=0.7,
        ),
        event_bus=EventBus(),
    )


async def demo_parallel_agents():
    """
    Demo 1: Parallel Agent Execution
    
    Run multiple analysts in parallel on the same question.
    Compare sequential vs parallel execution time.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: PARALLEL AGENT EXECUTION")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create 3 analysts with different perspectives
    analysts = [
        Agent(
            config=AgentConfig(
                name="TechAnalyst",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="You are a technology analyst. Analyze from a tech innovation perspective. Be concise (2-3 sentences).",
            ),
            event_bus=event_bus,
        ),
        Agent(
            config=AgentConfig(
                name="MarketAnalyst",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="You are a market analyst. Analyze from a market trends perspective. Be concise (2-3 sentences).",
            ),
            event_bus=event_bus,
        ),
        Agent(
            config=AgentConfig(
                name="RiskAnalyst",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="You are a risk analyst. Analyze from a risk assessment perspective. Be concise (2-3 sentences).",
            ),
            event_bus=event_bus,
        ),
    ]
    
    # Create mesh topology with parallel execution enabled
    topology = MeshTopology(
        config=TopologyConfig(
            name="parallel-analysts",
            max_iterations=5,
        ),
        agents=analysts,
    )
    
    task = "What are the implications of AI agents in enterprise software?"
    
    # Run in parallel
    print(f"\n📋 Task: {task}")
    print("\n🔄 Running 3 analysts in PARALLEL...")
    
    start = time.perf_counter()
    results = await topology.run_parallel(
        task,
        on_agent_complete=lambda name, _: print(f"  ✓ {name} completed"),
    )
    parallel_time = time.perf_counter() - start
    
    print(f"\n⏱️  Parallel execution time: {parallel_time:.2f}s")
    
    # Show results
    print("\n📊 Results from each analyst:")
    for name, thought in results["results"].items():
        print(f"\n  [{name}]:")
        print(f"    {thought[:200]}...")
    
    # Show timing breakdown
    print("\n⏱️  Individual agent timings:")
    for name, ms in results["timing"].items():
        print(f"    {name}: {ms:.0f}ms")
    
    # For comparison, run sequentially
    print("\n🔄 Running same 3 analysts SEQUENTIALLY for comparison...")
    start = time.perf_counter()
    for agent in analysts:
        await agent.think(task)
    sequential_time = time.perf_counter() - start
    
    print(f"\n⏱️  Sequential execution time: {sequential_time:.2f}s")
    print(f"🚀 Parallel speedup: {sequential_time / parallel_time:.2f}x faster!")


async def demo_fan_out_pattern():
    """
    Demo 2: Fan-Out Pattern
    
    Supervisor delegates to multiple workers in parallel,
    then aggregates their results.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: FAN-OUT PATTERN (Parallel Workers)")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create supervisor
    supervisor = Agent(
        config=AgentConfig(
            name="Coordinator",
            role=AgentRole.ORCHESTRATOR,
            model="gpt-4o-mini",
            system_prompt="You coordinate research tasks and synthesize results.",
        ),
        event_bus=event_bus,
    )
    
    # Create specialized workers
    workers = [
        Agent(
            config=AgentConfig(
                name="HistoryResearcher",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="Research historical context. Be concise (2-3 sentences).",
            ),
            event_bus=event_bus,
        ),
        Agent(
            config=AgentConfig(
                name="CurrentEventsResearcher",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="Research current developments. Be concise (2-3 sentences).",
            ),
            event_bus=event_bus,
        ),
        Agent(
            config=AgentConfig(
                name="FutureResearcher",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="Research future predictions and trends. Be concise (2-3 sentences).",
            ),
            event_bus=event_bus,
        ),
    ]
    
    # Create supervisor topology with parallel workers
    policy = TopologyPolicy.supervisor(
        supervisor="Coordinator",
        workers=[w.config.name for w in workers],
        parallel_workers=True,  # Enable parallel execution
    )
    
    topology = SupervisorTopology(
        config=TopologyConfig(
            name="research-team",
            max_iterations=10,
        ),
        agents=[supervisor] + workers,
        supervisor_name="Coordinator",
    )
    
    # Override policy to enable parallel
    topology._policy = policy
    
    task = "Research the evolution of renewable energy"
    
    print(f"\n📋 Task: {task}")
    print("\n🔄 Fan-out to 3 workers in PARALLEL...")
    
    start = time.perf_counter()
    
    # Run workers in parallel
    worker_results = await topology.run_parallel(
        task,
        agent_names=[w.config.name for w in workers],
        on_agent_complete=lambda name, _: print(f"  ✓ {name} completed"),
    )
    
    parallel_time = time.perf_counter() - start
    
    print(f"\n⏱️  Parallel fan-out time: {parallel_time:.2f}s")
    
    # Show results
    print("\n📊 Worker contributions:")
    for name, thought in worker_results["results"].items():
        print(f"\n  [{name}]:")
        print(f"    {thought[:200]}...")
    
    # Supervisor synthesizes
    print("\n🔄 Coordinator synthesizing results...")
    synthesis_prompt = f"""
Task: {task}

Worker contributions:
{chr(10).join(f'- {name}: {thought}' for name, thought in worker_results["results"].items())}

Please synthesize these into a coherent summary.
"""
    
    synthesis = await supervisor.think(synthesis_prompt)
    print(f"\n📝 [Coordinator] Synthesis:")
    print(f"    {synthesis[:300]}...")


async def demo_parallel_perspectives():
    """
    Demo 3: Parallel Perspectives with Merge Strategies
    
    Get multiple perspectives and merge them using different strategies.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: MERGE STRATEGIES")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create debaters with opposing views
    agents = [
        Agent(
            config=AgentConfig(
                name="Optimist",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="You are an optimist. Always find the positive angle. Answer in ONE word: positive, negative, or neutral.",
            ),
            event_bus=event_bus,
        ),
        Agent(
            config=AgentConfig(
                name="Pessimist",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="You are a pessimist. Always find the negative angle. Answer in ONE word: positive, negative, or neutral.",
            ),
            event_bus=event_bus,
        ),
        Agent(
            config=AgentConfig(
                name="Realist",
                role=AgentRole.WORKER,
                model="gpt-4o-mini",
                system_prompt="You are a realist. Give a balanced view. Answer in ONE word: positive, negative, or neutral.",
            ),
            event_bus=event_bus,
        ),
    ]
    
    topology = MeshTopology(
        config=TopologyConfig(name="sentiment-team"),
        agents=agents,
    )
    
    task = "What is the outlook for the tech job market in 2025?"
    
    print(f"\n📋 Task: {task}")
    
    # Test different merge strategies
    strategies = ["combine", "first", "vote"]
    
    for strategy in strategies:
        print(f"\n🔄 Running with merge_strategy='{strategy}'...")
        
        results = await topology.run_parallel(
            task,
            merge_strategy=strategy,
        )
        
        print(f"   Raw results: {results['results']}")
        print(f"   Merged ({strategy}): {results['merged']}")


async def main():
    """Run all parallel execution demos."""
    print("🚀 AgenticFlow - Parallel Execution Demo")
    print("=" * 60)
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please set it in your .env file")
        return
    
    print("✅ OpenAI API key found")
    
    # Run demos
    await demo_parallel_agents()
    await demo_fan_out_pattern()
    await demo_parallel_perspectives()
    
    print("\n" + "=" * 60)
    print("🎉 ALL DEMOS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
