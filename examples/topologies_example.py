"""
Topologies Example - demonstrates multi-agent coordination patterns.

This example shows:
1. Supervisor Topology - one agent coordinates workers
2. Pipeline Topology - sequential processing through stages
3. Mesh Topology - peer-to-peer collaboration
4. Hierarchical Topology - nested team structure

Usage:
    uv run python examples/topologies_example.py
"""

import asyncio

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    ToolRegistry,
)
from agenticflow.topologies import (
    TopologyFactory,
    TopologyType,
    TopologyConfig,
    TopologyState,
    SupervisorTopology,
    PipelineTopology,
    MeshTopology,
    HierarchicalTopology,
)

# For memory, use LangGraph directly:
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


# =============================================================================
# Helper: Create test agents (no LLM required)
# =============================================================================

def create_test_agents(event_bus: EventBus, tool_registry: ToolRegistry | None = None) -> dict[str, Agent]:
    """Create a set of test agents with different roles."""
    agents = {}
    
    # Orchestrator agent (supervisor)
    agents["orchestrator"] = Agent(
        config=AgentConfig(
            name="Orchestrator",
            role=AgentRole.ORCHESTRATOR,
            description="Coordinates team work and delegates tasks",
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    # Researcher agent
    agents["researcher"] = Agent(
        config=AgentConfig(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            description="Researches topics and gathers information",
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    # Planner agent
    agents["planner"] = Agent(
        config=AgentConfig(
            name="Planner",
            role=AgentRole.PLANNER,
            description="Creates execution plans",
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    # Worker agent
    agents["worker"] = Agent(
        config=AgentConfig(
            name="Worker",
            role=AgentRole.WORKER,
            description="Executes tasks",
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    # Validator agent
    agents["validator"] = Agent(
        config=AgentConfig(
            name="Validator",
            role=AgentRole.VALIDATOR,
            description="Reviews and validates outputs",
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )
    
    return agents


# =============================================================================
# Example 1: Topology Types Overview
# =============================================================================

async def topology_types_example():
    """Demonstrate available topology types."""
    print("\n" + "=" * 60)
    print("Example 1: Topology Types Overview")
    print("=" * 60)
    
    print("""
   📋 Available Topology Types:
   
   1. SUPERVISOR (TopologyType.SUPERVISOR)
      - One orchestrator coordinates multiple workers
      - Orchestrator receives tasks and delegates
      - Workers report results back to orchestrator
      - Use: Clear hierarchy, centralized control
      
   2. PIPELINE (TopologyType.PIPELINE)
      - Sequential processing through stages
      - Each agent processes and passes to next
      - Data flows in one direction
      - Use: Sequential workflows, data processing
      
   3. MESH (TopologyType.MESH)
      - Peer-to-peer collaboration
      - Any agent can communicate with any other
      - Distributed decision making
      - Use: Collaborative tasks, brainstorming
      
   4. HIERARCHICAL (TopologyType.HIERARCHICAL)
      - Nested team structure
      - Multiple levels of orchestrators
      - Complex organization patterns
      - Use: Large teams, department-like structure
   """)
    
    print("📋 TopologyType enum values:")
    for t in TopologyType:
        print(f"   - {t.name}: {t.value}")
    
    print("\n✅ Topology types overview complete")


# =============================================================================
# Example 2: Topology Configuration
# =============================================================================

async def topology_config_example():
    """Demonstrate topology configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Topology Configuration")
    print("=" * 60)
    
    event_bus = EventBus()
    agents = create_test_agents(event_bus)
    
    # Create configuration
    config = TopologyConfig(
        name="content-team",
        description="A team for content creation",
        max_iterations=10,
        recursion_limit=50,
        metadata={"team": "content", "version": "1.0"},
    )
    
    print(f"\n📋 Topology Configuration:")
    print(f"   Name: {config.name}")
    print(f"   Description: {config.description}")
    print(f"   Max iterations: {config.max_iterations}")
    print(f"   Recursion limit: {config.recursion_limit}")
    print(f"   Metadata: {config.metadata}")
    
    print("\n✅ Topology configuration complete")


# =============================================================================
# Example 3: Supervisor Topology
# =============================================================================

async def supervisor_topology_example():
    """Demonstrate supervisor topology setup."""
    print("\n" + "=" * 60)
    print("Example 3: Supervisor Topology")
    print("=" * 60)
    
    print("""
   Pattern: One orchestrator coordinates multiple workers
            Orchestrator → Worker1, Worker2, Worker3
            Workers report back to orchestrator
   """)
    
    event_bus = EventBus()
    agents = create_test_agents(event_bus)
    
    # Create supervisor topology using factory
    supervisor_agents = [
        agents["orchestrator"],
        agents["researcher"],
        agents["worker"],
        agents["validator"],
    ]
    
    topology = TopologyFactory.create(
        topology_type=TopologyType.SUPERVISOR,
        name="content-production",
        agents=supervisor_agents,
        supervisor_name="Orchestrator",
    )
    
    print(f"\n🔗 Created Topology:")
    print(f"   Name: {topology.config.name}")
    print(f"   Type: Supervisor")
    print(f"   Agents: {list(topology.agents.keys())}")
    print(f"   Supervisor: Orchestrator")
    
    # Show agent roles
    print(f"\n👥 Agent Roles:")
    for name, agent in topology.agents.items():
        role = "SUPERVISOR" if name == "Orchestrator" else "WORKER"
        print(f"   {name}: {role}")
    
    print("\n✅ Supervisor topology complete")


# =============================================================================
# Example 4: Pipeline Topology
# =============================================================================

async def pipeline_topology_example():
    """Demonstrate pipeline topology setup."""
    print("\n" + "=" * 60)
    print("Example 4: Pipeline Topology")
    print("=" * 60)
    
    print("""
   Pattern: Sequential processing through stages
            Input → Stage1 → Stage2 → Stage3 → Output
            Each stage processes and passes to next
   """)
    
    event_bus = EventBus()
    agents = create_test_agents(event_bus)
    
    # Create pipeline with specific order
    pipeline_agents = [
        agents["researcher"],  # Stage 1: Research
        agents["planner"],     # Stage 2: Plan
        agents["worker"],      # Stage 3: Execute
        agents["validator"],   # Stage 4: Validate
    ]
    
    topology = TopologyFactory.create(
        topology_type=TopologyType.PIPELINE,
        name="content-pipeline",
        agents=pipeline_agents,
    )
    
    print(f"\n🔗 Created Pipeline:")
    print(f"   Name: {topology.config.name}")
    
    print(f"\n📊 Pipeline Stages:")
    for i, (name, agent) in enumerate(topology.agents.items(), 1):
        arrow = "→" if i < len(topology.agents) else "✓"
        print(f"   Stage {i}: {name} ({agent.role.value}) {arrow}")
    
    print("\n✅ Pipeline topology complete")


# =============================================================================
# Example 5: Mesh Topology
# =============================================================================

async def mesh_topology_example():
    """Demonstrate mesh topology setup."""
    print("\n" + "=" * 60)
    print("Example 5: Mesh Topology")
    print("=" * 60)
    
    print("""
   Pattern: Peer-to-peer collaboration
            Agent1 ↔ Agent2
              ↕   ╲╱   ↕
            Agent3 ↔ Agent4
            Any agent can communicate with any other
   """)
    
    event_bus = EventBus()
    agents = create_test_agents(event_bus)
    
    # Create mesh topology
    mesh_agents = [
        agents["researcher"],
        agents["planner"],
        agents["worker"],
    ]
    
    topology = TopologyFactory.create(
        topology_type=TopologyType.MESH,
        name="collaborative-team",
        agents=mesh_agents,
    )
    
    print(f"\n🔗 Created Mesh:")
    print(f"   Name: {topology.config.name}")
    print(f"   Agents: {list(topology.agents.keys())}")
    
    # Show connections
    print(f"\n📊 Mesh Connections:")
    for name in topology.agents.keys():
        others = [n for n in topology.agents.keys() if n != name]
        print(f"   {name} ↔ {', '.join(others)}")
    
    print("\n✅ Mesh topology complete")


# =============================================================================
# Example 6: Hierarchical Topology
# =============================================================================

async def hierarchical_topology_example():
    """Demonstrate hierarchical topology setup."""
    print("\n" + "=" * 60)
    print("Example 6: Hierarchical Topology")
    print("=" * 60)
    
    print("""
   Pattern: Nested team structure
            
                 Top Orchestrator
                    /       \\
           Team Lead 1    Team Lead 2
              /  \\           /  \\
           W1    W2       W3    W4
   """)
    
    event_bus = EventBus()
    
    # Create teams
    top_orchestrator = Agent(
        config=AgentConfig(name="CEO", role=AgentRole.ORCHESTRATOR),
        event_bus=event_bus,
    )
    
    team_lead_1 = Agent(
        config=AgentConfig(name="Research_Lead", role=AgentRole.ORCHESTRATOR),
        event_bus=event_bus,
    )
    
    team_lead_2 = Agent(
        config=AgentConfig(name="Production_Lead", role=AgentRole.ORCHESTRATOR),
        event_bus=event_bus,
    )
    
    worker1 = Agent(config=AgentConfig(name="Researcher1", role=AgentRole.WORKER), event_bus=event_bus)
    worker2 = Agent(config=AgentConfig(name="Researcher2", role=AgentRole.WORKER), event_bus=event_bus)
    worker3 = Agent(config=AgentConfig(name="Producer1", role=AgentRole.WORKER), event_bus=event_bus)
    worker4 = Agent(config=AgentConfig(name="Producer2", role=AgentRole.WORKER), event_bus=event_bus)
    
    workers = [worker1, worker2, worker3, worker4]
    all_agents = [top_orchestrator, team_lead_1, team_lead_2] + workers
    
    # Define the hierarchy: manager -> direct reports
    hierarchy = {
        "CEO": ["Research_Lead", "Production_Lead"],
        "Research_Lead": ["Researcher1", "Researcher2"],
        "Production_Lead": ["Producer1", "Producer2"],
    }
    
    topology = TopologyFactory.create(
        topology_type=TopologyType.HIERARCHICAL,
        name="organization",
        agents=all_agents,
        hierarchy=hierarchy,
        root="CEO",
    )
    
    print(f"\n🔗 Created Hierarchy:")
    print(f"   Name: {topology.config.name}")
    print(f"   Total agents: {len(topology.agents)}")
    print(f"   Root: {topology.root}")
    
    print(f"\n📊 Organization Structure:")
    print(f"   Level 0: CEO (Top)")
    print(f"   Level 1: Research_Lead, Production_Lead (Leads)")
    print(f"   Level 2: {', '.join(w.name for w in workers)} (Workers)")
    
    print("\n✅ Hierarchical topology complete")


# =============================================================================
# Example 7: Topology with Memory
# =============================================================================

async def topology_with_memory_example():
    """Demonstrate topology with memory integration."""
    print("\n" + "=" * 60)
    print("Example 7: Topology with Memory")
    print("=" * 60)
    
    event_bus = EventBus()
    agents = create_test_agents(event_bus)
    
    # Create memory components using LangGraph directly
    checkpointer = MemorySaver()
    store = InMemoryStore()
    
    # Store team context in long-term memory
    team_namespace = ("team", "content")
    store.put(team_namespace, "guidelines", {"value": "Be concise and accurate"})
    store.put(team_namespace, "style", {"value": "Professional tone"})
    
    # Create topology with memory
    topology = TopologyFactory.create(
        topology_type=TopologyType.SUPERVISOR,
        name="memory-team",
        agents=[agents["orchestrator"], agents["researcher"], agents["worker"]],
        supervisor_name="Orchestrator",
        checkpointer=checkpointer,
        store=store,
    )
    
    print(f"\n🔗 Topology with Memory:")
    print(f"   Name: {topology.config.name}")
    print(f"   Has checkpointer: {topology.checkpointer is not None}")
    print(f"   Has store: {topology.store is not None}")
    
    # Show stored team context
    items = store.search(team_namespace)
    print(f"\n📦 Team Context:")
    for item in items:
        print(f"   {item.key}: {item.value['value']}")
    
    print("""
   Memory Integration:
   - Checkpointer: Saves conversation state per thread
   - Store: Shared team knowledge and context
   """)
    
    print("\n✅ Topology with memory complete")


# =============================================================================
# Example 8: Topology State
# =============================================================================

async def topology_state_example():
    """Demonstrate topology state tracking."""
    print("\n" + "=" * 60)
    print("Example 8: Topology State")
    print("=" * 60)
    
    # Show TopologyState structure
    print("""
   TopologyState tracks:
   - messages: Conversation history
   - current_agent: Currently active agent
   - task: The current task being processed
   - context: Additional context data
   - iteration: Current iteration count
   - completed: Whether processing is complete
   - error: Any error that occurred
   - results: Results from completed processing
   
   Example state flow:
   ```python
   state = TopologyState(
       messages=[],
       current_agent="researcher",
       task="Analyze this data",
       iteration=0,
   )
   
   # After processing
   state.results.append({"agent": "researcher", "output": "Found 5 insights"})
   state.current_agent = "writer"
   state.iteration = 1
   ```
   """)
    
    # Create sample state
    state = TopologyState(
        messages=[],
        current_agent="orchestrator",
        task="Write a report",
        context={"started_at": "2024-01-01T10:00:00"},
        iteration=0,
    )
    
    print(f"\n📊 Sample State:")
    print(f"   Current agent: {state.current_agent}")
    print(f"   Task: {state.task}")
    print(f"   Iteration: {state.iteration}")
    print(f"   Messages: {len(state.messages)}")
    print(f"   Results: {state.results}")
    print(f"   Completed: {state.completed}")
    
    # Convert to dict for serialization
    state_dict = state.to_dict()
    print(f"\n📦 State as dict (for serialization):")
    print(f"   {state_dict}")
    
    print("\n✅ Topology state complete")


# =============================================================================
# Example 9: Choosing the Right Topology
# =============================================================================

async def choosing_topology_example():
    """Guide for choosing the right topology."""
    print("\n" + "=" * 60)
    print("Example 9: Choosing the Right Topology")
    print("=" * 60)
    
    print("""
   📋 Topology Selection Guide:
   
   ┌─────────────────┬────────────────────────────────────────┐
   │ Topology        │ Best For                               │
   ├─────────────────┼────────────────────────────────────────┤
   │ SUPERVISOR      │ • Clear task delegation                │
   │                 │ • Centralized decision making          │
   │                 │ • Quality control needs                │
   │                 │ • Teams with distinct roles            │
   ├─────────────────┼────────────────────────────────────────┤
   │ PIPELINE        │ • Sequential workflows                 │
   │                 │ • Data transformation                  │
   │                 │ • Content creation flows               │
   │                 │ • Each stage has specific function     │
   ├─────────────────┼────────────────────────────────────────┤
   │ MESH            │ • Collaborative brainstorming          │
   │                 │ • Peer review processes                │
   │                 │ • Complex problem solving              │
   │                 │ • No clear hierarchy needed            │
   ├─────────────────┼────────────────────────────────────────┤
   │ HIERARCHICAL    │ • Large organizations                  │
   │                 │ • Multi-level management               │
   │                 │ • Department-based structure           │
   │                 │ • Complex delegation chains            │
   └─────────────────┴────────────────────────────────────────┘
   
   📝 Example Use Cases:
   
   • Blog post creation → PIPELINE
     (Research → Write → Edit → Publish)
   
   • Customer support → SUPERVISOR
     (Supervisor routes to specialists)
   
   • Code review → MESH
     (Multiple reviewers collaborate)
   
   • Enterprise workflow → HIERARCHICAL
     (CEO → Directors → Managers → Workers)
   """)
    
    print("\n✅ Topology selection guide complete")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all topology examples."""
    print("\n" + "🔗 " * 20)
    print("AgenticFlow Topologies Examples")
    print("🔗 " * 20)
    
    await topology_types_example()
    await topology_config_example()
    await supervisor_topology_example()
    await pipeline_topology_example()
    await mesh_topology_example()
    await hierarchical_topology_example()
    await topology_with_memory_example()
    await topology_state_example()
    await choosing_topology_example()
    
    print("\n" + "=" * 60)
    print("All topology examples complete!")
    print("=" * 60)
    
    print("\n💡 Topology features:")
    print("   - Multiple coordination patterns")
    print("   - Factory for easy creation")
    print("   - Memory integration")
    print("   - State tracking")
    print("   - Flexible configuration")


if __name__ == "__main__":
    asyncio.run(main())
