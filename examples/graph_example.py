"""
Graph Example - demonstrates LangGraph integration with handoffs.

This example shows:
1. Building graphs with GraphBuilder
2. Custom state schemas
3. Agent nodes and router nodes
4. Handoffs between agents using Command
5. Running graphs with different stream modes

Usage:
    uv run python examples/graph_example.py
"""

import asyncio

from langchain_core.messages import HumanMessage, AIMessage

from agenticflow.graph import (
    # State
    AgentGraphState,
    create_state_schema,
    merge_states,
    # Builder
    GraphBuilder,
    NodeConfig,
    EdgeConfig,
    # Nodes
    AgentNode,
    RouterNode,
    # Handoffs
    Handoff,
    HandoffType,
    create_handoff,
    # Runner
    GraphRunner,
    RunConfig,
    StreamMode,
)
from agenticflow.memory import memory_checkpointer


# =============================================================================
# Example 1: Basic Graph Building
# =============================================================================

async def basic_graph_example():
    """Demonstrate basic graph building."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Graph Building")
    print("=" * 60)
    
    # Create a graph builder with a name
    builder = GraphBuilder(name="basic-graph")
    
    # Define simple node functions
    def greet(state: dict) -> dict:
        name = state.get("name", "World")
        return {"greeting": f"Hello, {name}!"}
    
    def process(state: dict) -> dict:
        greeting = state.get("greeting", "")
        return {"result": f"Processed: {greeting.upper()}"}
    
    # Add nodes using the fluent API (returns self for chaining)
    builder.add_node("greeter", greet)
    builder.add_node("processor", process)
    
    # Add edges
    builder.add_edge("greeter", "processor")
    builder.add_edge_to_end("processor")
    
    # Set entry point
    builder.set_entry("greeter")
    
    # Build/compile the graph
    graph = builder.compile()
    
    print(f"\n🔧 Built graph with {len(builder._nodes)} nodes")
    print(f"   Nodes: {list(builder._nodes.keys())}")
    
    # Run the graph
    result = await graph.ainvoke({"name": "Alice"})
    
    print(f"\n📤 Result:")
    print(f"   Greeting: {result.get('greeting')}")
    print(f"   Result: {result.get('result')}")
    
    print("\n✅ Basic graph complete")


# =============================================================================
# Example 2: Agent Graph State
# =============================================================================

async def state_schema_example():
    """Demonstrate custom state schemas."""
    print("\n" + "=" * 60)
    print("Example 2: State Schemas")
    print("=" * 60)
    
    # Use the built-in AgentGraphState
    print("\n📋 Built-in AgentGraphState fields:")
    state = AgentGraphState()
    state_dict = state.to_dict()
    for key, value in state_dict.items():
        print(f"   {key}: {type(value).__name__} = {value}")
    
    # Create a custom state with extra fields
    CustomState = create_state_schema(
        extra_fields={
            "sentiment": (str, "neutral"),
            "confidence": (float, 0.0),
            "tags": (list, []),
        }
    )
    
    print(f"\n🔧 Custom state class: {CustomState.__name__}")
    
    # Merge states with proper behavior
    base_state = {
        "messages": [{"role": "user", "content": "Hello"}],
        "task": "greeting",
        "iteration": 1,
    }
    
    update_state = {
        "messages": [{"role": "assistant", "content": "Hi!"}],
        "iteration": 1,  # Will add due to Annotated[int, add]
    }
    
    merged = merge_states(base_state, update_state)
    
    print(f"\n🔀 Merged state:")
    print(f"   Messages: {len(merged.get('messages', []))} (appended)")
    print(f"   Iteration: {merged.get('iteration')} (added)")
    
    print("\n✅ State schema complete")


# =============================================================================
# Example 3: Agent and Router Nodes
# =============================================================================

async def nodes_example():
    """Demonstrate agent and router nodes."""
    print("\n" + "=" * 60)
    print("Example 3: Agent and Router Nodes")
    print("=" * 60)
    
    # Create an agent (to be used with AgentNode)
    from agenticflow.agents import Agent, AgentConfig
    from agenticflow.core import AgentRole
    from agenticflow.events import EventBus
    
    event_bus = EventBus()
    agent = Agent(
        config=AgentConfig(
            name="researcher",
            role=AgentRole.RESEARCHER,
            system_prompt="You are a research assistant.",
        ),
        event_bus=event_bus,
    )
    
    # AgentNode wraps an Agent
    agent_node = AgentNode(name="researcher", agent=agent)
    print(f"\n🤖 Agent Node: {agent_node.name}")
    print(f"   Agent role: {agent_node.agent.role.value}")
    print(f"   Include history: {agent_node.include_history}")
    
    # Create a router node with a routing function
    def route_decision(state: dict) -> str:
        """Decide which agent to route to."""
        task = state.get("task", "")
        if "research" in task.lower():
            return "researcher"
        elif "write" in task.lower():
            return "writer"
        else:
            return "default"
    
    router = RouterNode(name="task_router", route_fn=route_decision)
    
    print(f"\n🔀 Router Node: {router.name}")
    
    # Test routing
    test_states = [
        {"task": "Research AI trends"},
        {"task": "Write a blog post"},
        {"task": "Do something else"},
    ]
    
    print(f"\n📊 Routing decisions:")
    for state in test_states:
        result = await router(state)
        print(f"   '{state['task']}' → {result.get('next_agent')}")
    
    print("\n✅ Nodes example complete")


# =============================================================================
# Example 4: Handoffs Between Agents
# =============================================================================

async def handoffs_example():
    """Demonstrate handoffs between agents."""
    print("\n" + "=" * 60)
    print("Example 4: Handoffs Between Agents")
    print("=" * 60)
    
    # Direct handoff
    direct = Handoff(
        target="writer",
        handoff_type=HandoffType.DIRECT,
        message="Research complete, handing off to writer",
    )
    
    print(f"\n➡️ Direct Handoff:")
    print(f"   Target: {direct.target}")
    print(f"   Type: {direct.handoff_type.value}")
    print(f"   Message: {direct.message}")
    
    # Broadcast handoff (to multiple targets)
    broadcast = Handoff(
        target=["reviewer1", "reviewer2"],
        handoff_type=HandoffType.BROADCAST,
        message="Content ready for parallel review",
    )
    
    print(f"\n📢 Broadcast Handoff:")
    print(f"   Targets: {broadcast.target}")
    print(f"   Type: {broadcast.handoff_type.value}")
    
    # Create handoff using helper (returns Command directly)
    command = create_handoff(
        target="editor",
        message="Content ready for editing",
        state_update={"stage": "editing"},
    )
    
    print(f"\n🛠️ Helper-created Handoff (as Command):")
    print(f"   Type: {type(command).__name__}")
    print(f"   Goto: {command.goto}")
    
    # Convert direct Handoff to LangGraph Command
    direct_command = direct.to_command()
    print(f"\n📦 Direct Handoff as LangGraph Command:")
    print(f"   Type: {type(direct_command).__name__}")
    print(f"   Goto: {direct_command.goto}")
    
    print("\n✅ Handoffs example complete")


# =============================================================================
# Example 5: Graph Runner
# =============================================================================

async def runner_example():
    """Demonstrate graph runner with different modes."""
    print("\n" + "=" * 60)
    print("Example 5: Graph Runner")
    print("=" * 60)
    
    # Build a simple graph
    builder = GraphBuilder(name="runner-graph")
    
    def step1(state: dict) -> dict:
        return {"step": 1, "data": "Step 1 complete"}
    
    def step2(state: dict) -> dict:
        prev = state.get("data", "")
        return {"step": 2, "data": f"{prev} → Step 2 complete"}
    
    def step3(state: dict) -> dict:
        prev = state.get("data", "")
        return {"step": 3, "data": f"{prev} → Step 3 complete", "final": True}
    
    builder.add_node("step1", step1)
    builder.add_node("step2", step2)
    builder.add_node("step3", step3)
    
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", "step3")
    builder.add_edge_to_end("step3")
    builder.set_entry("step1")
    
    graph = builder.compile()
    
    # Create runner
    runner = GraphRunner(graph)
    
    # Run configuration
    config = RunConfig(
        thread_id="test-thread",
        recursion_limit=50,
        timeout_seconds=30,
    )
    
    print(f"\n⚙️ Run Config:")
    print(f"   Thread ID: {config.thread_id}")
    print(f"   Recursion limit: {config.recursion_limit}")
    print(f"   Timeout: {config.timeout_seconds}s")
    
    # Run with invoke (blocking)
    print(f"\n🏃 Running graph...")
    result = await runner.run(
        initial_state={"input": "test"},
        config=config,
    )
    
    print(f"\n📤 Final Result:")
    print(f"   Success: {result.success}")
    print(f"   Steps: {result.steps}")
    print(f"   Final state keys: {list(result.final_state.keys()) if result.final_state else 'None'}")
    
    print(f"\n✅ Runner example complete")


# =============================================================================
# Example 6: Complete Multi-Agent Graph
# =============================================================================

async def multi_agent_graph_example():
    """Demonstrate a complete multi-agent graph."""
    print("\n" + "=" * 60)
    print("Example 6: Complete Multi-Agent Graph")
    print("=" * 60)
    
    # Build a simpler sequential graph that demonstrates multi-agent coordination
    builder = GraphBuilder(name="multi-agent-graph")
    
    # Sequential processing: research -> write -> review -> done
    def research_step(state: dict) -> dict:
        return {
            "current_step": "research",
            "research_result": "Found relevant data about AI",
        }
    
    def write_step(state: dict) -> dict:
        research = state.get("research_result", "")
        return {
            "current_step": "write",
            "draft": f"Based on: {research[:30]}... - Draft content created",
        }
    
    def review_step(state: dict) -> dict:
        draft = state.get("draft", "")
        return {
            "current_step": "review",
            "review_result": "Approved",
            "final_output": f"Reviewed: {draft[:40]}...",
        }
    
    # Add nodes
    builder.add_node("research", research_step)
    builder.add_node("write", write_step)
    builder.add_node("review", review_step)
    
    # Sequential edges
    builder.add_edge("research", "write")
    builder.add_edge("write", "review")
    builder.add_edge_to_end("review")
    builder.set_entry("research")
    
    # Compile the graph
    graph = builder.compile()
    
    print(f"\n🔧 Built multi-agent graph:")
    print(f"   Nodes: research, write, review")
    print(f"   Flow: research → write → review → end")
    
    # Run the graph
    print(f"\n🏃 Running multi-agent workflow...")
    
    initial_state = {"task": "Research AI and write a summary"}
    
    result = await graph.ainvoke(initial_state)
    
    print(f"\n📤 Final Results:")
    print(f"   Current step: {result.get('current_step')}")
    print(f"   Research: {result.get('research_result', '')[:50]}...")
    print(f"   Draft: {result.get('draft', '')[:50]}...")
    print(f"   Review: {result.get('review_result')}")
    
    print("\n✅ Multi-agent graph complete")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all graph examples."""
    print("\n" + "📈 " * 20)
    print("AgenticFlow Graph Examples")
    print("📈 " * 20)
    
    await basic_graph_example()
    await state_schema_example()
    await nodes_example()
    await handoffs_example()
    await runner_example()
    await multi_agent_graph_example()
    
    print("\n" + "=" * 60)
    print("All graph examples complete!")
    print("=" * 60)
    
    print("\n💡 Key concepts:")
    print("   - GraphBuilder: Fluent API for building LangGraph graphs")
    print("   - AgentGraphState: Standard state with proper merge behaviors")
    print("   - Handoff: Explicit agent-to-agent transitions")
    print("   - GraphRunner: Execute graphs with streaming support")


if __name__ == "__main__":
    asyncio.run(main())
