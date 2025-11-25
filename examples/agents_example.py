"""
Agents Example - demonstrates agents and orchestration.

This example shows:
1. Creating agents with configuration
2. Agent state and lifecycle
3. Tools and tool registry
4. Orchestrator patterns

Usage:
    uv run python examples/agents_example.py

Note: Running agents requires API keys for LLM providers.
Set OPENAI_API_KEY environment variable if using OpenAI models.
"""

import asyncio
from langchain.tools import tool

from agenticflow import (
    EventBus,
    Event,
    Agent,
    AgentConfig,
    AgentRole,
    AgentStatus,
    TaskManager,
    Task,
    TaskStatus,
    ToolRegistry,
    create_tool_from_function,
    Orchestrator,
)


# =============================================================================
# Example 1: Basic Agent Creation
# =============================================================================

async def basic_agent_example():
    """Demonstrate basic agent creation."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Agent Creation")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create an agent with basic configuration
    config = AgentConfig(
        name="Research Assistant",
        role=AgentRole.WORKER,
        description="An agent that helps with research tasks",
        system_prompt="You are a helpful research assistant. Be thorough and accurate.",
    )
    
    agent = Agent(
        config=config,
        event_bus=event_bus,
    )
    
    print(f"\n🤖 Created Agent:")
    print(f"   ID: {agent.id[:8]}...")
    print(f"   Name: {agent.name}")
    print(f"   Role: {agent.role.value}")
    print(f"   Status: {agent.status.value}")
    print(f"   Description: {agent.config.description}")
    
    print("\n✅ Basic agent creation complete")


# =============================================================================
# Example 2: Agent Configuration Options
# =============================================================================

async def configuration_example():
    """Demonstrate agent configuration options."""
    print("\n" + "=" * 60)
    print("Example 2: Agent Configuration Options")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Models should be created using LangChain directly:
    # from langchain_openai import ChatOpenAI
    # model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    # Different role types (showing config without actual model for demo)
    configs = [
        AgentConfig(
            name="Orchestrator",
            role=AgentRole.ORCHESTRATOR,
            description="Coordinates other agents",
            # model=ChatOpenAI(model="gpt-4o", temperature=0.3),
            temperature=0.3,  # Lower for consistency
        ),
        AgentConfig(
            name="Specialist",
            role=AgentRole.SPECIALIST,
            description="Expert in data analysis",
            # model=ChatOpenAI(model="gpt-4o-mini"),
            temperature=0.5,
            tools=["analyze_data", "create_chart"],  # Allowed tools
        ),
        AgentConfig(
            name="Worker",
            role=AgentRole.WORKER,
            description="General purpose worker",
            # model=ChatOpenAI(model="gpt-4o-mini"),
            max_concurrent_tasks=3,
            timeout_seconds=60.0,
        ),
    ]
    
    print("\n📋 Agent Configurations:")
    print("\n   💡 Models are passed as LangChain objects:")
    print("      from langchain_openai import ChatOpenAI")
    print("      model = ChatOpenAI(model='gpt-4o')")
    for config in configs:
        print(f"\n   🤖 {config.name} ({config.role.value})")
        print(f"      Temperature: {config.temperature}")
        print(f"      Tools: {config.tools or 'all'}")
        print(f"      Max concurrent: {config.max_concurrent_tasks}")
        print(f"      Timeout: {config.timeout_seconds}s")
    
    print("\n✅ Agent configuration complete")


# =============================================================================
# Example 3: Tool Registry
# =============================================================================

async def tool_registry_example():
    """Demonstrate tool registry."""
    print("\n" + "=" * 60)
    print("Example 3: Tool Registry")
    print("=" * 60)
    
    # Define tools using the @tool decorator
    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            result = eval(expression)  # Note: in production, use a safe evaluator
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    @tool
    def search_knowledge(query: str) -> str:
        """Search the knowledge base for information."""
        # Simulated search
        return f"Found 5 results for: {query}"
    
    @tool
    def summarize_text(text: str, max_length: int = 100) -> str:
        """Summarize text to a maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    # Create registry and register tools
    registry = ToolRegistry()
    registry.register(calculate)
    registry.register(search_knowledge)
    registry.register(summarize_text)
    
    print(f"\n🔧 Registered Tools: {len(registry)}")
    
    for name in registry.tool_names:
        tool_obj = registry.get(name)
        print(f"\n   📦 {name}")
        print(f"      Description: {tool_obj.description}")
    
    # Get tool descriptions for LLM prompts
    print(f"\n📝 Tool descriptions for LLM:")
    print(registry.get_tool_descriptions())
    
    # Test tool execution
    print(f"\n⚡ Testing tools:")
    result = calculate.invoke({"expression": "2 + 2 * 3"})
    print(f"   calculate('2 + 2 * 3') = {result}")
    
    result = search_knowledge.invoke({"query": "AI trends"})
    print(f"   search_knowledge('AI trends') = {result}")
    
    print("\n✅ Tool registry complete")


# =============================================================================
# Example 4: Creating Tool from Function
# =============================================================================

async def create_tool_example():
    """Demonstrate creating tools from functions."""
    print("\n" + "=" * 60)
    print("Example 4: Creating Tool from Function")
    print("=" * 60)
    
    # Define a regular function
    def fetch_weather(city: str, units: str = "celsius") -> dict:
        """Fetch weather for a city."""
        # Simulated weather data
        return {
            "city": city,
            "temperature": 22 if units == "celsius" else 72,
            "units": units,
            "condition": "sunny",
        }
    
    # Convert to LangChain tool
    weather_tool = create_tool_from_function(
        fetch_weather,
        name="get_weather",
        description="Get current weather for a city",
    )
    
    print(f"\n🔧 Created Tool:")
    print(f"   Name: {weather_tool.name}")
    print(f"   Description: {weather_tool.description}")
    
    # Test the tool
    result = weather_tool.invoke({"city": "Seattle", "units": "celsius"})
    print(f"\n⚡ Test: {result}")
    
    # Register in registry
    registry = ToolRegistry()
    registry.register(weather_tool)
    
    print(f"\n📋 Registry now has: {registry.tool_names}")
    
    print("\n✅ Create tool complete")


# =============================================================================
# Example 5: Agent with Tools
# =============================================================================

async def agent_with_tools_example():
    """Demonstrate agent with tools (without actual LLM calls)."""
    print("\n" + "=" * 60)
    print("Example 5: Agent with Tools")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create tools
    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        return f"Web results for: {query}"
    
    @tool
    def write_file(filename: str, content: str) -> str:
        """Write content to a file."""
        return f"Wrote {len(content)} characters to {filename}"
    
    # Create registry
    registry = ToolRegistry()
    registry.register(search_web)
    registry.register(write_file)
    
    # Create agent with specific tools allowed
    config = AgentConfig(
        name="Research Writer",
        role=AgentRole.SPECIALIST,
        description="Researches topics and writes summaries",
        tools=["search_web", "write_file"],  # Only these tools allowed
    )
    
    agent = Agent(
        config=config,
        event_bus=event_bus,
        tool_registry=registry,
    )
    
    print(f"\n🤖 Agent: {agent.name}")
    print(f"   Allowed tools: {agent.config.tools}")
    
    # Check tool permissions
    print(f"\n🔐 Tool Permissions:")
    print(f"   Can use 'search_web': {config.can_use_tool('search_web')}")
    print(f"   Can use 'write_file': {config.can_use_tool('write_file')}")
    print(f"   Can use 'delete_file': {config.can_use_tool('delete_file')}")
    
    print("\n✅ Agent with tools complete")


# =============================================================================
# Example 6: Agent Events
# =============================================================================

async def agent_events_example():
    """Demonstrate agent events."""
    print("\n" + "=" * 60)
    print("Example 6: Agent Events")
    print("=" * 60)
    
    event_bus = EventBus()
    events: list[Event] = []
    
    # Track agent events
    async def track_events(event: Event):
        if event.type.value.startswith("agent."):
            events.append(event)
            print(f"   📨 {event.type.value}: {event.data.get('agent_name', 'unknown')}")
    
    event_bus.subscribe_all(track_events)
    
    # Create agent
    config = AgentConfig(name="Event Agent")
    agent = Agent(config=config, event_bus=event_bus)
    
    print(f"\n🤖 Agent: {agent.name}")
    print(f"\n   Simulating agent lifecycle events:")
    
    # Simulate events (these would normally be emitted by agent operations)
    await event_bus.publish(Event(
        type="agent.registered",
        data={"agent_id": agent.id, "agent_name": agent.name},
        source=f"agent:{agent.id}",
    ))
    
    await event_bus.publish(Event(
        type="agent.status_changed",
        data={
            "agent_id": agent.id,
            "agent_name": agent.name,
            "old_status": "idle",
            "new_status": "thinking",
        },
        source=f"agent:{agent.id}",
    ))
    
    await event_bus.publish(Event(
        type="agent.thinking",
        data={"agent_id": agent.id, "agent_name": agent.name},
        source=f"agent:{agent.id}",
    ))
    
    print(f"\n   Total agent events: {len(events)}")
    
    print("\n✅ Agent events complete")


# =============================================================================
# Example 7: Orchestrator Setup
# =============================================================================

async def orchestrator_example():
    """Demonstrate orchestrator setup."""
    print("\n" + "=" * 60)
    print("Example 7: Orchestrator Setup")
    print("=" * 60)
    
    # Create infrastructure
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    tool_registry = ToolRegistry()
    
    # Create orchestrator
    orchestrator = Orchestrator(
        event_bus=event_bus,
        task_manager=task_manager,
        tool_registry=tool_registry,
    )
    
    print(f"\n🎯 Orchestrator Created")
    print(f"   Task Manager: ✓")
    print(f"   Tool Registry: ✓")
    print(f"   Event Bus: ✓")
    
    # Create and register agents
    agents = [
        Agent(
            config=AgentConfig(
                name="Researcher",
                role=AgentRole.SPECIALIST,
                description="Gathers information",
            ),
            event_bus=event_bus,
            tool_registry=tool_registry,
        ),
        Agent(
            config=AgentConfig(
                name="Analyst",
                role=AgentRole.SPECIALIST,
                description="Analyzes data",
            ),
            event_bus=event_bus,
            tool_registry=tool_registry,
        ),
        Agent(
            config=AgentConfig(
                name="Writer",
                role=AgentRole.WORKER,
                description="Creates content",
            ),
            event_bus=event_bus,
            tool_registry=tool_registry,
        ),
    ]
    
    for agent in agents:
        orchestrator.register_agent(agent)
        print(f"   Registered: {agent.name} ({agent.role.value})")
    
    print(f"\n👥 Total agents: {len(orchestrator.agents)}")
    
    # Check agent availability
    print(f"\n   Agent Status:")
    for agent_id, agent in orchestrator.agents.items():
        print(f"      {agent.name}: {agent.status.value} (available: {agent.is_available()})")
    
    print("\n✅ Orchestrator setup complete")


# =============================================================================
# Example 8: Task Execution Flow (Simulated)
# =============================================================================

async def task_execution_example():
    """Demonstrate task execution flow (simulated, no LLM)."""
    print("\n" + "=" * 60)
    print("Example 8: Task Execution Flow (Simulated)")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Track events
    events: list[Event] = []
    async def track(event: Event):
        events.append(event)
    event_bus.subscribe_all(track)
    
    # Create task
    task = await task_manager.create_task(
        name="Analyze customer feedback",
        description="Review customer feedback and identify trends",
        tool="analyze_data",
        args={"data": "customer_feedback.csv"},
    )
    
    print(f"\n📋 Task: {task.name}")
    print(f"   Tool: {task.tool}")
    print(f"   Args: {task.args}")
    
    # Simulate execution flow
    print(f"\n⚡ Execution Flow:")
    
    # 1. Assign to agent
    task.assigned_agent_id = "agent-001"
    print(f"   1. Assigned to: {task.assigned_agent_id}")
    
    # 2. Schedule
    await task_manager.update_status(task.id, TaskStatus.SCHEDULED)
    print(f"   2. Scheduled: {(await task_manager.get_task(task.id)).status.value}")
    
    # 3. Start
    await task_manager.update_status(task.id, TaskStatus.RUNNING)
    print(f"   3. Running: {(await task_manager.get_task(task.id)).status.value}")
    
    # 4. Complete
    await task_manager.update_status(
        task.id,
        TaskStatus.COMPLETED,
        result={"trends": ["positive sentiment", "feature requests"]},
    )
    completed = await task_manager.get_task(task.id)
    print(f"   4. Completed: {completed.status.value}")
    print(f"      Result: {completed.result}")
    print(f"      Duration: {completed.duration_ms:.1f}ms")
    
    print(f"\n📊 Events emitted: {len(events)}")
    
    print("\n✅ Task execution flow complete")


# =============================================================================
# Example 9: Multi-Agent Team
# =============================================================================

async def multi_agent_team_example():
    """Demonstrate multi-agent team setup."""
    print("\n" + "=" * 60)
    print("Example 9: Multi-Agent Team")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Define team roles
    team_configs = [
        AgentConfig(
            name="Team Lead",
            role=AgentRole.ORCHESTRATOR,
            description="Coordinates team activities and delegates tasks",
            system_prompt="You coordinate a team. Break tasks into subtasks and delegate.",
        ),
        AgentConfig(
            name="Data Scientist",
            role=AgentRole.SPECIALIST,
            description="Performs data analysis and modeling",
            tools=["analyze_data", "train_model", "visualize"],
        ),
        AgentConfig(
            name="Backend Dev",
            role=AgentRole.SPECIALIST,
            description="Develops backend services and APIs",
            tools=["write_code", "deploy_service", "run_tests"],
        ),
        AgentConfig(
            name="Frontend Dev",
            role=AgentRole.SPECIALIST,
            description="Develops user interfaces",
            tools=["write_code", "design_ui", "run_tests"],
        ),
        AgentConfig(
            name="QA Engineer",
            role=AgentRole.WORKER,
            description="Tests and validates features",
            tools=["run_tests", "write_tests", "report_bugs"],
        ),
    ]
    
    print(f"\n👥 Team Structure:")
    
    # Group by role
    by_role: dict[AgentRole, list[AgentConfig]] = {}
    for config in team_configs:
        if config.role not in by_role:
            by_role[config.role] = []
        by_role[config.role].append(config)
    
    for role, configs in by_role.items():
        print(f"\n   {role.value.upper()}:")
        for config in configs:
            tools_str = ", ".join(config.tools) if config.tools else "all"
            print(f"      🤖 {config.name}")
            print(f"         {config.description}")
            print(f"         Tools: {tools_str}")
    
    # Create agents
    agents = [
        Agent(config=config, event_bus=event_bus)
        for config in team_configs
    ]
    
    print(f"\n   Total team members: {len(agents)}")
    
    # Simulate team metrics
    print(f"\n📊 Team Metrics:")
    print(f"   Orchestrators: {len([a for a in agents if a.role == AgentRole.ORCHESTRATOR])}")
    print(f"   Specialists: {len([a for a in agents if a.role == AgentRole.SPECIALIST])}")
    print(f"   Workers: {len([a for a in agents if a.role == AgentRole.WORKER])}")
    
    print("\n✅ Multi-agent team complete")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all agent examples."""
    print("\n" + "🤖 " * 20)
    print("AgenticFlow Agents Examples")
    print("🤖 " * 20)
    
    await basic_agent_example()
    await configuration_example()
    await tool_registry_example()
    await create_tool_example()
    await agent_with_tools_example()
    await agent_events_example()
    await orchestrator_example()
    await task_execution_example()
    await multi_agent_team_example()
    
    print("\n" + "=" * 60)
    print("All agent examples complete!")
    print("=" * 60)
    
    print("\n💡 Agent features demonstrated:")
    print("   - Agent creation and configuration")
    print("   - Role-based organization (Supervisor, Specialist, Worker)")
    print("   - Tool registry and tool permissions")
    print("   - Event-driven agent lifecycle")
    print("   - Orchestrator for coordination")
    print("   - Multi-agent team setup")
    
    print("\n📝 Note: For examples with LLM calls, set OPENAI_API_KEY env var")


if __name__ == "__main__":
    asyncio.run(main())
