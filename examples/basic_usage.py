"""
Basic Usage Example - demonstrates core AgenticFlow concepts.

This example shows how to:
1. Create an event bus and task manager
2. Register tools
3. Create agents
4. Run the orchestrator

Usage:
    uv run python examples/basic_usage.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.tools import tool

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    ConsoleEventHandler,
    EventBus,
    Orchestrator,
    TaskManager,
    ToolRegistry,
)

load_dotenv()


# =============================================================================
# Define your tools - these are the capabilities your agents will have
# =============================================================================


@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    # In a real application, this would call a search API
    return f"Search results for '{query}': [Example results would appear here]"


@tool
def write_content(topic: str, style: str = "informative") -> str:
    """Write content about a topic in a specific style."""
    # In a real application, this might call an LLM or template system
    return f"Content about '{topic}' in {style} style: [Generated content here]"


@tool
def analyze_data(data: str) -> str:
    """Analyze data and provide insights."""
    return f"Analysis of data: [Insights about '{data[:50]}...' would appear here]"


# =============================================================================
# Main example
# =============================================================================


async def main() -> None:
    """Run the basic usage example."""
    print("\n" + "=" * 60)
    print("AgenticFlow Basic Usage Example")
    print("=" * 60)

    # 1. Create the core infrastructure
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)

    # 2. Add a console handler to see events
    console_handler = ConsoleEventHandler(verbose=False)
    event_bus.subscribe_all(console_handler)

    # 3. Create and register tools
    tool_registry = ToolRegistry()
    tool_registry.register_many([search_web, write_content, analyze_data])

    print(f"\n📦 Registered {len(tool_registry)} tools:")
    print(tool_registry.get_tool_descriptions())

    # 4. Create agents with different roles
    researcher = Agent(
        config=AgentConfig(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            description="Researches topics and gathers information",
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            tools=["search_web"],
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )

    writer = Agent(
        config=AgentConfig(
            name="Writer",
            role=AgentRole.WORKER,
            description="Writes content based on research",
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            tools=["write_content"],
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )

    analyst = Agent(
        config=AgentConfig(
            name="Analyst",
            role=AgentRole.SPECIALIST,
            description="Analyzes data and provides insights",
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            tools=["analyze_data"],
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )

    print(f"\n🤖 Created {3} agents:")
    for agent in [researcher, writer, analyst]:
        print(f"   - {agent.name} ({agent.role.value})")

    # 5. Create the orchestrator
    orchestrator = Orchestrator(
        event_bus=event_bus,
        task_manager=task_manager,
        tool_registry=tool_registry,
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )

    # Register agents with orchestrator
    orchestrator.register_agent(researcher)
    orchestrator.register_agent(writer)
    orchestrator.register_agent(analyst)

    # 6. Run a simple task directly
    print("\n" + "-" * 60)
    print("Running a single task...")
    print("-" * 60)

    result = await orchestrator.run_task(
        name="Search for AI trends",
        tool="search_web",
        args={"query": "AI trends 2024"},
    )

    print(f"\n✅ Task result: {result}")

    # 7. Run a complex multi-step request (requires LLM)
    if os.getenv("OPENAI_API_KEY"):
        print("\n" + "-" * 60)
        print("Running a complex request with planning...")
        print("-" * 60)

        result = await orchestrator.run(
            "Search for information about sustainable energy, "
            "then write a brief summary about it."
        )

        print(f"\n📊 Execution Summary:")
        print(f"   Correlation ID: {result.get('correlation_id')}")
        print(f"   Events emitted: {result.get('event_count', 0)}")
        print(f"   Results: {len(result.get('results', {}))}")
    else:
        print("\n⚠️  Set OPENAI_API_KEY to run the planning example")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
