"""
Example 31: Spawning Agents - Dynamic Agent Creation

Demonstrates the spawning capability where a supervisor agent can
dynamically spawn ephemeral specialist agents to handle specific tasks.
This enables truly parallel, elastic agent swarms.

Key Features:
- Supervisor spawns specialist agents on-demand
- Parallel task execution via spawned agents
- Configurable spawn limits and specs
- Ephemeral agents auto-cleanup after task completion
"""

import asyncio
from agenticflow import Agent
from agenticflow.agent.spawning import SpawningConfig, AgentSpec
from agenticflow.tools import tool
from agenticflow.observability import Observer
from config import get_model


# Define tools for specialists
@tool
def search_web(query: str) -> str:
    """Search the web for information on a topic."""
    # Simulated search results
    results = {
        "python advantages": "Python: Easy syntax, large ecosystem, great for AI/ML, dynamic typing, extensive libraries like NumPy and Pandas.",
        "rust advantages": "Rust: Memory safety without GC, zero-cost abstractions, excellent performance, fearless concurrency, no null pointers.",
        "python disadvantages": "Python: Slower execution, GIL limits threading, dynamic typing can hide bugs, high memory usage.",
        "rust disadvantages": "Rust: Steep learning curve, longer compile times, complex ownership system, smaller ecosystem than Python.",
    }
    # Match query to results
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': General programming information found."


@tool
def analyze_data(data: str, analysis_type: str) -> str:
    """Analyze provided data with specified analysis type."""
    return f"Analysis ({analysis_type}): Processed '{data[:50]}...' - Found patterns and insights."


@tool
def write_report(title: str, content: str) -> str:
    """Write a structured report with the given content."""
    return f"📄 Report: {title}\n\n{content}\n\n[Report generated successfully]"


async def example_basic_spawning():
    """Basic example: Supervisor spawns a single specialist."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Spawning - Single Specialist")
    print("=" * 60)

    # Get the configured model
    model = get_model()
    print(f"Using model: {model.__class__.__name__}")

    # Define specialist specs
    specs = {
        "researcher": AgentSpec(
            role="Research Specialist",
            system_prompt="""You are a research specialist. When given a topic:
1. Use the search_web tool to find information
2. Summarize what you found clearly
3. Return your findings.""",
            tools=[search_web],
            description="Researches topics using web search",
        ),
    }

    # Create supervisor with spawning capability
    # Note: Supervisors use role="supervisor" which can delegate and finish
    supervisor = Agent(
        name="Supervisor",
        model=model,
        role="supervisor",
        system_prompt="""You are a supervisor that coordinates research tasks.
When asked to research a topic, use the spawn_agent tool to create a researcher specialist.
The researcher will handle the actual research work.

Example: To research Python, call spawn_agent with:
- spec_name: "researcher"
- task: "Research the advantages and disadvantages of Python programming language"
""",
        spawning=SpawningConfig(
            max_concurrent=3,
            max_total_spawns=5,
            ephemeral=True,
            specs=specs,
            available_tools=[search_web],  # Tools available for spawned agents
        ),
    )

    # Add observer for detailed output including tool calls and spawn events
    # Observer.detailed() shows: tool calls, results, spawn events, and agent actions
    observer = Observer.detailed()
    supervisor.add_observer(observer)

    # Run the supervisor with a task
    print("\n📋 Task: Research Python programming language advantages\n")

    response = await supervisor.run(
        "Please spawn a researcher to investigate the advantages of Python programming language. "
        "After the researcher completes their work, summarize the findings."
    )

    print(f"\n📝 Supervisor Response:\n{response}")

    # Show spawn stats
    if supervisor.spawn_manager:
        print(f"\n📊 Spawn Statistics:")
        print(f"   Total spawns: {supervisor.spawn_manager.total_spawns}")
        print(f"   Active spawns: {supervisor.spawn_manager.active_count}")


async def example_parallel_research():
    """Example: Parallel research with multiple spawned agents."""
    print("\n" + "=" * 60)
    print("Example 2: Parallel Research - Multiple Specialists")
    print("=" * 60)

    model = get_model()

    specs = {
        "researcher": AgentSpec(
            role="Research Specialist",
            system_prompt="""You are a research specialist. Research the given topic using search_web.
Be thorough and return key findings.""",
            tools=[search_web],
            description="Researches topics",
        ),
        "writer": AgentSpec(
            role="Technical Writer",
            system_prompt="""You are a technical writer. Create clear, structured reports.
Use the write_report tool to format your output.""",
            tools=[write_report],
            description="Writes reports",
        ),
    }

    supervisor = Agent(
        name="ResearchCoordinator",
        model=model,
        role="supervisor",
        system_prompt="""You coordinate research projects. You can spawn specialists:

1. "researcher" - For researching topics (has search_web tool)
2. "writer" - For writing reports (has write_report tool)

To compare two topics:
1. Spawn researchers to research each topic separately
2. Combine the findings yourself
3. Provide a clear comparison

IMPORTANT: Use spawn_agent to delegate research tasks. Each spawn_agent call creates a specialist.""",
        spawning=SpawningConfig(
            max_concurrent=4,
            max_total_spawns=10,
            ephemeral=True,
            specs=specs,
            available_tools=[search_web, write_report],
        ),
    )

    observer = Observer.detailed()
    supervisor.add_observer(observer)

    print("\n📋 Task: Compare Python and Rust programming languages\n")

    response = await supervisor.run(
        "Compare Python and Rust programming languages. "
        "Spawn researchers to gather information about advantages of each language, "
        "then provide a comparison summary."
    )

    print(f"\n📝 Research Coordinator Response:\n{response}")

    if supervisor.spawn_manager:
        print(f"\n📊 Spawn Statistics:")
        print(f"   Total spawns: {supervisor.spawn_manager.total_spawns}")


async def example_parallel_map():
    """Example: Using parallel_map for batch processing."""
    print("\n" + "=" * 60)
    print("Example 3: Parallel Map - Batch Processing")
    print("=" * 60)

    model = get_model()

    specs = {
        "analyzer": AgentSpec(
            role="Data Analyzer",
            system_prompt="""You are a data analyzer. When given data to analyze:
1. Use the analyze_data tool with appropriate analysis_type
2. Summarize the results concisely
3. Return your analysis.""",
            tools=[analyze_data],
            description="Analyzes data",
        ),
    }

    supervisor = Agent(
        name="BatchProcessor",
        model=model,
        role="supervisor",
        system_prompt="You coordinate batch analysis tasks.",
        spawning=SpawningConfig(
            max_concurrent=3,
            ephemeral=True,
            specs=specs,
            available_tools=[analyze_data],
        ),
    )

    # Data items to process in parallel
    data_items = [
        "Sales data for Q1 2024: Revenue increased 15%",
        "Customer feedback: 85% satisfaction rate",
        "Website traffic: 1M monthly visitors",
    ]

    print(f"\n📋 Processing {len(data_items)} items in parallel...\n")

    # Use parallel_map to process all items concurrently
    # Each item spawns an "analyzer" agent to process it
    results = await supervisor.parallel_map(
        items=data_items,
        task_template="Analyze this data and provide key insights: {item}",
        role="analyzer",  # Matches the spec name defined above
    )

    print("📊 Parallel Processing Results:")
    for i, result in enumerate(results, 1):
        # Results are strings from each spawned agent
        # Truncate long responses for display
        display = result[:150] + "..." if len(result) > 150 else result
        print(f"\n   📌 Item {i}:\n      {display}")

    if supervisor.spawn_manager:
        print(f"\n   Total agents spawned: {supervisor.spawn_manager.total_spawns}")


async def main():
    """Run all spawning examples."""
    print("🚀 Spawning Agents Demo")
    print("=" * 60)
    print("Demonstrating dynamic agent spawning for parallel processing")

    try:
        # Run examples
        await example_basic_spawning()
        await example_parallel_research()
        await example_parallel_map()

        print("\n" + "=" * 60)
        print("✅ All spawning examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
