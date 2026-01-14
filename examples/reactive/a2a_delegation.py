"""Agent-to-Agent (A2A) Delegation Example.

This example demonstrates how agents can delegate tasks to each other
using the ExecutionContext for direct agent-to-agent communication.

Key Concepts:
- Coordinator agents that delegate specialized tasks
- Specialist agents that handle specific types of work
- Request/Response pattern with correlation IDs
- Wait for responses or fire-and-forget delegation

Setup:
    1. Copy examples/.env.example to examples/.env
    2. Set LLM_PROVIDER and appropriate API key
    3. Run: uv run python examples/reactive/a2a_delegation.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import get_model, settings
except Exception as e:
    print(f"\n❌ Configuration error: {e}")
    print("\n💡 Setup instructions:")
    print("   1. Copy examples/.env.example to examples/.env")
    print("   2. Set LLM_PROVIDER (e.g., LLM_PROVIDER=openai)")
    print("   3. Add your API key (e.g., OPENAI_API_KEY=sk-...)")
    print("   4. Run again\n")
    sys.exit(1)

from agenticflow import Agent
from agenticflow.reactive import ReactiveFlow
from agenticflow.observability import Observer, ObservabilityLevel, Channel


# =============================================================================
# Example 1: Simple Delegation - Coordinator → Specialist
# =============================================================================


async def example_1_simple_delegation():
    """Coordinator delegates to a specialist and waits for result."""
    print("\n" + "=" * 80)
    print("Example 1: Simple Delegation")
    print("=" * 80 + "\n")

    model = get_model()

    # Coordinator agent that delegates specialized work
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="""You coordinate projects. When asked to analyze data:
1. Say you're delegating to the data analyst
2. Explain what you're asking them to do
Keep responses under 3 sentences.""",
    )

    # Specialist agent that processes delegated tasks
    data_analyst = Agent(
        name="data_analyst",
        model=model,
        system_prompt="""You are a data analysis specialist. 
When you receive a task, provide a brief analysis in 2-3 sentences.""",
    )

    # Create reactive flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    # Coordinator reacts to user requests
    flow.register(coordinator, on="task.created")

    # Data analyst reacts to agent requests targeted at them
    flow.register(data_analyst, handles=True)

    # Run the flow
    await flow.run(
        "Please analyze our sales data from last quarter",
        initial_event="task.created",
    )


# =============================================================================
# Example 2: Multi-Specialist Team
# =============================================================================


async def example_2_multi_specialist_team():
    """Coordinator delegates to multiple specialists based on task type."""
    print("\n" + "=" * 80)
    print("Example 2: Multi-Specialist Team")
    print("=" * 80 + "\n")

    model = get_model()

    # Team coordinator
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="""You coordinate a team with different specialists:
- data_analyst: for data analysis tasks
- writer: for writing tasks
- researcher: for research tasks

When you receive a task, identify what type it is and say you're delegating.
Keep responses very brief (1-2 sentences).""",
    )

    # Specialist agents
    data_analyst = Agent(
        name="data_analyst",
        model=model,
        system_prompt="You analyze data. Provide brief insights (2-3 sentences).",
    )

    writer = Agent(
        name="writer",
        model=model,
        system_prompt="You write content. Be creative and concise (2-3 sentences).",
    )

    researcher = Agent(
        name="researcher",
        model=model,
        system_prompt="You research topics. Provide key findings (2-3 sentences).",
    )

    # Create flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    # Register coordinator
    flow.register(coordinator, on="task.created")

    # Register specialists (each reacts to requests for them)
    for agent in [data_analyst, writer, researcher]:
        flow.register(agent, handles=True)

    # Test with different task types
    tasks = [
        "Analyze user engagement metrics",
        "Write a blog post intro about AI",
        "Research quantum computing trends",
    ]

    for task in tasks:
        print(f"📋 Task: {task}\n")
        await flow.run(task, initial_event="task.created")
        print()


# =============================================================================
# Example 3: Chain of Delegation
# =============================================================================


async def example_3_chain_delegation():
    """Agents delegate to each other in a chain: A → B → C."""
    print("\n" + "=" * 80)
    print("Example 3: Chain of Delegation")
    print("=" * 80 + "\n")

    model = get_model()

    # Project manager starts the chain
    pm = Agent(
        name="project_manager",
        model=model,
        system_prompt="""You're a project manager. When you receive a task:
1. Break it down into requirements
2. Say you're delegating to the architect
Keep it brief (2 sentences).""",
    )

    # Architect in the middle of the chain
    architect = Agent(
        name="architect",
        model=model,
        system_prompt="""You're a solution architect. When you receive requirements:
1. Design the solution
2. Say you're delegating implementation to the developer
Keep it brief (2 sentences).""",
    )

    # Developer at the end of the chain
    developer = Agent(
        name="developer",
        model=model,
        system_prompt="""You're a developer. When you receive a design:
1. Describe the implementation approach
2. Provide a brief code example
Keep it very concise (3 sentences max).""",
    )

    # Create flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    # Each agent reacts to requests targeted at them
    flow.register(pm, on="task.created")
    flow.register(architect, handles=True)
    flow.register(developer, handles=True)

    # Run
    await flow.run(
        "Build a user authentication system",
        initial_event="task.created",
    )


# =============================================================================
# Example 4: Parallel Delegation (Fan-out)
# =============================================================================


async def example_4_parallel_delegation():
    """Coordinator delegates to multiple specialists in parallel."""
    print("\n" + "=" * 80)
    print("Example 4: Parallel Delegation (Fan-out)")
    print("=" * 80 + "\n")

    model = get_model()

    # Coordinator that fans out work
    coordinator = Agent(
        name="coordinator",
        model=model,
        system_prompt="""You coordinate multiple specialists. For code review tasks:
1. Say you're delegating to security, performance, and style reviewers
2. List what each will check
Keep it brief (2-3 sentences).""",
    )

    # Multiple reviewers working in parallel
    security_reviewer = Agent(
        name="security_reviewer",
        model=model,
        system_prompt="You review code for security issues. Be brief (1-2 sentences).",
    )

    performance_reviewer = Agent(
        name="performance_reviewer",
        model=model,
        system_prompt="You review code for performance. Be brief (1-2 sentences).",
    )

    style_reviewer = Agent(
        name="style_reviewer",
        model=model,
        system_prompt="You review code style. Be brief (1-2 sentences).",
    )

    # Create flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    flow.register(coordinator, on="task.created")

    # All reviewers react to requests for them
    for agent in [security_reviewer, performance_reviewer, style_reviewer]:
        flow.register(agent, handles=True)

    # Run
    await flow.run(
        "Review this authentication code for production",
        initial_event="task.created",
    )


# =============================================================================
# Example 5: Request-Response Pattern
# =============================================================================


async def example_5_request_response_pattern():
    """Demonstrate explicit request/response with correlation IDs."""
    print("\n" + "=" * 80)
    print("Example 5: Request-Response Pattern")
    print("=" * 80 + "\n")

    model = get_model()

    # Requester agent
    requester = Agent(
        name="requester",
        model=model,
        system_prompt="""You request data processing. Say:
1. What you're requesting
2. Who you're asking
Be very brief (1 sentence).""",
    )

    # Processor that sends back responses
    processor = Agent(
        name="processor",
        model=model,
        system_prompt="""You process data and send results back.
Provide a brief processing result (1-2 sentences).""",
    )

    # Agent that listens for responses
    response_handler = Agent(
        name="response_handler",
        model=model,
        system_prompt="You receive processed results. Acknowledge briefly (1 sentence).",
    )

    # Create flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    flow.register(requester, on="task.created")
    flow.register(processor, handles=True)
    flow.register(response_handler, on="agent.response")

    # Run
    await flow.run(
        "Process customer data",
        initial_event="task.created",
    )


# =============================================================================
# Main Runner
# =============================================================================


async def main():
    """Run all examples."""
    print("\n🚀 Agent-to-Agent (A2A) Delegation Examples\n")

    examples = [
        ("Simple Delegation", example_1_simple_delegation),
        ("Multi-Specialist Team", example_2_multi_specialist_team),
        ("Chain of Delegation", example_3_chain_delegation),
        ("Parallel Delegation", example_4_parallel_delegation),
        ("Request-Response Pattern", example_5_request_response_pattern),
    ]

    for i, (name, example_fn) in enumerate(examples, 1):
        try:
            await example_fn()
        except Exception as e:
            print(f"\n❌ Example {i} ({name}) failed: {e}")

        # Small delay between examples
        await asyncio.sleep(0.5)

    print("\n" + "=" * 80)
    print("✅ All examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
