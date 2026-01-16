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
        system_prompt="You coordinate projects and delegate work to specialists.",
    )

    # Specialist agent that processes delegated tasks
    data_analyst = Agent(
        name="data_analyst",
        model=model,
        system_prompt="You are a data analysis specialist. Provide brief insights.",
    )

    # Create reactive flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    # Register agents with delegation policy - framework auto-injects tools and enhances prompts
    flow.register(coordinator, on="task.created", can_delegate=["data_analyst"])
    flow.register(data_analyst, handles=True)  # auto-sets can_reply=True

    # Run the flow
    await flow.run("Please analyze our sales data from last quarter")


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
        system_prompt="You coordinate a team of specialists.",
    )

    # Specialist agents
    data_analyst = Agent(
        name="data_analyst",
        model=model,
        system_prompt="You analyze data. Provide brief insights.",
    )

    writer = Agent(
        name="writer",
        model=model,
        system_prompt="You write creative content.",
    )

    researcher = Agent(
        name="researcher",
        model=model,
        system_prompt="You research topics thoroughly.",
    )

    # Create flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    # Declarative delegation policy - framework handles tools and prompts
    flow.register(
        coordinator,
        on="task.created",
        can_delegate=["data_analyst", "writer", "researcher"]
    )
    
    # Register specialists
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
        await flow.run(task)
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
        system_prompt="You're a project manager. Break down tasks into requirements.",
    )

    # Architect in the middle of the chain
    architect = Agent(
        name="architect",
        model=model,
        system_prompt="You're a solution architect. Design systems and delegate implementation.",
    )

    # Developer at the end of the chain
    developer = Agent(
        name="developer",
        model=model,
        system_prompt="You're a developer. Implement solutions with clean code.",
    )

    # Create flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    # Declarative chain: PM → Architect → Developer
    flow.register(pm, on="task.created", can_delegate=["architect"])
    flow.register(architect, handles=True, can_delegate=["developer"])
    flow.register(developer, handles=True)

    # Run
    await flow.run("Build a user authentication system")


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
        system_prompt="You coordinate code reviews across multiple dimensions.",
    )

    # Multiple reviewers working in parallel
    security_reviewer = Agent(
        name="security_reviewer",
        model=model,
        system_prompt="Review code for security vulnerabilities.",
    )

    performance_reviewer = Agent(
        name="performance_reviewer",
        model=model,
        system_prompt="Review code for performance optimization opportunities.",
    )

    style_reviewer = Agent(
        name="style_reviewer",
        model=model,
        system_prompt="Review code style and best practices.",
    )

    # Create flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    # Parallel delegation: coordinator → all reviewers
    flow.register(
        coordinator,
        on="task.created",
        can_delegate=["security_reviewer", "performance_reviewer", "style_reviewer"]
    )
    
    for agent in [security_reviewer, performance_reviewer, style_reviewer]:
        flow.register(agent, handles=True)

    # Run
    await flow.run("Review this authentication code for production")


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
        system_prompt="You request data processing tasks.",
    )

    # Processor that sends back responses
    processor = Agent(
        name="processor",
        model=model,
        system_prompt="You process data and return results.",
    )

    # Agent that listens for responses
    response_handler = Agent(
        name="response_handler",
        model=model,
        system_prompt="You receive and acknowledge processing results.",
    )

    # Create flow with observability
    flow = ReactiveFlow(
        observer=Observer(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.REACTIVE, Channel.AGENTS],
            show_duration=True,
        )
    )

    # Declarative request-response pattern
    flow.register(requester, on="task.created", can_delegate=["processor"])
    flow.register(processor, handles=True)
    flow.register(response_handler, on="agent.response")

    # Run
    await flow.run("Process customer data")


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
