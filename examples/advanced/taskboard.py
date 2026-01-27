"""
TaskBoard - Agent task tracking and verification.

This example demonstrates the TaskBoard feature which gives agents
tools to track their work, add notes, and verify completion.

## TaskBoard Tools

When `taskboard=True`, the agent gets these tools:
- `add_task`: Create a new task to track
- `update_task`: Update task status (pending, in_progress, completed, failed, blocked)
- `add_note`: Record observations and findings
- `verify_task`: Verify a task was completed correctly
- `get_taskboard_status`: See overall progress

## Run

    uv run python examples/advanced/taskboard.py
"""

import asyncio

from cogent import Agent
from cogent.observability import Observer


async def main() -> None:
    print("=" * 70)
    print("TASKBOARD - Agent Task Tracking")
    print("=" * 70)

    # Create agent with taskboard enabled
    observer = Observer.trace()

    agent = Agent(
        name="ProjectManager",
        model="gpt-4o-mini",
        instructions="You are a helpful project manager who plans and organizes work.",
        taskboard=True,  # Enables taskboard tools + instructions automatically
        observer=observer,
    )

    task = """
    Plan a simple REST API for a todo app. I need:
    1. List the endpoints needed
    2. Define the data model
    3. Recommend a tech stack
    """

    print(f"\n📋 Task:\n{task}")
    print("\n" + "-" * 70)
    print("EXECUTION (watch the agent use taskboard tools)")
    print("-" * 70 + "\n")

    result = await agent.run(task)

    print("\n" + "=" * 70)
    print("FINAL RESPONSE")
    print("=" * 70)
    print(result.content)

    print("\n" + "=" * 70)
    print("TASKBOARD STATUS")
    print("=" * 70)
    if agent.taskboard:
        print(agent.taskboard.summary())
    else:
        print("Taskboard not enabled")

    print("\n" + "=" * 70)
    print("EXECUTION METADATA")
    print("=" * 70)
    tokens = result.metadata.tokens if result.metadata else None
    print(f"  • Total tokens: {tokens.total_tokens if tokens else 'N/A'}")
    print(f"  • Tool calls: {len(result.tool_calls) if result.tool_calls else 0}")
    print(f"  • Duration: {result.metadata.duration:.1f}s" if result.metadata else "")


if __name__ == "__main__":
    asyncio.run(main())
