"""
Custom Event Handlers Example - demonstrates creating custom event handlers.

This example shows how to:
1. Create custom event handlers
2. Filter events by type or category
3. Collect metrics
4. Log to files

Usage:
    uv run python examples/custom_handlers.py
"""

import asyncio
import tempfile
from pathlib import Path

from langchain.tools import tool

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    ConsoleEventHandler,
    EventBus,
    EventType,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
    Orchestrator,
    TaskManager,
    ToolRegistry,
)
from agenticflow.models.event import Event


# =============================================================================
# Custom Event Handler
# =============================================================================


class SlackNotificationHandler:
    """
    Example custom handler that would send Slack notifications.
    
    In a real application, this would use the Slack API.
    """

    def __init__(self, channel: str = "#alerts") -> None:
        self.channel = channel
        self.notification_count = 0

    def __call__(self, event: Event) -> None:
        """Send notification for important events."""
        # Only notify on errors and completions
        if event.type in (
            EventType.TASK_FAILED,
            EventType.SYSTEM_ERROR,
            EventType.AGENT_ERROR,
        ):
            self._send_notification(f"🚨 Error: {event.type.value}", event.data)
        elif event.type == EventType.SYSTEM_STOPPED:
            self._send_notification("✅ System completed", event.data)

    def _send_notification(self, title: str, data: dict) -> None:
        """Simulate sending a Slack notification."""
        self.notification_count += 1
        print(f"   📨 [Slack → {self.channel}] {title}")


class DatabaseEventHandler:
    """
    Example custom handler that would persist events to a database.
    
    In a real application, this would use SQLAlchemy or similar.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []  # Simulate database

    def __call__(self, event: Event) -> None:
        """Persist event to database."""
        # In reality, this might use asyncio.create_task for async DB insert
        self.events.append(event.to_dict())

    def get_event_count(self) -> int:
        return len(self.events)


# =============================================================================
# Tools for demo
# =============================================================================


@tool
def risky_operation(action: str) -> str:
    """Perform a risky operation that might fail."""
    import random

    if random.random() < 0.3:  # 30% chance of failure
        raise RuntimeError(f"Operation '{action}' failed randomly!")
    return f"Successfully completed: {action}"


@tool
def safe_operation(action: str) -> str:
    """Perform a safe operation that always succeeds."""
    return f"Safely completed: {action}"


# =============================================================================
# Main example
# =============================================================================


async def main() -> None:
    """Run the custom handlers example."""
    print("\n" + "=" * 60)
    print("AgenticFlow Custom Event Handlers Example")
    print("=" * 60)

    # Create infrastructure
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    tool_registry = ToolRegistry()
    tool_registry.register_many([risky_operation, safe_operation])

    # =============================================================================
    # Setup various handlers
    # =============================================================================

    print("\n📋 Setting up event handlers...")

    # 1. Console handler (verbose mode)
    console = ConsoleEventHandler(verbose=False)
    event_bus.subscribe_all(console)
    print("   ✓ Console handler (all events)")

    # 2. File handler with filtering (only task events)
    temp_dir = Path(tempfile.mkdtemp())
    log_file = temp_dir / "task_events.jsonl"

    task_file_handler = FilteringEventHandler(
        FileEventHandler(log_file),
        categories=["task"],
    )
    event_bus.subscribe_all(task_file_handler)
    print(f"   ✓ File handler (task events only) → {log_file}")

    # 3. Metrics handler
    metrics = MetricsEventHandler()
    event_bus.subscribe_all(metrics)
    print("   ✓ Metrics handler")

    # 4. Custom Slack handler (errors only)
    slack = SlackNotificationHandler(channel="#agent-alerts")
    event_bus.subscribe_many(
        [
            EventType.TASK_FAILED,
            EventType.AGENT_ERROR,
            EventType.SYSTEM_ERROR,
            EventType.SYSTEM_STOPPED,
        ],
        slack,
    )
    print("   ✓ Slack handler (errors + completion)")

    # 5. Database handler (async)
    db_handler = DatabaseEventHandler()
    event_bus.subscribe_all(db_handler)
    print("   ✓ Database handler (all events)")

    # =============================================================================
    # Create agent and orchestrator
    # =============================================================================

    worker = Agent(
        config=AgentConfig(
            name="RiskyWorker",
            role=AgentRole.WORKER,
            tools=["risky_operation", "safe_operation"],
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )

    orchestrator = Orchestrator(
        event_bus=event_bus,
        task_manager=task_manager,
        tool_registry=tool_registry,
    )
    orchestrator.register_agent(worker)

    # =============================================================================
    # Run some tasks
    # =============================================================================

    print("\n" + "-" * 60)
    print("Running tasks (some may fail)...")
    print("-" * 60 + "\n")

    tasks = [
        ("Safe task 1", "safe_operation", {"action": "backup"}),
        ("Risky task 1", "risky_operation", {"action": "deploy"}),
        ("Safe task 2", "safe_operation", {"action": "validate"}),
        ("Risky task 2", "risky_operation", {"action": "migrate"}),
        ("Risky task 3", "risky_operation", {"action": "upgrade"}),
    ]

    for name, tool_name, args in tasks:
        try:
            await orchestrator.run_task(name=name, tool=tool_name, args=args)
        except Exception as e:
            print(f"   ⚠️  Task '{name}' failed: {e}")

    # =============================================================================
    # Show metrics
    # =============================================================================

    print("\n" + "=" * 60)
    print("📊 Metrics Summary")
    print("=" * 60)

    m = metrics.get_metrics()
    print(f"\n   Total events:      {m['total_events']}")
    print(f"   Tasks completed:   {m['tasks_completed']}")
    print(f"   Errors:            {m['error_count']}")
    print(f"   Error rate:        {m['error_rate']:.1%}")
    print(f"   Avg task duration: {m['avg_task_duration_ms']:.1f}ms")
    print(f"   Uptime:            {m['uptime_seconds']:.1f}s")

    print("\n   Event counts by type:")
    for event_type, count in sorted(m["event_counts"].items()):
        print(f"      {event_type}: {count}")

    print(f"\n   📨 Slack notifications sent: {slack.notification_count}")
    print(f"   💾 Events in database: {db_handler.get_event_count()}")

    # Show file contents
    print(f"\n   📁 Log file contents ({log_file}):")
    with open(log_file) as f:
        lines = f.readlines()
        print(f"      {len(lines)} task events logged")

    # Cleanup
    task_file_handler.inner.close()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
