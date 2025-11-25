"""
Events Example - demonstrates the event-driven architecture.

This example shows:
1. EventBus publish/subscribe patterns
2. Built-in event handlers
3. Custom event handlers
4. Event filtering and metrics

Usage:
    uv run python examples/events_example.py
"""

import asyncio
import tempfile
from pathlib import Path

from agenticflow import (
    EventBus,
    Event,
    EventType,
)
from agenticflow.events import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)


# =============================================================================
# Example 1: Basic Publish/Subscribe
# =============================================================================

async def basic_pubsub_example():
    """Demonstrate basic event publish/subscribe."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Publish/Subscribe")
    print("=" * 60)
    
    event_bus = EventBus()
    received_events: list[Event] = []
    
    # Subscribe to all events
    async def capture_all(event: Event):
        received_events.append(event)
        print(f"   📨 Received: {event.type.value}")
    
    event_bus.subscribe_all(capture_all)
    
    # Publish various events
    print("\n   Publishing events:")
    
    await event_bus.publish(Event(
        type=EventType.SYSTEM_STARTED,
        data={"version": "1.0.0"},
        source="example",
    ))
    
    await event_bus.publish(Event(
        type=EventType.TASK_CREATED,
        data={"name": "Test task", "priority": "high"},
        source="example",
    ))
    
    await event_bus.publish(Event(
        type=EventType.TASK_COMPLETED,
        data={"task": {"name": "Test task", "duration_ms": 100}},
        source="example",
    ))
    
    print(f"\n   Total events received: {len(received_events)}")
    print("\n✅ Basic publish/subscribe complete")


# =============================================================================
# Example 2: Typed Subscriptions
# =============================================================================

async def typed_subscription_example():
    """Demonstrate subscribing to specific event types."""
    print("\n" + "=" * 60)
    print("Example 2: Typed Subscriptions")
    print("=" * 60)
    
    event_bus = EventBus()
    task_events: list[Event] = []
    agent_events: list[Event] = []
    
    # Subscribe to specific event types
    async def handle_task_created(event: Event):
        task_events.append(event)
        print(f"   📋 Task handler: {event.data.get('name', 'unknown')}")
    
    async def handle_agent_thinking(event: Event):
        agent_events.append(event)
        print(f"   🧠 Agent handler: {event.data.get('agent_name', 'unknown')} thinking")
    
    event_bus.subscribe(EventType.TASK_CREATED, handle_task_created)
    event_bus.subscribe(EventType.AGENT_THINKING, handle_agent_thinking)
    
    # Publish mixed events
    print("\n   Publishing mixed events:")
    
    await event_bus.publish(Event(
        type=EventType.TASK_CREATED,
        data={"name": "Research task"},
    ))
    
    await event_bus.publish(Event(
        type=EventType.AGENT_THINKING,
        data={"agent_name": "Researcher"},
    ))
    
    await event_bus.publish(Event(
        type=EventType.TASK_CREATED,
        data={"name": "Write task"},
    ))
    
    await event_bus.publish(Event(
        type=EventType.TOOL_CALLED,  # No handler for this
        data={"tool": "search_web"},
    ))
    
    print(f"\n   Task events: {len(task_events)}")
    print(f"   Agent events: {len(agent_events)}")
    print("\n✅ Typed subscriptions complete")


# =============================================================================
# Example 3: Console Event Handler
# =============================================================================

async def console_handler_example():
    """Demonstrate the console event handler."""
    print("\n" + "=" * 60)
    print("Example 3: Console Event Handler")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create console handler with verbose mode
    console = ConsoleEventHandler(
        verbose=False,
        show_timestamp=True,
        show_source=True,
    )
    event_bus.subscribe_all(console)
    
    print("\n   Events with formatted output:")
    
    await event_bus.publish(Event(
        type=EventType.SYSTEM_STARTED,
        source="main",
    ))
    
    await event_bus.publish(Event(
        type=EventType.AGENT_REGISTERED,
        data={"agent_name": "Researcher"},
        source="orchestrator",
    ))
    
    await event_bus.publish(Event(
        type=EventType.TASK_CREATED,
        data={"name": "Analyze data", "depends_on": ["fetch_data"]},
        source="task_manager",
    ))
    
    await event_bus.publish(Event(
        type=EventType.TASK_STARTED,
        data={"task": {"name": "Analyze data"}},
    ))
    
    await event_bus.publish(Event(
        type=EventType.TASK_COMPLETED,
        data={"task": {"name": "Analyze data", "duration_ms": 150.5}},
    ))
    
    print("\n✅ Console handler complete")


# =============================================================================
# Example 4: File Event Handler
# =============================================================================

async def file_handler_example():
    """Demonstrate the file event handler."""
    print("\n" + "=" * 60)
    print("Example 4: File Event Handler")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_file = Path(f.name)
    
    event_bus = EventBus()
    
    # Create file handler
    file_handler = FileEventHandler(log_file, append=True)
    event_bus.subscribe_all(file_handler)
    
    print(f"\n   Writing events to: {log_file}")
    
    await event_bus.publish(Event(
        type=EventType.TASK_CREATED,
        data={"name": "Task 1"},
    ))
    
    await event_bus.publish(Event(
        type=EventType.TASK_COMPLETED,
        data={"task": {"name": "Task 1", "result": "success"}},
    ))
    
    await event_bus.publish(Event(
        type=EventType.TASK_FAILED,
        data={"task": {"name": "Task 2", "error": "timeout"}},
    ))
    
    # Close the handler
    file_handler.close()
    
    # Read and display the log
    print("\n   Log file contents:")
    content = log_file.read_text()
    lines = content.strip().split('\n')
    print(f"   Lines written: {len(lines)}")
    
    # Show first line
    import json
    first_event = json.loads(lines[0])
    print(f"   First event type: {first_event['type']}")
    
    # Clean up
    log_file.unlink()
    
    print("\n✅ File handler complete")


# =============================================================================
# Example 5: Filtering Event Handler
# =============================================================================

async def filtering_handler_example():
    """Demonstrate the filtering event handler."""
    print("\n" + "=" * 60)
    print("Example 5: Filtering Event Handler")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create filtered console handler - only task events
    task_only = FilteringEventHandler(
        ConsoleEventHandler(verbose=False, show_timestamp=False),
        categories=["task"],
    )
    event_bus.subscribe_all(task_only)
    
    print("\n   Publishing mixed events (only task events shown):")
    
    await event_bus.publish(Event(type=EventType.SYSTEM_STARTED))
    await event_bus.publish(Event(
        type=EventType.TASK_CREATED,
        data={"name": "Will show"},
    ))
    await event_bus.publish(Event(
        type=EventType.AGENT_THINKING,
        data={"agent_name": "Won't show"},
    ))
    await event_bus.publish(Event(
        type=EventType.TASK_COMPLETED,
        data={"task": {"name": "Will show", "duration_ms": 50}},
    ))
    await event_bus.publish(Event(type=EventType.TOOL_CALLED, data={"tool": "hidden"}))
    
    print("\n✅ Filtering handler complete")


# =============================================================================
# Example 6: Metrics Event Handler
# =============================================================================

async def metrics_handler_example():
    """Demonstrate the metrics event handler."""
    print("\n" + "=" * 60)
    print("Example 6: Metrics Event Handler")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Create metrics handler
    metrics = MetricsEventHandler()
    event_bus.subscribe_all(metrics)
    
    # Simulate a workload
    print("\n   Simulating workload...")
    
    await event_bus.publish(Event(type=EventType.SYSTEM_STARTED))
    
    for i in range(5):
        await event_bus.publish(Event(
            type=EventType.TASK_CREATED,
            data={"name": f"Task {i}"},
        ))
        await event_bus.publish(Event(
            type=EventType.TASK_STARTED,
            data={"task": {"name": f"Task {i}"}},
        ))
        await event_bus.publish(Event(
            type=EventType.TASK_COMPLETED,
            data={"task": {"name": f"Task {i}", "duration_ms": 100 + i * 20}},
        ))
    
    # Add some errors
    await event_bus.publish(Event(
        type=EventType.TASK_FAILED,
        data={"task": {"name": "Failed task", "error": "timeout"}},
    ))
    await event_bus.publish(Event(
        type=EventType.AGENT_ERROR,
        data={"agent_name": "Agent1", "error": "model unavailable"},
    ))
    
    # Get metrics
    m = metrics.get_metrics()
    
    print(f"\n   📊 Metrics Summary:")
    print(f"      Total events: {m['total_events']}")
    print(f"      Tasks completed: {m['tasks_completed']}")
    print(f"      Avg duration: {m['avg_task_duration_ms']:.1f}ms")
    print(f"      Min duration: {m['min_task_duration_ms']:.1f}ms")
    print(f"      Max duration: {m['max_task_duration_ms']:.1f}ms")
    print(f"      Errors: {m['error_count']}")
    print(f"      Error rate: {m['error_rate']:.1%}")
    
    print(f"\n   📈 Event counts by type:")
    for event_type, count in sorted(m['event_counts'].items()):
        print(f"      {event_type}: {count}")
    
    print("\n✅ Metrics handler complete")


# =============================================================================
# Example 7: Custom Event Handler
# =============================================================================

async def custom_handler_example():
    """Demonstrate creating custom event handlers."""
    print("\n" + "=" * 60)
    print("Example 7: Custom Event Handler")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Custom handler class
    class AlertHandler:
        """Handler that alerts on errors."""
        
        def __init__(self):
            self.alerts: list[str] = []
        
        def __call__(self, event: Event) -> None:
            if event.type in (
                EventType.TASK_FAILED,
                EventType.AGENT_ERROR,
                EventType.TOOL_ERROR,
                EventType.SYSTEM_ERROR,
            ):
                alert = f"🚨 ALERT: {event.type.value} - {event.data}"
                self.alerts.append(alert)
                print(f"   {alert}")
    
    alert_handler = AlertHandler()
    event_bus.subscribe_all(alert_handler)
    
    print("\n   Publishing events (errors will trigger alerts):")
    
    await event_bus.publish(Event(type=EventType.TASK_STARTED, data={}))
    await event_bus.publish(Event(type=EventType.TASK_COMPLETED, data={"task": {}}))
    await event_bus.publish(Event(
        type=EventType.TASK_FAILED,
        data={"task": {"name": "Critical task", "error": "Connection lost"}},
    ))
    await event_bus.publish(Event(
        type=EventType.SYSTEM_ERROR,
        data={"error": "Database unavailable"},
    ))
    
    print(f"\n   Total alerts: {len(alert_handler.alerts)}")
    print("\n✅ Custom handler complete")


# =============================================================================
# Example 8: Async Event Handler
# =============================================================================

async def async_handler_example():
    """Demonstrate async event handlers."""
    print("\n" + "=" * 60)
    print("Example 8: Async Event Handler")
    print("=" * 60)
    
    event_bus = EventBus()
    processed: list[str] = []
    
    async def async_processor(event: Event):
        """Async handler that simulates processing."""
        # Simulate async work
        await asyncio.sleep(0.01)
        processed.append(event.type.value)
        print(f"   ⚡ Async processed: {event.type.value}")
    
    event_bus.subscribe_all(async_processor)
    
    print("\n   Publishing events with async processing:")
    
    await event_bus.publish(Event(type=EventType.TASK_CREATED, data={"name": "Task 1"}))
    await event_bus.publish(Event(type=EventType.TASK_STARTED, data={}))
    await event_bus.publish(Event(type=EventType.TASK_COMPLETED, data={"task": {}}))
    
    print(f"\n   Processed {len(processed)} events asynchronously")
    print("\n✅ Async handler complete")


# =============================================================================
# Example 9: Multiple Handlers
# =============================================================================

async def multiple_handlers_example():
    """Demonstrate multiple handlers on same event bus."""
    print("\n" + "=" * 60)
    print("Example 9: Multiple Handlers")
    print("=" * 60)
    
    event_bus = EventBus()
    
    # Handler 1: Console output
    console = ConsoleEventHandler(verbose=False, show_timestamp=False)
    
    # Handler 2: Metrics collection
    metrics = MetricsEventHandler()
    
    # Handler 3: Custom logging
    log: list[str] = []
    async def logger(event: Event):
        log.append(f"{event.type.value}: {event.data}")
    
    # Register all handlers
    event_bus.subscribe_all(console)
    event_bus.subscribe_all(metrics)
    event_bus.subscribe_all(logger)
    
    print("\n   Publishing events (handled by 3 handlers):")
    
    await event_bus.publish(Event(
        type=EventType.TASK_CREATED,
        data={"name": "Multi-handler task"},
    ))
    await event_bus.publish(Event(
        type=EventType.TASK_COMPLETED,
        data={"task": {"name": "Multi-handler task", "duration_ms": 75}},
    ))
    
    print(f"\n   Metrics handler tracked: {metrics.get_metrics()['total_events']} events")
    print(f"   Logger captured: {len(log)} entries")
    print("\n✅ Multiple handlers complete")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all event examples."""
    print("\n" + "📨 " * 20)
    print("AgenticFlow Events Examples")
    print("📨 " * 20)
    
    await basic_pubsub_example()
    await typed_subscription_example()
    await console_handler_example()
    await file_handler_example()
    await filtering_handler_example()
    await metrics_handler_example()
    await custom_handler_example()
    await async_handler_example()
    await multiple_handlers_example()
    
    print("\n" + "=" * 60)
    print("All event examples complete!")
    print("=" * 60)
    
    print("\n💡 EventBus features:")
    print("   - Pub/sub with sync and async handlers")
    print("   - Type-specific and global subscriptions")
    print("   - Built-in handlers (console, file, filtering, metrics)")
    print("   - Easy custom handler creation")
    print("   - Multiple handlers per event bus")


if __name__ == "__main__":
    asyncio.run(main())
