"""
Tasks Example - demonstrates hierarchical task management.

This example shows:
1. Creating and managing tasks
2. Task hierarchies with subtasks
3. Task dependencies
4. Task lifecycle and status transitions

Usage:
    uv run python examples/tasks_example.py
"""

import asyncio

from agenticflow import (
    EventBus,
    TaskManager,
    Task,
    TaskStatus,
    Priority,
    Event,
)


# =============================================================================
# Example 1: Basic Task Creation
# =============================================================================

async def basic_task_example():
    """Demonstrate basic task creation and management."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Task Creation")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Create a simple task
    task = await task_manager.create_task(
        name="Research AI trends",
        description="Find the latest developments in AI",
        priority=Priority.HIGH,
        metadata={"category": "research", "estimated_time": "30min"},
    )
    
    print(f"\n📋 Created Task:")
    print(f"   ID: {task.id[:8]}...")
    print(f"   Name: {task.name}")
    print(f"   Status: {task.status.value}")
    print(f"   Priority: {task.priority.value}")
    print(f"   Metadata: {task.metadata}")
    
    # Get task by ID
    retrieved = await task_manager.get_task(task.id)
    print(f"\n🔍 Retrieved task: {retrieved.name}")
    
    print("\n✅ Basic task creation complete")


# =============================================================================
# Example 2: Task Status Transitions
# =============================================================================

async def task_status_example():
    """Demonstrate task status transitions."""
    print("\n" + "=" * 60)
    print("Example 2: Task Status Transitions")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Track events
    events = []
    async def event_tracker(event: Event):
        events.append(event)
    event_bus.subscribe_all(event_tracker)
    
    task = await task_manager.create_task(name="Demo task")
    
    print(f"\n📋 Task lifecycle:")
    print(f"   1. Created: {task.status.value}")
    
    # Start the task
    await task_manager.update_status(task.id, TaskStatus.RUNNING)
    print(f"   2. Running: {(await task_manager.get_task(task.id)).status.value}")
    
    # Complete the task
    await task_manager.update_status(
        task.id,
        TaskStatus.COMPLETED,
        result={"output": "Task completed successfully"},
    )
    updated = await task_manager.get_task(task.id)
    print(f"   3. Completed: {updated.status.value}")
    print(f"      Result: {updated.result}")
    
    print(f"\n📊 Events emitted: {len(events)}")
    for event in events:
        print(f"   - {event.type.value}")
    
    # Show all valid transitions
    print(f"\n🔄 Valid status transitions:")
    print(f"   PENDING → SCHEDULED, RUNNING, CANCELLED")
    print(f"   RUNNING → COMPLETED, FAILED, CANCELLED")
    print(f"   FAILED → PENDING (retry)")
    
    print("\n✅ Task status transitions complete")


# =============================================================================
# Example 3: Subtasks (Hierarchical Tasks)
# =============================================================================

async def subtask_example():
    """Demonstrate hierarchical task structure."""
    print("\n" + "=" * 60)
    print("Example 3: Subtasks (Hierarchical Tasks)")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Create parent task
    parent = await task_manager.create_task(
        name="Write blog post",
        description="Create a comprehensive blog post about AI",
    )
    
    # Create subtasks
    research = await task_manager.create_subtask(
        parent_id=parent.id,
        name="Research",
    )
    
    outline = await task_manager.create_subtask(
        parent_id=parent.id,
        name="Create outline",
    )
    
    write = await task_manager.create_subtask(
        parent_id=parent.id,
        name="Write content",
    )
    
    review = await task_manager.create_subtask(
        parent_id=parent.id,
        name="Review and edit",
    )
    
    print(f"\n📋 Task Hierarchy:")
    print(f"   📁 {parent.name} ({parent.id[:8]}...)")
    
    # Re-fetch parent to get updated subtask_ids
    parent = await task_manager.get_task(parent.id)
    for subtask_id in parent.subtask_ids:
        subtask = await task_manager.get_task(subtask_id)
        print(f"      └── {subtask.name} ({subtask.status.value})")
    
    # Get task tree
    tree = task_manager.get_task_tree(parent.id)
    print(f"\n🌳 Task Tree:")
    print(f"   Root: {tree['name']}")
    print(f"   Subtask count: {len(tree['subtasks'])}")
    
    print("\n✅ Subtask hierarchy complete")


# =============================================================================
# Example 4: Task Dependencies
# =============================================================================

async def dependencies_example():
    """Demonstrate task dependencies."""
    print("\n" + "=" * 60)
    print("Example 4: Task Dependencies")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Create tasks with dependencies
    # Task flow: fetch_data → process_data → generate_report
    
    fetch_task = await task_manager.create_task(
        name="Fetch data",
        description="Retrieve data from source",
    )
    
    process_task = await task_manager.create_task(
        name="Process data",
        description="Transform and clean data",
        depends_on=[fetch_task.id],  # Depends on fetch
    )
    
    report_task = await task_manager.create_task(
        name="Generate report",
        description="Create final report",
        depends_on=[process_task.id],  # Depends on process
    )
    
    print(f"\n📋 Tasks with dependencies:")
    print(f"   1. {fetch_task.name} (no dependencies)")
    print(f"   2. {process_task.name} (depends on: fetch)")
    print(f"   3. {report_task.name} (depends on: process)")
    
    # Check ready tasks
    ready = await task_manager.get_ready_tasks()
    print(f"\n🟢 Ready to run (no pending dependencies):")
    for task in ready:
        print(f"   - {task.name}")
    
    # Complete fetch task
    await task_manager.update_status(fetch_task.id, TaskStatus.RUNNING)
    await task_manager.update_status(fetch_task.id, TaskStatus.COMPLETED)
    
    # Now process should be ready
    ready = await task_manager.get_ready_tasks()
    print(f"\n🟢 Ready after completing 'Fetch data':")
    for task in ready:
        print(f"   - {task.name}")
    
    print("\n✅ Task dependencies complete")


# =============================================================================
# Example 5: Task Retry
# =============================================================================

async def retry_example():
    """Demonstrate task retry mechanism."""
    print("\n" + "=" * 60)
    print("Example 5: Task Retry")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Create task with retry configuration
    task = await task_manager.create_task(
        name="Flaky API call",
        description="Call an unreliable external API",
        max_retries=3,
    )
    
    print(f"\n📋 Task: {task.name}")
    print(f"   Max retries: {task.max_retries}")
    print(f"   Current retry count: {task.retry_count}")
    
    # Simulate failures and retries
    for attempt in range(1, 4):
        print(f"\n   Attempt {attempt}:")
        
        await task_manager.update_status(task.id, TaskStatus.RUNNING)
        
        # Simulate failure
        await task_manager.update_status(
            task.id,
            TaskStatus.FAILED,
            error=f"API timeout on attempt {attempt}",
        )
        
        current = await task_manager.get_task(task.id)
        print(f"      Status: {current.status.value}")
        print(f"      Error: {current.error}")
        
        # Retry if not at max
        if attempt < 3:
            can_retry = await task_manager.retry_task(task.id)
            if can_retry:
                current = await task_manager.get_task(task.id)
                print(f"      Retrying... (retry #{current.retry_count})")
    
    # Final attempt succeeds
    await task_manager.retry_task(task.id)
    await task_manager.update_status(task.id, TaskStatus.RUNNING)
    await task_manager.update_status(task.id, TaskStatus.COMPLETED, result={"data": "success"})
    
    final = await task_manager.get_task(task.id)
    print(f"\n   Final status: {final.status.value}")
    print(f"   Total retries: {final.retry_count}")
    
    print("\n✅ Task retry complete")


# =============================================================================
# Example 6: Task Cancellation
# =============================================================================

async def cancellation_example():
    """Demonstrate task cancellation."""
    print("\n" + "=" * 60)
    print("Example 6: Task Cancellation")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Create some tasks
    task1 = await task_manager.create_task(name="Task 1 (pending)")
    task2 = await task_manager.create_task(name="Task 2 (running)")
    task3 = await task_manager.create_task(name="Task 3 (completed)")
    
    # Set different states
    await task_manager.update_status(task2.id, TaskStatus.RUNNING)
    await task_manager.update_status(task3.id, TaskStatus.RUNNING)
    await task_manager.update_status(task3.id, TaskStatus.COMPLETED)
    
    print(f"\n📋 Tasks before cancellation:")
    for task_id in [task1.id, task2.id, task3.id]:
        current = await task_manager.get_task(task_id)
        print(f"   - {current.name}: {current.status.value}")
    
    # Cancel pending task
    await task_manager.cancel_task(task1.id)
    t1 = await task_manager.get_task(task1.id)
    print(f"\n🚫 Cancel pending task: {t1.status.value}")
    
    # Cancel running task
    await task_manager.cancel_task(task2.id)
    t2 = await task_manager.get_task(task2.id)
    print(f"🚫 Cancel running task: {t2.status.value}")
    
    # Cancel completed task (will still mark as cancelled)
    await task_manager.cancel_task(task3.id)
    t3 = await task_manager.get_task(task3.id)
    print(f"🚫 Cancel completed task: {t3.status.value}")
    
    print(f"\n📋 Tasks after cancellation:")
    for task_id in [task1.id, task2.id, task3.id]:
        current = await task_manager.get_task(task_id)
        print(f"   - {current.name}: {current.status.value}")
    
    print("\n✅ Task cancellation complete")


# =============================================================================
# Example 7: Task Statistics
# =============================================================================

async def stats_example():
    """Demonstrate task statistics."""
    print("\n" + "=" * 60)
    print("Example 7: Task Statistics")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Create various tasks
    tasks_config = [
        ("Task A", TaskStatus.COMPLETED, Priority.HIGH),
        ("Task B", TaskStatus.COMPLETED, Priority.NORMAL),
        ("Task C", TaskStatus.COMPLETED, Priority.LOW),
        ("Task D", TaskStatus.RUNNING, Priority.HIGH),
        ("Task E", TaskStatus.RUNNING, Priority.NORMAL),
        ("Task F", TaskStatus.PENDING, Priority.HIGH),
        ("Task G", TaskStatus.FAILED, Priority.CRITICAL),
    ]
    
    for name, target_status, priority in tasks_config:
        task = await task_manager.create_task(name=name, priority=priority)
        if target_status != TaskStatus.PENDING:
            await task_manager.update_status(task.id, TaskStatus.RUNNING)
            if target_status == TaskStatus.COMPLETED:
                await task_manager.update_status(task.id, TaskStatus.COMPLETED)
            elif target_status == TaskStatus.FAILED:
                await task_manager.update_status(task.id, TaskStatus.FAILED, error="Simulated error")
    
    # Get statistics
    stats = task_manager.get_stats()
    
    print(f"\n📊 Task Statistics:")
    print(f"\n   Total tasks: {stats['total']}")
    
    print(f"\n   By Status:")
    for status, count in stats['by_status'].items():
        print(f"      {status}: {count}")
    
    print("\n✅ Task statistics complete")


# =============================================================================
# Example 8: Aggregate Subtask Results
# =============================================================================

async def aggregate_results_example():
    """Demonstrate aggregating subtask results."""
    print("\n" + "=" * 60)
    print("Example 8: Aggregate Subtask Results")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Create parent task
    parent = await task_manager.create_task(
        name="Data analysis",
        description="Analyze multiple data sources",
    )
    
    # Create subtasks with results
    data_sources = ["sales", "inventory", "customers"]
    
    for source in data_sources:
        subtask = await task_manager.create_subtask(
            parent_id=parent.id,
            name=f"Analyze {source}",
        )
        await task_manager.update_status(subtask.id, TaskStatus.RUNNING)
        await task_manager.update_status(
            subtask.id,
            TaskStatus.COMPLETED,
            result={
                "source": source,
                "records": len(source) * 1000,
                "insights": f"Key findings from {source}",
            },
        )
    
    # Check if all subtasks complete
    all_complete = await task_manager.check_subtasks_complete(parent.id)
    print(f"\n✓ All subtasks complete: {all_complete}")
    
    # Aggregate results
    aggregated = await task_manager.aggregate_subtasks(parent.id)
    
    print(f"\n📊 Aggregated Results:")
    print(f"   Subtask count: {len(aggregated)}")
    print(f"\n   Individual results:")
    for name, result in aggregated.items():
        print(f"      - {name}: {result.get('source', 'N/A')}")
    
    print("\n✅ Aggregate results complete")


# =============================================================================
# Example 9: Task Assignment to Agents
# =============================================================================

async def task_assignment_example():
    """Demonstrate assigning tasks to agents."""
    print("\n" + "=" * 60)
    print("Example 9: Task Assignment to Agents")
    print("=" * 60)
    
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)
    
    # Simulate agent IDs
    researcher_id = "agent-researcher-001"
    writer_id = "agent-writer-002"
    
    # Create tasks
    task1 = await task_manager.create_task(
        name="Research topic",
        description="Gather information about AI",
    )
    
    task2 = await task_manager.create_task(
        name="Write article",
        description="Write an article based on research",
        depends_on=[task1.id],
    )
    
    # Assign tasks to agents
    await task_manager.assign_task(task1.id, researcher_id)
    await task_manager.assign_task(task2.id, writer_id)
    
    print(f"\n📋 Task Assignments:")
    t1 = await task_manager.get_task(task1.id)
    t2 = await task_manager.get_task(task2.id)
    print(f"   {t1.name} → {t1.assigned_agent_id}")
    print(f"   {t2.name} → {t2.assigned_agent_id}")
    
    # Get tasks by agent
    researcher_tasks = await task_manager.get_tasks_by_agent(researcher_id)
    writer_tasks = await task_manager.get_tasks_by_agent(writer_id)
    
    print(f"\n👤 Researcher's tasks: {len(researcher_tasks)}")
    for task in researcher_tasks:
        print(f"   - {task.name} ({task.status.value})")
    
    print(f"\n👤 Writer's tasks: {len(writer_tasks)}")
    for task in writer_tasks:
        print(f"   - {task.name} ({task.status.value})")
    
    print("\n✅ Task assignment complete")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all task examples."""
    print("\n" + "📋 " * 20)
    print("AgenticFlow Tasks Examples")
    print("📋 " * 20)
    
    await basic_task_example()
    await task_status_example()
    await subtask_example()
    await dependencies_example()
    await retry_example()
    await cancellation_example()
    await stats_example()
    await aggregate_results_example()
    await task_assignment_example()
    
    print("\n" + "=" * 60)
    print("All task examples complete!")
    print("=" * 60)
    
    print("\n💡 Task management features:")
    print("   - Hierarchical tasks with subtasks")
    print("   - Task dependencies for sequencing")
    print("   - Automatic retry with backoff")
    print("   - Event emission for each state change")
    print("   - Statistics and aggregation")
    print("   - Agent assignment")


if __name__ == "__main__":
    asyncio.run(main())
