# Tasks Module

The `agenticflow.tasks` module provides hierarchical task tracking with full lifecycle management.

## Overview

Tasks represent units of work with:
- Full lifecycle tracking (created → scheduled → running → completed)
- Hierarchical relationships (parent/subtasks)
- Dependency management
- Priority-based scheduling
- Retry logic

```python
from agenticflow.tasks import TaskManager, Task
from agenticflow.observability import EventBus

bus = EventBus()
manager = TaskManager(bus)

# Create a task
task = await manager.create_task(
    name="Analyze data",
    tool="analyze",
    args={"file": "data.csv"},
)
```

---

## Task

A single unit of work:

```python
from agenticflow.tasks import Task
from agenticflow.core import Priority, TaskStatus

task = Task(
    name="Process report",
    description="Generate monthly sales report",
    tool="generate_report",
    args={"month": "January", "year": 2024},
    priority=Priority.HIGH,
)

print(task.id)       # Unique ID
print(task.status)   # TaskStatus.PENDING
print(task.created_at)
```

### Task Properties

```python
@dataclass
class Task:
    # Identity
    name: str
    description: str = ""
    id: str  # Auto-generated
    
    # Execution
    tool: str | None = None
    args: dict = field(default_factory=dict)
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.NORMAL
    
    # Assignment
    assigned_agent_id: str | None = None
    
    # Hierarchy
    parent_id: str | None = None
    subtask_ids: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    
    # Results
    result: Any = None
    error: str | None = None
    
    # Timestamps
    created_at: datetime
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # Metrics
    duration_ms: float | None = None
    retry_count: int = 0
    max_retries: int = 3
```

### TaskStatus

```python
from agenticflow.core import TaskStatus

TaskStatus.PENDING     # Created, not scheduled
TaskStatus.SCHEDULED   # Scheduled for execution
TaskStatus.BLOCKED     # Waiting on dependencies
TaskStatus.RUNNING     # Actively executing
TaskStatus.SPAWNING    # Creating subtasks
TaskStatus.COMPLETED   # Finished successfully
TaskStatus.FAILED      # Failed with error
TaskStatus.CANCELLED   # Cancelled

# Check categories
status.is_terminal()  # COMPLETED, FAILED, CANCELLED
status.is_active()    # RUNNING, SPAWNING
```

### Priority

```python
from agenticflow.core import Priority

Priority.LOW
Priority.NORMAL
Priority.HIGH
Priority.CRITICAL
```

---

## TaskManager

Manages task lifecycle and relationships:

```python
from agenticflow.tasks import TaskManager
from agenticflow.observability import EventBus

bus = EventBus()
manager = TaskManager(bus)
```

### Creating Tasks

```python
# Basic task
task = await manager.create_task(
    name="Research topic",
    description="Research AI developments",
)

# With tool and args
task = await manager.create_task(
    name="Search web",
    tool="web_search",
    args={"query": "AI news 2024"},
    priority=Priority.HIGH,
)

# Assigned to agent
task = await manager.create_task(
    name="Write report",
    assigned_agent_id="writer-001",
)
```

### Subtasks

```python
# Create parent task
parent = await manager.create_task(name="Complete project")

# Create subtasks
subtask1 = await manager.create_subtask(
    parent_id=parent.id,
    name="Research phase",
    tool="research",
)

subtask2 = await manager.create_subtask(
    parent_id=parent.id,
    name="Writing phase",
    tool="write",
)

# Subtasks are linked to parent
print(parent.subtask_ids)  # [subtask1.id, subtask2.id]
```

### Dependencies

```python
# Create independent tasks
task_a = await manager.create_task(name="Task A")
task_b = await manager.create_task(name="Task B")

# Create dependent task
task_c = await manager.create_task(
    name="Task C",
    depends_on=[task_a.id, task_b.id],  # Requires A and B
)

# Task C won't be ready until A and B complete
```

### Status Updates

```python
from agenticflow.core import TaskStatus

# Update status
await manager.update_status(task.id, TaskStatus.RUNNING)

# Complete with result
await manager.complete_task(task.id, result="Analysis complete")

# Fail with error
await manager.fail_task(task.id, error="Connection timeout")

# Cancel
await manager.cancel_task(task.id)
```

### Querying Tasks

```python
# Get task by ID
task = await manager.get_task(task_id)

# Get all tasks
all_tasks = await manager.get_all_tasks()

# Get tasks by status
running = await manager.get_tasks_by_status(TaskStatus.RUNNING)

# Get ready tasks (no blocking dependencies)
ready = await manager.get_ready_tasks()

# Get tasks for an agent
agent_tasks = await manager.get_agent_tasks("agent-001")
```

### Priority Scheduling

```python
# Get next task by priority
next_task = await manager.get_next_task()

# Get next for specific agent
next_task = await manager.get_next_task(agent_id="agent-001")
```

---

## Event Integration

TaskManager publishes events for all lifecycle changes:

```python
from agenticflow.core import EventType

bus.subscribe(EventType.TASK_CREATED, on_task_created)
bus.subscribe(EventType.TASK_STARTED, on_task_started)
bus.subscribe(EventType.TASK_COMPLETED, on_task_completed)
bus.subscribe(EventType.TASK_FAILED, on_task_failed)
bus.subscribe(EventType.SUBTASK_SPAWNED, on_subtask_spawned)

async def on_task_completed(event):
    task_data = event.data
    print(f"Task {task_data['name']} completed: {task_data['result']}")
```

---

## Task Serialization

```python
# Convert to dict
task_dict = task.to_dict()

# Convert to JSON
task_json = task.to_json()

# Create from dict
task = Task.from_dict(task_dict)

# Create from JSON
task = Task.from_json(task_json)
```

---

## Task Lifecycle Methods

```python
task = Task(name="Example")

# State transitions
task.schedule()     # PENDING → SCHEDULED
task.start()        # → RUNNING
task.spawn()        # → SPAWNING (creating subtasks)
task.complete(result="Done")  # → COMPLETED
task.fail("Error")  # → FAILED
task.cancel()       # → CANCELLED

# Subtask management
task.add_subtask(subtask_id)
task.remove_subtask(subtask_id)
task.add_dependency(other_task_id)
```

---

## Retry Logic

```python
task = Task(
    name="Flaky operation",
    max_retries=3,
)

# On failure
try:
    result = await execute_task(task)
except Exception as e:
    if task.retry_count < task.max_retries:
        task.retry_count += 1
        task.status = TaskStatus.PENDING  # Retry
    else:
        task.fail(str(e))
```

---

## Workflow Example

```python
from agenticflow.tasks import TaskManager, Task
from agenticflow.observability import EventBus

bus = EventBus()
manager = TaskManager(bus)

async def run_workflow():
    # Create main task
    main = await manager.create_task(
        name="Generate Report",
        description="Create Q4 sales report",
    )
    
    # Spawn subtasks
    research = await manager.create_subtask(
        parent_id=main.id,
        name="Gather Data",
        tool="query_database",
    )
    
    analyze = await manager.create_subtask(
        parent_id=main.id,
        name="Analyze Data",
        tool="analyze",
        depends_on=[research.id],  # Needs research first
    )
    
    write = await manager.create_subtask(
        parent_id=main.id,
        name="Write Report",
        tool="write",
        depends_on=[analyze.id],  # Needs analysis first
    )
    
    # Execute in dependency order
    while ready := await manager.get_ready_tasks():
        for task in ready:
            await manager.update_status(task.id, TaskStatus.RUNNING)
            result = await execute(task)
            await manager.complete_task(task.id, result=result)
    
    # Check completion
    main = await manager.get_task(main.id)
    print(f"Workflow complete: {main.status}")
```

---

## API Reference

### Task

| Method | Description |
|--------|-------------|
| `to_dict()` | Convert to dictionary |
| `to_json()` | Convert to JSON string |
| `from_dict(data)` | Create from dictionary |
| `from_json(json_str)` | Create from JSON |
| `schedule()` | Transition to SCHEDULED |
| `start()` | Transition to RUNNING |
| `complete(result)` | Transition to COMPLETED |
| `fail(error)` | Transition to FAILED |
| `cancel()` | Transition to CANCELLED |
| `add_subtask(id)` | Add subtask ID |
| `add_dependency(id)` | Add dependency |

### TaskManager

| Method | Description |
|--------|-------------|
| `create_task(...)` | Create new task |
| `create_subtask(parent_id, ...)` | Create subtask |
| `get_task(id)` | Get task by ID |
| `get_all_tasks()` | Get all tasks |
| `get_tasks_by_status(status)` | Filter by status |
| `get_ready_tasks()` | Get tasks with satisfied deps |
| `get_next_task(agent_id?)` | Get highest priority ready task |
| `update_status(id, status)` | Update task status |
| `complete_task(id, result)` | Complete with result |
| `fail_task(id, error)` | Fail with error |
| `cancel_task(id)` | Cancel task |
