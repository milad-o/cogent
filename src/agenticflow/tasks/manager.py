"""
TaskManager - manages tasks with hierarchy and dependency support.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agenticflow.core.enums import EventType, Priority, TaskStatus
from agenticflow.core.utils import generate_id, now_utc
from agenticflow.models.event import Event
from agenticflow.models.task import Task

if TYPE_CHECKING:
    from agenticflow.events.bus import EventBus


class TaskManager:
    """
    Manages tasks with hierarchy and dependency support.
    
    The TaskManager is responsible for:
    - Creating and tracking tasks
    - Managing parent/child relationships
    - Tracking dependencies between tasks
    - Emitting lifecycle events
    - Aggregating subtask results
    
    Attributes:
        event_bus: EventBus for publishing events
        tasks: Dictionary of all managed tasks
        
    Example:
        ```python
        manager = TaskManager(event_bus)
        
        # Create a task
        task = await manager.create_task(
            name="Analyze data",
            tool="analyze",
            args={"file": "data.csv"},
        )
        
        # Update status
        await manager.update_status(task.id, TaskStatus.RUNNING)
        
        # Get ready tasks
        ready = await manager.get_ready_tasks()
        ```
    """

    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize the TaskManager.
        
        Args:
            event_bus: EventBus for publishing events
        """
        self.event_bus = event_bus
        self.tasks: dict[str, Task] = {}
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        name: str,
        description: str = "",
        tool: str | None = None,
        args: dict | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
        priority: Priority = Priority.NORMAL,
        assigned_agent_id: str | None = None,
        correlation_id: str | None = None,
        **kwargs: Any,
    ) -> Task:
        """
        Create a new task and emit event.
        
        Args:
            name: Human-readable task name
            description: Detailed task description
            tool: Tool to execute (if any)
            args: Arguments for the tool
            parent_id: ID of parent task (for subtasks)
            depends_on: IDs of tasks that must complete first
            priority: Task priority level
            assigned_agent_id: ID of agent to assign to
            correlation_id: Correlation ID for event tracking
            **kwargs: Additional task attributes
            
        Returns:
            The created Task
        """
        task = Task(
            id=generate_id(),
            name=name,
            description=description,
            tool=tool,
            args=args or {},
            parent_id=parent_id,
            depends_on=depends_on or [],
            priority=priority,
            assigned_agent_id=assigned_agent_id,
            **kwargs,
        )

        async with self._lock:
            self.tasks[task.id] = task

            # Link to parent if specified
            if parent_id and parent_id in self.tasks:
                self.tasks[parent_id].add_subtask(task.id)

        # Emit appropriate event
        event_type = EventType.SUBTASK_SPAWNED if parent_id else EventType.TASK_CREATED
        await self.event_bus.publish(
            Event(
                type=event_type,
                data=task.to_dict(),
                source="task_manager",
                correlation_id=correlation_id,
            )
        )

        return task

    async def create_subtask(
        self,
        parent_id: str,
        name: str,
        tool: str | None = None,
        args: dict | None = None,
        correlation_id: str | None = None,
        **kwargs: Any,
    ) -> Task:
        """
        Create a subtask linked to a parent task.
        
        Args:
            parent_id: ID of the parent task
            name: Subtask name
            tool: Tool to execute
            args: Tool arguments
            correlation_id: Correlation ID for event tracking
            **kwargs: Additional task attributes
            
        Returns:
            The created subtask
            
        Raises:
            ValueError: If parent task doesn't exist
        """
        if parent_id not in self.tasks:
            raise ValueError(f"Parent task {parent_id} not found")

        return await self.create_task(
            name=name,
            tool=tool,
            args=args,
            parent_id=parent_id,
            correlation_id=correlation_id,
            **kwargs,
        )

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Any = None,
        error: str | None = None,
        correlation_id: str | None = None,
    ) -> Task:
        """
        Update task status and emit appropriate event.
        
        Args:
            task_id: ID of the task to update
            status: New status
            result: Task result (for completion)
            error: Error message (for failure)
            correlation_id: Correlation ID for event tracking
            
        Returns:
            The updated Task
            
        Raises:
            ValueError: If task doesn't exist
        """
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            old_status = task.status

            # Update task state
            if status == TaskStatus.SCHEDULED:
                task.mark_scheduled()
            elif status == TaskStatus.RUNNING:
                task.mark_started()
            elif status == TaskStatus.COMPLETED:
                task.mark_completed(result)
            elif status == TaskStatus.FAILED:
                task.mark_failed(error or "Unknown error")
            elif status == TaskStatus.CANCELLED:
                task.mark_cancelled()
            else:
                task.status = status

        # Map status to event type
        event_map = {
            TaskStatus.SCHEDULED: EventType.TASK_SCHEDULED,
            TaskStatus.RUNNING: EventType.TASK_STARTED,
            TaskStatus.BLOCKED: EventType.TASK_BLOCKED,
            TaskStatus.COMPLETED: EventType.TASK_COMPLETED,
            TaskStatus.FAILED: EventType.TASK_FAILED,
            TaskStatus.CANCELLED: EventType.TASK_CANCELLED,
        }

        event_type = event_map.get(status)
        if event_type:
            await self.event_bus.publish(
                Event(
                    type=event_type,
                    data={
                        "task": task.to_dict(),
                        "old_status": old_status.value,
                        "new_status": status.value,
                    },
                    source="task_manager",
                    correlation_id=correlation_id,
                )
            )

        return task

    async def get_task(self, task_id: str) -> Task | None:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            The Task or None if not found
        """
        return self.tasks.get(task_id)

    async def get_ready_tasks(self) -> list[Task]:
        """
        Get tasks ready to execute (dependencies met).
        
        Returns:
            List of tasks sorted by priority (highest first)
        """
        ready = []

        async with self._lock:
            for task in self.tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue

                # Check dependencies
                deps_met = all(
                    self.tasks.get(dep_id, Task(name="")).status == TaskStatus.COMPLETED
                    for dep_id in task.depends_on
                )

                if deps_met:
                    ready.append(task)

        # Sort by priority (highest first)
        ready.sort(key=lambda t: t.priority.value, reverse=True)
        return ready

    async def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """
        Get all tasks with a specific status.
        
        Args:
            status: The status to filter by
            
        Returns:
            List of matching tasks
        """
        return [t for t in self.tasks.values() if t.status == status]

    async def get_tasks_by_agent(self, agent_id: str) -> list[Task]:
        """
        Get all tasks assigned to an agent.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            List of tasks assigned to the agent
        """
        return [t for t in self.tasks.values() if t.assigned_agent_id == agent_id]

    async def assign_task(
        self,
        task_id: str,
        agent_id: str,
        correlation_id: str | None = None,
    ) -> Task:
        """
        Assign a task to an agent.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            correlation_id: Correlation ID for event tracking
            
        Returns:
            The updated Task
        """
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            task.assigned_agent_id = agent_id

        return await self.update_status(
            task_id,
            TaskStatus.SCHEDULED,
            correlation_id=correlation_id,
        )

    async def retry_task(
        self,
        task_id: str,
        correlation_id: str | None = None,
    ) -> bool:
        """
        Retry a failed task.
        
        Args:
            task_id: ID of the task to retry
            correlation_id: Correlation ID for event tracking
            
        Returns:
            True if retry was successful, False if max retries exceeded
        """
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            if not task.increment_retry():
                return False

        await self.event_bus.publish(
            Event(
                type=EventType.TASK_RETRYING,
                data={
                    "task_id": task_id,
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries,
                },
                source="task_manager",
                correlation_id=correlation_id,
            )
        )

        return True

    async def cancel_task(
        self,
        task_id: str,
        correlation_id: str | None = None,
    ) -> Task:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task to cancel
            correlation_id: Correlation ID for event tracking
            
        Returns:
            The cancelled Task
        """
        return await self.update_status(
            task_id,
            TaskStatus.CANCELLED,
            correlation_id=correlation_id,
        )

    async def aggregate_subtasks(
        self,
        parent_id: str,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate results from all subtasks of a parent.
        
        Args:
            parent_id: ID of the parent task
            correlation_id: Correlation ID for event tracking
            
        Returns:
            Dictionary mapping subtask names to results
        """
        results: dict[str, Any] = {}

        async with self._lock:
            parent = self.tasks.get(parent_id)
            if not parent:
                return results

            for subtask_id in parent.subtask_ids:
                subtask = self.tasks.get(subtask_id)
                if subtask and subtask.status == TaskStatus.COMPLETED:
                    results[subtask.name] = subtask.result

        await self.event_bus.publish(
            Event(
                type=EventType.SUBTASKS_AGGREGATED,
                data={
                    "parent_id": parent_id,
                    "subtask_count": len(results),
                    "results": {k: str(v)[:100] for k, v in results.items()},
                },
                source="task_manager",
                correlation_id=correlation_id,
            )
        )

        return results

    async def check_subtasks_complete(self, parent_id: str) -> bool:
        """
        Check if all subtasks of a parent are complete.
        
        Args:
            parent_id: ID of the parent task
            
        Returns:
            True if all subtasks are complete
        """
        async with self._lock:
            parent = self.tasks.get(parent_id)
            if not parent or not parent.subtask_ids:
                return True

            return all(
                self.tasks.get(sub_id, Task(name="")).is_complete()
                for sub_id in parent.subtask_ids
            )

    def get_task_tree(self, task_id: str) -> dict:
        """
        Get a task and all its subtasks as a tree.
        
        Args:
            task_id: ID of the root task
            
        Returns:
            Dictionary with task data and nested subtasks
        """
        task = self.tasks.get(task_id)
        if not task:
            return {}

        tree = task.to_dict()
        tree["subtasks"] = [
            self.get_task_tree(sub_id) for sub_id in task.subtask_ids
        ]
        return tree

    def get_stats(self) -> dict:
        """
        Get task statistics.
        
        Returns:
            Dictionary with task counts by status
        """
        stats: dict[str, int] = {}
        for task in self.tasks.values():
            status = task.status.value
            stats[status] = stats.get(status, 0) + 1

        return {
            "total": len(self.tasks),
            "by_status": stats,
        }

    async def clear(self) -> None:
        """Clear all tasks."""
        async with self._lock:
            self.tasks.clear()
