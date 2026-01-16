"""
Task - a unit of work with lifecycle tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agenticflow.core.enums import Priority, TaskStatus
from agenticflow.core.utils import format_duration_ms, generate_id, now_utc


@dataclass
class Task:
    """
    A unit of work with full lifecycle tracking.

    Tasks support:
    - Hierarchical relationships (parent/child)
    - Dependency management
    - Full audit trail with timestamps
    - Result and error tracking
    - Priority-based scheduling
    - Retry logic

    Attributes:
        name: Human-readable task name
        description: Detailed task description
        tool: Tool to execute (if any)
        args: Arguments for the tool
        id: Unique task identifier
        status: Current lifecycle status
        priority: Task priority level
        assigned_agent_id: ID of agent assigned to this task
        parent_id: ID of parent task (for subtasks)
        subtask_ids: IDs of child tasks
        depends_on: IDs of tasks that must complete first
        result: Task execution result
        error: Error message if failed
        created_at: When task was created
        scheduled_at: When task was scheduled
        started_at: When execution started
        completed_at: When task completed/failed
        duration_ms: Execution duration in milliseconds
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts allowed

    Example:
        ```python
        task = Task(
            name="Analyze sales data",
            description="Review Q4 sales and identify trends",
            tool="analyze_data",
            args={"dataset": "sales_q4"},
            priority=Priority.HIGH,
        )
        ```
    """

    name: str
    description: str = ""
    tool: str | None = None
    args: dict = field(default_factory=dict)
    id: str = field(default_factory=generate_id)
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

    # Timestamps (full audit trail)
    created_at: datetime = field(default_factory=now_utc)
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Metrics
    duration_ms: float | None = None
    retry_count: int = 0
    max_retries: int = 3

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation of the task
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tool": self.tool,
            "args": self.args,
            "status": self.status.value,
            "priority": self.priority.value,
            "assigned_agent_id": self.assigned_agent_id,
            "parent_id": self.parent_id,
            "subtask_ids": self.subtask_ids,
            "depends_on": self.depends_on,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """
        Convert to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        """
        Create a Task from a dictionary.

        Args:
            data: Dictionary with task data

        Returns:
            New Task instance
        """
        task = cls(
            id=data.get("id", generate_id()),
            name=data["name"],
            description=data.get("description", ""),
            tool=data.get("tool"),
            args=data.get("args", {}),
            status=TaskStatus(data.get("status", "pending")),
            priority=Priority(data.get("priority", 2)),
            assigned_agent_id=data.get("assigned_agent_id"),
            parent_id=data.get("parent_id"),
            subtask_ids=data.get("subtask_ids", []),
            depends_on=data.get("depends_on", []),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {}),
        )

        # Parse timestamps
        if data.get("created_at"):
            task.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("scheduled_at"):
            task.scheduled_at = datetime.fromisoformat(data["scheduled_at"])
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])

        return task

    def is_complete(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status.is_terminal()

    def is_active(self) -> bool:
        """Check if task is currently being executed."""
        return self.status.is_active()

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED

    def has_subtasks(self) -> bool:
        """Check if task has subtasks."""
        return len(self.subtask_ids) > 0

    def has_dependencies(self) -> bool:
        """Check if task has unmet dependencies."""
        return len(self.depends_on) > 0

    def is_root_task(self) -> bool:
        """Check if task is a root task (no parent)."""
        return self.parent_id is None

    def mark_scheduled(self) -> None:
        """Mark task as scheduled."""
        self.status = TaskStatus.SCHEDULED
        self.scheduled_at = now_utc()

    def mark_started(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = now_utc()

    def mark_completed(self, result: Any = None) -> None:
        """Mark task as completed with optional result."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = now_utc()
        self.result = result
        self.duration_ms = format_duration_ms(self.started_at, self.completed_at)

    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.completed_at = now_utc()
        self.error = error
        self.duration_ms = format_duration_ms(self.started_at, self.completed_at)

    def mark_cancelled(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = now_utc()

    def increment_retry(self) -> bool:
        """
        Increment retry count and reset for retry.

        Returns:
            True if retry is allowed, False if max retries exceeded
        """
        if not self.can_retry():
            return False

        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.error = None
        self.started_at = None
        self.completed_at = None
        return True

    def add_subtask(self, subtask_id: str) -> None:
        """Add a subtask ID to this task."""
        if subtask_id not in self.subtask_ids:
            self.subtask_ids.append(subtask_id)

    def add_dependency(self, task_id: str) -> None:
        """Add a dependency to this task."""
        if task_id not in self.depends_on:
            self.depends_on.append(task_id)

    def create_subtask(
        self,
        name: str,
        tool: str | None = None,
        args: dict | None = None,
        **kwargs: Any,
    ) -> Task:
        """
        Create a subtask linked to this task.

        Args:
            name: Subtask name
            tool: Tool to execute
            args: Tool arguments
            **kwargs: Additional task attributes

        Returns:
            New Task linked as a subtask
        """
        subtask = Task(
            name=name,
            tool=tool,
            args=args or {},
            parent_id=self.id,
            **kwargs,
        )
        self.add_subtask(subtask.id)
        return subtask

    def __repr__(self) -> str:
        return f"Task(id={self.id}, name={self.name}, status={self.status.value})"
