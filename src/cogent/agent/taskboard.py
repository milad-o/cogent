"""
TaskBoard - Agent task tracking and verification system.

Provides tools for agents to:
- Track tasks and their status
- Add notes and observations
- Verify task completion
- Learn from successes/failures (Reflexion-style)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from cogent.tools.base import BaseTool, tool

if TYPE_CHECKING:
    from cogent.observability.bus import TraceBus


class TaskStatus(StrEnum):
    """Status of a task on the taskboard."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskItem:
    """A task on the taskboard."""

    id: str
    name: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class TaskBoardConfig:
    """Configuration for TaskBoard feature."""

    include_instructions: bool = True
    max_tasks: int = 50
    track_time: bool = True


TASKBOARD_INSTRUCTIONS = """
## Task Tracking

You have a taskboard to track your work. Use it to:
1. **Plan**: Break down complex tasks into subtasks with `add_task`
2. **Track**: Update task status as you work with `update_task`
3. **Note**: Record observations and findings with `add_note`
4. **Verify**: Check task completion with `verify_task`
5. **Review**: See overall progress with `get_taskboard_status`

Best practices:
- Create tasks BEFORE starting work
- Update status to "in_progress" when you begin
- Add notes as you discover important information
- Mark tasks "completed" only after verifying the result
- Use "blocked" status if waiting on dependencies
"""


class TaskBoard:
    """
    Task tracking system for agents.

    Provides structured task management with:
    - Task creation and status tracking
    - Notes and observations
    - Completion verification
    - Progress summaries
    """

    def __init__(
        self,
        config: TaskBoardConfig | None = None,
        event_bus: TraceBus | None = None,
    ) -> None:
        self.config = config or TaskBoardConfig()
        self.event_bus = event_bus
        self._tasks: dict[str, TaskItem] = {}
        self._notes: list[tuple[datetime, str]] = []
        self._task_counter = 0

    def add_task(
        self,
        name: str,
        description: str = "",
    ) -> str:
        """Add a new task to the taskboard.

        Args:
            name: Short task name
            description: Detailed description

        Returns:
            Task ID
        """
        self._task_counter += 1
        task_id = f"task_{self._task_counter}"

        task = TaskItem(
            id=task_id,
            name=name,
            description=description,
        )
        self._tasks[task_id] = task
        return task_id

    def update_task(
        self,
        task_id: str,
        status: str | TaskStatus,
        result: str | None = None,
    ) -> bool:
        """Update task status.

        Args:
            task_id: Task identifier
            status: New status (pending, in_progress, completed, failed, blocked)
            result: Optional result/output for completed tasks

        Returns:
            True if updated, False if task not found
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]

        if isinstance(status, str):
            status = TaskStatus(status)

        task.status = status
        if result:
            task.result = result

        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now()

        return True

    def add_note(self, note: str) -> None:
        """Add an observation or note.

        Args:
            note: The note content
        """
        self._notes.append((datetime.now(), note))

    def add_task_note(self, task_id: str, note: str) -> bool:
        """Add a note to a specific task.

        Args:
            task_id: Task identifier
            note: The note content

        Returns:
            True if added, False if task not found
        """
        if task_id not in self._tasks:
            return False

        self._tasks[task_id].notes.append(note)
        return True

    def get_task(self, task_id: str) -> TaskItem | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_tasks_by_status(self, status: TaskStatus) -> list[TaskItem]:
        """Get all tasks with a specific status."""
        return [t for t in self._tasks.values() if t.status == status]

    def is_complete(self) -> bool:
        """Check if all tasks are completed or failed."""
        if not self._tasks:
            return True
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self._tasks.values()
        )

    def get_progress(self) -> dict[str, int]:
        """Get task counts by status."""
        counts: dict[str, int] = {s.value: 0 for s in TaskStatus}
        for task in self._tasks.values():
            counts[task.status.value] += 1
        return counts

    def summary(self) -> str:
        """Get a formatted summary of the taskboard."""
        lines = ["📋 TaskBoard Summary", "=" * 40]

        progress = self.get_progress()
        total = len(self._tasks)
        completed = progress["completed"]

        lines.append(f"Progress: {completed}/{total} tasks completed")
        lines.append("")

        # Tasks by status
        for status in TaskStatus:
            tasks = self.get_tasks_by_status(status)
            if tasks:
                emoji = {
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.IN_PROGRESS: "🔄",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌",
                    TaskStatus.BLOCKED: "🚫",
                }[status]
                lines.append(f"{emoji} {status.value.upper()} ({len(tasks)}):")
                for task in tasks:
                    result_str = f" → {task.result[:50]}..." if task.result and len(task.result) > 50 else (f" → {task.result}" if task.result else "")
                    lines.append(f"   • {task.name}{result_str}")
                lines.append("")

        # Recent notes
        if self._notes:
            lines.append("📝 Notes:")
            for _, note in self._notes[-5:]:  # Last 5 notes
                lines.append(f"   • {note}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export taskboard state as dictionary."""
        return {
            "tasks": {
                tid: {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "status": t.status.value,
                    "result": t.result,
                    "notes": t.notes,
                }
                for tid, t in self._tasks.items()
            },
            "notes": [(ts.isoformat(), note) for ts, note in self._notes],
            "progress": self.get_progress(),
        }


def create_taskboard_tools(taskboard: TaskBoard) -> list[BaseTool]:
    """Create tools for interacting with a taskboard.

    Args:
        taskboard: The TaskBoard instance to wrap

    Returns:
        List of tools for task management
    """

    @tool
    def add_task(name: str, description: str = "") -> str:
        """Add a new task to track.

        Use this to break down work into trackable subtasks.
        Create tasks BEFORE starting work on them.

        Args:
            name: Short, descriptive task name
            description: Detailed description of what needs to be done

        Returns:
            Task ID for reference
        """
        task_id = taskboard.add_task(name, description)
        return f"Created task '{name}' with ID: {task_id}"

    @tool
    def update_task(task_id: str, status: str, result: str = "") -> str:
        """Update the status of a task.

        Args:
            task_id: The task ID (e.g., "task_1")
            status: New status - one of: pending, in_progress, completed, failed, blocked
            result: Result or output (required when marking as completed)

        Returns:
            Confirmation message
        """
        if taskboard.update_task(task_id, status, result if result else None):
            return f"Updated {task_id} to '{status}'" + (f" with result: {result}" if result else "")
        return f"Task {task_id} not found"

    @tool
    def add_note(note: str) -> str:
        """Record an observation or finding.

        Use this to capture important information discovered during work.
        Notes help with later analysis and learning.

        Args:
            note: The observation or finding to record

        Returns:
            Confirmation message
        """
        taskboard.add_note(note)
        return f"Note recorded: {note}"

    @tool
    def verify_task(task_id: str, verification_result: str) -> str:
        """Verify that a task was completed correctly.

        Use this after completing a task to confirm the result is valid.
        If verification fails, update task status to 'failed'.

        Args:
            task_id: The task ID to verify
            verification_result: Description of verification outcome

        Returns:
            Verification status
        """
        task = taskboard.get_task(task_id)
        if not task:
            return f"Task {task_id} not found"

        taskboard.add_task_note(task_id, f"Verification: {verification_result}")

        if task.status == TaskStatus.COMPLETED:
            return f"Task {task_id} verified: {verification_result}"
        return f"Task {task_id} status is '{task.status.value}', not completed. Verification: {verification_result}"

    @tool
    def get_taskboard_status() -> str:
        """Get current taskboard status and progress.

        Shows all tasks organized by status, overall progress,
        and recent notes. Use this to review your work.

        Returns:
            Formatted taskboard summary
        """
        return taskboard.summary()

    return [add_task, update_task, add_note, verify_task, get_taskboard_status]
