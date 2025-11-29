"""
TaskBoard - Task tracking and working memory for agents.

Provides agents with human-like task management:
- **Tasks**: Track work items with status (pending/in-progress/done/failed)
- **Notes**: Capture observations, insights, and questions
- **Learning**: Remember what worked and what didn't (Reflexion-style)

When enabled on an agent, it gets tools to manage its own work:
- `plan_tasks` - Break down work into trackable tasks
- `update_task` - Mark tasks as started, done, or failed
- `add_note` - Record observations and insights
- `check_progress` - Review current status
- `verify_done` - Confirm all work is complete

Example:
    ```python
    # Enable task tracking on an agent
    agent = Agent(
        name="Researcher",
        model=model,
        tools=[search, summarize],
        taskboard=True,  # Adds task management tools + instructions
    )
    
    # Or with configuration
    agent = Agent(
        name="Researcher",
        model=model,
        taskboard=TaskBoardConfig(
            auto_verify=True,      # Require verification before completing
            max_tasks=20,          # Limit concurrent tasks
        ),
    )
    
    # Agent now naturally plans, tracks, and verifies work
    result = await agent.run("Research Python async patterns")
    
    # Check what the agent tracked
    print(agent.taskboard.summary())
    ```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.core.utils import generate_id, now_utc

if TYPE_CHECKING:
    from agenticflow.tools.base import BaseTool


# ============================================================================
# Task Status
# ============================================================================

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Task:
    """A trackable task item."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=now_utc)
    completed_at: datetime | None = None
    result: str | None = None
    error: str | None = None
    
    def start(self) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS
    
    def complete(self, result: str | None = None) -> None:
        """Mark task as done with optional result."""
        self.status = TaskStatus.DONE
        self.completed_at = now_utc()
        self.result = result
    
    def fail(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.completed_at = now_utc()
        self.error = error
    
    def skip(self) -> None:
        """Mark task as skipped."""
        self.status = TaskStatus.SKIPPED
        self.completed_at = now_utc()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }
    
    def __str__(self) -> str:
        icons = {
            TaskStatus.PENDING: "○",
            TaskStatus.IN_PROGRESS: "→",
            TaskStatus.DONE: "✓",
            TaskStatus.FAILED: "✗",
            TaskStatus.SKIPPED: "○",
        }
        return f"[{icons.get(self.status, '?')}] {self.description}"


@dataclass
class Note:
    """A note or observation."""
    id: str
    content: str
    category: str = "observation"  # observation, insight, question
    created_at: datetime = field(default_factory=now_utc)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
        }


@dataclass
class ErrorRecord:
    """Record of an error for learning."""
    tool_name: str
    error_type: str
    error_message: str
    args: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)
    fix_applied: str | None = None
    fix_worked: bool = False


@dataclass
class LearnedPattern:
    """A learned failure pattern with optional fix."""
    tool_name: str
    pattern: str
    count: int = 1
    fix: str | None = None
    last_seen: datetime = field(default_factory=now_utc)


@dataclass 
class Reflection:
    """A reflection or lesson learned."""
    content: str
    category: str  # success, failure, insight, strategy
    context: str | None = None
    timestamp: datetime = field(default_factory=now_utc)


# ============================================================================
# TaskBoard Configuration
# ============================================================================

@dataclass
class TaskBoardConfig:
    """Configuration for TaskBoard behavior.
    
    Args:
        auto_verify: Require agent to verify completion before finishing.
        max_tasks: Maximum number of tasks to track.
        max_notes: Maximum notes to retain.
        track_errors: Track errors for learning.
        include_instructions: Add task management instructions to system prompt.
    """
    auto_verify: bool = True
    max_tasks: int = 50
    max_notes: int = 30
    track_errors: bool = True
    include_instructions: bool = True


# ============================================================================
# TaskBoard
# ============================================================================

class TaskBoard:
    """
    Task tracking and working memory for an agent.
    
    Provides structured storage for:
    - Tasks with status tracking
    - Notes and observations
    - Error history for self-correction
    - Learned patterns (Reflexion-style memory)
    
    Example:
        ```python
        board = TaskBoard()
        
        # Add tasks
        board.add_task("Search for tutorials")
        board.add_task("Summarize findings")
        
        # Track progress
        board.start_task("Search")
        board.complete_task("Search", result="Found 5 articles")
        
        # Take notes
        board.add_note("Python async is event-loop based")
        
        # Check status
        print(board.summary())
        ```
    """
    
    def __init__(self, config: TaskBoardConfig | None = None) -> None:
        """Initialize TaskBoard.
        
        Args:
            config: Configuration options.
        """
        self.config = config or TaskBoardConfig()
        
        self._tasks: dict[str, Task] = {}
        self._notes: list[Note] = []
        self._errors: list[ErrorRecord] = []
        self._patterns: dict[str, LearnedPattern] = {}
        self._reflections: list[Reflection] = []
        self._goal: str | None = None
    
    # ========================================
    # Task Management
    # ========================================
    
    def add_task(self, description: str) -> str:
        """Add a task.
        
        Args:
            description: What needs to be done.
            
        Returns:
            Task ID.
        """
        task_id = generate_id("task")
        self._tasks[task_id] = Task(id=task_id, description=description)
        
        # Trim if over limit
        if len(self._tasks) > self.config.max_tasks:
            # Remove oldest completed tasks first
            done = [t for t in self._tasks.values() if t.status == TaskStatus.DONE]
            if done:
                oldest = min(done, key=lambda t: t.completed_at or t.created_at)
                del self._tasks[oldest.id]
        
        return task_id
    
    def add_tasks(self, descriptions: list[str]) -> list[str]:
        """Add multiple tasks.
        
        Args:
            descriptions: List of task descriptions.
            
        Returns:
            List of task IDs.
        """
        return [self.add_task(d) for d in descriptions]
    
    def find_task(self, query: str) -> Task | None:
        """Find a task by ID or description.
        
        Args:
            query: Task ID or partial description match.
            
        Returns:
            Task if found, None otherwise.
        """
        # Exact ID match
        if query in self._tasks:
            return self._tasks[query]
        
        # Partial description match (case-insensitive)
        query_lower = query.lower()
        for task in self._tasks.values():
            if query_lower in task.description.lower():
                return task
        
        return None
    
    def start_task(self, query: str) -> bool:
        """Mark a task as in progress.
        
        Args:
            query: Task ID or description.
            
        Returns:
            True if found and updated.
        """
        task = self.find_task(query)
        if task:
            task.start()
            return True
        return False
    
    def complete_task(self, query: str, result: str | None = None) -> bool:
        """Mark a task as done.
        
        Args:
            query: Task ID or description.
            result: Optional result/output.
            
        Returns:
            True if found and updated.
        """
        task = self.find_task(query)
        if task:
            task.complete(result)
            return True
        return False
    
    def fail_task(self, query: str, error: str) -> bool:
        """Mark a task as failed.
        
        Args:
            query: Task ID or description.
            error: Error message.
            
        Returns:
            True if found and updated.
        """
        task = self.find_task(query)
        if task:
            task.fail(error)
            return True
        return False
    
    def skip_task(self, query: str) -> bool:
        """Mark a task as skipped.
        
        Args:
            query: Task ID or description.
            
        Returns:
            True if found and updated.
        """
        task = self.find_task(query)
        if task:
            task.skip()
            return True
        return False
    
    def get_pending(self) -> list[Task]:
        """Get pending/in-progress tasks."""
        return [
            t for t in self._tasks.values() 
            if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
        ]
    
    def get_done(self) -> list[Task]:
        """Get completed tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.DONE]
    
    def get_failed(self) -> list[Task]:
        """Get failed tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.FAILED]
    
    def get_all_tasks(self) -> list[Task]:
        """Get all tasks."""
        return list(self._tasks.values())
    
    def clear_tasks(self) -> None:
        """Clear all tasks."""
        self._tasks.clear()
    
    def is_complete(self) -> bool:
        """Check if all tasks are done (none pending)."""
        return len(self.get_pending()) == 0
    
    # ========================================
    # Notes
    # ========================================
    
    def add_note(self, content: str, category: str = "observation") -> str:
        """Add a note.
        
        Args:
            content: Note content.
            category: observation, insight, or question.
            
        Returns:
            Note ID.
        """
        note_id = generate_id("note")
        self._notes.append(Note(id=note_id, content=content, category=category))
        
        # Trim if over limit
        if len(self._notes) > self.config.max_notes:
            self._notes = self._notes[-self.config.max_notes:]
        
        return note_id
    
    def get_notes(self, category: str | None = None, limit: int | None = None) -> list[Note]:
        """Get notes.
        
        Args:
            category: Filter by category.
            limit: Maximum to return.
        """
        notes = self._notes
        if category:
            notes = [n for n in notes if n.category == category]
        if limit:
            notes = notes[-limit:]
        return notes
    
    # ========================================
    # Goal
    # ========================================
    
    def set_goal(self, goal: str) -> None:
        """Set the current goal."""
        self._goal = goal
    
    def get_goal(self) -> str | None:
        """Get the current goal."""
        return self._goal
    
    # ========================================
    # Error Tracking & Learning
    # ========================================
    
    def record_error(
        self,
        tool_name: str,
        error: Exception,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Record an error for learning.
        
        Args:
            tool_name: Tool that failed.
            error: The exception.
            args: Arguments that caused failure.
        """
        if not self.config.track_errors:
            return
        
        self._errors.append(ErrorRecord(
            tool_name=tool_name,
            error_type=type(error).__name__,
            error_message=str(error),
            args=args or {},
        ))
        
        # Keep recent errors only
        if len(self._errors) > 20:
            self._errors = self._errors[-20:]
        
        # Learn pattern
        self._learn_pattern(tool_name, str(error))
    
    def _learn_pattern(self, tool_name: str, error_message: str) -> None:
        """Learn a failure pattern."""
        # Generalize error message
        pattern = self._generalize_error(error_message)
        key = f"{tool_name}:{pattern}"
        
        if key in self._patterns:
            self._patterns[key].count += 1
            self._patterns[key].last_seen = now_utc()
        else:
            self._patterns[key] = LearnedPattern(
                tool_name=tool_name,
                pattern=pattern,
            )
    
    def _generalize_error(self, message: str) -> str:
        """Generalize an error message to a pattern."""
        pattern = message
        pattern = re.sub(r"'[^']*'", "'...'", pattern)
        pattern = re.sub(r'"[^"]*"', '"..."', pattern)
        pattern = re.sub(r'\b\d+\b', '#', pattern)
        pattern = re.sub(r'/[\w/.-]+', '<path>', pattern)
        return pattern[:100] if len(pattern) > 100 else pattern
    
    def get_known_fix(self, tool_name: str, error_message: str) -> str | None:
        """Check if we have a known fix for this error."""
        pattern = self._generalize_error(error_message)
        key = f"{tool_name}:{pattern}"
        
        if key in self._patterns:
            return self._patterns[key].fix
        return None
    
    def record_fix(self, tool_name: str, error_message: str, fix: str) -> None:
        """Record a successful fix for an error pattern."""
        pattern = self._generalize_error(error_message)
        key = f"{tool_name}:{pattern}"
        
        if key in self._patterns:
            self._patterns[key].fix = fix
    
    # ========================================
    # Reflections
    # ========================================
    
    def add_reflection(
        self, 
        content: str, 
        category: str = "insight",
        context: str | None = None,
    ) -> None:
        """Add a reflection or lesson learned.
        
        Args:
            content: The insight or lesson.
            category: success, failure, insight, or strategy.
            context: Optional context.
        """
        self._reflections.append(Reflection(
            content=content,
            category=category,
            context=context or self._goal,
        ))
        
        # Keep recent reflections
        if len(self._reflections) > 20:
            self._reflections = self._reflections[-20:]
    
    def get_reflections(self, category: str | None = None) -> list[Reflection]:
        """Get reflections."""
        if category:
            return [r for r in self._reflections if r.category == category]
        return self._reflections.copy()
    
    # ========================================
    # Summary & Context
    # ========================================
    
    def summary(self) -> str:
        """Get a summary of the taskboard state.
        
        Returns:
            Formatted summary string.
        """
        lines = []
        
        # Goal
        if self._goal:
            lines.append(f"🎯 Goal: {self._goal}")
            lines.append("")
        
        # Tasks
        tasks = self.get_all_tasks()
        if tasks:
            done = len(self.get_done())
            total = len(tasks)
            pct = (done / total * 100) if total > 0 else 0
            lines.append(f"📋 Tasks: {done}/{total} complete ({pct:.0f}%)")
            for task in tasks:
                lines.append(f"  {task}")
            lines.append("")
        
        # Notes
        notes = self.get_notes(limit=5)
        if notes:
            lines.append(f"📝 Notes ({len(self._notes)} total):")
            for note in notes:
                icon = {"observation": "•", "insight": "💡", "question": "❓"}.get(note.category, "•")
                lines.append(f"  {icon} {note.content}")
        
        return "\n".join(lines) if lines else "Empty taskboard"
    
    def get_context(self) -> str:
        """Get context for LLM prompt.
        
        Returns:
            Formatted context for inclusion in prompts.
        """
        sections = []
        
        # Goal
        if self._goal:
            sections.append(f"**Goal**: {self._goal}")
        
        # Tasks
        pending = self.get_pending()
        done = self.get_done()
        if pending or done:
            lines = ["**Progress**:"]
            for t in done[-3:]:
                result_str = f" → {t.result[:30]}..." if t.result and len(t.result) > 30 else f" → {t.result}" if t.result else ""
                lines.append(f"  ✓ {t.description}{result_str}")
            for t in pending:
                icon = "→" if t.status == TaskStatus.IN_PROGRESS else "○"
                lines.append(f"  {icon} {t.description}")
            sections.append("\n".join(lines))
        
        # Recent notes
        notes = self.get_notes(limit=3)
        if notes:
            lines = ["**Notes**:"]
            for n in notes:
                lines.append(f"  - {n.content}")
            sections.append("\n".join(lines))
        
        # Reflections
        reflections = self._reflections[-3:] if self._reflections else []
        if reflections:
            lines = ["**Learned**:"]
            for r in reflections:
                icon = {"success": "✓", "failure": "✗", "insight": "💡", "strategy": "📋"}.get(r.category, "•")
                lines.append(f"  {icon} {r.content}")
            sections.append("\n".join(lines))
        
        return "\n\n".join(sections) if sections else ""
    
    def verify_completion(self) -> dict[str, Any]:
        """Verify that work is complete.
        
        Returns:
            Dict with verification results.
        """
        pending = self.get_pending()
        failed = self.get_failed()
        done = self.get_done()
        
        issues = []
        
        if pending:
            issues.append(f"{len(pending)} task(s) still pending")
        
        if failed:
            issues.append(f"{len(failed)} task(s) failed")
        
        # Check for tasks without results
        no_result = [t for t in done if not t.result]
        if no_result:
            issues.append(f"{len(no_result)} completed task(s) have no result documented")
        
        return {
            "complete": len(pending) == 0 and len(failed) == 0,
            "done_count": len(done),
            "pending_count": len(pending),
            "failed_count": len(failed),
            "issues": issues,
        }
    
    # ========================================
    # Serialization
    # ========================================
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "goal": self._goal,
            "tasks": [t.to_dict() for t in self._tasks.values()],
            "notes": [n.to_dict() for n in self._notes],
            "reflections": [{"content": r.content, "category": r.category} for r in self._reflections],
        }
    
    def clear(self) -> None:
        """Clear tasks and notes (keep learned patterns)."""
        self._tasks.clear()
        self._notes.clear()
        self._goal = None
    
    def clear_all(self) -> None:
        """Clear everything including learned patterns."""
        self.clear()
        self._errors.clear()
        self._patterns.clear()
        self._reflections.clear()


# ============================================================================
# TaskBoard Tools
# ============================================================================

# System instructions for agents with taskboard enabled
TASKBOARD_INSTRUCTIONS = """
## Task Management

You have a taskboard to track your work. Use it like a checklist:

1. **Plan First**: For multi-step tasks, use `plan_tasks()` to create a checklist.
2. **Track Progress**: Use `update_task()` to mark tasks as started/done/failed.
3. **Take Notes**: Use `add_note()` for important observations.
4. **Check Status**: Use `check_progress()` periodically.
5. **Verify Completion**: Use `verify_done()` before finishing to ensure nothing was missed.

Be thorough - don't rush. Quality over speed.
"""


def create_taskboard_tools(board: TaskBoard) -> list["BaseTool"]:
    """Create tools for taskboard interaction.
    
    These tools let the agent manage its own work.
    Uses flexible parameter handling to work with various LLM outputs.
    
    Args:
        board: TaskBoard instance to operate on.
        
    Returns:
        List of tools for task management.
    """
    from agenticflow.tools.base import BaseTool
    
    def _plan_tasks_impl(**kwargs: Any) -> str:
        """Create a task list for multi-step work."""
        # Handle various LLM parameter naming
        task_list = (
            kwargs.get("tasks")
            or kwargs.get("task_list")
            or kwargs.get("items")
            or []
        )
        if not task_list:
            return "❌ No tasks provided. Pass a list of task descriptions."
        
        goal_text = kwargs.get("goal") or kwargs.get("objective") or kwargs.get("target")
        
        if goal_text:
            board.set_goal(goal_text)
        
        board.clear_tasks()
        board.add_tasks(task_list)
        
        lines = [f"✅ Created {len(task_list)} tasks:"]
        for i, desc in enumerate(task_list, 1):
            lines.append(f"  {i}. {desc}")
        
        if goal_text:
            lines.append(f"\n🎯 Goal: {goal_text}")
        
        return "\n".join(lines)
    
    plan_tasks = BaseTool(
        name="plan_tasks",
        description="Create a task list for multi-step work. Use at the start of complex tasks to break them into steps.",
        func=_plan_tasks_impl,
        args_schema={
            "tasks": {"type": "array", "items": {"type": "string"}, "description": "List of task descriptions"},
            "goal": {"type": "string", "description": "Optional overall goal"},
        },
    )
    
    def _update_task_impl(**kwargs: Any) -> str:
        """Mark a task as started, done, or failed."""
        # Extract task identifier from various possible parameter names
        task_id = (
            kwargs.get("task_name")
            or kwargs.get("task_id")
            or kwargs.get("task")
            or kwargs.get("name")
            or kwargs.get("id")
            or kwargs.get("task_number")
            or kwargs.get("number")
        )
        
        if task_id is None:
            return "❌ No task specified. Provide task_name (e.g., 'Search for async')."
        
        # Convert numeric task_id to task description by index
        if isinstance(task_id, int | float):
            idx = int(task_id) - 1  # 1-indexed to 0-indexed
            all_tasks = board.get_all_tasks()
            if 0 <= idx < len(all_tasks):
                task_id = all_tasks[idx].description
            else:
                return f"❌ Task #{int(task_id)} not found. Tasks are numbered 1-{len(all_tasks)}."
        
        # Extract status from various parameter names
        status = (
            kwargs.get("status")
            or kwargs.get("new_status")
            or kwargs.get("new_state")
            or kwargs.get("state")
            or "done"  # default to done if not specified
        )
        
        # Extract result
        result = kwargs.get("result") or kwargs.get("message") or kwargs.get("note")
        
        found = board.find_task(str(task_id))
        if not found:
            return f"❌ Task not found: {task_id}"
        
        status_lower = str(status).lower()
        
        if status_lower in ("started", "start", "in_progress", "working", "in-progress"):
            board.start_task(str(task_id))
            return f"→ Started: {found.description}"
        
        elif status_lower in ("done", "complete", "completed", "finished"):
            board.complete_task(str(task_id), result)
            remaining = len(board.get_pending())
            msg = f"✓ Completed: {found.description}"
            if result:
                msg += f"\n  Result: {result}"
            if remaining > 0:
                msg += f"\n  ({remaining} task(s) remaining)"
            else:
                msg += "\n  🎉 All tasks complete!"
            return msg
        
        elif status_lower in ("failed", "fail", "error"):
            board.fail_task(str(task_id), result or "Unknown error")
            return f"✗ Failed: {found.description}\n  Error: {result or 'Unknown error'}"
        
        elif status_lower in ("skip", "skipped"):
            board.skip_task(str(task_id))
            return f"○ Skipped: {found.description}"
        
        else:
            return f"❌ Unknown status: {status}. Use: started, done, failed, or skip."
    
    update_task = BaseTool(
        name="update_task",
        description="Mark a task as started, done, or failed. Use task name or number to identify the task.",
        func=_update_task_impl,
        args_schema={
            "task_name": {"type": "string", "description": "Task name/description (partial match OK) or task number (1-based)"},
            "status": {"type": "string", "description": "New status: 'started', 'done', 'failed', or 'skip'"},
            "result": {"type": "string", "description": "Optional result message or error description"},
        },
    )
    
    def _add_note_impl(**kwargs: Any) -> str:
        """Record a note or observation."""
        text = (
            kwargs.get("note")
            or kwargs.get("text")
            or kwargs.get("content")
            or kwargs.get("message")
        )
        if not text:
            return "❌ No note content provided."
        
        cat = kwargs.get("category") or kwargs.get("type") or "observation"
        
        board.add_note(text, cat)
        icon = {"observation": "📝", "insight": "💡", "question": "❓"}.get(cat, "📝")
        return f"{icon} Noted: {text}"
    
    add_note = BaseTool(
        name="add_note",
        description="Record a note or observation during work.",
        func=_add_note_impl,
        args_schema={
            "note": {"type": "string", "description": "The note content"},
            "category": {"type": "string", "description": "Category: observation, insight, or question"},
        },
    )
    
    def _check_progress_impl(**kwargs: Any) -> str:
        """Check current task progress."""
        tasks = board.get_all_tasks()
        if not tasks:
            return "📋 No tasks. Use plan_tasks() to create a task list."
        
        done = board.get_done()
        pending = board.get_pending()
        failed = board.get_failed()
        
        total = len(tasks)
        pct = (len(done) / total * 100) if total > 0 else 0
        
        lines = [
            "📋 **Task Progress**",
            f"Progress: {len(done)}/{total} complete ({pct:.0f}%)",
        ]
        
        if failed:
            lines.append(f"⚠️ {len(failed)} failed")
        
        lines.append("")
        
        for t in tasks:
            lines.append(f"  {t}")
        
        if pending:
            lines.append(f"\n👉 Next: {pending[0].description}")
        elif not failed:
            lines.append("\n🎉 All tasks complete! Use verify_done() to confirm.")
        
        return "\n".join(lines)
    
    check_progress = BaseTool(
        name="check_progress",
        description="Check current task progress and see remaining work.",
        func=_check_progress_impl,
        args_schema={},
    )
    
    def _verify_done_impl(**kwargs: Any) -> str:
        """Verify all work is complete before finishing."""
        result = board.verify_completion()
        
        lines = ["🔍 **Completion Check**", ""]
        
        if result["issues"]:
            lines.append("❌ **Issues Found:**")
            for issue in result["issues"]:
                lines.append(f"  ⚠️ {issue}")
            lines.append("\nPlease address these before completing.")
        else:
            lines.append("✅ **All checks passed!**")
            lines.append(f"  - {result['done_count']} task(s) completed")
            lines.append("  - All tasks have documented results")
            lines.append("\nYou can now provide the final answer.")
        
        return "\n".join(lines)
    
    verify_done = BaseTool(
        name="verify_done",
        description="Verify all work is complete before finishing. Use this to confirm nothing was missed.",
        func=_verify_done_impl,
        args_schema={},
    )
    
    return [plan_tasks, update_task, add_note, check_progress, verify_done]
