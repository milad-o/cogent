"""
Scratchpad - Working memory and todo list for agents.

Provides:
- **Todo List**: Track tasks with status (pending/done/failed)
- **Notes**: Intermediate observations and reasoning
- **Plan**: Current execution plan
- **Error Log**: Recent errors for self-correction context

This gives agents internal state to track complex multi-step work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agenticflow.core.utils import generate_id, now_utc


class TodoStatus(Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TodoItem:
    """A single todo item."""
    id: str
    description: str
    status: TodoStatus = TodoStatus.PENDING
    created_at: datetime = field(default_factory=now_utc)
    completed_at: datetime | None = None
    result: str | None = None
    error: str | None = None
    
    def mark_done(self, result: str | None = None) -> None:
        """Mark as done with optional result."""
        self.status = TodoStatus.DONE
        self.completed_at = now_utc()
        self.result = result
    
    def mark_failed(self, error: str) -> None:
        """Mark as failed with error."""
        self.status = TodoStatus.FAILED
        self.completed_at = now_utc()
        self.error = error
    
    def mark_in_progress(self) -> None:
        """Mark as in progress."""
        self.status = TodoStatus.IN_PROGRESS
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }
    
    def __str__(self) -> str:
        status_icon = {
            TodoStatus.PENDING: "○",
            TodoStatus.IN_PROGRESS: "◐",
            TodoStatus.DONE: "●",
            TodoStatus.FAILED: "✗",
            TodoStatus.SKIPPED: "◌",
        }
        icon = status_icon.get(self.status, "?")
        return f"[{icon}] {self.description}"


@dataclass
class Note:
    """A scratchpad note."""
    id: str
    content: str
    category: str = "observation"  # observation, insight, question, plan
    created_at: datetime = field(default_factory=now_utc)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ErrorRecord:
    """Record of an error for self-correction context."""
    tool_name: str
    args: dict[str, Any]
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=now_utc)
    correction_attempted: bool = False
    correction_successful: bool = False
    correction_strategy: str | None = None  # What was tried to fix it
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "args": self.args,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "correction_attempted": self.correction_attempted,
            "correction_successful": self.correction_successful,
            "correction_strategy": self.correction_strategy,
        }


@dataclass
class FailurePattern:
    """
    Learned failure pattern for Reflexion-style memory.
    
    Accumulates knowledge about what fails and why, enabling the agent
    to avoid repeating the same mistakes across attempts.
    """
    tool_name: str
    error_pattern: str  # Generalized pattern (e.g., "missing required arg: x")
    failure_count: int = 1
    successful_fix: str | None = None  # What worked to fix it
    last_seen: datetime = field(default_factory=now_utc)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "pattern": self.error_pattern,
            "count": self.failure_count,
            "fix": self.successful_fix,
        }


@dataclass
class Reflection:
    """
    A reflection entry for episodic memory (Reflexion-style).
    
    Captures high-level insights about what worked or didn't work
    during task execution. Persists across attempts.
    """
    content: str
    reflection_type: str  # "success", "failure", "insight", "strategy"
    task_context: str | None = None  # What task this was about
    timestamp: datetime = field(default_factory=now_utc)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "type": self.reflection_type,
            "context": self.task_context,
        }


class Scratchpad:
    """
    Working memory for an agent with Reflexion-style episodic learning.
    
    Provides structured storage for:
    - Todo list with status tracking
    - Notes and observations
    - Current plan
    - Error history for self-correction
    - **Failure patterns** - learned patterns of what doesn't work
    - **Reflections** - episodic memory that persists across attempts
    
    The failure patterns and reflections implement Reflexion-style memory,
    allowing agents to learn from trial-and-error without weight updates.
    
    Example:
        ```python
        pad = Scratchpad()
        
        # Todo list
        pad.add_todo("Search for Python tutorials")
        pad.add_todo("Summarize findings")
        pad.mark_done("Search for Python tutorials", "Found 5 articles")
        
        # Notes
        pad.note("Python is popular for data science")
        pad.note("Consider async tutorials", category="question")
        
        # Reflexion-style learning
        pad.learn_failure_pattern("search", "timeout", fix="use smaller query")
        pad.add_reflection("Breaking queries into chunks works better", "strategy")
        
        # Get context for LLM
        context = pad.get_context()
        ```
    """
    
    def __init__(
        self, 
        max_notes: int = 50, 
        max_errors: int = 10,
        max_reflections: int = 20,
        max_patterns: int = 30,
    ) -> None:
        """Initialize scratchpad.
        
        Args:
            max_notes: Maximum notes to retain.
            max_errors: Maximum errors to retain for context.
            max_reflections: Maximum reflections to retain.
            max_patterns: Maximum failure patterns to track.
        """
        self._todos: dict[str, TodoItem] = {}
        self._notes: list[Note] = []
        self._errors: list[ErrorRecord] = []
        self._plan: str | None = None
        self._goal: str | None = None
        
        # Reflexion-style episodic memory
        self._failure_patterns: dict[str, FailurePattern] = {}
        self._reflections: list[Reflection] = []
        
        self.max_notes = max_notes
        self.max_errors = max_errors
        self.max_reflections = max_reflections
        self.max_patterns = max_patterns
    
    # ========================================
    # Todo List
    # ========================================
    
    def add_todo(self, description: str) -> str:
        """Add a todo item.
        
        Args:
            description: What needs to be done.
            
        Returns:
            ID of the created todo.
        """
        todo_id = generate_id("todo")
        self._todos[todo_id] = TodoItem(id=todo_id, description=description)
        return todo_id
    
    def add_todos(self, descriptions: list[str]) -> list[str]:
        """Add multiple todos at once.
        
        Args:
            descriptions: List of todo descriptions.
            
        Returns:
            List of created todo IDs.
        """
        return [self.add_todo(d) for d in descriptions]
    
    def mark_done(
        self, 
        description_or_id: str, 
        result: str | None = None,
    ) -> bool:
        """Mark a todo as done.
        
        Args:
            description_or_id: Todo ID or description (partial match).
            result: Optional result/output.
            
        Returns:
            True if found and marked, False otherwise.
        """
        todo = self._find_todo(description_or_id)
        if todo:
            todo.mark_done(result)
            return True
        return False
    
    def mark_failed(self, description_or_id: str, error: str) -> bool:
        """Mark a todo as failed.
        
        Args:
            description_or_id: Todo ID or description.
            error: Error message.
            
        Returns:
            True if found and marked.
        """
        todo = self._find_todo(description_or_id)
        if todo:
            todo.mark_failed(error)
            return True
        return False
    
    def mark_in_progress(self, description_or_id: str) -> bool:
        """Mark a todo as in progress.
        
        Args:
            description_or_id: Todo ID or description.
            
        Returns:
            True if found and marked.
        """
        todo = self._find_todo(description_or_id)
        if todo:
            todo.mark_in_progress()
            return True
        return False
    
    def _find_todo(self, description_or_id: str) -> TodoItem | None:
        """Find a todo by ID or description match."""
        # Try exact ID match first
        if description_or_id in self._todos:
            return self._todos[description_or_id]
        
        # Try description match (case-insensitive partial)
        desc_lower = description_or_id.lower()
        for todo in self._todos.values():
            if desc_lower in todo.description.lower():
                return todo
        
        return None
    
    def get_pending(self) -> list[TodoItem]:
        """Get all pending/in-progress todos."""
        return [
            t for t in self._todos.values() 
            if t.status in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS)
        ]
    
    def get_done(self) -> list[TodoItem]:
        """Get all completed todos."""
        return [t for t in self._todos.values() if t.status == TodoStatus.DONE]
    
    def get_failed(self) -> list[TodoItem]:
        """Get all failed todos."""
        return [t for t in self._todos.values() if t.status == TodoStatus.FAILED]
    
    def get_all_todos(self) -> list[TodoItem]:
        """Get all todos."""
        return list(self._todos.values())
    
    def clear_todos(self) -> None:
        """Clear all todos."""
        self._todos.clear()
    
    def has_pending(self) -> bool:
        """Check if there are pending todos."""
        return len(self.get_pending()) > 0
    
    def all_done(self) -> bool:
        """Check if all todos are done (none pending/in-progress)."""
        return not self.has_pending()
    
    def todo_summary(self) -> str:
        """Get a summary of todo status."""
        todos = self.get_all_todos()
        if not todos:
            return "No todos."
        
        pending = len(self.get_pending())
        done = len(self.get_done())
        failed = len(self.get_failed())
        
        lines = [f"Todos: {done}/{len(todos)} done"]
        if failed:
            lines.append(f"({failed} failed)")
        
        lines.append("")
        for todo in todos:
            lines.append(str(todo))
        
        return "\n".join(lines)
    
    # ========================================
    # Notes
    # ========================================
    
    def note(self, content: str, category: str = "observation") -> str:
        """Add a note.
        
        Args:
            content: Note content.
            category: Category (observation, insight, question, plan).
            
        Returns:
            Note ID.
        """
        note_id = generate_id("note")
        note = Note(id=note_id, content=content, category=category)
        self._notes.append(note)
        
        # Trim if over limit
        if len(self._notes) > self.max_notes:
            self._notes = self._notes[-self.max_notes:]
        
        return note_id
    
    def observe(self, content: str) -> str:
        """Add an observation note."""
        return self.note(content, "observation")
    
    def insight(self, content: str) -> str:
        """Add an insight note."""
        return self.note(content, "insight")
    
    def question(self, content: str) -> str:
        """Add a question note."""
        return self.note(content, "question")
    
    def get_notes(self, category: str | None = None, limit: int | None = None) -> list[Note]:
        """Get notes, optionally filtered by category.
        
        Args:
            category: Filter by category (None for all).
            limit: Maximum notes to return.
            
        Returns:
            List of notes.
        """
        notes = self._notes
        if category:
            notes = [n for n in notes if n.category == category]
        if limit:
            notes = notes[-limit:]
        return notes
    
    def clear_notes(self) -> None:
        """Clear all notes."""
        self._notes.clear()
    
    # ========================================
    # Plan & Goal
    # ========================================
    
    def set_goal(self, goal: str) -> None:
        """Set the current goal."""
        self._goal = goal
    
    def get_goal(self) -> str | None:
        """Get the current goal."""
        return self._goal
    
    def set_plan(self, plan: str) -> None:
        """Set the current plan."""
        self._plan = plan
    
    def get_plan(self) -> str | None:
        """Get the current plan."""
        return self._plan
    
    # ========================================
    # Error Tracking
    # ========================================
    
    def record_error(
        self,
        tool_name: str,
        args: dict[str, Any],
        error: Exception,
    ) -> None:
        """Record an error for self-correction context.
        
        Args:
            tool_name: Tool that failed.
            args: Arguments that caused failure.
            error: The exception.
        """
        record = ErrorRecord(
            tool_name=tool_name,
            args=args,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        self._errors.append(record)
        
        # Trim if over limit
        if len(self._errors) > self.max_errors:
            self._errors = self._errors[-self.max_errors:]
    
    def get_recent_errors(self, limit: int = 5) -> list[ErrorRecord]:
        """Get recent errors."""
        return self._errors[-limit:]
    
    def get_error_context(self) -> str:
        """Get error context string for LLM."""
        errors = self.get_recent_errors(3)
        if not errors:
            return ""
        
        lines = ["Recent errors:"]
        for err in errors:
            lines.append(f"- {err.tool_name}({err.args}): {err.error_type}: {err.error_message}")
        
        return "\n".join(lines)
    
    def mark_error_corrected(self, tool_name: str, success: bool = True, strategy: str | None = None) -> None:
        """Mark that an error correction was attempted.
        
        Args:
            tool_name: Tool that was corrected.
            success: Whether correction succeeded.
            strategy: What strategy was used for correction.
        """
        for err in reversed(self._errors):
            if err.tool_name == tool_name and not err.correction_attempted:
                err.correction_attempted = True
                err.correction_successful = success
                err.correction_strategy = strategy
                
                # Learn from this for Reflexion memory
                if success and strategy:
                    self.learn_failure_pattern(
                        tool_name, 
                        err.error_message,
                        fix=strategy,
                    )
                break
    
    # ========================================
    # Reflexion-Style Memory
    # ========================================
    
    def learn_failure_pattern(
        self, 
        tool_name: str, 
        error_message: str,
        fix: str | None = None,
    ) -> None:
        """Learn a failure pattern for future avoidance.
        
        This implements Reflexion-style verbal reinforcement learning.
        The agent accumulates knowledge about what fails and how to fix it.
        
        Args:
            tool_name: Tool that failed.
            error_message: The error message (will be generalized).
            fix: What worked to fix it (if known).
        """
        # Generalize the error message to a pattern
        pattern = self._generalize_error(error_message)
        key = f"{tool_name}:{pattern}"
        
        if key in self._failure_patterns:
            # Update existing pattern
            self._failure_patterns[key].failure_count += 1
            self._failure_patterns[key].last_seen = now_utc()
            if fix and not self._failure_patterns[key].successful_fix:
                self._failure_patterns[key].successful_fix = fix
        else:
            # New pattern
            self._failure_patterns[key] = FailurePattern(
                tool_name=tool_name,
                error_pattern=pattern,
                successful_fix=fix,
            )
        
        # Trim if over limit (remove oldest)
        if len(self._failure_patterns) > self.max_patterns:
            oldest_key = min(
                self._failure_patterns.keys(),
                key=lambda k: self._failure_patterns[k].last_seen
            )
            del self._failure_patterns[oldest_key]
    
    def _generalize_error(self, error_message: str) -> str:
        """Generalize an error message to a reusable pattern.
        
        Strips specific values to create a pattern that matches
        similar future errors.
        
        Args:
            error_message: Specific error message.
            
        Returns:
            Generalized pattern.
        """
        import re
        
        # Remove specific values, keep structure
        pattern = error_message
        
        # Replace quoted strings with placeholder
        pattern = re.sub(r"'[^']*'", "'<VALUE>'", pattern)
        pattern = re.sub(r'"[^"]*"', '"<VALUE>"', pattern)
        
        # Replace numbers with placeholder
        pattern = re.sub(r'\b\d+\b', '<NUM>', pattern)
        
        # Replace file paths
        pattern = re.sub(r'/[\w/.-]+', '<PATH>', pattern)
        
        # Truncate if too long
        if len(pattern) > 100:
            pattern = pattern[:100] + "..."
        
        return pattern
    
    def get_failure_patterns(self, tool_name: str | None = None) -> list[FailurePattern]:
        """Get learned failure patterns.
        
        Args:
            tool_name: Filter by tool (None for all).
            
        Returns:
            List of failure patterns.
        """
        patterns = list(self._failure_patterns.values())
        if tool_name:
            patterns = [p for p in patterns if p.tool_name == tool_name]
        return sorted(patterns, key=lambda p: p.failure_count, reverse=True)
    
    def get_known_fix(self, tool_name: str, error_message: str) -> str | None:
        """Check if we have a known fix for this error.
        
        Args:
            tool_name: Tool that failed.
            error_message: The error message.
            
        Returns:
            Known fix if one exists, None otherwise.
        """
        pattern = self._generalize_error(error_message)
        key = f"{tool_name}:{pattern}"
        
        if key in self._failure_patterns:
            return self._failure_patterns[key].successful_fix
        return None
    
    def add_reflection(
        self, 
        content: str, 
        reflection_type: str = "insight",
        task_context: str | None = None,
    ) -> None:
        """Add a reflection to episodic memory.
        
        Reflections capture high-level insights about what worked
        or didn't work. They persist across attempts and inform
        future decision-making.
        
        Args:
            content: The reflection content.
            reflection_type: Type (success, failure, insight, strategy).
            task_context: Optional task context.
        """
        reflection = Reflection(
            content=content,
            reflection_type=reflection_type,
            task_context=task_context or self._goal,
        )
        self._reflections.append(reflection)
        
        # Trim if over limit
        if len(self._reflections) > self.max_reflections:
            self._reflections = self._reflections[-self.max_reflections:]
    
    def get_reflections(
        self, 
        reflection_type: str | None = None,
        limit: int | None = None,
    ) -> list[Reflection]:
        """Get reflections from episodic memory.
        
        Args:
            reflection_type: Filter by type (None for all).
            limit: Maximum to return.
            
        Returns:
            List of reflections.
        """
        reflections = self._reflections
        if reflection_type:
            reflections = [r for r in reflections if r.reflection_type == reflection_type]
        if limit:
            reflections = reflections[-limit:]
        return reflections
    
    def get_reflexion_context(self) -> str:
        """Get Reflexion-style context for LLM prompt.
        
        Returns a summary of learned patterns and reflections
        to help the agent avoid past mistakes.
        
        Returns:
            Formatted context string.
        """
        sections = []
        
        # Failure patterns with known fixes
        patterns_with_fixes = [
            p for p in self._failure_patterns.values() 
            if p.successful_fix
        ]
        if patterns_with_fixes:
            lines = ["**Learned Fixes** (from past failures):"]
            for p in sorted(patterns_with_fixes, key=lambda x: x.failure_count, reverse=True)[:5]:
                lines.append(f"  - {p.tool_name}: {p.error_pattern} → Fix: {p.successful_fix}")
            sections.append("\n".join(lines))
        
        # Common failures to avoid
        common_failures = [
            p for p in self._failure_patterns.values() 
            if p.failure_count >= 2 and not p.successful_fix
        ]
        if common_failures:
            lines = ["**Known Pitfalls** (avoid these):"]
            for p in sorted(common_failures, key=lambda x: x.failure_count, reverse=True)[:3]:
                lines.append(f"  - {p.tool_name}: {p.error_pattern} (failed {p.failure_count}x)")
            sections.append("\n".join(lines))
        
        # Recent reflections
        reflections = self.get_reflections(limit=5)
        if reflections:
            lines = ["**Reflections**:"]
            for r in reflections:
                icon = {"success": "✓", "failure": "✗", "insight": "💡", "strategy": "📋"}.get(r.reflection_type, "•")
                lines.append(f"  {icon} {r.content}")
            sections.append("\n".join(lines))
        
        return "\n\n".join(sections) if sections else ""
    
    # ========================================
    # Context Generation
    # ========================================
    
    def get_context(self, include_all: bool = False) -> str:
        """Get scratchpad context for LLM prompt.
        
        Args:
            include_all: Include all details (vs summary).
            
        Returns:
            Formatted context string.
        """
        sections = []
        
        # Goal
        if self._goal:
            sections.append(f"**Goal**: {self._goal}")
        
        # Plan
        if self._plan:
            sections.append(f"**Plan**: {self._plan}")
        
        # Todos
        pending = self.get_pending()
        done = self.get_done()
        if pending or done:
            todo_lines = ["**Progress**:"]
            for t in done[-5:]:  # Last 5 completed
                todo_lines.append(f"  ✓ {t.description}")
            for t in pending:
                icon = "→" if t.status == TodoStatus.IN_PROGRESS else "○"
                todo_lines.append(f"  {icon} {t.description}")
            sections.append("\n".join(todo_lines))
        
        # Notes (recent)
        notes = self.get_notes(limit=5)
        if notes:
            note_lines = ["**Notes**:"]
            for n in notes:
                note_lines.append(f"  - {n.content}")
            sections.append("\n".join(note_lines))
        
        # Errors
        error_ctx = self.get_error_context()
        if error_ctx:
            sections.append(error_ctx)
        
        # Reflexion context (learned patterns and reflections)
        reflexion_ctx = self.get_reflexion_context()
        if reflexion_ctx:
            sections.append(reflexion_ctx)
        
        return "\n\n".join(sections) if sections else ""
    
    def get_completion_checklist(self) -> dict[str, Any]:
        """Get checklist for completion verification.
        
        Returns:
            Dict with completion status info.
        """
        pending = self.get_pending()
        failed = self.get_failed()
        done = self.get_done()
        uncorrected_errors = [e for e in self._errors if not e.correction_attempted]
        
        return {
            "all_todos_done": len(pending) == 0,
            "pending_count": len(pending),
            "done_count": len(done),
            "failed_count": len(failed),
            "has_uncorrected_errors": len(uncorrected_errors) > 0,
            "pending_todos": [t.description for t in pending],
            "failed_todos": [t.description for t in failed],
            "ready_to_complete": len(pending) == 0 and len(uncorrected_errors) == 0,
        }
    
    # ========================================
    # Serialization
    # ========================================
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "goal": self._goal,
            "plan": self._plan,
            "todos": [t.to_dict() for t in self._todos.values()],
            "notes": [n.to_dict() for n in self._notes],
            "errors": [e.to_dict() for e in self._errors],
            "failure_patterns": [p.to_dict() for p in self._failure_patterns.values()],
            "reflections": [r.to_dict() for r in self._reflections],
        }
    
    def clear(self) -> None:
        """Clear all scratchpad state except learned patterns."""
        self._todos.clear()
        self._notes.clear()
        self._errors.clear()
        self._plan = None
        self._goal = None
        # Note: We keep failure_patterns and reflections for Reflexion-style learning
    
    def clear_all(self) -> None:
        """Clear ALL scratchpad state including learned patterns."""
        self.clear()
        self._failure_patterns.clear()
        self._reflections.clear()
