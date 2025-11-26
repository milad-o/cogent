"""
Base executor class and common types.

All execution strategies inherit from BaseExecutor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.observability.progress import ProgressTracker


class ExecutionStrategy(Enum):
    """Available execution strategies.
    
    Choose based on task complexity:
    - REACT: Simple tasks, sequential execution
    - PLAN_EXECUTE: Clear structure, plan then execute
    - DAG: Complex tasks, maximize parallelism (RECOMMENDED)
    - ADAPTIVE: Let the system choose
    """
    
    REACT = "react"  # Think-Act-Observe loop
    PLAN_EXECUTE = "plan_execute"  # Plan then execute
    DAG = "dag"  # Dependency graph with parallel execution
    ADAPTIVE = "adaptive"  # Auto-select based on task


class BaseExecutor(ABC):
    """Base class for all execution strategies.
    
    Provides common infrastructure for tool tracking and step emission.
    Subclasses implement the specific execution pattern.
    
    Attributes:
        agent: The agent to execute with.
        max_iterations: Maximum iterations before stopping.
        on_step: Optional callback for step events.
        tracker: Optional progress tracker for observability.
    """
    
    def __init__(self, agent: Agent) -> None:
        """Initialize executor with an agent.
        
        Args:
            agent: The agent that will think and act.
        """
        self.agent = agent
        self.max_iterations: int = 10
        self.on_step: Callable[[str, Any], None] | None = None
        self.tracker: ProgressTracker | None = None
    
    @abstractmethod
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a task and return the result.
        
        Args:
            task: The task description.
            context: Optional context dictionary.
            
        Returns:
            The result of executing the task.
        """
        ...
    
    def _emit_step(self, step_type: str, data: Any) -> None:
        """Emit a step event if callback is set.
        
        Args:
            step_type: Type of step (e.g., "think", "act", "planning").
            data: Step data to emit.
        """
        if self.on_step:
            self.on_step(step_type, data)
    
    def _track_tool_call(self, tool: str, args: dict[str, Any]) -> None:
        """Track tool call via progress tracker if available.
        
        Args:
            tool: Tool name.
            args: Tool arguments.
        """
        if self.tracker:
            self.tracker.tool_call(tool, args, agent=self.agent.name)
    
    def _track_tool_result(self, tool: str, result: Any, duration_ms: float = 0) -> None:
        """Track tool result via progress tracker if available.
        
        Args:
            tool: Tool name.
            result: Tool result.
            duration_ms: Execution duration.
        """
        if self.tracker:
            self.tracker.tool_result(tool, str(result)[:200], duration_ms=duration_ms)
    
    def _track_tool_error(self, tool: str, error: str) -> None:
        """Track tool error via progress tracker if available.
        
        Args:
            tool: Tool name.
            error: Error message.
        """
        if self.tracker:
            self.tracker.tool_error(tool, error)
