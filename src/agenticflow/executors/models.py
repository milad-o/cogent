"""
Data models for execution planning.

These are the core data structures used across all execution strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A planned tool call with dependencies.
    
    Represents a single tool invocation in an execution plan,
    tracking its dependencies, status, and result.
    
    Attributes:
        id: Unique identifier for this call (e.g., "call_0")
        tool_name: Name of the tool to invoke
        args: Arguments to pass to the tool
        depends_on: IDs of tool calls this depends on
        result: Result from execution (set after completion)
        error: Error message if failed
        status: Current status (pending, running, completed, failed)
        duration_ms: Execution time in milliseconds
    """
    
    id: str
    tool_name: str
    args: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    status: str = "pending"  # pending, running, completed, failed
    duration_ms: float = 0.0

    def is_ready(self, completed: set[str]) -> bool:
        """Check if all dependencies are satisfied.
        
        Args:
            completed: Set of completed call IDs.
            
        Returns:
            True if all dependencies are in the completed set.
        """
        return all(dep in completed for dep in self.depends_on)


@dataclass
class ExecutionPlan:
    """A plan of tool calls with dependencies.
    
    Represents a complete execution plan as a DAG of tool calls.
    Supports parallel execution of independent calls.
    
    Example:
        plan = ExecutionPlan()
        id1 = plan.add_call("search", {"query": "topic A"})
        id2 = plan.add_call("search", {"query": "topic B"})
        id3 = plan.add_call("combine", {"a": "$call_0", "b": "$call_1"}, 
                           depends_on=[id1, id2])
        
        # Get execution waves
        waves = plan.get_execution_order()
        # [["call_0", "call_1"], ["call_2"]]
        # First wave runs in parallel, second wave depends on first
    """
    
    calls: list[ToolCall] = field(default_factory=list)
    final_answer_template: str = ""
    
    def add_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        depends_on: list[str] | None = None,
    ) -> str:
        """Add a tool call to the plan.
        
        Args:
            tool_name: Name of tool to call.
            args: Arguments for the tool.
            depends_on: IDs of calls this depends on.
            
        Returns:
            ID of the new call (e.g., "call_0").
        """
        call_id = f"call_{len(self.calls)}"
        self.calls.append(ToolCall(
            id=call_id,
            tool_name=tool_name,
            args=args,
            depends_on=depends_on or [],
        ))
        return call_id
    
    def get_ready_calls(self, completed: set[str]) -> list[ToolCall]:
        """Get all calls ready to execute (dependencies satisfied).
        
        Args:
            completed: Set of completed call IDs.
            
        Returns:
            List of calls that can be executed now.
        """
        return [
            call for call in self.calls
            if call.status == "pending" and call.is_ready(completed)
        ]
    
    def get_execution_order(self) -> list[list[str]]:
        """Get execution waves (calls that can run in parallel).
        
        Returns:
            List of waves, where each wave is a list of call IDs
            that can execute in parallel.
            
        Example:
            # For a plan where call_2 depends on call_0 and call_1:
            [["call_0", "call_1"], ["call_2"]]
        """
        waves: list[list[str]] = []
        completed: set[str] = set()
        remaining = {c.id for c in self.calls}
        
        while remaining:
            wave = [
                c.id for c in self.calls
                if c.id in remaining and c.is_ready(completed)
            ]
            if not wave:
                # Circular dependency or error
                break
            waves.append(wave)
            completed.update(wave)
            remaining -= set(wave)
        
        return waves
    
    def __len__(self) -> int:
        """Number of calls in the plan."""
        return len(self.calls)
    
    def __bool__(self) -> bool:
        """True if plan has any calls."""
        return len(self.calls) > 0
