"""
Base executor class and common types.

All execution strategies inherit from BaseExecutor.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.observability.progress import ProgressTracker


class ExecutionStrategy(Enum):
    """Available execution strategies.
    
    Choose based on task complexity:
    - NATIVE: High-performance parallel execution (DEFAULT)
    - SEQUENTIAL: Sequential tool execution for ordered tasks
    - TREE_SEARCH: LATS-style exploration with backtracking (BEST ACCURACY)
    """
    
    NATIVE = "native"  # Default - parallel tool execution
    SEQUENTIAL = "sequential"  # Sequential tool execution
    TREE_SEARCH = "tree_search"  # LATS Monte Carlo tree search


@dataclass
class CompletionCheck:
    """Result of completion verification.
    
    Attributes:
        is_complete: Whether the task is fully complete.
        confidence: Confidence score (0.0-1.0).
        missing: List of missing elements.
        summary: Brief summary of completion status.
    """
    
    is_complete: bool
    confidence: float
    missing: list[str] = field(default_factory=list)
    summary: str = ""


class BaseExecutor(ABC):
    """Base class for all execution strategies.
    
    Provides common infrastructure for tool tracking, step emission,
    and completion verification.
    
    Subclasses implement the specific execution pattern.
    
    Attributes:
        agent: The agent to execute with.
        max_iterations: Maximum iterations before stopping.
        on_step: Optional callback for step events.
        tracker: Optional progress tracker for observability.
        verify_completion: Whether to verify completion before returning.
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
        self.verify_completion: bool = False  # Enable for strict verification
    
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
    
    async def _verify_completion(
        self,
        task: str,
        result: str,
        context: dict[str, Any] | None = None,
    ) -> CompletionCheck:
        """Verify that the task is truly complete.
        
        Asks the LLM to verify the result addresses the original task.
        
        Args:
            task: Original task description.
            result: The proposed final result.
            context: Optional execution context.
            
        Returns:
            CompletionCheck with verification results.
        """
        self._emit_step("verifying_completion", {"task": task[:100]})
        
        verification_prompt = f"""Verify if this result fully addresses the original task.

ORIGINAL TASK:
{task}

PROPOSED RESULT:
{result[:2000]}

Analyze:
1. Does the result directly answer/address the task?
2. Are there any missing elements or incomplete parts?
3. Is the result accurate and well-formed?

Respond with ONLY JSON:
{{
    "is_complete": true/false,
    "confidence": 0.0-1.0,
    "missing": ["list", "of", "missing", "elements"],
    "summary": "Brief explanation"
}}"""

        try:
            response = await self.agent.think(
                verification_prompt,
                include_tools=False,
                system_prompt_override="You are a task completion verifier. Analyze results objectively and output JSON only.",
            )
            
            # Parse the verification response
            import re
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                return CompletionCheck(
                    is_complete=data.get("is_complete", True),
                    confidence=data.get("confidence", 0.8),
                    missing=data.get("missing", []),
                    summary=data.get("summary", ""),
                )
        except Exception:
            pass
        
        # Default to complete if verification fails
        return CompletionCheck(is_complete=True, confidence=0.5, summary="Verification inconclusive")
    
    async def _address_missing_elements(
        self,
        task: str,
        result: str,
        missing: list[str],
    ) -> str:
        """Attempt to address missing elements in the result.
        
        Args:
            task: Original task.
            result: Current result.
            missing: List of missing elements.
            
        Returns:
            Enhanced result addressing missing elements.
        """
        self._emit_step("addressing_missing", {"missing": missing})
        
        enhancement_prompt = f"""The result is incomplete. Address the missing elements.

ORIGINAL TASK:
{task}

CURRENT RESULT:
{result[:2000]}

MISSING ELEMENTS:
{json.dumps(missing)}

Provide an enhanced result that addresses ALL missing elements.
Integrate the additions naturally into a complete response."""

        return await self.agent.think(enhancement_prompt)
    
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
