"""
BudgetGuard - Enforce call limits for cost control.

Limits the number of model calls and/or tool calls an agent can make
in a single run, preventing runaway costs and infinite loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
    Phase,
)


class ExitBehavior(Enum):
    """What to do when budget is exhausted."""
    STOP = "stop"      # Stop execution, return final response
    ERROR = "error"    # Raise an exception


@dataclass
class BudgetGuard(Interceptor):
    """Enforce call limits for cost control.
    
    Tracks and limits the number of model calls and tool calls
    an agent can make during execution. Useful for:
    - Preventing runaway agents from making excessive API calls
    - Enforcing cost controls in production
    - Testing agent behavior within specific budgets
    
    Attributes:
        model_calls: Maximum model calls allowed (0 = unlimited).
        tool_calls: Maximum tool calls allowed (0 = unlimited).
        exit_behavior: What to do when limit reached (stop or error).
        warning_threshold: Warn when this % of budget used (0-1).
    
    Example:
        # Limit to 10 model calls and 50 tool calls
        guard = BudgetGuard(model_calls=10, tool_calls=50)
        
        agent = Agent(
            name="assistant",
            model=model,
            intercept=[guard],
        )
        
        # After run, check usage
        print(f"Model calls: {guard.current_model_calls}")
        print(f"Tool calls: {guard.current_tool_calls}")
    """
    
    model_calls: int = 0  # 0 = unlimited
    tool_calls: int = 0   # 0 = unlimited
    exit_behavior: ExitBehavior | str = ExitBehavior.STOP
    warning_threshold: float = 0.8
    
    # Tracking (reset per run)
    _current_model_calls: int = 0
    _current_tool_calls: int = 0
    
    def __post_init__(self) -> None:
        """Normalize exit_behavior to enum."""
        if isinstance(self.exit_behavior, str):
            self.exit_behavior = ExitBehavior(self.exit_behavior)
    
    @property
    def current_model_calls(self) -> int:
        """Current model call count."""
        return self._current_model_calls
    
    @property
    def current_tool_calls(self) -> int:
        """Current tool call count."""
        return self._current_tool_calls
    
    @property
    def model_budget_remaining(self) -> int | None:
        """Remaining model calls, or None if unlimited."""
        if self.model_calls == 0:
            return None
        return max(0, self.model_calls - self._current_model_calls)
    
    @property
    def tool_budget_remaining(self) -> int | None:
        """Remaining tool calls, or None if unlimited."""
        if self.tool_calls == 0:
            return None
        return max(0, self.tool_calls - self._current_tool_calls)
    
    def reset(self) -> None:
        """Reset counters (called automatically at PRE_RUN)."""
        self._current_model_calls = 0
        self._current_tool_calls = 0
    
    async def pre_run(self, ctx: InterceptContext) -> InterceptResult:
        """Reset counters at start of run."""
        self.reset()
        return InterceptResult.ok()
    
    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Check model call budget before calling model."""
        if self.model_calls > 0:
            if self._current_model_calls >= self.model_calls:
                return self._handle_limit_reached(
                    "model", 
                    self._current_model_calls, 
                    self.model_calls,
                )
            
            # Warn if approaching limit
            if self._is_near_limit(self._current_model_calls, self.model_calls):
                self._warn("model", self._current_model_calls, self.model_calls)
        
        return InterceptResult.ok()
    
    async def post_think(self, ctx: InterceptContext) -> InterceptResult:
        """Increment model call counter after model responds."""
        self._current_model_calls += 1
        return InterceptResult.ok()
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        """Check tool call budget before executing tool."""
        if self.tool_calls > 0:
            if self._current_tool_calls >= self.tool_calls:
                return self._handle_limit_reached(
                    "tool",
                    self._current_tool_calls,
                    self.tool_calls,
                )
            
            # Warn if approaching limit
            if self._is_near_limit(self._current_tool_calls, self.tool_calls):
                self._warn("tool", self._current_tool_calls, self.tool_calls)
        
        return InterceptResult.ok()
    
    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        """Increment tool call counter after tool returns."""
        self._current_tool_calls += 1
        return InterceptResult.ok()
    
    def _is_near_limit(self, current: int, limit: int) -> bool:
        """Check if approaching the limit."""
        if limit == 0:
            return False
        return current >= int(limit * self.warning_threshold)
    
    def _warn(self, call_type: str, current: int, limit: int) -> None:
        """Emit warning when approaching limit."""
        import warnings
        remaining = limit - current
        warnings.warn(
            f"BudgetGuard: {remaining} {call_type} calls remaining "
            f"({current}/{limit} used)",
            stacklevel=4,
        )
    
    def _handle_limit_reached(
        self, 
        call_type: str, 
        current: int, 
        limit: int,
    ) -> InterceptResult:
        """Handle when a limit is reached."""
        message = (
            f"Budget exhausted: {call_type} call limit reached "
            f"({current}/{limit})"
        )
        
        if self.exit_behavior == ExitBehavior.ERROR:
            raise BudgetExhaustedError(message, call_type, current, limit)
        
        return InterceptResult.stop(
            f"I've reached my {call_type} call limit ({limit} calls). "
            f"Please try again with a simpler request."
        )


class BudgetExhaustedError(Exception):
    """Raised when budget is exhausted and exit_behavior is ERROR."""
    
    def __init__(
        self, 
        message: str, 
        call_type: str, 
        current: int, 
        limit: int,
    ) -> None:
        self.call_type = call_type
        self.current = current
        self.limit = limit
        super().__init__(message)


__all__ = [
    "BudgetGuard",
    "BudgetExhaustedError",
    "ExitBehavior",
]
