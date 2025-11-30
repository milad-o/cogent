"""
Rate limiting interceptors.

Interceptors for controlling the rate of tool calls to prevent
API abuse and respect rate limits.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
    StopExecution,
)


@dataclass
class RateLimiter(Interceptor):
    """Rate limits tool calls using a sliding window.
    
    Tracks tool calls within a time window and either delays or blocks
    calls that exceed the limit.
    
    Attributes:
        calls_per_window: Maximum calls allowed in the time window.
        window_seconds: Size of the sliding window in seconds.
        action: What to do when limit hit - "wait" or "block".
        per_tool: If True, limits each tool separately.
        
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.interceptors import RateLimiter
        
        # Max 10 tool calls per minute, wait if exceeded
        agent = Agent(
            name="assistant",
            model=model,
            tools=[search, api_call],
            intercept=[
                RateLimiter(
                    calls_per_window=10,
                    window_seconds=60,
                    action="wait",
                ),
            ],
        )
        
        # Separate limits per tool
        agent = Agent(
            name="api_bot",
            model=model,
            tools=[free_api, paid_api],
            intercept=[
                RateLimiter(
                    calls_per_window=5,
                    window_seconds=10,
                    per_tool=True,  # Each tool gets its own limit
                ),
            ],
        )
        ```
    """
    
    calls_per_window: int = 10
    window_seconds: float = 60.0
    action: str = "wait"  # "wait" or "block"
    per_tool: bool = False
    
    def __post_init__(self) -> None:
        """Initialize call tracking."""
        if self.calls_per_window < 1:
            raise ValueError("calls_per_window must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.action not in ("wait", "block"):
            raise ValueError("action must be 'wait' or 'block'")
        
        # Timestamps of recent calls: deque of (timestamp, tool_name)
        self._call_times: deque[tuple[float, str]] = deque()
        # Per-tool tracking if needed
        self._tool_times: dict[str, deque[float]] = {}
    
    def _prune_old_calls(self, now: float) -> None:
        """Remove calls outside the current window."""
        cutoff = now - self.window_seconds
        
        # Prune global calls
        while self._call_times and self._call_times[0][0] < cutoff:
            self._call_times.popleft()
        
        # Prune per-tool calls
        for tool_times in self._tool_times.values():
            while tool_times and tool_times[0] < cutoff:
                tool_times.popleft()
    
    def _get_wait_time(self, tool_name: str | None = None) -> float:
        """Calculate wait time until next call is allowed.
        
        Returns 0 if call is allowed now, otherwise seconds to wait.
        """
        now = time.monotonic()
        self._prune_old_calls(now)
        
        if self.per_tool and tool_name:
            # Check per-tool limit
            tool_times = self._tool_times.get(tool_name, deque())
            if len(tool_times) < self.calls_per_window:
                return 0.0
            oldest = tool_times[0]
        else:
            # Check global limit
            if len(self._call_times) < self.calls_per_window:
                return 0.0
            oldest = self._call_times[0][0]
        
        # Calculate wait time
        wait_until = oldest + self.window_seconds
        return max(0.0, wait_until - now)
    
    def _record_call(self, tool_name: str) -> None:
        """Record a tool call."""
        now = time.monotonic()
        self._call_times.append((now, tool_name))
        
        if self.per_tool:
            if tool_name not in self._tool_times:
                self._tool_times[tool_name] = deque()
            self._tool_times[tool_name].append(now)
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        """Check rate limit before tool execution."""
        tool_name = ctx.tool_name or ""
        wait_time = self._get_wait_time(tool_name if self.per_tool else None)
        
        if wait_time > 0:
            if self.action == "block":
                raise StopExecution(
                    f"Rate limit exceeded. Try again in {wait_time:.1f}s.",
                    f"Tool {tool_name} blocked by rate limit",
                )
            else:
                # Wait action - sleep until allowed
                ctx.state.setdefault("rate_limiter", {})
                ctx.state["rate_limiter"]["waited"] = wait_time
                await asyncio.sleep(wait_time)
        
        return InterceptResult.ok()
    
    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        """Record the tool call after execution."""
        tool_name = ctx.tool_name or ""
        self._record_call(tool_name)
        return InterceptResult.ok()
    
    def reset(self) -> None:
        """Reset all rate limit counters."""
        self._call_times.clear()
        self._tool_times.clear()
    
    @property
    def current_usage(self) -> dict[str, Any]:
        """Get current rate limit usage stats."""
        now = time.monotonic()
        self._prune_old_calls(now)
        
        result = {
            "global_calls": len(self._call_times),
            "limit": self.calls_per_window,
            "window_seconds": self.window_seconds,
        }
        
        if self.per_tool:
            result["per_tool"] = {
                name: len(times) 
                for name, times in self._tool_times.items()
            }
        
        return result


@dataclass
class ThrottleInterceptor(Interceptor):
    """Throttle tool calls with a minimum delay between calls.
    
    Simpler than RateLimiter - just enforces a minimum gap between
    consecutive tool calls.
    
    Attributes:
        min_delay: Minimum seconds between tool calls.
        per_tool: If True, track delay per tool separately.
        
    Example:
        ```python
        # At least 0.5 seconds between any tool calls
        agent = Agent(
            name="polite_bot",
            model=model,
            intercept=[
                ThrottleInterceptor(min_delay=0.5),
            ],
        )
        ```
    """
    
    min_delay: float = 0.5
    per_tool: bool = False
    
    def __post_init__(self) -> None:
        """Initialize timing."""
        if self.min_delay < 0:
            raise ValueError("min_delay must be non-negative")
        
        self._last_call: float = 0.0
        self._tool_last_call: dict[str, float] = {}
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        """Wait if needed before tool execution."""
        now = time.monotonic()
        tool_name = ctx.tool_name or ""
        
        if self.per_tool:
            last = self._tool_last_call.get(tool_name, 0.0)
        else:
            last = self._last_call
        
        elapsed = now - last
        if elapsed < self.min_delay:
            wait = self.min_delay - elapsed
            await asyncio.sleep(wait)
        
        return InterceptResult.ok()
    
    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        """Record call time after execution."""
        now = time.monotonic()
        tool_name = ctx.tool_name or ""
        
        self._last_call = now
        if self.per_tool:
            self._tool_last_call[tool_name] = now
        
        return InterceptResult.ok()


__all__ = [
    "RateLimiter",
    "ThrottleInterceptor",
]
