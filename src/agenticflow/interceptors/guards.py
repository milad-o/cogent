"""
ToolGuard - Tool retry interceptor.

Automatically retry failed tool calls with exponential backoff.

Example:
    from agenticflow import Agent
    from agenticflow.interceptors import ToolGuard
    
    agent = Agent(
        name="assistant",
        model=model,
        tools=[search, database_query],
        intercept=[
            ToolGuard(
                max_retries=3,
                backoff=2.0,
                retry_on=[TimeoutError, ConnectionError],
            ),
        ],
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Type

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
)


@dataclass
class ToolRetryState:
    """Tracks retry state for a tool call."""
    tool_name: str
    attempts: int = 0
    last_error: Exception | None = None
    last_attempt_time: float = 0.0


class ToolGuard(Interceptor):
    """Automatic retry for failed tool calls.
    
    Implements exponential backoff retry for transient failures.
    Tracks state per-tool and can be configured for specific error types.
    
    Args:
        max_retries: Maximum retry attempts per tool call.
        backoff: Backoff multiplier (delay = initial_delay * backoff^attempt).
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay between retries.
        retry_on: List of exception types to retry on.
        skip_on: List of exception types to never retry.
        on_retry: Callback(tool_name, attempt, error, delay) when retrying.
        
    Example:
        # Basic retry with defaults
        guard = ToolGuard(max_retries=3)
        
        # Custom configuration
        guard = ToolGuard(
            max_retries=5,
            backoff=2.0,
            initial_delay=0.5,
            retry_on=[TimeoutError, ConnectionError, IOError],
            on_retry=lambda t, a, e, d: print(f"Retry {t} #{a}: {e}"),
        )
    """
    
    STATE_KEY = "_toolguard_state"
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff: float = 2.0,
        initial_delay: float = 0.5,
        max_delay: float = 30.0,
        retry_on: list[Type[Exception]] | None = None,
        skip_on: list[Type[Exception]] | None = None,
        on_retry: Callable[[str, int, Exception, float], None] | None = None,
    ) -> None:
        """Initialize ToolGuard.
        
        Args:
            max_retries: Max retry attempts per tool call (default: 3).
            backoff: Backoff multiplier (default: 2.0).
            initial_delay: Initial delay in seconds (default: 0.5).
            max_delay: Maximum delay between retries (default: 30.0).
            retry_on: Exception types to retry on (default: all).
            skip_on: Exception types to never retry.
            on_retry: Callback when retrying.
        """
        self.max_retries = max_retries
        self.backoff = backoff
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retry_on = tuple(retry_on) if retry_on else None
        self.skip_on = tuple(skip_on) if skip_on else ()
        self.on_retry = on_retry
    
    def _get_states(self, ctx: InterceptContext) -> dict[str, ToolRetryState]:
        """Get or create retry states dict."""
        if self.STATE_KEY not in ctx.state:
            ctx.state[self.STATE_KEY] = {}
        return ctx.state[self.STATE_KEY]
    
    def _should_retry(self, error: Exception) -> bool:
        """Check if error should trigger retry."""
        # Never retry these
        if isinstance(error, self.skip_on):
            return False
        
        # If retry_on specified, only retry those
        if self.retry_on is not None:
            return isinstance(error, self.retry_on)
        
        # Default: retry on common transient errors
        return isinstance(error, (
            TimeoutError,
            ConnectionError,
            OSError,
        )) or "timeout" in str(error).lower() or "connection" in str(error).lower()
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.initial_delay * (self.backoff ** attempt)
        return min(delay, self.max_delay)
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        """Initialize retry state before tool execution."""
        if ctx.tool_name is None:
            return InterceptResult.ok()
        
        states = self._get_states(ctx)
        
        # Check if this is a retry
        tool_key = f"{ctx.tool_name}:{id(ctx.tool_args)}"
        if tool_key in states:
            state = states[tool_key]
            # Check if last attempt failed and we should wait
            if state.last_error is not None:
                delay = self._calculate_delay(state.attempts - 1)
                elapsed = time.time() - state.last_attempt_time
                if elapsed < delay:
                    await asyncio.sleep(delay - elapsed)
        else:
            states[tool_key] = ToolRetryState(tool_name=ctx.tool_name)
        
        return InterceptResult.ok()
    
    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        """Handle tool result and potentially trigger retry."""
        if ctx.tool_name is None:
            return InterceptResult.ok()
        
        states = self._get_states(ctx)
        tool_key = f"{ctx.tool_name}:{id(ctx.tool_args)}"
        state = states.get(tool_key)
        
        if state is None:
            return InterceptResult.ok()
        
        # Check if result indicates an error
        result_str = str(ctx.tool_result) if ctx.tool_result else ""
        is_error = result_str.startswith("Error:")
        
        if not is_error:
            # Success - clean up state
            states.pop(tool_key, None)
            return InterceptResult.ok()
        
        # Parse error from result
        state.attempts += 1
        state.last_attempt_time = time.time()
        
        # Check if we should retry
        if state.attempts >= self.max_retries:
            # Max retries exceeded
            states.pop(tool_key, None)
            return InterceptResult.ok()
        
        # Calculate delay and notify
        delay = self._calculate_delay(state.attempts)
        
        if self.on_retry:
            error = RuntimeError(result_str)
            self.on_retry(ctx.tool_name, state.attempts, error, delay)
        
        # Return metadata indicating retry needed
        # The executor will need to handle this
        return InterceptResult(
            proceed=True,
            metadata={
                "retry_tool": ctx.tool_name,
                "retry_attempt": state.attempts,
                "retry_delay": delay,
            },
        )


class CircuitBreaker(Interceptor):
    """Circuit breaker for tool calls.
    
    Prevents repeated calls to failing tools by "opening" the circuit
    after too many failures. The circuit stays open for a cooling period,
    then allows a test call to determine if the tool is healthy again.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Tool is failing, calls are blocked
    - HALF_OPEN: Testing if tool is healthy again
    
    Args:
        failure_threshold: Failures before opening circuit.
        reset_timeout: Seconds before testing again.
        tools: Specific tools to protect (None = all).
        
    Example:
        breaker = CircuitBreaker(
            failure_threshold=5,
            reset_timeout=30.0,
        )
    """
    
    STATE_KEY = "_circuit_breaker_state"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        tools: list[str] | None = None,
    ) -> None:
        """Initialize CircuitBreaker.
        
        Args:
            failure_threshold: Failures before opening (default: 5).
            reset_timeout: Seconds before half-open test (default: 30.0).
            tools: Tools to protect (default: all).
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.protected_tools = set(tools) if tools else None
    
    def _get_circuit_state(self, ctx: InterceptContext) -> dict[str, Any]:
        """Get or create circuit state."""
        if self.STATE_KEY not in ctx.state:
            ctx.state[self.STATE_KEY] = {}
        return ctx.state[self.STATE_KEY]
    
    def _is_protected(self, tool_name: str) -> bool:
        """Check if tool is protected by circuit breaker."""
        if self.protected_tools is None:
            return True
        return tool_name in self.protected_tools
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        """Check if circuit is open and block if so."""
        if ctx.tool_name is None or not self._is_protected(ctx.tool_name):
            return InterceptResult.ok()
        
        circuits = self._get_circuit_state(ctx)
        circuit = circuits.get(ctx.tool_name, {})
        
        state = circuit.get("state", "closed")
        
        if state == "open":
            # Check if reset timeout has passed
            open_time = circuit.get("open_time", 0)
            if time.time() - open_time >= self.reset_timeout:
                # Move to half-open
                circuit["state"] = "half_open"
                circuits[ctx.tool_name] = circuit
            else:
                # Block the call
                return InterceptResult.stop(
                    f"Circuit breaker open for {ctx.tool_name}"
                )
        
        return InterceptResult.ok()
    
    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        """Track failures and update circuit state."""
        if ctx.tool_name is None or not self._is_protected(ctx.tool_name):
            return InterceptResult.ok()
        
        circuits = self._get_circuit_state(ctx)
        circuit = circuits.get(ctx.tool_name, {"failures": 0, "state": "closed"})
        
        result_str = str(ctx.tool_result) if ctx.tool_result else ""
        is_error = result_str.startswith("Error:")
        
        if is_error:
            circuit["failures"] = circuit.get("failures", 0) + 1
            
            if circuit.get("state") == "half_open":
                # Test failed, reopen circuit
                circuit["state"] = "open"
                circuit["open_time"] = time.time()
            elif circuit["failures"] >= self.failure_threshold:
                # Open the circuit
                circuit["state"] = "open"
                circuit["open_time"] = time.time()
        else:
            # Success
            if circuit.get("state") == "half_open":
                # Test passed, close circuit
                circuit["state"] = "closed"
                circuit["failures"] = 0
            else:
                # Reset failure count on success
                circuit["failures"] = max(0, circuit.get("failures", 0) - 1)
        
        circuits[ctx.tool_name] = circuit
        return InterceptResult.ok()


__all__ = [
    "ToolGuard",
    "ToolRetryState",
    "CircuitBreaker",
]
