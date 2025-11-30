"""
Base classes for interceptors.

Interceptors are composable units that can intercept agent execution at key phases.
They enable cross-cutting concerns like:
- Cost control (limit calls)
- Context management (summarization)
- Security (PII detection)
- Observability (logging, metrics)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.schemas.message import Message


class Phase(Enum):
    """Execution phase where interceptor runs.
    
    The agent execution loop has these phases:
    
    1. PRE_RUN: Before agent.run() starts processing
    2. PRE_THINK: Before each model call
    3. POST_THINK: After model responds (before tool execution)
    4. PRE_ACT: Before each tool execution
    5. POST_ACT: After each tool returns
    6. POST_RUN: After agent.run() completes
    7. ON_ERROR: When an error occurs
    """
    PRE_RUN = "pre_run"
    PRE_THINK = "pre_think"
    POST_THINK = "post_think"
    PRE_ACT = "pre_act"
    POST_ACT = "post_act"
    POST_RUN = "post_run"
    ON_ERROR = "on_error"


@dataclass
class InterceptContext:
    """Context passed to interceptors at each phase.
    
    Contains all relevant information about current execution state.
    The `state` dict is mutable and shared across all interceptors,
    allowing them to communicate and track information.
    
    Attributes:
        agent: The agent being executed.
        phase: Current execution phase.
        task: The original task/prompt.
        messages: Current message history.
        state: Mutable shared state dict for interceptors.
        tool_name: Name of tool (only in PRE_ACT/POST_ACT).
        tool_args: Tool arguments (only in PRE_ACT/POST_ACT).
        tool_result: Tool result (only in POST_ACT).
        error: Exception if ON_ERROR phase.
        model_response: Model response (only in POST_THINK).
    """
    agent: Agent
    phase: Phase
    task: str
    messages: list[dict[str, Any]]
    state: dict[str, Any] = field(default_factory=dict)
    
    # Phase-specific data
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    error: Exception | None = None
    model_response: Any = None
    
    # Execution counters (updated by executor)
    model_calls: int = 0
    tool_calls: int = 0
    
    def __post_init__(self) -> None:
        """Initialize state if not provided."""
        if self.state is None:
            self.state = {}


@dataclass
class InterceptResult:
    """Result from an interceptor.
    
    Interceptors return this to indicate what should happen next.
    
    Attributes:
        proceed: Whether to continue execution (False = stop).
        modified_messages: If set, replace messages with these.
        modified_task: If set, replace task with this.
        modified_tool_args: If set, replace tool args (PRE_ACT only).
        skip_action: Skip the current action but continue loop.
        final_response: If stopping, use this as final response.
        metadata: Optional metadata to attach to result.
    """
    proceed: bool = True
    modified_messages: list[dict[str, Any]] | None = None
    modified_task: str | None = None
    modified_tool_args: dict[str, Any] | None = None
    skip_action: bool = False
    final_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls) -> InterceptResult:
        """Continue execution normally."""
        return cls(proceed=True)
    
    @classmethod
    def stop(cls, response: str | None = None) -> InterceptResult:
        """Stop execution with optional response."""
        return cls(proceed=False, final_response=response)
    
    @classmethod
    def skip(cls) -> InterceptResult:
        """Skip current action but continue execution."""
        return cls(proceed=True, skip_action=True)
    
    @classmethod
    def modify_messages(cls, messages: list[dict[str, Any]]) -> InterceptResult:
        """Continue with modified messages."""
        return cls(proceed=True, modified_messages=messages)
    
    @classmethod
    def modify_args(cls, args: dict[str, Any]) -> InterceptResult:
        """Continue with modified tool arguments."""
        return cls(proceed=True, modified_tool_args=args)


class StopExecution(Exception):
    """Raised by interceptors to halt execution immediately.
    
    Use this when you need to stop execution and return a specific response.
    Preferable to returning InterceptResult.stop() when you need to
    bail out from deep in the execution stack.
    
    Attributes:
        response: The response to return.
        reason: Why execution was stopped.
    """
    
    def __init__(self, response: str, reason: str = ""):
        self.response = response
        self.reason = reason
        super().__init__(reason or response)


class Interceptor(ABC):
    """Base class for all interceptors.
    
    Interceptors can hook into any execution phase by implementing
    the corresponding method. Default implementations return ok().
    
    Example:
        class LoggingInterceptor(Interceptor):
            async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
                print(f"Model call #{ctx.model_calls + 1}")
                return InterceptResult.ok()
                
            async def post_act(self, ctx: InterceptContext) -> InterceptResult:
                print(f"Tool {ctx.tool_name} returned: {ctx.tool_result}")
                return InterceptResult.ok()
    
    Note:
        Override only the phases you need. Unimplemented phases pass through.
    """
    
    @property
    def name(self) -> str:
        """Interceptor name for logging/debugging."""
        return self.__class__.__name__
    
    async def intercept(self, ctx: InterceptContext) -> InterceptResult:
        """Main dispatch method - routes to phase-specific handlers.
        
        You typically don't override this. Override the phase methods instead.
        """
        match ctx.phase:
            case Phase.PRE_RUN:
                return await self.pre_run(ctx)
            case Phase.PRE_THINK:
                return await self.pre_think(ctx)
            case Phase.POST_THINK:
                return await self.post_think(ctx)
            case Phase.PRE_ACT:
                return await self.pre_act(ctx)
            case Phase.POST_ACT:
                return await self.post_act(ctx)
            case Phase.POST_RUN:
                return await self.post_run(ctx)
            case Phase.ON_ERROR:
                return await self.on_error(ctx)
            case _:
                return InterceptResult.ok()
    
    # Phase handlers - override as needed
    
    async def pre_run(self, ctx: InterceptContext) -> InterceptResult:
        """Called before agent.run() starts."""
        return InterceptResult.ok()
    
    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Called before each model call."""
        return InterceptResult.ok()
    
    async def post_think(self, ctx: InterceptContext) -> InterceptResult:
        """Called after model responds."""
        return InterceptResult.ok()
    
    async def pre_act(self, ctx: InterceptContext) -> InterceptResult:
        """Called before tool execution."""
        return InterceptResult.ok()
    
    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        """Called after tool returns."""
        return InterceptResult.ok()
    
    async def post_run(self, ctx: InterceptContext) -> InterceptResult:
        """Called after agent.run() completes."""
        return InterceptResult.ok()
    
    async def on_error(self, ctx: InterceptContext) -> InterceptResult:
        """Called when an error occurs."""
        return InterceptResult.ok()


async def run_interceptors(
    interceptors: list[Interceptor],
    ctx: InterceptContext,
) -> InterceptResult:
    """Run all interceptors for a given phase.
    
    Interceptors run in order. If any returns proceed=False,
    execution stops and that result is returned.
    
    Modifications accumulate:
    - modified_messages from later interceptors override earlier ones
    - modified_tool_args from later interceptors override earlier ones
    
    Args:
        interceptors: List of interceptors to run.
        ctx: The context for this phase.
        
    Returns:
        Combined InterceptResult from all interceptors.
    """
    result = InterceptResult.ok()
    
    for interceptor in interceptors:
        try:
            r = await interceptor.intercept(ctx)
            
            # Stop immediately if told to
            if not r.proceed:
                return r
            
            # Accumulate modifications
            if r.modified_messages is not None:
                result.modified_messages = r.modified_messages
                ctx.messages = r.modified_messages  # Update for next interceptor
            
            if r.modified_task is not None:
                result.modified_task = r.modified_task
                ctx.task = r.modified_task
            
            if r.modified_tool_args is not None:
                result.modified_tool_args = r.modified_tool_args
                ctx.tool_args = r.modified_tool_args
            
            if r.skip_action:
                result.skip_action = True
            
            # Merge metadata
            result.metadata.update(r.metadata)
            
        except StopExecution as e:
            return InterceptResult.stop(e.response)
    
    return result


__all__ = [
    "Phase",
    "InterceptContext",
    "InterceptResult",
    "StopExecution",
    "Interceptor",
    "run_interceptors",
]
