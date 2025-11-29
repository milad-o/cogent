"""
NativeExecutor - High-performance native executor.

Key optimizations:
1. Direct asyncio loop (no graph framework overhead)
2. Parallel tool execution with asyncio.gather
3. Cached model binding for zero-overhead tool calls
4. Minimal prompt construction
5. LLM resilience with automatic retry for rate limits

Includes standalone `run()` function for quick execution without Agent class.
All native - no LangChain/LangGraph dependencies.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

# Native message types (no LangChain)
from agenticflow.core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from agenticflow.models.base import BaseChatModel
from agenticflow.models.openai import OpenAIChat
from agenticflow.tools.base import BaseTool
from agenticflow.agent.resilience import ModelResilience, RetryPolicy

if TYPE_CHECKING:
    from agenticflow.agent import Agent


async def run(
    task: str,
    tools: list[BaseTool] | None = None,
    model: str | BaseChatModel = "gpt-4o-mini",
    system_prompt: str | None = None,
    max_iterations: int = 10,
    max_tool_calls: int = 20,
    resilience: bool = True,
    verbose: bool = False,
) -> str:
    """Execute a task with tools using native execution.
    
    Standalone function - no Agent class required.
    Uses native SDK for maximum performance.
    
    Args:
        task: The task to execute.
        tools: List of tools (created with @tool decorator).
        model: Model name or any native chat model instance.
        system_prompt: Optional system prompt.
        max_iterations: Maximum LLM call iterations.
        max_tool_calls: Maximum total tool calls.
        resilience: Enable automatic retry for rate limits (default: True).
        verbose: Print retry/error information (default: False).
        
    Returns:
        The final answer string.
        
    Example:
        from agenticflow import run, tool
        
        @tool
        def search(query: str) -> str:
            '''Search the web.'''
            return f"Results for {query}"
        
        result = await run(
            "Search for Python tutorials",
            tools=[search],
        )
        
        # With different providers
        from agenticflow.models.anthropic import AnthropicChat
        
        result = await run(
            "Explain quantum computing",
            model=AnthropicChat(model="claude-sonnet-4-20250514"),
        )
    """
    # Create or use model
    if isinstance(model, str):
        chat_model = OpenAIChat(model=model)
    else:
        chat_model = model
    
    # Bind tools if provided
    tools = tools or []
    tool_map = {t.name: t for t in tools}
    
    if tools:
        bound_model = chat_model.bind_tools(tools, parallel_tool_calls=True)
    else:
        bound_model = chat_model
    
    # Setup resilience for LLM calls
    model_resilience = None
    if resilience:
        def on_retry(attempt: int, error: Exception, delay: float) -> None:
            if verbose:
                print(f"⏳ LLM retry {attempt}: {str(error)[:80]}... (waiting {delay:.1f}s)")
        
        def on_error(error: Exception, attempts: int) -> None:
            if verbose:
                print(f"❌ LLM failed after {attempts} attempts: {error}")
        
        model_resilience = ModelResilience(
            retry_policy=RetryPolicy.aggressive(),
            on_retry=on_retry,
            on_error=on_error,
        )
    
    # Build initial messages using native message types
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(system_prompt))
    messages.append(HumanMessage(task))
    
    # Execution loop
    total_tool_calls = 0
    
    for _ in range(max_iterations):
        # LLM call with resilience
        if model_resilience:
            exec_result = await model_resilience.execute(bound_model.ainvoke, messages)
            if not exec_result.success:
                raise exec_result.error or RuntimeError("LLM call failed after retries")
            response: AIMessage = exec_result.result
        else:
            response: AIMessage = await bound_model.ainvoke(messages)
        
        messages.append(response)
        
        if not response.tool_calls:
            return response.content or ""
        
        # Check tool limit
        if total_tool_calls + len(response.tool_calls) > max_tool_calls:
            return response.content or "Tool call limit reached"
        
        # Execute tools in parallel
        tool_results = await asyncio.gather(
            *[_execute_tool_native(tc, tool_map) for tc in response.tool_calls],
            return_exceptions=True,
        )
        
        for result, tc in zip(tool_results, response.tool_calls):
            tool_id = tc.get("id", "")
            if isinstance(result, Exception):
                messages.append(ToolMessage(f"Error: {result}", tool_call_id=tool_id))
            else:
                messages.append(result)
        
        total_tool_calls += len(response.tool_calls)
    
    return "Max iterations reached"


async def _execute_tool_native(
    tool_call: dict[str, Any],
    tool_map: dict[str, BaseTool],
) -> ToolMessage:
    """Execute a single tool call (returns native ToolMessage)."""
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    tool_id = tool_call.get("id", "")
    
    tool = tool_map.get(tool_name)
    if tool is None:
        return ToolMessage(f"Error: Unknown tool '{tool_name}'", tool_call_id=tool_id)
    
    try:
        result = await tool.ainvoke(args)
        return ToolMessage(str(result) if result is not None else "", tool_call_id=tool_id)
    except Exception as e:
        return ToolMessage(f"Error: {e}", tool_call_id=tool_id)


# We need BaseExecutor at class definition time
from agenticflow.executors.base import BaseExecutor


class NativeExecutor(BaseExecutor):
    """
    High-performance native executor - the default choice.
    
    Features:
    - Direct asyncio loop (no graph framework)
    - Parallel tool execution with asyncio.gather
    - Cached bound model for fast tool binding
    - Minimal prompt construction
    - LLM resilience with automatic retry for rate limits
    
    Trade-offs:
    - No taskboard/working memory
    - No event streaming
    - Limited tool error recovery (but LLM calls are resilient)
    
    Use for:
    - Most agent tasks (default executor)
    - Latency-critical applications
    - Simple to complex multi-tool tasks
    
    Example:
        from agenticflow.executors import NativeExecutor
        
        executor = NativeExecutor(agent)
        result = await executor.execute("Research ACME Corp and calculate metrics")
    """
    
    __slots__ = ("_bound_model", "_tool_map", "_max_tool_calls", "_model_resilience")
    
    def __init__(
        self,
        agent: "Agent",
        max_tool_calls: int = 20,
        resilience: bool = True,
    ) -> None:
        """Initialize NativeExecutor.
        
        Args:
            agent: The agent to execute.
            max_tool_calls: Maximum total tool calls across all iterations.
            resilience: Enable automatic retry for LLM rate limits (default: True).
        """
        super().__init__(agent)
        self._max_tool_calls = max_tool_calls
        
        # Pre-cache everything at construction time
        self._bound_model = None
        self._tool_map: dict[str, Any] = {}
        self._model_resilience: ModelResilience | None = None
        
        # Setup model resilience
        if resilience:
            # Create on_retry callback that logs to event bus
            async def _log_retry(attempt: int, error: Exception, delay: float) -> None:
                """Log retry attempt via event bus."""
                event_bus = getattr(self.agent, "event_bus", None)
                if event_bus:
                    from agenticflow.core.enums import EventType
                    await event_bus.publish(EventType.AGENT_THINKING.value, {
                        "agent": self.agent.name or "agent",
                        "agent_name": self.agent.name or "agent",
                        "message": f"⏳ LLM retry {attempt}: {str(error)[:60]}... (waiting {delay:.1f}s)",
                    })
            
            def on_retry_sync(attempt: int, error: Exception, delay: float) -> None:
                """Sync wrapper that prints retry info."""
                import sys
                print(
                    f"⏳ LLM retry {attempt}: {str(error)[:80]}... (waiting {delay:.1f}s)",
                    file=sys.stderr,
                )
            
            self._model_resilience = ModelResilience(
                retry_policy=RetryPolicy.aggressive(),
                tracker=getattr(self, "tracker", None),
                on_retry=on_retry_sync,
            )
        
        self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Pre-cache model binding and tool map."""
        # Cache bound model with parallel tool calls enabled
        if self.agent.all_tools:
            # Enable parallel_tool_calls for batched tool execution (like Agno)
            self._bound_model = self.agent.model.bind_tools(
                self.agent.all_tools,
                parallel_tool_calls=True,  # Key optimization from Agno
            )
            # Cache tool lookup map
            self._tool_map = {t.name: t for t in self.agent.all_tools}
        else:
            self._bound_model = self.agent.model
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute task with observability support.
        
        Uses a tight asyncio loop with parallel tool execution.
        Emits events for observability when event_bus is attached.
        Automatically retries on rate limits and transient errors.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The final answer string.
        """
        import time
        from agenticflow.core.enums import EventType
        
        # Get observability components
        event_bus = getattr(self.agent, 'event_bus', None)
        agent_name = self.agent.name or "agent"
        
        # Update resilience tracker if we have one
        if self._model_resilience and hasattr(self, "tracker"):
            self._model_resilience.tracker = self.tracker
        
        # Track execution timing
        execution_start = time.perf_counter()
        
        # Emit agent invoked event
        if event_bus:
            await event_bus.publish(EventType.AGENT_INVOKED.value, {
                "agent": agent_name,
                "agent_name": agent_name,
                "task": task[:200] + "..." if len(task) > 200 else task,
            })
        
        # Build messages once
        messages = self._build_messages(task, context)
        
        # Track tool calls to enforce limit
        total_tool_calls = 0
        
        # Main execution loop (no graph, just async)
        for iteration in range(self.max_iterations):
            iteration_start = time.perf_counter()
            
            # Emit thinking event
            if event_bus:
                await event_bus.publish(EventType.AGENT_THINKING.value, {
                    "agent": agent_name,
                    "agent_name": agent_name,
                    "iteration": iteration + 1,
                })
            
            # LLM call with resilience (automatic retry for rate limits)
            if self._model_resilience:
                exec_result = await self._model_resilience.execute(
                    self._bound_model.ainvoke, messages
                )
                if not exec_result.success:
                    # Emit error event
                    if event_bus:
                        await event_bus.publish(EventType.AGENT_ERROR.value, {
                            "agent": agent_name,
                            "agent_name": agent_name,
                            "error": str(exec_result.error),
                            "error_type": exec_result.error_type,
                            "attempts": exec_result.attempts,
                        })
                    raise exec_result.error or RuntimeError(
                        f"LLM call failed after {exec_result.attempts} attempts"
                    )
                response: AIMessage = exec_result.result
            else:
                response: AIMessage = await self._bound_model.ainvoke(messages)
            
            messages.append(response)
            
            # Check for tool calls
            if not response.tool_calls:
                # No tool calls - we have our final answer
                duration_ms = (time.perf_counter() - execution_start) * 1000
                if event_bus:
                    content = response.content or ""
                    await event_bus.publish(EventType.AGENT_RESPONDED.value, {
                        "agent": agent_name,
                        "agent_name": agent_name,
                        "response": content,
                        "response_preview": content[:500] if len(content) > 500 else content,
                        "thought": content,  # Legacy field
                        "content": content,  # Legacy field
                        "duration_ms": duration_ms,
                        "iteration": iteration + 1,
                    })
                return response.content or ""
            
            # Emit tool call events
            if event_bus:
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    await event_bus.publish(EventType.TOOL_CALLED.value, {
                        "agent": agent_name,
                        "agent_name": agent_name,
                        "tool": tool_name,
                        "tool_name": tool_name,
                        "args": tc.get("args", {}),
                    })
            
            # Check tool call limit
            num_calls = len(response.tool_calls)
            if total_tool_calls + num_calls > self._max_tool_calls:
                # Approaching limit - return what we have
                return response.content or "Tool call limit reached"
            
            # Execute ALL tool calls in parallel (like Agno)
            tool_results = await self._execute_tools_parallel(response.tool_calls)
            messages.extend(tool_results)
            total_tool_calls += num_calls
            
            # Emit tool result events
            if event_bus:
                for i, result in enumerate(tool_results):
                    tc = response.tool_calls[i] if i < len(response.tool_calls) else {}
                    tool_name = tc.get("name", "unknown")
                    await event_bus.publish(EventType.TOOL_RESULT.value, {
                        "agent": agent_name,
                        "agent_name": agent_name,
                        "tool": tool_name,
                        "tool_name": tool_name,
                        "result": result.content[:500] if len(result.content) > 500 else result.content,
                    })
        
        # Max iterations reached
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage):
                return last.content or "Max iterations reached"
        return "Max iterations reached"
    
    def _build_messages(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> list[BaseMessage]:
        """Build minimal message list.
        
        Args:
            task: The task string.
            context: Optional context dict.
            
        Returns:
            List of messages for the LLM.
        """
        messages: list[BaseMessage] = []
        
        # System prompt (if any)
        sys_prompt = self.agent.config.system_prompt
        if sys_prompt:
            messages.append(SystemMessage(content=sys_prompt))
        
        # User message with task
        if context:
            user_content = f"{task}\n\nContext: {context}"
        else:
            user_content = task
        
        messages.append(HumanMessage(content=user_content))
        return messages
    
    async def _execute_tools_parallel(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[ToolMessage]:
        """Execute all tool calls in parallel using asyncio.gather.
        
        This is the key optimization from Agno's architecture.
        
        Args:
            tool_calls: List of tool call dictionaries from the LLM.
            
        Returns:
            List of ToolMessage results in the same order.
        """
        # Execute all tools concurrently
        results = await asyncio.gather(
            *(self._run_single_tool(tc) for tc in tool_calls),
            return_exceptions=True,
        )
        
        # Convert results to ToolMessages
        messages: list[ToolMessage] = []
        for result, tc in zip(results, tool_calls):
            tool_id = tc.get("id", "")
            
            if isinstance(result, Exception):
                messages.append(ToolMessage(
                    content=f"Error: {result}",
                    tool_call_id=tool_id,
                ))
            else:
                messages.append(result)
        
        return messages
    
    async def _run_single_tool(
        self,
        tool_call: dict[str, Any],
    ) -> ToolMessage:
        """Execute a single tool call.
        
        Args:
            tool_call: Tool call dict with name, args, id.
            
        Returns:
            ToolMessage with the result.
        """
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")
        
        # Track for metrics (but don't emit events)
        self._track_tool_call(tool_name, args)
        
        # Get tool from cache
        tool = self._tool_map.get(tool_name)
        if tool is None:
            return ToolMessage(
                content=f"Error: Unknown tool '{tool_name}'",
                tool_call_id=tool_id,
            )
        
        try:
            # Direct invocation - no resilience wrapper
            if asyncio.iscoroutinefunction(getattr(tool, "func", None)):
                result = await tool.ainvoke(args)
            else:
                # Run sync tools in thread pool to not block
                result = await asyncio.to_thread(tool.invoke, args)
            
            self._track_tool_result(tool_name, result, 0)
            
            return ToolMessage(
                content=str(result) if result is not None else "",
                tool_call_id=tool_id,
            )
            
        except Exception as e:
            self._track_tool_error(tool_name, str(e))
            return ToolMessage(
                content=f"Error: {e}",
                tool_call_id=tool_id,
            )


class SequentialExecutor(NativeExecutor):
    """
    NativeExecutor with sequential tool execution.
    
    Like NativeExecutor but executes tools one at a time for
    tasks that require step-by-step reasoning. Also includes
    LLM resilience for automatic retry on rate limits.
    
    Use when:
    - Tools depend on previous tool results
    - Order of execution matters
    - Debugging tool sequences
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute with sequential tool handling.
        
        Args:
            task: The task to execute.
            context: Optional context.
            
        Returns:
            Final answer string.
        """
        messages = self._build_messages(task, context)
        total_tool_calls = 0
        
        for _ in range(self.max_iterations):
            # LLM call with resilience
            if self._model_resilience:
                exec_result = await self._model_resilience.execute(
                    self._bound_model.ainvoke, messages
                )
                if not exec_result.success:
                    raise exec_result.error or RuntimeError("LLM call failed")
                response: AIMessage = exec_result.result
            else:
                response: AIMessage = await self._bound_model.ainvoke(messages)
            
            messages.append(response)
            
            if not response.tool_calls:
                return response.content or ""
            
            # Execute tools sequentially
            for tc in response.tool_calls:
                if total_tool_calls >= self._max_tool_calls:
                    return response.content or "Tool call limit reached"
                
                result = await self._run_single_tool(tc)
                messages.append(result)
                total_tool_calls += 1
        
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage):
                return last.content or "Max iterations reached"
        return "Max iterations reached"
