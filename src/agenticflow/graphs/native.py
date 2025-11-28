"""
NativeExecutor - High-performance native executor.

Key optimizations:
1. Direct asyncio loop (no graph framework overhead)
2. Parallel tool execution with asyncio.gather
3. Cached model binding for zero-overhead tool calls
4. Minimal prompt construction
5. No event emission in hot path

Includes standalone `run()` function for quick execution without Agent class.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

# Native message types for standalone run() function (uses native OpenAIChat)
from agenticflow.core.messages import (
    AIMessage as NativeAIMessage,
    BaseMessage as NativeBaseMessage,
    HumanMessage as NativeHumanMessage,
    SystemMessage as NativeSystemMessage,
    ToolMessage as NativeToolMessage,
)
from agenticflow.models.openai import OpenAIChat
from agenticflow.tools.base import BaseTool

# LangChain message types for NativeExecutor (which works with Agent's LangChain model)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

if TYPE_CHECKING:
    from agenticflow.agent import Agent


async def run(
    task: str,
    tools: list[BaseTool] | None = None,
    model: str | OpenAIChat = "gpt-4o-mini",
    system_prompt: str | None = None,
    max_iterations: int = 10,
    max_tool_calls: int = 20,
) -> str:
    """Execute a task with tools using native execution.
    
    Standalone function - no Agent class or LangChain required.
    Uses native OpenAI SDK for maximum performance.
    
    Args:
        task: The task to execute.
        tools: List of tools (created with @tool decorator).
        model: Model name or OpenAIChat instance.
        system_prompt: Optional system prompt.
        max_iterations: Maximum LLM call iterations.
        max_tool_calls: Maximum total tool calls.
        
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
    
    # Build initial messages using NATIVE message types
    messages: list[NativeBaseMessage] = []
    if system_prompt:
        messages.append(NativeSystemMessage(system_prompt))
    messages.append(NativeHumanMessage(task))
    
    # Execution loop
    total_tool_calls = 0
    
    for _ in range(max_iterations):
        # Native ChatModel returns NativeAIMessage
        response: NativeAIMessage = await bound_model.ainvoke(messages)
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
                messages.append(NativeToolMessage(f"Error: {result}", tool_call_id=tool_id))
            else:
                messages.append(result)
        
        total_tool_calls += len(response.tool_calls)
    
    return "Max iterations reached"


async def _execute_tool_native(
    tool_call: dict[str, Any],
    tool_map: dict[str, BaseTool],
) -> NativeToolMessage:
    """Execute a single tool call (returns native ToolMessage)."""
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    tool_id = tool_call.get("id", "")
    
    tool = tool_map.get(tool_name)
    if tool is None:
        return NativeToolMessage(f"Error: Unknown tool '{tool_name}'", tool_call_id=tool_id)
    
    try:
        result = await tool.ainvoke(args)
        return NativeToolMessage(str(result) if result is not None else "", tool_call_id=tool_id)
    except Exception as e:
        return NativeToolMessage(f"Error: {e}", tool_call_id=tool_id)


# Import BaseExecutor here to avoid circular import at module level
def _get_base_executor():
    from agenticflow.graphs.base import BaseExecutor
    return BaseExecutor


# We need BaseExecutor at class definition time, so import it
from agenticflow.graphs.base import BaseExecutor


class NativeExecutor(BaseExecutor):
    """
    High-performance native executor - the default choice.
    
    Features:
    - Direct asyncio loop (no graph framework)
    - Parallel tool execution with asyncio.gather
    - Cached bound model for fast tool binding
    - Minimal prompt construction
    - No event emission in hot path
    
    Trade-offs:
    - No resilience features (retry, circuit breaker)
    - No scratchpad/working memory
    - No event streaming
    - Limited error recovery
    
    Use for:
    - Most agent tasks (default executor)
    - Latency-critical applications
    - Simple to complex multi-tool tasks
    
    Example:
        from agenticflow.graphs import NativeExecutor
        
        executor = NativeExecutor(agent)
        result = await executor.execute("Research ACME Corp and calculate metrics")
    """
    
    __slots__ = ("_bound_model", "_tool_map", "_max_tool_calls")
    
    def __init__(
        self,
        agent: "Agent",
        max_tool_calls: int = 20,
    ) -> None:
        """Initialize NativeExecutor.
        
        Args:
            agent: The agent to execute.
            max_tool_calls: Maximum total tool calls across all iterations.
        """
        super().__init__(agent)
        self._max_tool_calls = max_tool_calls
        
        # Pre-cache everything at construction time
        self._bound_model = None
        self._tool_map: dict[str, Any] = {}
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
        """Execute task with maximum performance.
        
        Uses a tight asyncio loop with parallel tool execution.
        No LangGraph overhead, no event emission in hot path.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The final answer string.
        """
        # Build messages once
        messages = self._build_messages(task, context)
        
        # Track tool calls to enforce limit
        total_tool_calls = 0
        
        # Main execution loop (no graph, just async)
        for iteration in range(self.max_iterations):
            # Direct LLM call with bound tools
            response: AIMessage = await self._bound_model.ainvoke(messages)
            messages.append(response)
            
            # Check for tool calls
            if not response.tool_calls:
                # No tool calls - we have our final answer
                return response.content or ""
            
            # Check tool call limit
            num_calls = len(response.tool_calls)
            if total_tool_calls + num_calls > self._max_tool_calls:
                # Approaching limit - return what we have
                return response.content or "Tool call limit reached"
            
            # Execute ALL tool calls in parallel (like Agno)
            tool_results = await self._execute_tools_parallel(response.tool_calls)
            messages.extend(tool_results)
            total_tool_calls += num_calls
        
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
    tasks that require step-by-step reasoning.
    
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
