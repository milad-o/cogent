"""
NativeExecutor - High-performance native executor.

Key optimizations:
1. Direct asyncio loop (no graph framework overhead)
2. Parallel tool execution with asyncio.gather
3. Cached model binding for zero-overhead tool calls
4. Minimal prompt construction
5. LLM resilience with automatic retry for rate limits

Includes standalone `run()` function for quick execution without Agent class.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

# Native message types
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
from agenticflow.agent.output import (
    ResponseSchema,
    OutputMethod,
    StructuredResult,
    OutputValidationError,
    validate_and_parse,
    build_structured_prompt,
    get_best_method,
)
from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
    Phase,
    StopExecution,
    run_interceptors,
)
from agenticflow.context import RunContext, EMPTY_CONTEXT
from agenticflow.core.utils import model_identifier

if TYPE_CHECKING:
    from agenticflow.agent import Agent


async def run(
    task: str,
    tools: list[BaseTool] | None = None,
    model: str | BaseChatModel = "gpt-4o-mini",
    system_prompt: str | None = None,
    max_iterations: int = 25,
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
            # If the agent has an explicit resilience config, honor it for LLM calls.
            # This prevents long hangs in reactive/high-concurrency flows by allowing
            # fast-fail or tighter timeouts.
            agent_resilience_cfg = None
            try:
                agent_resilience_cfg = getattr(getattr(self.agent, "config", None), "resilience_config", None)
            except Exception:
                agent_resilience_cfg = None

            retry_policy = RetryPolicy.aggressive()
            timeout_seconds = 120.0
            if agent_resilience_cfg is not None:
                retry_policy = agent_resilience_cfg.retry_policy
                timeout_seconds = float(agent_resilience_cfg.timeout_seconds)

            # Create on_retry callback that logs to event bus
            async def _log_retry(attempt: int, error: Exception, delay: float) -> None:
                """Log retry attempt via event bus."""
                event_bus = getattr(self.agent, "event_bus", None)
                if event_bus:
                    from agenticflow.observability.event import EventType
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
                retry_policy=retry_policy,
                timeout_seconds=timeout_seconds,
                tracker=getattr(self, "tracker", None),
                on_retry=on_retry_sync,
            )
        
        self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Pre-cache model binding and tool map."""
        # Cache bound model with parallel tool calls enabled
        if self.agent.all_tools:
            # Enable parallel_tool_calls for batched tool execution
            self._bound_model = self.agent.model.bind_tools(
                self.agent.all_tools,
                parallel_tool_calls=True,
            )
            # Cache tool lookup map
            self._tool_map = {t.name: t for t in self.agent.all_tools}
        else:
            self._bound_model = self.agent.model
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | RunContext | None = None,
    ) -> Any:
        """Execute task with observability support.
        
        Uses a tight asyncio loop with parallel tool execution.
        Emits events for observability when event_bus is attached.
        Automatically retries on rate limits and transient errors.
        
        If the agent has reasoning enabled, performs a thinking phase
        before the main execution loop.
        
        If the agent has structured output configured, validates and parses
        the response according to the schema.
        
        Args:
            task: The task to execute.
            context: Optional context - can be a dict or RunContext.
                     If dict, it's used for message building only.
                     If RunContext, it's also passed to tools and interceptors.
            
        Returns:
            If output schema configured: StructuredResult with parsed data.
            Otherwise: The final answer string.
        """
        import time
        from agenticflow.observability.event import EventType
        
        # Normalize context: convert dict to RunContext with metadata
        run_context: RunContext
        context_dict: dict[str, Any] | None = None
        if context is None:
            run_context = EMPTY_CONTEXT
        elif isinstance(context, RunContext):
            run_context = context
            context_dict = context.metadata if context.metadata else None
        else:
            # Dict passed - store in metadata
            run_context = RunContext(metadata=context)
            context_dict = context
        
        # Get observability components
        event_bus = getattr(self.agent, 'event_bus', None)
        agent_name = self.agent.name or "agent"
        
        # Update resilience tracker if we have one
        if self._model_resilience and hasattr(self, "tracker"):
            self._model_resilience.tracker = self.tracker
        
        # Track execution timing
        execution_start = time.perf_counter()
        
        # Emit agent invoked event (not USER_INPUT - that's for user-facing input)
        if event_bus:
            await event_bus.publish(EventType.AGENT_INVOKED.value, {
                "agent": agent_name,
                "agent_name": agent_name,
                "task": task[:200] + "..." if len(task) > 200 else task,
            })
        
        # Build messages once
        messages = self._build_messages(task, context_dict)
        
        # === REASONING PHASE ===
        # If reasoning is enabled, think through the problem first
        reasoning_config = getattr(self.agent, '_reasoning_config', None)
        if reasoning_config is not None:
            reasoning_result = await self._execute_reasoning(
                task, context_dict, messages, event_bus, agent_name
            )
        
        # === STRUCTURED OUTPUT ===
        # If structured output is configured, use specialized execution with retry
        output_config = getattr(self.agent, '_output_config', None)
        if output_config is not None:
            return await self._execute_with_structured_output(
                task, context_dict, messages, event_bus, agent_name, run_context
            )
        
        # === STANDARD EXECUTION ===
        # No structured output - use main loop directly
        return await self._execute_main_loop(
            task, context_dict, messages, event_bus, agent_name, execution_start, run_context
        )

    async def execute_messages(
        self,
        *,
        task: str,
        messages: list[BaseMessage],
        context: dict[str, Any] | RunContext | None = None,
    ) -> Any:
        """Execute using a pre-built message list.

        This enables structured/reactive execution without converting event
        context into a single prompt string. The provided `messages` are used
        as-is (interceptors may still modify them).

        Args:
            task: High-level task label used for observability and completion checks.
            messages: Pre-built message list to send to the model.
            context: Optional context (dict or RunContext). Dict is stored as metadata.

        Returns:
            Final answer string (or structured output result if configured).
        """
        import time
        from agenticflow.observability.event import EventType

        # Normalize context
        run_context: RunContext
        context_dict: dict[str, Any] | None = None
        if context is None:
            run_context = EMPTY_CONTEXT
        elif isinstance(context, RunContext):
            run_context = context
            context_dict = context.metadata if context.metadata else None
        else:
            run_context = RunContext(metadata=context)
            context_dict = context

        if not messages:
            raise ValueError("execute_messages requires at least one message")

        # Ensure system prompt is present if agent has one and caller didn't include it
        sys_prompt = self.agent.config.system_prompt
        if sys_prompt and not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=sys_prompt), *messages]

        event_bus = getattr(self.agent, 'event_bus', None)
        agent_name = self.agent.name or "agent"

        if self._model_resilience and hasattr(self, "tracker"):
            self._model_resilience.tracker = self.tracker

        execution_start = time.perf_counter()

        if event_bus:
            await event_bus.publish(EventType.AGENT_INVOKED.value, {
                "agent": agent_name,
                "agent_name": agent_name,
                "task": task[:200] + "..." if len(task) > 200 else task,
            })

        # If reasoning is enabled, run it against the provided messages context.
        reasoning_config = getattr(self.agent, '_reasoning_config', None)
        if reasoning_config is not None:
            await self._execute_reasoning(task, context_dict, messages, event_bus, agent_name)

        # Structured output path: keep existing behavior for now.
        output_config = getattr(self.agent, '_output_config', None)
        if output_config is not None:
            # Reuse the structured output executor; it rebuilds messages from an effective task.
            # This preserves correctness, even though it doesn't preserve custom message shapes.
            return await self._execute_with_structured_output(
                task, context_dict, messages, event_bus, agent_name, run_context
            )

        return await self._execute_main_loop(
            task, context_dict, messages, event_bus, agent_name, execution_start, run_context
        )
    
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
        
        # Include conversation history from agent state (loaded from thread_id)
        if self.agent.state.message_history:
            messages.extend(self.agent.state.message_history)
        
        # User message with task
        if context:
            user_content = f"{task}\n\nContext: {context}"
        else:
            user_content = task
        
        messages.append(HumanMessage(content=user_content))
        return messages
    
    def _process_output(
        self,
        raw_content: str,
    ) -> Any:
        """Process raw output through structured output schema if configured.
        
        Args:
            raw_content: Raw LLM response content.
            
        Returns:
            If output schema configured: StructuredResult with parsed data.
            Otherwise: The raw content string.
        """
        output_config = getattr(self.agent, '_output_config', None)
        if output_config is None:
            return raw_content
        
        # Try to validate and parse
        try:
            data = validate_and_parse(raw_content, output_config.schema)
            return StructuredResult(
                data=data,
                raw=raw_content if output_config.include_raw else None,
                valid=True,
                attempts=1,
                method=get_best_method(self.agent.model, output_config),
            )
        except OutputValidationError as e:
            # Return invalid result
            return StructuredResult(
                data=None,
                raw=raw_content,
                valid=False,
                error=str(e),
                attempts=1,
                method=get_best_method(self.agent.model, output_config),
            )
        except Exception as e:
            return StructuredResult(
                data=None,
                raw=raw_content,
                valid=False,
                error=f"Unexpected error: {e}",
                attempts=1,
            )
    
    async def _execute_with_structured_output(
        self,
        task: str,
        context: dict[str, Any] | None,
        messages: list[BaseMessage],
        event_bus: Any | None,
        agent_name: str,
        run_context: RunContext | None = None,
    ) -> Any:
        """Execute with structured output support and retry on validation errors.
        
        This wraps the main execution loop with structured output handling,
        including retry logic when validation fails.
        
        Args:
            task: The task to execute.
            context: Optional context dict.
            messages: Initial messages.
            event_bus: Event bus for observability.
            agent_name: Agent name for events.
            run_context: Optional RunContext for tools and interceptors.
            
        Returns:
            StructuredResult if output schema configured, else raw string.
        """
        import time
        from agenticflow.observability.event import EventType
        
        output_config: ResponseSchema | None = getattr(self.agent, '_output_config', None)
        if output_config is None:
            # No structured output - delegate to normal flow
            # This should not be called, but handle gracefully
            return await self._execute_main_loop(
                task, context, messages, event_bus, agent_name, time.perf_counter(), run_context
            )
        
        # Structured output enabled
        max_attempts = output_config.max_retries + 1 if output_config.retry_on_error else 1
        last_error: str | None = None
        
        for attempt in range(1, max_attempts + 1):
            # Modify task with structured output instructions
            effective_task = build_structured_prompt(
                task,
                output_config,
                error_context=last_error,
            )
            
            # Rebuild messages with structured prompt
            structured_messages = self._build_messages(effective_task, context)
            
            # Execute main loop
            raw_result = await self._execute_main_loop(
                effective_task, context, structured_messages, 
                event_bus, agent_name, time.perf_counter(), run_context
            )
            
            # Try to validate
            try:
                data = validate_and_parse(raw_result, output_config.schema)
                return StructuredResult(
                    data=data,
                    raw=raw_result if output_config.include_raw else None,
                    valid=True,
                    attempts=attempt,
                    method=get_best_method(self.agent.model, output_config),
                )
            except OutputValidationError as e:
                last_error = str(e)
                if event_bus:
                    await event_bus.publish(EventType.AGENT_ERROR.value, {
                        "agent": agent_name,
                        "agent_name": agent_name,
                        "error": f"Structured output validation failed (attempt {attempt}): {last_error}",
                        "error_type": "validation",
                    })
                
                if attempt >= max_attempts:
                    # Return failed result
                    return StructuredResult(
                        data=None,
                        raw=raw_result,
                        valid=False,
                        error=last_error,
                        attempts=attempt,
                        method=get_best_method(self.agent.model, output_config),
                    )
        
        # Should not reach here
        return StructuredResult(data=None, valid=False, error="Max retries exceeded")
    
    async def _execute_main_loop(
        self,
        task: str,
        context: dict[str, Any] | None,
        messages: list[BaseMessage],
        event_bus: Any | None,
        agent_name: str,
        execution_start: float,
        run_context: RunContext | None = None,
    ) -> str:
        """Execute the main agent loop (extracted for reuse with structured output).
        
        Args:
            task: The task string.
            context: Optional context dict.
            messages: Message list.
            event_bus: Event bus for observability.
            agent_name: Agent name for events.
            execution_start: Start time for duration tracking.
            run_context: Optional RunContext for tools and interceptors.
            
        Returns:
            Raw response content string.
        """
        import time
        from agenticflow.observability.event import EventType
        
        total_tool_calls = 0
        model_calls = 0
        
        # Get interceptors from agent
        interceptors: list[Interceptor] = getattr(self.agent, '_interceptors', [])
        
        # Shared state for interceptors across the execution
        intercept_state: dict[str, Any] = {}
        
        # Current tools (can be modified by ToolGate interceptors)
        current_tools = list(self.agent.all_tools) if self.agent.all_tools else []
        
        # Current model (can be modified by Failover interceptors)
        current_model = self._bound_model
        
        # Current system prompt (can be modified by PromptAdapter interceptors)
        current_prompt = self.agent.config.system_prompt
        
        # Helper to create context for a phase
        def make_ctx(
            phase: Phase,
            **extra: Any,
        ) -> InterceptContext:
            return InterceptContext(
                agent=self.agent,
                phase=phase,
                task=task,
                messages=[{"role": getattr(m, "role", "unknown"), "content": getattr(m, "content", "")} for m in messages],
                state=intercept_state,
                run_context=run_context,
                tools=current_tools,
                model_calls=model_calls,
                tool_calls=total_tool_calls,
                **extra,
            )
        
        # PRE_RUN interceptors
        if interceptors:
            try:
                result = await run_interceptors(interceptors, make_ctx(Phase.PRE_RUN))
                if not result.proceed:
                    return result.final_response or "Execution stopped by interceptor"
                # Apply tool filtering if modified
                if result.modified_tools is not None:
                    current_tools = result.modified_tools
                # Apply prompt modification if set
                if result.modified_prompt is not None:
                    current_prompt = result.modified_prompt
                    # Rebuild messages with new prompt
                    if messages and isinstance(messages[0], SystemMessage):
                        messages[0] = SystemMessage(content=current_prompt)
                    elif current_prompt:
                        messages.insert(0, SystemMessage(content=current_prompt))
            except StopExecution as e:
                return e.response
        
        for iteration in range(self.max_iterations):
            # PRE_THINK interceptors
            if interceptors:
                try:
                    result = await run_interceptors(interceptors, make_ctx(Phase.PRE_THINK))
                    if not result.proceed:
                        return result.final_response or "Execution stopped by interceptor"
                    # Apply message modification if set
                    if result.modified_messages is not None:
                        # Convert dict messages to LangChain messages
                        new_messages: list[BaseMessage] = []
                        for msg in result.modified_messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            if role == "system":
                                new_messages.append(SystemMessage(content=content))
                            elif role == "assistant":
                                new_messages.append(AIMessage(content=content))
                            else:  # user or unknown
                                new_messages.append(HumanMessage(content=content))
                        messages = new_messages
                    # Apply tool filtering if modified
                    if result.modified_tools is not None:
                        current_tools = result.modified_tools
                    # Apply model switch if modified
                    if result.modified_model is not None:
                        current_model = result.modified_model
                        if current_tools:
                            current_model = current_model.bind_tools(current_tools, parallel_tool_calls=True)
                    # Apply prompt modification if set
                    if result.modified_prompt is not None:
                        current_prompt = result.modified_prompt
                        if messages and isinstance(messages[0], SystemMessage):
                            messages[0] = SystemMessage(content=current_prompt)
                except StopExecution as e:
                    return e.response
            
            # Emit thinking event
            if event_bus:
                await event_bus.publish(EventType.AGENT_THINKING.value, {
                    "agent": agent_name,
                    "agent_name": agent_name,
                    "iteration": iteration + 1,
                })
            
            # Emit LLM request event (for deep observability)
            if event_bus:
                await event_bus.publish(EventType.LLM_REQUEST.value, {
                    "agent_name": agent_name,
                    "model": model_identifier(self.agent.model),
                    "messages": [{"role": getattr(m, "role", "unknown"), "content": str(getattr(m, "content", ""))[:500]} for m in messages],
                    "message_count": len(messages),
                    "system_prompt": (current_prompt or "")[:300] if current_prompt else "",
                    "tools_available": [t.name for t in current_tools] if current_tools else [],
                    "prompt": task,
                    "iteration": iteration + 1,
                })
            
            loop_start = time.perf_counter()
            
            # LLM call with resilience
            if self._model_resilience:
                exec_result = await self._model_resilience.execute(
                    self._bound_model.ainvoke, messages
                )
                if not exec_result.success:
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
            
            loop_duration_ms = (time.perf_counter() - loop_start) * 1000
            model_calls += 1
            messages.append(response)
            
            # Emit LLM response event (for deep observability)
            tool_calls = response.tool_calls or []
            # Defensive: some providers/SDKs may omit tool_call IDs.
            # Azure/OpenAI validators require each tool result to reference a
            # preceding assistant message with tool_calls containing matching IDs.
            for i, tc in enumerate(tool_calls):
                if isinstance(tc, dict) and not tc.get("id"):
                    tc["id"] = f"call_{iteration + 1}_{i}"
            if event_bus:
                await event_bus.publish(EventType.LLM_RESPONSE.value, {
                    "agent_name": agent_name,
                    "iteration": iteration + 1,
                    "content": (response.content or "")[:500],
                    "content_length": len(response.content or ""),
                    "tool_calls": [
                        {"name": tc.get("name", "?"), "args": str(tc.get("args", {}))[:100]}
                        for tc in tool_calls[:5]
                    ] if tool_calls else [],
                    "has_tool_calls": bool(tool_calls),
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "duration_ms": loop_duration_ms,
                })
            
            # Emit tool decision event if tools were selected
            if tool_calls and event_bus:
                await event_bus.publish(EventType.LLM_TOOL_DECISION.value, {
                    "agent_name": agent_name,
                    "iteration": iteration + 1,
                    "tools_selected": [tc.get("name", "?") for tc in tool_calls],
                    "reasoning": (response.content or "")[:200] if response.content else "",
                })
            
            # POST_THINK interceptors
            if interceptors:
                try:
                    result = await run_interceptors(
                        interceptors, 
                        make_ctx(Phase.POST_THINK, model_response=response),
                    )
                    if not result.proceed:
                        return result.final_response or "Execution stopped by interceptor"
                except StopExecution as e:
                    return e.response
            
            # Check for tool calls (use already extracted tool_calls)
            if not tool_calls:
                duration_ms = (time.perf_counter() - execution_start) * 1000
                final_output = response.content or ""
                
                # POST_RUN interceptors
                if interceptors:
                    try:
                        await run_interceptors(interceptors, make_ctx(Phase.POST_RUN))
                    except StopExecution:
                        pass  # Already completing, ignore stop
                
                if event_bus:
                    await event_bus.publish(EventType.AGENT_RESPONDED.value, {
                        "agent": agent_name,
                        "agent_name": agent_name,
                        "response": final_output,
                        "response_preview": final_output[:500] if len(final_output) > 500 else final_output,
                        "thought": final_output,
                        "content": final_output,
                        "duration_ms": duration_ms,
                        "iteration": iteration + 1,
                    })
                    await event_bus.publish(EventType.OUTPUT_GENERATED.value, {
                        "agent": agent_name,
                        "agent_name": agent_name,
                        "output": final_output,
                        "content": final_output,
                        "duration_ms": duration_ms,
                    })
                return final_output
            
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
                limit_output = response.content or "Tool call limit reached"
                if event_bus:
                    await event_bus.publish(EventType.OUTPUT_GENERATED.value, {
                        "agent": agent_name,
                        "agent_name": agent_name,
                        "output": limit_output,
                        "content": limit_output,
                        "limited": True,
                    })
                return limit_output
            
            # Execute tools with PRE_ACT/POST_ACT interceptors
            tool_results = await self._execute_tools_with_interceptors(
                response.tool_calls,
                interceptors,
                make_ctx,
                run_context,
            )
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
        
        # Max iterations reached - POST_RUN
        if interceptors:
            try:
                await run_interceptors(interceptors, make_ctx(Phase.POST_RUN))
            except StopExecution:
                pass
        
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage):
                return last.content or "Max iterations reached"
        return "Max iterations reached"
    
    async def _execute_tools_with_interceptors(
        self,
        tool_calls: list[dict[str, Any]],
        interceptors: list[Interceptor],
        make_ctx: Any,
        run_context: RunContext | None = None,
    ) -> list[ToolMessage]:
        """Execute tools with PRE_ACT/POST_ACT interceptor hooks.
        
        Args:
            tool_calls: List of tool call dicts from LLM.
            interceptors: List of interceptors.
            make_ctx: Context factory function.
            run_context: Optional RunContext to pass to tools.
            
        Returns:
            List of ToolMessage results.
        """
        if not interceptors:
            # No interceptors - use fast parallel path
            return await self._execute_tools_parallel(tool_calls, run_context)
        
        results: list[ToolMessage] = []
        
        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            tool_id = tc.get("id", f"call_{tool_name}")
            tool_args = tc.get("args", {})
            
            # PRE_ACT
            try:
                pre_result = await run_interceptors(
                    interceptors,
                    make_ctx(
                        Phase.PRE_ACT,
                        tool_name=tool_name,
                        tool_args=tool_args,
                    ),
                )
                if not pre_result.proceed:
                    # Tool blocked by interceptor
                    results.append(ToolMessage(
                        content=pre_result.final_response or "Tool call blocked",
                        tool_call_id=tool_id,
                    ))
                    continue
                if pre_result.skip_action:
                    results.append(ToolMessage(
                        content="Tool call skipped",
                        tool_call_id=tool_id,
                    ))
                    continue
                # Apply modified args if any
                if pre_result.modified_tool_args:
                    tool_args = pre_result.modified_tool_args
            except StopExecution as e:
                results.append(ToolMessage(
                    content=e.response,
                    tool_call_id=tool_id,
                ))
                continue
            
            # Execute the tool
            tool = self._tool_map.get(tool_name)
            if tool:
                try:
                    if hasattr(tool, "ainvoke"):
                        result = await tool.ainvoke(tool_args, ctx=run_context)
                    else:
                        result = tool.invoke(tool_args, ctx=run_context)
                    result_str = str(result) if result is not None else ""
                except Exception as e:
                    result_str = f"Error: {e}"
            else:
                result_str = f"Unknown tool: {tool_name}"
            
            # POST_ACT
            try:
                post_result = await run_interceptors(
                    interceptors,
                    make_ctx(
                        Phase.POST_ACT,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=result_str,
                    ),
                )
                # POST_ACT can't stop execution, just observe
            except StopExecution:
                pass  # Ignore stop in POST_ACT
            
            results.append(ToolMessage(
                content=result_str,
                tool_call_id=tool_id,
            ))
        
        return results

    async def _execute_reasoning(
        self,
        task: str,
        context: dict[str, Any] | None,
        messages: list[BaseMessage],
        event_bus: Any | None,
        agent_name: str,
    ) -> dict[str, Any]:
        """Execute reasoning phase before main loop.
        
        Performs a thinking phase where the agent analyzes the task,
        breaks it down, and plans the approach.
        
        Args:
            task: The task to reason about.
            context: Optional context dict.
            messages: Current message list (will be modified in-place).
            event_bus: Event bus for observability.
            agent_name: Name of the agent.
            
        Returns:
            Dict with reasoning results (thinking_steps, plan, etc).
        """
        from agenticflow.observability.event import EventType
        from agenticflow.agent.reasoning import (
            ReasoningConfig,
            build_reasoning_prompt,
            extract_thinking,
            estimate_confidence,
            ThinkingStep,
            REASONING_SYSTEM_PROMPT,
            STYLE_INSTRUCTIONS,
        )
        
        config: ReasoningConfig = getattr(self.agent, '_reasoning_config', None)
        if config is None:
            return {}
        
        # Prepare reasoning results
        thinking_steps: list[ThinkingStep] = []
        
        # Build reasoning-enhanced system prompt
        style_instructions = STYLE_INSTRUCTIONS.get(config.style, "")
        reasoning_system = REASONING_SYSTEM_PROMPT.format(
            style_instructions=style_instructions
        )
        
        # Emit reasoning event (start)
        if event_bus:
            await event_bus.publish(EventType.AGENT_REASONING.value, {
                "agent": agent_name,
                "agent_name": agent_name,
                "phase": "start",
                "reasoning_type": "analysis",
                "round": 0,
                "style": config.style.value,
                "thought_preview": f"Beginning {config.style.value} reasoning...",
            })
        
        # Create reasoning messages
        reasoning_messages: list[BaseMessage] = [
            SystemMessage(content=reasoning_system),
            HumanMessage(content=build_reasoning_prompt(config, task, context)),
        ]
        
        # Use raw model without tools for reasoning (prevents tool_calls during thinking)
        reasoning_model = self.agent.model
        
        # Run thinking rounds
        final_thinking = ""
        for round_num in range(1, config.max_thinking_rounds + 1):
            # LLM call for thinking (use raw model, not bound model)
            if self._model_resilience:
                exec_result = await self._model_resilience.execute(
                    reasoning_model.ainvoke, reasoning_messages
                )
                if not exec_result.success:
                    break  # Stop reasoning on error, continue to main loop
                response: AIMessage = exec_result.result
            else:
                response: AIMessage = await reasoning_model.ainvoke(reasoning_messages)
            
            content = response.content or ""
            
            # Extract thinking from response
            thinking, cleaned = extract_thinking(content)
            if thinking:
                final_thinking = thinking
                confidence = estimate_confidence(thinking)
                
                step = ThinkingStep(
                    round=round_num,
                    thought=thinking,
                    reasoning_type="analysis" if round_num == 1 else "refinement",
                    confidence=confidence,
                )
                thinking_steps.append(step)
                
                # Emit reasoning event (step)
                if event_bus:
                    await event_bus.publish(EventType.AGENT_REASONING.value, {
                        "agent": agent_name,
                        "agent_name": agent_name,
                        "phase": "thinking",
                        "reasoning_type": step.reasoning_type,
                        "round": round_num,
                        "thought_preview": thinking[:200] + "..." if len(thinking) > 200 else thinking,
                        "confidence": confidence,
                    })
                
                # Check if confident enough to stop
                if config.require_confidence and confidence >= config.require_confidence:
                    break
                
                # Add thinking to messages for next round (if doing multiple rounds)
                if round_num < config.max_thinking_rounds:
                    reasoning_messages.append(response)
                    reasoning_messages.append(HumanMessage(
                        content="Good analysis. Can you refine your approach further? "
                        "Consider edge cases or alternative approaches."
                    ))
            else:
                # No <thinking> block found, use content as final plan
                final_thinking = content[:500]
                break
        
        # Emit reasoning event (complete)
        if event_bus:
            await event_bus.publish(EventType.AGENT_REASONING.value, {
                "agent": agent_name,
                "agent_name": agent_name,
                "phase": "complete",
                "reasoning_type": "reflection",
                "round": len(thinking_steps),
                "thinking_rounds": len(thinking_steps),
                "final_confidence": thinking_steps[-1].confidence if thinking_steps else None,
                "thought_preview": f"✓ Reasoning complete ({len(thinking_steps)} rounds)",
            })
        
        # Inject reasoning context into main messages
        if final_thinking and config.show_thinking:
            # Add thinking as an assistant message so the LLM sees it
            messages.append(AIMessage(content=f"<thinking>\n{final_thinking}\n</thinking>"))
        
        return {
            "thinking_steps": thinking_steps,
            "final_thinking": final_thinking,
            "rounds": len(thinking_steps),
        }

    async def _execute_tools_parallel(
        self,
        tool_calls: list[dict[str, Any]],
        run_context: RunContext | None = None,
    ) -> list[ToolMessage]:
        """Execute all tool calls in parallel using asyncio.gather.
        
        Args:
            tool_calls: List of tool call dictionaries from the LLM.
            run_context: Optional RunContext to pass to tools.
            
        Returns:
            List of ToolMessage results in the same order.
        """
        # Execute all tools concurrently
        results = await asyncio.gather(
            *(self._run_single_tool(tc, run_context) for tc in tool_calls),
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
        run_context: RunContext | None = None,
    ) -> ToolMessage:
        """Execute a single tool call.
        
        Args:
            tool_call: Tool call dict with name, args, id.
            run_context: Optional RunContext to pass to the tool.
            
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
            # Direct invocation with context support
            if asyncio.iscoroutinefunction(getattr(tool, "func", None)):
                result = await tool.ainvoke(args, ctx=run_context)
            else:
                # Run sync tools in thread pool to not block
                result = await asyncio.to_thread(tool.invoke, args, run_context)
            
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
