"""
Agent - an autonomous entity that can think, act, and communicate.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Sequence, overload

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agenticflow.agent.config import AgentConfig
from agenticflow.agent.state import AgentState
from agenticflow.agent.scratchpad import Scratchpad
from agenticflow.agent.hitl import (
    should_interrupt,
    PendingAction,
    HumanDecision,
    InterruptedState,
    InterruptReason,
    DecisionType,
    InterruptedException,
    AbortedException,
)
from agenticflow.core.enums import AgentRole, AgentStatus, EventType
from agenticflow.core.utils import generate_id, now_utc
from agenticflow.models.event import Event
from agenticflow.models.message import Message
from agenticflow.models.task import Task

if TYPE_CHECKING:
    from agenticflow.events.bus import EventBus
    from agenticflow.tools.registry import ToolRegistry
    from agenticflow.observability.progress import ProgressTracker
    from agenticflow.agent.resilience import ToolResilience, FallbackRegistry, ResilienceConfig
    from agenticflow.agent.streaming import StreamChunk, StreamEvent


class Agent:
    """
    An autonomous entity that can think, act, and communicate.
    
    Agents are the primary actors in the system. Each agent has:
    - A unique identity and role
    - Configuration defining its capabilities
    - Runtime state tracking its activity
    - Access to tools and the event bus
    
    Agents communicate through the event bus and can:
    - Receive and process tasks
    - Use tools to accomplish goals
    - Spawn subtasks for complex work
    - Send messages to other agents
    
    Simplified API (recommended):
        ```python
        from langchain_openai import ChatOpenAI
        from langchain.tools import tool
        
        @tool
        def search(query: str) -> str:
            '''Search for information.'''
            return f"Results for {query}"
        
        model = ChatOpenAI(model="gpt-4o")
        
        agent = Agent(
            name="Researcher",
            model=model,
            tools=[search],  # Pass tool functions directly
            description="Researches topics",
        )
        
        result = await agent.think("What should I research?")
        ```
    
    Advanced API (with AgentConfig):
        ```python
        from agenticflow import Agent, AgentConfig, AgentRole
        
        config = AgentConfig(
            name="Writer",
            role=AgentRole.WORKER,
            model=model,
            tools=["write_poem", "write_story"],
            resilience_config=ResilienceConfig.aggressive(),
        )
        
        agent = Agent(config=config, event_bus=event_bus)
        ```
    
    When used with Flow (recommended):
        ```python
        from agenticflow import Flow, Agent
        
        # No need to pass event_bus or tool_registry
        agent = Agent(name="Worker", model=model, tools=[my_tool])
        
        # Flow wires everything automatically
        flow = Flow(name="my-flow", agents=[agent], topology="pipeline")
        ```
    """

    @overload
    def __init__(
        self,
        *,
        name: str,
        model: BaseChatModel | None = None,
        role: AgentRole | str = AgentRole.WORKER,
        description: str = "",
        instructions: str | None = None,
        tools: Sequence[BaseTool | str] | None = None,
        capabilities: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        resilience: ResilienceConfig | None = None,
        memory: Any = None,
        store: Any = None,
    ) -> None:
        """Simplified constructor - create agent with direct parameters."""
        ...

    @overload
    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus | None = None,
        tool_registry: ToolRegistry | None = None,
        memory: Any = None,
        store: Any = None,
    ) -> None:
        """Advanced constructor - create agent with AgentConfig."""
        ...

    def __init__(
        self,
        config: AgentConfig | None = None,
        event_bus: EventBus | None = None,
        tool_registry: ToolRegistry | None = None,
        memory: Any = None,
        store: Any = None,
        *,
        # Simplified API parameters
        name: str | None = None,
        model: BaseChatModel | None = None,
        role: AgentRole | str = AgentRole.WORKER,
        description: str = "",
        instructions: str | None = None,
        tools: Sequence[BaseTool | str] | None = None,
        capabilities: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        resilience: ResilienceConfig | None = None,
        interrupt_on: dict[str, Any] | None = None,  # HITL: tool approval rules
        stream: bool = False,  # Enable streaming by default for this agent
    ) -> None:
        """
        Initialize an Agent.
        
        Can be called two ways:
        
        1. Simplified (recommended for use with Flow):
            Agent(name="Worker", model=model, tools=[...], instructions="...")
        
        2. Advanced (for full control):
            Agent(config=AgentConfig(...), event_bus=bus, tool_registry=registry)
        
        Args:
            config: Agent configuration (advanced API)
            event_bus: Event bus for communication (optional, Flow provides this)
            tool_registry: Registry of available tools (optional, Flow provides this)
            memory: Memory backend for conversation persistence. Accepts:
                - True: Use built-in InMemoryCheckpointer (for testing)
                - LangGraph checkpointer: MemorySaver, SqliteSaver, PostgresSaver
                - AgentMemory instance: For custom configuration
            store: LangGraph store for long-term memory across threads.
            name: Agent name (simplified API)
            model: LangChain model (simplified API)
            role: Agent role (simplified API)
            description: Agent description (simplified API)
            instructions: Instructions for the agent - defines behavior and personality
            tools: List of tools - can be BaseTool objects or strings (simplified API)
            system_prompt: System prompt (alias for instructions, for compatibility)
            resilience: Resilience configuration (simplified API)
            stream: Enable token-by-token streaming by default for this agent.
                   When True, chat() and think() return async iterators.
            
        Example with memory:
            ```python
            from langgraph.checkpoint.memory import MemorySaver
            
            # Simple: just pass True for in-memory
            agent = Agent(name="Assistant", model=model, memory=True)
            
            # LangGraph checkpointer for persistence
            agent = Agent(name="Assistant", model=model, memory=MemorySaver())
            
            # Chat with thread-based memory
            response = await agent.chat("Hi, I'm Alice", thread_id="conv-1")
            response = await agent.chat("What's my name?", thread_id="conv-1")  # Remembers!
            ```
        """
        # Handle simplified API
        if config is None:
            if name is None:
                raise ValueError(
                    "Either provide 'config' (AgentConfig) or 'name' parameter.\n"
                    "Simplified: Agent(name='Worker', model=model)\n"
                    "Advanced: Agent(config=AgentConfig(...))"
                )
            
            # Parse role if string
            if isinstance(role, str):
                role = AgentRole(role.lower())
            
            # Extract tool names and store tool objects
            tool_names: list[str] = []
            self._direct_tools: list[BaseTool] = []
            
            if tools:
                for tool in tools:
                    if isinstance(tool, str):
                        tool_names.append(tool)
                    elif isinstance(tool, BaseTool):
                        tool_names.append(tool.name)
                        self._direct_tools.append(tool)
                    else:
                        # Try to get name attribute
                        tool_names.append(getattr(tool, "name", str(tool)))
            
            # instructions takes priority over system_prompt
            # If neither provided, use role-specific default prompt
            effective_prompt = instructions or system_prompt
            if not effective_prompt:
                from agenticflow.agent.roles import get_role_prompt
                effective_prompt = get_role_prompt(role)
            
            # Create config from simplified params
            config = AgentConfig(
                name=name,
                role=role,
                description=description,
                model=model,
                system_prompt=effective_prompt,
                tools=tool_names,
                resilience_config=resilience,
                interrupt_on=interrupt_on or {},
                stream=stream,
            )
        else:
            # Using AgentConfig directly - still need to process tools parameter
            self._direct_tools = []
            if tools:
                for tool in tools:
                    if isinstance(tool, str):
                        if tool not in config.tools:
                            config.tools.append(tool)
                    elif isinstance(tool, BaseTool):
                        if tool.name not in config.tools:
                            config.tools.append(tool.name)
                        self._direct_tools.append(tool)
                    else:
                        tool_name = getattr(tool, "name", str(tool))
                        if tool_name not in config.tools:
                            config.tools.append(tool_name)
            
            # Apply default role prompt if no system_prompt provided
            if not config.system_prompt:
                from agenticflow.agent.roles import get_role_prompt
                config = AgentConfig(
                    name=config.name,
                    role=config.role,
                    description=config.description,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    system_prompt=get_role_prompt(config.role),
                    model_kwargs=config.model_kwargs,
                    stream=config.stream,
                    tools=config.tools,
                    max_concurrent_tasks=config.max_concurrent_tasks,
                    timeout_seconds=config.timeout_seconds,
                    retry_on_error=config.retry_on_error,
                    max_retries=config.max_retries,
                    resilience_config=config.resilience_config,
                    fallback_tools=config.fallback_tools,
                    interrupt_on=config.interrupt_on,
                    metadata=config.metadata,
                )
        
        self.id = generate_id()
        self.config = config
        self.state = AgentState()
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self._model = None
        self._lock = asyncio.Lock()
        self._resilience: ToolResilience | None = None
        self._capabilities: list[Any] = []
        self._setup_resilience()
        
        # Human-in-the-loop state
        self._pending_actions: dict[str, PendingAction] = {}  # action_id -> pending action
        self._interrupted_state: InterruptedState | None = None
        
        # Setup capabilities (adds tools from each capability)
        if capabilities:
            self._setup_capabilities(capabilities)
        
        # Setup memory
        self._setup_memory(memory, store)
        
        # Setup scratchpad (working memory for todos, notes, errors)
        self._scratchpad = Scratchpad()

    def _setup_resilience(self) -> None:
        """Setup resilience layer if configured."""
        from agenticflow.agent.resilience import (
            FallbackRegistry,
            ResilienceConfig,
            ToolResilience,
        )
        
        # Use explicit resilience config or create from legacy settings
        if self.config.resilience_config:
            resilience_config = self.config.resilience_config
        elif self.config.retry_on_error:
            # Create config from legacy settings
            from agenticflow.agent.resilience import RetryPolicy, RetryStrategy
            resilience_config = ResilienceConfig(
                retry_policy=RetryPolicy(
                    max_retries=self.config.max_retries,
                    strategy=RetryStrategy.EXPONENTIAL_JITTER,
                ),
                timeout_seconds=self.config.timeout_seconds,
            )
        else:
            resilience_config = ResilienceConfig.fast_fail()
        
        # Setup fallback registry
        fallback_registry = FallbackRegistry()
        for primary, fallbacks in self.config.fallback_tools.items():
            fallback_registry.register(primary, fallbacks)
        
        self._resilience = ToolResilience(
            config=resilience_config,
            fallback_registry=fallback_registry,
        )
    
    def _setup_capabilities(self, capabilities: Sequence[Any]) -> None:
        """Setup capabilities and extract their tools.
        
        Args:
            capabilities: List of BaseCapability instances.
        """
        from agenticflow.capabilities.base import BaseCapability
        
        for cap in capabilities:
            if not isinstance(cap, BaseCapability):
                raise TypeError(
                    f"Expected BaseCapability, got {type(cap).__name__}. "
                    "Capabilities must extend BaseCapability."
                )
            
            self._capabilities.append(cap)
            
            # Add capability's tools to agent's direct tools
            for tool in cap.tools:
                if tool not in self._direct_tools:
                    self._direct_tools.append(tool)
                    # Also update config.tools for consistency
                    if tool.name not in self.config.tools:
                        self.config.tools.append(tool.name)
    
    async def initialize_capabilities(self) -> None:
        """Initialize all capabilities (call after agent is fully set up).
        
        This is called automatically by Flow, but can be called manually
        for standalone agents with capabilities.
        """
        for cap in self._capabilities:
            await cap.initialize(self)
    
    async def shutdown_capabilities(self) -> None:
        """Shutdown all capabilities (cleanup resources)."""
        for cap in self._capabilities:
            await cap.shutdown()
    
    @property
    def capabilities(self) -> list[Any]:
        """List of capabilities attached to this agent."""
        return self._capabilities.copy()
    
    def get_capability(self, name: str) -> Any | None:
        """Get a capability by name."""
        for cap in self._capabilities:
            if cap.name == name:
                return cap
        return None
    
    def _setup_memory(self, memory: Any = None, store: Any = None) -> None:
        """Setup memory manager.
        
        Args:
            memory: Memory backend - can be:
                - None: No persistence (in-memory only for session)
                - True: Use built-in InMemorySaver
                - LangGraph saver (MemorySaver, SqliteSaver, etc.)
                - AgentMemory instance
            store: LangGraph BaseStore for long-term memory.
        """
        from agenticflow.agent.memory import AgentMemory, InMemorySaver
        
        # Handle different memory input types
        if isinstance(memory, AgentMemory):
            # Already an AgentMemory instance
            self._memory = memory
        elif memory is True:
            # Convenience: True means use built-in in-memory
            self._memory = AgentMemory(
                backend=InMemorySaver(),
                store=store,
            )
        elif memory is not None:
            # Assume it's a LangGraph saver
            self._memory = AgentMemory(
                backend=memory,
                store=store,
            )
        else:
            # No persistence
            self._memory = AgentMemory(store=store)
    
    @property
    def memory(self):
        """Access the agent's memory manager.
        
        Returns:
            AgentMemory instance for managing conversation history and long-term memory.
            
        Example:
            ```python
            # Get messages from a thread
            messages = await agent.memory.get_messages(thread_id="conv-1")
            
            # Store long-term memory
            await agent.memory.remember("user_preference", {"theme": "dark"})
            
            # Recall long-term memory
            prefs = await agent.memory.recall("user_preference")
            ```
        """
        return self._memory
    
    @property
    def has_memory(self) -> bool:
        """Whether the agent has a memory persistence backend configured."""
        return self._memory.has_persistence
    
    @property
    def scratchpad(self) -> Scratchpad:
        """Access the agent's scratchpad (working memory).
        
        The scratchpad provides:
        - Todo list: Track tasks with status
        - Notes: Store observations and insights
        - Error log: Context for self-correction
        
        Example:
            ```python
            # Track todos
            agent.scratchpad.add_todo("Search for Python tutorials")
            agent.scratchpad.mark_done("Search", "Found 5 articles")
            
            # Take notes
            agent.scratchpad.note("Python is popular for data science")
            
            # Check completion
            if agent.scratchpad.all_done():
                print("All tasks complete!")
            ```
        """
        return self._scratchpad
    
    @property
    def resilience(self) -> ToolResilience:
        """Access the resilience layer for this agent."""
        if self._resilience is None:
            self._setup_resilience()
        return self._resilience
    
    @property
    def name(self) -> str:
        """Agent's display name."""
        return self.config.name

    @property
    def role(self) -> AgentRole:
        """Agent's role in the system."""
        return self.config.role

    @property
    def role_behavior(self):
        """Get behavior definition for this agent's role."""
        from agenticflow.agent.roles import RoleBehavior, get_role_behavior
        return get_role_behavior(self.role)

    @property
    def can_delegate(self) -> bool:
        """Whether this agent can delegate to other agents."""
        return self.role_behavior.can_delegate

    @property
    def can_finish(self) -> bool:
        """Whether this agent can produce final answers."""
        return self.role_behavior.can_finish

    @property
    def can_use_tools(self) -> bool:
        """Whether this agent can use tools directly."""
        return self.role_behavior.can_use_tools

    @property
    def status(self) -> AgentStatus:
        """Current agent status."""
        return self.state.status

    @property
    def direct_tools(self) -> list[BaseTool]:
        """Get tools passed directly to the agent (not via registry)."""
        return self._direct_tools

    @property
    def all_tools(self) -> list[BaseTool]:
        """Get all available tools (direct + registry).
        
        Returns tools in priority order:
        1. Direct tools (passed to Agent constructor)
        2. Registry tools (if tool_registry is set)
        """
        tools: list[BaseTool] = list(self._direct_tools)
        
        # Add registry tools that aren't already in direct tools
        if self.tool_registry:
            direct_names = {t.name for t in tools}
            for name in self.config.tools:
                if name not in direct_names:
                    tool = self.tool_registry.get(name)
                    if tool:
                        tools.append(tool)
        
        return tools

    def _get_tool(self, tool_name: str) -> BaseTool | None:
        """Get a tool by name from direct tools or registry.
        
        Args:
            tool_name: Name of the tool to retrieve.
            
        Returns:
            The tool object, or None if not found.
        """
        # Check direct tools first (higher priority)
        for tool in self._direct_tools:
            if tool.name == tool_name:
                return tool
        
        # Fall back to registry
        if self.tool_registry:
            return self.tool_registry.get(tool_name)
        
        return None

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools.
        
        Returns:
            Formatted string with tool names and descriptions.
        """
        tools = self.all_tools
        if not tools:
            return "No tools available."
        
        descriptions = []
        for tool in tools:
            desc = getattr(tool, "description", "No description")
            descriptions.append(f"- {tool.name}: {desc}")
        
        return "\n".join(descriptions)

    def get_capabilities_description(self) -> str:
        """Get formatted descriptions of all capabilities and their tools.
        
        Returns:
            Formatted string describing each capability and its tools.
        """
        if not self._capabilities:
            return ""
        
        sections = []
        for cap in self._capabilities:
            cap_tools = cap.tools
            if cap_tools:
                tool_list = "\n".join(f"  - {t.name}: {t.description}" for t in cap_tools)
                sections.append(f"**{cap.name}** - {cap.description}\n{tool_list}")
            else:
                sections.append(f"**{cap.name}** - {cap.description}")
        
        return "\n\n".join(sections)

    def get_effective_system_prompt(self) -> str | None:
        """Get the system prompt with capabilities and tools automatically injected.
        
        Supports placeholders:
        - {tools}: Replaced with tool descriptions
        - {capabilities}: Replaced with capability descriptions
        
        If no placeholders are present and tools exist, they are appended.
        If no system prompt but tools exist, generates a minimal prompt with tools.
        
        Returns:
            The system prompt with tools/capabilities injected, or None if no prompt and no tools.
        """
        base_prompt = self.config.system_prompt
        tools = self.all_tools
        tools_desc = self.get_tool_descriptions()
        caps_desc = self.get_capabilities_description()
        
        # If no base prompt but we have tools, generate a minimal prompt
        if not base_prompt:
            if tools or caps_desc:
                from agenticflow.agent.roles import get_role_prompt
                base_prompt = get_role_prompt(self.config.role)
            else:
                return None
        
        result = base_prompt
        
        # Replace placeholders if present
        has_tools_placeholder = "{tools}" in result
        has_caps_placeholder = "{capabilities}" in result
        
        if has_tools_placeholder:
            result = result.replace("{tools}", tools_desc)
        
        if has_caps_placeholder:
            result = result.replace("{capabilities}", caps_desc if caps_desc else "No capabilities.")
        
        # Auto-append if no placeholders and we have tools/capabilities
        if not has_tools_placeholder and not has_caps_placeholder:
            appendix_parts = []
            
            if caps_desc:
                appendix_parts.append(f"## Capabilities\n\n{caps_desc}")
            elif tools:
                # Only show flat tool list if no capabilities (to avoid duplication)
                appendix_parts.append(f"## Available Tools\n\n{tools_desc}")
            
            if appendix_parts:
                result = f"{result}\n\n" + "\n\n".join(appendix_parts)
        
        return result

    @property
    def instructions(self) -> str | None:
        """Agent's instructions (system prompt)."""
        return self.config.system_prompt
    
    @instructions.setter
    def instructions(self, value: str | None) -> None:
        """Set agent's instructions."""
        self.config = AgentConfig(
            name=self.config.name,
            role=self.config.role,
            description=self.config.description,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            system_prompt=value,
            model_kwargs=self.config.model_kwargs.copy(),
            tools=self.config.tools.copy(),
            max_concurrent_tasks=self.config.max_concurrent_tasks,
            timeout_seconds=self.config.timeout_seconds,
            retry_on_error=self.config.retry_on_error,
            max_retries=self.config.max_retries,
            resilience_config=self.config.resilience_config,
            fallback_tools=self.config.fallback_tools.copy(),
            metadata=self.config.metadata.copy(),
        )

    @property
    def model(self) -> BaseChatModel | None:
        """Get the LLM model.
        
        The model should be passed directly as a LangChain BaseChatModel
        instance in the AgentConfig.
        
        Example:
            ```python
            from langchain_openai import ChatOpenAI
            
            model = ChatOpenAI(model="gpt-4o")
            config = AgentConfig(name="Agent", model=model)
            ```
        """
        if self._model is None:
            model_spec = self.config.effective_model
            if model_spec is not None:
                if isinstance(model_spec, BaseChatModel):
                    self._model = model_spec
                else:
                    raise TypeError(
                        f"model must be a LangChain BaseChatModel instance, "
                        f"got {type(model_spec).__name__}. "
                        f"Use: from langchain_openai import ChatOpenAI; "
                        f"model = ChatOpenAI(model='gpt-4o')"
                    )
        return self._model

    async def _set_status(
        self,
        status: AgentStatus,
        correlation_id: str | None = None,
    ) -> None:
        """Update agent status and emit event."""
        old_status = self.state.status
        self.state.status = status
        self.state.record_activity()

        if self.event_bus:
            await self.event_bus.publish(
                Event(
                    type=EventType.AGENT_STATUS_CHANGED,
                    data={
                        "agent_id": self.id,
                        "agent_name": self.name,
                        "old_status": old_status.value,
                        "new_status": status.value,
                    },
                    source=f"agent:{self.id}",
                    correlation_id=correlation_id,
                )
            )

    async def _emit_event(
        self,
        event_type: EventType,
        data: dict,
        correlation_id: str | None = None,
    ) -> None:
        """Emit an event from this agent."""
        if self.event_bus:
            await self.event_bus.publish(
                Event(
                    type=event_type,
                    data=data,
                    source=f"agent:{self.id}",
                    correlation_id=correlation_id,
                )
            )

    def think(
        self,
        prompt: str,
        correlation_id: str | None = None,
        *,
        stream: bool | None = None,
        include_tools: bool = True,
        system_prompt_override: str | None = None,
    ) -> "str | AsyncIterator[StreamChunk]":
        """
        Process a prompt through the agent's reasoning.
        
        This is the agent's "thinking" phase where it uses its LLM
        to reason about the input and determine what to do.
        
        Args:
            prompt: The prompt to process
            correlation_id: Optional correlation ID for event tracking
            stream: Enable streaming response. If None, uses agent's default
                (set via stream=True in constructor). When True, returns an
                async iterator of StreamChunk objects.
            include_tools: Whether to include tools in system prompt (default: True).
                Set to False for planning prompts that already list tools.
            system_prompt_override: If provided, use this instead of agent's system prompt.
                Useful for planning where we need a neutral persona.
            
        Returns:
            If stream=False: Coroutine that returns the LLM's response string.
            If stream=True: AsyncIterator[StreamChunk] for token-by-token streaming.
            
        Raises:
            RuntimeError: If no model is configured
            
        Example - Non-streaming:
            ```python
            response = await agent.think("What should I do?")
            ```
            
        Example - Streaming:
            ```python
            async for chunk in agent.think("Write a poem", stream=True):
                print(chunk.content, end="", flush=True)
            ```
        """
        # Determine if streaming based on parameter or agent default
        use_streaming = stream if stream is not None else self.config.stream
        
        if use_streaming:
            # Return async iterator directly (not wrapped in coroutine)
            return self.think_stream(
                prompt,
                correlation_id=correlation_id,
                include_tools=include_tools,
                system_prompt_override=system_prompt_override,
            )
        
        # Return coroutine for non-streaming
        return self._think_impl(
            prompt,
            correlation_id=correlation_id,
            include_tools=include_tools,
            system_prompt_override=system_prompt_override,
        )
    
    async def _think_impl(
        self,
        prompt: str,
        correlation_id: str | None = None,
        *,
        include_tools: bool = True,
        system_prompt_override: str | None = None,
    ) -> str:
        """Internal implementation of non-streaming think."""
        if not self.model:
            raise RuntimeError(f"Agent {self.name} has no model configured")

        await self._set_status(AgentStatus.THINKING, correlation_id)

        await self._emit_event(
            EventType.AGENT_THINKING,
            {
                "agent_id": self.id,
                "agent_name": self.name,
                "prompt_preview": prompt[:200],
            },
            correlation_id,
        )

        start_time = now_utc()

        # Build messages - determine which system prompt to use
        messages = []
        if system_prompt_override is not None:
            effective_prompt = system_prompt_override
        elif include_tools:
            effective_prompt = self.get_effective_system_prompt()
        else:
            effective_prompt = self.config.system_prompt
        if effective_prompt:
            messages.append(SystemMessage(content=effective_prompt))

        # Add relevant history
        messages.extend(self.state.get_recent_history(10))
        messages.append(HumanMessage(content=prompt))

        try:
            response = await self.model.ainvoke(messages)
            result = response.content

            # Track timing
            duration_ms = (now_utc() - start_time).total_seconds() * 1000
            self.state.add_thinking_time(duration_ms)

            # Update history
            self.state.add_message(HumanMessage(content=prompt))
            self.state.add_message(AIMessage(content=result))

            await self._set_status(AgentStatus.IDLE, correlation_id)
            
            # Emit response event so observer can show the thought
            await self._emit_event(
                EventType.AGENT_RESPONDED,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "response": result,
                    "response_preview": result[:500] if result else "",
                    "duration_ms": duration_ms,
                },
                correlation_id,
            )
            
            return result

        except Exception as e:
            self.state.record_error(str(e))
            await self._set_status(AgentStatus.ERROR, correlation_id)
            await self._emit_event(
                EventType.AGENT_ERROR,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "error": str(e),
                    "phase": "thinking",
                },
                correlation_id,
            )
            raise

    def chat(
        self,
        message: str,
        thread_id: str | None = None,
        *,
        stream: bool | None = None,
        correlation_id: str | None = None,
        metadata: dict | None = None,
    ) -> "str | AsyncIterator[StreamChunk]":
        """
        Chat with the agent, maintaining conversation history per thread.
        
        This is the recommended way to have multi-turn conversations with
        an agent. Each thread_id maintains its own conversation history,
        enabling context-aware responses.
        
        Requires a checkpointer to be configured for cross-session persistence.
        Without a checkpointer, history is only maintained in-memory for the
        current session.
        
        Args:
            message: The user's message.
            thread_id: Unique identifier for the conversation thread.
                If None, uses a default thread for this agent.
            stream: Enable streaming response. If None, uses agent's default
                (set via stream=True in constructor). When True, returns an
                async iterator of StreamChunk objects.
            correlation_id: Optional correlation ID for event tracking.
            metadata: Optional metadata to store with the conversation.
            
        Returns:
            If stream=False: Coroutine that returns the agent's response string.
            If stream=True: AsyncIterator[StreamChunk] for token-by-token streaming.
            
        Example - Non-streaming:
            ```python
            agent = Agent(name="Assistant", model=model)
            response = await agent.chat("Hello!")
            print(response)
            ```
            
        Example - Streaming (explicit):
            ```python
            agent = Agent(name="Assistant", model=model)
            async for chunk in agent.chat("Tell me a story", stream=True):
                print(chunk.content, end="", flush=True)
            ```
            
        Example - Streaming (default):
            ```python
            # Set stream=True as default for the agent
            agent = Agent(name="Assistant", model=model, stream=True)
            
            async for chunk in agent.chat("Tell me a story"):
                print(chunk.content, end="", flush=True)
            ```
            
        Example - Multi-turn with memory:
            ```python
            agent = Agent(name="Assistant", model=model, memory=True)
            
            await agent.chat("I'm Alice", thread_id="user-123")
            response = await agent.chat("What's my name?", thread_id="user-123")
            # Response will mention Alice!
            ```
        """
        # Determine if streaming based on parameter or agent default
        use_streaming = stream if stream is not None else self.config.stream
        
        if use_streaming:
            # Return the streaming iterator directly (not wrapped in coroutine)
            return self.chat_stream(
                message,
                thread_id=thread_id,
                correlation_id=correlation_id,
                metadata=metadata,
            )
        
        # Return coroutine for non-streaming
        return self._chat_impl(
            message,
            thread_id=thread_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )
    
    async def _chat_impl(
        self,
        message: str,
        thread_id: str | None = None,
        *,
        correlation_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Internal implementation of non-streaming chat."""
        if not self.model:
            raise RuntimeError(f"Agent {self.name} has no model configured")
        
        # Default thread ID based on agent
        if thread_id is None:
            thread_id = f"agent-{self.id}-default"
        
        await self._set_status(AgentStatus.THINKING, correlation_id)
        
        await self._emit_event(
            EventType.AGENT_THINKING,
            {
                "agent_id": self.id,
                "agent_name": self.name,
                "thread_id": thread_id,
                "prompt_preview": message[:200],
            },
            correlation_id,
        )
        
        start_time = now_utc()
        
        try:
            # Load conversation history from memory
            history = await self._memory.get_messages(thread_id)
            
            # Build messages with effective system prompt (includes tools)
            messages = []
            effective_prompt = self.get_effective_system_prompt()
            if effective_prompt:
                messages.append(SystemMessage(content=effective_prompt))
            
            # Add conversation history
            messages.extend(history)
            
            # Add new user message
            user_message = HumanMessage(content=message)
            messages.append(user_message)
            
            # Get response from model
            response = await self.model.ainvoke(messages)
            result = response.content
            
            # Track timing
            duration_ms = (now_utc() - start_time).total_seconds() * 1000
            self.state.add_thinking_time(duration_ms)
            
            # Save messages to memory (both user and assistant)
            ai_message = AIMessage(content=result)
            await self._memory.add_messages(
                thread_id=thread_id,
                messages=[user_message, ai_message],
                metadata={
                    **(metadata or {}),
                    "agent_name": self.name,
                    "agent_id": self.id,
                },
            )
            
            # Also update local state
            self.state.add_message(user_message)
            self.state.add_message(ai_message)
            
            await self._set_status(AgentStatus.IDLE, correlation_id)
            
            await self._emit_event(
                EventType.AGENT_RESPONDED,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "thread_id": thread_id,
                    "response_preview": result[:200] if result else "",
                    "duration_ms": duration_ms,
                },
                correlation_id,
            )
            
            return result
            
        except Exception as e:
            self.state.record_error(str(e))
            await self._set_status(AgentStatus.ERROR, correlation_id)
            await self._emit_event(
                EventType.AGENT_ERROR,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "thread_id": thread_id,
                    "error": str(e),
                    "phase": "chat",
                },
                correlation_id,
            )
            raise

    async def get_thread_history(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> list:
        """
        Get conversation history for a thread.
        
        Args:
            thread_id: Thread identifier.
            limit: Maximum number of messages to return.
            
        Returns:
            List of messages in the thread.
        """
        return await self._memory.get_messages(thread_id, limit=limit)
    
    async def clear_thread(self, thread_id: str) -> None:
        """
        Clear conversation history for a thread.
        
        Args:
            thread_id: Thread identifier to clear.
        """
        await self._memory.clear_thread(thread_id)

    # =========================================================================
    # STREAMING METHODS
    # =========================================================================

    async def think_stream(
        self,
        prompt: str,
        correlation_id: str | None = None,
        *,
        include_tools: bool = True,
        system_prompt_override: str | None = None,
    ) -> "AsyncIterator[StreamChunk]":
        """
        Stream the agent's thinking response token by token.
        
        This is the streaming version of think(). Tokens are yielded
        as they are generated by the LLM, enabling real-time display.
        
        Args:
            prompt: The prompt to process.
            correlation_id: Optional correlation ID for event tracking.
            include_tools: Whether to include tools in system prompt.
            system_prompt_override: Optional system prompt override.
            
        Yields:
            StreamChunk objects containing token content.
            
        Raises:
            RuntimeError: If no model is configured.
            
        Example:
            ```python
            async for chunk in agent.think_stream("Write a poem"):
                print(chunk.content, end="", flush=True)
            print()  # Newline after streaming
            ```
        """
        from agenticflow.agent.streaming import (
            StreamChunk,
            chunk_from_langchain,
            extract_tool_calls,
        )
        
        if not self.model:
            raise RuntimeError(f"Agent {self.name} has no model configured")

        await self._set_status(AgentStatus.THINKING, correlation_id)

        await self._emit_event(
            EventType.AGENT_THINKING,
            {
                "agent_id": self.id,
                "agent_name": self.name,
                "prompt_preview": prompt[:200],
                "streaming": True,
            },
            correlation_id,
        )

        start_time = now_utc()

        # Build messages
        messages = []
        if system_prompt_override is not None:
            effective_prompt = system_prompt_override
        elif include_tools:
            effective_prompt = self.get_effective_system_prompt()
        else:
            effective_prompt = self.config.system_prompt
        if effective_prompt:
            messages.append(SystemMessage(content=effective_prompt))

        messages.extend(self.state.get_recent_history(10))
        messages.append(HumanMessage(content=prompt))

        accumulated_content = ""
        index = 0

        try:
            async for chunk in self.model.astream(messages):
                stream_chunk = chunk_from_langchain(chunk, index)
                accumulated_content += stream_chunk.content
                index += 1
                
                # Emit token event
                if stream_chunk.content:
                    await self._emit_event(
                        EventType.TOKEN_STREAMED,
                        {
                            "agent_id": self.id,
                            "agent_name": self.name,
                            "token": stream_chunk.content,
                            "index": index,
                        },
                        correlation_id,
                    )
                
                yield stream_chunk

            # Track timing
            duration_ms = (now_utc() - start_time).total_seconds() * 1000
            self.state.add_thinking_time(duration_ms)

            # Update history with final result
            self.state.add_message(HumanMessage(content=prompt))
            self.state.add_message(AIMessage(content=accumulated_content))

            await self._set_status(AgentStatus.IDLE, correlation_id)

            await self._emit_event(
                EventType.AGENT_RESPONDED,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "response": accumulated_content,
                    "response_preview": accumulated_content[:500] if accumulated_content else "",
                    "duration_ms": duration_ms,
                    "streaming": True,
                    "token_count": index,
                },
                correlation_id,
            )

        except Exception as e:
            self.state.record_error(str(e))
            await self._set_status(AgentStatus.ERROR, correlation_id)
            await self._emit_event(
                EventType.AGENT_ERROR,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "error": str(e),
                    "phase": "thinking_stream",
                },
                correlation_id,
            )
            raise

    async def chat_stream(
        self,
        message: str,
        thread_id: str | None = None,
        *,
        correlation_id: str | None = None,
        metadata: dict | None = None,
    ) -> "AsyncIterator[StreamChunk]":
        """
        Stream a chat response token by token with conversation history.
        
        This is the streaming version of chat(). Maintains conversation
        history per thread while yielding tokens as they arrive.
        
        Args:
            message: The user's message.
            thread_id: Unique identifier for the conversation thread.
            correlation_id: Optional correlation ID for event tracking.
            metadata: Optional metadata to store with the conversation.
            
        Yields:
            StreamChunk objects containing token content.
            
        Example:
            ```python
            async for chunk in agent.chat_stream("Tell me a story", thread_id="user-123"):
                print(chunk.content, end="", flush=True)
            print()
            ```
        """
        from agenticflow.agent.streaming import StreamChunk, chunk_from_langchain
        
        if not self.model:
            raise RuntimeError(f"Agent {self.name} has no model configured")

        if thread_id is None:
            thread_id = f"agent-{self.id}-default"

        await self._set_status(AgentStatus.THINKING, correlation_id)

        await self._emit_event(
            EventType.AGENT_THINKING,
            {
                "agent_id": self.id,
                "agent_name": self.name,
                "thread_id": thread_id,
                "prompt_preview": message[:200],
                "streaming": True,
            },
            correlation_id,
        )

        start_time = now_utc()

        try:
            # Load conversation history
            history = await self._memory.get_messages(thread_id)

            # Build messages
            messages = []
            effective_prompt = self.get_effective_system_prompt()
            if effective_prompt:
                messages.append(SystemMessage(content=effective_prompt))
            messages.extend(history)
            
            user_message = HumanMessage(content=message)
            messages.append(user_message)

            accumulated_content = ""
            index = 0

            async for chunk in self.model.astream(messages):
                stream_chunk = chunk_from_langchain(chunk, index)
                accumulated_content += stream_chunk.content
                index += 1
                
                if stream_chunk.content:
                    await self._emit_event(
                        EventType.TOKEN_STREAMED,
                        {
                            "agent_id": self.id,
                            "agent_name": self.name,
                            "thread_id": thread_id,
                            "token": stream_chunk.content,
                            "index": index,
                        },
                        correlation_id,
                    )
                
                yield stream_chunk

            # Track timing
            duration_ms = (now_utc() - start_time).total_seconds() * 1000
            self.state.add_thinking_time(duration_ms)

            # Save to memory
            ai_message = AIMessage(content=accumulated_content)
            await self._memory.add_messages(
                thread_id=thread_id,
                messages=[user_message, ai_message],
                metadata={
                    **(metadata or {}),
                    "agent_name": self.name,
                    "agent_id": self.id,
                },
            )

            # Update local state
            self.state.add_message(user_message)
            self.state.add_message(ai_message)

            await self._set_status(AgentStatus.IDLE, correlation_id)

            await self._emit_event(
                EventType.AGENT_RESPONDED,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "thread_id": thread_id,
                    "response_preview": accumulated_content[:200] if accumulated_content else "",
                    "duration_ms": duration_ms,
                    "streaming": True,
                    "token_count": index,
                },
                correlation_id,
            )

        except Exception as e:
            self.state.record_error(str(e))
            await self._set_status(AgentStatus.ERROR, correlation_id)
            await self._emit_event(
                EventType.AGENT_ERROR,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "thread_id": thread_id,
                    "error": str(e),
                    "phase": "chat_stream",
                },
                correlation_id,
            )
            raise

    async def stream_events(
        self,
        prompt: str,
        correlation_id: str | None = None,
        *,
        include_tools: bool = True,
        system_prompt_override: str | None = None,
    ) -> "AsyncIterator[StreamEvent]":
        """
        Stream structured events during agent thinking.
        
        This provides more detailed events than think_stream(), including:
        - STREAM_START: Beginning of stream with metadata
        - TOKEN: Each token as it arrives
        - TOOL_CALL_START: When a tool call begins
        - TOOL_CALL_ARGS: Tool call argument chunks
        - TOOL_CALL_END: Tool call complete
        - STREAM_END: End of stream with full content
        - ERROR: If an error occurs
        
        Args:
            prompt: The prompt to process.
            correlation_id: Optional correlation ID for event tracking.
            include_tools: Whether to include tools in system prompt.
            system_prompt_override: Optional system prompt override.
            
        Yields:
            StreamEvent objects with structured information.
            
        Example:
            ```python
            async for event in agent.stream_events("Search for Python tutorials"):
                if event.type == StreamEventType.TOKEN:
                    print(event.content, end="", flush=True)
                elif event.type == StreamEventType.TOOL_CALL_START:
                    print(f"\\n[Calling {event.tool_name}...]")
            ```
        """
        from agenticflow.agent.streaming import (
            StreamEvent,
            StreamEventType,
            chunk_from_langchain,
            extract_tool_calls,
            ToolCallChunk,
        )
        
        if not self.model:
            raise RuntimeError(f"Agent {self.name} has no model configured")

        await self._set_status(AgentStatus.THINKING, correlation_id)

        start_time = now_utc()

        # Build messages
        messages = []
        if system_prompt_override is not None:
            effective_prompt = system_prompt_override
        elif include_tools:
            effective_prompt = self.get_effective_system_prompt()
        else:
            effective_prompt = self.config.system_prompt
        if effective_prompt:
            messages.append(SystemMessage(content=effective_prompt))

        messages.extend(self.state.get_recent_history(10))
        messages.append(HumanMessage(content=prompt))

        # Emit start event
        yield StreamEvent(
            type=StreamEventType.STREAM_START,
            metadata={
                "agent_id": self.id,
                "agent_name": self.name,
                "prompt": prompt[:200],
            },
            index=0,
        )

        accumulated_content = ""
        index = 1
        active_tool_calls: dict[int, ToolCallChunk] = {}

        try:
            async for chunk in self.model.astream(messages):
                stream_chunk = chunk_from_langchain(chunk, index)
                
                # Handle text content
                if stream_chunk.content:
                    accumulated_content += stream_chunk.content
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        content=stream_chunk.content,
                        accumulated=accumulated_content,
                        index=index,
                    )
                    index += 1
                
                # Handle tool calls
                tool_chunks = extract_tool_calls(chunk)
                for tc in tool_chunks:
                    tc_index = tc.index
                    
                    if tc_index not in active_tool_calls:
                        # New tool call
                        active_tool_calls[tc_index] = tc
                        if tc.name:
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call=tc,
                                tool_name=tc.name,
                                index=index,
                            )
                            index += 1
                    else:
                        # Update existing tool call
                        existing = active_tool_calls[tc_index]
                        if tc.args:
                            existing.args += tc.args
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_ARGS,
                                tool_call=existing,
                                tool_name=existing.name,
                                index=index,
                            )
                            index += 1

            # Emit tool call end events
            for tc in active_tool_calls.values():
                yield StreamEvent(
                    type=StreamEventType.TOOL_CALL_END,
                    tool_call=tc,
                    tool_name=tc.name,
                    index=index,
                )
                index += 1

            # Track timing
            duration_ms = (now_utc() - start_time).total_seconds() * 1000
            self.state.add_thinking_time(duration_ms)

            # Update history
            self.state.add_message(HumanMessage(content=prompt))
            self.state.add_message(AIMessage(content=accumulated_content))

            await self._set_status(AgentStatus.IDLE, correlation_id)

            # Emit end event
            yield StreamEvent(
                type=StreamEventType.STREAM_END,
                accumulated=accumulated_content,
                metadata={
                    "duration_ms": duration_ms,
                    "token_count": index,
                    "tool_calls": len(active_tool_calls),
                },
                index=index,
            )

        except Exception as e:
            self.state.record_error(str(e))
            await self._set_status(AgentStatus.ERROR, correlation_id)
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error=str(e),
                accumulated=accumulated_content,
                index=index,
            )
            raise

    # =========================================================================
    # TOOL EXECUTION (ACT)
    # =========================================================================

    async def act(
        self,
        tool_name: str,
        args: dict,
        correlation_id: str | None = None,
        tracker: ProgressTracker | None = None,
        use_resilience: bool = True,
    ) -> Any:
        """
        Execute an action using a tool with intelligent retry and recovery.
        
        This is the agent's "acting" phase where it interacts with
        the world through tools. When use_resilience=True (default),
        the agent will:
        - Retry on transient failures with exponential backoff
        - Use circuit breakers to prevent cascading failures
        - Fall back to alternative tools if configured
        - Learn from failures to adapt future behavior
        
        Args:
            tool_name: Name of the tool to use
            args: Arguments for the tool
            correlation_id: Optional correlation ID for event tracking
            tracker: Optional progress tracker for real-time output
            use_resilience: Whether to use intelligent retry/recovery (default: True)
            
        Returns:
            The tool's result
            
        Raises:
            RuntimeError: If no tool registry is configured
            ValueError: If tool is not found
            PermissionError: If agent is not authorized to use the tool
            
        Example:
            ```python
            # With full resilience (retry, circuit breaker, fallback)
            result = await agent.act("web_search", {"query": "Python async"})
            
            # Disable resilience for testing
            result = await agent.act("web_search", {"query": "test"}, use_resilience=False)
            ```
        """
        tool_obj = self._get_tool(tool_name)
        if not tool_obj:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {[t.name for t in self.all_tools]}")

        # Check if agent is allowed to use this tool
        if not self.config.can_use_tool(tool_name):
            raise PermissionError(
                f"Agent {self.name} not authorized to use tool: {tool_name}"
            )
        
        # Check for human-in-the-loop interrupt
        if should_interrupt(tool_name, args, self.config.interrupt_on):
            action_id = str(uuid.uuid4())
            pending = PendingAction(
                action_id=action_id,
                tool_name=tool_name,
                args=args,
                agent_name=self.name,
                reason=InterruptReason.TOOL_APPROVAL,
                context={
                    "correlation_id": correlation_id,
                    "use_resilience": use_resilience,
                },
            )
            self._pending_actions[action_id] = pending
            
            await self._emit_event(
                EventType.AGENT_INTERRUPTED,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "reason": "tool_approval",
                    "pending_action": pending.to_dict(),
                },
                correlation_id,
            )
            
            if tracker:
                tracker.update(f"⏸️ Waiting for approval: {pending.describe()}")
            
            # Create interrupted state and raise exception
            interrupted = InterruptedState(
                thread_id=correlation_id or self.id,
                pending_actions=[pending],
                agent_state=self.state.to_dict(),
                interrupt_reason=InterruptReason.TOOL_APPROVAL,
            )
            self._interrupted_state = interrupted
            raise InterruptedException(interrupted, f"Tool '{tool_name}' requires human approval")

        await self._set_status(AgentStatus.ACTING, correlation_id)

        await self._emit_event(
            EventType.TOOL_CALLED,
            {
                "agent_id": self.id,
                "agent_name": self.name,
                "tool": tool_name,
                "args": args,
            },
            correlation_id,
        )
        
        # Emit to progress tracker if provided
        if tracker:
            tracker.tool_call(tool_name, args, agent=self.name)
            # Attach tracker to resilience layer for retry/fallback tracking
            if self._resilience:
                self._resilience.tracker = tracker

        start_time = now_utc()

        try:
            if use_resilience and self._resilience:
                # Use resilience layer for intelligent retry/recovery
                def get_fallback_tool(name: str):
                    t = self._get_tool(name)
                    return t.invoke if t else None
                
                exec_result = await self._resilience.execute(
                    tool_fn=tool_obj.invoke,
                    tool_name=tool_name,
                    args=args,
                    fallback_fn=get_fallback_tool,
                    tool_obj=tool_obj,  # Pass tool object for ainvoke support
                )
                
                if exec_result.success:
                    result = exec_result.result
                    duration_ms = exec_result.total_time_ms
                    
                    # Emit additional info if fallback was used
                    if exec_result.used_fallback:
                        await self._emit_event(
                            EventType.TOOL_RESULT,
                            {
                                "agent_id": self.id,
                                "tool": tool_name,
                                "actual_tool": exec_result.tool_used,
                                "used_fallback": True,
                                "fallback_chain": exec_result.fallback_chain,
                                "attempts": exec_result.attempts,
                                "result_preview": str(result)[:200],
                                "duration_ms": duration_ms,
                            },
                            correlation_id,
                        )
                        if tracker:
                            tracker.update(
                                f"✅ {tool_name} succeeded via fallback → {exec_result.tool_used}"
                            )
                    else:
                        await self._emit_event(
                            EventType.TOOL_RESULT,
                            {
                                "agent_id": self.id,
                                "tool": tool_name,
                                "attempts": exec_result.attempts,
                                "result_preview": str(result)[:200],
                                "duration_ms": duration_ms,
                            },
                            correlation_id,
                        )
                        if tracker and exec_result.attempts > 1:
                            tracker.update(
                                f"✅ {tool_name} succeeded after {exec_result.attempts} attempts"
                            )
                else:
                    # All retries and fallbacks exhausted
                    error = exec_result.error or RuntimeError(f"Tool {tool_name} failed")
                    
                    await self._emit_event(
                        EventType.TOOL_ERROR,
                        {
                            "agent_id": self.id,
                            "tool": tool_name,
                            "error": str(error),
                            "attempts": exec_result.attempts,
                            "fallback_chain": exec_result.fallback_chain,
                            "circuit_state": exec_result.circuit_state.value if exec_result.circuit_state else None,
                            "recovery_action": exec_result.recovery_action.value if exec_result.recovery_action else None,
                        },
                        correlation_id,
                    )
                    
                    if tracker:
                        tracker.tool_error(
                            tool_name,
                            f"{error} (after {exec_result.attempts} attempts, "
                            f"fallbacks: {exec_result.fallback_chain or 'none'})"
                        )
                    
                    self.state.record_error(str(error))
                    await self._set_status(AgentStatus.ERROR, correlation_id)
                    raise error
            else:
                # Direct execution without resilience - use ainvoke for async support
                if hasattr(tool_obj, "ainvoke"):
                    result = await tool_obj.ainvoke(args)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: tool_obj.invoke(args))
                duration_ms = (now_utc() - start_time).total_seconds() * 1000

            # Track timing
            self.state.add_acting_time(duration_ms)

            if tracker:
                tracker.tool_result(tool_name, str(result)[:200], duration_ms=duration_ms)

            await self._set_status(AgentStatus.IDLE, correlation_id)
            return result

        except Exception as e:
            # Only catch if not already handled by resilience layer
            if not (use_resilience and self._resilience):
                self.state.record_error(str(e))
                await self._set_status(AgentStatus.ERROR, correlation_id)
                await self._emit_event(
                    EventType.TOOL_ERROR,
                    {
                        "agent_id": self.id,
                        "tool": tool_name,
                        "error": str(e),
                    },
                    correlation_id,
                )
                
                if tracker:
                    tracker.tool_error(tool_name, str(e))
            
            raise

    async def resume_action(
        self,
        decision: HumanDecision,
        correlation_id: str | None = None,
        tracker: ProgressTracker | None = None,
        use_resilience: bool = True,
    ) -> Any:
        """
        Resume a pending action after human decision.
        
        When a tool call is interrupted for human approval, this method
        continues execution based on the human's decision.
        
        Args:
            decision: Human decision (approve, reject, edit, guide, respond, etc.)
            correlation_id: Optional correlation ID for event tracking
            tracker: Optional progress tracker
            use_resilience: Whether to use resilience (default: True)
            
        Returns:
            - Tool result if approved/edited
            - None if rejected/skipped
            - GuidanceResult if human provided guidance
            - HumanResponse if human provided direct response
            
        Raises:
            AbortedException: If human aborts the workflow
            ValueError: If action_id not found in pending actions
            
        Example:
            ```python
            try:
                result = await agent.act("delete_file", {"path": "/important.txt"})
            except InterruptedException as e:
                pending = e.state.pending_actions[0]
                
                # Option 1: Approve
                decision = HumanDecision.approve(pending.action_id)
                
                # Option 2: Provide guidance for agent to reconsider
                decision = HumanDecision.guide(
                    pending.action_id,
                    guidance="Archive the file first, then delete it."
                )
                
                result = await agent.resume_action(decision)
                if isinstance(result, GuidanceResult):
                    # Agent should reconsider with this guidance
                    print(f"Guidance: {result.guidance}")
            ```
        """
        from agenticflow.agent.hitl import GuidanceResult, HumanResponse
        
        action_id = decision.action_id
        
        if action_id not in self._pending_actions:
            raise ValueError(f"No pending action with id: {action_id}")
        
        pending = self._pending_actions.pop(action_id)
        
        await self._emit_event(
            EventType.AGENT_RESUMED,
            {
                "agent_id": self.id,
                "agent_name": self.name,
                "action_id": action_id,
                "decision": decision.decision.value,
                "tool": pending.tool_name,
                "has_guidance": decision.guidance is not None,
                "has_response": decision.response is not None,
            },
            correlation_id,
        )
        
        if tracker:
            tracker.update(f"▶️ Resumed: {decision.decision.value} - {pending.describe()}")
        
        # Handle decision types
        if decision.decision == DecisionType.ABORT:
            self._interrupted_state = None
            raise AbortedException(decision)
        
        if decision.decision == DecisionType.REJECT:
            if tracker:
                tracker.update(f"❌ Rejected: {pending.describe()}")
            return None
        
        if decision.decision == DecisionType.SKIP:
            if tracker:
                tracker.update(f"⏭️ Skipped: {pending.describe()}")
            return None
        
        if decision.decision == DecisionType.GUIDE:
            # Human provided guidance - return it for the agent to reconsider
            if tracker:
                tracker.update(f"💡 Guidance received: {decision.guidance[:100]}...")
            self._interrupted_state = None
            return GuidanceResult(
                action_id=action_id,
                guidance=decision.guidance or "",
                original_action=pending,
                feedback=decision.feedback,
                should_retry=True,
            )
        
        if decision.decision == DecisionType.RESPOND:
            # Human provided a direct response
            if tracker:
                tracker.update(f"💬 Response received: {str(decision.response)[:100]}")
            self._interrupted_state = None
            return HumanResponse(
                action_id=action_id,
                response=decision.response,
                original_action=pending,
                feedback=decision.feedback,
            )
        
        # APPROVE or EDIT - execute the tool
        args = decision.modified_args if decision.decision == DecisionType.EDIT else pending.args
        
        if decision.decision == DecisionType.EDIT:
            if tracker:
                tracker.update(f"✏️ Edited args: {args}")
        
        # Clear interrupted state since we're resuming
        self._interrupted_state = None
        
        # Execute the tool (bypass interrupt check since already approved)
        tool_obj = self._get_tool(pending.tool_name)
        if not tool_obj:
            raise ValueError(f"Tool no longer available: {pending.tool_name}")
        
        await self._set_status(AgentStatus.ACTING, correlation_id)
        
        await self._emit_event(
            EventType.TOOL_CALLED,
            {
                "agent_id": self.id,
                "agent_name": self.name,
                "tool": pending.tool_name,
                "args": args,
                "approved_by_human": True,
            },
            correlation_id,
        )
        
        if tracker:
            tracker.tool_call(pending.tool_name, args, agent=self.name)
        
        start_time = now_utc()
        
        try:
            if use_resilience and self._resilience:
                def get_fallback_tool(name: str):
                    t = self._get_tool(name)
                    return t.invoke if t else None
                
                exec_result = await self._resilience.execute(
                    tool_fn=tool_obj.invoke,
                    tool_name=pending.tool_name,
                    args=args,
                    fallback_fn=get_fallback_tool,
                    tool_obj=tool_obj,
                )
                
                if exec_result.success:
                    result = exec_result.result
                    duration_ms = exec_result.total_time_ms
                else:
                    error = exec_result.error or RuntimeError(f"Tool {pending.tool_name} failed")
                    raise error
            else:
                if hasattr(tool_obj, "ainvoke"):
                    result = await tool_obj.ainvoke(args)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: tool_obj.invoke(args))
                duration_ms = (now_utc() - start_time).total_seconds() * 1000
            
            self.state.add_acting_time(duration_ms)
            
            if tracker:
                tracker.tool_result(pending.tool_name, str(result)[:200], duration_ms=duration_ms)
            
            await self._set_status(AgentStatus.IDLE, correlation_id)
            return result
            
        except Exception as e:
            self.state.record_error(str(e))
            await self._set_status(AgentStatus.ERROR, correlation_id)
            if tracker:
                tracker.tool_error(pending.tool_name, str(e))
            raise
    
    @property
    def pending_actions(self) -> list[PendingAction]:
        """Get list of pending actions awaiting human decision."""
        return list(self._pending_actions.values())
    
    @property
    def is_interrupted(self) -> bool:
        """Check if agent has pending actions awaiting human decision."""
        return len(self._pending_actions) > 0
    
    def get_interrupted_state(self) -> InterruptedState | None:
        """Get the current interrupted state, if any."""
        return self._interrupted_state
    
    def clear_pending_actions(self) -> None:
        """Clear all pending actions (e.g., on abort or reset)."""
        self._pending_actions.clear()
        self._interrupted_state = None

    async def act_many(
        self,
        tool_calls: list[tuple[str, dict]],
        correlation_id: str | None = None,
        fail_fast: bool = False,
        tracker: ProgressTracker | None = None,
        use_resilience: bool = True,
    ) -> list[Any]:
        """
        Execute multiple tool calls in parallel with intelligent retry and recovery.
        
        This enables concurrent tool execution when an LLM requests
        multiple tools simultaneously. Results are returned in the
        same order as the input tool_calls. Each tool call uses the
        resilience layer independently for retry and fallback.
        
        Args:
            tool_calls: List of (tool_name, args) tuples
            correlation_id: Optional correlation ID for event tracking
            fail_fast: If True, cancel remaining on first error
            tracker: Optional progress tracker for real-time output
            use_resilience: Whether to use intelligent retry/recovery (default: True)
            
        Returns:
            List of results (or exceptions if fail_fast=False)
            
        Example:
            ```python
            results = await agent.act_many([
                ("search_web", {"query": "Python async"}),
                ("read_file", {"path": "data.json"}),
                ("calculate", {"expr": "2 + 2"}),
            ])
            ```
        """
        if not tool_calls:
            return []
        
        await self._set_status(AgentStatus.ACTING, correlation_id)
        
        await self._emit_event(
            EventType.TOOL_CALLED,
            {
                "agent_id": self.id,
                "agent_name": self.name,
                "tool": f"parallel:{len(tool_calls)} tools",
                "tools": [tc[0] for tc in tool_calls],
            },
            correlation_id,
        )
        
        # Emit to progress tracker
        if tracker:
            tracker.update(f"Executing {len(tool_calls)} tools in parallel")
            for tool_name, args in tool_calls:
                tracker.tool_call(tool_name, args, agent=self.name, parallel=True)
            # Attach tracker to resilience layer
            if self._resilience:
                self._resilience.tracker = tracker
        
        start_time = now_utc()
        
        async def execute_single(tool_name: str, args: dict) -> Any:
            """Execute a single tool with resilience (for parallel execution)."""
            tool_obj = self._get_tool(tool_name)
            if not tool_obj:
                raise ValueError(f"Unknown tool: {tool_name}. Available: {[t.name for t in self.all_tools]}")
            
            if not self.config.can_use_tool(tool_name):
                raise PermissionError(
                    f"Agent {self.name} not authorized to use tool: {tool_name}"
                )
            
            if use_resilience and self._resilience:
                # Use resilience layer
                def get_fallback_tool(name: str):
                    t = self._get_tool(name)
                    return t.invoke if t else None
                
                exec_result = await self._resilience.execute(
                    tool_fn=tool_obj.invoke,
                    tool_name=tool_name,
                    args=args,
                    fallback_fn=get_fallback_tool,
                    tool_obj=tool_obj,  # Pass tool object for ainvoke support
                )
                
                if exec_result.success:
                    return exec_result.result
                else:
                    raise exec_result.error or RuntimeError(f"Tool {tool_name} failed")
            else:
                # Direct execution - use ainvoke for async support
                if hasattr(tool_obj, "ainvoke"):
                    return await tool_obj.ainvoke(args)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, lambda: tool_obj.invoke(args))
        
        # Create tasks for parallel execution
        tasks = [execute_single(name, args) for name, args in tool_calls]
        
        try:
            if fail_fast:
                # Cancel all on first error
                results = await asyncio.gather(*tasks)
            else:
                # Return exceptions instead of raising
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Track timing
            duration_ms = (now_utc() - start_time).total_seconds() * 1000
            self.state.add_acting_time(duration_ms)
            
            # Count successes/failures
            successes = sum(1 for r in results if not isinstance(r, Exception))
            failures = len(results) - successes
            
            await self._emit_event(
                EventType.TOOL_RESULT,
                {
                    "agent_id": self.id,
                    "tools": [tc[0] for tc in tool_calls],
                    "successes": successes,
                    "failures": failures,
                    "duration_ms": duration_ms,
                },
                correlation_id,
            )
            
            # Emit results to progress tracker
            if tracker:
                for (tool_name, _), result in zip(tool_calls, results):
                    if isinstance(result, Exception):
                        tracker.tool_error(tool_name, str(result))
                    else:
                        tracker.tool_result(tool_name, str(result)[:100])
                tracker.update(f"Completed {successes}/{len(tool_calls)} tools ({duration_ms:.0f}ms)")
            
            await self._set_status(AgentStatus.IDLE, correlation_id)
            return results
            
        except Exception as e:
            self.state.record_error(str(e))
            await self._set_status(AgentStatus.ERROR, correlation_id)
            await self._emit_event(
                EventType.TOOL_ERROR,
                {
                    "agent_id": self.id,
                    "tools": [tc[0] for tc in tool_calls],
                    "error": str(e),
                },
                correlation_id,
            )
            
            # Emit error to progress tracker
            if tracker:
                tracker.tool_error("parallel_execution", str(e))
            
            raise

    async def execute_task(
        self,
        task: Task,
        correlation_id: str | None = None,
    ) -> Any:
        """
        Execute a task assigned to this agent.
        
        Handles the full task lifecycle:
        1. Mark task as started
        2. Think about the task (if no tool specified)
        3. Act using the specified tool (if any)
        4. Return the result
        
        Args:
            task: The task to execute
            correlation_id: Optional correlation ID for event tracking
            
        Returns:
            The task result
        """
        self.state.start_task(task.id)

        try:
            await self._emit_event(
                EventType.AGENT_INVOKED,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "task_id": task.id,
                    "task_name": task.name,
                },
                correlation_id,
            )

            result = None

            if task.tool:
                # Direct tool execution
                result = await self.act(task.tool, task.args, correlation_id)
            else:
                # Use LLM to process the task
                prompt = f"Task: {task.name}\n"
                if task.description:
                    prompt += f"Description: {task.description}\n"
                if task.args:
                    prompt += f"Context: {json.dumps(task.args)}\n"
                result = await self.think(prompt, correlation_id)

            self.state.finish_task(task.id, success=True)

            await self._emit_event(
                EventType.AGENT_RESPONDED,
                {
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "task_id": task.id,
                    "result_preview": str(result)[:200],
                },
                correlation_id,
            )

            return result

        except Exception as e:
            self.state.finish_task(task.id, success=False)
            raise

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        strategy: str = "dag",
        on_step: Callable[[str, Any], None] | None = None,
        tracker: ProgressTracker | None = None,
    ) -> Any:
        """
        Execute a complex task using an advanced execution strategy.
        
        This is the main entry point for complex multi-step tasks.
        It uses sophisticated execution patterns to maximize speed
        and efficiency.
        
        Args:
            task: The task description.
            context: Optional context dict.
            strategy: Execution strategy:
                - "react": Classic think-act-observe loop (slowest)
                - "plan": Plan all steps, then execute sequentially
                - "dag": Build dependency graph, execute in parallel (fastest)
                - "adaptive": Auto-select based on task complexity
            on_step: Optional callback(step_type, data) for progress.
            tracker: Optional ProgressTracker for real-time output.
            
        Returns:
            The final result.
            
        Example:
            ```python
            # Fast parallel execution with progress tracking
            tracker = ProgressTracker(OutputConfig.verbose())
            result = await agent.run(
                "Search for Python tutorials and Rust tutorials, then compare them",
                strategy="dag",
                tracker=tracker,
            )
            ```
        """
        from agenticflow.graphs import (
            ExecutionStrategy,
            create_executor,
        )
        
        strategy_map = {
            "react": ExecutionStrategy.REACT,
            "plan": ExecutionStrategy.PLAN_EXECUTE,
            "dag": ExecutionStrategy.DAG,
            "adaptive": ExecutionStrategy.ADAPTIVE,
        }
        
        exec_strategy = strategy_map.get(strategy, ExecutionStrategy.DAG)
        executor = create_executor(self, exec_strategy)
        executor.on_step = on_step
        executor.tracker = tracker  # Pass tracker to executor
        
        return await executor.execute(task, context)

    async def send_message(
        self,
        content: str,
        receiver_id: str | None = None,
        correlation_id: str | None = None,
    ) -> Message:
        """
        Send a message to another agent or broadcast.
        
        Args:
            content: Message content
            receiver_id: ID of receiving agent (None for broadcast)
            correlation_id: Optional correlation ID for event tracking
            
        Returns:
            The sent Message
        """
        message = Message(
            content=content,
            sender_id=self.id,
            receiver_id=receiver_id,
        )

        event_type = EventType.MESSAGE_BROADCAST if receiver_id is None else EventType.MESSAGE_SENT

        await self._emit_event(
            event_type,
            message.to_dict(),
            correlation_id,
        )

        return message

    async def receive_message(
        self,
        message: Message,
        correlation_id: str | None = None,
    ) -> str | None:
        """
        Receive and process a message.
        
        Args:
            message: The message to process
            correlation_id: Optional correlation ID for event tracking
            
        Returns:
            Response content, if any
        """
        await self._emit_event(
            EventType.MESSAGE_RECEIVED,
            {
                "agent_id": self.id,
                "message": message.to_dict(),
            },
            correlation_id,
        )

        # Process the message through thinking
        if self.model:
            response = await self.think(
                f"Message from {message.sender_id}: {message.content}",
                correlation_id,
            )
            return response

        return None

    def is_available(self) -> bool:
        """Check if agent can accept new work."""
        return (
            self.state.is_available()
            and self.state.has_capacity(self.config.max_concurrent_tasks)
        )
    
    # --- Resilience Control Methods ---
    
    def get_circuit_status(self, tool_name: str | None = None) -> dict[str, Any]:
        """
        Get circuit breaker status for tools.
        
        Args:
            tool_name: Specific tool name, or None for all tools
            
        Returns:
            Circuit breaker status dict
            
        Example:
            ```python
            # Check specific tool
            status = agent.get_circuit_status("web_search")
            if status["state"] == "open":
                print(f"web_search circuit is open!")
            
            # Check all tools
            all_status = agent.get_circuit_status()
            ```
        """
        if not self._resilience:
            return {"state": "no_resilience_configured"}
        
        if tool_name:
            return self._resilience.get_circuit_status(tool_name)
        return self._resilience.get_all_circuit_status()
    
    def reset_circuit(self, tool_name: str | None = None) -> None:
        """
        Reset circuit breaker(s) to closed state.
        
        Args:
            tool_name: Specific tool, or None to reset all
            
        Example:
            ```python
            # Reset specific tool
            agent.reset_circuit("web_search")
            
            # Reset all circuits
            agent.reset_circuit()
            ```
        """
        if not self._resilience:
            return
        
        if tool_name:
            self._resilience.reset_circuit(tool_name)
        else:
            self._resilience.reset_all_circuits()
    
    def get_failure_suggestions(self, tool_name: str) -> dict[str, Any] | None:
        """
        Get suggestions for a failing tool based on learned patterns.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Suggestions dict with failure_rate, common_errors, recommendations
            
        Example:
            ```python
            suggestions = agent.get_failure_suggestions("web_search")
            print(f"Failure rate: {suggestions['failure_rate']:.1%}")
            print(f"Common errors: {suggestions['common_errors']}")
            for rec in suggestions['recommendations']:
                print(f"  - {rec}")
            ```
        """
        if not self._resilience:
            return None
        return self._resilience.get_failure_suggestions(tool_name)
    
    def clear_failure_memory(self) -> None:
        """Clear all learned failure patterns (reset memory)."""
        if self._resilience and self._resilience.failure_memory:
            self._resilience.failure_memory.clear()

    def to_dict(self) -> dict:
        """Convert agent info to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "description": self.config.description,
            "tools": self.config.tools,
            "state": self.state.to_dict(),
        }

    def draw_mermaid(
        self,
        *,
        theme: str = "default",
        direction: str = "TB",
        title: str | None = None,
        show_tools: bool = True,
        show_roles: bool = True,
        show_config: bool = False,
    ) -> str:
        """
        Generate a Mermaid diagram showing this agent and its tools.
        
        Args:
            theme: Mermaid theme (default, forest, dark, neutral, base)
            direction: Graph direction (TB, TD, BT, LR, RL)
            title: Optional diagram title
            show_tools: Whether to show agent tools
            show_roles: Whether to show agent role
            show_config: Whether to show model configuration
            
        Returns:
            Mermaid diagram code as string
        """
        from agenticflow.visualization.mermaid import (
            AgentDiagram,
            MermaidConfig,
            MermaidDirection,
            MermaidTheme,
        )
        
        # Parse theme and direction enums
        theme_enum = MermaidTheme(theme)
        direction_enum = MermaidDirection(direction)
        
        config = MermaidConfig(
            title=title or "",
            theme=theme_enum,
            direction=direction_enum,
            show_tools=show_tools,
            show_roles=show_roles,
            show_config=show_config,
        )
        diagram = AgentDiagram(self, config=config)
        return diagram.to_mermaid()

    def draw_mermaid_png(
        self,
        *,
        theme: str = "default",
        direction: str = "TB",
        title: str | None = None,
        show_tools: bool = True,
        show_roles: bool = True,
        show_config: bool = False,
    ) -> bytes:
        """
        Generate a PNG image of this agent's Mermaid diagram.
        
        Requires httpx to be installed for mermaid.ink API.
        
        Args:
            theme: Mermaid theme (default, forest, dark, neutral, base)
            direction: Graph direction (TB, TD, BT, LR, RL)
            title: Optional diagram title
            show_tools: Whether to show agent tools
            show_roles: Whether to show agent role
            show_config: Whether to show model configuration
            
        Returns:
            PNG image as bytes
        """
        from agenticflow.visualization.mermaid import (
            AgentDiagram,
            MermaidConfig,
            MermaidDirection,
            MermaidTheme,
        )
        
        theme_enum = MermaidTheme(theme)
        direction_enum = MermaidDirection(direction)
        
        config = MermaidConfig(
            title=title or "",
            theme=theme_enum,
            direction=direction_enum,
            show_tools=show_tools,
            show_roles=show_roles,
            show_config=show_config,
        )
        diagram = AgentDiagram(self, config=config)
        return diagram.draw_png()

    # =========================================================================
    # Factory Methods - Create role-specific agents
    # =========================================================================
    
    @classmethod
    def as_supervisor(
        cls,
        name: str,
        model: Any,
        *,
        workers: list[str] | None = None,
        instructions: str | None = None,
        **kwargs,
    ) -> "Agent":
        """Create a supervisor agent.
        
        Supervisors delegate tasks to workers and synthesize results.
        
        Args:
            name: Agent name
            model: LLM model to use
            workers: List of worker agent names this supervisor manages
            instructions: Additional instructions (appended to role prompt)
            **kwargs: Additional Agent parameters
            
        Returns:
            Agent configured as a supervisor
            
        Example:
            supervisor = Agent.as_supervisor(
                name="Manager",
                model=ChatOpenAI(),
                workers=["Researcher", "Analyst"],
            )
        """
        from agenticflow.agent.roles import get_role_prompt
        
        base_prompt = get_role_prompt(AgentRole.SUPERVISOR)
        if workers:
            base_prompt += f"\n\nYour team members: {', '.join(workers)}"
        if instructions:
            base_prompt += f"\n\n{instructions}"
        
        return cls(
            name=name,
            model=model,
            role=AgentRole.SUPERVISOR,
            instructions=base_prompt,
            **kwargs,
        )
    
    @classmethod
    def as_worker(
        cls,
        name: str,
        model: Any,
        *,
        specialty: str | None = None,
        instructions: str | None = None,
        tools: list | None = None,
        **kwargs,
    ) -> "Agent":
        """Create a worker agent.
        
        Workers execute tasks using their tools.
        
        Args:
            name: Agent name
            model: LLM model to use  
            specialty: Description of worker's specialty
            instructions: Additional instructions
            tools: List of tools for this worker
            **kwargs: Additional Agent parameters
            
        Returns:
            Agent configured as a worker
            
        Example:
            researcher = Agent.as_worker(
                name="Researcher",
                model=ChatOpenAI(),
                specialty="web research and data gathering",
                tools=[search_tool, scrape_tool],
            )
        """
        from agenticflow.agent.roles import get_role_prompt
        
        base_prompt = get_role_prompt(AgentRole.WORKER)
        if specialty:
            base_prompt += f"\n\nYour specialty: {specialty}"
        if instructions:
            base_prompt += f"\n\n{instructions}"
        
        return cls(
            name=name,
            model=model,
            role=AgentRole.WORKER,
            instructions=base_prompt,
            tools=tools,
            **kwargs,
        )
    
    @classmethod
    def as_critic(
        cls,
        name: str,
        model: Any,
        *,
        criteria: list[str] | None = None,
        instructions: str | None = None,
        **kwargs,
    ) -> "Agent":
        """Create a reviewer/critic agent.
        
        Reviewers evaluate work and can approve (finish) or request revisions.
        
        Args:
            name: Agent name
            model: LLM model to use
            criteria: List of criteria to evaluate against
            instructions: Additional instructions
            **kwargs: Additional Agent parameters
            
        Returns:
            Agent configured as a reviewer
        """
        from agenticflow.agent.roles import get_role_prompt
        
        base_prompt = get_role_prompt(AgentRole.REVIEWER)
        if criteria:
            base_prompt += f"\n\nEvaluation criteria:\n- " + "\n- ".join(criteria)
        if instructions:
            base_prompt += f"\n\n{instructions}"
        
        return cls(
            name=name,
            model=model,
            role=AgentRole.REVIEWER,
            instructions=base_prompt,
            **kwargs,
        )
    
    # Alias for as_critic
    as_reviewer = as_critic
    
    @classmethod
    def as_autonomous(
        cls,
        name: str,
        model: Any,
        *,
        tools: list | None = None,
        instructions: str | None = None,
        **kwargs,
    ) -> "Agent":
        """Create an autonomous agent.
        
        Autonomous agents work independently - they use tools AND can finish.
        Perfect for single-agent flows or independent tasks.
        
        Args:
            name: Agent name
            model: LLM model to use
            tools: Tools available to this agent
            instructions: Additional instructions
            **kwargs: Additional Agent parameters
            
        Returns:
            Agent configured as autonomous
            
        Example:
            assistant = Agent.as_autonomous(
                name="Assistant",
                model=ChatOpenAI(),
                tools=[search, calculator],
            )
        """
        from agenticflow.agent.roles import get_role_prompt
        
        base_prompt = get_role_prompt(AgentRole.AUTONOMOUS)
        if instructions:
            base_prompt += f"\n\n{instructions}"
        
        return cls(
            name=name,
            model=model,
            role=AgentRole.AUTONOMOUS,
            tools=tools,
            instructions=base_prompt,
            **kwargs,
        )
    
    @classmethod
    def as_planner(
        cls,
        name: str,
        model: Any,
        *,
        available_agents: list[str] | None = None,
        instructions: str | None = None,
        **kwargs,
    ) -> "Agent":
        """Create a planner agent (autonomous with planning focus).
        
        Planners create execution plans with steps and dependencies.
        They are autonomous agents with planning-focused instructions.
        
        Args:
            name: Agent name
            model: LLM model to use
            available_agents: List of agents that can execute plan steps
            instructions: Additional instructions
            **kwargs: Additional Agent parameters
            
        Returns:
            Agent configured as a planner
        """
        base_prompt = """You create execution plans with steps and dependencies.

Output plans in this format:
PLAN:
1. [Step 1] - [assigned_to] - [dependencies: none/step_n]
2. [Step 2] - [assigned_to] - [dependencies: step_1]
...
END PLAN

When plan is complete: "FINAL ANSWER: [plan summary]"
"""
        if available_agents:
            base_prompt += f"\n\nAvailable agents for task assignment: {', '.join(available_agents)}"
        if instructions:
            base_prompt += f"\n\n{instructions}"
        
        return cls(
            name=name,
            model=model,
            role=AgentRole.AUTONOMOUS,  # Planners are autonomous - they can finish
            instructions=base_prompt,
            **kwargs,
        )
    
    @classmethod
    def as_researcher(
        cls,
        name: str,
        model: Any,
        *,
        tools: list | None = None,
        focus_areas: list[str] | None = None,
        instructions: str | None = None,
        **kwargs,
    ) -> "Agent":
        """Create a researcher agent (worker with research focus).
        
        Researchers gather and synthesize information using tools.
        They are workers with research-focused instructions.
        
        Args:
            name: Agent name
            model: LLM model to use
            tools: Research tools (search, scrape, etc.)
            focus_areas: Areas of research focus
            instructions: Additional instructions
            **kwargs: Additional Agent parameters
            
        Returns:
            Agent configured as a researcher
        """
        base_prompt = """You gather and synthesize information using your tools.

Structure your findings:
- Key facts discovered
- Sources used
- Confidence level (high/medium/low)
- Areas needing more research
"""
        if focus_areas:
            base_prompt += f"\n\nFocus areas: {', '.join(focus_areas)}"
        if instructions:
            base_prompt += f"\n\n{instructions}"
        
        return cls(
            name=name,
            model=model,
            role=AgentRole.WORKER,  # Researchers are workers - they use tools but can't finish
            tools=tools,
            instructions=base_prompt,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, name={self.name}, role={self.role.value})"
