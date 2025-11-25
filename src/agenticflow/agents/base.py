"""
Agent - an autonomous entity that can think, act, and communicate.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agenticflow.agents.config import AgentConfig
from agenticflow.agents.state import AgentState
from agenticflow.core.enums import AgentRole, AgentStatus, EventType
from agenticflow.core.utils import generate_id, now_utc
from agenticflow.models.event import Event
from agenticflow.models.message import Message
from agenticflow.models.task import Task

if TYPE_CHECKING:
    from agenticflow.events.bus import EventBus
    from agenticflow.tools.registry import ToolRegistry


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
    
    Attributes:
        id: Unique agent identifier
        config: Agent configuration
        state: Agent runtime state
        event_bus: Event bus for communication
        tool_registry: Registry of available tools
        
    Example:
        ```python
        agent = Agent(
            config=AgentConfig(
                name="Writer",
                role=AgentRole.WORKER,
                model_name="gpt-4o",
                tools=["write_poem", "write_story"],
            ),
            event_bus=event_bus,
            tool_registry=tool_registry,
        )
        
        # Execute a task
        result = await agent.execute_task(task)
        
        # Direct thinking
        response = await agent.think("What should I write about?")
        
        # Direct action
        result = await agent.act("write_poem", {"subject": "sunset"})
        ```
    """

    def __init__(
        self,
        config: AgentConfig,
        event_bus: EventBus,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """
        Initialize an Agent.
        
        Args:
            config: Agent configuration
            event_bus: Event bus for communication
            tool_registry: Registry of available tools (optional)
        """
        self.id = generate_id()
        self.config = config
        self.state = AgentState()
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self._model = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Agent's display name."""
        return self.config.name

    @property
    def role(self) -> AgentRole:
        """Agent's role in the system."""
        return self.config.role

    @property
    def status(self) -> AgentStatus:
        """Current agent status."""
        return self.state.status

    @property
    def model(self):
        """Lazy-load the LLM model."""
        if self._model is None and self.config.model_name:
            self._model = init_chat_model(
                self.config.model_name,
                temperature=self.config.temperature,
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
        await self.event_bus.publish(
            Event(
                type=event_type,
                data=data,
                source=f"agent:{self.id}",
                correlation_id=correlation_id,
            )
        )

    async def think(
        self,
        prompt: str,
        correlation_id: str | None = None,
    ) -> str:
        """
        Process a prompt through the agent's reasoning.
        
        This is the agent's "thinking" phase where it uses its LLM
        to reason about the input and determine what to do.
        
        Args:
            prompt: The prompt to process
            correlation_id: Optional correlation ID for event tracking
            
        Returns:
            The LLM's response
            
        Raises:
            RuntimeError: If no model is configured
        """
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

        # Build messages
        messages = []
        if self.config.system_prompt:
            messages.append(SystemMessage(content=self.config.system_prompt))

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

    async def act(
        self,
        tool_name: str,
        args: dict,
        correlation_id: str | None = None,
    ) -> Any:
        """
        Execute an action using a tool.
        
        This is the agent's "acting" phase where it interacts with
        the world through tools.
        
        Args:
            tool_name: Name of the tool to use
            args: Arguments for the tool
            correlation_id: Optional correlation ID for event tracking
            
        Returns:
            The tool's result
            
        Raises:
            RuntimeError: If no tool registry is configured
            ValueError: If tool is not found
            PermissionError: If agent is not authorized to use the tool
        """
        if not self.tool_registry:
            raise RuntimeError(f"Agent {self.name} has no tool registry")

        tool_obj = self.tool_registry.get(tool_name)
        if not tool_obj:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Check if agent is allowed to use this tool
        if not self.config.can_use_tool(tool_name):
            raise PermissionError(
                f"Agent {self.name} not authorized to use tool: {tool_name}"
            )

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

        start_time = now_utc()

        try:
            # Execute tool in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: tool_obj.invoke(args))

            # Track timing
            duration_ms = (now_utc() - start_time).total_seconds() * 1000
            self.state.add_acting_time(duration_ms)

            await self._emit_event(
                EventType.TOOL_RESULT,
                {
                    "agent_id": self.id,
                    "tool": tool_name,
                    "result_preview": str(result)[:200],
                    "duration_ms": duration_ms,
                },
                correlation_id,
            )

            await self._set_status(AgentStatus.IDLE, correlation_id)
            return result

        except Exception as e:
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

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, name={self.name}, role={self.role.value})"
