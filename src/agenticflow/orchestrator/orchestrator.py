"""
Orchestrator - coordinates multiple agents to achieve complex goals.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from agenticflow.core.enums import AgentRole, EventType, Priority, TaskStatus
from agenticflow.core.utils import generate_id, now_utc
from agenticflow.models.event import Event
from agenticflow.models.task import Task

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.events.bus import EventBus
    from agenticflow.tasks.manager import TaskManager
    from agenticflow.tools.registry import ToolRegistry


class Orchestrator:
    """
    Coordinates multiple agents to achieve complex goals.
    
    The Orchestrator is responsible for:
    - Planning: Breaking down requests into executable tasks
    - Delegation: Assigning tasks to appropriate agents
    - Monitoring: Tracking task progress and handling failures
    - Aggregation: Collecting and combining results
    
    Attributes:
        event_bus: EventBus for communication
        task_manager: TaskManager for task tracking
        tool_registry: ToolRegistry for available tools
        agents: Registered agents
        model_name: LLM model for planning
        
    Example:
        ```python
        orchestrator = Orchestrator(
            event_bus=event_bus,
            task_manager=task_manager,
            tool_registry=tool_registry,
            model_name="gpt-4o",
        )
        
        # Register agents
        orchestrator.register_agent(writer_agent)
        orchestrator.register_agent(analyst_agent)
        
        # Run a complex request
        result = await orchestrator.run(
            "Analyze the data and write a summary report"
        )
        ```
    """

    def __init__(
        self,
        event_bus: EventBus,
        task_manager: TaskManager,
        tool_registry: ToolRegistry,
        model_name: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the Orchestrator.
        
        Args:
            event_bus: EventBus for communication
            task_manager: TaskManager for task tracking
            tool_registry: ToolRegistry for available tools
            model_name: LLM model for planning (optional)
            temperature: LLM temperature for planning
        """
        self.id = generate_id()
        self.event_bus = event_bus
        self.task_manager = task_manager
        self.tool_registry = tool_registry
        self.agents: dict[str, Agent] = {}
        self._model_name = model_name
        self._temperature = temperature
        self._model = None

        # Subscribe to events
        self.event_bus.subscribe(EventType.TASK_COMPLETED, self._on_task_completed)
        self.event_bus.subscribe(EventType.TASK_FAILED, self._on_task_failed)

    @property
    def model(self):
        """Lazy-load the LLM model for planning."""
        if self._model is None and self._model_name:
            self._model = init_chat_model(
                self._model_name,
                temperature=self._temperature,
            )
        return self._model

    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.id] = agent

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if agent was removed
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_available_agents(self) -> list[Agent]:
        """Get all agents that can accept new work."""
        return [a for a in self.agents.values() if a.is_available()]

    def get_agents_by_role(self, role: AgentRole) -> list[Agent]:
        """Get all agents with a specific role."""
        return [a for a in self.agents.values() if a.role == role]

    def find_agent_for_tool(self, tool_name: str) -> Agent | None:
        """
        Find an available agent that can use a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            An available agent or None
        """
        for agent in self.get_available_agents():
            if agent.config.can_use_tool(tool_name):
                return agent
        return None

    async def _emit_event(
        self,
        event_type: EventType,
        data: dict,
        correlation_id: str | None = None,
    ) -> None:
        """Emit an event from the orchestrator."""
        await self.event_bus.publish(
            Event(
                type=event_type,
                data=data,
                source=f"orchestrator:{self.id}",
                correlation_id=correlation_id,
            )
        )

    async def plan(self, request: str, correlation_id: str) -> list[dict]:
        """
        Create an execution plan for a request.
        
        Args:
            request: The user's request
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of planned steps
        """
        if not self.model:
            raise RuntimeError("No planning model configured")

        await self._emit_event(
            EventType.AGENT_INVOKED,
            {"action": "planning", "request": request},
            correlation_id,
        )

        tool_descriptions = self.tool_registry.get_tool_descriptions()
        agent_descriptions = self._get_agent_descriptions()

        planning_prompt = f"""Create an execution plan as a DAG (Directed Acyclic Graph).

AVAILABLE TOOLS:
{tool_descriptions if tool_descriptions else "No tools registered"}

AVAILABLE AGENTS:
{agent_descriptions if agent_descriptions else "No agents registered"}

OUTPUT FORMAT (JSON):
{{
  "steps": [
    {{"id": "unique_id", "name": "descriptive name", "tool": "tool_name", "args": {{}}, "depends_on": []}},
    {{"id": "step2", "name": "another task", "tool": "tool_name", "args": {{"text": "$ref:unique_id"}}, "depends_on": ["unique_id"]}}
  ]
}}

RULES:
1. Each step has unique "id" and descriptive "name"
2. "tool" specifies which tool to use (must be from available tools)
3. "args" are the arguments to pass to the tool
4. "depends_on" lists IDs of steps that must complete first
5. Use "$ref:step_id" in args to reference another step's result
6. Steps with NO dependencies can run in PARALLEL
7. Keep the plan minimal and efficient

User request: {request}

JSON:"""

        response = await self.model.ainvoke([HumanMessage(content=planning_prompt)])

        try:
            content = response.content
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            plan = json.loads(content.strip())
            steps = plan.get("steps", [])

            await self._emit_event(
                EventType.PLAN_CREATED,
                {"step_count": len(steps), "steps": steps},
                correlation_id,
            )

            return steps

        except Exception as e:
            await self._emit_event(
                EventType.PLAN_FAILED,
                {"error": str(e), "response": str(response.content)[:200]},
                correlation_id,
            )
            return []

    def _get_agent_descriptions(self) -> str:
        """Get formatted descriptions of available agents."""
        descriptions = []
        for agent in self.agents.values():
            tools = ", ".join(agent.config.tools) if agent.config.tools else "all"
            descriptions.append(
                f"- {agent.name} ({agent.role.value}): {agent.config.description or 'No description'} [tools: {tools}]"
            )
        return "\n".join(descriptions)

    def _resolve_args(self, args: dict, correlation_id: str) -> dict:
        """
        Resolve $ref: patterns in arguments.
        
        Args:
            args: Arguments that may contain references
            correlation_id: For looking up completed tasks
            
        Returns:
            Resolved arguments
        """
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str):
                def replacer(match: re.Match) -> str:
                    ref_id = match.group(1)
                    # Look up task by plan ID stored in metadata
                    for task in self.task_manager.tasks.values():
                        if task.metadata.get("plan_id") == ref_id:
                            return str(task.result) if task.result else match.group(0)
                    return match.group(0)

                resolved[key] = re.sub(r"\$ref:(\w+)", replacer, value)
            else:
                resolved[key] = value
        return resolved

    async def execute_task(self, task: Task, correlation_id: str) -> Any:
        """
        Execute a single task.
        
        Args:
            task: The task to execute
            correlation_id: Correlation ID for tracking
            
        Returns:
            Task result
        """
        # Update status to running
        await self.task_manager.update_status(
            task.id, TaskStatus.RUNNING, correlation_id=correlation_id
        )

        try:
            # Find an agent for this task
            agent = None
            if task.assigned_agent_id:
                agent = self.agents.get(task.assigned_agent_id)

            if not agent and task.tool:
                agent = self.find_agent_for_tool(task.tool)

            if not agent:
                # Use any available agent
                available = self.get_available_agents()
                if available:
                    agent = available[0]

            if not agent:
                raise RuntimeError("No available agent for task")

            # Resolve argument references
            resolved_args = self._resolve_args(task.args, correlation_id)
            task.args = resolved_args

            # Execute via agent
            result = await agent.execute_task(task, correlation_id)

            # Update task with result
            await self.task_manager.update_status(
                task.id,
                TaskStatus.COMPLETED,
                result=result,
                correlation_id=correlation_id,
            )

            return result

        except Exception as e:
            await self.task_manager.update_status(
                task.id,
                TaskStatus.FAILED,
                error=str(e),
                correlation_id=correlation_id,
            )
            raise

    async def _on_task_completed(self, event: Event) -> None:
        """Handle task completion - check for subtask aggregation."""
        task_data = event.data.get("task", {})
        parent_id = task_data.get("parent_id")

        if parent_id:
            # Check if all sibling subtasks are complete
            if await self.task_manager.check_subtasks_complete(parent_id):
                await self.task_manager.aggregate_subtasks(
                    parent_id, correlation_id=event.correlation_id
                )

    async def _on_task_failed(self, event: Event) -> None:
        """Handle task failure - attempt retry if configured."""
        task_data = event.data.get("task", {})
        task_id = task_data.get("id")

        if task_id:
            task = await self.task_manager.get_task(task_id)
            if task and task.can_retry():
                await self.task_manager.retry_task(
                    task_id, correlation_id=event.correlation_id
                )

    async def run(self, request: str) -> dict:
        """
        Run the full orchestration workflow.
        
        Args:
            request: The user's request
            
        Returns:
            Dictionary with results and metadata
        """
        correlation_id = generate_id()

        # Emit system started
        await self._emit_event(
            EventType.SYSTEM_STARTED,
            {"request": request},
            correlation_id,
        )

        try:
            # Plan
            steps = await self.plan(request, correlation_id)

            if not steps:
                return {
                    "error": "Planning failed - no steps generated",
                    "correlation_id": correlation_id,
                }

            # Create tasks from plan
            task_id_map: dict[str, str] = {}  # plan_id -> task_id

            for step in steps:
                # Map dependencies to task IDs
                depends_on = [
                    task_id_map[dep]
                    for dep in step.get("depends_on", [])
                    if dep in task_id_map
                ]

                task = await self.task_manager.create_task(
                    name=step.get("name", step["id"]),
                    tool=step.get("tool"),
                    args=step.get("args", {}),
                    depends_on=depends_on,
                    correlation_id=correlation_id,
                    metadata={"plan_id": step["id"]},
                )
                task_id_map[step["id"]] = task.id

            # Execute in waves
            step_num = 0
            max_iterations = 100  # Safety limit

            while step_num < max_iterations:
                ready_tasks = await self.task_manager.get_ready_tasks()

                if not ready_tasks:
                    # Check if all done
                    all_done = all(
                        t.is_complete()
                        for t in self.task_manager.tasks.values()
                    )
                    if all_done:
                        break
                    else:
                        await asyncio.sleep(0.1)
                        continue

                step_num += 1

                await self._emit_event(
                    EventType.PLAN_STEP_STARTED,
                    {"step": step_num, "task_count": len(ready_tasks)},
                    correlation_id,
                )

                # Execute all ready tasks in parallel
                await asyncio.gather(
                    *[self.execute_task(task, correlation_id) for task in ready_tasks],
                    return_exceptions=True,
                )

                await self._emit_event(
                    EventType.PLAN_STEP_COMPLETED,
                    {"step": step_num},
                    correlation_id,
                )

            # Aggregate results
            results = {
                task.name: task.result
                for task in self.task_manager.tasks.values()
                if task.status == TaskStatus.COMPLETED
            }

            # Emit system stopped
            await self._emit_event(
                EventType.SYSTEM_STOPPED,
                {"correlation_id": correlation_id, "task_count": len(results)},
                correlation_id,
            )

            return {
                "correlation_id": correlation_id,
                "results": results,
                "event_count": len(
                    self.event_bus.get_history(correlation_id=correlation_id)
                ),
                "stats": self.task_manager.get_stats(),
            }

        except Exception as e:
            await self._emit_event(
                EventType.SYSTEM_ERROR,
                {"error": str(e)},
                correlation_id,
            )
            return {
                "error": str(e),
                "correlation_id": correlation_id,
            }

    async def run_task(
        self,
        name: str,
        tool: str | None = None,
        args: dict | None = None,
        agent_id: str | None = None,
    ) -> Any:
        """
        Run a single task directly without planning.
        
        Args:
            name: Task name
            tool: Tool to use
            args: Tool arguments
            agent_id: Specific agent to use
            
        Returns:
            Task result
        """
        correlation_id = generate_id()

        task = await self.task_manager.create_task(
            name=name,
            tool=tool,
            args=args or {},
            assigned_agent_id=agent_id,
            correlation_id=correlation_id,
        )

        return await self.execute_task(task, correlation_id)

    def to_dict(self) -> dict:
        """Convert orchestrator info to dictionary."""
        return {
            "id": self.id,
            "agent_count": len(self.agents),
            "agents": [a.to_dict() for a in self.agents.values()],
            "tool_count": len(self.tool_registry),
            "task_stats": self.task_manager.get_stats(),
        }
