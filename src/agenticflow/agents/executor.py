"""
Advanced Execution Strategies for Agents
========================================

Provides sophisticated execution patterns beyond simple ReAct:

1. **ReActExecutor**: Classic think-act-observe loop (baseline)
2. **PlanExecutor**: Plan all steps upfront, then execute
3. **DAGExecutor**: Build dependency graph, execute with max parallelism (LLMCompiler-style)
4. **AdaptiveExecutor**: Dynamically choose strategy based on task

The DAGExecutor is the fastest for complex tasks with tool dependencies.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from agenticflow.core.utils import generate_id, now_utc

if TYPE_CHECKING:
    from agenticflow.agents.base import Agent
    from agenticflow.observability.progress import ProgressTracker


class ExecutionStrategy(Enum):
    """Available execution strategies."""
    
    REACT = "react"  # Think-Act-Observe loop
    PLAN_EXECUTE = "plan_execute"  # Plan then execute
    DAG = "dag"  # Dependency graph with parallel execution
    ADAPTIVE = "adaptive"  # Auto-select based on task


@dataclass
class ToolCall:
    """A planned tool call with dependencies."""
    
    id: str
    tool_name: str
    args: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)  # IDs of tools this depends on
    result: Any = None
    error: str | None = None
    status: str = "pending"  # pending, running, completed, failed
    duration_ms: float = 0.0

    def is_ready(self, completed: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed for dep in self.depends_on)


@dataclass
class ExecutionPlan:
    """A plan of tool calls with dependencies."""
    
    calls: list[ToolCall] = field(default_factory=list)
    final_answer_template: str = ""
    
    def add_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        depends_on: list[str] | None = None,
    ) -> str:
        """Add a tool call to the plan, returns its ID."""
        call_id = f"call_{len(self.calls)}"
        self.calls.append(ToolCall(
            id=call_id,
            tool_name=tool_name,
            args=args,
            depends_on=depends_on or [],
        ))
        return call_id
    
    def get_ready_calls(self, completed: set[str]) -> list[ToolCall]:
        """Get all calls that are ready to execute (dependencies satisfied)."""
        return [
            call for call in self.calls
            if call.status == "pending" and call.is_ready(completed)
        ]
    
    def get_execution_order(self) -> list[list[str]]:
        """Get execution waves (calls that can run in parallel)."""
        waves: list[list[str]] = []
        completed: set[str] = set()
        remaining = {c.id for c in self.calls}
        
        while remaining:
            wave = [
                c.id for c in self.calls
                if c.id in remaining and c.is_ready(completed)
            ]
            if not wave:
                # Circular dependency or error
                break
            waves.append(wave)
            completed.update(wave)
            remaining -= set(wave)
        
        return waves


class BaseExecutor(ABC):
    """Base class for execution strategies."""
    
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.max_iterations: int = 10
        self.on_step: Callable[[str, Any], None] | None = None
        self.tracker: ProgressTracker | None = None
    
    @abstractmethod
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a task and return the result."""
        pass
    
    def _emit_step(self, step_type: str, data: Any) -> None:
        """Emit a step event if callback is set."""
        if self.on_step:
            self.on_step(step_type, data)
    
    def _track_tool_call(self, tool: str, args: dict[str, Any]) -> None:
        """Track tool call via progress tracker if available."""
        if self.tracker:
            self.tracker.tool_call(tool, args, agent=self.agent.name)
    
    def _track_tool_result(self, tool: str, result: Any, duration_ms: float = 0) -> None:
        """Track tool result via progress tracker if available."""
        if self.tracker:
            self.tracker.tool_result(tool, str(result)[:200], duration_ms=duration_ms)
    
    def _track_tool_error(self, tool: str, error: str) -> None:
        """Track tool error via progress tracker if available."""
        if self.tracker:
            self.tracker.tool_error(tool, error)


class ReActExecutor(BaseExecutor):
    """
    Classic ReAct (Reason + Act) execution.
    
    Pattern: Think → Act → Observe → Repeat
    
    Simple but slow - each step waits for the previous one.
    Good for simple tasks or when steps truly depend on each other.
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using ReAct pattern."""
        observations: list[dict[str, Any]] = []
        
        for i in range(self.max_iterations):
            # Build prompt with observations
            prompt = self._build_prompt(task, observations, context)
            
            # Think
            self._emit_step("think", {"iteration": i})
            response = await self.agent.think(prompt)
            
            # Parse for tool calls or final answer
            action = self._parse_response(response)
            
            if action["type"] == "final_answer":
                return action["answer"]
            
            if action["type"] == "tool_call":
                # Act
                self._emit_step("act", {"tool": action["tool"], "args": action["args"]})
                self._track_tool_call(action["tool"], action["args"])
                start_time = now_utc()
                try:
                    result = await self.agent.act(action["tool"], action["args"])
                    duration_ms = (now_utc() - start_time).total_seconds() * 1000
                    self._track_tool_result(action["tool"], result, duration_ms)
                    observations.append({
                        "tool": action["tool"],
                        "args": action["args"],
                        "result": str(result),
                    })
                except Exception as e:
                    self._track_tool_error(action["tool"], str(e))
                    observations.append({
                        "tool": action["tool"],
                        "args": action["args"],
                        "error": str(e),
                    })
        
        return f"Max iterations reached. Last observations: {observations[-1] if observations else 'none'}"
    
    def _build_prompt(
        self,
        task: str,
        observations: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> str:
        """Build prompt with task and observations."""
        parts = [f"Task: {task}"]
        
        if context:
            parts.append(f"Context: {json.dumps(context)}")
        
        if observations:
            parts.append("\nPrevious observations:")
            for obs in observations:
                if "error" in obs:
                    parts.append(f"  - {obs['tool']}({obs['args']}) → ERROR: {obs['error']}")
                else:
                    parts.append(f"  - {obs['tool']}({obs['args']}) → {obs['result'][:200]}")
        
        parts.append("\nRespond with either:")
        parts.append("1. TOOL: <tool_name>(<args as json>)")
        parts.append("2. FINAL ANSWER: <your answer>")
        
        return "\n".join(parts)
    
    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse agent response for tool call or final answer."""
        response_lower = response.lower()
        
        # Check for final answer
        if "final answer:" in response_lower:
            idx = response_lower.index("final answer:")
            answer = response[idx + 13:].strip()
            return {"type": "final_answer", "answer": answer}
        
        # Check for tool call
        tool_match = re.search(r"TOOL:\s*(\w+)\s*\((.+?)\)", response, re.IGNORECASE | re.DOTALL)
        if tool_match:
            tool_name = tool_match.group(1)
            try:
                args = json.loads(tool_match.group(2))
            except json.JSONDecodeError:
                # Try to parse as simple key=value
                args = {"input": tool_match.group(2).strip()}
            return {"type": "tool_call", "tool": tool_name, "args": args}
        
        # Default to final answer if no tool pattern found
        return {"type": "final_answer", "answer": response}


class PlanExecutor(BaseExecutor):
    """
    Plan-and-Execute strategy.
    
    Pattern: Plan all steps → Execute sequentially
    
    Faster than ReAct because planning is done once.
    Good when task structure is clear upfront.
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using plan-and-execute pattern."""
        # Phase 1: Create plan
        self._emit_step("planning", {"task": task})
        plan = await self._create_plan(task, context)
        
        if not plan.calls:
            # No tools needed, just think
            return await self.agent.think(task)
        
        # Phase 2: Execute plan sequentially
        results: dict[str, Any] = {}
        for call in plan.calls:
            self._emit_step("executing", {"tool": call.tool_name, "id": call.id})
            
            # Substitute results from previous calls
            args = self._substitute_args(call.args, results)
            
            # Track the tool call
            self._track_tool_call(call.tool_name, args)
            
            try:
                start = now_utc()
                result = await self.agent.act(call.tool_name, args)
                call.duration_ms = (now_utc() - start).total_seconds() * 1000
                call.result = result
                call.status = "completed"
                results[call.id] = result
                
                # Track the result
                self._track_tool_result(call.tool_name, result, call.duration_ms)
            except Exception as e:
                call.error = str(e)
                call.status = "failed"
                results[call.id] = f"ERROR: {e}"
                
                # Track the error
                self._track_tool_error(call.tool_name, str(e))
        
        # Phase 3: Synthesize final answer
        return await self._synthesize(task, plan, results)
    
    async def _create_plan(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> ExecutionPlan:
        """Ask agent to create an execution plan."""
        tools_desc = self.agent.tool_registry.get_tool_descriptions() if self.agent.tool_registry else "No tools available"
        
        prompt = f"""Create a step-by-step plan to accomplish this task:

Task: {task}
{f"Context: {json.dumps(context)}" if context else ""}

Available tools:
{tools_desc}

Respond with a JSON plan:
{{
  "steps": [
    {{"tool": "tool_name", "args": {{}}, "depends_on": []}},
    ...
  ]
}}

If no tools are needed, respond with: {{"steps": []}}
"""
        
        response = await self.agent.think(prompt)
        return self._parse_plan(response)
    
    def _parse_plan(self, response: str) -> ExecutionPlan:
        """Parse plan from agent response."""
        plan = ExecutionPlan()
        
        # Try to extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for i, step in enumerate(data.get("steps", [])):
                    plan.add_call(
                        tool_name=step["tool"],
                        args=step.get("args", {}),
                        depends_on=step.get("depends_on", []),
                    )
            except (json.JSONDecodeError, KeyError):
                pass
        
        return plan
    
    def _substitute_args(
        self,
        args: dict[str, Any],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Substitute $call_N references with actual results."""
        substituted = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                ref = value[1:]  # Remove $
                if ref in results:
                    substituted[key] = results[ref]
                else:
                    substituted[key] = value
            else:
                substituted[key] = value
        return substituted
    
    async def _synthesize(
        self,
        task: str,
        plan: ExecutionPlan,
        results: dict[str, Any],
    ) -> str:
        """Synthesize final answer from results."""
        prompt = f"""Task: {task}

Tool results:
{json.dumps(results, indent=2, default=str)}

Based on these results, provide a final answer."""
        
        return await self.agent.think(prompt)


class DAGExecutor(BaseExecutor):
    """
    DAG-based execution with maximum parallelism (LLMCompiler-style).
    
    Pattern: Plan DAG → Execute in parallel waves → Synthesize
    
    FASTEST approach for complex tasks. Identifies dependencies
    and executes independent steps in parallel.
    
    Example DAG:
        search_A ─┐
                  ├─► combine ─► final
        search_B ─┘
        
    search_A and search_B run in parallel, then combine runs.
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using DAG with parallel execution."""
        # Phase 1: Create DAG plan
        self._emit_step("planning_dag", {"task": task})
        plan = await self._create_dag_plan(task, context)
        
        if not plan.calls:
            return await self.agent.think(task)
        
        # Phase 2: Execute DAG with parallelism
        waves = plan.get_execution_order()
        self._emit_step("dag_waves", {"waves": len(waves), "total_calls": len(plan.calls)})
        
        results: dict[str, Any] = {}
        
        for wave_idx, wave in enumerate(waves):
            wave_calls = [c for c in plan.calls if c.id in wave]
            self._emit_step("executing_wave", {
                "wave": wave_idx + 1,
                "total_waves": len(waves),
                "parallel_calls": len(wave_calls),
            })
            
            # Execute wave in parallel
            wave_results = await self._execute_wave(wave_calls, results)
            results.update(wave_results)
        
        # Phase 3: Synthesize
        return await self._synthesize(task, plan, results)
    
    async def _create_dag_plan(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> ExecutionPlan:
        """Create a DAG plan identifying dependencies."""
        tools_desc = self.agent.tool_registry.get_tool_descriptions() if self.agent.tool_registry else "No tools available"
        
        prompt = f"""Create an execution plan as a Directed Acyclic Graph (DAG).
Identify which steps can run IN PARALLEL (no dependencies between them).

Task: {task}
{f"Context: {json.dumps(context)}" if context else ""}

Available tools:
{tools_desc}

Respond with a JSON DAG plan:
{{
  "steps": [
    {{"id": "call_0", "tool": "tool_name", "args": {{}}, "depends_on": []}},
    {{"id": "call_1", "tool": "other_tool", "args": {{}}, "depends_on": []}},
    {{"id": "call_2", "tool": "combine", "args": {{"a": "$call_0", "b": "$call_1"}}, "depends_on": ["call_0", "call_1"]}}
  ]
}}

Rules:
- Steps with NO dependencies can run in parallel
- Use $call_N to reference results from previous steps
- List all dependencies in depends_on array
"""
        
        response = await self.agent.think(prompt)
        return self._parse_dag_plan(response)
    
    def _parse_dag_plan(self, response: str) -> ExecutionPlan:
        """Parse DAG plan from agent response."""
        plan = ExecutionPlan()
        
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for step in data.get("steps", []):
                    call = ToolCall(
                        id=step.get("id", f"call_{len(plan.calls)}"),
                        tool_name=step["tool"],
                        args=step.get("args", {}),
                        depends_on=step.get("depends_on", []),
                    )
                    plan.calls.append(call)
            except (json.JSONDecodeError, KeyError):
                pass
        
        return plan
    
    async def _execute_wave(
        self,
        calls: list[ToolCall],
        prior_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a wave of calls in parallel."""
        
        async def execute_one(call: ToolCall) -> tuple[str, Any]:
            """Execute a single call."""
            args = self._substitute_args(call.args, prior_results)
            
            call.status = "running"
            start = now_utc()
            
            # Track the tool call
            self._track_tool_call(call.tool_name, args)
            
            try:
                result = await self.agent.act(call.tool_name, args)
                call.duration_ms = (now_utc() - start).total_seconds() * 1000
                call.result = result
                call.status = "completed"
                
                # Track the result
                self._track_tool_result(call.tool_name, result, call.duration_ms)
                
                return (call.id, result)
            except Exception as e:
                call.duration_ms = (now_utc() - start).total_seconds() * 1000
                call.error = str(e)
                call.status = "failed"
                
                # Track the error
                self._track_tool_error(call.tool_name, str(e))
                
                return (call.id, f"ERROR: {e}")
        
        # Execute all calls in parallel
        results = await asyncio.gather(*[execute_one(c) for c in calls])
        return dict(results)
    
    def _substitute_args(
        self,
        args: dict[str, Any],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Substitute $call_N references with actual results."""
        substituted = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                ref = value[1:]
                substituted[key] = results.get(ref, value)
            else:
                substituted[key] = value
        return substituted
    
    async def _synthesize(
        self,
        task: str,
        plan: ExecutionPlan,
        results: dict[str, Any],
    ) -> str:
        """Synthesize final answer from DAG results."""
        # Show execution summary
        successful = [c for c in plan.calls if c.status == "completed"]
        failed = [c for c in plan.calls if c.status == "failed"]
        
        prompt = f"""Task: {task}

Execution completed:
- Successful: {len(successful)} calls
- Failed: {len(failed)} calls

Results:
{json.dumps(results, indent=2, default=str)}

Provide the final answer based on these results."""
        
        return await self.agent.think(prompt)


class AdaptiveExecutor(BaseExecutor):
    """
    Adaptive execution that chooses the best strategy.
    
    - Simple task (no tools) → Direct thinking
    - Single tool → ReAct
    - Multiple independent tools → DAG (parallel)
    - Complex dependencies → DAG
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using automatically selected strategy."""
        # Analyze task to choose strategy
        strategy = await self._choose_strategy(task, context)
        self._emit_step("strategy_selected", {"strategy": strategy.value})
        
        if strategy == ExecutionStrategy.REACT:
            executor = ReActExecutor(self.agent)
        elif strategy == ExecutionStrategy.PLAN_EXECUTE:
            executor = PlanExecutor(self.agent)
        else:  # DAG is default for complex
            executor = DAGExecutor(self.agent)
        
        executor.max_iterations = self.max_iterations
        executor.on_step = self.on_step
        
        return await executor.execute(task, context)
    
    async def _choose_strategy(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> ExecutionStrategy:
        """Analyze task to choose best strategy."""
        if not self.agent.tool_registry or len(self.agent.tool_registry) == 0:
            return ExecutionStrategy.REACT  # No tools, just think
        
        # Ask agent to analyze complexity
        prompt = f"""Analyze this task's complexity:

Task: {task}

Respond with ONE word:
- SIMPLE: No tools needed, or single straightforward tool call
- SEQUENTIAL: Multiple tools, each depends on previous result  
- PARALLEL: Multiple independent tool calls possible
"""
        
        response = await self.agent.think(prompt)
        response_lower = response.lower()
        
        if "parallel" in response_lower:
            return ExecutionStrategy.DAG
        elif "sequential" in response_lower:
            return ExecutionStrategy.PLAN_EXECUTE
        else:
            return ExecutionStrategy.REACT


def create_executor(
    agent: Agent,
    strategy: ExecutionStrategy = ExecutionStrategy.DAG,
) -> BaseExecutor:
    """Create an executor with the specified strategy.
    
    Args:
        agent: The agent to execute with.
        strategy: Execution strategy to use.
        
    Returns:
        Configured executor instance.
        
    Example:
        >>> executor = create_executor(agent, ExecutionStrategy.DAG)
        >>> result = await executor.execute("Search for X and Y, then combine")
    """
    executors = {
        ExecutionStrategy.REACT: ReActExecutor,
        ExecutionStrategy.PLAN_EXECUTE: PlanExecutor,
        ExecutionStrategy.DAG: DAGExecutor,
        ExecutionStrategy.ADAPTIVE: AdaptiveExecutor,
    }
    
    return executors[strategy](agent)
