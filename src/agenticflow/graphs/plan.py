"""
Plan-and-Execute execution strategy.

Plans all steps upfront, then executes sequentially.
Faster than ReAct because planning is done once.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agenticflow.core.utils import now_utc
from agenticflow.graphs.base import BaseExecutor
from agenticflow.graphs.models import ExecutionPlan


class PlanExecutor(BaseExecutor):
    """
    Plan-and-Execute strategy with self-correction.
    
    Pattern: Plan all steps → Execute sequentially → Self-correct on errors
    
    The agent first creates a complete plan of what tools to call,
    then executes them one by one, with self-correction on failures.
    
    Pros:
        - Planning is done once upfront
        - Faster than ReAct for multi-step tasks
        - Clear execution structure
        - Self-corrects tool call errors
        
    Cons:
        - Still sequential execution
        - Plan may need revision if steps fail
        - No parallelism
        
    Example:
        executor = PlanExecutor(agent)
        result = await executor.execute("Research topic X and summarize")
    """
    
    # Maximum correction attempts per failed call
    max_correction_attempts: int = 2
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using plan-and-execute pattern with self-correction.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The synthesized final answer.
        """
        # Phase 1: Create plan
        self._emit_step("planning", {"task": task})
        plan = await self._create_plan(task, context)
        
        if not plan.calls:
            # No tools needed, just think
            return await self.agent.think(task)
        
        # Phase 2: Execute plan sequentially with self-correction
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
                error_str = str(e)
                # Track the error
                self._track_tool_error(call.tool_name, error_str)
                
                # Attempt self-correction
                corrected_result = await self._attempt_correction(
                    call, args, error_str, results
                )
                
                if corrected_result is not None:
                    call.result = corrected_result
                    call.status = "completed"
                    results[call.id] = corrected_result
                else:
                    call.error = error_str
                    call.status = "failed"
                    results[call.id] = f"ERROR: {e}"
        
        # Phase 3: Synthesize final answer
        return await self._synthesize(task, plan, results)
    
    async def _attempt_correction(
        self,
        call: Any,
        original_args: dict[str, Any],
        error: str,
        prior_results: dict[str, Any],
    ) -> Any | None:
        """Attempt to self-correct a failed tool call.
        
        Asks the LLM to analyze the error and provide corrected arguments.
        
        Args:
            call: The failed ToolCall.
            original_args: The arguments that caused the error.
            error: The error message.
            prior_results: Results from previous calls.
            
        Returns:
            The result if correction succeeds, None otherwise.
        """
        for attempt in range(self.max_correction_attempts):
            self._emit_step("self_correction", {
                "call_id": call.id,
                "tool": call.tool_name,
                "attempt": attempt + 1,
                "max_attempts": self.max_correction_attempts,
                "error": error,
            })
            
            # Ask LLM to analyze and correct
            correction_prompt = self._build_correction_prompt(
                call, original_args, error, prior_results
            )
            
            try:
                response = await self.agent.think(
                    correction_prompt,
                    include_tools=False,
                    system_prompt_override="You are debugging a tool call error. Analyze the error and provide corrected arguments as JSON only.",
                )
                
                # Parse corrected args from response
                corrected_args = self._parse_corrected_args(response)
                
                if corrected_args and corrected_args != original_args:
                    # Track the correction attempt
                    self._track_tool_call(call.tool_name, corrected_args)
                    
                    result = await self.agent.act(call.tool_name, corrected_args)
                    self._track_tool_result(call.tool_name, result, 0)
                    return result
                    
            except Exception as retry_error:
                error = str(retry_error)
                self._track_tool_error(call.tool_name, error)
                continue
        
        return None
    
    def _build_correction_prompt(
        self,
        call: Any,
        args: dict[str, Any],
        error: str,
        prior_results: dict[str, Any],
    ) -> str:
        """Build a prompt asking LLM to correct the tool call.
        
        Args:
            call: The failed ToolCall.
            args: The arguments that caused the error.
            error: The error message.
            prior_results: Results from previous calls for context.
            
        Returns:
            Prompt for the correction request.
        """
        # Get tool schema for reference
        tool = self.agent._get_tool(call.tool_name)
        schema_info = ""
        if tool and hasattr(tool, "parameters"):
            schema_info = f"\nTool schema: {json.dumps(tool.parameters, default=str)}"
        
        return f"""A tool call failed. Analyze the error and provide corrected arguments.

Tool: {call.tool_name}{schema_info}

Original arguments:
{json.dumps(args, indent=2, default=str)}

Error:
{error}

{f"Available results from prior steps: {list(prior_results.keys())}" if prior_results else ""}

Think about what caused this error:
1. Are the argument types correct?
2. Are required arguments missing?
3. Are there invalid values?

Respond with ONLY a JSON object containing the corrected arguments:
{{"arg_name": "corrected_value", ...}}"""
    
    def _parse_corrected_args(self, response: str) -> dict[str, Any] | None:
        """Parse corrected arguments from LLM response.
        
        Args:
            response: LLM response with corrected args.
            
        Returns:
            Corrected arguments dict or None if parsing fails.
        """
        json_match = re.search(r"\{[\s\S]*?\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
    
    async def _create_plan(
        self,
        task: str,
        context: dict[str, Any] | None,
    ) -> ExecutionPlan:
        """Ask agent to create an execution plan.
        
        Args:
            task: The task to plan for.
            context: Optional context.
            
        Returns:
            ExecutionPlan with tool calls.
        """
        tools_desc = self.agent.get_tool_descriptions()
        
        prompt = f"""Analyze this task and create an execution plan.

Task: {task}
{f"Context: {json.dumps(context)}" if context else ""}

Available tools (you can ONLY use these):
{tools_desc}

First, think step by step:
1. What information do I need to accomplish this task?
2. Which of the available tools can provide that information?
3. In what order should I call the tools?
4. What will I do with the results after getting them?

Then output a JSON plan:
{{
  "reasoning": "Brief explanation of your approach and what you'll do with results",
  "steps": [
    {{"tool": "tool_name", "args": {{}}}}
  ]
}}

Rules:
- Only include steps that require tools from the list above
- Use $call_N in args to reference the result from step N (0-indexed), e.g., $call_0 for first step's result
- After tool execution, you will synthesize the final answer yourself
- If no tools needed, use: {{"reasoning": "...", "steps": []}}
"""
        
        # Use neutral planner persona
        planner_prompt = "You are a task planner. Your job is to analyze tasks and output JSON execution plans. Never execute tools, only plan them."
        response = await self.agent.think(
            prompt, 
            include_tools=False, 
            system_prompt_override=planner_prompt,
        )
        return self._parse_plan(response)
    
    def _parse_plan(self, response: str) -> ExecutionPlan:
        """Parse plan from agent response.
        
        Args:
            response: Raw response containing JSON plan.
            
        Returns:
            Parsed ExecutionPlan.
        """
        plan = ExecutionPlan()
        
        # Try to extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for step in data.get("steps", []):
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
        """Substitute $call_N references with actual results.
        
        Args:
            args: Arguments with potential $references.
            results: Map of call_id -> result.
            
        Returns:
            Args with references resolved.
        """
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
        """Synthesize final answer from results.
        
        Args:
            task: Original task.
            plan: The execution plan.
            results: Map of call_id -> result.
            
        Returns:
            Synthesized answer string.
        """
        prompt = f"""Task: {task}

Tool results:
{json.dumps(results, indent=2, default=str)}

Based on these results, provide a final answer."""
        
        return await self.agent.think(prompt)
