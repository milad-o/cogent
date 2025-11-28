"""
ReAct (Reason + Act) execution strategy with self-correction.

The classic think-act-observe loop with intelligent error recovery.
When a tool call fails, the agent analyzes the error and retries with corrected arguments.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agenticflow.core.utils import now_utc
from agenticflow.graphs.base import BaseExecutor


class ReActExecutor(BaseExecutor):
    """
    ReAct execution with self-correction.
    
    Pattern: Think → Act → Observe → Self-Correct → Repeat
    
    The agent thinks about what to do, takes an action (tool call),
    observes the result, and repeats. When errors occur, it analyzes
    the error and attempts to correct its approach.
    
    Self-Correction Features:
        - Analyzes error messages to understand failures
        - Asks LLM to fix arguments when tool calls fail
        - Tracks error patterns to avoid repeating mistakes
        - Uses scratchpad for working memory
    
    Example:
        executor = ReActExecutor(agent)
        result = await executor.execute("What's the weather in NYC?")
    """
    
    # Max self-correction attempts per tool call
    max_corrections: int = 2
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using ReAct pattern with self-correction.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The final answer from the agent.
        """
        observations: list[dict[str, Any]] = []
        
        # Set goal in scratchpad
        self.agent.scratchpad.set_goal(task)
        
        for i in range(self.max_iterations):
            # Build prompt with observations and scratchpad context
            prompt = self._build_prompt(task, observations, context)
            
            # Think
            self._emit_step("think", {"iteration": i})
            response = await self.agent.think(prompt)
            
            # Parse for tool calls or final answer
            action = self._parse_response(response)
            
            if action["type"] == "final_answer":
                # Verify completion before returning
                checklist = self.agent.scratchpad.get_completion_checklist()
                if checklist["has_uncorrected_errors"] and i < self.max_iterations - 1:
                    # Still have uncorrected errors, add observation
                    observations.append({
                        "warning": "You indicated completion but there are uncorrected errors. Please verify."
                    })
                    continue
                return action["answer"]
            
            if action["type"] == "tool_call":
                # Try tool call with self-correction on failure
                result = await self._execute_with_correction(
                    action["tool"], 
                    action["args"],
                    observations,
                )
                observations.append(result)
        
        return f"Max iterations reached. Last observations: {observations[-1] if observations else 'none'}"
    
    async def _execute_with_correction(
        self,
        tool_name: str,
        args: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute a tool call with self-correction on failure.
        
        Args:
            tool_name: Name of the tool.
            args: Tool arguments.
            observations: List to append correction history.
            
        Returns:
            Observation dict with result or final error.
        """
        self._emit_step("act", {"tool": tool_name, "args": args})
        self._track_tool_call(tool_name, args)
        
        last_error: str | None = None
        current_args = args
        
        for attempt in range(self.max_corrections + 1):
            start_time = now_utc()
            try:
                result = await self.agent.act(tool_name, current_args)
                duration_ms = (now_utc() - start_time).total_seconds() * 1000
                self._track_tool_result(tool_name, result, duration_ms)
                
                # Success! Mark any previous error as corrected
                if attempt > 0:
                    self.agent.scratchpad.mark_error_corrected(tool_name, success=True)
                    self._emit_step("self_correct_success", {
                        "tool": tool_name,
                        "attempts": attempt + 1,
                    })
                
                return {
                    "tool": tool_name,
                    "args": current_args,
                    "result": str(result),
                    "corrected": attempt > 0,
                }
                
            except Exception as e:
                last_error = str(e)
                self._track_tool_error(tool_name, last_error)
                
                # Record in scratchpad for context
                self.agent.scratchpad.record_error(tool_name, current_args, e)
                
                # If we have more correction attempts, try to fix
                if attempt < self.max_corrections:
                    self._emit_step("self_correct_attempt", {
                        "tool": tool_name,
                        "error": last_error,
                        "attempt": attempt + 1,
                    })
                    
                    # Ask LLM to fix the arguments
                    corrected = await self._correct_tool_call(
                        tool_name, current_args, last_error
                    )
                    
                    if corrected and corrected != current_args:
                        current_args = corrected
                        continue
                
                # No more attempts or couldn't correct
                break
        
        # All attempts failed
        self.agent.scratchpad.mark_error_corrected(tool_name, success=False)
        return {
            "tool": tool_name,
            "args": args,
            "error": last_error,
            "correction_failed": True,
        }
    
    async def _correct_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        error: str,
    ) -> dict[str, Any] | None:
        """Ask LLM to correct a failed tool call.
        
        Uses Reflexion-style memory to check for known fixes first.
        
        Args:
            tool_name: Tool that failed.
            args: Arguments that caused failure.
            error: Error message.
            
        Returns:
            Corrected arguments dict, or None if can't correct.
        """
        # Check if we have a known fix from Reflexion memory
        known_fix = self.agent.scratchpad.get_known_fix(tool_name, error)
        
        # Get tool description if available
        tool = self.agent._get_tool(tool_name)
        tool_desc = getattr(tool, "description", "No description") if tool else "Unknown tool"
        
        # Build prompt with Reflexion context
        reflexion_hint = ""
        if known_fix:
            reflexion_hint = f"\n\n**Known fix from past experience**: {known_fix}\nTry applying this fix first."
        
        prompt = f"""A tool call failed. Analyze the error and provide corrected arguments.

Tool: {tool_name}
Description: {tool_desc}
Original args: {json.dumps(args)}
Error: {error}{reflexion_hint}

Common fixes:
- Check if required arguments are missing
- Check if argument types are correct (string vs int vs list)
- Check if values are valid (e.g., valid file paths, correct IDs)
- Check if argument names match the tool's expected parameters

Respond with ONLY the corrected JSON arguments, nothing else:
{{"param": "corrected_value"}}

If you cannot fix this error, respond with: CANNOT_FIX"""

        try:
            response = await self.agent.think(
                prompt, 
                include_tools=False,
                system_prompt_override="You are a debugging assistant. Output only valid JSON or CANNOT_FIX.",
            )
            
            response = response.strip()
            
            if "CANNOT_FIX" in response.upper():
                return None
            
            # Try to parse corrected args
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                corrected = json.loads(json_match.group())
                
                # Learn from this correction for future use
                if corrected != args:
                    strategy = f"Changed args from {args} to {corrected}"
                    self.agent.scratchpad.mark_error_corrected(
                        tool_name, 
                        success=True, 
                        strategy=strategy[:100]  # Truncate
                    )
                
                return corrected
            
        except Exception:
            pass
        
        return None
    
    def _build_prompt(
        self,
        task: str,
        observations: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> str:
        """Build prompt with task, observations, and scratchpad context.
        
        Args:
            task: The original task.
            observations: List of previous tool observations.
            context: Optional context.
            
        Returns:
            Formatted prompt string.
        """
        parts = [f"Task: {task}"]
        
        if context:
            parts.append(f"Context: {json.dumps(context)}")
        
        # Add scratchpad context (todos, notes, errors)
        scratchpad_ctx = self.agent.scratchpad.get_context()
        if scratchpad_ctx:
            parts.append(f"\n{scratchpad_ctx}")
        
        if observations:
            parts.append("\nPrevious observations:")
            for obs in observations:
                if "error" in obs:
                    corrected = " (correction failed)" if obs.get("correction_failed") else ""
                    parts.append(f"  - {obs['tool']}({obs['args']}) → ERROR: {obs['error']}{corrected}")
                elif "warning" in obs:
                    parts.append(f"  ⚠️ {obs['warning']}")
                else:
                    corrected = " (self-corrected)" if obs.get("corrected") else ""
                    parts.append(f"  - {obs['tool']}({obs['args']}) → {str(obs['result'])[:200]}{corrected}")
        
        parts.append("\nRespond with either:")
        parts.append("1. TOOL: <tool_name>(<args as json>)")
        parts.append("2. FINAL ANSWER: <your answer>")
        
        return "\n".join(parts)
    
    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse agent response for tool call or final answer.
        
        Args:
            response: Raw response from agent.
            
        Returns:
            Dict with "type" (tool_call or final_answer) and relevant data.
        """
        response_lower = response.lower()
        
        # Check for final answer
        if "final answer:" in response_lower:
            idx = response_lower.index("final answer:")
            answer = response[idx + 13:].strip()
            return {"type": "final_answer", "answer": answer}
        
        # Check for tool call - support both JSON and simple argument formats
        # Pattern 1: TOOL: name({json})
        # Pattern 2: TOOL: name(simple_value)
        # Pattern 3: TOOL: name("string")
        tool_match = re.search(r"TOOL:\s*(\w+)\s*\(([^)]*)\)", response, re.IGNORECASE | re.DOTALL)
        if tool_match:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2).strip()
            
            # Try to parse as JSON first
            if args_str:
                try:
                    args = json.loads(args_str)
                    if isinstance(args, str):
                        # JSON parsed to string like "ETLPipeline" -> need to wrap
                        args = self._infer_arg_name(tool_name, args)
                    elif not isinstance(args, dict):
                        args = self._infer_arg_name(tool_name, args)
                except json.JSONDecodeError:
                    # Not valid JSON - treat as simple value
                    # Remove quotes if present
                    value = args_str.strip('\'"')
                    args = self._infer_arg_name(tool_name, value)
            else:
                args = {}
                
            return {"type": "tool_call", "tool": tool_name, "args": args}
        
        # Default to final answer if no tool pattern found
        return {"type": "final_answer", "answer": response}
    
    def _infer_arg_name(self, tool_name: str, value: Any) -> dict[str, Any]:
        """Infer the argument name based on tool name.
        
        Different tools expect different parameter names. This maps
        common tool names to their expected first argument.
        
        Args:
            tool_name: Name of the tool being called.
            value: The value to wrap.
            
        Returns:
            Dict with inferred argument name.
        """
        # Common parameter name mappings
        param_mappings = {
            # CodebaseAnalyzer tools
            "find_classes": "pattern",
            "find_functions": "pattern",
            "find_callers": "function_name",
            "find_usages": "name",
            "find_subclasses": "base_class",
            "find_imports": "module_name",
            "get_definition": "name",
            # KnowledgeGraph tools
            "remember": "entity",
            "recall": "entity",
            "connect": "source",
            "query_knowledge": "pattern",
            "forget": "entity",
            "list_knowledge": "entity_type",
            # Generic fallbacks
            "search": "query",
            "get": "name",
            "find": "query",
        }
        
        param_name = param_mappings.get(tool_name, "input")
        return {param_name: value}
