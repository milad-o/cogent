"""
ReAct (Reason + Act) execution strategy.

The classic think-act-observe loop. Simple but sequential.
Use for simple tasks or when steps truly depend on each other.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agenticflow.core.utils import now_utc
from agenticflow.graphs.base import BaseExecutor


class ReActExecutor(BaseExecutor):
    """
    Classic ReAct (Reason + Act) execution.
    
    Pattern: Think → Act → Observe → Repeat
    
    The agent thinks about what to do, takes an action (tool call),
    observes the result, and repeats until it has a final answer.
    
    Pros:
        - Simple and predictable
        - Good for debugging
        - Works well for simple tasks
        
    Cons:
        - Sequential - no parallelism
        - Slow for complex tasks
        - Each step waits for the previous
        
    Example:
        executor = ReActExecutor(agent)
        result = await executor.execute("What's the weather in NYC?")
    """
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using ReAct pattern.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The final answer from the agent.
        """
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
        """Build prompt with task and observations.
        
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
