"""
LATS (Language Agent Tree Search) execution strategy.

Implements Monte Carlo Tree Search for LLM agents, enabling:
- Exploration of multiple reasoning paths
- Backtracking when paths fail
- Self-reflection to learn from failures
- LLM-powered value functions for path selection

Based on: "Language Agent Tree Search Unifies Reasoning Acting and Planning"
(Zhou et al., 2024) - achieves 92.7% on HumanEval with GPT-4.

Key insight: Instead of committing to a single reasoning path (like ReAct),
LATS explores multiple paths and uses self-reflection to guide search.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticflow.core.utils import generate_id, now_utc
from agenticflow.executors.base import BaseExecutor


class NodeState(Enum):
    """State of a search tree node."""
    PENDING = "pending"      # Not yet expanded
    EXPANDED = "expanded"    # Children generated
    TERMINAL = "terminal"    # Reached final answer or max depth
    FAILED = "failed"        # This path failed
    SUCCESS = "success"      # Found correct answer


@dataclass
class SearchNode:
    """A node in the LATS search tree.
    
    Each node represents a state in the reasoning process,
    including the action taken to reach it and its evaluation.
    
    Attributes:
        id: Unique node identifier.
        parent: Parent node (None for root).
        action: Action that led to this state (tool call or thought).
        observation: Result of the action.
        state: Current node state.
        value: LLM-estimated value (0.0-1.0) of this path.
        visits: Number of times this node was visited (for UCB1).
        children: Child nodes from this state.
        depth: Depth in the tree (root = 0).
        reflection: Self-reflection if this path failed.
    """
    id: str
    parent: "SearchNode | None" = None
    action: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    state: NodeState = NodeState.PENDING
    value: float = 0.5  # Prior value estimate
    visits: int = 0
    children: list["SearchNode"] = field(default_factory=list)
    depth: int = 0
    reflection: str | None = None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.state in (NodeState.TERMINAL, NodeState.SUCCESS, NodeState.FAILED)
    
    def ucb1_score(self, exploration_weight: float = 1.414) -> float:
        """Calculate UCB1 score for node selection.
        
        UCB1 balances exploitation (high value) and exploration (low visits).
        
        Args:
            exploration_weight: Controls exploration vs exploitation.
            
        Returns:
            UCB1 score for this node.
        """
        if self.visits == 0:
            return float("inf")  # Unvisited nodes get priority
        
        parent_visits = self.parent.visits if self.parent else 1
        exploitation = self.value
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploitation + exploration
    
    def backpropagate(self, value: float) -> None:
        """Backpropagate value up the tree.
        
        Updates visit counts and values for this node and all ancestors.
        
        Args:
            value: The value to propagate (0.0-1.0).
        """
        node: SearchNode | None = self
        while node is not None:
            node.visits += 1
            # Running average of values
            node.value = node.value + (value - node.value) / node.visits
            node = node.parent
    
    def get_path(self) -> list["SearchNode"]:
        """Get the path from root to this node."""
        path = []
        node: SearchNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "action": self.action,
            "observation": self.observation[:200] if self.observation else "",
            "state": self.state.value,
            "value": round(self.value, 3),
            "visits": self.visits,
            "depth": self.depth,
            "children_count": len(self.children),
        }


@dataclass
class TreeSearchResult:
    """Result of tree search execution.
    
    Attributes:
        answer: The final answer found.
        success: Whether a successful path was found.
        best_path: The best path through the tree.
        nodes_explored: Total nodes explored.
        iterations: Number of MCTS iterations.
        reflections: Reflections generated during search.
    """
    answer: str
    success: bool
    best_path: list[SearchNode]
    nodes_explored: int
    iterations: int
    reflections: list[str] = field(default_factory=list)


class TreeSearchExecutor(BaseExecutor):
    """
    LATS (Language Agent Tree Search) execution strategy.
    
    Uses Monte Carlo Tree Search to explore multiple reasoning paths,
    with LLM-powered value functions and self-reflection on failures.
    
    Pattern:
        1. SELECT: Choose promising node using UCB1
        2. EXPAND: Generate possible actions from that state
        3. EVALUATE: Use LLM to estimate path value
        4. BACKPROPAGATE: Update values up the tree
        5. REFLECT: Learn from failed paths
    
    When to use:
        - Complex multi-step tasks where initial attempts may fail
        - Tasks requiring exploration of multiple approaches
        - When you want the agent to learn from mistakes
        
    Compared to other executors:
        - ReAct: Single path, no backtracking
        - DAG: Parallel execution, no exploration
        - TreeSearch: Multiple paths, backtracking, self-reflection
    
    Example:
        executor = TreeSearchExecutor(agent)
        executor.max_iterations = 10  # MCTS iterations
        executor.max_depth = 5        # Maximum tree depth
        executor.num_candidates = 3   # Actions to consider per expansion
        
        result = await executor.execute("Solve this complex problem")
    
    Performance:
        - 92.7% pass@1 on HumanEval (GPT-4) per LATS paper
        - Best for tasks where exploration matters
        - Higher latency but better accuracy than single-path methods
    """
    
    # MCTS parameters
    max_depth: int = 5              # Maximum tree depth
    num_candidates: int = 3         # Number of actions to generate per expansion
    exploration_weight: float = 1.414  # UCB1 exploration constant
    value_threshold: float = 0.3    # Minimum value to continue exploring
    
    # Reflection settings
    enable_reflection: bool = True  # Generate reflections on failures
    max_reflections: int = 5        # Maximum reflections to store
    
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute using LATS tree search.
        
        Args:
            task: The task to execute.
            context: Optional context dictionary.
            
        Returns:
            The best answer found, or summary if no success.
        """
        self._emit_step("tree_search_start", {"task": task})
        
        # Initialize root node
        root = SearchNode(
            id=generate_id("node"),
            action={"type": "root", "task": task},
            observation=f"Task: {task}",
            state=NodeState.EXPANDED,
        )
        
        reflections: list[str] = []
        nodes_explored = 1
        best_terminal: SearchNode | None = None
        
        for iteration in range(self.max_iterations):
            self._emit_step("mcts_iteration", {"iteration": iteration + 1})
            
            # 1. SELECT: Find most promising leaf node
            node = self._select(root)
            
            if node.is_terminal():
                # Already at terminal, check if it's the best
                if node.state == NodeState.SUCCESS:
                    if best_terminal is None or node.value > best_terminal.value:
                        best_terminal = node
                continue
            
            # 2. EXPAND: Generate candidate actions
            if node.depth < self.max_depth:
                children = await self._expand(node, task, context, reflections)
                nodes_explored += len(children)
                
                if not children:
                    node.state = NodeState.TERMINAL
                    continue
                
                # 3. SIMULATE & EVALUATE: Execute and evaluate each child
                for child in children:
                    # Execute the action
                    result = await self._simulate(child, task)
                    
                    # Check if terminal
                    if self._is_final_answer(result):
                        child.state = NodeState.SUCCESS
                        child.observation = result
                        
                        # Evaluate success
                        value = await self._evaluate(child, task, success=True)
                        child.value = value
                        
                        # 4. BACKPROPAGATE
                        child.backpropagate(value)
                        
                        if best_terminal is None or value > best_terminal.value:
                            best_terminal = child
                    else:
                        child.observation = result
                        
                        # Evaluate partial progress
                        value = await self._evaluate(child, task, success=False)
                        child.value = value
                        
                        # 4. BACKPROPAGATE
                        child.backpropagate(value)
                        
                        # Check if path is worth continuing
                        if value < self.value_threshold:
                            child.state = NodeState.FAILED
                            
                            # 5. REFLECT on failure
                            if self.enable_reflection and len(reflections) < self.max_reflections:
                                reflection = await self._reflect(child, task)
                                if reflection:
                                    reflections.append(reflection)
                                    child.reflection = reflection
                                    
                                    # Store in agent's Reflexion memory
                                    self.agent.taskboard.add_reflection(
                                        reflection, 
                                        "failure",
                                        task[:100],
                                    )
            else:
                # Max depth reached
                node.state = NodeState.TERMINAL
        
        self._emit_step("tree_search_complete", {
            "nodes_explored": nodes_explored,
            "iterations": self.max_iterations,
            "found_success": best_terminal is not None,
        })
        
        # Return best result
        if best_terminal and best_terminal.state == NodeState.SUCCESS:
            return best_terminal.observation
        
        # No successful path found - synthesize best effort answer
        return await self._synthesize_best_effort(root, task, reflections)
    
    def _select(self, root: SearchNode) -> SearchNode:
        """Select the most promising leaf node using UCB1.
        
        Traverses tree from root, always selecting child with
        highest UCB1 score until reaching a leaf.
        
        Args:
            root: Root node of search tree.
            
        Returns:
            Selected leaf node for expansion.
        """
        node = root
        
        while not node.is_leaf() and not node.is_terminal():
            # Select child with highest UCB1 score
            best_child = max(
                [c for c in node.children if not c.is_terminal()],
                key=lambda c: c.ucb1_score(self.exploration_weight),
                default=None,
            )
            
            if best_child is None:
                break
            
            node = best_child
        
        return node
    
    async def _expand(
        self,
        node: SearchNode,
        task: str,
        context: dict[str, Any] | None,
        reflections: list[str],
    ) -> list[SearchNode]:
        """Expand a node by generating candidate actions.
        
        Asks the LLM to generate multiple possible next actions,
        incorporating reflections from previous failures.
        
        Args:
            node: Node to expand.
            task: Original task.
            context: Optional context.
            reflections: Reflections from failed paths.
            
        Returns:
            List of child nodes for candidate actions.
        """
        self._emit_step("expand", {"node_id": node.id, "depth": node.depth})
        
        # Build path context
        path = node.get_path()
        path_context = self._format_path(path)
        
        # Build reflection context
        reflection_context = ""
        if reflections:
            reflection_context = "\n\n**Lessons learned from failed attempts:**\n"
            for i, r in enumerate(reflections[-3:], 1):  # Last 3 reflections
                reflection_context += f"{i}. {r}\n"
        
        # Get tool descriptions
        tools_desc = self.agent.get_tool_descriptions()
        
        prompt = f"""You are solving a task step by step. Generate {self.num_candidates} different possible next actions.

**Task:** {task}
{f"**Context:** {json.dumps(context)}" if context else ""}

**Progress so far:**
{path_context}
{reflection_context}
**Available tools:**
{tools_desc}

Generate {self.num_candidates} DIFFERENT possible next actions. For each action, provide either:
1. A tool call: TOOL: tool_name({{"arg": "value"}})
2. A final answer: FINAL ANSWER: <your answer>

Think about DIFFERENT approaches - don't just vary parameters, try different strategies.

Format your response as:
ACTION 1:
<reasoning>
<action>

ACTION 2:
<reasoning>
<action>

ACTION 3:
<reasoning>
<action>
"""
        
        response = await self.agent.think(
            prompt,
            include_tools=False,
            system_prompt_override="You are an expert problem solver exploring multiple approaches. Generate diverse, creative solutions.",
        )
        
        # Parse candidate actions
        children = []
        actions = self._parse_candidate_actions(response)
        
        for action in actions:
            child = SearchNode(
                id=generate_id("node"),
                parent=node,
                action=action,
                depth=node.depth + 1,
            )
            node.children.append(child)
            children.append(child)
        
        node.state = NodeState.EXPANDED
        return children
    
    def _parse_candidate_actions(self, response: str) -> list[dict[str, Any]]:
        """Parse candidate actions from LLM response.
        
        Args:
            response: LLM response with multiple actions.
            
        Returns:
            List of action dictionaries.
        """
        actions = []
        
        # Split by ACTION markers
        parts = re.split(r"ACTION\s*\d+\s*:", response, flags=re.IGNORECASE)
        
        for part in parts[1:]:  # Skip first (before ACTION 1)
            part = part.strip()
            if not part:
                continue
            
            # Check for FINAL ANSWER
            if "FINAL ANSWER:" in part.upper():
                idx = part.upper().index("FINAL ANSWER:")
                answer = part[idx + 13:].strip()
                actions.append({
                    "type": "final_answer",
                    "answer": answer,
                    "reasoning": part[:idx].strip(),
                })
                continue
            
            # Check for TOOL call
            tool_match = re.search(
                r"TOOL:\s*(\w+)\s*\(([^)]*)\)",
                part,
                re.IGNORECASE | re.DOTALL,
            )
            if tool_match:
                tool_name = tool_match.group(1)
                args_str = tool_match.group(2).strip()
                
                # Parse arguments
                try:
                    args = json.loads(args_str) if args_str else {}
                    if not isinstance(args, dict):
                        args = {"input": args}
                except json.JSONDecodeError:
                    args = {"input": args_str.strip('\'"')}
                
                # Extract reasoning (text before TOOL:)
                tool_idx = part.upper().index("TOOL:")
                reasoning = part[:tool_idx].strip()
                
                actions.append({
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": args,
                    "reasoning": reasoning,
                })
        
        return actions[:self.num_candidates]  # Limit to num_candidates
    
    async def _simulate(self, node: SearchNode, task: str) -> str:
        """Simulate executing an action at a node.
        
        Args:
            node: Node with action to simulate.
            task: Original task.
            
        Returns:
            Result of the action (observation or answer).
        """
        action = node.action
        
        if action.get("type") == "final_answer":
            return action.get("answer", "")
        
        if action.get("type") == "tool_call":
            tool_name = action.get("tool", "")
            args = action.get("args", {})
            
            self._emit_step("simulate_tool", {
                "node_id": node.id,
                "tool": tool_name,
            })
            
            self._track_tool_call(tool_name, args)
            
            try:
                start = now_utc()
                result = await self.agent.act(tool_name, args)
                duration_ms = (now_utc() - start).total_seconds() * 1000
                self._track_tool_result(tool_name, result, duration_ms)
                return str(result)
            except Exception as e:
                error_msg = str(e)
                self._track_tool_error(tool_name, error_msg)
                
                # Learn from this error
                self.agent.taskboard.record_error(tool_name, e)
                
                return f"ERROR: {error_msg}"
        
        return "Unknown action type"
    
    def _is_final_answer(self, result: str) -> bool:
        """Check if a result is a final answer.
        
        Args:
            result: Result string to check.
            
        Returns:
            True if this appears to be a final answer.
        """
        # Check if it's an error
        if result.startswith("ERROR:"):
            return False
        
        # Tool results are observations, not final answers
        # Final answers come from "final_answer" action type
        return False  # Let evaluation determine if we should stop
    
    async def _evaluate(
        self,
        node: SearchNode,
        task: str,
        success: bool,
    ) -> float:
        """Evaluate the value of a node using LLM.
        
        This is the "critic" that estimates how promising a path is.
        
        Args:
            node: Node to evaluate.
            task: Original task.
            success: Whether this is a successful terminal state.
            
        Returns:
            Value estimate (0.0-1.0).
        """
        if success and node.action.get("type") == "final_answer":
            # Verify the answer
            answer = node.action.get("answer", "")
            prompt = f"""Evaluate this answer to the task.

Task: {task}

Proposed Answer: {answer}

Rate the answer on a scale of 0.0 to 1.0:
- 1.0: Completely correct and well-explained
- 0.7-0.9: Mostly correct with minor issues
- 0.4-0.6: Partially correct or incomplete
- 0.1-0.3: Mostly incorrect
- 0.0: Completely wrong

Respond with ONLY a number between 0.0 and 1.0."""

            try:
                response = await self.agent.think(
                    prompt,
                    include_tools=False,
                    system_prompt_override="You are a precise evaluator. Output only a number.",
                )
                
                # Parse value
                match = re.search(r"(\d+\.?\d*)", response)
                if match:
                    value = float(match.group(1))
                    return max(0.0, min(1.0, value))
            except Exception:
                pass
            
            return 0.8  # Default for final answers
        
        # Evaluate partial progress
        path = node.get_path()
        path_context = self._format_path(path)
        
        prompt = f"""Evaluate progress toward solving this task.

Task: {task}

Progress so far:
{path_context}

Latest result: {node.observation[:500]}

Rate the progress on a scale of 0.0 to 1.0:
- 0.8-1.0: Very close to solution, just needs final step
- 0.5-0.7: Good progress, on the right track
- 0.3-0.4: Some progress but significant work remains
- 0.1-0.2: Little progress or wrong direction
- 0.0: No progress or completely wrong approach

Respond with ONLY a number between 0.0 and 1.0."""

        try:
            response = await self.agent.think(
                prompt,
                include_tools=False,
                system_prompt_override="You are a precise evaluator. Output only a number.",
            )
            
            match = re.search(r"(\d+\.?\d*)", response)
            if match:
                value = float(match.group(1))
                return max(0.0, min(1.0, value))
        except Exception:
            pass
        
        return 0.4  # Default for partial progress
    
    async def _reflect(self, node: SearchNode, task: str) -> str | None:
        """Generate reflection on a failed path.
        
        This implements the self-reflection component of LATS,
        helping the agent learn from failures.
        
        Args:
            node: Failed terminal node.
            task: Original task.
            
        Returns:
            Reflection string, or None if generation failed.
        """
        self._emit_step("reflect", {"node_id": node.id})
        
        path = node.get_path()
        path_context = self._format_path(path)
        
        prompt = f"""Analyze why this approach failed and what should be tried differently.

Task: {task}

Failed approach:
{path_context}

Final result: {node.observation[:500]}

Provide a brief (1-2 sentence) reflection on:
1. Why this approach didn't work
2. What different strategy should be tried

Be specific and actionable. Start with "Instead of..." or "The problem was..."."""

        try:
            reflection = await self.agent.think(
                prompt,
                include_tools=False,
                system_prompt_override="You are analyzing failures to improve future attempts. Be concise and specific.",
            )
            return reflection.strip()[:200]  # Limit length
        except Exception:
            return None
    
    def _format_path(self, path: list[SearchNode]) -> str:
        """Format a path through the tree for prompts.
        
        Args:
            path: List of nodes from root to current.
            
        Returns:
            Formatted string describing the path.
        """
        lines = []
        for i, node in enumerate(path):
            if node.action.get("type") == "root":
                continue
            
            action = node.action
            if action.get("type") == "tool_call":
                lines.append(f"Step {i}: Called {action.get('tool')}({action.get('args')})")
                if node.observation:
                    obs = node.observation[:200]
                    lines.append(f"  Result: {obs}")
            elif action.get("type") == "final_answer":
                lines.append(f"Step {i}: Proposed answer: {action.get('answer', '')[:200]}")
        
        return "\n".join(lines) if lines else "No actions taken yet."
    
    async def _synthesize_best_effort(
        self,
        root: SearchNode,
        task: str,
        reflections: list[str],
    ) -> str:
        """Synthesize best-effort answer when no successful path found.
        
        Args:
            root: Root of search tree.
            task: Original task.
            reflections: Reflections from failed attempts.
            
        Returns:
            Best-effort answer based on exploration.
        """
        # Find the highest-value leaf
        best_node = self._find_best_node(root)
        path_context = self._format_path(best_node.get_path()) if best_node else "No exploration completed."
        
        reflection_context = ""
        if reflections:
            reflection_context = "\n\nLessons learned:\n" + "\n".join(f"- {r}" for r in reflections)
        
        prompt = f"""The tree search did not find a definitive answer. Synthesize the best possible answer from what was learned.

Task: {task}

Best exploration path:
{path_context}
{reflection_context}

Based on the exploration, provide the best answer you can. If uncertain, explain what's known and what's missing."""

        return await self.agent.think(prompt)
    
    def _find_best_node(self, root: SearchNode) -> SearchNode | None:
        """Find the highest-value node in the tree.
        
        Args:
            root: Root node.
            
        Returns:
            Node with highest value, or None.
        """
        best: SearchNode | None = None
        best_value = -1.0
        
        def traverse(node: SearchNode) -> None:
            nonlocal best, best_value
            if node.visits > 0 and node.value > best_value:
                best = node
                best_value = node.value
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return best
    
    def get_tree_stats(self, root: SearchNode) -> dict[str, Any]:
        """Get statistics about the search tree.
        
        Args:
            root: Root node.
            
        Returns:
            Dictionary of tree statistics.
        """
        total_nodes = 0
        max_depth = 0
        terminal_nodes = 0
        success_nodes = 0
        
        def traverse(node: SearchNode) -> None:
            nonlocal total_nodes, max_depth, terminal_nodes, success_nodes
            total_nodes += 1
            max_depth = max(max_depth, node.depth)
            if node.is_terminal():
                terminal_nodes += 1
            if node.state == NodeState.SUCCESS:
                success_nodes += 1
            for child in node.children:
                traverse(child)
        
        traverse(root)
        
        return {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "terminal_nodes": terminal_nodes,
            "success_nodes": success_nodes,
        }
