"""
Reasoning - Extended thinking and self-correction for agents.

This module provides deliberate reasoning capabilities for agents,
enabling them to think through problems step-by-step, self-correct,
and iterate until confident in their solution.

Inspired by how humans and advanced AI systems (o1, Claude) approach
complex problems: reason first, then act, then verify.

Example - Enable reasoning for complex tasks:
    ```python
    from agenticflow import Agent
    from agenticflow.agent.reasoning import ReasoningConfig
    
    agent = Agent(
        name="Analyst",
        model=model,
        reasoning=ReasoningConfig.standard(),  # Enable reasoning
    )
    
    # Agent will think through the problem before acting
    result = await agent.run(
        "Analyze the codebase and suggest architectural improvements"
    )
    ```

Example - Per-call reasoning:
    ```python
    # Simple tasks - no reasoning needed
    result = await agent.run("What time is it?")
    
    # Complex tasks - enable reasoning for this call
    result = await agent.run(
        "Design a database schema for an e-commerce platform",
        reasoning=True,  # Enable reasoning for this call
    )
    ```

Example - Custom reasoning configuration:
    ```python
    from agenticflow.agent.reasoning import ReasoningConfig
    
    result = await agent.run(
        "Debug this complex issue",
        reasoning=ReasoningConfig(
            max_thinking_rounds=5,
            show_thinking=True,  # Show reasoning in output
            thinking_budget=2000,  # Max tokens for thinking
            style="analytical",  # Reasoning style
        ),
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


class ReasoningStyle(str, Enum):
    """
    Reasoning styles that guide how the agent thinks.
    
    - ANALYTICAL: Step-by-step logical breakdown
    - EXPLORATORY: Consider multiple approaches
    - CRITICAL: Question assumptions, find flaws
    - CREATIVE: Generate novel solutions
    """
    
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"
    CRITICAL = "critical"
    CREATIVE = "creative"


@dataclass
class ReasoningConfig:
    """
    Configuration for agent reasoning mode.
    
    When reasoning is enabled, the agent will:
    1. Analyze the problem before acting
    2. Break down complex tasks into steps
    3. Consider multiple approaches
    4. Self-correct and refine its plan
    5. Execute with confidence
    
    Attributes:
        enabled: Whether reasoning is enabled (default: True)
        max_thinking_rounds: Maximum rounds of thinking before acting (1-10)
        thinking_budget: Token budget for thinking (None = unlimited)
        show_thinking: Whether to include thinking in output (default: False)
        style: Reasoning style (analytical, exploratory, critical, creative)
        require_confidence: Minimum confidence before acting (0.0-1.0, None = disabled)
        self_correct: Enable self-correction after tool results (default: True)
        
    Example:
        ```python
        # Standard reasoning
        config = ReasoningConfig.standard()
        
        # Deep analysis
        config = ReasoningConfig.deep()
        
        # Quick reasoning
        config = ReasoningConfig.quick()
        
        # Custom
        config = ReasoningConfig(
            max_thinking_rounds=3,
            style=ReasoningStyle.ANALYTICAL,
            show_thinking=True,
        )
        ```
    """
    
    enabled: bool = True
    max_thinking_rounds: int = 3
    thinking_budget: int | None = None  # Token limit for thinking
    show_thinking: bool = False  # Include <thinking> in output
    style: ReasoningStyle = ReasoningStyle.ANALYTICAL
    require_confidence: float | None = None  # 0.0-1.0 confidence threshold
    self_correct: bool = True  # Re-evaluate after tool results
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_thinking_rounds < 1:
            raise ValueError("max_thinking_rounds must be at least 1")
        if self.max_thinking_rounds > 10:
            raise ValueError("max_thinking_rounds cannot exceed 10")
        if self.require_confidence is not None:
            if not 0.0 <= self.require_confidence <= 1.0:
                raise ValueError("require_confidence must be between 0.0 and 1.0")
    
    @classmethod
    def quick(cls) -> ReasoningConfig:
        """Quick reasoning - minimal overhead, 1 thinking round."""
        return cls(
            max_thinking_rounds=1,
            style=ReasoningStyle.ANALYTICAL,
            show_thinking=False,
            self_correct=False,
        )
    
    @classmethod
    def standard(cls) -> ReasoningConfig:
        """Standard reasoning - balanced speed and depth."""
        return cls(
            max_thinking_rounds=3,
            style=ReasoningStyle.ANALYTICAL,
            show_thinking=False,
            self_correct=True,
        )
    
    @classmethod
    def deep(cls) -> ReasoningConfig:
        """Deep reasoning - thorough analysis for complex problems."""
        return cls(
            max_thinking_rounds=5,
            style=ReasoningStyle.EXPLORATORY,
            show_thinking=True,
            self_correct=True,
            require_confidence=0.7,
        )
    
    @classmethod
    def critical(cls) -> ReasoningConfig:
        """Critical reasoning - question assumptions, find flaws."""
        return cls(
            max_thinking_rounds=4,
            style=ReasoningStyle.CRITICAL,
            show_thinking=True,
            self_correct=True,
        )


@dataclass
class ThinkingStep:
    """A single step in the reasoning process."""
    
    round: int
    thought: str
    reasoning_type: str  # "analysis", "plan", "reflection", "correction"
    confidence: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "thought": self.thought,
            "reasoning_type": self.reasoning_type,
            "confidence": self.confidence,
        }


@dataclass 
class ReasoningResult:
    """Result of the reasoning process."""
    
    thinking_steps: list[ThinkingStep] = field(default_factory=list)
    final_plan: str | None = None
    total_thinking_tokens: int = 0
    thinking_rounds: int = 0
    
    @property
    def thinking_summary(self) -> str:
        """Get a summary of the thinking process."""
        if not self.thinking_steps:
            return ""
        return "\n".join(
            f"[Round {s.round}] {s.reasoning_type}: {s.thought[:100]}..."
            for s in self.thinking_steps
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "thinking_steps": [s.to_dict() for s in self.thinking_steps],
            "final_plan": self.final_plan,
            "total_thinking_tokens": self.total_thinking_tokens,
            "thinking_rounds": self.thinking_rounds,
        }


# =============================================================================
# Reasoning Prompts
# =============================================================================

REASONING_SYSTEM_PROMPT = """You are a thoughtful AI assistant that reasons carefully before acting.

When given a task, you MUST follow this process:

1. **ANALYZE**: Understand what is being asked. Identify key requirements and constraints.

2. **PLAN**: Break down the task into clear steps. Consider what tools or information you need.

3. **REFLECT**: Check your understanding. Are there ambiguities? What could go wrong?

4. **EXECUTE**: Carry out your plan, one step at a time.

5. **VERIFY**: After acting, check if the result makes sense. Self-correct if needed.

{style_instructions}

IMPORTANT: You MUST wrap your reasoning in <thinking></thinking> tags like this:
<thinking>
Step 1: First, I need to understand what is being asked...
Step 2: The key requirements are...
Step 3: My plan is to...
</thinking>

Then proceed with your response or tool calls."""


STYLE_INSTRUCTIONS = {
    ReasoningStyle.ANALYTICAL: """
**Analytical Style**:
- Break problems into logical components
- Identify cause and effect relationships
- Use structured step-by-step reasoning
- Support conclusions with evidence""",
    
    ReasoningStyle.EXPLORATORY: """
**Exploratory Style**:
- Consider multiple possible approaches
- Weigh pros and cons of each option
- Think about edge cases and alternatives
- Be open to unexpected solutions""",
    
    ReasoningStyle.CRITICAL: """
**Critical Style**:
- Question assumptions in the problem
- Look for potential flaws or gaps
- Consider what could go wrong
- Validate information before using it""",
    
    ReasoningStyle.CREATIVE: """
**Creative Style**:
- Think beyond conventional solutions
- Combine ideas in novel ways
- Consider unconventional approaches
- Don't be afraid to propose bold ideas""",
}


SELF_CORRECTION_PROMPT = """
Based on the tool results above, reflect on your approach:

<thinking>
1. Did the results match your expectations?
2. Is there anything you missed or should reconsider?
3. Should you adjust your approach?
</thinking>

If corrections are needed, explain what you'll do differently. Otherwise, proceed with the next step."""


def build_reasoning_prompt(
    config: ReasoningConfig,
    task: str,
    context: dict[str, Any] | None = None,
) -> str:
    """
    Build the reasoning-enhanced prompt for a task.
    
    Args:
        config: Reasoning configuration
        task: The task to reason about
        context: Optional context dict
        
    Returns:
        Enhanced prompt with reasoning instructions
    """
    style_instructions = STYLE_INSTRUCTIONS.get(config.style, "")
    
    system_context = REASONING_SYSTEM_PROMPT.format(
        style_instructions=style_instructions
    )
    
    prompt_parts = [
        f"Task: {task}",
    ]
    
    if context:
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        prompt_parts.append(f"\nContext:\n{context_str}")
    
    prompt_parts.append("\nBefore acting, think through this carefully:")
    
    return "\n".join(prompt_parts)


def extract_thinking(response: str) -> tuple[str | None, str]:
    """
    Extract <thinking> blocks from a response.
    
    Args:
        response: The full response text
        
    Returns:
        Tuple of (thinking_content, response_without_thinking)
    """
    import re
    
    thinking_pattern = r"<thinking>(.*?)</thinking>"
    matches = re.findall(thinking_pattern, response, re.DOTALL)
    
    thinking_content = "\n".join(m.strip() for m in matches) if matches else None
    
    # Remove thinking blocks from response
    cleaned_response = re.sub(thinking_pattern, "", response, flags=re.DOTALL).strip()
    
    return thinking_content, cleaned_response


def estimate_confidence(thinking: str) -> float:
    """
    Estimate confidence level from thinking content.
    
    This is a heuristic - looks for confidence indicators in the text.
    
    Args:
        thinking: The thinking content
        
    Returns:
        Estimated confidence (0.0-1.0)
    """
    thinking_lower = thinking.lower()
    
    # High confidence indicators
    high_confidence = [
        "i'm confident", "i am confident", "clearly", "definitely",
        "this will work", "straightforward", "simple", "obvious",
    ]
    
    # Low confidence indicators  
    low_confidence = [
        "not sure", "uncertain", "unclear", "might", "maybe",
        "could be", "need more", "ambiguous", "risky",
    ]
    
    high_count = sum(1 for phrase in high_confidence if phrase in thinking_lower)
    low_count = sum(1 for phrase in low_confidence if phrase in thinking_lower)
    
    # Base confidence
    confidence = 0.6
    
    # Adjust based on indicators
    confidence += high_count * 0.1
    confidence -= low_count * 0.1
    
    # Clamp to valid range
    return max(0.1, min(0.95, confidence))
