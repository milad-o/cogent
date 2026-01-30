"""Tactical Multi-Agent Delegation: LLM-as-Judge with HITL.

This example demonstrates the Generator + Verifier pattern based on:
arXiv:2601.14652 - Multi-agent only helps for verification + parallel tasks

Features:
- Orchestrator with reasoning (o3-mini) manages the workflow
- TaskBoard for explicit state/task tracking
- Writer and Judge as agent-tools (different models for diversity)
- Dynamic model selection based on content complexity
- HITL when judge escalates

Usage:
    uv run python examples/advanced/tactical_delegation.py
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from cogent import Agent, HumanDecision, InterruptedException, Observer
from cogent.agent import ReasoningConfig
from cogent.tools.base import tool


# =============================================================================
# Structured Outputs
# =============================================================================


class ComplexityAssessment(BaseModel):
    """Assessment of content brief complexity."""

    level: Literal["simple", "moderate", "complex"] = Field(
        description="Complexity level of the content task"
    )
    factors: list[str] = Field(
        description="Factors contributing to complexity"
    )
    recommended_model: str = Field(
        description="Recommended model: gpt4-mini for simple, gpt4 for moderate, o3-mini for complex"
    )


class JudgeVerdict(BaseModel):
    """Structured verdict from the quality judge."""

    decision: Literal["APPROVE", "REVISE", "ESCALATE"] = Field(
        description="APPROVE if ready, REVISE for auto-fix, ESCALATE for human review"
    )
    quality_score: int = Field(ge=1, le=10, description="Overall quality score 1-10")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    suggestions: list[str] = Field(default_factory=list, description="Improvements to make")
    escalation_reason: str | None = Field(default=None, description="Why human review needed")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Content workflow with reasoning orchestrator and taskboard."""
    print("=" * 60)
    print("Tactical Delegation: Reasoning + TaskBoard + Dynamic Models")
    print("=" * 60)
    print()

    observer = Observer(level="progress")

    # -------------------------------------------------------------------------
    # Complexity assessor - determines which model to use for writing
    # -------------------------------------------------------------------------
    complexity_assessor = Agent(
        name="ComplexityAssessor",
        model="gpt4-mini",  # Fast, cheap for assessment
        output=ComplexityAssessment,
        instructions="""Assess content brief complexity.

simple: Basic announcement, single product, clear audience
moderate: Multiple features, nuanced tone, some constraints
complex: Sensitive topics, legal/regulatory, multiple stakeholders, ambiguity

Recommend model based on complexity:
- simple → gpt4-mini (fast, cost-effective)
- moderate → gpt4 (balanced)
- complex → o3-mini (reasoning for nuanced content)""",
    )

    # -------------------------------------------------------------------------
    # Writers - different models for different complexity levels
    # -------------------------------------------------------------------------
    writer_simple = Agent(
        name="ContentWriter-Fast",
        model="gpt4-mini",
        instructions="""You are a marketing copywriter for straightforward content.
Write clear, professional content. Follow the brief exactly.""",
    )

    writer_standard = Agent(
        name="ContentWriter-Standard",
        model="gpt4",
        instructions="""You are a marketing copywriter.
Write compelling, professional content. Balance creativity with clarity.
If feedback provided, address ALL issues.""",
    )

    writer_complex = Agent(
        name="ContentWriter-Advanced",
        model="o3-mini",  # Reasoning model for complex/sensitive content
        instructions="""You are a senior marketing copywriter for sensitive content.
Think carefully about tone, implications, and audience perception.
Balance marketing goals with accuracy and responsibility.""",
    )

    # -------------------------------------------------------------------------
    # Judge - uses Gemini for diversity
    # -------------------------------------------------------------------------
    judge = Agent(
        name="QualityJudge",
        model="gemini",
        output=JudgeVerdict,
        instructions="""You are a content quality judge.

Evaluate marketing copy for:
1. Clarity - Is the message clear?
2. Tone - Appropriate for the audience?
3. Accuracy - No misleading claims?
4. Effectiveness - Would it achieve its goal?

Decisions:
- APPROVE: Quality score 8+ with no major issues
- REVISE: Fixable issues, provide specific suggestions
- ESCALATE: Sensitive topics, legal concerns, or ambiguous requirements""",
    )

    # -------------------------------------------------------------------------
    # HITL escalation tool
    # -------------------------------------------------------------------------
    @tool
    def escalate_to_human(content: str, reason: str, score: int) -> str:
        """Escalate content to human for review when judge flags concerns."""
        return f"Escalated (score: {score}/10): {reason}"

    # -------------------------------------------------------------------------
    # Orchestrator with reasoning + taskboard
    # -------------------------------------------------------------------------
    orchestrator = Agent(
        name="ContentOrchestrator",
        model="gpt4",  # GPT-4 with reasoning config
        reasoning=ReasoningConfig.standard(),  # Enable reasoning for orchestration
        tools=[
            complexity_assessor.as_tool(
                name="assess_complexity",
                description="Assess brief complexity to select appropriate writer model",
            ),
            writer_simple.as_tool(
                name="draft_simple",
                description="Generate content for SIMPLE briefs (fast, cost-effective)",
            ),
            writer_standard.as_tool(
                name="draft_standard",
                description="Generate content for MODERATE complexity briefs",
            ),
            writer_complex.as_tool(
                name="draft_complex",
                description="Generate content for COMPLEX/sensitive briefs (uses reasoning)",
            ),
            judge.as_tool(
                name="judge_content",
                description="Evaluate content quality and decide: APPROVE, REVISE, or ESCALATE",
            ),
            escalate_to_human,
        ],
        instructions="""You orchestrate content creation with explicit task tracking.

WORKFLOW (use taskboard to track each step):

1. ASSESS: Use assess_complexity to determine brief complexity
   → Add task: "Assess complexity"
   → Result tells you which writer to use

2. DRAFT: Use the appropriate writer based on complexity
   - simple → draft_simple
   - moderate → draft_standard  
   - complex → draft_complex
   → Add task: "Generate draft"
   → SAVE the generated content for the next step

3. JUDGE: Use judge_content to evaluate
   → IMPORTANT: Pass BOTH the original brief AND the full content to judge_content
   → Format: "Brief: [brief]\n\nContent to evaluate:\n[full content]"
   → Add task: "Quality review"
   → If APPROVE: Complete workflow
   → If REVISE: Go back to step 2 with feedback (max 3 attempts)
   → If ESCALATE: Use escalate_to_human tool

4. COMPLETE: Return final approved content

CRITICAL: When calling judge_content, you MUST include the actual content text.
Do not just describe the content - pass the full text.

Use add_note to record important observations (complexity factors, judge feedback).
Use verify_task after each major step.""",
        taskboard=True,  # Enable taskboard for state tracking
        interrupt_on={"escalate_to_human": True},
        observer=observer,
    )

    # -------------------------------------------------------------------------
    # Execute workflow
    # -------------------------------------------------------------------------
    brief = """Write a short product announcement (2-3 sentences) for:
Product: EcoFlow Solar Panel
Audience: Environmentally-conscious homeowners
Key message: 30% more efficient than competitors, easy installation"""

    print(f"Brief: {brief}")
    print()

    try:
        result = await orchestrator.run(
            f"Create content for this brief:\n\n{brief}\n\nMax revision attempts: 3",
            max_iterations=40,  # Allow enough iterations for the full workflow
        )

        print()
        print("=" * 60)
        print("WORKFLOW COMPLETE")
        print("=" * 60)
        print(result.content)

        # Show taskboard state
        if orchestrator.taskboard:
            print()
            print("=" * 60)
            print("TASKBOARD STATUS")
            print("=" * 60)
            print(orchestrator.taskboard.summary())

    except InterruptedException as e:
        # HITL: Judge escalated to human
        pending = e.state.pending_actions[0]
        print()
        print("=" * 60)
        print("🚨 HUMAN REVIEW REQUIRED")
        print("=" * 60)
        print(f"Content: {pending.args.get('content', 'N/A')[:300]}...")
        print(f"Reason: {pending.args.get('reason', 'N/A')}")
        print(f"Score: {pending.args.get('score', 'N/A')}/10")
        print()

        # Simulate human decision
        print("[Simulating: Human APPROVES]")
        decision = HumanDecision.approve(pending.action_id)
        result = await orchestrator.resume(decision)

        print()
        print("✅ Human approved!")
        print(result.content)

    print()
    print("=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(observer.summary())


if __name__ == "__main__":
    asyncio.run(main())
