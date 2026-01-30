"""Tactical Multi-Agent Delegation: LLM-as-Judge with HITL.

This example demonstrates the Generator + Verifier pattern based on:
arXiv:2601.14652 - Multi-agent only helps for verification + parallel tasks

Features:
- Orchestrator (Gemini 2.5 Flash) manages the workflow with TaskBoard
- Writer agents at different complexity tiers (Flash/Flash/Pro)
- Judge agent (Gemini 2.5 Flash) for quality verification
- Dynamic model selection based on content complexity
- HITL when judge escalates

Usage:
    uv run python examples/advanced/tactical_delegation.py
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from cogent import Agent, HumanDecision, InterruptedException, Observer
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
        description="Recommended model: gemini-2.0-flash for simple, gemini-2.5-flash for moderate, gemini-2.5-pro for complex"
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
        model="gemini-2.0-flash",  # Fast, cheap for assessment
        output=ComplexityAssessment,
        instructions="""Assess content brief complexity.

simple: Basic announcement, single product, clear audience
moderate: Multiple features, nuanced tone, some constraints
complex: Sensitive topics, legal/regulatory, multiple stakeholders, ambiguity

Recommend model based on complexity:
- simple → gemini-2.0-flash (fast, cost-effective)
- moderate → gemini-2.5-flash (balanced)
- complex → gemini-2.5-pro (reasoning for nuanced content)""",
    )

    # -------------------------------------------------------------------------
    # Writers - different models for different complexity levels
    # -------------------------------------------------------------------------
    writer_simple = Agent(
        name="ContentWriter-Fast",
        model="gemini-2.0-flash",
        instructions="""You are a marketing copywriter for straightforward content.
Write clear, professional content. Follow the brief exactly.""",
    )

    writer_standard = Agent(
        name="ContentWriter-Standard",
        model="gemini-2.5-flash",
        instructions="""You are a marketing copywriter.
Write compelling, professional content. Balance creativity with clarity.
If feedback provided, address ALL issues.""",
    )

    writer_complex = Agent(
        name="ContentWriter-Advanced",
        model="gemini-2.5-pro",  # Pro model for complex/sensitive content
        instructions="""You are a senior marketing copywriter for sensitive content.
Think carefully about tone, implications, and audience perception.
Balance marketing goals with accuracy and responsibility.""",
    )

    # -------------------------------------------------------------------------
    # Judge - uses Gemini for diversity
    # -------------------------------------------------------------------------
    judge = Agent(
        name="QualityJudge",
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",  # Gemini 2.5 Flash
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
   → Execute assess_complexity
   → Update task to completed with result

2. DRAFT: Use the appropriate writer based on complexity
   - simple → draft_simple
   - moderate → draft_standard  
   - complex → draft_complex
   → Add task: "Generate draft"
   → Execute the writer
   → Update task to completed, SAVE the generated content

3. JUDGE: Use judge_content to evaluate
   → Add task: "Quality review"
   → Pass BOTH brief AND full content: "Brief: [brief]\n\nContent:\n[content]"
   → Update task to completed with verdict
   → If APPROVE: Go to step 4
   → If REVISE: Go back to step 2 with feedback (max 3 attempts)
   → If ESCALATE: Use escalate_to_human tool

4. COMPLETE: Return final approved content

IMPORTANT: Always update_task to 'completed' BEFORE moving to next step or finishing.""",
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
