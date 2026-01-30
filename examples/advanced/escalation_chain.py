"""Escalation chain: subagent → orchestrator → human.

Scenario:
- A compliance subagent reviews a press release for risky claims.
- If it finds a high-severity issue, it escalates to the orchestrator.
- The orchestrator then escalates to a human via HITL.

Usage:
    uv run python examples/advanced/escalation_chain.py
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from cogent import Agent, HumanDecision, InterruptedException, Observer
from cogent.tools.base import tool


# =============================================================================
# Structured Outputs
# =============================================================================


class ComplianceFinding(BaseModel):
    """Result of compliance review."""

    decision: Literal["OK", "ESCALATE"] = Field(
        description="OK if safe, ESCALATE if risky or ambiguous"
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Severity of the risk"
    )
    issues: list[str] = Field(default_factory=list, description="Issues found")
    recommendation: str = Field(description="What to do next")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Demonstrate escalation chain with HITL."""
    print("=" * 60)
    print("Escalation Chain: Subagent → Orchestrator → Human")
    print("=" * 60)
    print()

    observer = Observer(level="progress")

    # -------------------------------------------------------------------------
    # Subagent: compliance reviewer
    # -------------------------------------------------------------------------
    compliance_reviewer = Agent(
        name="ComplianceReviewer",
        model="gemini-2.5-flash",
        output=ComplianceFinding,
        instructions="""Review the press release for risky or unverifiable claims.

Escalate if you see:
- Absolute guarantees (e.g., "guaranteed results")
- Medical/financial claims without evidence
- Ambiguous or unverified data points

Return:
- decision: OK or ESCALATE
- severity: low/medium/high
- issues: list of issues found
- recommendation: what to do next""",
    )

    # -------------------------------------------------------------------------
    # HITL escalation tool
    # -------------------------------------------------------------------------
    @tool
    def escalate_to_human(summary: str, issues: list[str], severity: str) -> str:
        """Escalate to a human reviewer when risks are detected."""
        return f"Escalated to human ({severity}). Summary: {summary}"

    # -------------------------------------------------------------------------
    # Orchestrator
    # -------------------------------------------------------------------------
    orchestrator = Agent(
        name="ReleaseOrchestrator",
        model="gemini-2.5-flash",
        tools=[
            compliance_reviewer.as_tool(
                name="review_compliance",
                description="Review press release for compliance risks",
            ),
            escalate_to_human,
        ],
        instructions="""You orchestrate review of a press release.

Steps:
1. Call review_compliance with the full press release.
2. If decision is ESCALATE or severity is high, call escalate_to_human.
3. Otherwise, approve and return the final text.

Always include the full press release when calling review_compliance.""",
        interrupt_on={"escalate_to_human": True},
        observer=observer,
    )

    # -------------------------------------------------------------------------
    # Example input (intentionally risky claim)
    # -------------------------------------------------------------------------
    press_release = (
        "Introducing the ZenoSleep Patch, the only wearable that guarantees "
        "a 100% improvement in sleep quality within 7 days. Backed by "
        "proprietary research and clinically proven results, it works for "
        "everyone, every night."
    )

    print("Press release:")
    print(press_release)
    print()

    try:
        result = await orchestrator.run(
            f"Review this press release for compliance risks:\n\n{press_release}"
        )
        print("Approved:\n")
        print(result.content)

    except InterruptedException as e:
        pending = e.state.pending_actions[0]
        print("\n" + "=" * 60)
        print("🚨 HUMAN REVIEW REQUIRED")
        print("=" * 60)
        print(f"Severity: {pending.args.get('severity', 'N/A')}")
        print(f"Issues: {pending.args.get('issues', [])}")
        print(f"Summary: {pending.args.get('summary', 'N/A')}")
        print()

        # Simulate human decision
        print("[Simulating: Human REVISES]")
        decision = HumanDecision.reject(pending.action_id)
        result = await orchestrator.resume(decision)
        print("\nHuman decision recorded.")
        print(result.content)

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(observer.summary())


if __name__ == "__main__":
    asyncio.run(main())
