"""
Role-specific agent behaviors and prompts.

Roles define CAPABILITIES (what an agent CAN do), not personalities.
The actual behavior comes from the agent's instructions.

Role System:
────────────
┌─────────────┬────────────┬──────────────┬───────────────┐
│ Role        │ can_finish │ can_delegate │ can_use_tools │
├─────────────┼────────────┼──────────────┼───────────────┤
│ WORKER      │     ❌     │      ❌      │      ✅       │
│ SUPERVISOR  │     ✅     │      ✅      │      ❌       │
│ AUTONOMOUS  │     ✅     │      ❌      │      ✅       │
│ REVIEWER    │     ✅     │      ❌      │      ❌       │
└─────────────┴────────────┴──────────────┴───────────────┘

Usage:
    # Worker does tasks, can't finish
    analyst = Agent(name="Analyst", role="worker", tools=[search])
    
    # Supervisor manages, can finish
    manager = Agent(name="Manager", role="supervisor")
    
    # Autonomous works alone, can finish
    assistant = Agent(name="Assistant", role="autonomous", tools=[search])
    
    # Reviewer approves/rejects, can finish
    qa = Agent(name="QA", role="reviewer")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.core.enums import AgentRole, get_role_capabilities

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


# =============================================================================
# Role Prompts - Default system prompts for each role
# =============================================================================

ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.WORKER: """You are a worker that executes tasks.

Your capabilities:
- Execute tasks using your tools
- Report results clearly
- Pass work to other agents when needed

You CANNOT finish the flow - pass work to a supervisor or reviewer when done.
""",

    AgentRole.SUPERVISOR: """You are a supervisor that manages work.

Your capabilities:
- Delegate tasks to workers
- Review and synthesize results
- Make final decisions

When delegating: "DELEGATE TO [agent]: [task]"
When done: "FINAL ANSWER: [result]"

You do NOT use tools directly - delegate tool work to workers.
""",

    AgentRole.AUTONOMOUS: """You are an autonomous agent that works independently.

Your capabilities:
- Execute tasks using your tools
- Make decisions
- Finish when done

When done: "FINAL ANSWER: [result]"
""",

    AgentRole.REVIEWER: """You are a reviewer that evaluates work.

Your capabilities:
- Review submitted work
- Approve or request revisions
- Make final decisions

Structure your review:
- APPROVED: [reason] → ends the flow
- REVISION NEEDED: [feedback] → work continues

When approved: "FINAL ANSWER: [approved result]"
""",
}


# =============================================================================
# Role Behaviors - Runtime behavior configuration
# =============================================================================

@dataclass
class RoleBehavior:
    """Runtime behavior for a role."""
    
    can_delegate: bool = False
    can_finish: bool = True
    can_use_tools: bool = True
    max_iterations: int | None = None

    @classmethod
    def for_role(cls, role: AgentRole) -> "RoleBehavior":
        """Create behavior from role capabilities."""
        caps = get_role_capabilities(role)
        return cls(
            can_delegate=caps["can_delegate"],
            can_finish=caps["can_finish"],
            can_use_tools=caps["can_use_tools"],
        )


def get_role_prompt(role: AgentRole) -> str:
    """Get default prompt for a role."""
    return ROLE_PROMPTS.get(role, ROLE_PROMPTS[AgentRole.AUTONOMOUS])


def get_role_behavior(role: AgentRole) -> RoleBehavior:
    """Get behavior for a role."""
    return RoleBehavior.for_role(role)


# Backward compatibility - ROLE_BEHAVIORS dict
ROLE_BEHAVIORS: dict[AgentRole, RoleBehavior] = {
    role: RoleBehavior.for_role(role) for role in AgentRole
}


# =============================================================================
# Delegation Parsing - Extract commands from agent output
# =============================================================================

@dataclass
class DelegationCommand:
    """A parsed command from agent output."""
    
    action: str  # "delegate", "final_answer", "route"
    target: str | None = None
    task: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_delegation(text: str) -> DelegationCommand | None:
    """Parse delegation commands from output.
    
    Supported formats:
    - DELEGATE TO [agent]: [task]
    - FINAL ANSWER: [answer]
    - ROUTE TO [agent]: [message]
    """
    import re
    
    text = text.strip()
    
    # DELEGATE TO
    match = re.search(
        r"DELEGATE\s+TO\s+\[?(\w+)\]?\s*:\s*(.+)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match:
        return DelegationCommand(
            action="delegate",
            target=match.group(1),
            task=match.group(2).strip(),
        )
    
    # FINAL ANSWER
    match = re.search(
        r"FINAL\s+ANSWER\s*:\s*(.+)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match:
        return DelegationCommand(
            action="final_answer",
            task=match.group(1).strip(),
        )
    
    # ROUTE TO
    match = re.search(
        r"ROUTE\s+TO\s+\[?(\w+)\]?\s*:\s*(.+)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match:
        return DelegationCommand(
            action="route",
            target=match.group(1),
            task=match.group(2).strip(),
        )
    
    return None


def has_final_answer(text: str) -> bool:
    """Check if text contains a final answer."""
    import re
    return bool(re.search(r"FINAL\s+ANSWER\s*:", text, re.IGNORECASE))


def extract_final_answer(text: str) -> str | None:
    """Extract final answer from text."""
    cmd = parse_delegation(text)
    if cmd and cmd.action == "final_answer":
        return cmd.task
    return None
