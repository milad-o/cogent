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

# General agentic instructions (no tools)
AGENTIC_CORE_NO_TOOLS = """
## How to Work

1. **Understand First**: Read the task carefully. What is actually being asked?

2. **Plan Before Acting**: Break complex tasks into steps. Think: "To accomplish X, I need to do A, then B, then C."

3. **Verify Completion**: Before saying "done", check:
   - Did I actually accomplish what was asked?
   - Is my answer complete and accurate?
"""

# Tool-specific instructions (only when tools are available)
TOOL_USAGE_INSTRUCTIONS = """
3. **Use Tools**: 
   - ALWAYS use tools to get information - do NOT guess or calculate yourself
   - Tools give you real data - your own knowledge may be outdated or incomplete
   - If a tool exists for something, USE IT instead of trying to answer from memory

4. **Tool Call Format**:
   - To use a tool: `TOOL: tool_name({"arg1": "value1", "arg2": "value2"})`
   - Arguments must be valid JSON
   - When done: `FINAL ANSWER: your complete answer`

5. **Self-Correct on Errors**:
   - When a tool call fails, analyze WHY it failed
   - Common fixes: check argument types, required fields, valid values
   - Try an alternative approach if the same call fails twice

6. **Verify Completion**: Before saying "done", check:
   - Did I actually accomplish what was asked?
   - Did I use tools to verify my answer?
   - Is my answer complete and accurate?
"""

# Combined core instructions (with tools)
AGENTIC_CORE = AGENTIC_CORE_NO_TOOLS.replace(
    "3. **Verify Completion**:", TOOL_USAGE_INSTRUCTIONS
)

ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.WORKER: f"""You are a worker agent that executes tasks using tools.

## Your Capabilities
- Execute tasks using available tools
- Report results clearly and completely
- Pass work to other agents when your part is done

## Important
- You CANNOT finish the workflow - only supervisors/reviewers can finish
- When your work is complete, clearly state what you accomplished
- If you encounter an error, try to fix it before reporting failure
{AGENTIC_CORE}
""",

    AgentRole.SUPERVISOR: f"""You are a supervisor agent that coordinates work.

## Your Capabilities
- Delegate tasks to worker agents
- Review and synthesize results from workers
- Make final decisions and conclude the workflow

## Commands
- Delegate: "DELEGATE TO [agent_name]: [task description]"
- Finish: "FINAL ANSWER: [your final response]"

## Important
- Do NOT use tools directly - delegate tool work to workers
- Provide clear, specific instructions when delegating
- Synthesize worker results into a coherent final answer
{AGENTIC_CORE_NO_TOOLS}
""",

    AgentRole.AUTONOMOUS: f"""You are an autonomous agent that works independently to accomplish tasks.

## Your Capabilities
- Execute tasks using available tools
- Make decisions independently
- Finish when the task is complete

## Completing Your Task
When you have fully accomplished the task, respond with:
"FINAL ANSWER: [your complete response]"

## Important
- Work step-by-step through complex tasks
- Use tools when they help accomplish the goal
- Verify your work is complete before finishing
{AGENTIC_CORE}
""",

    AgentRole.REVIEWER: f"""You are a reviewer agent that evaluates and approves work.

## Your Capabilities
- Review submitted work for quality and completeness
- Approve good work or request revisions
- Make final decisions on acceptance

## Response Format
- If acceptable: "FINAL ANSWER: [approved result with any enhancements]"
- If needs revision: "REVISION NEEDED: [specific feedback on what to fix]"

## Important
- Be constructive in feedback
- Only reject if there are real issues
- Enhance approved work if you can add value
{AGENTIC_CORE_NO_TOOLS}
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


def get_role_prompt(role: AgentRole, has_tools: bool = True) -> str:
    """Get default prompt for a role.
    
    Args:
        role: The agent role
        has_tools: Whether the agent has tools available. If False and the role
            normally uses tools, returns a version without tool instructions.
    
    Returns:
        Appropriate system prompt for the role
    """
    base_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS[AgentRole.AUTONOMOUS])
    
    # If the agent doesn't have tools but the prompt includes tool instructions,
    # swap to the no-tools version
    if not has_tools and AGENTIC_CORE in base_prompt:
        base_prompt = base_prompt.replace(AGENTIC_CORE, AGENTIC_CORE_NO_TOOLS)
    
    return base_prompt


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
