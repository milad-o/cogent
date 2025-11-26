"""
Role-specific agent behaviors and prompts.

Each role has:
- Default system prompt defining behavior
- Role-specific methods (e.g., supervisor.delegate())
- Graph builder for appropriate execution pattern

Usage:
    # Quick creation with role defaults
    supervisor = Agent.as_supervisor(name="Manager", model=model)
    worker = Agent.as_worker(name="Analyst", model=model)
    
    # Or use role-specific classes directly
    from agenticflow.agent.roles import SupervisorAgent, WorkerAgent
    
    supervisor = SupervisorAgent(name="Manager", model=model, workers=["A", "B"])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agenticflow.core.enums import AgentRole

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


# =============================================================================
# Role Prompts - Default instructions for each role
# =============================================================================

ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.SUPERVISOR: """You are a supervisor agent responsible for coordinating a team.

Your responsibilities:
1. Analyze the task and break it down into subtasks
2. Delegate subtasks to appropriate workers based on their capabilities
3. Monitor progress and provide guidance when needed
4. Synthesize results from workers into a final output
5. Make final decisions when workers disagree

When delegating, specify:
- Which worker should handle the task
- What specific outcome you expect
- Any constraints or guidelines

Always end with either:
- A delegation: "DELEGATE TO [worker_name]: [task description]"
- A final answer: "FINAL ANSWER: [your synthesized response]"
""",

    AgentRole.COORDINATOR: """You are a coordinator agent that facilitates collaboration.

Your responsibilities:
1. Route messages between agents efficiently
2. Balance workload across the team
3. Help agents reach consensus when needed
4. Track conversation state and ensure nothing is dropped
5. Identify when tasks are complete

You don't make final decisions - you help others collaborate effectively.

When routing, use:
- "ROUTE TO [agent_name]: [message]"
- "BROADCAST: [message to all]"
- "CONSENSUS REACHED: [summary]"
""",

    AgentRole.PLANNER: """You are a planner agent that creates execution plans.

Your responsibilities:
1. Analyze complex tasks and break them into steps
2. Identify dependencies between steps
3. Estimate effort and assign priorities
4. Create clear, actionable plans

Output plans in this format:
PLAN:
1. [Step 1] - [assigned_to] - [dependencies: none/step_n]
2. [Step 2] - [assigned_to] - [dependencies: step_1]
...
END PLAN
""",

    AgentRole.WORKER: """You are a worker agent that executes tasks.

Your responsibilities:
1. Execute the task you're given thoroughly
2. Use your tools when appropriate
3. Report your progress and results clearly
4. Ask for clarification if the task is unclear

Be concise but complete in your responses.
""",

    AgentRole.SPECIALIST: """You are a specialist agent with domain expertise.

Your responsibilities:
1. Apply your specialized knowledge to tasks
2. Provide expert-level analysis and recommendations
3. Flag potential issues in your domain
4. Educate others when they need context

Always explain your reasoning, especially for non-obvious conclusions.
""",

    AgentRole.RESEARCHER: """You are a researcher agent that gathers information.

Your responsibilities:
1. Search for relevant information using available tools
2. Verify information from multiple sources when possible
3. Summarize findings clearly
4. Cite sources and note confidence levels

Structure your findings:
- Key facts discovered
- Sources used
- Confidence level (high/medium/low)
- Areas needing more research
""",

    AgentRole.CRITIC: """You are a critic agent that reviews and improves work.

Your responsibilities:
1. Review work products for quality and correctness
2. Identify issues, gaps, and areas for improvement
3. Provide constructive, actionable feedback
4. Suggest specific improvements

Structure your feedback:
- STRENGTHS: What's done well
- ISSUES: Problems to fix (with specifics)
- SUGGESTIONS: Improvements to consider
- VERDICT: Approved / Needs revision / Rejected
""",

    AgentRole.VALIDATOR: """You are a validator agent that ensures correctness.

Your responsibilities:
1. Verify outputs against requirements
2. Check for errors, inconsistencies, and edge cases
3. Ensure compliance with standards
4. Provide a clear pass/fail assessment

Structure your validation:
- CHECKS PERFORMED: [list]
- ISSUES FOUND: [list or "none"]
- STATUS: PASS / FAIL / CONDITIONAL PASS
- REQUIRED FIXES: [if any]
""",

    AgentRole.ORCHESTRATOR: """You are an orchestrator managing the overall flow.

Your responsibilities:
1. Start and stop agent workflows
2. Handle system-level events and errors
3. Ensure resources are properly allocated
4. Monitor overall progress toward goals

You operate at the system level, not on individual tasks.
""",
}


# =============================================================================
# Role Behaviors - What each role can do
# =============================================================================

@dataclass
class RoleBehavior:
    """Defines what a role can do."""
    
    can_delegate: bool = False
    can_finish: bool = True
    can_spawn_subtasks: bool = False
    can_use_tools: bool = True
    can_route_messages: bool = False
    max_iterations: int | None = None  # None = use default
    
    # Handoff rules
    allowed_targets: list[str] | None = None  # None = use topology default
    requires_approval: bool = False


ROLE_BEHAVIORS: dict[AgentRole, RoleBehavior] = {
    AgentRole.SUPERVISOR: RoleBehavior(
        can_delegate=True,
        can_finish=True,
        can_spawn_subtasks=True,
        can_use_tools=False,  # Supervisors delegate, not execute
        max_iterations=10,
    ),
    AgentRole.COORDINATOR: RoleBehavior(
        can_delegate=False,
        can_finish=False,  # Coordinators route, don't finish
        can_route_messages=True,
        can_use_tools=False,
    ),
    AgentRole.PLANNER: RoleBehavior(
        can_delegate=False,
        can_finish=True,
        can_spawn_subtasks=True,
        can_use_tools=False,
    ),
    AgentRole.WORKER: RoleBehavior(
        can_delegate=False,
        can_finish=True,
        can_use_tools=True,
    ),
    AgentRole.SPECIALIST: RoleBehavior(
        can_delegate=False,
        can_finish=True,
        can_use_tools=True,
    ),
    AgentRole.RESEARCHER: RoleBehavior(
        can_delegate=False,
        can_finish=True,
        can_use_tools=True,
    ),
    AgentRole.CRITIC: RoleBehavior(
        can_delegate=False,
        can_finish=True,
        can_use_tools=False,
    ),
    AgentRole.VALIDATOR: RoleBehavior(
        can_delegate=False,
        can_finish=True,
        can_use_tools=False,
    ),
    AgentRole.ORCHESTRATOR: RoleBehavior(
        can_delegate=True,
        can_finish=True,
        can_spawn_subtasks=True,
        can_use_tools=False,
        max_iterations=5,
    ),
}


def get_role_prompt(role: AgentRole) -> str:
    """Get default prompt for a role."""
    return ROLE_PROMPTS.get(role, ROLE_PROMPTS[AgentRole.WORKER])


def get_role_behavior(role: AgentRole) -> RoleBehavior:
    """Get behavior definition for a role."""
    return ROLE_BEHAVIORS.get(role, ROLE_BEHAVIORS[AgentRole.WORKER])


# =============================================================================
# Delegation Parsing - Extract delegation commands from LLM output
# =============================================================================

@dataclass
class DelegationCommand:
    """A parsed delegation command from supervisor output."""
    
    action: str  # "delegate", "final_answer", "route", "broadcast"
    target: str | None = None  # Worker/agent name
    task: str = ""  # Task description or content
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_delegation(text: str) -> DelegationCommand | None:
    """Parse delegation commands from LLM output.
    
    Supported formats:
    - DELEGATE TO [worker]: [task]
    - FINAL ANSWER: [answer]
    - ROUTE TO [agent]: [message]
    - BROADCAST: [message]
    """
    import re
    
    text = text.strip()
    
    # Check for DELEGATE TO
    delegate_match = re.search(
        r"DELEGATE\s+TO\s+\[?(\w+)\]?\s*:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if delegate_match:
        return DelegationCommand(
            action="delegate",
            target=delegate_match.group(1),
            task=delegate_match.group(2).strip(),
        )
    
    # Check for FINAL ANSWER
    final_match = re.search(
        r"FINAL\s+ANSWER\s*:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if final_match:
        return DelegationCommand(
            action="final_answer",
            task=final_match.group(1).strip(),
        )
    
    # Check for ROUTE TO
    route_match = re.search(
        r"ROUTE\s+TO\s+\[?(\w+)\]?\s*:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if route_match:
        return DelegationCommand(
            action="route",
            target=route_match.group(1),
            task=route_match.group(2).strip(),
        )
    
    # Check for BROADCAST
    broadcast_match = re.search(
        r"BROADCAST\s*:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if broadcast_match:
        return DelegationCommand(
            action="broadcast",
            task=broadcast_match.group(1).strip(),
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
