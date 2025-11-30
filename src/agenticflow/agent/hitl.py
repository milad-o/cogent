"""
Human-in-the-Loop (HITL) support for AgenticFlow.

Provides types and utilities for implementing human oversight of agent actions,
including tool approval workflows and interactive decision-making.

Uses an interrupt/resume pattern for seamless integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from datetime import datetime, timezone


class InterruptReason(Enum):
    """Reasons why agent execution was interrupted."""
    
    TOOL_APPROVAL = "tool_approval"  # Tool requires human approval
    HUMAN_INPUT = "human_input"  # Agent requests human input
    CONFIRMATION = "confirmation"  # Agent wants confirmation before proceeding
    ERROR_RECOVERY = "error_recovery"  # Error occurred, human decision needed
    CHECKPOINT = "checkpoint"  # Periodic checkpoint for review
    CUSTOM = "custom"  # Custom interrupt reason


class DecisionType(Enum):
    """Types of human decisions for pending actions."""
    
    APPROVE = "approve"  # Approve the action as-is
    REJECT = "reject"  # Reject the action entirely
    EDIT = "edit"  # Approve with modifications
    SKIP = "skip"  # Skip this action but continue
    ABORT = "abort"  # Abort the entire workflow
    GUIDE = "guide"  # Provide guidance for agent to reconsider/retry
    RESPOND = "respond"  # Provide a direct response (for human_input interrupts)


@dataclass
class PendingAction:
    """
    Represents an action pending human approval.
    
    Captures all context needed for a human to make an informed decision
    about whether to approve, modify, or reject an agent action.
    
    Attributes:
        action_id: Unique identifier for this pending action
        tool_name: Name of the tool to be executed
        args: Arguments that will be passed to the tool
        agent_name: Name of the agent requesting the action
        reason: Why this action requires approval
        context: Additional context for decision-making
        timestamp: When the action was queued
        metadata: Additional metadata
    """
    
    action_id: str
    tool_name: str
    args: dict[str, Any]
    agent_name: str
    reason: InterruptReason = InterruptReason.TOOL_APPROVAL
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/transmission."""
        return {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "agent_name": self.agent_name,
            "reason": self.reason.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingAction:
        """Deserialize from storage/transmission."""
        return cls(
            action_id=data["action_id"],
            tool_name=data["tool_name"],
            args=data["args"],
            agent_name=data["agent_name"],
            reason=InterruptReason(data.get("reason", "tool_approval")),
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
        )
    
    def describe(self) -> str:
        """Human-readable description of the pending action."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.tool_name}({args_str})"


@dataclass
class HumanDecision:
    """
    A human's decision about a pending action.
    
    Represents the human's response to a pending action, including
    any modifications, guidance, or direct responses.
    
    Attributes:
        action_id: ID of the pending action this decision applies to
        decision: Type of decision (approve, reject, edit, guide, respond, etc.)
        modified_args: If decision is EDIT, the modified arguments
        feedback: Optional human feedback/explanation
        guidance: If decision is GUIDE, instructions for agent to reconsider
        response: If decision is RESPOND, the human's direct answer
        timestamp: When the decision was made
    """
    
    action_id: str
    decision: DecisionType
    modified_args: dict[str, Any] | None = None
    feedback: str | None = None
    guidance: str | None = None  # Instructions for agent to reconsider
    response: Any = None  # Direct response value for RESPOND decisions
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/transmission."""
        return {
            "action_id": self.action_id,
            "decision": self.decision.value,
            "modified_args": self.modified_args,
            "feedback": self.feedback,
            "guidance": self.guidance,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanDecision:
        """Deserialize from storage/transmission."""
        return cls(
            action_id=data["action_id"],
            decision=DecisionType(data["decision"]),
            modified_args=data.get("modified_args"),
            feedback=data.get("feedback"),
            guidance=data.get("guidance"),
            response=data.get("response"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.now(timezone.utc),
        )

    @classmethod
    def approve(cls, action_id: str, feedback: str | None = None) -> HumanDecision:
        """Create an approval decision."""
        return cls(action_id=action_id, decision=DecisionType.APPROVE, feedback=feedback)
    
    @classmethod
    def reject(cls, action_id: str, feedback: str | None = None) -> HumanDecision:
        """Create a rejection decision."""
        return cls(action_id=action_id, decision=DecisionType.REJECT, feedback=feedback)
    
    @classmethod
    def edit(cls, action_id: str, modified_args: dict[str, Any], feedback: str | None = None) -> HumanDecision:
        """Create an edit decision with modified arguments."""
        return cls(action_id=action_id, decision=DecisionType.EDIT, modified_args=modified_args, feedback=feedback)
    
    @classmethod
    def skip(cls, action_id: str, feedback: str | None = None) -> HumanDecision:
        """Create a skip decision."""
        return cls(action_id=action_id, decision=DecisionType.SKIP, feedback=feedback)
    
    @classmethod
    def abort(cls, action_id: str, feedback: str | None = None) -> HumanDecision:
        """Create an abort decision."""
        return cls(action_id=action_id, decision=DecisionType.ABORT, feedback=feedback)
    
    @classmethod
    def guide(cls, action_id: str, guidance: str, feedback: str | None = None) -> HumanDecision:
        """
        Create a guidance decision - tells the agent to reconsider with new instructions.
        
        Use this when you want the agent to think again with your input, rather than
        simply approving or rejecting the proposed action.
        
        Args:
            action_id: ID of the pending action
            guidance: Instructions/guidance for the agent to consider
            feedback: Optional additional feedback
            
        Example:
            ```python
            # Agent wants to delete important.txt
            decision = HumanDecision.guide(
                action_id,
                guidance="Don't delete that file. Instead, archive it to /backup/ first, "
                         "then delete the original. Also check if any other files depend on it."
            )
            ```
        """
        return cls(action_id=action_id, decision=DecisionType.GUIDE, guidance=guidance, feedback=feedback)
    
    @classmethod
    def respond(cls, action_id: str, response: Any, feedback: str | None = None) -> HumanDecision:
        """
        Create a direct response - provides a value the agent requested.
        
        Use this when the agent asks for human input (not approval) and you
        want to provide a direct answer.
        
        Args:
            action_id: ID of the pending action
            response: The response value (string, dict, etc.)
            feedback: Optional additional context
            
        Example:
            ```python
            # Agent asks "What should I name the report?"
            decision = HumanDecision.respond(
                action_id,
                response="Q4-2024-Sales-Report"
            )
            ```
        """
        return cls(action_id=action_id, decision=DecisionType.RESPOND, response=response, feedback=feedback)


@dataclass
class InterruptedState:
    """
    State captured when agent execution is interrupted.
    
    Contains everything needed to resume execution after human decision.
    
    Attributes:
        thread_id: Thread identifier for resumption
        pending_actions: Actions awaiting human decision
        agent_state: Captured agent state at interrupt
        conversation_history: Conversation context
        interrupt_reason: Why execution was interrupted
        metadata: Additional state metadata
    """
    
    thread_id: str
    pending_actions: list[PendingAction]
    agent_state: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    interrupt_reason: InterruptReason = InterruptReason.TOOL_APPROVAL
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_interrupted(self) -> bool:
        """Check if there are pending actions requiring decisions."""
        return len(self.pending_actions) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "thread_id": self.thread_id,
            "pending_actions": [a.to_dict() for a in self.pending_actions],
            "agent_state": self.agent_state,
            "conversation_history": self.conversation_history,
            "interrupt_reason": self.interrupt_reason.value,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterruptedState:
        """Deserialize from storage."""
        return cls(
            thread_id=data["thread_id"],
            pending_actions=[PendingAction.from_dict(a) for a in data.get("pending_actions", [])],
            agent_state=data.get("agent_state", {}),
            conversation_history=data.get("conversation_history", []),
            interrupt_reason=InterruptReason(data.get("interrupt_reason", "tool_approval")),
            metadata=data.get("metadata", {}),
        )


# Type alias for interrupt_on configuration
# Can be:
# - bool: True means always require approval, False means auto-approve
# - Callable[[str, dict], bool]: Function that decides based on tool_name and args
InterruptRule = bool | Callable[[str, dict[str, Any]], bool]


def should_interrupt(
    tool_name: str,
    args: dict[str, Any],
    interrupt_on: dict[str, InterruptRule] | None,
) -> bool:
    """
    Determine if a tool call should be interrupted for human approval.
    
    Args:
        tool_name: Name of the tool being called
        args: Arguments to the tool
        interrupt_on: Configuration mapping tool names to interrupt rules
        
    Returns:
        True if the tool should be interrupted, False if auto-approved
        
    Example:
        ```python
        # Always require approval for delete_file
        interrupt_on = {
            "delete_file": True,
            "read_file": False,  # Never interrupt
            "write_file": lambda name, args: args.get("path", "").startswith("/important/"),
        }
        
        should_interrupt("delete_file", {}, interrupt_on)  # True
        should_interrupt("read_file", {}, interrupt_on)  # False
        should_interrupt("write_file", {"path": "/important/data.txt"}, interrupt_on)  # True
        should_interrupt("write_file", {"path": "/tmp/scratch.txt"}, interrupt_on)  # False
        ```
    """
    if interrupt_on is None:
        return False
    
    # Check for exact tool match
    if tool_name in interrupt_on:
        rule = interrupt_on[tool_name]
        if isinstance(rule, bool):
            return rule
        elif callable(rule):
            return rule(tool_name, args)
    
    # Check for wildcard rule
    if "*" in interrupt_on:
        rule = interrupt_on["*"]
        if isinstance(rule, bool):
            return rule
        elif callable(rule):
            return rule(tool_name, args)
    
    # Default: no interrupt
    return False


class HITLException(Exception):
    """Base exception for HITL operations."""
    pass


class InterruptedException(HITLException):
    """
    Raised when agent execution is interrupted for human input.
    
    Contains the interrupted state needed for resumption.
    """
    
    def __init__(self, state: InterruptedState, message: str = "Execution interrupted for human input"):
        super().__init__(message)
        self.state = state


class DecisionRequiredException(HITLException):
    """Raised when a human decision is required to continue."""
    
    def __init__(self, pending_action: PendingAction):
        super().__init__(f"Human decision required for: {pending_action.describe()}")
        self.pending_action = pending_action


class AbortedException(HITLException):
    """Raised when human aborts the workflow."""
    
    def __init__(self, decision: HumanDecision):
        super().__init__(f"Workflow aborted by human: {decision.feedback or 'No reason provided'}")
        self.decision = decision


@dataclass
class GuidanceResult:
    """
    Result returned when human provides guidance instead of approval.
    
    When a human chooses to guide rather than approve/reject, this result
    contains the guidance that should influence the agent's next action.
    
    Attributes:
        action_id: ID of the original pending action
        guidance: Human's guidance/instructions
        original_action: The action that was pending
        feedback: Optional additional feedback
        should_retry: Whether the agent should retry with the guidance
    """
    
    action_id: str
    guidance: str
    original_action: PendingAction
    feedback: str | None = None
    should_retry: bool = True
    
    def to_message(self) -> str:
        """Convert guidance to a message for the agent."""
        parts = [f"Human guidance for {self.original_action.tool_name}:"]
        parts.append(self.guidance)
        if self.feedback:
            parts.append(f"\nAdditional context: {self.feedback}")
        return "\n".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "action_id": self.action_id,
            "guidance": self.guidance,
            "original_action": self.original_action.to_dict(),
            "feedback": self.feedback,
            "should_retry": self.should_retry,
        }


@dataclass  
class HumanResponse:
    """
    Result returned when human provides a direct response.
    
    When an agent requests human input (not approval), this contains
    the human's direct answer.
    
    Attributes:
        action_id: ID of the original pending action
        response: The human's response value
        original_action: The action that was pending
        feedback: Optional additional context
    """
    
    action_id: str
    response: Any
    original_action: PendingAction
    feedback: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "action_id": self.action_id,
            "response": self.response,
            "original_action": self.original_action.to_dict(),
            "feedback": self.feedback,
        }
