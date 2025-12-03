"""
Enums for AgenticFlow - defines all status types, event types, and roles.
"""

from enum import Enum


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    BLOCKED = "blocked"
    RUNNING = "running"
    SPAWNING = "spawning"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Check if this status is a terminal state."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    def is_active(self) -> bool:
        """Check if this status indicates active work."""
        return self in (TaskStatus.RUNNING, TaskStatus.SPAWNING)


class AgentStatus(Enum):
    """Agent lifecycle states."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"

    def is_available(self) -> bool:
        """Check if agent can accept new work."""
        return self == AgentStatus.IDLE

    def is_working(self) -> bool:
        """Check if agent is currently working."""
        return self in (AgentStatus.THINKING, AgentStatus.ACTING)


class EventType(Enum):
    """All event types in the system."""

    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"

    # Task lifecycle events
    TASK_CREATED = "task.created"
    TASK_SCHEDULED = "task.scheduled"
    TASK_STARTED = "task.started"
    TASK_BLOCKED = "task.blocked"
    TASK_UNBLOCKED = "task.unblocked"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_RETRYING = "task.retrying"

    # Subtask events
    SUBTASK_SPAWNED = "subtask.spawned"
    SUBTASK_COMPLETED = "subtask.completed"
    SUBTASKS_AGGREGATED = "subtasks.aggregated"

    # Agent events
    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_INVOKED = "agent.invoked"
    AGENT_THINKING = "agent.thinking"
    AGENT_REASONING = "agent.reasoning"  # Extended thinking/chain-of-thought
    AGENT_ACTING = "agent.acting"
    AGENT_RESPONDED = "agent.responded"
    AGENT_ERROR = "agent.error"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_INTERRUPTED = "agent.interrupted"  # HITL: agent paused for human input
    AGENT_RESUMED = "agent.resumed"  # HITL: agent resumed after human decision
    
    # Agent spawning events (dynamic agent creation)
    AGENT_SPAWNED = "agent.spawned"  # New agent spawned by parent
    AGENT_SPAWN_COMPLETED = "agent.spawn_completed"  # Spawned agent finished task
    AGENT_SPAWN_FAILED = "agent.spawn_failed"  # Spawned agent failed
    AGENT_DESPAWNED = "agent.despawned"  # Ephemeral agent cleaned up
    
    # LLM Request/Response events (deep observability)
    LLM_REQUEST = "llm.request"  # Full request being sent to LLM
    LLM_RESPONSE = "llm.response"  # Full response from LLM (before parsing)
    LLM_TOOL_DECISION = "llm.tool_decision"  # LLM decided to call tool(s)

    # Streaming events (token-by-token LLM output)
    STREAM_START = "stream.start"  # Streaming has started
    TOKEN_STREAMED = "stream.token"  # A token was streamed from the LLM
    STREAM_TOOL_CALL = "stream.tool_call"  # Tool call detected during streaming
    STREAM_END = "stream.end"  # Streaming has completed
    STREAM_ERROR = "stream.error"  # Error during streaming

    # Tool events
    TOOL_REGISTERED = "tool.registered"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    
    # Deferred tool events (async/event-driven completion)
    TOOL_DEFERRED = "tool.deferred"  # Tool returned DeferredResult
    TOOL_DEFERRED_WAITING = "tool.deferred.waiting"  # Waiting for completion
    TOOL_DEFERRED_COMPLETED = "tool.deferred.completed"  # Deferred completed
    TOOL_DEFERRED_TIMEOUT = "tool.deferred.timeout"  # Deferred timed out
    TOOL_DEFERRED_CANCELLED = "tool.deferred.cancelled"  # Deferred cancelled
    
    # Webhook/external events
    WEBHOOK_RECEIVED = "webhook.received"  # External webhook callback

    # Planning events
    PLAN_CREATED = "plan.created"
    PLAN_STEP_STARTED = "plan.step.started"
    PLAN_STEP_COMPLETED = "plan.step.completed"
    PLAN_FAILED = "plan.failed"

    # Message events
    MESSAGE_SENT = "message.sent"
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_BROADCAST = "message.broadcast"

    # WebSocket/Client events
    CLIENT_CONNECTED = "client.connected"
    CLIENT_DISCONNECTED = "client.disconnected"
    CLIENT_MESSAGE = "client.message"

    CUSTOM = "custom"

    @property
    def category(self) -> str:
        """Get the category of this event type."""
        return self.value.split(".")[0]


class Priority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: "Priority") -> bool:
        if isinstance(other, Priority):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other: "Priority") -> bool:
        if isinstance(other, Priority):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other: "Priority") -> bool:
        if isinstance(other, Priority):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other: "Priority") -> bool:
        if isinstance(other, Priority):
            return self.value >= other.value
        return NotImplemented


class AgentRole(Enum):
    """
    Agent roles define CAPABILITIES, not job titles.
    
    Each role is a combination of three core capabilities:
    - can_finish: Can end the flow with FINAL ANSWER
    - can_delegate: Can assign/route work to other agents
    - can_use_tools: Can call external tools
    
    ┌─────────────┬────────────┬──────────────┬───────────────┐
    │ Role        │ can_finish │ can_delegate │ can_use_tools │
    ├─────────────┼────────────┼──────────────┼───────────────┤
    │ WORKER      │     ❌     │      ❌      │      ✅       │
    │ SUPERVISOR  │     ✅     │      ✅      │      ❌       │
    │ AUTONOMOUS  │     ✅     │      ❌      │      ✅       │
    │ REVIEWER    │     ✅     │      ❌      │      ❌       │
    └─────────────┴────────────┴──────────────┴───────────────┘
    
    Usage Guide:
    ─────────────
    WORKER: The doer. Executes tasks, uses tools, reports results.
            Cannot finish the flow - must pass work to someone who can.
            Example: Researcher gathering data, Analyst running queries
    
    SUPERVISOR: The manager. Delegates work, reviews results, decides.
                Can finish the flow. Does NOT use tools directly.
                Example: Team lead, Project manager, Director
    
    AUTONOMOUS: The solo expert. Works independently, uses tools, decides.
                Can finish the flow. Perfect for single-agent flows.
                Example: Assistant, Solo analyst, Independent researcher
    
    REVIEWER: The gatekeeper. Reviews work, approves or rejects.
              Can finish (approve) but doesn't do work or delegate.
              Example: QA reviewer, Editor, Validator
    
    Topology Patterns:
    ──────────────────
    Pipeline:     WORKER → WORKER → REVIEWER
    Supervisor:   SUPERVISOR ↔ [WORKER, WORKER, WORKER]
    Mesh:         [WORKER, WORKER] ↔ SUPERVISOR (as lead)
    Single:       AUTONOMOUS
    Review:       WORKER → REVIEWER
    """

    # Core roles - just 4, each with distinct capabilities
    WORKER = "worker"
    SUPERVISOR = "supervisor"
    AUTONOMOUS = "autonomous"
    REVIEWER = "reviewer"

    def can_finish(self) -> bool:
        """Check if this role can end the flow with FINAL ANSWER."""
        return _ROLE_CAPABILITIES[self]["can_finish"]

    def can_delegate(self) -> bool:
        """Check if this role can assign/route work to other agents."""
        return _ROLE_CAPABILITIES[self]["can_delegate"]

    def can_use_tools(self) -> bool:
        """Check if this role can call external tools."""
        return _ROLE_CAPABILITIES[self]["can_use_tools"]


# =============================================================================
# Role Capabilities - THE source of truth for what each role can do
# =============================================================================

_ROLE_CAPABILITIES: dict[AgentRole, dict[str, bool]] = {
    AgentRole.WORKER: {
        "can_finish": False,
        "can_delegate": False,
        "can_use_tools": True,
    },
    AgentRole.SUPERVISOR: {
        "can_finish": True,
        "can_delegate": True,
        "can_use_tools": False,
    },
    AgentRole.AUTONOMOUS: {
        "can_finish": True,
        "can_delegate": False,
        "can_use_tools": True,
    },
    AgentRole.REVIEWER: {
        "can_finish": True,
        "can_delegate": False,
        "can_use_tools": False,
    },
}


def get_role_capabilities(role: AgentRole) -> dict[str, bool]:
    """Get capabilities for a role."""
    return _ROLE_CAPABILITIES.get(role, _ROLE_CAPABILITIES[AgentRole.AUTONOMOUS])
