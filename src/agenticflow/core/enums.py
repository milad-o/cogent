"""
Enums for AgenticFlow - defines all status types, event types, and roles.
"""

from enum import Enum


class TaskStatus(Enum):
    """
    Task lifecycle states.
    
    State Transitions:
        PENDING → SCHEDULED → RUNNING → COMPLETED
                    ↓           ↓          ↓
                 BLOCKED ← SPAWNING → FAILED
                              ↓
                          CANCELLED
    """

    PENDING = "pending"  # Created, waiting to be scheduled
    SCHEDULED = "scheduled"  # Assigned to execution queue
    BLOCKED = "blocked"  # Waiting on dependencies
    RUNNING = "running"  # Currently executing
    SPAWNING = "spawning"  # Creating subtasks
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Error occurred
    CANCELLED = "cancelled"  # Manually cancelled

    def is_terminal(self) -> bool:
        """Check if this status is a terminal state."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    def is_active(self) -> bool:
        """Check if this status indicates active work."""
        return self in (TaskStatus.RUNNING, TaskStatus.SPAWNING)


class AgentStatus(Enum):
    """
    Agent lifecycle states.
    
    State Transitions:
        IDLE ↔ THINKING ↔ ACTING
          ↓        ↓         ↓
       WAITING  ERROR    ERROR
          ↓        ↓         ↓
       OFFLINE ←────────────────
    """

    IDLE = "idle"  # Ready to accept work
    THINKING = "thinking"  # Processing/reasoning with LLM
    ACTING = "acting"  # Executing an action/tool
    WAITING = "waiting"  # Waiting for external input
    ERROR = "error"  # Encountered an error
    OFFLINE = "offline"  # Not available

    def is_available(self) -> bool:
        """Check if agent can accept new work."""
        return self == AgentStatus.IDLE

    def is_working(self) -> bool:
        """Check if agent is currently working."""
        return self in (AgentStatus.THINKING, AgentStatus.ACTING)


class EventType(Enum):
    """
    All event types in the system.
    
    Events are organized by category:
    - system.*: System lifecycle events
    - task.*: Task state changes
    - subtask.*: Subtask-specific events
    - agent.*: Agent activity events
    - tool.*: Tool execution events
    - plan.*: Planning events
    - message.*: Inter-agent communication
    - client.*: External client events
    """

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
    AGENT_ACTING = "agent.acting"
    AGENT_RESPONDED = "agent.responded"
    AGENT_ERROR = "agent.error"
    AGENT_STATUS_CHANGED = "agent.status_changed"

    # Tool events
    TOOL_REGISTERED = "tool.registered"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

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

    @property
    def category(self) -> str:
        """Get the category of this event type."""
        return self.value.split(".")[0]


class Priority(Enum):
    """
    Task priority levels.
    
    Higher values indicate higher priority.
    """

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
    Common agent roles in a multi-agent system.
    
    Roles help the orchestrator understand agent capabilities
    and make appropriate task assignments.
    """

    ORCHESTRATOR = "orchestrator"  # Coordinates other agents
    WORKER = "worker"  # General-purpose task executor
    PLANNER = "planner"  # Creates execution plans
    CRITIC = "critic"  # Reviews and provides feedback
    SPECIALIST = "specialist"  # Domain-specific expertise
    ASSISTANT = "assistant"  # General-purpose helper
    RESEARCHER = "researcher"  # Gathers information
    VALIDATOR = "validator"  # Validates outputs

    def can_delegate(self) -> bool:
        """Check if this role can delegate tasks to others."""
        return self in (AgentRole.ORCHESTRATOR, AgentRole.PLANNER)
