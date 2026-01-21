"""Standard event names and conventions for AgenticFlow.

This module defines semantic event taxonomies to encourage consistent
event naming across applications.

Users can use these constants or define their own domain-specific events.
"""

from __future__ import annotations
from enum import StrEnum


class TaskEvents(StrEnum):
    """Standard task lifecycle events.
    
    Use these for work item tracking and coordination.
    """
    
    CREATED = "task.created"
    """New task defined - default initial event."""
    
    ASSIGNED = "task.assigned"
    """Task assigned to specific agent(s)."""
    
    STARTED = "task.started"
    """Work began on task."""
    
    COMPLETED = "task.completed"
    """Task finished successfully."""
    
    FAILED = "task.failed"
    """Task failed with error."""
    
    CANCELLED = "task.cancelled"
    """Task was cancelled."""


class AgentEvents(StrEnum):
    """Standard agent lifecycle events.
    
    Emitted automatically by AgentReactor.
    """
    
    STARTED = "agent.started"
    """Agent began processing."""
    
    DONE = "agent.done"
    """Agent finished processing - default output event."""
    
    FAILED = "agent.failed"
    """Agent encountered error."""
    
    IDLE = "agent.idle"
    """Agent waiting for work."""


class FlowEvents(StrEnum):
    """Standard flow orchestration events."""
    
    STARTED = "flow.started"
    """Flow execution began."""
    
    DONE = "flow.done"
    """Flow completed successfully - common stop event."""
    
    FAILED = "flow.failed"
    """Flow failed with error."""
    
    CANCELLED = "flow.cancelled"
    """Flow was cancelled."""


class BatchEvents(StrEnum):
    """Standard batch processing events."""
    
    QUEUED = "batch.queued"
    """Batch added to queue."""
    
    READY = "batch.ready"
    """Batch ready for processing."""
    
    ASSIGNED = "batch.assigned"
    """Batch assigned to worker(s)."""
    
    PROCESSING = "batch.processing"
    """Batch currently being processed."""
    
    PROCESSED = "batch.processed"
    """Batch processing completed."""
    
    FAILED = "batch.failed"
    """Batch processing failed."""


class ReviewEvents(StrEnum):
    """Standard review/approval events."""
    
    SUBMITTED = "review.submitted"
    """Document/item submitted for review."""
    
    STARTED = "review.started"
    """Review in progress."""
    
    COMPLETED = "review.completed"
    """Review finished (may be approved or rejected)."""
    
    APPROVED = "review.approved"
    """Review approved."""
    
    REJECTED = "review.rejected"
    """Review rejected."""
    
    CHANGES_REQUESTED = "review.changes_requested"
    """Review requires changes."""


class DeploymentEvents(StrEnum):
    """Standard deployment events."""
    
    REQUESTED = "deployment.requested"
    """Deployment requested."""
    
    VALIDATED = "deployment.validated"
    """Pre-deployment validation passed."""
    
    STARTED = "deployment.started"
    """Deployment in progress."""
    
    COMPLETED = "deployment.completed"
    """Deployment finished successfully."""
    
    FAILED = "deployment.failed"
    """Deployment failed."""
    
    ROLLED_BACK = "deployment.rolled_back"
    """Deployment rolled back."""


class IncidentEvents(StrEnum):
    """Standard incident management events."""
    
    DETECTED = "incident.detected"
    """Incident detected."""
    
    REPORTED = "incident.reported"
    """Incident reported."""
    
    ASSIGNED = "incident.assigned"
    """Incident assigned for investigation."""
    
    INVESTIGATING = "incident.investigating"
    """Incident under investigation."""
    
    ESCALATED = "incident.escalated"
    """Incident escalated."""
    
    RESOLVED = "incident.resolved"
    """Incident resolved."""
    
    CLOSED = "incident.closed"
    """Incident closed."""


# Convenience exports - common patterns
TASK_LIFECYCLE = [
    TaskEvents.CREATED,
    TaskEvents.ASSIGNED,
    TaskEvents.STARTED,
    TaskEvents.COMPLETED,
]

AGENT_LIFECYCLE = [
    AgentEvents.STARTED,
    AgentEvents.DONE,
]

FLOW_LIFECYCLE = [
    FlowEvents.STARTED,
    FlowEvents.DONE,
]
