"""Checkpointing for ReactiveFlow.

Provides persistent state management for long-running flows, enabling:
- Resume after crashes/restarts
- Distributed flow execution
- Audit and debugging of flow history
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agenticflow.events.event import Event
    from agenticflow.reactive.core import Reaction


@dataclass(slots=True, kw_only=True)
class FlowState:
    """Serializable snapshot of a ReactiveFlow's execution state.
    
    Captures everything needed to resume a flow from a checkpoint.
    
    Attributes:
        flow_id: Unique identifier for the flow instance
        checkpoint_id: Unique identifier for this checkpoint
        task: The original task string
        events_processed: Number of events processed so far
        pending_events: Events waiting to be processed
        context: Shared context dict
        reactions: List of reactions (agent responses) so far
        last_output: Most recent successful agent output
        round: Current processing round
        timestamp: When this checkpoint was created
        metadata: Optional user-defined metadata
    """
    
    flow_id: str
    checkpoint_id: str
    task: str
    events_processed: int = 0
    pending_events: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    reactions: list[dict[str, Any]] = field(default_factory=list)
    last_output: str = ""
    round: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "flow_id": self.flow_id,
            "checkpoint_id": self.checkpoint_id,
            "task": self.task,
            "events_processed": self.events_processed,
            "pending_events": self.pending_events,
            "context": self.context,
            "reactions": self.reactions,
            "last_output": self.last_output,
            "round": self.round,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowState:
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        return cls(
            flow_id=data["flow_id"],
            checkpoint_id=data["checkpoint_id"],
            task=data.get("task", ""),
            events_processed=data.get("events_processed", 0),
            pending_events=data.get("pending_events", []),
            context=data.get("context", {}),
            reactions=data.get("reactions", []),
            last_output=data.get("last_output", ""),
            round=data.get("round", 0),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@runtime_checkable
class Checkpointer(Protocol):
    """Protocol for checkpoint storage backends.
    
    Implementations must be async-compatible for use in ReactiveFlow.
    
    Example:
        ```python
        checkpointer = MemoryCheckpointer()
        flow = ReactiveFlow(checkpointer=checkpointer)
        
        # Resume from checkpoint after crash
        state = await checkpointer.load("my-checkpoint-id")
        result = await flow.resume(state)
        ```
    """
    
    async def save(self, state: FlowState) -> None:
        """Save a checkpoint.
        
        Args:
            state: The flow state to checkpoint
        """
        ...
    
    async def load(self, checkpoint_id: str) -> FlowState | None:
        """Load a checkpoint by ID.
        
        Args:
            checkpoint_id: The checkpoint ID to load
            
        Returns:
            The flow state if found, None otherwise
        """
        ...
    
    async def load_latest(self, flow_id: str) -> FlowState | None:
        """Load the most recent checkpoint for a flow.
        
        Args:
            flow_id: The flow ID to find checkpoints for
            
        Returns:
            The most recent flow state if any exist, None otherwise
        """
        ...
    
    async def list_checkpoints(self, flow_id: str) -> list[str]:
        """List all checkpoint IDs for a flow.
        
        Args:
            flow_id: The flow ID to list checkpoints for
            
        Returns:
            List of checkpoint IDs, newest first
        """
        ...
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.
        
        Args:
            checkpoint_id: The checkpoint ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        ...


class MemoryCheckpointer:
    """In-memory checkpointer for development and testing.
    
    Checkpoints are lost when the process exits. Use FileCheckpointer
    or a database-backed implementation for production.
    
    Example:
        ```python
        checkpointer = MemoryCheckpointer()
        flow = ReactiveFlow(checkpointer=checkpointer)
        ```
    """
    
    def __init__(self, max_checkpoints_per_flow: int = 100) -> None:
        self._checkpoints: dict[str, FlowState] = {}
        self._flow_index: dict[str, list[str]] = {}  # flow_id -> [checkpoint_ids]
        self._max_per_flow = max_checkpoints_per_flow
    
    async def save(self, state: FlowState) -> None:
        """Save checkpoint to memory."""
        self._checkpoints[state.checkpoint_id] = state
        
        # Update flow index
        if state.flow_id not in self._flow_index:
            self._flow_index[state.flow_id] = []
        self._flow_index[state.flow_id].append(state.checkpoint_id)
        
        # Prune old checkpoints if needed
        if len(self._flow_index[state.flow_id]) > self._max_per_flow:
            old_id = self._flow_index[state.flow_id].pop(0)
            self._checkpoints.pop(old_id, None)
    
    async def load(self, checkpoint_id: str) -> FlowState | None:
        """Load checkpoint from memory."""
        return self._checkpoints.get(checkpoint_id)
    
    async def load_latest(self, flow_id: str) -> FlowState | None:
        """Load most recent checkpoint for a flow."""
        checkpoint_ids = self._flow_index.get(flow_id, [])
        if not checkpoint_ids:
            return None
        return self._checkpoints.get(checkpoint_ids[-1])
    
    async def list_checkpoints(self, flow_id: str) -> list[str]:
        """List checkpoints for a flow (newest first)."""
        return list(reversed(self._flow_index.get(flow_id, [])))
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        state = self._checkpoints.pop(checkpoint_id, None)
        if state is None:
            return False
        
        flow_checkpoints = self._flow_index.get(state.flow_id, [])
        if checkpoint_id in flow_checkpoints:
            flow_checkpoints.remove(checkpoint_id)
        return True
    
    def clear(self) -> None:
        """Clear all checkpoints (for testing)."""
        self._checkpoints.clear()
        self._flow_index.clear()


class FileCheckpointer:
    """File-based checkpointer for simple persistence.
    
    Stores each checkpoint as a JSON file. Suitable for single-process
    deployments. For distributed systems, use a database-backed implementation.
    
    Example:
        ```python
        checkpointer = FileCheckpointer(Path("./checkpoints"))
        flow = ReactiveFlow(checkpointer=checkpointer)
        ```
    """
    
    def __init__(
        self,
        directory: Path | str,
        *,
        max_checkpoints_per_flow: int = 100,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._max_per_flow = max_checkpoints_per_flow
    
    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for a checkpoint file."""
        return self._directory / f"{checkpoint_id}.json"
    
    async def save(self, state: FlowState) -> None:
        """Save checkpoint to file."""
        path = self._checkpoint_path(state.checkpoint_id)
        data = state.to_dict()
        path.write_text(json.dumps(data, indent=2, default=str))
        
        # Prune old checkpoints
        await self._prune_flow(state.flow_id)
    
    async def load(self, checkpoint_id: str) -> FlowState | None:
        """Load checkpoint from file."""
        path = self._checkpoint_path(checkpoint_id)
        if not path.exists():
            return None
        
        try:
            data = json.loads(path.read_text())
            return FlowState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None
    
    async def load_latest(self, flow_id: str) -> FlowState | None:
        """Load most recent checkpoint for a flow."""
        checkpoints = await self.list_checkpoints(flow_id)
        if not checkpoints:
            return None
        return await self.load(checkpoints[0])
    
    async def list_checkpoints(self, flow_id: str) -> list[str]:
        """List checkpoints for a flow (newest first)."""
        checkpoints: list[tuple[str, datetime]] = []
        
        for path in self._directory.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if data.get("flow_id") == flow_id:
                    ts = datetime.fromisoformat(data.get("timestamp", ""))
                    checkpoints.append((data["checkpoint_id"], ts))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        
        # Sort by timestamp descending
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return [cp_id for cp_id, _ in checkpoints]
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint file."""
        path = self._checkpoint_path(checkpoint_id)
        if path.exists():
            path.unlink()
            return True
        return False
    
    async def _prune_flow(self, flow_id: str) -> None:
        """Remove old checkpoints beyond the limit."""
        checkpoints = await self.list_checkpoints(flow_id)
        for checkpoint_id in checkpoints[self._max_per_flow:]:
            await self.delete(checkpoint_id)


def generate_checkpoint_id() -> str:
    """Generate a unique checkpoint ID."""
    return f"cp_{uuid.uuid4().hex[:12]}"


def generate_flow_id() -> str:
    """Generate a unique flow ID."""
    return f"flow_{uuid.uuid4().hex[:12]}"
