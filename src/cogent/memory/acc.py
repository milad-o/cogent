"""Agent Cognitive Compressor (ACC) - Bounded memory control.

Based on arXiv:2601.11653: "AI Agents Need Memory Control Over More Context"

Core principle: Replace unbounded transcript replay with bounded internal state
to prevent memory poisoning, drift, and hallucinations.

Architecture:
    1. Bounded State - Fixed-size memory containers
    2. Semantic Forget Gate - Relevance-based retention
    3. Memory Update Rules - Bio-inspired selective updates
    4. Context Injection - Format memory for LLM consumption

Example:
    ```python
    from cogent.memory.acc import AgentCognitiveCompressor, BoundedMemoryState

    # Create bounded state
    state = BoundedMemoryState()
    
    # Create compressor
    acc = AgentCognitiveCompressor(state=state)
    
    # Update from conversation turn
    await acc.update_from_turn(
        user_message="What's the weather?",
        assistant_message="It's sunny today.",
        tool_calls=[],
        current_task="Help with weather queries",
    )
    
    # Format for LLM context
    context = acc.format_for_prompt("Check weather tomorrow")
    ```
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from cogent.core.utils import now_utc

if TYPE_CHECKING:
    from cogent.models.base import BaseChatModel
    from cogent.vectorstore import VectorStore


# =============================================================================
# BOUNDED STATE - Fixed-size memory containers
# =============================================================================


@dataclass
class MemoryItem:
    """Single item in bounded memory state.
    
    Each item has:
    - content: The actual memory content
    - type: Category (constraint, entity, action, context)
    - relevance: Score 0.0-1.0 (updated by forget gate)
    - created_at: When this memory was created
    - last_accessed: Last time this memory was used
    - decay_rate: How fast relevance decays over time
    - verified: Whether this passed verification (prevents poisoning)
    """

    content: str
    type: Literal["constraint", "entity", "action", "context"]
    relevance: float = 1.0  # 0.0-1.0
    created_at: datetime = field(default_factory=now_utc)
    last_accessed: datetime = field(default_factory=now_utc)
    decay_rate: float = 0.1  # How fast relevance decays
    verified: bool = False  # Prevent memory poisoning
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate memory item."""
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError(f"Relevance must be 0.0-1.0, got {self.relevance}")
        if self.decay_rate < 0:
            raise ValueError(f"Decay rate must be >= 0, got {self.decay_rate}")

    def access(self) -> None:
        """Mark this memory as accessed (updates timestamp)."""
        self.last_accessed = now_utc()

    def age_seconds(self) -> float:
        """How old is this memory in seconds?"""
        return (now_utc() - self.created_at).total_seconds()

    def time_since_access(self) -> float:
        """Seconds since last access."""
        return (now_utc() - self.last_accessed).total_seconds()

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        return {
            "content": self.content,
            "type": self.type,
            "relevance": self.relevance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "decay_rate": self.decay_rate,
            "verified": self.verified,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryItem:
        """Deserialize from dict."""
        return cls(
            content=data["content"],
            type=data["type"],
            relevance=data.get("relevance", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            decay_rate=data.get("decay_rate", 0.1),
            verified=data.get("verified", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BoundedMemoryState:
    """Fixed-size internal state for agent memory.
    
    Instead of unbounded transcript replay, we maintain bounded memory
    across 4 categories:
    
    1. Constraints (max 10): Task requirements, goals, rules
    2. Entities (max 50): Facts, names, data, knowledge
    3. Actions (max 30): What worked/failed, execution history
    4. Context (max 20): Relevant conversation snippets
    
    Total bounded size: ~110 items regardless of conversation length.
    """

    # Core memories (task requirements, goals, rules)
    constraints: list[MemoryItem] = field(default_factory=list)
    max_constraints: int = 10

    # Extracted facts, entities, knowledge
    entities: list[MemoryItem] = field(default_factory=list)
    max_entities: int = 50

    # Action history (what worked/failed)
    actions: list[MemoryItem] = field(default_factory=list)
    max_actions: int = 30

    # Relevant context snippets
    context: list[MemoryItem] = field(default_factory=list)
    max_context: int = 20

    @property
    def total_items(self) -> int:
        """Total number of memory items across all categories."""
        return (
            len(self.constraints)
            + len(self.entities)
            + len(self.actions)
            + len(self.context)
        )

    @property
    def is_full(self) -> bool:
        """Whether memory is at capacity."""
        max_total = (
            self.max_constraints
            + self.max_entities
            + self.max_actions
            + self.max_context
        )
        return self.total_items >= max_total

    @property
    def utilization(self) -> float:
        """Memory utilization percentage (0.0-1.0)."""
        max_total = (
            self.max_constraints
            + self.max_entities
            + self.max_actions
            + self.max_context
        )
        return self.total_items / max_total if max_total > 0 else 0.0

    def get_all_items(self) -> list[MemoryItem]:
        """Get all memory items across all categories."""
        return (
            list(self.constraints)
            + list(self.entities)
            + list(self.actions)
            + list(self.context)
        )

    def clear(self) -> None:
        """Clear all memory (useful for testing)."""
        self.constraints.clear()
        self.entities.clear()
        self.actions.clear()
        self.context.clear()

    def to_dict(self) -> dict:
        """Serialize to dict for persistence."""
        return {
            "constraints": [item.to_dict() for item in self.constraints],
            "entities": [item.to_dict() for item in self.entities],
            "actions": [item.to_dict() for item in self.actions],
            "context": [item.to_dict() for item in self.context],
            "max_constraints": self.max_constraints,
            "max_entities": self.max_entities,
            "max_actions": self.max_actions,
            "max_context": self.max_context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BoundedMemoryState:
        """Deserialize from dict."""
        return cls(
            constraints=[MemoryItem.from_dict(item) for item in data.get("constraints", [])],
            entities=[MemoryItem.from_dict(item) for item in data.get("entities", [])],
            actions=[MemoryItem.from_dict(item) for item in data.get("actions", [])],
            context=[MemoryItem.from_dict(item) for item in data.get("context", [])],
            max_constraints=data.get("max_constraints", 10),
            max_entities=data.get("max_entities", 50),
            max_actions=data.get("max_actions", 30),
            max_context=data.get("max_context", 20),
        )


# =============================================================================
# SEMANTIC FORGET GATE - Relevance-based retention
# =============================================================================


class SemanticForgetGate:
    """Decides what to keep/discard from bounded memory.
    
    Uses relevance scoring based on:
    - Semantic similarity to current task (if embedder available)
    - Time decay (older memories fade)
    - Access frequency (frequently used memories stay)
    - Type priority (constraints > entities > actions > context)
    
    Example:
        ```python
        gate = SemanticForgetGate(decay_rate=0.1)
        
        # Update relevance scores
        for item in memory_items:
            item.relevance = gate.compute_relevance(item, current_task, now())
        
        # Prune to max size
        kept = gate.prune_memory(memory_items, max_size=50, current_task)
        ```
    """

    def __init__(
        self,
        embedder: BaseChatModel | None = None,
        decay_rate: float = 0.1,
        type_weights: dict[str, float] | None = None,
    ):
        """Initialize forget gate.
        
        Args:
            embedder: Optional model for computing semantic similarity
            decay_rate: How fast relevance decays over time (higher = faster)
            type_weights: Priority weights for memory types
        """
        self.embedder = embedder
        self.decay_rate = decay_rate
        self.type_weights = type_weights or {
            "constraint": 1.0,  # Always prioritize constraints
            "entity": 0.8,
            "action": 0.7,
            "context": 0.6,
        }

    def compute_relevance(
        self,
        item: MemoryItem,
        current_task: str,
        current_time: datetime | None = None,
    ) -> float:
        """Compute relevance score for memory item.
        
        Combines multiple factors:
        - Semantic similarity to current task (40%)
        - Time decay (30%)
        - Type priority (30%)
        
        Args:
            item: Memory item to score
            current_task: Current task/query for semantic matching
            current_time: Current time (defaults to now)
        
        Returns:
            Relevance score 0.0-1.0
        """
        if current_time is None:
            current_time = now_utc()

        # 1. Time decay - exponential decay based on time since last access
        age_hours = (current_time - item.last_accessed).total_seconds() / 3600
        time_decay = math.exp(-self.decay_rate * age_hours)

        # 2. Semantic similarity (if embedder available)
        semantic_score = 0.5  # Default neutral score
        if self.embedder and current_task and item.content:
            try:
                # TODO: Implement semantic similarity when embedder is available
                # For now, use simple keyword matching as fallback
                semantic_score = self._simple_similarity(current_task, item.content)
            except Exception:
                # Fall back to neutral if embedding fails
                semantic_score = 0.5

        # 3. Type priority
        type_weight = self.type_weights.get(item.type, 0.5)

        # Combine factors with weights
        relevance = 0.4 * semantic_score + 0.3 * time_decay + 0.3 * type_weight

        # Clamp to [0, 1]
        return max(0.0, min(1.0, relevance))

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple keyword-based similarity (fallback when no embedder).
        
        Computes Jaccard similarity on word sets.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def should_forget(
        self,
        item: MemoryItem,
        threshold: float = 0.3,
    ) -> bool:
        """Whether to discard this memory item.
        
        Forgets items that:
        - Have low relevance (below threshold)
        - Are not verified (prevents keeping unverified content)
        
        Args:
            item: Memory item to check
            threshold: Relevance threshold (0.0-1.0)
        
        Returns:
            True if item should be forgotten
        """
        # Never forget verified high-priority items
        if item.verified and item.type == "constraint":
            return False

        # Forget low-relevance unverified items
        return item.relevance < threshold and not item.verified

    def prune_memory(
        self,
        items: list[MemoryItem],
        max_size: int,
        current_task: str,
    ) -> list[MemoryItem]:
        """Remove least relevant items to stay within max_size.
        
        Process:
        1. Update relevance scores for all items
        2. Sort by relevance (highest first)
        3. Keep top max_size items
        4. Mark accessed for kept items
        
        Args:
            items: List of memory items
            max_size: Maximum items to keep
            current_task: Current task for relevance scoring
        
        Returns:
            Pruned list of items (sorted by relevance)
        """
        if not items:
            return []

        # Update relevance scores
        current_time = now_utc()
        for item in items:
            item.relevance = self.compute_relevance(item, current_task, current_time)

        # Sort by relevance (keep highest)
        sorted_items = sorted(items, key=lambda x: x.relevance, reverse=True)

        # Keep top max_size
        kept = sorted_items[:max_size]

        # Mark as accessed
        for item in kept:
            item.access()

        return kept


# =============================================================================
# AGENT COGNITIVE COMPRESSOR - Bio-inspired memory controller
# =============================================================================


class AgentCognitiveCompressor:
    """Bio-inspired memory controller for agents.
    
    Implements the ACC algorithm from arXiv:2601.11653:
    1. Artifact Recall - Retrieve potential memories
    2. Verification - Check before committing (prevent poisoning)
    3. State Commitment - Selective update to bounded state
    4. Pruning - Maintain bounds via semantic forget gate
    
    Example:
        ```python
        state = BoundedMemoryState()
        gate = SemanticForgetGate()
        acc = AgentCognitiveCompressor(state=state, forget_gate=gate)
        
        # Update from conversation turn
        await acc.update_from_turn(
            user_message="Book a flight to Paris",
            assistant_message="I'll help you book that flight.",
            tool_calls=[{"name": "search_flights", "args": {"destination": "Paris"}}],
            current_task="Help book travel",
        )
        
        # Get formatted context for LLM
        context = acc.format_for_prompt("Check flight prices")
        ```
    """

    def __init__(
        self,
        state: BoundedMemoryState,
        forget_gate: SemanticForgetGate | None = None,
        vectorstore: VectorStore | None = None,
        extractor_model: BaseChatModel | None = None,
    ):
        """Initialize ACC.
        
        Args:
            state: Bounded memory state to manage
            forget_gate: Forget gate for relevance-based pruning
            vectorstore: Optional vectorstore for artifact recall
            extractor_model: Optional model for memory extraction
        """
        self.state = state
        self.forget_gate = forget_gate or SemanticForgetGate()
        self.vectorstore = vectorstore
        self.extractor_model = extractor_model

    async def update_from_turn(
        self,
        user_message: str,
        assistant_message: str,
        tool_calls: list[dict],
        current_task: str,
    ) -> None:
        """Update memory state from a single conversation turn.
        
        This is the core ACC algorithm:
        1. Extract memory artifacts from turn
        2. Verify artifacts (prevent poisoning)
        3. Commit to bounded state
        4. Prune to maintain bounds
        
        Args:
            user_message: User's message this turn
            assistant_message: Assistant's response
            tool_calls: Tools called this turn
            current_task: Current task context
        """
        # 1. Extract memory artifacts from turn
        artifacts = await self._extract_artifacts(
            user_message,
            assistant_message,
            tool_calls,
        )

        # 2. Verify artifacts (prevent poisoning)
        verified_artifacts = await self._verify_artifacts(artifacts)

        # 3. Commit to bounded state
        for artifact in verified_artifacts:
            await self._commit_to_state(artifact, current_task)

        # 4. Prune to maintain bounds
        await self._prune_state(current_task)

    async def _extract_artifacts(
        self,
        user_message: str,
        assistant_message: str,
        tool_calls: list[dict],
    ) -> list[MemoryItem]:
        """Extract memory-worthy content from conversation turn.
        
        For now, uses simple rule-based extraction.
        Future: Use LLM to extract structured memory.
        
        Args:
            user_message: User's message
            assistant_message: Assistant's response
            tool_calls: Tools called
        
        Returns:
            List of memory artifacts
        """
        artifacts: list[MemoryItem] = []

        # Extract constraints from user message (simple heuristic)
        if any(keyword in user_message.lower() for keyword in ["must", "need", "require", "should"]):
            artifacts.append(
                MemoryItem(
                    content=f"User requirement: {user_message}",
                    type="constraint",
                    relevance=1.0,
                    verified=False,
                )
            )

        # Extract entities (simple heuristic - capitalize words)
        # Future: Use NER or LLM extraction
        words = user_message.split() + assistant_message.split()
        capitalized = [w.strip(".,!?") for w in words if w and w[0].isupper()]
        if capitalized:
            entities_text = ", ".join(set(capitalized[:10]))  # Top 10
            artifacts.append(
                MemoryItem(
                    content=f"Entities mentioned: {entities_text}",
                    type="entity",
                    relevance=0.8,
                    verified=False,
                )
            )

        # Extract action history from tool calls
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            artifacts.append(
                MemoryItem(
                    content=f"Called tool: {tool_name}",
                    type="action",
                    relevance=0.7,
                    verified=False,
                )
            )

        # Store recent context snippet
        if user_message or assistant_message:
            context_snippet = f"User: {user_message[:100]}... | Assistant: {assistant_message[:100]}..."
            artifacts.append(
                MemoryItem(
                    content=context_snippet,
                    type="context",
                    relevance=0.6,
                    verified=False,
                )
            )

        return artifacts

    async def _verify_artifacts(
        self,
        artifacts: list[MemoryItem],
    ) -> list[MemoryItem]:
        """Verify artifacts before committing (prevent poisoning).
        
        Simple verification for now:
        - Mark all as verified (basic implementation)
        
        Future improvements:
        - Check consistency with existing verified memories
        - Detect hallucinations via fact-checking
        - Cross-reference with vectorstore
        
        Args:
            artifacts: Unverified memory artifacts
        
        Returns:
            Verified artifacts safe to commit
        """
        verified = []
        for artifact in artifacts:
            # Basic verification: just mark as verified
            # Future: Add consistency checks, hallucination detection
            artifact.verified = True
            verified.append(artifact)

        return verified

    async def _commit_to_state(
        self,
        artifact: MemoryItem,
        current_task: str,
    ) -> None:
        """Commit verified artifact to bounded state.
        
        Args:
            artifact: Verified memory artifact
            current_task: Current task (unused for now)
        """
        # Add to appropriate list based on type
        if artifact.type == "constraint":
            self.state.constraints.append(artifact)
        elif artifact.type == "entity":
            self.state.entities.append(artifact)
        elif artifact.type == "action":
            self.state.actions.append(artifact)
        elif artifact.type == "context":
            self.state.context.append(artifact)

    async def _prune_state(self, current_task: str) -> None:
        """Prune each memory category to stay within bounds.
        
        Args:
            current_task: Current task for relevance scoring
        """
        self.state.constraints = self.forget_gate.prune_memory(
            self.state.constraints,
            self.state.max_constraints,
            current_task,
        )
        self.state.entities = self.forget_gate.prune_memory(
            self.state.entities,
            self.state.max_entities,
            current_task,
        )
        self.state.actions = self.forget_gate.prune_memory(
            self.state.actions,
            self.state.max_actions,
            current_task,
        )
        self.state.context = self.forget_gate.prune_memory(
            self.state.context,
            self.state.max_context,
            current_task,
        )

    def format_for_prompt(self, current_task: str) -> str:
        """Format bounded memory for LLM consumption.
        
        Returns a structured prompt section with relevant memory organized by type.
        
        Args:
            current_task: Current task for relevance filtering
        
        Returns:
            Formatted memory context string
        """
        sections = []

        # 1. Task constraints (always include all)
        if self.state.constraints:
            constraints_text = "\n".join(
                f"- {item.content}" for item in self.state.constraints
            )
            sections.append(f"## Task Requirements\n{constraints_text}")

        # 2. Relevant entities (top 10 by relevance)
        if self.state.entities:
            entities = sorted(
                self.state.entities,
                key=lambda x: x.relevance,
                reverse=True,
            )[:10]
            entities_text = "\n".join(f"- {item.content}" for item in entities)
            sections.append(f"## Known Facts\n{entities_text}")

        # 3. Action history (5 most recent)
        if self.state.actions:
            actions = sorted(
                self.state.actions,
                key=lambda x: x.created_at,
                reverse=True,
            )[:5]
            actions_text = "\n".join(f"- {item.content}" for item in actions)
            sections.append(f"## Previous Actions\n{actions_text}")

        # 4. Relevant context (top 5 by relevance)
        if self.state.context:
            context_items = sorted(
                self.state.context,
                key=lambda x: x.relevance,
                reverse=True,
            )[:5]
            context_text = "\n".join(f"- {item.content}" for item in context_items)
            sections.append(f"## Relevant Context\n{context_text}")

        if not sections:
            return ""

        return "\n\n".join(sections)

    def get_stats(self) -> dict[str, object]:
        """Get memory statistics for monitoring.
        
        Returns:
            Dict with memory stats (counts, utilization, etc.)
        """
        return {
            "total_items": self.state.total_items,
            "utilization": self.state.utilization,
            "constraints": len(self.state.constraints),
            "entities": len(self.state.entities),
            "actions": len(self.state.actions),
            "context": len(self.state.context),
            "max_constraints": self.state.max_constraints,
            "max_entities": self.state.max_entities,
            "max_actions": self.state.max_actions,
            "max_context": self.state.max_context,
        }
