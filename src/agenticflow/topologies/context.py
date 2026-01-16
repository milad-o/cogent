"""
Context management strategies for topologies.

Strategies for managing context between topology rounds to prevent
context explosion in multi-round collaboration patterns.

Example:
    >>> from agenticflow.topologies.context import (
    ...     SlidingWindowStrategy,
    ...     SummarizationStrategy,
    ... )
    >>>
    >>> # Simple: keep only last 3 rounds
    >>> mesh = Mesh(
    ...     agents=[...],
    ...     max_rounds=10,
    ...     context_strategy=SlidingWindowStrategy(max_rounds=3),
    ... )
    >>>
    >>> # Advanced: summarize older rounds
    >>> mesh = Mesh(
    ...     agents=[...],
    ...     context_strategy=SummarizationStrategy(model=model),
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agenticflow.memory import Memory
    from agenticflow.models.base import BaseChatModel
    from agenticflow.observability.bus import TraceBus
    from agenticflow.vectorstore import VectorStore


@runtime_checkable
class ContextStrategy(Protocol):
    """Protocol for context management strategies.

    Implement this to create custom context management for topologies.
    """

    async def build_context(
        self,
        round_history: list[dict[str, str]],
        current_round: int,
        task: str,
    ) -> str:
        """Build context string from round history.

        Args:
            round_history: List of dicts mapping agent names to outputs.
            current_round: The current round number (1-indexed).
            task: The original task being worked on.

        Returns:
            Formatted context string to include in agent prompts.
        """
        ...


def _format_round(round_num: int, round_data: dict[str, str]) -> str:
    """Format a single round of outputs."""
    lines = [f"\n--- Round {round_num} ---"]
    for name, output in round_data.items():
        lines.append(f"\n{name}:\n{output}")
    return "\n".join(lines)


# =============================================================================
# Strategy 1: Sliding Window
# =============================================================================


@dataclass
class SlidingWindowStrategy:
    """Keep only the last N rounds of history.

    The simplest strategy - no LLM calls, just truncation.
    Useful when recent context is most relevant.

    Args:
        max_rounds: Maximum number of rounds to keep (default: 3).
        trace_bus: Optional TraceBus for emitting observability events.

    Example:
        >>> strategy = SlidingWindowStrategy(max_rounds=2)
        >>> mesh = Mesh(agents=[...], context_strategy=strategy)
    """

    max_rounds: int = 3
    trace_bus: TraceBus | None = None

    async def build_context(
        self,
        round_history: list[dict[str, str]],
        current_round: int,
        task: str,
    ) -> str:
        """Build context from last N rounds only."""
        if not round_history:
            return ""

        # Keep only the most recent rounds
        total_rounds = len(round_history)
        dropped_rounds = max(0, total_rounds - self.max_rounds)
        recent = round_history[-self.max_rounds:]

        # Calculate starting round number
        start_round = max(1, total_rounds - self.max_rounds + 1)

        context = "\n\nPREVIOUS ROUNDS:"
        for i, round_data in enumerate(recent):
            context += _format_round(start_round + i, round_data)

        # Emit trace event
        if self.trace_bus and dropped_rounds > 0:
            await self.trace_bus.publish("context.truncated", {
                "strategy": "SlidingWindowStrategy",
                "total_rounds": total_rounds,
                "kept_rounds": len(recent),
                "dropped_rounds": dropped_rounds,
                "current_round": current_round,
            })

        return context


# =============================================================================
# Strategy 2: Summarization
# =============================================================================


@dataclass
class SummarizationStrategy:
    """Summarize older rounds, keep recent ones full.

    Uses an LLM to compress older rounds into a summary while
    keeping the most recent rounds in full detail.

    Args:
        model: Chat model for summarization.
        keep_full_rounds: Number of recent rounds to keep unsummarized.
        max_summary_tokens: Approximate max tokens for summary.
        trace_bus: Optional TraceBus for emitting observability events.

    Example:
        >>> strategy = SummarizationStrategy(model=model, keep_full_rounds=2)
        >>> mesh = Mesh(agents=[...], context_strategy=strategy)
    """

    model: BaseChatModel
    keep_full_rounds: int = 2
    max_summary_tokens: int = 300
    trace_bus: TraceBus | None = None
    _cached_summary: str = field(default="", init=False)
    _summarized_through: int = field(default=0, init=False)

    async def build_context(
        self,
        round_history: list[dict[str, str]],
        current_round: int,
        task: str,
    ) -> str:
        """Build context with summarized older rounds."""
        if not round_history:
            return ""

        total_rounds = len(round_history)

        # If we have few rounds, just show them all
        if total_rounds <= self.keep_full_rounds:
            context = "\n\nPREVIOUS ROUNDS:"
            for i, round_data in enumerate(round_history, 1):
                context += _format_round(i, round_data)
            return context

        # Split into rounds to summarize and rounds to keep full
        rounds_to_summarize = round_history[:-self.keep_full_rounds]
        rounds_to_keep = round_history[-self.keep_full_rounds:]

        # Only re-summarize if we have new rounds to summarize
        if len(rounds_to_summarize) > self._summarized_through:
            self._cached_summary = await self._summarize_rounds(
                rounds_to_summarize, task
            )
            self._summarized_through = len(rounds_to_summarize)

            # Emit trace event
            if self.trace_bus:
                await self.trace_bus.publish("context.summarized", {
                    "strategy": "SummarizationStrategy",
                    "rounds_summarized": len(rounds_to_summarize),
                    "summary_length": len(self._cached_summary),
                    "current_round": current_round,
                })

        # Build final context
        context = "\n\nSUMMARY OF EARLIER ROUNDS:\n" + self._cached_summary
        context += "\n\nRECENT ROUNDS:"
        start_round = total_rounds - self.keep_full_rounds + 1
        for i, round_data in enumerate(rounds_to_keep):
            context += _format_round(start_round + i, round_data)

        return context

    async def _summarize_rounds(
        self,
        rounds: list[dict[str, str]],
        task: str,
    ) -> str:
        """Use LLM to summarize rounds."""
        # Format rounds for summarization
        history_text = ""
        for i, round_data in enumerate(rounds, 1):
            history_text += _format_round(i, round_data)

        prompt = f"""Summarize the key points from this team collaboration history.

TASK: {task}

HISTORY:
{history_text}

Provide a concise summary (~{self.max_summary_tokens // 4} words) capturing:
- Key decisions made
- Important findings
- Areas of agreement/disagreement
- Open questions

SUMMARY:"""

        response = await self.model.ainvoke([{"role": "user", "content": prompt}])
        return response.content


# =============================================================================
# Strategy 3: Retrieval-Augmented
# =============================================================================


@dataclass
class RetrievalStrategy:
    """Store outputs in vectorstore, retrieve only relevant context.

    Uses semantic search to find the most relevant prior contributions
    instead of passing all history.

    Args:
        vectorstore: VectorStore instance for storage and retrieval.
        k: Number of relevant chunks to retrieve.
        include_metadata: Whether to include round/agent info.

    Example:
        >>> from agenticflow.vectorstore import VectorStore
        >>> vs = VectorStore()
        >>> strategy = RetrievalStrategy(vectorstore=vs, k=5)
        >>> mesh = Mesh(agents=[...], context_strategy=strategy)
    """

    vectorstore: VectorStore
    k: int = 5
    include_metadata: bool = True
    _added_rounds: int = field(default=0, init=False)

    async def build_context(
        self,
        round_history: list[dict[str, str]],
        current_round: int,
        task: str,
    ) -> str:
        """Build context by retrieving relevant prior contributions."""
        if not round_history:
            return ""

        # Add new rounds to vectorstore
        if len(round_history) > self._added_rounds:
            new_rounds = round_history[self._added_rounds:]
            for round_num, round_data in enumerate(new_rounds, self._added_rounds + 1):
                texts = []
                metadatas = []
                for agent_name, output in round_data.items():
                    texts.append(output)
                    metadatas.append({
                        "agent": agent_name,
                        "round": round_num,
                    })
                await self.vectorstore.add_texts(texts, metadatas)
            self._added_rounds = len(round_history)

        # Retrieve relevant context for current task
        results = await self.vectorstore.search(task, k=self.k)

        if not results:
            return ""

        context = "\n\nRELEVANT PRIOR CONTRIBUTIONS:"
        for result in results:
            doc = result.document
            if self.include_metadata and doc.metadata:
                agent = doc.metadata.get("agent", "unknown")
                round_num = doc.metadata.get("round", "?")
                context += f"\n\n[{agent} - Round {round_num}]:\n{doc.text}"
            else:
                context += f"\n\n{doc.text}"

        return context


# =============================================================================
# Strategy 4: Structured Handoff
# =============================================================================


@dataclass
class StructuredHandoff:
    """Structured data extracted from agent contributions."""

    decisions: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)

    def to_context(self) -> str:
        """Format as context string."""
        lines = []
        if self.decisions:
            lines.append("DECISIONS MADE:")
            lines.extend(f"  • {d}" for d in self.decisions)
        if self.key_findings:
            lines.append("\nKEY FINDINGS:")
            lines.extend(f"  • {f}" for f in self.key_findings)
        if self.open_questions:
            lines.append("\nOPEN QUESTIONS:")
            lines.extend(f"  • {q}" for q in self.open_questions)
        if self.action_items:
            lines.append("\nACTION ITEMS:")
            lines.extend(f"  • {a}" for a in self.action_items)
        return "\n".join(lines)


@dataclass
class StructuredHandoffStrategy:
    """Extract structured data from contributions instead of raw text.

    Uses an LLM to extract key decisions, findings, and questions
    from each round, passing only the structured data forward.

    Args:
        model: Chat model for extraction.
        max_items_per_category: Max items to keep per category.
        trace_bus: Optional TraceBus for emitting observability events.

    Example:
        >>> strategy = StructuredHandoffStrategy(model=model)
        >>> mesh = Mesh(agents=[...], context_strategy=strategy)
    """

    model: BaseChatModel
    max_items_per_category: int = 5
    trace_bus: TraceBus | None = None
    _accumulated: StructuredHandoff = field(default_factory=StructuredHandoff, init=False)
    _processed_rounds: int = field(default=0, init=False)

    async def build_context(
        self,
        round_history: list[dict[str, str]],
        current_round: int,
        task: str,
    ) -> str:
        """Build context from structured data only."""
        if not round_history:
            return ""

        # Process new rounds
        if len(round_history) > self._processed_rounds:
            new_rounds = round_history[self._processed_rounds:]
            for round_data in new_rounds:
                extracted = await self._extract_structured(round_data, task)
                self._merge_handoff(extracted)
            self._processed_rounds = len(round_history)

            # Emit trace event
            if self.trace_bus:
                await self.trace_bus.publish("context.structured", {
                    "strategy": "StructuredHandoffStrategy",
                    "rounds_processed": len(new_rounds),
                    "decisions_count": len(self._accumulated.decisions),
                    "findings_count": len(self._accumulated.key_findings),
                    "current_round": current_round,
                })

        context = self._accumulated.to_context()
        return f"\n\nCOLLABORATION STATE:\n{context}" if context else ""

    async def _extract_structured(
        self,
        round_data: dict[str, str],
        task: str,
    ) -> StructuredHandoff:
        """Use LLM to extract structured data from round."""
        contributions = "\n\n".join(
            f"[{name}]: {output}" for name, output in round_data.items()
        )

        prompt = f"""Extract structured information from these team contributions.

TASK: {task}

CONTRIBUTIONS:
{contributions}

Respond in this exact JSON format:
{{
  "decisions": ["decision 1", "decision 2"],
  "key_findings": ["finding 1", "finding 2"],
  "open_questions": ["question 1"],
  "action_items": ["action 1"]
}}

Only include items that are clearly stated. Keep each item under 20 words.

JSON:"""

        response = await self.model.ainvoke([{"role": "user", "content": prompt}])

        # Parse JSON response
        import json
        try:
            # Try to extract JSON from response
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return StructuredHandoff(
                    decisions=data.get("decisions", [])[:self.max_items_per_category],
                    key_findings=data.get("key_findings", [])[:self.max_items_per_category],
                    open_questions=data.get("open_questions", [])[:self.max_items_per_category],
                    action_items=data.get("action_items", [])[:self.max_items_per_category],
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return StructuredHandoff()

    def _merge_handoff(self, new: StructuredHandoff) -> None:
        """Merge new handoff into accumulated state."""
        # Add new items, keep most recent up to max
        for attr in ["decisions", "key_findings", "open_questions", "action_items"]:
            existing = getattr(self._accumulated, attr)
            new_items = getattr(new, attr)
            combined = existing + new_items
            setattr(self._accumulated, attr, combined[-self.max_items_per_category:])


# =============================================================================
# Strategy 5: Blackboard (Shared Memory)
# =============================================================================


@dataclass
class BlackboardStrategy:
    """Use shared Memory as a blackboard for context.

    Agents read/write to specific keys in shared memory instead
    of accumulating full text history.

    Args:
        memory: Memory instance for shared state.
        keys: List of keys to include in context.

    Example:
        >>> from agenticflow.memory import Memory
        >>> memory = Memory()
        >>> strategy = BlackboardStrategy(memory=memory)
        >>> mesh = Mesh(agents=[...], context_strategy=strategy)
    """

    memory: Memory
    keys: list[str] = field(default_factory=lambda: [
        "decisions", "findings", "questions", "current_focus"
    ])

    async def build_context(
        self,
        round_history: list[dict[str, str]],
        current_round: int,
        task: str,
    ) -> str:
        """Build context from blackboard keys."""
        context_parts = []

        for key in self.keys:
            value = await self.memory.recall(key)
            if value:
                context_parts.append(f"{key.upper()}:\n{value}")

        if not context_parts:
            return ""

        return "\n\nSHARED KNOWLEDGE:\n" + "\n\n".join(context_parts)

    async def write(self, key: str, value: Any) -> None:
        """Helper to write to the blackboard."""
        await self.memory.remember(key, value)

    async def read(self, key: str) -> Any:
        """Helper to read from the blackboard."""
        return await self.memory.recall(key)


# =============================================================================
# Composite Strategy
# =============================================================================


@dataclass
class CompositeStrategy:
    """Combine multiple strategies.

    Useful for combining approaches, e.g., sliding window + summarization.

    Args:
        strategies: List of strategies to apply in order.
        separator: String to separate outputs from each strategy.
    """

    strategies: list[ContextStrategy]
    separator: str = "\n\n---\n\n"

    async def build_context(
        self,
        round_history: list[dict[str, str]],
        current_round: int,
        task: str,
    ) -> str:
        """Build context by combining all strategies."""
        contexts = []
        for strategy in self.strategies:
            ctx = await strategy.build_context(round_history, current_round, task)
            if ctx:
                contexts.append(ctx)
        return self.separator.join(contexts)


__all__ = [
    # Protocol
    "ContextStrategy",
    # Strategies
    "SlidingWindowStrategy",
    "SummarizationStrategy",
    "RetrievalStrategy",
    "StructuredHandoffStrategy",
    "StructuredHandoff",
    "BlackboardStrategy",
    "CompositeStrategy",
]
