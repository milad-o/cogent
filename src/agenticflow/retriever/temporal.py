"""Time-based index for temporal document retrieval.

Prioritizes recent information and supports time-aware queries:
- Time-decay scoring (recent documents score higher)
- Time-range filtering ("docs from last 30 days")
- Point-in-time queries ("what was the policy in 2023?")
- Automatic timestamp extraction from content
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, RetrievalResult
from agenticflow.vectorstore import Document

if TYPE_CHECKING:
    from agenticflow.vectorstore import VectorStore


class DecayFunction(Enum):
    """Time decay functions for scoring."""
    EXPONENTIAL = "exponential"  # score * exp(-decay * days)
    LINEAR = "linear"            # score * max(0, 1 - decay * days)
    STEP = "step"                # Full score within window, zero outside
    LOGARITHMIC = "logarithmic"  # score * 1/(1 + log(1 + days))
    NONE = "none"                # No decay, just filtering


@dataclass
class TimeRange:
    """Time range specification."""
    start: datetime | None = None
    end: datetime | None = None

    @classmethod
    def last_days(cls, days: int) -> TimeRange:
        """Create range for last N days."""
        now = datetime.now(UTC)
        return cls(start=now - timedelta(days=days), end=now)

    @classmethod
    def last_hours(cls, hours: int) -> TimeRange:
        """Create range for last N hours."""
        now = datetime.now(UTC)
        return cls(start=now - timedelta(hours=hours), end=now)

    @classmethod
    def year(cls, year: int) -> TimeRange:
        """Create range for a specific year."""
        return cls(
            start=datetime(year, 1, 1, tzinfo=UTC),
            end=datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC),
        )

    @classmethod
    def between(cls, start: str, end: str) -> TimeRange:
        """Create range from ISO date strings."""
        return cls(
            start=datetime.fromisoformat(start.replace("Z", "+00:00")),
            end=datetime.fromisoformat(end.replace("Z", "+00:00")),
        )

    def contains(self, dt: datetime) -> bool:
        """Check if datetime is within range."""
        # Ensure timezone-aware for comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        if self.start:
            start = self.start
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            if dt < start:
                return False

        if self.end:
            end = self.end
            if end.tzinfo is None:
                end = end.replace(tzinfo=UTC)
            if dt > end:
                return False

        return True


class TimeBasedIndex(BaseRetriever):
    """Index with time-aware retrieval and decay scoring.

    Features:
    - Automatic timestamp extraction from content
    - Time-decay scoring (recent = higher score)
    - Time-range filtering
    - Relative time queries ("last 7 days")
    - Point-in-time queries

    Use cases:
    - News and articles
    - Financial reports
    - Changelogs and release notes
    - Policy documents with versions
    - Any evolving knowledge base

    Example:
        ```python
        from agenticflow.retriever import TimeBasedIndex, TimeRange, DecayFunction

        index = TimeBasedIndex(
            vectorstore=vs,
            decay_function=DecayFunction.EXPONENTIAL,
            decay_rate=0.01,  # Halve score every ~70 days
        )

        await index.add_documents(docs)

        # Recent docs score higher
        results = await index.retrieve("market trends")

        # Filter to last 30 days
        results = await index.retrieve(
            "market trends",
            time_range=TimeRange.last_days(30),
        )

        # Point-in-time query
        results = await index.retrieve(
            "company policy",
            time_range=TimeRange.year(2023),
        )
        ```
    """

    _name: str = "time_based"

    # Common date patterns for extraction
    DATE_PATTERNS = [
        # ISO format: 2024-01-15, 2024-01-15T10:30:00
        (r'\b(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?(?:Z|[+-]\d{2}:\d{2})?)\b', "%Y-%m-%d"),
        # US format: 01/15/2024, 1/15/24
        (r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b', "%m/%d/%Y"),
        # EU format: 15-01-2024, 15.01.2024
        (r'\b(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})\b', "%d-%m-%Y"),
        # Written: January 15, 2024 / Jan 15, 2024
        (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b', "%B %d, %Y"),
        # Written: 15 January 2024
        (r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b', "%d %B %Y"),
    ]

    def __init__(
        self,
        vectorstore: VectorStore,
        *,
        decay_function: DecayFunction | str = DecayFunction.EXPONENTIAL,
        decay_rate: float = 0.01,
        reference_date: datetime | None = None,
        auto_extract_timestamps: bool = True,
        timestamp_field: str = "timestamp",
        name: str | None = None,
    ) -> None:
        """Create a time-based index.

        Args:
            vectorstore: Vector store for embeddings.
            decay_function: How to decay scores over time.
            decay_rate: Rate of decay (meaning depends on function).
            reference_date: Reference date for decay calculation (default: now).
            auto_extract_timestamps: Try to extract timestamps from content.
            timestamp_field: Metadata field name for timestamps.
            name: Optional custom name.
        """
        self._vectorstore = vectorstore

        if isinstance(decay_function, str):
            decay_function = DecayFunction(decay_function)
        self._decay_function = decay_function
        self._decay_rate = decay_rate
        self._reference_date = reference_date
        self._auto_extract = auto_extract_timestamps
        self._timestamp_field = timestamp_field

        # Document storage with timestamps
        self._documents: dict[str, Document] = {}
        self._timestamps: dict[str, datetime] = {}

        if name:
            self._name = name

    def _get_reference_date(self) -> datetime:
        """Get reference date for decay calculation."""
        return self._reference_date or datetime.now(UTC)

    def _extract_timestamp(self, text: str, metadata: dict[str, Any]) -> datetime | None:
        """Extract timestamp from text or metadata."""
        # First check metadata
        if self._timestamp_field in metadata:
            ts = metadata[self._timestamp_field]
            if isinstance(ts, datetime):
                return ts
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    pass

        # Check common metadata fields
        for field in ["date", "created_at", "published_at", "updated_at", "created", "published"]:
            if field in metadata:
                val = metadata[field]
                if isinstance(val, datetime):
                    return val
                if isinstance(val, str):
                    try:
                        return datetime.fromisoformat(val.replace("Z", "+00:00"))
                    except ValueError:
                        pass

        # Try to extract from text
        if self._auto_extract:
            for pattern, date_format in self.DATE_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date_str = match.group(1)
                    try:
                        # Handle various formats
                        if "T" in date_str or "Z" in date_str or "+" in date_str:
                            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))

                        # Try the specified format
                        dt = datetime.strptime(date_str, date_format)
                        return dt.replace(tzinfo=UTC)
                    except ValueError:
                        continue

        return None

    def _get_age_days(self, timestamp: datetime) -> int:
        """Calculate age in days, handling timezone-naive datetimes."""
        ref = self._get_reference_date()

        # Ensure both are timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=UTC)

        return (ref - timestamp).days

    def _calculate_decay(self, doc_timestamp: datetime) -> float:
        """Calculate decay multiplier based on document age."""
        ref = self._get_reference_date()

        # Ensure both are timezone-aware
        if doc_timestamp.tzinfo is None:
            doc_timestamp = doc_timestamp.replace(tzinfo=UTC)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=UTC)

        age_days = (ref - doc_timestamp).total_seconds() / 86400

        if age_days < 0:
            age_days = 0  # Future dates get no decay

        if self._decay_function == DecayFunction.NONE:
            return 1.0

        elif self._decay_function == DecayFunction.EXPONENTIAL:
            # Half-life approach: decay_rate of 0.01 means ~70 day half-life
            return math.exp(-self._decay_rate * age_days)

        elif self._decay_function == DecayFunction.LINEAR:
            # Linear decay to zero at 1/decay_rate days
            return max(0.0, 1.0 - self._decay_rate * age_days)

        elif self._decay_function == DecayFunction.STEP:
            # Full score within window, zero outside
            window_days = 1.0 / self._decay_rate if self._decay_rate > 0 else 365
            return 1.0 if age_days <= window_days else 0.0

        elif self._decay_function == DecayFunction.LOGARITHMIC:
            # Slow decay: 1 / (1 + log(1 + days * rate))
            return 1.0 / (1.0 + math.log(1.0 + age_days * self._decay_rate))

        return 1.0

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents with timestamp extraction.

        Args:
            documents: Documents to index.

        Returns:
            List of document IDs.
        """
        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            doc_id = doc.id or hashlib.md5(doc.text.encode()).hexdigest()[:12]
            self._documents[doc_id] = doc

            # Extract or use provided timestamp
            timestamp = self._extract_timestamp(doc.text, doc.metadata)
            if timestamp:
                self._timestamps[doc_id] = timestamp
            else:
                # Default to now if no timestamp found
                self._timestamps[doc_id] = datetime.now(UTC)

            # Prepare for vector store
            metadata = {
                **doc.metadata,
                "doc_id": doc_id,
                self._timestamp_field: self._timestamps[doc_id].isoformat(),
            }

            texts.append(doc.text)
            metadatas.append(metadata)
            ids.append(doc_id)

        # Add to vector store
        await self._vectorstore.add_texts(texts, metadatas=metadatas)

        return ids

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        time_range: TimeRange | None = None,
        apply_decay: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve with time-aware scoring.

        Args:
            query: Search query.
            k: Number of documents to retrieve.
            filter: Additional metadata filters.
            time_range: Optional time range filter.
            apply_decay: Whether to apply time decay to scores.

        Returns:
            Results sorted by decay-adjusted scores.
        """
        # Fetch more to account for filtering
        fetch_k = k * 3 if time_range else k

        search_results = await self._vectorstore.search(query, k=fetch_k, filter=filter)

        results: list[RetrievalResult] = []

        for sr in search_results:
            doc_id = sr.document.metadata.get("doc_id")

            # Get timestamp
            timestamp = None
            if doc_id and doc_id in self._timestamps:
                timestamp = self._timestamps[doc_id]
            else:
                # Try to parse from metadata
                ts_str = sr.document.metadata.get(self._timestamp_field)
                if ts_str:
                    with contextlib.suppress(ValueError):
                        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

            # Apply time range filter
            if time_range and timestamp:
                if not time_range.contains(timestamp):
                    continue

            # Calculate decay-adjusted score
            original_score = sr.score
            if apply_decay and timestamp:
                decay = self._calculate_decay(timestamp)
                adjusted_score = original_score * decay
            else:
                decay = 1.0
                adjusted_score = original_score

            results.append(RetrievalResult(
                document=sr.document,
                score=adjusted_score,
                retriever_name=self.name,
                metadata={
                    "original_score": original_score,
                    "decay_factor": decay,
                    "timestamp": timestamp.isoformat() if timestamp else None,
                    "age_days": self._get_age_days(timestamp) if timestamp else None,
                },
            ))

        # Sort by adjusted score
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:k]

    async def retrieve_latest(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve most recent matching documents.

        Combines semantic relevance with strong recency preference.
        """
        return await self.retrieve_with_scores(
            query, k=k, filter=filter, apply_decay=True
        )

    async def retrieve_at_time(
        self,
        query: str,
        point_in_time: datetime | str,
        k: int = 4,
        window_days: int = 30,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents as they existed at a point in time.

        Args:
            query: Search query.
            point_in_time: The reference date.
            k: Number of documents.
            window_days: How far back from point_in_time to search.
            filter: Additional filters.

        Returns:
            Documents that existed at that point in time.
        """
        if isinstance(point_in_time, str):
            point_in_time = datetime.fromisoformat(point_in_time.replace("Z", "+00:00"))

        time_range = TimeRange(
            start=point_in_time - timedelta(days=window_days),
            end=point_in_time,
        )

        return await self.retrieve_with_scores(
            query, k=k, filter=filter, time_range=time_range, apply_decay=False
        )

    def get_document_timeline(self) -> list[tuple[datetime, str, str]]:
        """Get timeline of all indexed documents.

        Returns:
            List of (timestamp, doc_id, title) sorted by time.
        """
        timeline = []
        for doc_id, timestamp in self._timestamps.items():
            doc = self._documents.get(doc_id)
            title = doc.metadata.get("title", doc_id) if doc else doc_id
            timeline.append((timestamp, doc_id, title))

        timeline.sort(key=lambda x: x[0], reverse=True)
        return timeline

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        k_default: int = 4,
        include_scores: bool = False,
        include_metadata: bool = True,
        allow_time_range: bool = True,
        allow_decay_control: bool = True,
    ):
        """Expose this retriever as a tool with time-aware parameters.

        Args:
            name: Optional tool name.
            description: Optional tool description.
            k_default: Default number of results.
            include_scores: Include scores in output.
            include_metadata: Include metadata in output.
            allow_time_range: Allow specifying time_range parameters.
            allow_decay_control: Allow controlling decay application.

        Returns:
            BaseTool configured for this retriever.
        """
        from agenticflow.tools.base import BaseTool

        tool_name = name or f"{self.name}_retrieve"
        tool_description = description or (
            f"Retrieve documents using {self.name} with time-based scoring. "
            "Recent documents are ranked higher based on recency decay."
        )

        args_schema: dict[str, Any] = {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return.",
                "default": k_default,
                "minimum": 1,
            },
            "filter": {
                "type": "object",
                "description": "Optional metadata filter (field -> value).",
                "additionalProperties": True,
            },
        }

        if allow_time_range:
            args_schema["time_start"] = {
                "type": "string",
                "description": "Start of time range (ISO format, e.g., '2024-01-01T00:00:00Z').",
            }
            args_schema["time_end"] = {
                "type": "string",
                "description": "End of time range (ISO format, e.g., '2024-12-31T23:59:59Z').",
            }

        if allow_decay_control:
            args_schema["apply_decay"] = {
                "type": "boolean",
                "description": "Apply time decay to scores (recent documents ranked higher).",
                "default": True,
            }

        async def _tool(
            query: str,
            k: int = k_default,
            filter: dict[str, Any] | None = None,
            time_start: str | None = None,
            time_end: str | None = None,
            apply_decay: bool = True,
        ) -> list[dict[str, Any]]:
            # Parse time range if provided
            time_range = None
            if allow_time_range and (time_start or time_end):
                from datetime import datetime

                start = datetime.fromisoformat(time_start.replace("Z", "+00:00")) if time_start else None
                end = datetime.fromisoformat(time_end.replace("Z", "+00:00")) if time_end else None

                if start or end:
                    time_range = TimeRange(start=start, end=end)

            results = await self.retrieve(
                query,
                k=k,
                filter=filter,
                time_range=time_range,
                apply_decay=apply_decay if allow_decay_control else True,
                include_scores=True,
            )

            payload: list[dict[str, Any]] = []
            for r in results:
                entry: dict[str, Any] = {"text": r.document.text}
                if include_metadata:
                    entry["metadata"] = r.document.metadata
                if include_scores:
                    entry["score"] = r.score
                payload.append(entry)

            return payload

        return BaseTool(
            name=tool_name,
            description=tool_description,
            func=_tool,
            args_schema=args_schema,
        )
