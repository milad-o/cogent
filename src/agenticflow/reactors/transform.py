"""Transform reactor - transforms event data."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agenticflow.events import Event
from agenticflow.reactors.base import BaseReactor

if TYPE_CHECKING:
    from agenticflow.flow.context import Context


# Transform function type
TransformFn = Callable[[dict[str, Any]], dict[str, Any]]


class Transform(BaseReactor):
    """Transforms event data before forwarding.

    Useful for data normalization, enrichment, or extraction.

    Example:
        ```python
        # Extract specific fields
        flow.register(
            Transform(
                transform=lambda d: {"summary": d.get("output", "")[:100]},
                emit="summary.ready",
            ),
            on="agent.done",
        )

        # Enrich with additional data
        flow.register(
            Transform(
                transform=lambda d: {**d, "timestamp": datetime.now().isoformat()},
                emit="enriched.event",
            ),
            on="raw.event",
        )
        ```
    """

    def __init__(
        self,
        transform: TransformFn,
        emit: str,
        *,
        name: str | None = None,
    ) -> None:
        """Initialize transform reactor.

        Args:
            transform: Function to transform event data
            emit: Event name to emit
            name: Reactor name
        """
        super().__init__(name or "transform")
        self._transform = transform
        self._emit = emit

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event:
        """Transform event data and emit."""
        transformed = self._transform(event.data)

        return Event(
            name=self._emit,
            source=self.name,
            data=transformed,
            correlation_id=event.correlation_id,
        )

    @classmethod
    def extract(
        cls,
        keys: list[str],
        emit: str,
        **kwargs: Any,
    ) -> Transform:
        """Create a transform that extracts specific keys.

        Args:
            keys: Keys to extract from event data
            emit: Event name to emit
        """
        return cls(
            transform=lambda d: {k: d.get(k) for k in keys if k in d},
            emit=emit,
            **kwargs,
        )

    @classmethod
    def rename(
        cls,
        mapping: dict[str, str],
        emit: str,
        **kwargs: Any,
    ) -> Transform:
        """Create a transform that renames keys.

        Args:
            mapping: Old key -> new key mapping
            emit: Event name to emit
        """
        def rename_fn(data: dict[str, Any]) -> dict[str, Any]:
            result = {}
            for k, v in data.items():
                new_key = mapping.get(k, k)
                result[new_key] = v
            return result

        return cls(transform=rename_fn, emit=emit, **kwargs)

    @classmethod
    def flatten(
        cls,
        emit: str,
        prefix: str = "",
        **kwargs: Any,
    ) -> Transform:
        """Create a transform that flattens nested data.

        Args:
            emit: Event name to emit
            prefix: Prefix for flattened keys
        """
        def flatten_fn(data: dict[str, Any], _prefix: str = prefix) -> dict[str, Any]:
            result = {}
            for k, v in data.items():
                key = f"{_prefix}{k}" if _prefix else k
                if isinstance(v, dict):
                    result.update(flatten_fn(v, f"{key}."))
                else:
                    result[key] = v
            return result

        return cls(transform=flatten_fn, emit=emit, **kwargs)


class MapTransform(BaseReactor):
    """Applies multiple transforms and emits multiple events.

    Example:
        ```python
        flow.register(
            MapTransform([
                (lambda d: {"text": d["output"]}, "text.ready"),
                (lambda d: {"length": len(d["output"])}, "metrics.ready"),
            ]),
            on="agent.done",
        )
        ```
    """

    def __init__(
        self,
        transforms: list[tuple[TransformFn, str]],
        *,
        name: str | None = None,
    ) -> None:
        """Initialize map transform.

        Args:
            transforms: List of (transform_fn, emit_name) tuples
            name: Reactor name
        """
        super().__init__(name or "map_transform")
        self._transforms = transforms

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> list[Event]:
        """Apply all transforms and emit multiple events."""
        results = []

        for transform_fn, emit_name in self._transforms:
            try:
                transformed = transform_fn(event.data)
                results.append(Event(
                    name=emit_name,
                    source=self.name,
                    data=transformed,
                    correlation_id=event.correlation_id,
                ))
            except Exception:
                # Skip failed transforms
                continue

        return results
