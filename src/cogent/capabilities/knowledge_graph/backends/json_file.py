"""JSON file-backed graph storage with auto-save."""

from __future__ import annotations

from pathlib import Path

from cogent.capabilities.knowledge_graph.backends.base import (
    GraphBackend,
    _SyncGraphBackend,
)
from cogent.graph import Graph
from cogent.graph.storage import FileStorage


class JSONFileGraph(_SyncGraphBackend, GraphBackend):
    """Synchronous facade over Graph+FileStorage."""

    def __init__(self, path: str | Path, auto_save: bool = True) -> None:
        self._path = Path(path)
        self._auto_save = auto_save
        storage = FileStorage(str(self._path), format="json", auto_save=auto_save)
        super().__init__(Graph(storage=storage))

    def _maybe_save(self) -> None:
        if self._auto_save:
            self.save()

    def add_entity(self, *args, **kwargs):  # type: ignore[override]
        entity = super().add_entity(*args, **kwargs)
        self._maybe_save()
        return entity

    def add_relationship(self, *args, **kwargs):  # type: ignore[override]
        rel = super().add_relationship(*args, **kwargs)
        self._maybe_save()
        return rel

    def remove_entity(self, entity_id: str) -> bool:  # type: ignore[override]
        removed = super().remove_entity(entity_id)
        if removed:
            self._maybe_save()
        return removed

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self._maybe_save()

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("Path required for JSONFileGraph.save()")
        self._path = target
        self._graph.storage.path = self._path  # type: ignore[attr-defined]
        # FileStorage exposes async save
        self._runner.run(self._graph.storage.save())  # type: ignore[attr-defined]

    def load(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self._path
        if target is None or not target.exists():
            return
        self._path = target
        self._graph.storage.path = self._path  # type: ignore[attr-defined]
        self._runner.run(self._graph.storage.load())  # type: ignore[attr-defined]
