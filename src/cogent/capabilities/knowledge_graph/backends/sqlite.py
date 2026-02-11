"""SQLite-backed graph storage using the async Graph API."""

from __future__ import annotations

from pathlib import Path

from cogent.capabilities.knowledge_graph.backends.base import (
    GraphBackend,
    _SyncGraphBackend,
)
from cogent.graph import Graph
from cogent.graph.storage import SQLStorage


class SQLiteGraph(_SyncGraphBackend, GraphBackend):
    """Synchronous facade over Graph+SQLStorage (SQLite)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        connection = f"sqlite+aiosqlite:///{self._path}"
        storage = SQLStorage(connection)
        super().__init__(Graph(storage=storage))

    def close(self) -> None:  # type: ignore[override]
        # Close storage then stop runner
        self._runner.run(self._graph.storage.close())  # type: ignore[attr-defined]
        super().close()
