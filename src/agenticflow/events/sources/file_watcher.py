"""File watcher event source.

Watches directories for file changes and emits events into EventFlow.
Uses watchfiles for efficient cross-platform file system watching.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from agenticflow.events.event import Event
from agenticflow.events.sources.base import EmitCallback, EventSource


@dataclass
class FileWatcherSource(EventSource):
    """Watch directories for file changes.

    Monitors specified directories and emits events when files are
    created, modified, or deleted.

    Attributes:
        paths: List of directories to watch
        patterns: Glob patterns to filter files (e.g., ["*.json", "*.csv"])
        recursive: Watch subdirectories recursively (default: True)
        event_prefix: Prefix for event names (default: "file")

    Event Types Emitted:
        - file.created: New file created
        - file.modified: Existing file modified
        - file.deleted: File deleted

    Example:
        ```python
        source = FileWatcherSource(
            paths=["./incoming", "./uploads"],
            patterns=["*.json", "*.csv"],
        )
        flow.source(source)

        # When ./incoming/data.json is created:
        # Event(name="file.created", data={"path": "./incoming/data.json", ...})
        ```
    """

    paths: list[str | Path] = field(default_factory=list)
    patterns: list[str] | None = None
    recursive: bool = True
    event_prefix: str = "file"

    _task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)

    async def start(self, emit: EmitCallback) -> None:
        """Start watching for file changes.

        Args:
            emit: Callback to emit events into the flow.
        """
        try:
            from watchfiles import Change, awatch
        except ImportError as e:
            raise ImportError(
                "FileWatcherSource requires 'watchfiles'. "
                "Install with: uv add watchfiles"
            ) from e

        self._running = True
        watch_paths = [Path(p) for p in self.paths]

        # Map watchfiles Change enum to event names
        change_map = {
            Change.added: "created",
            Change.modified: "modified",
            Change.deleted: "deleted",
        }

        async for changes in awatch(*watch_paths, recursive=self.recursive):
            if not self._running:
                break

            for change_type, path_str in changes:
                path = Path(path_str)

                # Apply pattern filtering
                if self.patterns:
                    if not any(fnmatch(path.name, p) for p in self.patterns):
                        continue

                # Determine event name
                change_name = change_map.get(change_type, "changed")
                event_name = f"{self.event_prefix}.{change_name}"

                # Create event with file metadata
                event = Event(
                    name=event_name,
                    data={
                        "path": str(path),
                        "filename": path.name,
                        "extension": path.suffix,
                        "change_type": change_name,
                    },
                    source=f"file_watcher:{','.join(str(p) for p in self.paths)}",
                )

                await emit(event)

    async def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    @property
    def name(self) -> str:
        """Human-readable name for this source."""
        paths_str = ", ".join(str(p) for p in self.paths[:2])
        if len(self.paths) > 2:
            paths_str += f", +{len(self.paths) - 2} more"
        return f"FileWatcherSource({paths_str})"
