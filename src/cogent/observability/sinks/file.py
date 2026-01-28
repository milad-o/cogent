"""
File Sink - Write to file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from cogent.observability.sinks.base import BaseSink


class FileSink(BaseSink):
    """
    Writes output to a file.

    Example:
        ```python
        sink = FileSink("logs/agent.log")
        sink.write("Event logged")
        sink.close()

        # Or with context manager
        with FileSink("logs/agent.log") as sink:
            sink.write("Event logged")
        ```
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "a",
        encoding: str = "utf-8",
        buffer_size: int = 1,  # Line buffered by default
    ) -> None:
        """
        Initialize file sink.

        Args:
            path: File path to write to
            mode: File mode ('a' for append, 'w' for overwrite)
            encoding: File encoding
            buffer_size: Buffer size (1 = line buffered)
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = open(  # noqa: SIM115 - persistent handle, closed in close()
            self._path,
            mode=mode,
            encoding=encoding,
            buffering=buffer_size,
        )

    def write(self, output: str) -> None:
        """Write to file."""
        if not output.endswith("\n"):
            output = output + "\n"
        self._file.write(output)

    def flush(self) -> None:
        """Flush file buffer."""
        self._file.flush()

    def close(self) -> None:
        """Close the file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def path(self) -> Path:
        """Path to the log file."""
        return self._path
