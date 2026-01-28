"""
Console Sink - Write to stdout/stderr.
"""

from __future__ import annotations

import sys
from typing import TextIO

from cogent.observability.sinks.base import BaseSink


class ConsoleSink(BaseSink):
    """
    Writes output to console (stdout/stderr).

    Example:
        ```python
        sink = ConsoleSink()  # Default: stdout
        sink = ConsoleSink(stream=sys.stderr)

        sink.write("Hello, world!")
        ```
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        auto_newline: bool = True,
    ) -> None:
        """
        Initialize console sink.

        Args:
            stream: Output stream (default: sys.stderr)
            auto_newline: Add newline after each write if not present
        """
        self._stream = stream or sys.stderr
        self._auto_newline = auto_newline

    def write(self, output: str) -> None:
        """Write to console stream."""
        if self._auto_newline and not output.endswith("\n"):
            output = output + "\n"
        self._stream.write(output)

    def flush(self) -> None:
        """Flush the console stream."""
        self._stream.flush()

    def write_raw(self, output: str) -> None:
        """Write without auto-newline, for streaming tokens."""
        self._stream.write(output)
        self._stream.flush()
