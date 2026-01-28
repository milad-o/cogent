"""
Callback Sink - Write to a callback function.
"""

from __future__ import annotations

from collections.abc import Callable

from cogent.observability.sinks.base import BaseSink


class CallbackSink(BaseSink):
    """
    Writes output to a callback function.

    Useful for integrating with custom logging systems,
    message queues, or real-time displays.

    Example:
        ```python
        def my_handler(output: str) -> None:
            send_to_slack(output)

        sink = CallbackSink(my_handler)
        sink.write("Alert!")
        ```
    """

    def __init__(
        self,
        callback: Callable[[str], None],
        *,
        on_flush: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """
        Initialize callback sink.

        Args:
            callback: Function called for each write
            on_flush: Optional function called on flush
            on_close: Optional function called on close
        """
        self._callback = callback
        self._on_flush = on_flush
        self._on_close = on_close

    def write(self, output: str) -> None:
        """Call the callback with the output."""
        self._callback(output)

    def flush(self) -> None:
        """Call flush callback if provided."""
        if self._on_flush:
            self._on_flush()

    def close(self) -> None:
        """Call close callback if provided."""
        if self._on_close:
            self._on_close()
