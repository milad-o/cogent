"""
Sink Protocol - Interface for event output destinations.

Sinks receive formatted output and write it to their destination
(console, file, network, etc.).
"""

from __future__ import annotations

from typing import Protocol


class Sink(Protocol):
    """
    Protocol for output sinks.

    Implement this protocol to create custom output destinations.

    Example:
        ```python
        class MySink:
            def write(self, output: str) -> None:
                send_to_service(output)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass
        ```
    """

    def write(self, output: str) -> None:
        """
        Write formatted output to the sink.

        Args:
            output: Pre-formatted string to write
        """
        ...

    def flush(self) -> None:
        """
        Flush any buffered output.

        Called periodically and when the observer is closed.
        """
        ...

    def close(self) -> None:
        """
        Close the sink and release resources.

        Called when the observer is shutdown.
        """
        ...


class BaseSink:
    """
    Base class for sinks with default implementations.

    Provides no-op implementations for flush and close.
    Subclasses only need to implement write.
    """

    def write(self, output: str) -> None:
        """Write output. Subclasses must implement."""
        raise NotImplementedError

    def flush(self) -> None:
        """Flush buffered output. Default: no-op."""
        pass

    def close(self) -> None:
        """Close and release resources. Default: no-op."""
        pass

    def __enter__(self) -> BaseSink:
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Context manager exit."""
        self.close()
