"""Task handler base class and registry for SSIS task parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    from agenticflow.capabilities.ssis.capability import SSISAnalyzer


class TaskHandler(ABC):
    """
    Base class for custom task handlers.

    Implement this class to add parsing logic for specific SSIS task types.
    Register handlers with SSISAnalyzer.register_task_handler().

    Example:
        ```python
        class MyTaskHandler(TaskHandler):
            @property
            def task_patterns(self):
                return ["mycustomtask"]

            def handle(self, exe, analyzer, package_name, task_name):
                # Custom parsing logic
                pass

        analyzer.register_task_handler(MyTaskHandler())
        ```
    """

    @property
    @abstractmethod
    def task_patterns(self) -> list[str]:
        """Patterns to match against CreationName (case-insensitive)."""
        ...

    @abstractmethod
    def handle(
        self,
        exe: ET.Element,
        analyzer: "SSISAnalyzer",
        package_name: str,
        task_name: str,
    ) -> None:
        """Handle the task - extract additional info and add to KG."""
        ...


class TaskHandlerRegistry:
    """Registry for task handlers - enables extensibility."""

    def __init__(self) -> None:
        self._handlers: list[TaskHandler] = []

    def register(self, handler: TaskHandler) -> None:
        """Register a task handler."""
        self._handlers.append(handler)

    def get_handler(self, exe_type: str) -> TaskHandler | None:
        """Get handler for a task type."""
        exe_type_lower = exe_type.lower()
        for handler in self._handlers:
            for pattern in handler.task_patterns:
                if pattern.lower() in exe_type_lower:
                    return handler
        return None
