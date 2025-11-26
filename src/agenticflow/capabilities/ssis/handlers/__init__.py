"""SSIS task handlers for extensible task parsing."""

from agenticflow.capabilities.ssis.handlers.base import TaskHandler, TaskHandlerRegistry
from agenticflow.capabilities.ssis.handlers.builtin import (
    ExecuteProcessTaskHandler,
    ScriptTaskHandler,
    WebServiceTaskHandler,
    XMLTaskHandler,
    DEFAULT_HANDLERS,
)

__all__ = [
    # Base classes
    "TaskHandler",
    "TaskHandlerRegistry",
    # Built-in handlers
    "ExecuteProcessTaskHandler",
    "ScriptTaskHandler",
    "WebServiceTaskHandler",
    "XMLTaskHandler",
    "DEFAULT_HANDLERS",
]
