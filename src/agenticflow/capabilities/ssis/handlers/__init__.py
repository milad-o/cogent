"""SSIS task handlers for extensible task parsing."""

from agenticflow.capabilities.ssis.handlers.base import TaskHandler, TaskHandlerRegistry
from agenticflow.capabilities.ssis.handlers.builtin import (
    DEFAULT_HANDLERS,
    ExecuteProcessTaskHandler,
    ScriptTaskHandler,
    WebServiceTaskHandler,
    XMLTaskHandler,
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
