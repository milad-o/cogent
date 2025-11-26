"""
SSIS Package Analyzer Capability.

Parses Microsoft SQL Server Integration Services (SSIS) packages (.dtsx files)
and stores the structure in a knowledge graph for intelligent querying and
lineage tracing.

SSIS packages are XML-based and contain:
- Control Flow: Tasks, containers, precedence constraints
- Data Flow: Sources, transformations, destinations
- Connection Managers: Database connections, file connections
- Variables and Parameters
- Event Handlers
"""

from agenticflow.capabilities.ssis.capability import SSISAnalyzer
from agenticflow.capabilities.ssis.handlers import (
    TaskHandler,
    TaskHandlerRegistry,
    ExecuteProcessTaskHandler,
    ScriptTaskHandler,
    WebServiceTaskHandler,
    XMLTaskHandler,
    DEFAULT_HANDLERS,
)
from agenticflow.capabilities.ssis.classifiers import (
    classify_executable,
    classify_component,
)

__all__ = [
    # Main capability
    "SSISAnalyzer",
    # Task handler extensibility
    "TaskHandler",
    "TaskHandlerRegistry",
    "DEFAULT_HANDLERS",
    # Built-in handlers
    "ExecuteProcessTaskHandler",
    "ScriptTaskHandler",
    "WebServiceTaskHandler",
    "XMLTaskHandler",
    # Classifiers (for custom handlers)
    "classify_executable",
    "classify_component",
]
