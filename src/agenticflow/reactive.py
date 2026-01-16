"""Compatibility module for reactive imports.

This module provides backward compatibility for code that imports
from agenticflow.reactive. All reactive functionality is now in
agenticflow.flow.reactive.

For new code, prefer importing from agenticflow.flow or the main package:
    from agenticflow import ReactiveFlow, react_to, skill
    # or
    from agenticflow.flow import ReactiveFlow, react_to, skill
"""

from agenticflow.flow.reactive import (
    EventFlow,
    EventFlowConfig,
    EventFlowResult,
    ReactiveFlow,
    ReactiveFlowConfig,
    ReactiveFlowResult,
)
from agenticflow.flow.streaming import ReactiveStreamChunk
from agenticflow.flow.triggers import react_to
from agenticflow.flow.skills import skill

# Legacy imports for backward compatibility
from agenticflow.observability import Observer

__all__ = [
    # Flow classes
    "EventFlow",
    "EventFlowConfig",
    "EventFlowResult",
    "ReactiveFlow",
    "ReactiveFlowConfig",
    "ReactiveFlowResult",
    "ReactiveStreamChunk",
    # Decorators
    "react_to",
    "skill",
    # Events
    "Observer",
]
