"""Sinks - Output destinations for events."""

from cogent.observability.sinks.base import BaseSink, Sink
from cogent.observability.sinks.callback import CallbackSink
from cogent.observability.sinks.console import ConsoleSink
from cogent.observability.sinks.file import FileSink

__all__ = [
    "Sink",
    "BaseSink",
    "ConsoleSink",
    "FileSink",
    "CallbackSink",
]
