"""Structured data file loaders."""

from agenticflow.document.loaders.data.csv import CSVLoader
from agenticflow.document.loaders.data.json import JSONLoader
from agenticflow.document.loaders.data.xlsx import XLSXLoader

__all__ = ["CSVLoader", "JSONLoader", "XLSXLoader"]
