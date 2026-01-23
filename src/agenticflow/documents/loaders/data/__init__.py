"""Structured data file loaders."""

from agenticflow.documents.loaders.data.csv import CSVLoader
from agenticflow.documents.loaders.data.json import JSONLoader
from agenticflow.documents.loaders.data.xlsx import XLSXLoader

__all__ = ["CSVLoader", "JSONLoader", "XLSXLoader"]
