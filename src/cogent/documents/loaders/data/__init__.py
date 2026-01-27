"""Structured data file loaders."""

from cogent.documents.loaders.data.csv import CSVLoader
from cogent.documents.loaders.data.json import JSONLoader
from cogent.documents.loaders.data.xlsx import XLSXLoader

__all__ = ["CSVLoader", "JSONLoader", "XLSXLoader"]
