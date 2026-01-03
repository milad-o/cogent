"""CSV file loader."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document


class CSVLoader(BaseLoader):
    """Loader for CSV files.
    
    Converts CSV rows to readable text format.
    Each row becomes "column1: value1, column2: value2, ...".
    
    Example:
        >>> loader = CSVLoader()
        >>> docs = await loader.load(Path("data.csv"))
        >>> print(docs[0].metadata["columns"])
    """
    
    supported_extensions = [".csv", ".tsv"]
    
    def __init__(
        self,
        encoding: str = "utf-8",
        delimiter: str | None = None,
        row_as_document: bool = False,
    ) -> None:
        """Initialize the loader.
        
        Args:
            encoding: Text encoding.
            delimiter: CSV delimiter (auto-detected if None).
            row_as_document: If True, each row becomes a separate Document.
        """
        super().__init__(encoding)
        self.delimiter = delimiter
        self.row_as_document = row_as_document
    
    async def load(self, path: str | Path, **kwargs: Any) -> list[Document]:
        """Load a CSV file.
        
        Args:
            path: Path to the CSV file (str or Path).
            **kwargs: Optional 'encoding', 'delimiter' overrides.
            
        Returns:
            List of Documents (one per file, or one per row if row_as_document).
        """
        path = Path(path)
        encoding = kwargs.get("encoding", self.encoding)
        content = path.read_text(encoding=encoding)
        
        # Determine delimiter
        delimiter = kwargs.get("delimiter", self.delimiter)
        if delimiter is None:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
        
        if self.row_as_document:
            # Each row becomes a document
            documents = []
            for i, row in enumerate(rows):
                row_text = ", ".join(f"{k}: {v}" for k, v in row.items() if v)
                documents.append(
                    self._create_document(
                        row_text,
                        path,
                        row_index=i,
                        columns=fieldnames,
                    )
                )
            return documents
        else:
            # All rows in one document
            text_rows = []
            for row in rows:
                row_text = ", ".join(f"{k}: {v}" for k, v in row.items() if v)
                text_rows.append(row_text)
            
            return [
                self._create_document(
                    "\n".join(text_rows),
                    path,
                    columns=fieldnames,
                    row_count=len(rows),
                )
            ]


__all__ = ["CSVLoader"]
