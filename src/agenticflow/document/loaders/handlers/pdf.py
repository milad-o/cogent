"""PDF file loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticflow.document.loaders.base import BaseLoader
from agenticflow.document.types import Document


class PDFLoader(BaseLoader):
    """Loader for PDF files.
    
    Extracts text from PDF files using pypdf or pdfplumber.
    Returns one Document per page.
    
    Example:
        >>> loader = PDFLoader()
        >>> docs = await loader.load(Path("document.pdf"))
        >>> print(f"Loaded {len(docs)} pages")
    """
    
    supported_extensions = [".pdf"]
    
    def __init__(
        self,
        encoding: str = "utf-8",
        extract_images: bool = False,
    ) -> None:
        """Initialize the loader.
        
        Args:
            encoding: Not used for PDFs (binary format).
            extract_images: Whether to extract text from images (requires OCR).
        """
        super().__init__(encoding)
        self.extract_images = extract_images
    
    async def load(self, path: Path, **kwargs: Any) -> list[Document]:
        """Load a PDF file.
        
        Args:
            path: Path to the PDF file.
            **kwargs: Additional options.
            
        Returns:
            List of Documents, one per page.
            
        Raises:
            ImportError: If pypdf or pdfplumber is not installed.
        """
        # Try pypdf first (lighter weight)
        try:
            return self._load_with_pypdf(path)
        except ImportError:
            # Try pdfplumber (better extraction)
            try:
                return self._load_with_pdfplumber(path)
            except ImportError:
                pass
        
        raise ImportError(
            "PDF loading requires 'pypdf' or 'pdfplumber'. "
            "Install with: uv add pypdf"
        )
    
    def _load_with_pypdf(self, path: Path) -> list[Document]:
        """Load PDF using pypdf library.
        
        Args:
            path: Path to PDF file.
            
        Returns:
            List of Documents.
        """
        from pypdf import PdfReader
        
        documents: list[Document] = []
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    self._create_document(
                        text,
                        path,
                        page=i + 1,
                        total_pages=total_pages,
                    )
                )
        
        return documents
    
    def _load_with_pdfplumber(self, path: Path) -> list[Document]:
        """Load PDF using pdfplumber library.
        
        Args:
            path: Path to PDF file.
            
        Returns:
            List of Documents.
        """
        import pdfplumber
        
        documents: list[Document] = []
        
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        self._create_document(
                            text,
                            path,
                            page=i + 1,
                            total_pages=total_pages,
                        )
                    )
        
        return documents


__all__ = ["PDFLoader"]
