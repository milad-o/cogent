"""
PDF capability - PDF reading and creation.

Provides tools for extracting text from PDFs and creating new PDF documents,
enabling agents to work with PDF files for document processing.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import PDF
    
    agent = Agent(
        name="DocumentProcessor",
        model=model,
        capabilities=[PDF()],
    )
    
    # Agent can now work with PDFs
    await agent.run("Extract all text from report.pdf and summarize it")
    await agent.run("Create a PDF report with the analysis results")
    ```
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenticflow.capabilities.base import BaseCapability
from agenticflow.tools.base import tool


@dataclass
class PDFInfo:
    """Information about a PDF file."""
    
    path: str
    page_count: int
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    created: datetime | None = None
    modified: datetime | None = None
    encrypted: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "page_count": self.page_count,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "creator": self.creator,
            "created": self.created.isoformat() if self.created else None,
            "modified": self.modified.isoformat() if self.modified else None,
            "encrypted": self.encrypted,
        }


@dataclass
class ExtractResult:
    """Result of text extraction from PDF."""
    
    text: str
    page_count: int
    pages: list[str] | None = None
    char_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:10000] + "..." if len(self.text) > 10000 else self.text,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "truncated": len(self.text) > 10000,
        }


@dataclass
class CreateResult:
    """Result of PDF creation."""
    
    path: str
    page_count: int
    success: bool = True
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "page_count": self.page_count,
            "success": self.success,
            "error": self.error,
        }


class PDF(BaseCapability):
    """
    PDF capability for reading and creating PDF documents.
    
    Provides tools for extracting text from PDFs, getting metadata,
    and creating new PDF documents.
    
    Args:
        allowed_paths: List of paths the agent can access. If empty, allows
            current working directory only.
        max_file_size_mb: Maximum file size to process in MB (default: 100)
        max_pages: Maximum pages to extract at once (default: 500)
        
    Tools provided:
        - read_pdf: Extract text from a PDF file
        - get_pdf_info: Get metadata about a PDF
        - read_pdf_page: Extract text from specific pages
        - create_pdf: Create a new PDF document
        - merge_pdfs: Merge multiple PDFs into one
        - extract_tables: Extract tables from PDF (if available)
    
    Note:
        Requires pypdf for reading: pip install pypdf
        Requires reportlab for creating: pip install reportlab
    """
    
    @property
    def name(self) -> str:
        """Unique name for this capability."""
        return "pdf"
    
    @property
    def tools(self) -> list:
        """Tools this capability provides to the agent."""
        return self.get_tools()
    
    def __init__(
        self,
        allowed_paths: list[str | Path] | None = None,
        max_file_size_mb: int = 100,
        max_pages: int = 500,
    ) -> None:
        self.allowed_paths = [Path(p).resolve() for p in (allowed_paths or ["."])]
        self.max_file_size_mb = max_file_size_mb
        self.max_pages = max_pages
        
        # Check for optional dependencies
        self._has_pypdf = False
        self._has_reportlab = False
        
        try:
            import pypdf
            self._has_pypdf = True
        except ImportError:
            pass
        
        try:
            import reportlab
            self._has_reportlab = True
        except ImportError:
            pass
    
    def _validate_path(self, path: str | Path) -> Path:
        """Validate that path is within allowed directories."""
        path = Path(path).resolve()
        
        for allowed in self.allowed_paths:
            try:
                path.relative_to(allowed)
                return path
            except ValueError:
                continue
        
        msg = f"Path {path} is not within allowed directories"
        raise PermissionError(msg)
    
    def _check_file_size(self, path: Path) -> None:
        """Check file size is within limits."""
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                msg = f"File size ({size_mb:.1f}MB) exceeds limit ({self.max_file_size_mb}MB)"
                raise ValueError(msg)
    
    def _require_pypdf(self) -> None:
        """Ensure pypdf is available."""
        if not self._has_pypdf:
            msg = "pypdf not installed. Install with: pip install pypdf"
            raise ImportError(msg)
    
    def _require_reportlab(self) -> None:
        """Ensure reportlab is available."""
        if not self._has_reportlab:
            msg = "reportlab not installed. Install with: pip install reportlab"
            raise ImportError(msg)
    
    # =========================================================================
    # PDF Reading (requires pypdf)
    # =========================================================================
    
    def _get_pdf_info(self, path: Path) -> PDFInfo:
        """Get PDF metadata."""
        self._require_pypdf()
        from pypdf import PdfReader
        
        reader = PdfReader(path)
        meta = reader.metadata or {}
        
        # Parse dates if available
        created = None
        modified = None
        
        if meta.get("/CreationDate"):
            try:
                date_str = str(meta["/CreationDate"])
                # PDF date format: D:YYYYMMDDHHmmSS
                if date_str.startswith("D:"):
                    date_str = date_str[2:16]
                    created = datetime.strptime(date_str, "%Y%m%d%H%M%S")
            except (ValueError, TypeError):
                pass
        
        return PDFInfo(
            path=str(path),
            page_count=len(reader.pages),
            title=str(meta.get("/Title", "")) or None,
            author=str(meta.get("/Author", "")) or None,
            subject=str(meta.get("/Subject", "")) or None,
            creator=str(meta.get("/Creator", "")) or None,
            created=created,
            modified=modified,
            encrypted=reader.is_encrypted,
        )
    
    def _extract_text(
        self,
        path: Path,
        start_page: int = 0,
        end_page: int | None = None,
        per_page: bool = False,
    ) -> ExtractResult:
        """Extract text from PDF."""
        self._require_pypdf()
        from pypdf import PdfReader
        
        reader = PdfReader(path)
        total_pages = len(reader.pages)
        
        end_page = min(end_page or total_pages, total_pages, start_page + self.max_pages)
        
        pages_text = []
        all_text = []
        
        for i in range(start_page, end_page):
            page = reader.pages[i]
            text = page.extract_text() or ""
            pages_text.append(text)
            all_text.append(text)
        
        full_text = "\n\n".join(all_text)
        
        return ExtractResult(
            text=full_text,
            page_count=end_page - start_page,
            pages=pages_text if per_page else None,
            char_count=len(full_text),
        )
    
    # =========================================================================
    # PDF Creation (requires reportlab)
    # =========================================================================
    
    def _create_pdf(
        self,
        path: Path,
        content: str | list[str],
        title: str | None = None,
        author: str | None = None,
        font_size: int = 12,
        margin: int = 72,  # 1 inch
    ) -> CreateResult:
        """Create a PDF document."""
        self._require_reportlab()
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        
        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
            title=title or "",
            author=author or "",
        )
        
        styles = getSampleStyleSheet()
        normal_style = ParagraphStyle(
            "CustomNormal",
            parent=styles["Normal"],
            fontSize=font_size,
            leading=font_size * 1.2,
        )
        
        # Build content
        story = []
        
        if isinstance(content, str):
            content = [content]
        
        for i, section in enumerate(content):
            # Split by paragraphs
            paragraphs = section.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    # Escape HTML-like characters
                    para = para.replace("&", "&amp;")
                    para = para.replace("<", "&lt;")
                    para = para.replace(">", "&gt;")
                    para = para.replace("\n", "<br/>")
                    
                    story.append(Paragraph(para, normal_style))
                    story.append(Spacer(1, 12))
            
            # Add page break between sections (except last)
            if i < len(content) - 1:
                story.append(PageBreak())
        
        doc.build(story)
        
        # Get page count
        from pypdf import PdfReader
        reader = PdfReader(path)
        page_count = len(reader.pages)
        
        return CreateResult(
            path=str(path),
            page_count=page_count,
        )
    
    def _merge_pdfs(self, paths: list[Path], output_path: Path) -> CreateResult:
        """Merge multiple PDFs into one."""
        self._require_pypdf()
        from pypdf import PdfWriter, PdfReader
        
        writer = PdfWriter()
        total_pages = 0
        
        for path in paths:
            reader = PdfReader(path)
            for page in reader.pages:
                writer.add_page(page)
                total_pages += 1
        
        with open(output_path, "wb") as f:
            writer.write(f)
        
        return CreateResult(
            path=str(output_path),
            page_count=total_pages,
        )
    
    # =========================================================================
    # Tool Methods
    # =========================================================================
    
    def get_tools(self) -> list:
        """Return the tools provided by this capability."""
        
        @tool
        def read_pdf(
            path: str,
            start_page: int = 0,
            end_page: int | None = None,
        ) -> dict[str, Any]:
            """Extract text from a PDF file.
            
            Args:
                path: Path to the PDF file
                start_page: Starting page (0-indexed, default: 0)
                end_page: Ending page (exclusive, default: all pages)
            
            Returns:
                Dictionary with text content, page count, and character count
            """
            file_path = self._validate_path(path)
            self._check_file_size(file_path)
            
            result = self._extract_text(file_path, start_page, end_page)
            return result.to_dict()
        
        @tool
        def get_pdf_info(path: str) -> dict[str, Any]:
            """Get metadata and information about a PDF file.
            
            Args:
                path: Path to the PDF file
            
            Returns:
                Dictionary with page count, title, author, and other metadata
            """
            file_path = self._validate_path(path)
            
            info = self._get_pdf_info(file_path)
            return info.to_dict()
        
        @tool
        def read_pdf_pages(
            path: str,
            pages: list[int],
        ) -> dict[str, Any]:
            """Extract text from specific pages of a PDF.
            
            Args:
                path: Path to the PDF file
                pages: List of page numbers to extract (0-indexed)
            
            Returns:
                Dictionary mapping page numbers to their text content
            """
            self._require_pypdf()
            from pypdf import PdfReader
            
            file_path = self._validate_path(path)
            self._check_file_size(file_path)
            
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            result = {}
            for page_num in pages:
                if 0 <= page_num < total_pages:
                    text = reader.pages[page_num].extract_text() or ""
                    result[page_num] = text
            
            return {
                "pages": result,
                "total_pages": total_pages,
                "extracted_pages": len(result),
            }
        
        @tool
        def create_pdf(
            path: str,
            content: str | list[str],
            title: str | None = None,
            author: str | None = None,
            font_size: int = 12,
        ) -> dict[str, Any]:
            """Create a new PDF document.
            
            Args:
                path: Output path for the PDF file
                content: Text content (string or list of strings for multiple pages)
                title: Document title (metadata)
                author: Document author (metadata)
                font_size: Font size in points (default: 12)
            
            Returns:
                Dictionary with path, page count, and success status
            """
            file_path = self._validate_path(path)
            
            result = self._create_pdf(
                file_path,
                content,
                title=title,
                author=author,
                font_size=font_size,
            )
            return result.to_dict()
        
        @tool
        def merge_pdfs(
            paths: list[str],
            output_path: str,
        ) -> dict[str, Any]:
            """Merge multiple PDF files into one.
            
            Args:
                paths: List of PDF file paths to merge (in order)
                output_path: Path for the merged output PDF
            
            Returns:
                Dictionary with output path, total page count, and success status
            """
            input_paths = [self._validate_path(p) for p in paths]
            output_file = self._validate_path(output_path)
            
            for path in input_paths:
                self._check_file_size(path)
            
            result = self._merge_pdfs(input_paths, output_file)
            return result.to_dict()
        
        @tool
        def search_pdf(
            path: str,
            query: str,
            case_sensitive: bool = False,
        ) -> dict[str, Any]:
            """Search for text in a PDF file.
            
            Args:
                path: Path to the PDF file
                query: Text to search for
                case_sensitive: Whether search is case-sensitive
            
            Returns:
                Dictionary with matching pages and contexts
            """
            self._require_pypdf()
            from pypdf import PdfReader
            
            file_path = self._validate_path(path)
            self._check_file_size(file_path)
            
            reader = PdfReader(file_path)
            matches = []
            
            search_query = query if case_sensitive else query.lower()
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                search_text = text if case_sensitive else text.lower()
                
                if search_query in search_text:
                    # Find context around match
                    idx = search_text.find(search_query)
                    start = max(0, idx - 100)
                    end = min(len(text), idx + len(query) + 100)
                    context = text[start:end]
                    
                    matches.append({
                        "page": i,
                        "context": f"...{context}...",
                    })
            
            return {
                "query": query,
                "total_pages": len(reader.pages),
                "matches": matches,
                "match_count": len(matches),
            }
        
        @tool
        def split_pdf(
            path: str,
            output_dir: str,
            pages_per_file: int = 1,
        ) -> dict[str, Any]:
            """Split a PDF into multiple files.
            
            Args:
                path: Path to the PDF file to split
                output_dir: Directory for output files
                pages_per_file: Number of pages per output file (default: 1)
            
            Returns:
                Dictionary with list of output files created
            """
            self._require_pypdf()
            from pypdf import PdfReader, PdfWriter
            
            file_path = self._validate_path(path)
            output_directory = self._validate_path(output_dir)
            
            self._check_file_size(file_path)
            output_directory.mkdir(parents=True, exist_ok=True)
            
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            output_files = []
            base_name = file_path.stem
            
            for start in range(0, total_pages, pages_per_file):
                end = min(start + pages_per_file, total_pages)
                
                writer = PdfWriter()
                for i in range(start, end):
                    writer.add_page(reader.pages[i])
                
                output_file = output_directory / f"{base_name}_pages_{start+1}-{end}.pdf"
                with open(output_file, "wb") as f:
                    writer.write(f)
                
                output_files.append(str(output_file))
            
            return {
                "source": str(file_path),
                "output_files": output_files,
                "files_created": len(output_files),
                "total_pages": total_pages,
            }
        
        return [
            read_pdf,
            get_pdf_info,
            read_pdf_pages,
            create_pdf,
            merge_pdfs,
            search_pdf,
            split_pdf,
        ]
