"""Table-aware document splitter for HTML content.

Splits HTML documents by tables, creating separate chunks for each table
with its caption and surrounding context. Ideal for RAG over structured data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agenticflow.document.splitters.base import BaseSplitter

if TYPE_CHECKING:
    from agenticflow.vectorstore import Document


@dataclass
class TableChunk:
    """A chunk containing a table with context."""
    
    table_number: int | None
    caption: str | None
    table_html: str
    preceding_context: str
    following_context: str
    
    def to_text(self, *, include_context: bool = True) -> str:
        """Convert to plain text representation.
        
        Args:
            include_context: Include preceding/following text.
            
        Returns:
            Combined text content.
        """
        parts = []
        
        if include_context and self.preceding_context:
            parts.append(self.preceding_context)
        
        if self.caption:
            parts.append(self.caption)
        
        parts.append(self.table_html)
        
        if include_context and self.following_context:
            parts.append(self.following_context)
        
        return '\n'.join(parts)


class TableAwareSplitter(BaseSplitter):
    """Split HTML documents by tables.
    
    Creates separate chunks for:
    1. Each table with its caption and surrounding context
    2. Text-only sections between tables
    
    This improves retrieval by:
    - Isolating each table for focused matching
    - Preserving table captions with their data
    - Reducing noise from multiple tables on same page
    
    Example:
        >>> from agenticflow.document.loaders.handlers import PDFHTMLLoader
        >>> from agenticflow.document.splitters import TableAwareSplitter
        >>> 
        >>> loader = PDFHTMLLoader()
        >>> docs = await loader.load("document.pdf")
        >>> 
        >>> splitter = TableAwareSplitter(
        ...     context_lines=2,
        ...     include_caption=True,
        ... )
        >>> chunks = splitter.split_documents(docs)
    """
    
    def __init__(
        self,
        *,
        context_before: int = 1,
        context_after: int = 1,
        min_text_chunk_size: int = 100,
        preserve_metadata: bool = True,
    ) -> None:
        """Create a table-aware splitter.
        
        Args:
            context_before: Number of elements before table to include as context.
            context_after: Number of elements after table to include as context.
            min_text_chunk_size: Minimum size for text-only chunks (chars).
            preserve_metadata: Preserve original document metadata in chunks.
        """
        self.context_before = context_before
        self.context_after = context_after
        self.min_text_chunk_size = min_text_chunk_size
        self.preserve_metadata = preserve_metadata
    
    def split_text(self, text: str) -> list[str]:
        """Split HTML text into table-aware chunks.
        
        Args:
            text: HTML content to split.
            
        Returns:
            List of HTML chunks.
        """
        chunks = []
        
        # Parse HTML into elements
        elements = self._parse_html_elements(text)
        
        # Group elements into table chunks
        table_chunks = self._extract_table_chunks(elements)
        
        for chunk in table_chunks:
            chunks.append(chunk.to_text(include_context=True))
        
        return chunks
    
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents by tables.
        
        Args:
            documents: Documents to split (should contain HTML).
            
        Returns:
            New documents with one table or text section per document.
        """
        from agenticflow.vectorstore import Document
        
        split_docs = []
        
        for doc in documents:
            # Parse HTML elements
            elements = self._parse_html_elements(doc.content)
            
            # Extract table chunks
            table_chunks = self._extract_table_chunks(elements)
            
            for i, chunk in enumerate(table_chunks):
                # Build metadata
                metadata = doc.metadata.copy() if self.preserve_metadata else {}
                metadata.update({
                    'chunk_index': i,
                    'chunk_type': 'table' if chunk.table_html else 'text',
                })
                
                if chunk.table_number is not None:
                    metadata['table_number'] = chunk.table_number
                if chunk.caption:
                    metadata['table_caption'] = chunk.caption
                
                # Create new document
                split_docs.append(Document(
                text=chunk.to_text(include_context=True),
                ))
        
        return split_docs
    
    def _parse_html_elements(self, html: str) -> list[dict[str, str]]:
        """Parse HTML into structured elements.
        
        Args:
            html: HTML content.
            
        Returns:
            List of elements with type and content.
        """
        elements = []
        
        # Pattern to match HTML tags
        # Matches: <tag ...>content</tag>
        pattern = r'<(h[1-6]|p|table)[^>]*>(.*?)</\1>'
        
        for match in re.finditer(pattern, html, re.DOTALL):
            tag = match.group(1)
            content = match.group(0)  # Full match including tags
            
            element_type = 'heading' if tag.startswith('h') else tag
            elements.append({
                'type': element_type,
                'tag': tag,
                'content': content,
                'text': match.group(2),  # Content without tags
            })
        
        return elements
    
    def _extract_table_chunks(self, elements: list[dict[str, str]]) -> list[TableChunk]:
        """Extract table chunks with context.
        
        Args:
            elements: Parsed HTML elements.
            
        Returns:
            List of table chunks with surrounding context.
        """
        chunks = []
        i = 0
        
        while i < len(elements):
            element = elements[i]
            
            # Check if this is a table
            if element['type'] == 'table':
                # Get context before (any elements)
                start_idx = max(0, i - self.context_before)
                preceding = []
                for j in range(start_idx, i):
                    preceding.append(elements[j]['content'])
                
                # Get context after (any elements, stop at next table)
                following = []
                for j in range(i + 1, min(i + 1 + self.context_after, len(elements))):
                    if elements[j]['type'] == 'table':
                        break  # Stop at next table
                    following.append(elements[j]['content'])
                
                # Try to extract table number from any surrounding text
                table_number = None
                caption = None
                all_context = '\n'.join(preceding + following)
                match = re.search(r'Table (\d+)[:\s]', all_context)
                if match:
                    table_number = int(match.group(1))
                    # Extract full caption line
                    for line in all_context.split('\n'):
                        if f'Table {table_number}' in line:
                            caption = line.strip()
                            break
                
                # Create table chunk
                chunks.append(TableChunk(
                    table_number=table_number,
                    caption=caption,
                    table_html=element['content'],
                    preceding_context='\n'.join(preceding),
                    following_context='\n'.join(following),
                ))
                
                # Move past this table
                i += 1
                
            else:
                # Accumulate text elements between tables
                text_elements = [element['content']]
                i += 1
                
                # Collect consecutive non-table elements
                while i < len(elements) and elements[i]['type'] != 'table':
                    text_elements.append(elements[i]['content'])
                    i += 1
                
                # Create text chunk if substantial
                text_content = '\n'.join(text_elements)
                if len(text_content) >= self.min_text_chunk_size:
                    chunks.append(TableChunk(
                        table_number=None,
                        caption=None,
                        table_html='',
                        preceding_context=text_content,
                        following_context='',
                    ))
        
        return chunks
