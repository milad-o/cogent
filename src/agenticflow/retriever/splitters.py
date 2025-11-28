"""
Text Splitters/Chunkers for AgenticFlow.

Advanced text splitting strategies for optimal retrieval:

**Character-based:**
- RecursiveCharacterSplitter: Split by separators hierarchically
- CharacterSplitter: Simple character-based splitting

**Semantic:**
- SemanticSplitter: Split by semantic similarity (requires embeddings)
- SentenceSplitter: Split by sentence boundaries

**Structure-aware:**
- MarkdownSplitter: Respects markdown headings and structure
- HTMLSplitter: Splits HTML by tags and structure
- CodeSplitter: Language-aware code splitting

**Token-based:**
- TokenSplitter: Split by token count (requires tokenizer)

Example:
    >>> from agenticflow.retriever.splitters import (
    ...     RecursiveCharacterSplitter,
    ...     MarkdownSplitter,
    ...     CodeSplitter,
    ... )
    >>> 
    >>> # Basic splitting
    >>> splitter = RecursiveCharacterSplitter(chunk_size=1000, overlap=200)
    >>> chunks = splitter.split_text(text)
    >>> 
    >>> # Split documents
    >>> chunks = splitter.split_documents(documents)
    >>> 
    >>> # Markdown-aware
    >>> md_splitter = MarkdownSplitter(chunk_size=1000)
    >>> chunks = md_splitter.split_text(markdown_text)
    >>> 
    >>> # Code-aware
    >>> code_splitter = CodeSplitter(language="python", chunk_size=1500)
    >>> chunks = code_splitter.split_text(python_code)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from agenticflow.retriever.loaders import Document


@dataclass
class TextChunk:
    """A chunk of text with metadata.
    
    Attributes:
        content: The chunk text content.
        metadata: Metadata inherited from source + chunk info.
        start_index: Character position in original text.
        end_index: End character position in original text.
    """
    
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_index: int | None = None
    end_index: int | None = None
    
    def __len__(self) -> int:
        return len(self.content)
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"TextChunk(len={len(self.content)}, content='{preview}')"


# =============================================================================
# Base Splitter
# =============================================================================

class TextSplitter(ABC):
    """Base class for text splitters.
    
    Args:
        chunk_size: Target size for each chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        length_function: Function to measure text length (default: len).
        keep_separator: Whether to keep separators in output.
        strip_whitespace: Whether to strip whitespace from chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] | None = None,
        keep_separator: bool = False,
        strip_whitespace: bool = True,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
    
    @abstractmethod
    def split_text(self, text: str) -> list[TextChunk]:
        """Split text into chunks.
        
        Args:
            text: The text to split.
            
        Returns:
            List of TextChunk objects.
        """
        ...
    
    def split_documents(
        self,
        documents: Sequence[Any],  # Document type
    ) -> list[TextChunk]:
        """Split multiple documents.
        
        Args:
            documents: List of Document objects with content and metadata.
            
        Returns:
            List of TextChunk objects with inherited metadata.
        """
        chunks = []
        for doc in documents:
            doc_chunks = self.split_text(doc.content)
            for chunk in doc_chunks:
                # Inherit document metadata
                chunk.metadata = {**doc.metadata, **chunk.metadata}
            chunks.extend(doc_chunks)
        return chunks
    
    def _merge_splits(
        self,
        splits: list[str],
        separator: str = "",
    ) -> list[TextChunk]:
        """Merge splits into chunks respecting size limits.
        
        Args:
            splits: List of text pieces to merge.
            separator: Separator to use when joining.
            
        Returns:
            List of merged TextChunk objects.
        """
        chunks: list[TextChunk] = []
        current_chunk: list[str] = []
        current_length = 0
        current_start = 0
        position = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # Check if adding this split exceeds chunk size
            total = current_length + split_length
            if current_chunk:
                total += self.length_function(separator)
            
            if total > self.chunk_size and current_chunk:
                # Save current chunk
                content = separator.join(current_chunk)
                if self.strip_whitespace:
                    content = content.strip()
                if content:
                    chunks.append(TextChunk(
                        content=content,
                        start_index=current_start,
                        end_index=position,
                        metadata={"chunk_index": len(chunks)},
                    ))
                
                # Handle overlap
                overlap_start = current_start
                while current_chunk and current_length > self.chunk_overlap:
                    popped = current_chunk.pop(0)
                    current_length -= self.length_function(popped)
                    if current_chunk:
                        current_length -= self.length_function(separator)
                    overlap_start += self.length_function(popped) + self.length_function(separator)
                
                current_start = overlap_start
            
            current_chunk.append(split)
            current_length += split_length
            if len(current_chunk) > 1:
                current_length += self.length_function(separator)
            position += split_length + self.length_function(separator)
        
        # Add final chunk
        if current_chunk:
            content = separator.join(current_chunk)
            if self.strip_whitespace:
                content = content.strip()
            if content:
                chunks.append(TextChunk(
                    content=content,
                    start_index=current_start,
                    end_index=position,
                    metadata={"chunk_index": len(chunks)},
                ))
        
        return chunks


# =============================================================================
# Character-based Splitters
# =============================================================================

class RecursiveCharacterSplitter(TextSplitter):
    """Split text recursively using a hierarchy of separators.
    
    Tries to split by larger semantic units first (paragraphs),
    then falls back to smaller units (sentences, words, characters).
    
    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        separators: List of separators to try in order.
        
    Example:
        >>> splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
        >>> chunks = splitter.split_text(long_text)
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",      # Paragraphs
        "\n",        # Lines
        ". ",        # Sentences (with space)
        "! ",        # Exclamations
        "? ",        # Questions
        "; ",        # Semicolons
        ", ",        # Clauses
        " ",         # Words
        "",          # Characters
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separators = separators or self.DEFAULT_SEPARATORS.copy()
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Split text using recursive separator strategy."""
        return self._split_text(text, self.separators)
    
    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[TextChunk]:
        """Recursively split text."""
        # Base case: no separators left
        if not separators:
            return [TextChunk(content=text, metadata={"chunk_index": 0})]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        if separator:
            if self.keep_separator:
                # Keep separator at end of each piece
                splits = re.split(f"({re.escape(separator)})", text)
                # Recombine: pair each content with its separator
                merged = []
                for i in range(0, len(splits) - 1, 2):
                    merged.append(splits[i] + (splits[i + 1] if i + 1 < len(splits) else ""))
                if len(splits) % 2 == 1:
                    merged.append(splits[-1])
                splits = merged
            else:
                splits = text.split(separator)
        else:
            # Character-level split
            splits = list(text)
        
        # Process splits
        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # If single split is too large, recursively split it
            if split_length > self.chunk_size:
                # First, save current accumulated parts
                if current_parts:
                    chunks.extend(
                        self._merge_splits(current_parts, separator if self.keep_separator else "")
                    )
                    current_parts = []
                    current_length = 0
                
                # Recursively split the large piece
                if remaining_separators:
                    sub_chunks = self._split_text(split, remaining_separators)
                    chunks.extend(sub_chunks)
                else:
                    # Can't split further, just add as is
                    chunks.append(TextChunk(
                        content=split,
                        metadata={"chunk_index": len(chunks)},
                    ))
                continue
            
            # Check if we need to start a new chunk
            total_length = current_length + split_length
            if current_parts:
                total_length += self.length_function(separator)
            
            if total_length > self.chunk_size and current_parts:
                # Create chunk from accumulated parts
                chunks.extend(
                    self._merge_splits(current_parts, separator if self.keep_separator else "")
                )
                
                # Start new chunk with overlap
                overlap_parts = []
                overlap_length = 0
                for part in reversed(current_parts):
                    part_len = self.length_function(part)
                    if overlap_length + part_len <= self.chunk_overlap:
                        overlap_parts.insert(0, part)
                        overlap_length += part_len
                    else:
                        break
                
                current_parts = overlap_parts
                current_length = overlap_length
            
            current_parts.append(split)
            current_length += split_length
        
        # Handle remaining parts
        if current_parts:
            chunks.extend(
                self._merge_splits(current_parts, separator if self.keep_separator else "")
            )
        
        # Renumber chunk indices
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks


class CharacterSplitter(TextSplitter):
    """Simple character-based splitter using a single separator.
    
    Args:
        separator: The separator to split on.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        
    Example:
        >>> splitter = CharacterSplitter(separator="\\n\\n", chunk_size=1000)
        >>> chunks = splitter.split_text(text)
    """
    
    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separator = separator
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Split text by separator."""
        if self.separator:
            splits = text.split(self.separator)
        else:
            splits = list(text)
        
        return self._merge_splits(splits, self.separator if self.keep_separator else " ")


# =============================================================================
# Sentence-based Splitter
# =============================================================================

class SentenceSplitter(TextSplitter):
    """Split text by sentence boundaries.
    
    Uses regex to identify sentence endings and respects
    common abbreviations to avoid false splits.
    
    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        min_sentence_length: Minimum characters for a valid sentence.
        
    Example:
        >>> splitter = SentenceSplitter(chunk_size=500)
        >>> chunks = splitter.split_text(text)
    """
    
    # Common abbreviations that don't end sentences
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "vs", "etc", "fig", "eg", "ie", "al", "vol",
        "inc", "ltd", "corp", "co",
    }
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_sentence_length: int = 10,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.min_sentence_length = min_sentence_length
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Split text into sentences then merge into chunks."""
        sentences = self._split_sentences(text)
        return self._merge_splits(sentences, " ")
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences."""
        # Pattern matches sentence-ending punctuation followed by space and capital
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # Split on the pattern
        raw_sentences = re.split(pattern, text)
        
        sentences = []
        current = ""
        
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Check if previous sentence ended with abbreviation
            if current:
                last_word = current.split()[-1].rstrip(".").lower() if current.split() else ""
                if last_word in self.ABBREVIATIONS:
                    current = current + " " + sent
                    continue
            
            if current and len(current) >= self.min_sentence_length:
                sentences.append(current)
            
            current = sent
        
        # Add final sentence
        if current and len(current) >= self.min_sentence_length:
            sentences.append(current)
        elif current and sentences:
            sentences[-1] = sentences[-1] + " " + current
        elif current:
            sentences.append(current)
        
        return sentences


# =============================================================================
# Markdown Splitter
# =============================================================================

class MarkdownSplitter(TextSplitter):
    """Split markdown text respecting document structure.
    
    Preserves heading hierarchy and splits at section boundaries.
    Each chunk includes its heading context.
    
    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        headers_to_split_on: List of header levels to split on.
        return_each_section: Return each section as separate chunk.
        
    Example:
        >>> splitter = MarkdownSplitter(chunk_size=1000)
        >>> chunks = splitter.split_text(markdown_text)
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        headers_to_split_on: list[str] | None = None,
        return_each_section: bool = False,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.headers_to_split_on = headers_to_split_on or ["#", "##", "###"]
        self.return_each_section = return_each_section
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Split markdown by headers and content."""
        sections = self._split_by_headers(text)
        
        if self.return_each_section:
            chunks = []
            for section in sections:
                chunks.append(TextChunk(
                    content=section["content"],
                    metadata={
                        "headers": section["headers"],
                        "chunk_index": len(chunks),
                    }
                ))
            return chunks
        
        # Merge sections into appropriately sized chunks
        chunks: list[TextChunk] = []
        current_content: list[str] = []
        current_length = 0
        current_headers: dict[str, str] = {}
        
        for section in sections:
            section_text = section["content"]
            section_length = self.length_function(section_text)
            
            # If section is too large, split it further
            if section_length > self.chunk_size:
                # Save current chunk first
                if current_content:
                    chunks.append(TextChunk(
                        content="\n\n".join(current_content),
                        metadata={
                            "headers": dict(current_headers),
                            "chunk_index": len(chunks),
                        }
                    ))
                    current_content = []
                    current_length = 0
                
                # Split large section with recursive splitter
                sub_splitter = RecursiveCharacterSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                sub_chunks = sub_splitter.split_text(section_text)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["headers"] = section["headers"]
                    sub_chunk.metadata["chunk_index"] = len(chunks)
                    chunks.append(sub_chunk)
                
                current_headers = section["headers"].copy()
                continue
            
            # Check if we need to start new chunk
            if current_length + section_length > self.chunk_size and current_content:
                chunks.append(TextChunk(
                    content="\n\n".join(current_content),
                    metadata={
                        "headers": dict(current_headers),
                        "chunk_index": len(chunks),
                    }
                ))
                current_content = []
                current_length = 0
            
            current_content.append(section_text)
            current_length += section_length
            current_headers.update(section["headers"])
        
        # Add final chunk
        if current_content:
            chunks.append(TextChunk(
                content="\n\n".join(current_content),
                metadata={
                    "headers": dict(current_headers),
                    "chunk_index": len(chunks),
                }
            ))
        
        return chunks
    
    def _split_by_headers(self, text: str) -> list[dict[str, Any]]:
        """Split markdown into sections by headers."""
        # Build pattern from header levels
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = text.split("\n")
        sections: list[dict[str, Any]] = []
        current_headers: dict[str, str] = {}
        current_content: list[str] = []
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        sections.append({
                            "headers": dict(current_headers),
                            "content": content_text,
                        })
                    current_content = []
                
                # Update header hierarchy
                level = header_match.group(1)
                title = header_match.group(2).strip()
                
                # Clear lower-level headers
                levels_to_clear = [h for h in current_headers if len(h) >= len(level)]
                for h in levels_to_clear:
                    del current_headers[h]
                
                current_headers[level] = title
                
                # Add header to content if desired
                if level in self.headers_to_split_on:
                    current_content.append(line)
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:
                sections.append({
                    "headers": dict(current_headers),
                    "content": content_text,
                })
        
        return sections


# =============================================================================
# HTML Splitter
# =============================================================================

class HTMLSplitter(TextSplitter):
    """Split HTML content by semantic tags.
    
    Respects HTML structure and splits at tag boundaries.
    
    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        tags_to_split_on: HTML tags to use as split points.
        
    Example:
        >>> splitter = HTMLSplitter(chunk_size=1000)
        >>> chunks = splitter.split_text(html_content)
    """
    
    DEFAULT_SPLIT_TAGS = [
        "article", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "table", "tr", "blockquote", "pre", "code",
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tags_to_split_on: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.tags_to_split_on = tags_to_split_on or self.DEFAULT_SPLIT_TAGS
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Split HTML by tags."""
        # Try to use BeautifulSoup
        try:
            from bs4 import BeautifulSoup
            return self._split_with_bs4(text)
        except ImportError:
            return self._split_with_regex(text)
    
    def _split_with_bs4(self, text: str) -> list[TextChunk]:
        """Split using BeautifulSoup."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(text, "html.parser")
        
        # Remove script and style
        for element in soup(["script", "style"]):
            element.decompose()
        
        sections: list[str] = []
        
        # Find all split-worthy elements
        for tag in self.tags_to_split_on:
            for element in soup.find_all(tag):
                text_content = element.get_text(separator=" ", strip=True)
                if text_content:
                    sections.append(text_content)
        
        # If no sections found, get all text
        if not sections:
            sections = [soup.get_text(separator=" ", strip=True)]
        
        return self._merge_splits(sections, "\n\n")
    
    def _split_with_regex(self, text: str) -> list[TextChunk]:
        """Fallback regex-based splitting."""
        # Build pattern for tags
        tag_pattern = "|".join(self.tags_to_split_on)
        pattern = rf'<({tag_pattern})[^>]*>(.*?)</\1>'
        
        sections = []
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            content = re.sub(r'<[^>]+>', ' ', match.group(2))
            content = re.sub(r'\s+', ' ', content).strip()
            if content:
                sections.append(content)
        
        if not sections:
            # Just strip all tags and split
            plain_text = re.sub(r'<[^>]+>', ' ', text)
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            sections = [plain_text]
        
        return self._merge_splits(sections, "\n\n")


# =============================================================================
# Code Splitter
# =============================================================================

class CodeSplitter(TextSplitter):
    """Split source code respecting language syntax.
    
    Splits at function/class boundaries and preserves code blocks.
    
    Args:
        language: Programming language (python, javascript, java, etc.)
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        
    Example:
        >>> splitter = CodeSplitter(language="python", chunk_size=1500)
        >>> chunks = splitter.split_text(python_code)
    """
    
    # Language-specific split patterns
    LANGUAGE_PATTERNS: dict[str, list[str]] = {
        "python": [
            r'\nclass\s+\w+',           # Class definitions
            r'\ndef\s+\w+',             # Function definitions
            r'\nasync\s+def\s+\w+',     # Async functions
            r'\n@\w+',                  # Decorators
            r'\n\n',                    # Double newlines
        ],
        "javascript": [
            r'\nclass\s+\w+',
            r'\nfunction\s+\w+',
            r'\nconst\s+\w+\s*=\s*(?:async\s+)?\(',
            r'\nlet\s+\w+\s*=\s*(?:async\s+)?\(',
            r'\nexport\s+',
            r'\n\n',
        ],
        "typescript": [
            r'\nclass\s+\w+',
            r'\ninterface\s+\w+',
            r'\ntype\s+\w+',
            r'\nfunction\s+\w+',
            r'\nconst\s+\w+\s*=\s*(?:async\s+)?\(',
            r'\nexport\s+',
            r'\n\n',
        ],
        "java": [
            r'\npublic\s+class\s+\w+',
            r'\nprivate\s+class\s+\w+',
            r'\nprotected\s+class\s+\w+',
            r'\npublic\s+\w+\s+\w+\s*\(',
            r'\nprivate\s+\w+\s+\w+\s*\(',
            r'\n\n',
        ],
        "go": [
            r'\nfunc\s+',
            r'\ntype\s+\w+\s+struct',
            r'\ntype\s+\w+\s+interface',
            r'\n\n',
        ],
        "rust": [
            r'\nfn\s+\w+',
            r'\nimpl\s+',
            r'\nstruct\s+\w+',
            r'\nenum\s+\w+',
            r'\ntrait\s+\w+',
            r'\n\n',
        ],
        "cpp": [
            r'\nclass\s+\w+',
            r'\nstruct\s+\w+',
            r'\n\w+\s+\w+\s*\([^)]*\)\s*{',  # Function definitions
            r'\n\n',
        ],
        "c": [
            r'\nstruct\s+\w+',
            r'\n\w+\s+\w+\s*\([^)]*\)\s*{',
            r'\n\n',
        ],
    }
    
    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.language = language.lower()
        self.patterns = self.LANGUAGE_PATTERNS.get(
            self.language,
            [r'\n\n', r'\n']  # Default fallback
        )
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Split code by language-specific patterns."""
        # Try each pattern in order
        for pattern in self.patterns:
            sections = self._split_by_pattern(text, pattern)
            if len(sections) > 1:
                chunks = self._merge_splits(sections, "\n")
                if chunks:
                    for chunk in chunks:
                        chunk.metadata["language"] = self.language
                    return chunks
        
        # Fallback to simple line splitting
        lines = text.split("\n")
        chunks = self._merge_splits(lines, "\n")
        for chunk in chunks:
            chunk.metadata["language"] = self.language
        return chunks
    
    def _split_by_pattern(self, text: str, pattern: str) -> list[str]:
        """Split text by regex pattern, keeping the matched delimiter."""
        splits = re.split(f'({pattern})', text)
        
        # Recombine: each split starts with its delimiter
        sections = []
        current = ""
        
        for i, part in enumerate(splits):
            if re.match(pattern, part):
                if current.strip():
                    sections.append(current)
                current = part
            else:
                current += part
        
        if current.strip():
            sections.append(current)
        
        return sections


# =============================================================================
# Semantic Splitter
# =============================================================================

class SemanticSplitter(TextSplitter):
    """Split text by semantic similarity using embeddings.
    
    Groups semantically similar sentences together and splits
    when semantic similarity drops below threshold.
    
    Args:
        embedding_model: Embedding model for computing similarity.
        chunk_size: Target chunk size.
        breakpoint_threshold: Similarity threshold for splits (0-1).
        buffer_size: Number of sentences to compare.
        
    Example:
        >>> from agenticflow.models import EmbeddingModel
        >>> embedder = EmbeddingModel()
        >>> splitter = SemanticSplitter(embedding_model=embedder)
        >>> chunks = await splitter.asplit_text(text)
    """
    
    def __init__(
        self,
        embedding_model: Any = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        breakpoint_threshold: float = 0.5,
        buffer_size: int = 1,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._embedding_model = embedding_model
        self.breakpoint_threshold = breakpoint_threshold
        self.buffer_size = buffer_size
    
    @property
    def embedding_model(self) -> Any:
        """Get or create embedding model."""
        if self._embedding_model is None:
            from agenticflow.models import EmbeddingModel
            self._embedding_model = EmbeddingModel()
        return self._embedding_model
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Synchronous split - falls back to sentence splitting."""
        # Can't do async embedding in sync context easily
        sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return sentence_splitter.split_text(text)
    
    async def asplit_text(self, text: str) -> list[TextChunk]:
        """Asynchronously split text by semantic similarity."""
        # First split into sentences
        sentence_splitter = SentenceSplitter(min_sentence_length=5)
        sentence_chunks = sentence_splitter.split_text(text)
        sentences = [c.content for c in sentence_chunks]
        
        if len(sentences) <= 1:
            return [TextChunk(content=text, metadata={"chunk_index": 0})]
        
        # Get embeddings for all sentences
        embeddings = await self.embedding_model.aembed(sentences)
        
        # Find breakpoints based on similarity
        breakpoints = self._find_breakpoints(embeddings)
        
        # Group sentences by breakpoints
        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        
        for i, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            
            if i in breakpoints or i == len(sentences) - 1:
                content = " ".join(current_sentences)
                
                # Check size and split if needed
                if self.length_function(content) > self.chunk_size:
                    # Split large chunk with recursive splitter
                    sub_splitter = RecursiveCharacterSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    sub_chunks = sub_splitter.split_text(content)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(TextChunk(
                        content=content,
                        metadata={"chunk_index": len(chunks)},
                    ))
                
                current_sentences = []
        
        # Renumber
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks
    
    def _find_breakpoints(self, embeddings: list[list[float]]) -> set[int]:
        """Find indices where semantic similarity drops."""
        import math
        
        breakpoints: set[int] = set()
        
        for i in range(1, len(embeddings)):
            # Compare with previous buffer
            start = max(0, i - self.buffer_size)
            prev_embeddings = embeddings[start:i]
            curr_embedding = embeddings[i]
            
            # Compute average similarity with previous sentences
            similarities = []
            for prev_emb in prev_embeddings:
                sim = self._cosine_similarity(prev_emb, curr_embedding)
                similarities.append(sim)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
            
            if avg_similarity < self.breakpoint_threshold:
                breakpoints.add(i - 1)  # Break after previous sentence
        
        return breakpoints
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


# =============================================================================
# Token-based Splitter
# =============================================================================

class TokenSplitter(TextSplitter):
    """Split text by token count.
    
    Uses tiktoken for accurate OpenAI token counting.
    
    Args:
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap in tokens.
        model_name: Model name for tiktoken encoding.
        
    Example:
        >>> splitter = TokenSplitter(chunk_size=500, model_name="gpt-4")
        >>> chunks = splitter.split_text(text)
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model_name: str = "gpt-4",
        **kwargs: Any,
    ):
        self._tokenizer = None
        self._model_name = model_name
        
        # Initialize parent with token length function
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            **kwargs,
        )
    
    @property
    def tokenizer(self) -> Any:
        """Lazy-load tiktoken encoder."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.encoding_for_model(self._model_name)
            except ImportError:
                raise ImportError(
                    "Token splitting requires 'tiktoken'. "
                    "Install with: pip install tiktoken"
                )
            except KeyError:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer
    
    def _token_length(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def split_text(self, text: str) -> list[TextChunk]:
        """Split text by token count."""
        # Use recursive character splitting with token length function
        splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
        )
        chunks = splitter.split_text(text)
        
        # Add token count metadata
        for chunk in chunks:
            chunk.metadata["token_count"] = self._token_length(chunk.content)
        
        return chunks


# =============================================================================
# Convenience Functions
# =============================================================================

def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter_type: str = "recursive",
    **kwargs: Any,
) -> list[TextChunk]:
    """Split text using specified strategy.
    
    Args:
        text: Text to split.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        splitter_type: Type of splitter (recursive, sentence, markdown, code, token).
        **kwargs: Additional splitter arguments.
        
    Returns:
        List of TextChunk objects.
        
    Example:
        >>> chunks = split_text(text, chunk_size=500, splitter_type="recursive")
        >>> chunks = split_text(markdown, splitter_type="markdown")
        >>> chunks = split_text(code, splitter_type="code", language="python")
    """
    splitters = {
        "recursive": RecursiveCharacterSplitter,
        "character": CharacterSplitter,
        "sentence": SentenceSplitter,
        "markdown": MarkdownSplitter,
        "html": HTMLSplitter,
        "code": CodeSplitter,
        "token": TokenSplitter,
    }
    
    splitter_class = splitters.get(splitter_type.lower())
    if not splitter_class:
        raise ValueError(
            f"Unknown splitter type: {splitter_type}. "
            f"Supported: {', '.join(splitters.keys())}"
        )
    
    splitter = splitter_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    
    return splitter.split_text(text)


__all__ = [
    # Core types
    "TextChunk",
    "TextSplitter",
    # Character-based
    "RecursiveCharacterSplitter",
    "CharacterSplitter",
    # Sentence-based
    "SentenceSplitter",
    # Structure-aware
    "MarkdownSplitter",
    "HTMLSplitter",
    "CodeSplitter",
    # Semantic
    "SemanticSplitter",
    # Token-based
    "TokenSplitter",
    # Convenience
    "split_text",
]
