"""Tests for the document module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agenticflow.document import (
    # Types
    Document,
    TextChunk,
    FileType,
    SplitterType,
    # Loaders
    DocumentLoader,
    BaseLoader,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    CodeLoader,
    CSVLoader,
    JSONLoader,
    LOADERS,
    load_documents,
    load_documents_sync,
    register_loader,
    # Splitters
    BaseSplitter,
    RecursiveCharacterSplitter,
    CharacterSplitter,
    SentenceSplitter,
    MarkdownSplitter,
    HTMLSplitter,
    CodeSplitter,
    split_text,
)


# ============================================================================
# Test Types
# ============================================================================


class TestFileType:
    """Tests for FileType enum."""
    
    def test_from_extension_basic(self) -> None:
        """Test basic extension lookup."""
        assert FileType.from_extension(".py") == FileType.PYTHON
        assert FileType.from_extension(".md") == FileType.MARKDOWN
        assert FileType.from_extension(".json") == FileType.JSON
    
    def test_from_extension_without_dot(self) -> None:
        """Test extension lookup without leading dot."""
        assert FileType.from_extension("py") == FileType.PYTHON
        assert FileType.from_extension("md") == FileType.MARKDOWN
    
    def test_from_extension_aliases(self) -> None:
        """Test extension aliases."""
        assert FileType.from_extension(".yml") == FileType.YAML
        assert FileType.from_extension(".jsx") == FileType.JAVASCRIPT
        assert FileType.from_extension(".ndjson") == FileType.JSONL
    
    def test_from_extension_unknown(self) -> None:
        """Test unknown extension returns None."""
        assert FileType.from_extension(".xyz123") is None


class TestSplitterType:
    """Tests for SplitterType enum."""
    
    def test_all_splitter_types_exist(self) -> None:
        """Test all expected splitter types are defined."""
        assert SplitterType.RECURSIVE is not None
        assert SplitterType.CHARACTER is not None
        assert SplitterType.SENTENCE is not None
        assert SplitterType.MARKDOWN is not None
        assert SplitterType.HTML is not None
        assert SplitterType.CODE is not None
        assert SplitterType.SEMANTIC is not None
        assert SplitterType.TOKEN is not None


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_create_document(self) -> None:
        """Test creating a document."""
        doc = Document(text="Test content", metadata={"key": "value"})
        assert doc.text == "Test content"
        assert doc.content == "Test content"  # Alias works
        assert doc.metadata["key"] == "value"
    
    def test_document_auto_source(self) -> None:
        """Test that source is auto-added to metadata."""
        doc = Document(text="Test")
        assert "source" in doc.metadata
        assert doc.metadata["source"] == "unknown"
    
    def test_document_length(self) -> None:
        """Test document length."""
        doc = Document(text="Hello")
        assert len(doc) == 5
    
    def test_document_repr(self) -> None:
        """Test document string representation."""
        doc = Document(text="Short content")
        assert "Short content" in repr(doc)
    
    def test_document_to_dict(self) -> None:
        """Test converting document to dict."""
        doc = Document(text="Test", metadata={"key": "value"})
        d = doc.to_dict()
        assert d["text"] == "Test"
        assert d["metadata"]["key"] == "value"
    
    def test_document_from_dict(self) -> None:
        """Test creating document from dict."""
        d = {"text": "Test", "metadata": {"key": "value"}}
        doc = Document.from_dict(d)
        assert doc.text == "Test"
        assert doc.metadata["key"] == "value"


class TestTextChunk:
    """Tests for TextChunk dataclass."""
    
    def test_create_chunk(self) -> None:
        """Test creating a text chunk."""
        chunk = TextChunk(text="Test content", metadata={"index": 0})
        assert chunk.text == "Test content"
        assert chunk.content == "Test content"  # Alias works
        assert chunk.metadata["index"] == 0
    
    def test_chunk_length(self) -> None:
        """Test chunk length."""
        chunk = TextChunk(text="Hello")
        assert len(chunk) == 5
    
    def test_chunk_with_indices(self) -> None:
        """Test chunk with start/end indices."""
        chunk = TextChunk(text="Test", start_index=0, end_index=4)
        assert chunk.start_index == 0
        assert chunk.end_index == 4
    
    def test_chunk_to_dict(self) -> None:
        """Test converting chunk to dict."""
        chunk = TextChunk(text="Test", start_index=0, end_index=4)
        d = chunk.to_dict()
        assert d["text"] == "Test"
        assert d["start_index"] == 0
        assert d["end_index"] == 4
    
    def test_chunk_from_dict(self) -> None:
        """Test creating chunk from dict."""
        d = {"text": "Test", "metadata": {}, "start_index": 0, "end_index": 4}
        chunk = TextChunk.from_dict(d)
        assert chunk.text == "Test"
        assert chunk.start_index == 0
    
    def test_chunk_to_document(self) -> None:
        """Test converting chunk to document."""
        chunk = TextChunk(text="Test", metadata={"source": "test.txt"})
        doc = chunk.to_document()
        assert isinstance(doc, Document)
        assert doc.content == "Test"
        assert doc.metadata["source"] == "test.txt"


# ============================================================================
# Test Loaders
# ============================================================================


class TestBaseLoader:
    """Tests for BaseLoader abstract class."""
    
    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()


class TestIndividualLoaders:
    """Tests for individual loader classes."""
    
    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)
    
    @pytest.mark.asyncio
    async def test_text_loader(self, temp_dir: Path) -> None:
        """Test TextLoader."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("Hello, World!")
        
        loader = TextLoader()
        docs = await loader.load(text_file)
        
        assert len(docs) == 1
        assert docs[0].content == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_markdown_loader(self, temp_dir: Path) -> None:
        """Test MarkdownLoader."""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Title\n\nContent here")
        
        loader = MarkdownLoader()
        docs = await loader.load(md_file)
        
        assert len(docs) == 1
        assert "Title" in docs[0].content
    
    @pytest.mark.asyncio
    async def test_markdown_loader_with_frontmatter(self, temp_dir: Path) -> None:
        """Test MarkdownLoader with YAML frontmatter."""
        md_file = temp_dir / "test.md"
        md_file.write_text("---\ntitle: My Doc\nauthor: Test\n---\n\n# Content")
        
        loader = MarkdownLoader()
        docs = await loader.load(md_file)
        
        assert docs[0].metadata.get("title") == "My Doc"
        assert docs[0].metadata.get("author") == "Test"
    
    @pytest.mark.asyncio
    async def test_html_loader(self, temp_dir: Path) -> None:
        """Test HTMLLoader."""
        html_file = temp_dir / "test.html"
        html_file.write_text("<html><body><p>Content</p></body></html>")
        
        loader = HTMLLoader()
        docs = await loader.load(html_file)
        
        assert len(docs) == 1
        assert "Content" in docs[0].content
        assert "<p>" not in docs[0].content  # Tags removed
    
    @pytest.mark.asyncio
    async def test_code_loader(self, temp_dir: Path) -> None:
        """Test CodeLoader."""
        py_file = temp_dir / "test.py"
        py_file.write_text("def hello():\n    print('Hello')")
        
        loader = CodeLoader()
        docs = await loader.load(py_file)
        
        assert len(docs) == 1
        assert "def hello" in docs[0].content
        assert docs[0].metadata["language"] == "python"
    
    @pytest.mark.asyncio
    async def test_csv_loader(self, temp_dir: Path) -> None:
        """Test CSVLoader."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25")
        
        loader = CSVLoader()
        docs = await loader.load(csv_file)
        
        assert len(docs) == 1
        assert "Alice" in docs[0].content
        assert docs[0].metadata["row_count"] == 2
    
    @pytest.mark.asyncio
    async def test_json_loader(self, temp_dir: Path) -> None:
        """Test JSONLoader."""
        json_file = temp_dir / "test.json"
        json_file.write_text('{"key": "value"}')
        
        loader = JSONLoader()
        docs = await loader.load(json_file)
        
        assert len(docs) == 1
        assert "key" in docs[0].content


class TestLoadersRegistry:
    """Tests for loader registry."""
    
    def test_loaders_proxy_contains(self) -> None:
        """Test LOADERS proxy __contains__."""
        assert ".py" in LOADERS
        assert ".xyz123" not in LOADERS
    
    def test_loaders_proxy_keys(self) -> None:
        """Test LOADERS proxy keys."""
        keys = LOADERS.keys()
        assert ".py" in keys
        assert ".md" in keys
        assert ".json" in keys


# ============================================================================
# Test Splitters
# ============================================================================


class TestBaseSplitter:
    """Tests for BaseSplitter abstract class."""
    
    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseSplitter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSplitter()
    
    def test_chunk_overlap_validation(self) -> None:
        """Test that chunk_overlap must be less than chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=150)


class TestRecursiveCharacterSplitter:
    """Tests for RecursiveCharacterSplitter."""
    
    def test_basic_split(self) -> None:
        """Test basic recursive splitting."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=0)
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 2
        assert all(isinstance(c, Document) for c in chunks)
    
    def test_split_respects_chunk_size(self) -> None:
        """Test that chunks respect size limit."""
        text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        splitter = RecursiveCharacterSplitter(chunk_size=150, chunk_overlap=0)
        chunks = splitter.split_text(text)
        
        for chunk in chunks:
            assert len(chunk.text) <= 150 + 10  # Some tolerance
    
    def test_split_documents(self) -> None:
        """Test splitting multiple documents."""
        docs = [
            Document(text="First document content", metadata={"id": "1"}),
            Document(text="Second document content", metadata={"id": "2"}),
        ]
        splitter = RecursiveCharacterSplitter(chunk_size=15, chunk_overlap=0)
        chunks = splitter.split_documents(docs)
        
        assert len(chunks) >= 2
        # Chunks should inherit document metadata
        ids = {c.metadata.get("id") for c in chunks}
        assert "1" in ids or "2" in ids


class TestCharacterSplitter:
    """Tests for CharacterSplitter."""
    
    def test_basic_split(self) -> None:
        """Test basic character splitting."""
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        splitter = CharacterSplitter(separator="\n\n", chunk_size=20, chunk_overlap=0)
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1


class TestSentenceSplitter:
    """Tests for SentenceSplitter."""
    
    def test_split_sentences(self) -> None:
        """Test sentence-based splitting."""
        text = "First sentence. Second sentence. Third sentence."
        splitter = SentenceSplitter(chunk_size=40, chunk_overlap=0)
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
    
    def test_handles_abbreviations(self) -> None:
        """Test that abbreviations don't cause false splits."""
        text = "Dr. Smith went to the store. He bought apples."
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text(text)
        
        assert any("Dr. Smith" in c.content for c in chunks)


class TestMarkdownSplitter:
    """Tests for MarkdownSplitter."""
    
    def test_split_by_headers(self) -> None:
        """Test markdown splitting by headers."""
        text = """# Introduction

This is the intro.

## Section 1

Content for section 1.
"""
        splitter = MarkdownSplitter(chunk_size=200, chunk_overlap=0)
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
        assert any("headers" in c.metadata for c in chunks)


class TestCodeSplitter:
    """Tests for CodeSplitter."""
    
    def test_split_python_code(self) -> None:
        """Test splitting Python code."""
        code = """
def function_one():
    print("one")

def function_two():
    print("two")
"""
        splitter = CodeSplitter(language="python", chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text(code)
        
        assert len(chunks) >= 1
        assert all(c.metadata.get("language") == "python" for c in chunks)


class TestSplitTextFunction:
    """Tests for split_text convenience function."""
    
    def test_recursive_splitter(self) -> None:
        """Test using recursive splitter type."""
        text = "Content " * 100
        chunks = split_text(text, chunk_size=500, chunk_overlap=50, splitter_type="recursive")
        assert len(chunks) >= 1
    
    def test_invalid_splitter_type(self) -> None:
        """Test that invalid splitter type raises error."""
        with pytest.raises(ValueError, match="Unknown splitter type"):
            split_text("text", splitter_type="invalid")


# ============================================================================
# Test Module Structure
# ============================================================================


class TestModuleExports:
    """Tests for module-level exports."""
    
    def test_document_module_exports(self) -> None:
        """Test that document module exports all components."""
        from agenticflow import document
        
        # Types
        assert hasattr(document, "Document")
        assert hasattr(document, "TextChunk")
        assert hasattr(document, "FileType")
        assert hasattr(document, "SplitterType")
        
        # Loaders
        assert hasattr(document, "DocumentLoader")
        assert hasattr(document, "BaseLoader")
        assert hasattr(document, "TextLoader")
        assert hasattr(document, "LOADERS")
        assert hasattr(document, "load_documents")
        assert hasattr(document, "load_documents_sync")
        
        # Splitters
        assert hasattr(document, "BaseSplitter")
        assert hasattr(document, "RecursiveCharacterSplitter")
        assert hasattr(document, "CharacterSplitter")
        assert hasattr(document, "SentenceSplitter")
        assert hasattr(document, "MarkdownSplitter")
        assert hasattr(document, "HTMLSplitter")
        assert hasattr(document, "CodeSplitter")
        assert hasattr(document, "split_text")
    
    def test_backward_compat_from_retriever(self) -> None:
        """Test backward compatibility imports from retriever."""
        from agenticflow.retriever import (
            Document,
            DocumentLoader,
            TextChunk,
            RecursiveCharacterSplitter,
            split_text,
        )
        
        # These should work and be the same types
        assert Document is not None
        assert DocumentLoader is not None
        assert TextChunk is not None
        assert RecursiveCharacterSplitter is not None
        assert split_text is not None
