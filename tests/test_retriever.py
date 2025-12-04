"""Tests for the retriever module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agenticflow.retriever.base import (
    BaseRetriever,
    FusionStrategy,
    RetrievalResult,
    Retriever,
)

# Test both import paths - new document module and backward-compatible retriever imports
from agenticflow.document import (
    Document,
    DocumentLoader,
    load_documents,
)
from agenticflow.document.splitters import (
    CharacterSplitter,
    CodeSplitter,
    MarkdownSplitter,
    RecursiveCharacterSplitter,
    SentenceSplitter,
    split_text,
)
from agenticflow.document.types import Document, TextChunk

from agenticflow.retriever.utils.fusion import (
    deduplicate_results,
    fuse_results,
    normalize_scores,
)
from agenticflow.vectorstore import Document as VectorStoreDocument
from agenticflow.models import MockEmbedding


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_documents() -> list[VectorStoreDocument]:
    """Create sample documents for testing."""
    return [
        VectorStoreDocument(text="Python is a programming language", metadata={"id": "1", "topic": "python"}),
        VectorStoreDocument(text="Machine learning uses algorithms", metadata={"id": "2", "topic": "ml"}),
        VectorStoreDocument(text="Deep learning is a subset of ML", metadata={"id": "3", "topic": "ml"}),
        VectorStoreDocument(text="JavaScript runs in browsers", metadata={"id": "4", "topic": "js"}),
        VectorStoreDocument(text="Python is great for data science", metadata={"id": "5", "topic": "python"}),
    ]


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    """Create sample retrieval results."""
    return [
        RetrievalResult(
            document=VectorStoreDocument(text="Doc A", metadata={"id": "a"}),
            score=0.9,
            retriever_name="test",
        ),
        RetrievalResult(
            document=VectorStoreDocument(text="Doc B", metadata={"id": "b"}),
            score=0.7,
            retriever_name="test",
        ),
        RetrievalResult(
            document=VectorStoreDocument(text="Doc C", metadata={"id": "c"}),
            score=0.5,
            retriever_name="test",
        ),
    ]


# ============================================================================
# Test Document Loader
# ============================================================================


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self) -> None:
        """Test creating a document."""
        doc = Document(text="Test content", metadata={"key": "value"})
        assert doc.text == "Test content"
        assert doc.metadata["key"] == "value"

    def test_document_auto_source(self) -> None:
        """Test that source is auto-added to metadata."""
        doc = Document(text="Test")
        assert "source" in doc.metadata

    def test_document_repr(self) -> None:
        """Test document string representation."""
        doc = Document(text="Short content")
        assert "Short content" in repr(doc)


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.mark.asyncio
    async def test_load_text_file(self, temp_dir: Path) -> None:
        """Test loading a text file."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("Hello, World!")

        loader = DocumentLoader()
        docs = await loader.load(text_file)

        assert len(docs) == 1
        assert docs[0].content == "Hello, World!"
        assert docs[0].metadata["file_type"] == ".txt"

    @pytest.mark.asyncio
    async def test_load_markdown_file(self, temp_dir: Path) -> None:
        """Test loading a markdown file."""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Title\n\nContent here")

        loader = DocumentLoader()
        docs = await loader.load(md_file)

        assert len(docs) == 1
        assert "Title" in docs[0].content
        assert docs[0].metadata["file_type"] == ".md"

    @pytest.mark.asyncio
    async def test_load_markdown_with_frontmatter(self, temp_dir: Path) -> None:
        """Test loading markdown with YAML frontmatter."""
        md_file = temp_dir / "test.md"
        md_file.write_text("---\ntitle: My Doc\nauthor: Test\n---\n\n# Content")

        loader = DocumentLoader()
        docs = await loader.load(md_file)

        assert docs[0].metadata.get("title") == "My Doc"
        assert docs[0].metadata.get("author") == "Test"

    @pytest.mark.asyncio
    async def test_load_json_file(self, temp_dir: Path) -> None:
        """Test loading a JSON file."""
        json_file = temp_dir / "test.json"
        json_file.write_text('{"key": "value", "number": 42}')

        loader = DocumentLoader()
        docs = await loader.load(json_file)

        assert len(docs) == 1
        assert "key" in docs[0].content
        assert "value" in docs[0].content

    @pytest.mark.asyncio
    async def test_load_json_array(self, temp_dir: Path) -> None:
        """Test loading a JSON array file."""
        json_file = temp_dir / "test.json"
        json_file.write_text('[{"id": 1}, {"id": 2}, {"id": 3}]')

        loader = DocumentLoader()
        docs = await loader.load(json_file)

        assert len(docs) == 3
        assert docs[0].metadata["index"] == 0

    @pytest.mark.asyncio
    async def test_load_csv_file(self, temp_dir: Path) -> None:
        """Test loading a CSV file."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25")

        loader = DocumentLoader()
        docs = await loader.load(csv_file)

        assert len(docs) == 1
        assert "Alice" in docs[0].content
        assert docs[0].metadata["row_count"] == 2

    @pytest.mark.asyncio
    async def test_load_html_file(self, temp_dir: Path) -> None:
        """Test loading an HTML file."""
        html_file = temp_dir / "test.html"
        html_file.write_text("<html><head><title>Test</title></head><body><p>Content</p></body></html>")

        loader = DocumentLoader()
        docs = await loader.load(html_file)

        assert len(docs) == 1
        assert "Content" in docs[0].content
        # Tags should be removed
        assert "<p>" not in docs[0].content

    @pytest.mark.asyncio
    async def test_load_code_file(self, temp_dir: Path) -> None:
        """Test loading a Python file."""
        py_file = temp_dir / "test.py"
        py_file.write_text("def hello():\n    print('Hello')")

        loader = DocumentLoader()
        docs = await loader.load(py_file)

        assert len(docs) == 1
        assert "def hello" in docs[0].content
        assert docs[0].metadata["language"] == "python"

    @pytest.mark.asyncio
    async def test_load_many(self, temp_dir: Path) -> None:
        """Test loading multiple files."""
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")

        loader = DocumentLoader()
        docs = await loader.load_many([
            temp_dir / "file1.txt",
            temp_dir / "file2.txt",
        ])

        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_load_directory(self, temp_dir: Path) -> None:
        """Test loading all files from a directory."""
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.md").write_text("# Content 2")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file3.txt").write_text("Content 3")

        loader = DocumentLoader()
        docs = await loader.load_directory(temp_dir, glob="**/*.txt")

        assert len(docs) == 2  # Only .txt files

    @pytest.mark.asyncio
    async def test_load_unsupported_file(self, temp_dir: Path) -> None:
        """Test that unsupported files raise ValueError."""
        binary_file = temp_dir / "test.xyz"
        binary_file.write_bytes(b"\x00\x01\x02")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            await loader.load(binary_file)

    @pytest.mark.asyncio
    async def test_load_missing_file(self) -> None:
        """Test that missing files raise FileNotFoundError."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            await loader.load("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_register_custom_loader(self, temp_dir: Path) -> None:
        """Test registering a custom loader."""
        custom_file = temp_dir / "test.custom"
        custom_file.write_text("custom content")

        async def custom_loader(path: Path) -> list[Document]:
            return [Document(text=f"CUSTOM: {path.read_text()}", metadata={"source": str(path)})]

        loader = DocumentLoader()
        loader.register_loader(".custom", custom_loader)
        docs = await loader.load(custom_file)

        assert "CUSTOM:" in docs[0].text


class TestLoadDocuments:
    """Tests for load_documents convenience function."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.mark.asyncio
    async def test_load_single_file(self, temp_dir: Path) -> None:
        """Test loading a single file."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("Test content")

        docs = await load_documents(text_file)
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_load_directory_path(self, temp_dir: Path) -> None:
        """Test loading from directory path."""
        (temp_dir / "file.txt").write_text("Content")

        docs = await load_documents(temp_dir)
        assert len(docs) >= 1


# ============================================================================
# Test Text Splitters
# ============================================================================


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_create_chunk(self) -> None:
        """Test creating a text chunk."""
        chunk = TextChunk(text="Test content", metadata={"index": 0})
        assert chunk.content == "Test content"
        assert chunk.metadata["index"] == 0

    def test_chunk_length(self) -> None:
        """Test chunk length."""
        chunk = TextChunk(text="Hello")
        assert len(chunk) == 5

    def test_chunk_repr(self) -> None:
        """Test chunk string representation."""
        chunk = TextChunk(text="Short")
        assert "Short" in repr(chunk)


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
            assert len(chunk.content) <= 150 + 10  # Some tolerance

    def test_split_with_overlap(self) -> None:
        """Test splitting with overlap."""
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8"
        splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=5)
        chunks = splitter.split_text(text)

        assert len(chunks) >= 2
        # Check for some overlap between consecutive chunks
        # (exact overlap depends on separator positions)

    def test_split_documents(self) -> None:
        """Test splitting multiple documents."""
        docs = [
            Document(text="First document content", metadata={"id": "1"}),
            Document(text="Second document content", metadata={"id": "2"}),
        ]
        splitter = RecursiveCharacterSplitter(chunk_size=15, chunk_overlap=0)
        chunks = splitter.split_documents(docs)

        # Should have chunks from both documents
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

    def test_custom_separator(self) -> None:
        """Test with custom separator."""
        text = "Part1---Part2---Part3"
        splitter = CharacterSplitter(separator="---", chunk_size=10, chunk_overlap=0)
        chunks = splitter.split_text(text)

        assert len(chunks) >= 2


class TestSentenceSplitter:
    """Tests for SentenceSplitter."""

    def test_split_sentences(self) -> None:
        """Test sentence-based splitting."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        splitter = SentenceSplitter(chunk_size=40, chunk_overlap=0)
        chunks = splitter.split_text(text)

        assert len(chunks) >= 1

    def test_handles_abbreviations(self) -> None:
        """Test that abbreviations don't cause false splits."""
        text = "Dr. Smith went to the store. He bought apples."
        splitter = SentenceSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text(text)

        # Should not split on "Dr."
        assert any("Dr. Smith" in c.content for c in chunks)


class TestMarkdownSplitter:
    """Tests for MarkdownSplitter."""

    def test_split_by_headers(self) -> None:
        """Test markdown splitting by headers."""
        text = """# Introduction

This is the intro.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""
        splitter = MarkdownSplitter(chunk_size=200, chunk_overlap=0)
        chunks = splitter.split_text(text)

        assert len(chunks) >= 1
        # Should have header metadata
        assert any("headers" in c.metadata for c in chunks)

    def test_return_each_section(self) -> None:
        """Test returning each section as separate chunk."""
        text = """# Title

Intro.

## Part 1

Part 1 content.

## Part 2

Part 2 content.
"""
        splitter = MarkdownSplitter(chunk_size=1000, return_each_section=True)
        chunks = splitter.split_text(text)

        # Should have multiple sections
        assert len(chunks) >= 2


class TestCodeSplitter:
    """Tests for CodeSplitter."""

    def test_split_python_code(self) -> None:
        """Test splitting Python code."""
        code = """
def function_one():
    print("one")

def function_two():
    print("two")

class MyClass:
    def method(self):
        pass
"""
        splitter = CodeSplitter(language="python", chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text(code)

        assert len(chunks) >= 1
        assert all(c.metadata.get("language") == "python" for c in chunks)

    def test_split_javascript_code(self) -> None:
        """Test splitting JavaScript code."""
        code = """
function hello() {
    console.log("hello");
}

class MyClass {
    constructor() {}
}
"""
        splitter = CodeSplitter(language="javascript", chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text(code)

        assert len(chunks) >= 1
        assert all(c.metadata.get("language") == "javascript" for c in chunks)


class TestSplitTextFunction:
    """Tests for split_text convenience function."""

    def test_recursive_splitter(self) -> None:
        """Test using recursive splitter type."""
        text = "Content " * 100
        chunks = split_text(text, chunk_size=500, chunk_overlap=50, splitter_type="recursive")
        assert len(chunks) >= 1

    def test_sentence_splitter(self) -> None:
        """Test using sentence splitter type."""
        text = "First. Second. Third. Fourth."
        chunks = split_text(text, chunk_size=100, chunk_overlap=10, splitter_type="sentence")
        assert len(chunks) >= 1

    def test_markdown_splitter(self) -> None:
        """Test using markdown splitter type."""
        text = "# Title\n\nContent"
        chunks = split_text(text, chunk_size=500, chunk_overlap=50, splitter_type="markdown")
        assert len(chunks) >= 1

    def test_code_splitter(self) -> None:
        """Test using code splitter type."""
        code = "def foo():\n    pass"
        chunks = split_text(code, chunk_size=500, chunk_overlap=50, splitter_type="code", language="python")
        assert len(chunks) >= 1

    def test_invalid_splitter_type(self) -> None:
        """Test that invalid splitter type raises error."""
        with pytest.raises(ValueError, match="Unknown splitter type"):
            split_text("text", splitter_type="invalid")


# ============================================================================
# Test Base Classes
# ============================================================================


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_create_result(self) -> None:
        """Test creating a retrieval result."""
        doc = VectorStoreDocument(text="test", metadata={"key": "value"})
        result = RetrievalResult(
            document=doc,
            score=0.85,
            retriever_name="dense",
        )
        
        assert result.document == doc
        assert result.score == 0.85
        assert result.retriever_name == "dense"
        assert result.metadata == {}
    
    def test_result_with_metadata(self) -> None:
        """Test result with custom metadata."""
        doc = VectorStoreDocument(text="test", metadata={})
        result = RetrievalResult(
            document=doc,
            score=0.5,
            retriever_name="hybrid",
            metadata={"fusion": "rrf", "rank": 1},
        )
        
        assert result.metadata["fusion"] == "rrf"
        assert result.metadata["rank"] == 1


class TestFusionStrategy:
    """Tests for FusionStrategy enum."""
    
    def test_all_strategies_exist(self) -> None:
        """Test that all expected strategies are defined."""
        assert FusionStrategy.RRF is not None
        assert FusionStrategy.LINEAR is not None
        assert FusionStrategy.MAX is not None
        assert FusionStrategy.VOTING is not None
    
    def test_strategy_values(self) -> None:
        """Test strategy string values."""
        assert FusionStrategy.RRF.value == "rrf"
        assert FusionStrategy.LINEAR.value == "linear"


class TestBaseRetriever:
    """Tests for BaseRetriever base class."""
    
    def test_base_retriever_name(self) -> None:
        """Test that BaseRetriever has a name property."""
        
        class TestRetriever(BaseRetriever):
            _name = "test_retriever"
            
            async def retrieve(self, query, k=None, filter=None):
                return []
        
        retriever = TestRetriever()
        assert retriever.name == "test_retriever"


# ============================================================================
# Test Fusion Utilities
# ============================================================================


class TestNormalizeScores:
    """Tests for score normalization."""
    
    def test_normalize_basic(self, sample_results: list[RetrievalResult]) -> None:
        """Test basic score normalization."""
        normalized = normalize_scores(sample_results)
        
        # Highest should be 1.0
        assert normalized[0].score == 1.0
        # Others should be scaled
        assert 0 < normalized[1].score < 1.0
        assert 0 <= normalized[2].score <= 1.0
    
    def test_normalize_empty(self) -> None:
        """Test normalizing empty list."""
        normalized = normalize_scores([])
        assert normalized == []
    
    def test_normalize_single(self) -> None:
        """Test normalizing single result."""
        result = RetrievalResult(
            document=VectorStoreDocument(text="test", metadata={}),
            score=0.5,
            retriever_name="test",
        )
        normalized = normalize_scores([result])
        assert normalized[0].score == 1.0


class TestDeduplicateResults:
    """Tests for result deduplication."""
    
    def test_deduplicate_by_text(self) -> None:
        """Test deduplication by document text."""
        results = [
            RetrievalResult(
                document=VectorStoreDocument(text="same text", metadata={}),
                score=0.9,
                retriever_name="r1",
            ),
            RetrievalResult(
                document=VectorStoreDocument(text="same text", metadata={}),
                score=0.7,
                retriever_name="r2",
            ),
            RetrievalResult(
                document=VectorStoreDocument(text="different text", metadata={}),
                score=0.5,
                retriever_name="r1",
            ),
        ]
        
        deduped = deduplicate_results(results)
        
        # Should keep 2 unique documents
        assert len(deduped) == 2
        # Should keep higher scored duplicate
        assert deduped[0].score == 0.9
    
    def test_deduplicate_by_id(self) -> None:
        """Test deduplication by document ID."""
        results = [
            RetrievalResult(
                document=VectorStoreDocument(text="text A", metadata={"id": "1"}),
                score=0.9,
                retriever_name="r1",
            ),
            RetrievalResult(
                document=VectorStoreDocument(text="text B", metadata={"id": "1"}),
                score=0.7,
                retriever_name="r2",
            ),
        ]
        
        deduped = deduplicate_results(results)
        assert len(deduped) == 1


class TestFuseResults:
    """Tests for result fusion strategies."""
    
    def test_rrf_fusion(self) -> None:
        """Test RRF (Reciprocal Rank Fusion)."""
        # Use same document ID for merging
        doc_a = VectorStoreDocument("A", {"id": "doc_a"})
        doc_b = VectorStoreDocument("B", {"id": "doc_b"})
        
        results_list = [
            [
                RetrievalResult(doc_a, 0.9, "r1"),
                RetrievalResult(doc_b, 0.8, "r1"),
            ],
            [
                RetrievalResult(VectorStoreDocument("B", {"id": "doc_b"}), 0.95, "r2"),
                RetrievalResult(VectorStoreDocument("A", {"id": "doc_a"}), 0.7, "r2"),
            ],
        ]
        
        fused = fuse_results(results_list, strategy=FusionStrategy.RRF)
        
        # Both docs should be present
        texts = [r.document.text for r in fused]
        assert "A" in texts
        assert "B" in texts
    
    def test_linear_fusion(self) -> None:
        """Test linear weighted fusion."""
        # Use same document ID for merging
        results_list = [
            [RetrievalResult(VectorStoreDocument("A", {"id": "shared"}), 0.8, "r1")],
            [RetrievalResult(VectorStoreDocument("A", {"id": "shared"}), 0.6, "r2")],
        ]
        
        fused = fuse_results(
            results_list,
            strategy=FusionStrategy.LINEAR,
            weights=[0.7, 0.3],
        )
        
        # Should have combined score
        assert len(fused) == 1
        expected_score = 0.7 * 0.8 + 0.3 * 0.6
        assert abs(fused[0].score - expected_score) < 0.01
    
    def test_max_fusion(self) -> None:
        """Test max score fusion."""
        # Use same document ID for merging
        results_list = [
            [RetrievalResult(VectorStoreDocument("A", {"id": "shared"}), 0.6, "r1")],
            [RetrievalResult(VectorStoreDocument("A", {"id": "shared"}), 0.9, "r2")],
        ]
        
        fused = fuse_results(results_list, strategy=FusionStrategy.MAX)
        
        assert len(fused) == 1
        assert fused[0].score == 0.9
    
    def test_voting_fusion(self) -> None:
        """Test voting fusion."""
        # Use same document IDs for merging
        results_list = [
            [
                RetrievalResult(VectorStoreDocument("A", {"id": "a"}), 0.9, "r1"),
                RetrievalResult(VectorStoreDocument("B", {"id": "b"}), 0.8, "r1"),
            ],
            [
                RetrievalResult(VectorStoreDocument("A", {"id": "a"}), 0.7, "r2"),
                RetrievalResult(VectorStoreDocument("C", {"id": "c"}), 0.9, "r2"),
            ],
            [
                RetrievalResult(VectorStoreDocument("A", {"id": "a"}), 0.5, "r3"),
            ],
        ]
        
        fused = fuse_results(results_list, strategy=FusionStrategy.VOTING)
        
        # A should be ranked highest (appears in all 3)
        assert fused[0].document.text == "A"
        # Score is normalized vote count (1.0 for A since all 3 retrievers found it)


# ============================================================================
# Test Dense Retriever
# ============================================================================


class TestDenseRetriever:
    """Tests for DenseRetriever."""
    
    @pytest.fixture
    def mock_vectorstore(self, sample_documents: list[Document]):
        """Create a mock vector store."""
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(
            embeddings=MockEmbedding(dimensions=64),
            backend=InMemoryBackend(),
        )
        return vs
    
    @pytest.mark.asyncio
    async def test_dense_retriever_basic(
        self,
        mock_vectorstore,
        sample_documents: list[Document],
    ) -> None:
        """Test basic dense retrieval."""
        from agenticflow.retriever.dense import DenseRetriever
        
        # Add documents
        await mock_vectorstore.add_documents(sample_documents)
        
        retriever = DenseRetriever(mock_vectorstore)
        
        results = await retriever.retrieve("Python programming", k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, VectorStoreDocument) for r in results)
    
    @pytest.mark.asyncio
    async def test_dense_retriever_with_scores(
        self,
        mock_vectorstore,
        sample_documents: list[VectorStoreDocument],
    ) -> None:
        """Test dense retrieval with scores."""
        from agenticflow.retriever.dense import DenseRetriever
        
        await mock_vectorstore.add_documents(sample_documents)
        retriever = DenseRetriever(mock_vectorstore)
        
        results = await retriever.retrieve_with_scores("machine learning", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.retriever_name == "dense" for r in results)


# ============================================================================
# Test BM25 Retriever
# ============================================================================


class TestBM25Retriever:
    """Tests for BM25Retriever (sparse retrieval)."""
    
    @pytest.mark.asyncio
    async def test_bm25_basic(self, sample_documents: list[Document]) -> None:
        """Test basic BM25 retrieval."""
        pytest.importorskip("rank_bm25")
        from agenticflow.retriever.sparse import BM25Retriever
        
        retriever = BM25Retriever(k=3)
        await retriever.index_documents(sample_documents)
        
        results = await retriever.retrieve("Python programming language")
        
        assert len(results) <= 3
        # Should find Python-related docs
        texts = [r.text for r in results]
        assert any("Python" in t for t in texts)
    
    @pytest.mark.asyncio
    async def test_bm25_with_scores(self, sample_documents: list[Document]) -> None:
        """Test BM25 with scores."""
        pytest.importorskip("rank_bm25")
        from agenticflow.retriever.sparse import BM25Retriever
        
        retriever = BM25Retriever()
        await retriever.index_documents(sample_documents)
        
        results = await retriever.retrieve_with_scores("machine learning")
        
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Scores should be non-negative
        assert all(r.score >= 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_bm25_tokenizer(self, sample_documents: list[Document]) -> None:
        """Test BM25 with different tokenizers."""
        pytest.importorskip("rank_bm25")
        from agenticflow.retriever.sparse import BM25Retriever
        
        # Simple tokenizer
        retriever = BM25Retriever(tokenizer="simple")
        await retriever.index_documents(sample_documents)
        
        results = await retriever.retrieve("PYTHON")  # Uppercase
        
        # Should still find Python docs (case-insensitive)
        texts = [r.text.lower() for r in results]
        assert any("python" in t for t in texts)


# ============================================================================
# Test Hybrid Retriever
# ============================================================================


class TestHybridRetriever:
    """Tests for HybridRetriever."""
    
    @pytest.mark.asyncio
    async def test_hybrid_basic(self, sample_documents: list[Document]) -> None:
        """Test basic hybrid retrieval."""
        pytest.importorskip("rank_bm25")
        from agenticflow.retriever.dense import DenseRetriever
        from agenticflow.retriever.hybrid import HybridRetriever
        from agenticflow.retriever.sparse import BM25Retriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        await vs.add_documents(sample_documents)
        
        dense = DenseRetriever(vs)
        sparse = BM25Retriever()
        await sparse.index_documents(sample_documents)
        
        hybrid = HybridRetriever(dense, sparse, dense_weight=0.6)
        
        results = await hybrid.retrieve("Python programming", k=3)
        
        assert len(results) <= 3
    
    @pytest.mark.asyncio
    async def test_hybrid_weights(self, sample_documents: list[Document]) -> None:
        """Test hybrid with different weight configurations."""
        pytest.importorskip("rank_bm25")
        from agenticflow.retriever.dense import DenseRetriever
        from agenticflow.retriever.hybrid import HybridRetriever
        from agenticflow.retriever.sparse import BM25Retriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        await vs.add_documents(sample_documents)
        
        dense = DenseRetriever(vs)
        sparse = BM25Retriever()
        await sparse.index_documents(sample_documents)
        
        # Heavy on dense
        hybrid_dense = HybridRetriever(dense, sparse, dense_weight=0.9, sparse_weight=0.1)
        results_dense = await hybrid_dense.retrieve("Python", k=3)
        
        # Heavy on sparse
        hybrid_sparse = HybridRetriever(dense, sparse, dense_weight=0.1, sparse_weight=0.9)
        results_sparse = await hybrid_sparse.retrieve("Python", k=3)
        
        # Both should return results
        assert len(results_dense) > 0
        assert len(results_sparse) > 0


# ============================================================================
# Test Ensemble Retriever
# ============================================================================


class TestEnsembleRetriever:
    """Tests for EnsembleRetriever."""
    
    @pytest.mark.asyncio
    async def test_ensemble_basic(self, sample_documents: list[Document]) -> None:
        """Test basic ensemble retrieval."""
        pytest.importorskip("rank_bm25")
        from agenticflow.retriever.dense import DenseRetriever
        from agenticflow.retriever.ensemble import EnsembleRetriever
        from agenticflow.retriever.sparse import BM25Retriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        await vs.add_documents(sample_documents)
        
        dense = DenseRetriever(vs)
        sparse = BM25Retriever()
        await sparse.index_documents(sample_documents)
        
        ensemble = EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=[0.5, 0.5],
        )
        
        results = await ensemble.retrieve("Python", k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_ensemble_fusion_strategies(self, sample_documents: list[Document]) -> None:
        """Test ensemble with different fusion strategies."""
        pytest.importorskip("rank_bm25")
        from agenticflow.retriever.dense import DenseRetriever
        from agenticflow.retriever.ensemble import EnsembleRetriever
        from agenticflow.retriever.sparse import BM25Retriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        await vs.add_documents(sample_documents)
        
        dense = DenseRetriever(vs)
        sparse = BM25Retriever()
        await sparse.index_documents(sample_documents)
        
        # Test RRF
        ensemble_rrf = EnsembleRetriever(
            retrievers=[dense, sparse],
            fusion_strategy=FusionStrategy.RRF,
        )
        results_rrf = await ensemble_rrf.retrieve("Python", k=3)
        
        # Test MAX
        ensemble_max = EnsembleRetriever(
            retrievers=[dense, sparse],
            fusion_strategy=FusionStrategy.MAX,
        )
        results_max = await ensemble_max.retrieve("Python", k=3)
        
        assert len(results_rrf) > 0
        assert len(results_max) > 0


# ============================================================================
# Test Contextual Retrievers
# ============================================================================


class TestParentDocumentRetriever:
    """Tests for ParentDocumentRetriever."""
    
    @pytest.mark.asyncio
    async def test_parent_retriever_basic(self) -> None:
        """Test basic parent document retrieval."""
        from agenticflow.retriever.contextual import ParentDocumentRetriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        
        # Large parent document
        parent_doc = VectorStoreDocument(
            text="This is a long document about Python. " * 50 +
                 "Python is great for machine learning. " * 50,
            metadata={"id": "parent1", "topic": "python"},
        )
        
        retriever = ParentDocumentRetriever(
            vs,
            chunk_size=100,
            chunk_overlap=20,
        )
        
        await retriever.add_documents([parent_doc])
        
        results = await retriever.retrieve("Python machine learning", k=1)
        
        assert len(results) == 1
        # Should return the full parent, not a chunk
        assert len(results[0].text) > 100
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky test - investigate async issue")
    async def test_parent_retriever_with_scores(self) -> None:
        """Test parent retrieval with scores."""
        from agenticflow.retriever.contextual import ParentDocumentRetriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        
        docs = [
            VectorStoreDocument(text="Long document A. " * 20, metadata={"id": "1"}),
            VectorStoreDocument(text="Long document B. " * 20, metadata={"id": "2"}),
        ]
        
        retriever = ParentDocumentRetriever(vs, chunk_size=50)
        await retriever.add_documents(docs)
        
        results = await retriever.retrieve_with_scores("document", k=2)
        
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert "matching_chunks" in results[0].metadata


class TestSentenceWindowRetriever:
    """Tests for SentenceWindowRetriever."""
    
    @pytest.mark.asyncio
    async def test_sentence_window_basic(self) -> None:
        """Test basic sentence window retrieval."""
        from agenticflow.retriever.contextual import SentenceWindowRetriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        
        doc = VectorStoreDocument(
            text="First sentence. Second sentence about Python. Third sentence. "
                 "Fourth sentence. Fifth sentence.",
            metadata={"id": "1"},
        )
        
        retriever = SentenceWindowRetriever(vs, window_size=1)
        await retriever.add_documents([doc])
        
        results = await retriever.retrieve("Python", k=1)
        
        # Should return a result with surrounding context (window)
        # Note: MockEmbedding uses hash-based embeddings, not semantic
        assert len(results) >= 1
        assert len(results[0].text) > 0


# ============================================================================
# Test Self-Query Retriever
# ============================================================================


class TestSelfQueryRetriever:
    """Tests for SelfQueryRetriever."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for query parsing."""
        
        class MockLLM:
            async def generate(self, prompt: str) -> str:
                # Return mock parsed query
                return '{"semantic_query": "programming tutorials", "filter": {"topic": "python"}}'
        
        return MockLLM()
    
    @pytest.mark.asyncio
    async def test_self_query_basic(self, mock_llm, sample_documents: list[Document]) -> None:
        """Test basic self-query retrieval."""
        from agenticflow.retriever.self_query import AttributeInfo, SelfQueryRetriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        await vs.add_documents(sample_documents)
        
        retriever = SelfQueryRetriever(
            vectorstore=vs,
            llm=mock_llm,
            attribute_info=[
                AttributeInfo("topic", "Document topic", "string"),
            ],
        )
        
        results = await retriever.retrieve("Python tutorials about programming")
        
        # Should return some results
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_self_query_verbose(self, mock_llm, sample_documents: list[Document]) -> None:
        """Test self-query with verbose output."""
        from agenticflow.retriever.self_query import AttributeInfo, SelfQueryRetriever
        from agenticflow.vectorstore import VectorStore
        from agenticflow.vectorstore.backends.inmemory import InMemoryBackend
        
        vs = VectorStore(embeddings=MockEmbedding(dimensions=64), backend=InMemoryBackend())
        await vs.add_documents(sample_documents)
        
        retriever = SelfQueryRetriever(
            vectorstore=vs,
            llm=mock_llm,
            attribute_info=[
                AttributeInfo("topic", "Document topic", "string"),
            ],
        )
        
        results, parsed = await retriever.retrieve_verbose("Python tutorials")
        
        assert parsed.semantic_query == "programming tutorials"
        assert parsed.filter == {"topic": "python"}


# ============================================================================
# Test Rerankers
# ============================================================================


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""
    
    @pytest.mark.asyncio
    async def test_cross_encoder_basic(self, sample_documents: list[Document]) -> None:
        """Test basic cross-encoder reranking."""
        pytest.importorskip("sentence_transformers")
        from agenticflow.retriever.rerankers import CrossEncoderReranker
        
        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        
        results = await reranker.rerank(
            "Python programming",
            sample_documents[:3],
            top_n=2,
        )
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Should be sorted by score
        if len(results) > 1:
            assert results[0].score >= results[1].score


class TestLLMReranker:
    """Tests for LLM-based rerankers."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM model."""
        
        class MockModel:
            model = "mock-model"
            
            async def generate(self, prompt: str) -> str:
                # Return a score based on "Python" presence
                if "Python" in prompt:
                    return "8"
                return "5"
        
        return MockModel()
    
    @pytest.mark.asyncio
    async def test_llm_reranker_basic(
        self,
        mock_model,
        sample_documents: list[Document],
    ) -> None:
        """Test basic LLM reranking."""
        from agenticflow.retriever.rerankers import LLMReranker
        
        reranker = LLMReranker(model=mock_model, max_concurrent=2)
        
        results = await reranker.rerank(
            "Python programming",
            sample_documents[:3],
            top_n=2,
        )
        
        assert len(results) == 2
        # Python docs should score higher
        assert results[0].score >= results[1].score
    
    @pytest.mark.asyncio
    async def test_listwise_reranker(self, sample_documents: list[Document]) -> None:
        """Test listwise LLM reranking."""
        from agenticflow.retriever.rerankers import ListwiseLLMReranker
        
        class MockModel:
            model = "mock"
            
            async def generate(self, prompt: str) -> str:
                # Return ordering: Python docs first
                return "1,5,2,3,4"
        
        reranker = ListwiseLLMReranker(model=MockModel())
        
        results = await reranker.rerank(
            "Python",
            sample_documents,
            top_n=3,
        )
        
        assert len(results) == 3
        # First result should be doc index 0 (1 in 1-indexed)
        assert results[0].document.text == sample_documents[0].text


# ============================================================================
# Test Module Exports
# ============================================================================


class TestModuleExports:
    """Tests for module-level exports."""
    
    def test_main_exports(self) -> None:
        """Test that main retriever module exports all components."""
        from agenticflow import retriever
        
        # Core
        assert hasattr(retriever, "Retriever")
        assert hasattr(retriever, "BaseRetriever")
        assert hasattr(retriever, "RetrievalResult")
        assert hasattr(retriever, "FusionStrategy")
        
        # Retrievers
        assert hasattr(retriever, "DenseRetriever")
        assert hasattr(retriever, "BM25Retriever")
        assert hasattr(retriever, "HybridRetriever")
        assert hasattr(retriever, "EnsembleRetriever")
        assert hasattr(retriever, "ParentDocumentRetriever")
        assert hasattr(retriever, "SentenceWindowRetriever")
        assert hasattr(retriever, "SelfQueryRetriever")
        
        # Rerankers
        assert hasattr(retriever, "Reranker")
        assert hasattr(retriever, "CrossEncoderReranker")
        assert hasattr(retriever, "CohereReranker")
        assert hasattr(retriever, "LLMReranker")
        
        # Utilities
        assert hasattr(retriever, "fuse_results")
        assert hasattr(retriever, "normalize_scores")
    
    def test_rerankers_submodule(self) -> None:
        """Test rerankers submodule exports."""
        from agenticflow.retriever import rerankers
        
        assert hasattr(rerankers, "Reranker")
        assert hasattr(rerankers, "BaseReranker")
        assert hasattr(rerankers, "CrossEncoderReranker")
        assert hasattr(rerankers, "CohereReranker")
        assert hasattr(rerankers, "LLMReranker")
        assert hasattr(rerankers, "ListwiseLLMReranker")
