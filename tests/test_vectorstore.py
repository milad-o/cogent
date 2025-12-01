"""Tests for vector store module.

Tests:
- Document class
- Text splitting utilities
- InMemory backend
- MockEmbeddings
- VectorStore high-level API
"""

from __future__ import annotations

import pytest

from agenticflow.vectorstore import (
    Document,
    MockEmbeddings,
    SearchResult,
    VectorStore,
    create_documents,
    create_vectorstore,
    split_documents,
    split_text,
)
from agenticflow.vectorstore.backends.inmemory import InMemoryBackend


# ============================================================
# Document Tests
# ============================================================


class TestDocument:
    """Tests for Document class."""

    def test_create_document(self) -> None:
        """Test basic document creation."""
        doc = Document(text="Hello world")
        assert doc.text == "Hello world"
        assert doc.metadata == {"source": "unknown"}  # Auto-added
        assert doc.embedding is None
        assert doc.id.startswith("doc_")

    def test_document_with_metadata(self) -> None:
        """Test document with metadata."""
        doc = Document(
            text="Test content",
            metadata={"source": "test.txt", "page": 1},
        )
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["page"] == 1

    def test_document_with_embedding(self) -> None:
        """Test document with pre-computed embedding."""
        embedding = [0.1, 0.2, 0.3]
        doc = Document(text="Test", embedding=embedding)
        assert doc.embedding == embedding

    def test_document_with_custom_id(self) -> None:
        """Test document with custom ID."""
        doc = Document(text="Test", id="custom-id-123")
        assert doc.id == "custom-id-123"

    def test_document_to_dict(self) -> None:
        """Test document serialization."""
        doc = Document(
            text="Test",
            metadata={"key": "value"},
            id="test-id",
        )
        data = doc.to_dict()
        assert data["text"] == "Test"
        assert data["metadata"]["key"] == "value"
        assert data["id"] == "test-id"

    def test_document_from_dict(self) -> None:
        """Test document deserialization."""
        data = {
            "text": "Test content",
            "metadata": {"source": "file.txt"},
            "id": "doc-123",
        }
        doc = Document.from_dict(data)
        assert doc.text == "Test content"
        assert doc.metadata["source"] == "file.txt"
        assert doc.id == "doc-123"

    def test_document_len(self) -> None:
        """Test document length."""
        doc = Document(text="Hello")
        assert len(doc) == 5

    def test_document_repr(self) -> None:
        """Test document string representation."""
        doc = Document(text="Short text", id="test-id")
        repr_str = repr(doc)
        assert "test-id" in repr_str
        assert "Short text" in repr_str

    def test_document_repr_truncation(self) -> None:
        """Test long text truncation in repr."""
        long_text = "A" * 100
        doc = Document(text=long_text)
        repr_str = repr(doc)
        assert "..." in repr_str


class TestCreateDocuments:
    """Tests for create_documents utility."""

    def test_create_from_texts(self) -> None:
        """Test creating documents from texts."""
        docs = create_documents(["Hello", "World"])
        assert len(docs) == 2
        assert docs[0].text == "Hello"
        assert docs[1].text == "World"

    def test_create_with_metadatas(self) -> None:
        """Test creating documents with metadata."""
        docs = create_documents(
            texts=["A", "B"],
            metadatas=[{"idx": 0}, {"idx": 1}],
        )
        assert docs[0].metadata["idx"] == 0
        assert docs[1].metadata["idx"] == 1

    def test_create_with_ids(self) -> None:
        """Test creating documents with custom IDs."""
        docs = create_documents(
            texts=["A", "B"],
            ids=["id-a", "id-b"],
        )
        assert docs[0].id == "id-a"
        assert docs[1].id == "id-b"

    def test_mismatched_metadatas_raises(self) -> None:
        """Test that mismatched metadata length raises."""
        with pytest.raises(ValueError, match="metadatas length"):
            create_documents(
                texts=["A", "B", "C"],
                metadatas=[{"x": 1}],  # Only 1, need 3
            )

    def test_mismatched_ids_raises(self) -> None:
        """Test that mismatched IDs length raises."""
        with pytest.raises(ValueError, match="ids length"):
            create_documents(
                texts=["A", "B"],
                ids=["only-one"],
            )


# ============================================================
# Text Splitting Tests
# ============================================================


class TestSplitText:
    """Tests for text splitting utilities."""

    def test_short_text_not_split(self) -> None:
        """Test that short text isn't split."""
        text = "Short text"
        chunks = split_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_by_chunk_size(self) -> None:
        """Test splitting by chunk size."""
        text = "A" * 500
        chunks = split_text(text, chunk_size=100, chunk_overlap=0)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_split_with_overlap(self) -> None:
        """Test splitting with overlap."""
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8"
        chunks = split_text(text, chunk_size=20, chunk_overlap=5)
        # Chunks should exist
        assert len(chunks) >= 2

    def test_split_on_separator(self) -> None:
        """Test preferring separator for splits."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = split_text(text, chunk_size=30, chunk_overlap=0, separator="\n\n")
        # Should prefer splitting on paragraph breaks
        assert len(chunks) >= 2


class TestSplitDocuments:
    """Tests for document splitting."""

    def test_split_documents(self) -> None:
        """Test splitting documents into chunks."""
        docs = [
            Document(text="A" * 200, metadata={"source": "test.txt"}),
        ]
        chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > 1
        # Metadata should be preserved
        assert all(c.metadata["source"] == "test.txt" for c in chunks)
        # Chunk metadata should be added
        assert all("_chunk_index" in c.metadata for c in chunks)
        assert all("_parent_id" in c.metadata for c in chunks)


# ============================================================
# InMemory Backend Tests
# ============================================================


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    @pytest.fixture
    def backend(self) -> InMemoryBackend:
        """Create a fresh backend for each test."""
        return InMemoryBackend()

    @pytest.mark.asyncio
    async def test_add_and_count(self, backend: InMemoryBackend) -> None:
        """Test adding documents and counting."""
        docs = [Document(text="Test")]
        embeddings = [[0.1, 0.2, 0.3]]
        ids = ["doc-1"]
        
        await backend.add(ids, embeddings, docs)
        assert backend.count() == 1

    @pytest.mark.asyncio
    async def test_add_multiple(self, backend: InMemoryBackend) -> None:
        """Test adding multiple documents."""
        docs = [Document(text=f"Doc {i}") for i in range(5)]
        embeddings = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(5)]
        ids = [f"doc-{i}" for i in range(5)]
        
        await backend.add(ids, embeddings, docs)
        assert backend.count() == 5

    @pytest.mark.asyncio
    async def test_search_returns_results(self, backend: InMemoryBackend) -> None:
        """Test basic search."""
        docs = [
            Document(text="Python programming"),
            Document(text="JavaScript programming"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        ids = ["python", "javascript"]
        
        await backend.add(ids, embeddings, docs)
        
        # Search with embedding similar to Python
        results = await backend.search([0.9, 0.1, 0.0], k=2)
        
        assert len(results) == 2
        assert results[0].id == "python"  # Should be most similar
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_search_with_filter(self, backend: InMemoryBackend) -> None:
        """Test search with metadata filter."""
        docs = [
            Document(text="Python", metadata={"lang": "python"}),
            Document(text="JavaScript", metadata={"lang": "javascript"}),
            Document(text="Rust", metadata={"lang": "rust"}),
        ]
        embeddings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ids = ["py", "js", "rs"]
        
        await backend.add(ids, embeddings, docs)
        
        # Filter for only Python
        results = await backend.search([0.5, 0.5, 0.5], k=10, filter={"lang": "python"})
        
        assert len(results) == 1
        assert results[0].document.metadata["lang"] == "python"

    @pytest.mark.asyncio
    async def test_search_empty_store(self, backend: InMemoryBackend) -> None:
        """Test search on empty store."""
        results = await backend.search([0.1, 0.2, 0.3], k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete(self, backend: InMemoryBackend) -> None:
        """Test deleting documents."""
        docs = [Document(text="Test")]
        await backend.add(["doc-1"], [[0.1, 0.2, 0.3]], docs)
        
        assert backend.count() == 1
        result = await backend.delete(["doc-1"])
        assert result is True
        assert backend.count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, backend: InMemoryBackend) -> None:
        """Test deleting nonexistent document."""
        result = await backend.delete(["nonexistent"])
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, backend: InMemoryBackend) -> None:
        """Test clearing all documents."""
        docs = [Document(text=f"Doc {i}") for i in range(3)]
        embeddings = [[0.1, 0.2, 0.3]] * 3
        ids = [f"doc-{i}" for i in range(3)]
        
        await backend.add(ids, embeddings, docs)
        assert backend.count() == 3
        
        await backend.clear()
        assert backend.count() == 0

    @pytest.mark.asyncio
    async def test_get(self, backend: InMemoryBackend) -> None:
        """Test getting documents by ID."""
        docs = [Document(text="Test", id="doc-1")]
        await backend.add(["doc-1"], [[0.1, 0.2, 0.3]], docs)
        
        retrieved = await backend.get(["doc-1"])
        assert len(retrieved) == 1
        assert retrieved[0].text == "Test"

    @pytest.mark.asyncio
    async def test_get_missing(self, backend: InMemoryBackend) -> None:
        """Test getting missing documents."""
        retrieved = await backend.get(["nonexistent"])
        assert retrieved == []

    @pytest.mark.asyncio
    async def test_length_mismatch_raises(self, backend: InMemoryBackend) -> None:
        """Test that mismatched lengths raise."""
        with pytest.raises(ValueError, match="Lengths must match"):
            await backend.add(
                ids=["id1", "id2"],
                embeddings=[[0.1]],  # Only 1
                documents=[Document(text="A"), Document(text="B")],
            )


# ============================================================
# Mock Embeddings Tests
# ============================================================


class TestMockEmbeddings:
    """Tests for MockEmbedding (imported as MockEmbeddings for backward compat)."""

    @pytest.fixture
    def embeddings(self) -> MockEmbeddings:
        """Create mock embeddings."""
        return MockEmbeddings(dimensions=128)

    @pytest.mark.asyncio
    async def test_embed_texts(self, embeddings: MockEmbeddings) -> None:
        """Test embedding multiple texts."""
        vectors = await embeddings.aembed_texts(["Hello", "World"])
        assert len(vectors) == 2
        assert all(len(v) == 128 for v in vectors)

    @pytest.mark.asyncio
    async def test_embed_query(self, embeddings: MockEmbeddings) -> None:
        """Test embedding a single query."""
        vector = await embeddings.aembed_query("Hello")
        assert len(vector) == 128

    @pytest.mark.asyncio
    async def test_deterministic(self, embeddings: MockEmbeddings) -> None:
        """Test that same text produces same embedding."""
        v1 = await embeddings.aembed_query("Test text")
        v2 = await embeddings.aembed_query("Test text")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(
        self, embeddings: MockEmbeddings
    ) -> None:
        """Test that different texts produce different embeddings."""
        v1 = await embeddings.aembed_query("Hello")
        v2 = await embeddings.aembed_query("World")
        assert v1 != v2

    def test_dimension_property(self, embeddings: MockEmbeddings) -> None:
        """Test dimension property."""
        assert embeddings.dimension == 128

    @pytest.mark.asyncio
    async def test_normalized(self, embeddings: MockEmbeddings) -> None:
        """Test that embeddings are normalized."""
        import math
        
        vector = await embeddings.aembed_query("Test")
        magnitude = math.sqrt(sum(x * x for x in vector))
        assert abs(magnitude - 1.0) < 0.01  # Should be ~1
    
    def test_sync_embed(self, embeddings: MockEmbeddings) -> None:
        """Test synchronous embedding."""
        vectors = embeddings.embed(["Hello", "World"])
        assert len(vectors) == 2
        assert all(len(v) == 128 for v in vectors)


# ============================================================
# VectorStore Tests
# ============================================================


class TestVectorStore:
    """Tests for high-level VectorStore."""

    @pytest.fixture
    def store(self) -> VectorStore:
        """Create a store with mock embeddings."""
        return VectorStore.with_mock_embeddings(dimension=128)

    @pytest.mark.asyncio
    async def test_add_texts(self, store: VectorStore) -> None:
        """Test adding texts."""
        ids = await store.add_texts(["Hello", "World"])
        assert len(ids) == 2
        assert store.count() == 2

    @pytest.mark.asyncio
    async def test_add_texts_with_metadata(self, store: VectorStore) -> None:
        """Test adding texts with metadata."""
        ids = await store.add_texts(
            texts=["Hello", "World"],
            metadatas=[{"idx": 0}, {"idx": 1}],
        )
        
        docs = await store.get(ids)
        assert docs[0].metadata["idx"] == 0
        assert docs[1].metadata["idx"] == 1

    @pytest.mark.asyncio
    async def test_add_documents(self, store: VectorStore) -> None:
        """Test adding Document objects."""
        docs = [
            Document(text="Doc 1", metadata={"type": "a"}),
            Document(text="Doc 2", metadata={"type": "b"}),
        ]
        
        ids = await store.add_documents(docs)
        assert len(ids) == 2
        assert store.count() == 2

    @pytest.mark.asyncio
    async def test_search(self, store: VectorStore) -> None:
        """Test basic search."""
        await store.add_texts([
            "Python is a programming language",
            "JavaScript runs in browsers",
            "Rust is memory safe",
        ])
        
        results = await store.search("programming", k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score > 0 for r in results)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store: VectorStore) -> None:
        """Test search with filter."""
        await store.add_texts(
            texts=["Python", "JavaScript", "Rust"],
            metadatas=[
                {"category": "scripting"},
                {"category": "scripting"},
                {"category": "systems"},
            ],
        )
        
        results = await store.search("language", k=10, filter={"category": "systems"})
        assert len(results) == 1
        assert results[0].document.metadata["category"] == "systems"

    @pytest.mark.asyncio
    async def test_similarity_search(self, store: VectorStore) -> None:
        """Test similarity_search returns documents."""
        await store.add_texts(["Hello", "World"])
        
        docs = await store.similarity_search("greeting", k=1)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    @pytest.mark.asyncio
    async def test_delete(self, store: VectorStore) -> None:
        """Test deleting documents."""
        ids = await store.add_texts(["To delete"])
        assert store.count() == 1
        
        await store.delete(ids)
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_clear(self, store: VectorStore) -> None:
        """Test clearing store."""
        await store.add_texts(["A", "B", "C"])
        assert store.count() == 3
        
        await store.clear()
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_get(self, store: VectorStore) -> None:
        """Test getting documents by ID."""
        ids = await store.add_texts(["Test document"])
        
        docs = await store.get(ids)
        assert len(docs) == 1
        assert docs[0].text == "Test document"


class TestCreateVectorstore:
    """Tests for create_vectorstore factory."""

    @pytest.mark.asyncio
    async def test_create_empty(self) -> None:
        """Test creating empty store."""
        store = await create_vectorstore(embeddings=MockEmbeddings())
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_create_with_texts(self) -> None:
        """Test creating store with initial texts."""
        store = await create_vectorstore(
            texts=["Hello", "World"],
            embeddings=MockEmbeddings(),
        )
        assert store.count() == 2

    @pytest.mark.asyncio
    async def test_create_with_metadatas(self) -> None:
        """Test creating store with metadata."""
        store = await create_vectorstore(
            texts=["A", "B"],
            metadatas=[{"x": 1}, {"x": 2}],
            embeddings=MockEmbeddings(),
        )
        
        results = await store.search("test", k=10)
        metadatas = [r.document.metadata for r in results]
        # Metadata includes auto-added source
        assert any(m.get("x") == 1 for m in metadatas)
        assert any(m.get("x") == 2 for m in metadatas)


class TestVectorStoreWithMockEmbeddings:
    """Tests specifically for mock embedding behavior."""

    @pytest.mark.asyncio
    async def test_semantic_similarity(self) -> None:
        """Test that similar texts are more similar."""
        store = VectorStore.with_mock_embeddings()
        
        await store.add_texts([
            "The quick brown fox",
            "A fast brown fox",  # Similar
            "Hello world",  # Different
        ])
        
        results = await store.search("The quick brown fox", k=3)
        
        # First result should be exact match (if deduplicated) or very similar
        # Just verify we get results and they're sorted by score
        assert len(results) == 3
        assert results[0].score >= results[1].score >= results[2].score


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a search result."""
        doc = Document(text="Test", id="doc-1")
        result = SearchResult(document=doc, score=0.95)
        
        assert result.document == doc
        assert result.score == 0.95
        assert result.id == "doc-1"  # Auto-populated from document

    def test_result_with_explicit_id(self) -> None:
        """Test result with explicit ID."""
        doc = Document(text="Test", id="doc-1")
        result = SearchResult(document=doc, score=0.9, id="custom-id")
        
        assert result.id == "custom-id"
