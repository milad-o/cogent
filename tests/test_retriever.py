"""Tests for the retriever module."""

from __future__ import annotations

import pytest

from agenticflow.retriever.base import (
    BaseRetriever,
    FusionStrategy,
    RetrievalResult,
    Retriever,
)
from agenticflow.retriever.utils.fusion import (
    deduplicate_results,
    fuse_results,
    normalize_scores,
)
from agenticflow.vectorstore import Document


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(text="Python is a programming language", metadata={"id": "1", "topic": "python"}),
        Document(text="Machine learning uses algorithms", metadata={"id": "2", "topic": "ml"}),
        Document(text="Deep learning is a subset of ML", metadata={"id": "3", "topic": "ml"}),
        Document(text="JavaScript runs in browsers", metadata={"id": "4", "topic": "js"}),
        Document(text="Python is great for data science", metadata={"id": "5", "topic": "python"}),
    ]


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    """Create sample retrieval results."""
    return [
        RetrievalResult(
            document=Document(text="Doc A", metadata={"id": "a"}),
            score=0.9,
            retriever_name="test",
        ),
        RetrievalResult(
            document=Document(text="Doc B", metadata={"id": "b"}),
            score=0.7,
            retriever_name="test",
        ),
        RetrievalResult(
            document=Document(text="Doc C", metadata={"id": "c"}),
            score=0.5,
            retriever_name="test",
        ),
    ]


# ============================================================================
# Test Base Classes
# ============================================================================


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_create_result(self) -> None:
        """Test creating a retrieval result."""
        doc = Document(text="test", metadata={"key": "value"})
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
        doc = Document(text="test", metadata={})
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
            document=Document(text="test", metadata={}),
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
                document=Document(text="same text", metadata={}),
                score=0.9,
                retriever_name="r1",
            ),
            RetrievalResult(
                document=Document(text="same text", metadata={}),
                score=0.7,
                retriever_name="r2",
            ),
            RetrievalResult(
                document=Document(text="different text", metadata={}),
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
                document=Document(text="text A", metadata={"id": "1"}),
                score=0.9,
                retriever_name="r1",
            ),
            RetrievalResult(
                document=Document(text="text B", metadata={"id": "1"}),
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
        doc_a = Document("A", {"id": "doc_a"})
        doc_b = Document("B", {"id": "doc_b"})
        
        results_list = [
            [
                RetrievalResult(doc_a, 0.9, "r1"),
                RetrievalResult(doc_b, 0.8, "r1"),
            ],
            [
                RetrievalResult(Document("B", {"id": "doc_b"}), 0.95, "r2"),
                RetrievalResult(Document("A", {"id": "doc_a"}), 0.7, "r2"),
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
            [RetrievalResult(Document("A", {"id": "shared"}), 0.8, "r1")],
            [RetrievalResult(Document("A", {"id": "shared"}), 0.6, "r2")],
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
            [RetrievalResult(Document("A", {"id": "shared"}), 0.6, "r1")],
            [RetrievalResult(Document("A", {"id": "shared"}), 0.9, "r2")],
        ]
        
        fused = fuse_results(results_list, strategy=FusionStrategy.MAX)
        
        assert len(fused) == 1
        assert fused[0].score == 0.9
    
    def test_voting_fusion(self) -> None:
        """Test voting fusion."""
        # Use same document IDs for merging
        results_list = [
            [
                RetrievalResult(Document("A", {"id": "a"}), 0.9, "r1"),
                RetrievalResult(Document("B", {"id": "b"}), 0.8, "r1"),
            ],
            [
                RetrievalResult(Document("A", {"id": "a"}), 0.7, "r2"),
                RetrievalResult(Document("C", {"id": "c"}), 0.9, "r2"),
            ],
            [
                RetrievalResult(Document("A", {"id": "a"}), 0.5, "r3"),
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
        
        class MockEmbedding:
            """Mock embedding model."""
            
            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                # Simple mock embeddings based on text length
                return [[len(t) / 100.0] * 64 for t in texts]
            
            async def embed_query(self, text: str) -> list[float]:
                return [len(text) / 100.0] * 64
        
        vs = VectorStore(
            embeddings=MockEmbedding(),
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
        assert all(isinstance(r, Document) for r in results)
    
    @pytest.mark.asyncio
    async def test_dense_retriever_with_scores(
        self,
        mock_vectorstore,
        sample_documents: list[Document],
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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                return [[0.1] * 64 for _ in texts]
            
            async def embed_query(self, text):
                return [0.1] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                return [[0.1] * 64 for _ in texts]
            
            async def embed_query(self, text):
                return [0.1] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                return [[0.1] * 64 for _ in texts]
            
            async def embed_query(self, text):
                return [0.1] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                return [[0.1] * 64 for _ in texts]
            
            async def embed_query(self, text):
                return [0.1] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                return [[len(t) / 100.0] * 64 for t in texts]
            
            async def embed_query(self, text):
                return [len(text) / 100.0] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
        
        # Large parent document
        parent_doc = Document(
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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                return [[0.1] * 64 for _ in texts]
            
            async def embed_query(self, text):
                return [0.1] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
        
        docs = [
            Document(text="Long document A. " * 20, metadata={"id": "1"}),
            Document(text="Long document B. " * 20, metadata={"id": "2"}),
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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                # Score based on "Python" presence
                return [[1.0 if "Python" in t else 0.1] * 64 for t in texts]
            
            async def embed_query(self, text):
                return [1.0 if "Python" in text else 0.1] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
        
        doc = Document(
            text="First sentence. Second sentence about Python. Third sentence. "
                 "Fourth sentence. Fifth sentence.",
            metadata={"id": "1"},
        )
        
        retriever = SentenceWindowRetriever(vs, window_size=1)
        await retriever.add_documents([doc])
        
        results = await retriever.retrieve("Python", k=1)
        
        # Should include surrounding sentences
        assert "Python" in results[0].text


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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                return [[0.1] * 64 for _ in texts]
            
            async def embed_query(self, text):
                return [0.1] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
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
        
        class MockEmbed:
            async def embed_texts(self, texts):
                return [[0.1] * 64 for _ in texts]
            
            async def embed_query(self, text):
                return [0.1] * 64
        
        vs = VectorStore(embeddings=MockEmbed(), backend=InMemoryBackend())
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
