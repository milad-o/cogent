"""Summary-based retrieval indexes.

These indexes use LLM summarization for efficient retrieval over large documents:

- SummaryIndex: Creates summaries of documents for quick scanning
- TreeIndex: Hierarchical tree of summaries for very large documents
- KeywordTableIndex: Keyword extraction with document mapping

These are particularly useful when:
1. Documents are too large for direct embedding
2. You need to understand document-level topics quickly
3. You want to combine with KnowledgeGraph for entity extraction
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, RetrievalResult
from agenticflow.retriever.utils.llm_adapter import adapt_llm
from agenticflow.vectorstore import Document

if TYPE_CHECKING:
    from agenticflow.models import Model
    from agenticflow.retriever.utils.llm_adapter import LLMProtocol
    from agenticflow.vectorstore import VectorStore


@dataclass
class DocumentSummary:
    """Summary of a document with metadata."""

    doc_id: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)  # For KG integration
    metadata: dict[str, Any] = field(default_factory=dict)


class SummaryIndex(BaseRetriever):
    """Index that summarizes documents for efficient retrieval.

    Creates LLM-generated summaries of each document, then retrieves
    by searching summaries. Returns full documents for matched summaries.

    Benefits:
    - Works well for long documents
    - Captures high-level themes
    - Can extract entities for KnowledgeGraph integration

    Example:
        ```python
        from agenticflow.retriever import SummaryIndex
        from agenticflow.models import OpenAIModel

        index = SummaryIndex(
            llm=OpenAIModel(),
            vectorstore=vectorstore,  # For summary embeddings
            extract_entities=True,    # For KG integration
        )

        await index.add_documents(long_documents)
        results = await index.retrieve("machine learning concepts")

        # Access extracted entities for KG
        for summary in index.summaries.values():
            for entity in summary.entities:
                kg.add_entity(entity["name"], entity["type"])
        ```
    """

    _name: str = "summary"

    # Prompts
    SUMMARY_PROMPT = '''Summarize the following document concisely, capturing the main topics and key points.
Keep the summary under 200 words.

Document:
{text}

Summary:'''

    SUMMARY_WITH_ENTITIES_PROMPT = '''Analyze the following document and provide:
1. A concise summary (under 200 words)
2. Key entities mentioned (people, organizations, concepts, technologies)
3. Important keywords

Document:
{text}

Respond in JSON format:
{{
    "summary": "concise summary here",
    "entities": [
        {{"name": "entity name", "type": "Person|Organization|Concept|Technology|Location|Other"}}
    ],
    "keywords": ["keyword1", "keyword2", ...]
}}'''

    def __init__(
        self,
        llm: Model,
        vectorstore: VectorStore | None = None,
        *,
        extract_entities: bool = False,
        extract_keywords: bool = True,
        name: str | None = None,
        verbose: bool = False,
        logger: Any | None = None,
    ) -> None:
        """Create a summary index.

        Args:
            llm: Language model for generating summaries.
                Automatically adapts chat models to .generate() interface.
            vectorstore: Optional vector store for summary embeddings.
                If not provided, uses keyword matching on summaries.
            extract_entities: Extract entities for KnowledgeGraph integration.
            extract_keywords: Extract keywords for additional matching.
            name: Optional custom name.
            verbose: If True, emit structured logs via ObservabilityLogger.
            logger: Optional ObservabilityLogger instance. If provided, overrides
                `verbose` and will be used for all SummaryIndex logs.
        """
        self._llm: LLMProtocol = adapt_llm(llm)  # Auto-adapt chat models
        self._vectorstore = vectorstore
        self._extract_entities = extract_entities
        self._extract_keywords = extract_keywords

        # Optional observability
        self._log = None
        if logger is not None:
            self._log = logger
        elif verbose:
            from agenticflow.observability.logger import LogLevel, ObservabilityLogger

            self._log = ObservabilityLogger(
                name="agenticflow.retriever.summary_index",
                level=LogLevel.DEBUG,
            )
            self._log.set_context(
                retriever="SummaryIndex",
                extract_entities=extract_entities,
                extract_keywords=extract_keywords,
                has_vectorstore=vectorstore is not None,
            )

        # Storage
        self._documents: dict[str, Document] = {}
        self._summaries: dict[str, DocumentSummary] = {}

        if name:
            self._name = name

    @property
    def summaries(self) -> dict[str, DocumentSummary]:
        """Access document summaries (useful for KG integration)."""
        return self._summaries

    def _generate_doc_id(self, text: str) -> str:
        """Generate a unique document ID."""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    async def _summarize_document(self, doc: Document) -> DocumentSummary:
        """Generate summary for a document."""
        doc_id = doc.id or self._generate_doc_id(doc.text)

        if self._extract_entities:
            prompt = self.SUMMARY_WITH_ENTITIES_PROMPT.format(text=doc.text[:8000])
            response = await self._llm.generate(prompt)

            try:
                # Parse JSON response
                data = json.loads(response)
                return DocumentSummary(
                    doc_id=doc_id,
                    summary=data.get("summary", response),
                    keywords=data.get("keywords", []),
                    entities=data.get("entities", []),
                    metadata=doc.metadata,
                )
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return DocumentSummary(
                    doc_id=doc_id,
                    summary=response,
                    metadata=doc.metadata,
                )
        else:
            prompt = self.SUMMARY_PROMPT.format(text=doc.text[:8000])
            response = await self._llm.generate(prompt)

            return DocumentSummary(
                doc_id=doc_id,
                summary=response,
                metadata=doc.metadata,
            )

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents and generate summaries.

        Args:
            documents: Documents to index.

        Returns:
            List of document IDs.
        """
        start = time.perf_counter()

        await self._emit(
            "retrieval.summary_index.start",
            {
                "documents_count": len(documents),
                "has_vectorstore": self._vectorstore is not None,
                "extract_keywords": self._extract_keywords,
                "extract_entities": self._extract_entities,
            },
        )

        if self._log is not None:
            self._log.info(
                "summary_index_add_documents_start",
                documents_count=len(documents),
            )

        ids: list[str] = []
        summary_texts: list[str] = []

        try:
            for idx, doc in enumerate(documents, start=1):
                doc_id = doc.id or self._generate_doc_id(doc.text)

                # Store original document
                self._documents[doc_id] = doc

                # Generate summary
                doc_start = time.perf_counter()
                summary = await self._summarize_document(doc)
                self._summaries[doc_id] = summary

                page = doc.metadata.get("page")
                duration_ms = (time.perf_counter() - doc_start) * 1000

                await self._emit(
                    "retrieval.summary_index.document_summarized",
                    {
                        "doc_index": idx,
                        "doc_total": len(documents),
                        "doc_id": doc_id,
                        "page": page,
                        "text_chars": len(doc.text),
                        "summary_chars": len(summary.summary),
                        "duration_ms": duration_ms,
                    },
                )

                if self._log is not None:
                    self._log.debug(
                        "summary_index_document_summarized",
                        doc_index=idx,
                        doc_total=len(documents),
                        doc_id=doc_id,
                        page=page,
                        text_chars=len(doc.text),
                        summary_chars=len(summary.summary),
                        duration_ms=duration_ms,
                    )

                summary_texts.append(summary.summary)
                ids.append(doc_id)

            # Add summaries to vector store if available
            if self._vectorstore:
                vs_start = time.perf_counter()
                await self._vectorstore.add_texts(
                    summary_texts,
                    metadatas=[{"doc_id": id_} for id_ in ids],
                )

                if self._log is not None:
                    self._log.info(
                        "summary_index_vectorstore_added",
                        summaries_count=len(summary_texts),
                        duration_ms=(time.perf_counter() - vs_start) * 1000,
                    )

            duration_ms = (time.perf_counter() - start) * 1000
            await self._emit(
                "retrieval.summary_index.complete",
                {
                    "documents_count": len(documents),
                    "summaries_count": len(summary_texts),
                    "duration_ms": duration_ms,
                },
            )

            if self._log is not None:
                self._log.info(
                    "summary_index_add_documents_complete",
                    documents_count=len(documents),
                    duration_ms=duration_ms,
                )

            return ids

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            await self._emit(
                "retrieval.summary_index.error",
                {
                    "documents_count": len(documents),
                    "duration_ms": duration_ms,
                    "error": str(exc),
                },
            )
            if self._log is not None:
                self._log.error(
                    "summary_index_add_documents_error",
                    documents_count=len(documents),
                    duration_ms=duration_ms,
                    error=str(exc),
                )
            raise

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents by searching summaries.

        Args:
            query: Search query.
            k: Number of documents to retrieve.
            filter: Optional metadata filter.

        Returns:
            Original documents matching summary search.
        """
        if self._vectorstore:
            # Use vector similarity on summaries
            results = await self._vectorstore.search(query, k=k, filter=filter)

            retrieval_results = []
            for r in results:
                doc_id = r.document.metadata.get("doc_id")
                if doc_id and doc_id in self._documents:
                    original_doc = self._documents[doc_id]
                    summary = self._summaries.get(doc_id)

                    retrieval_results.append(RetrievalResult(
                        document=original_doc,
                        score=r.score,
                        retriever_name=self.name,
                        metadata={
                            "summary": summary.summary if summary else None,
                            "keywords": summary.keywords if summary else [],
                            "entities": summary.entities if summary else [],
                        },
                    ))

            return retrieval_results
        else:
            # Fallback: keyword matching on summaries
            query_lower = query.lower()
            scored = []

            for doc_id, summary in self._summaries.items():
                # Simple keyword matching score
                score = 0.0
                summary_lower = summary.summary.lower()

                for word in query_lower.split():
                    if word in summary_lower:
                        score += 1.0
                    if word in [kw.lower() for kw in summary.keywords]:
                        score += 0.5

                if score > 0:
                    scored.append((doc_id, score))

            # Sort by score and return top k
            scored.sort(key=lambda x: x[1], reverse=True)

            results = []
            for doc_id, score in scored[:k]:
                doc = self._documents[doc_id]
                summary = self._summaries[doc_id]

                results.append(RetrievalResult(
                    document=doc,
                    score=score / len(query.split()),  # Normalize
                    retriever_name=self.name,
                    metadata={
                        "summary": summary.summary,
                        "keywords": summary.keywords,
                        "entities": summary.entities,
                    },
                ))

            return results

    def get_entities(self) -> list[dict[str, Any]]:
        """Get all extracted entities for KnowledgeGraph integration.

        Returns:
            List of entities with name, type, and source document.

        Example:
            ```python
            # Extract entities and add to KnowledgeGraph
            entities = summary_index.get_entities()
            for entity in entities:
                kg.add_entity(
                    entity["name"],
                    entity["type"],
                    {"source_doc": entity["source_doc"]}
                )
            ```
        """
        all_entities = []
        for doc_id, summary in self._summaries.items():
            for entity in summary.entities:
                all_entities.append({
                    **entity,
                    "source_doc": doc_id,
                })
        return all_entities

    def get_keywords(self) -> dict[str, list[str]]:
        """Get keyword to document mapping.

        Returns:
            Dictionary mapping keywords to document IDs.
        """
        keyword_map: dict[str, list[str]] = {}
        for doc_id, summary in self._summaries.items():
            for keyword in summary.keywords:
                if keyword not in keyword_map:
                    keyword_map[keyword] = []
                keyword_map[keyword].append(doc_id)
        return keyword_map


class TreeIndex(BaseRetriever):
    """Hierarchical tree index for very large documents.

    Builds a tree of summaries where:
    - Leaf nodes are chunks of the original document
    - Internal nodes are summaries of their children
    - Root is a summary of the entire document

    Retrieval traverses from root, following relevant branches.

    Best for:
    - Very long documents (books, reports)
    - When you need multi-level abstraction
    - Hierarchical topic exploration

    Example:
        ```python
        from agenticflow.retriever import TreeIndex

        index = TreeIndex(
            llm=OpenAIModel(),
            chunk_size=2000,
            max_children=4,  # 4-ary tree
        )

        await index.add_document(very_long_document)
        results = await index.retrieve("specific topic")
        ```
    """

    _name: str = "tree"

    SUMMARIZE_PROMPT = '''Summarize the following texts into a single coherent summary.
Capture the main themes and key points. Keep it under 300 words.

Texts:
{texts}

Summary:'''

    @dataclass
    class TreeNode:
        """Node in the summary tree."""
        node_id: str
        text: str
        summary: str | None = None
        children: list[str] = field(default_factory=list)
        parent: str | None = None
        is_leaf: bool = True
        depth: int = 0

    def __init__(
        self,
        llm: Model,
        *,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        max_children: int = 4,
        name: str | None = None,
    ) -> None:
        """Create a tree index.

        Args:
            llm: Language model for summaries.
            chunk_size: Size of leaf chunks.
            chunk_overlap: Overlap between chunks.
            max_children: Max children per node (tree branching factor).
            name: Optional custom name.
        """
        self._llm: LLMProtocol = adapt_llm(llm)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_children = max_children

        # Tree storage
        self._nodes: dict[str, TreeIndex.TreeNode] = {}
        self._root_ids: list[str] = []
        self._documents: dict[str, Document] = {}

        if name:
            self._name = name

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]

            # Try to break at paragraph/sentence
            if end < len(text):
                for sep in ["\n\n", ". ", ".\n", "\n"]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > self._chunk_size // 2:
                        chunk = chunk[:last_sep + len(sep)]
                        break

            if chunk.strip():
                chunks.append(chunk.strip())

            start += len(chunk) - self._chunk_overlap
            if start <= 0:
                start = end

        return chunks

    def _generate_node_id(self, text: str, depth: int) -> str:
        """Generate unique node ID."""
        hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"node_{depth}_{hash_val}"

    async def _build_tree(self, chunks: list[str], doc_id: str) -> str:
        """Build summary tree from chunks, return root node ID."""
        # Create leaf nodes
        current_level: list[str] = []

        for chunk in chunks:
            node_id = self._generate_node_id(chunk, 0)
            self._nodes[node_id] = self.TreeNode(
                node_id=node_id,
                text=chunk,
                is_leaf=True,
                depth=0,
            )
            current_level.append(node_id)

        # Build tree bottom-up
        depth = 1
        while len(current_level) > 1:
            next_level: list[str] = []

            # Group nodes
            for i in range(0, len(current_level), self._max_children):
                children = current_level[i:i + self._max_children]

                # Get texts to summarize
                child_texts = []
                for child_id in children:
                    node = self._nodes[child_id]
                    child_texts.append(node.summary or node.text)

                # Generate summary
                combined = "\n\n---\n\n".join(child_texts)
                prompt = self.SUMMARIZE_PROMPT.format(texts=combined[:6000])
                summary = await self._llm.generate(prompt)

                # Create parent node
                parent_id = self._generate_node_id(summary, depth)
                self._nodes[parent_id] = self.TreeNode(
                    node_id=parent_id,
                    text=combined[:500],  # Store truncated for reference
                    summary=summary,
                    children=children,
                    is_leaf=False,
                    depth=depth,
                )

                # Update children's parent
                for child_id in children:
                    self._nodes[child_id].parent = parent_id

                next_level.append(parent_id)

            current_level = next_level
            depth += 1

        return current_level[0] if current_level else ""

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents and build summary trees.

        Args:
            documents: Documents to index.

        Returns:
            List of root node IDs (one per document).
        """
        root_ids = []

        for doc in documents:
            doc_id = doc.id or self._generate_node_id(doc.text, -1)
            self._documents[doc_id] = doc

            # Chunk the document
            chunks = self._chunk_text(doc.text)

            # Build tree
            root_id = await self._build_tree(chunks, doc_id)
            root_ids.append(root_id)
            self._root_ids.append(root_id)

        return root_ids

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve by traversing tree from roots.

        Uses query similarity to navigate tree and find relevant leaves.
        """
        if not self._root_ids:
            return []

        # Score all nodes
        query_lower = query.lower()

        def score_node(node: TreeIndex.TreeNode) -> float:
            text = (node.summary or node.text).lower()
            score = 0.0
            for word in query_lower.split():
                if word in text:
                    score += 1.0
            return score

        # Find best leaf nodes
        scored_leaves: list[tuple[str, float]] = []

        for node_id, node in self._nodes.items():
            if node.is_leaf:
                score = score_node(node)
                if score > 0:
                    scored_leaves.append((node_id, score))

        # Sort and take top k
        scored_leaves.sort(key=lambda x: x[1], reverse=True)

        results = []
        for node_id, score in scored_leaves[:k]:
            node = self._nodes[node_id]

            # Find path to root for context
            path = []
            current = node
            while current.parent:
                parent = self._nodes[current.parent]
                if parent.summary:
                    path.append(parent.summary)
                current = parent

            results.append(RetrievalResult(
                document=Document(text=node.text, metadata={"node_id": node_id}),
                score=score / len(query.split()),
                retriever_name=self.name,
                metadata={
                    "depth": node.depth,
                    "path_summaries": path[::-1],  # Root to leaf
                },
            ))

        return results


class KeywordTableIndex(BaseRetriever):
    """Keyword-based index with document mapping.

    Extracts keywords from documents using LLM or simple extraction,
    builds an inverted index for fast keyword lookup.

    Best for:
    - When you know users search by specific terms
    - Fast exact-match retrieval
    - Complementing semantic search

    Example:
        ```python
        from agenticflow.retriever import KeywordTableIndex

        index = KeywordTableIndex(
            llm=model,  # Optional: for better keyword extraction
            max_keywords_per_doc=20,
        )

        await index.add_documents(documents)
        results = await index.retrieve("machine learning neural network")
        ```
    """

    _name: str = "keyword_table"

    EXTRACT_KEYWORDS_PROMPT = '''Extract the most important keywords and key phrases from this document.
Include technical terms, proper nouns, and important concepts.
Return as a JSON array of strings.

Document:
{text}

Keywords (JSON array):'''

    def __init__(
        self,
        llm: Model | None = None,
        *,
        max_keywords_per_doc: int = 20,
        use_llm_extraction: bool = True,
        name: str | None = None,
    ) -> None:
        """Create a keyword table index.

        Args:
            llm: Language model for keyword extraction (optional).
                Automatically adapts chat models to .generate() interface.
            max_keywords_per_doc: Maximum keywords to extract per document.
            use_llm_extraction: Use LLM for extraction (if False, uses simple extraction).
            name: Optional custom name.
        """
        self._llm: LLMProtocol | None = adapt_llm(llm) if llm else None
        self._max_keywords = max_keywords_per_doc
        self._use_llm = use_llm_extraction and llm is not None

        # Inverted index: keyword -> list of (doc_id, score)
        self._keyword_index: dict[str, list[tuple[str, float]]] = {}
        self._documents: dict[str, Document] = {}
        self._doc_keywords: dict[str, list[str]] = {}

        if name:
            self._name = name

    def _simple_extract_keywords(self, text: str) -> list[str]:
        """Simple keyword extraction without LLM."""
        import re
        from collections import Counter

        # Tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Remove common stopwords
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'has', 'have', 'will', 'your', 'was', 'were', 'been', 'being',
            'that', 'this', 'with', 'from', 'they', 'which', 'their', 'would',
            'there', 'what', 'about', 'when', 'make', 'like', 'just', 'over',
            'such', 'into', 'than', 'them', 'some', 'could', 'other', 'then',
        }
        words = [w for w in words if w not in stopwords]

        # Count and return top keywords
        counter = Counter(words)
        return [word for word, _ in counter.most_common(self._max_keywords)]

    async def _extract_keywords(self, doc: Document) -> list[str]:
        """Extract keywords from document."""
        if self._use_llm and self._llm:
            prompt = self.EXTRACT_KEYWORDS_PROMPT.format(text=doc.text[:4000])
            response = await self._llm.generate(prompt)

            try:
                # Parse JSON array
                keywords = json.loads(response)
                if isinstance(keywords, list):
                    return keywords[:self._max_keywords]
            except json.JSONDecodeError:
                pass

        # Fallback to simple extraction
        return self._simple_extract_keywords(doc.text)

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents and build keyword index.

        Args:
            documents: Documents to index.

        Returns:
            List of document IDs.
        """
        ids = []

        for doc in documents:
            doc_id = doc.id or hashlib.md5(doc.text.encode()).hexdigest()[:12]
            self._documents[doc_id] = doc

            # Extract keywords
            keywords = await self._extract_keywords(doc)
            self._doc_keywords[doc_id] = keywords

            # Build inverted index
            for i, keyword in enumerate(keywords):
                keyword_lower = keyword.lower()
                # Score based on position (earlier = more important)
                score = 1.0 - (i / len(keywords)) * 0.5

                if keyword_lower not in self._keyword_index:
                    self._keyword_index[keyword_lower] = []
                self._keyword_index[keyword_lower].append((doc_id, score))

            ids.append(doc_id)

        return ids

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents by keyword matching.

        Args:
            query: Search query (keywords).
            k: Number of documents to retrieve.
            filter: Optional metadata filter.

        Returns:
            Documents matching query keywords.
        """
        import re

        # Extract query keywords
        query_keywords = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())

        # Score documents
        doc_scores: dict[str, float] = {}
        doc_matched_keywords: dict[str, list[str]] = {}

        for keyword in query_keywords:
            if keyword in self._keyword_index:
                for doc_id, score in self._keyword_index[keyword]:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                        doc_matched_keywords[doc_id] = []
                    doc_scores[doc_id] += score
                    doc_matched_keywords[doc_id].append(keyword)

        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:k]:
            doc = self._documents[doc_id]

            results.append(RetrievalResult(
                document=doc,
                score=score / len(query_keywords) if query_keywords else 0.0,
                retriever_name=self.name,
                metadata={
                    "matched_keywords": doc_matched_keywords[doc_id],
                    "doc_keywords": self._doc_keywords[doc_id],
                },
            ))

        return results

    def get_keyword_table(self) -> dict[str, list[str]]:
        """Get the keyword to document mapping.

        Returns:
            Dictionary mapping keywords to document IDs.
        """
        return {
            keyword: [doc_id for doc_id, _ in docs]
            for keyword, docs in self._keyword_index.items()
        }


class KnowledgeGraphIndex(BaseRetriever):
    """Index that integrates with KnowledgeGraph capability.

    Extracts entities and relationships from documents and stores them
    in a KnowledgeGraph, enabling graph-based retrieval.

    Features:
    - Entity extraction from documents
    - Relationship extraction between entities
    - Graph traversal for retrieval
    - Combines with vector search for hybrid retrieval

    Example:
        ```python
        from agenticflow.retriever import KnowledgeGraphIndex
        from agenticflow.capabilities import KnowledgeGraph

        kg = KnowledgeGraph(backend="sqlite", path="knowledge.db")

        index = KnowledgeGraphIndex(
            llm=model,
            knowledge_graph=kg,
            vectorstore=vectorstore,  # Optional for hybrid
        )

        await index.add_documents(documents)

        # Graph-based retrieval
        results = await index.retrieve("Who works at Acme Corp?")

        # Direct graph queries
        kg.query("? -works_at-> Acme Corp")
        ```
    """

    _name: str = "knowledge_graph"

    EXTRACT_TRIPLETS_PROMPT = '''Extract entities and relationships from this text.

Text:
{text}

For each relationship found, output in this JSON format:
{{
    "entities": [
        {{"name": "entity name", "type": "Person|Organization|Location|Concept|Technology|Product|Event|Other"}}
    ],
    "relationships": [
        {{"source": "entity1", "relation": "relationship_type", "target": "entity2"}}
    ]
}}

Common relationship types: works_at, located_in, created_by, part_of, related_to, manages, owns, uses, depends_on

Extract all meaningful entities and relationships:'''

    def __init__(
        self,
        llm: Model,
        knowledge_graph: Any,  # KnowledgeGraph capability
        vectorstore: VectorStore | None = None,
        *,
        include_text_chunks: bool = True,
        chunk_size: int = 1000,
        name: str | None = None,
    ) -> None:
        """Create a knowledge graph index.

        Args:
            llm: Language model for extraction.
            knowledge_graph: KnowledgeGraph capability instance.
            vectorstore: Optional vector store for hybrid retrieval.
            include_text_chunks: Store text chunks with entity references.
            chunk_size: Size of text chunks.
            name: Optional custom name.
        """
        self._llm: LLMProtocol = adapt_llm(llm)
        self._kg = knowledge_graph
        self._vectorstore = vectorstore
        self._include_chunks = include_text_chunks
        self._chunk_size = chunk_size

        # Document storage
        self._documents: dict[str, Document] = {}
        self._chunk_entities: dict[str, list[str]] = {}  # chunk_id -> entity names

        if name:
            self._name = name

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks for processing."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]

            if end < len(text):
                for sep in ["\n\n", ". ", ".\n"]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > self._chunk_size // 2:
                        chunk = chunk[:last_sep + len(sep)]
                        break

            if chunk.strip():
                chunks.append(chunk.strip())

            start += len(chunk)

        return chunks

    async def _extract_graph_data(self, text: str, doc_id: str) -> tuple[list[dict], list[dict]]:
        """Extract entities and relationships from text."""
        prompt = self.EXTRACT_TRIPLETS_PROMPT.format(text=text[:4000])
        response = await self._llm.generate(prompt)

        try:
            data = json.loads(response)
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            return entities, relationships
        except json.JSONDecodeError:
            return [], []

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents, extract entities/relationships, populate KG.

        Args:
            documents: Documents to process.

        Returns:
            List of document IDs.
        """
        ids = []

        for doc in documents:
            doc_id = doc.id or hashlib.md5(doc.text.encode()).hexdigest()[:12]
            self._documents[doc_id] = doc

            # Process chunks
            chunks = self._chunk_text(doc.text)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"

                # Extract entities and relationships
                entities, relationships = await self._extract_graph_data(chunk, doc_id)

                # Add to knowledge graph
                entity_names = []
                for entity in entities:
                    self._kg.graph.add_entity(
                        entity["name"],
                        entity.get("type", "Entity"),
                        {"source_doc": doc_id, "chunk_id": chunk_id},
                    )
                    entity_names.append(entity["name"])

                for rel in relationships:
                    self._kg.graph.add_relationship(
                        rel["source"],
                        rel["relation"],
                        rel["target"],
                        {"source_doc": doc_id},
                    )

                self._chunk_entities[chunk_id] = entity_names

                # Add chunk to vector store if available
                if self._vectorstore and self._include_chunks:
                    await self._vectorstore.add_texts(
                        [chunk],
                        metadatas=[{
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "entities": entity_names,
                        }],
                    )

            ids.append(doc_id)

        return ids

    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using graph + optional vector search.

        First finds entities mentioned in query, then:
        1. Retrieves related entities via graph traversal
        2. Optionally combines with vector similarity
        """
        results = []

        # Try to find entities in query
        query_lower = query.lower()
        relevant_doc_ids: set[str] = set()
        related_entities: list[str] = []

        # Check which entities are mentioned
        for entity in self._kg.graph.get_all_entities():
            if entity.id.lower() in query_lower:
                # Get related entities
                rels = self._kg.graph.get_relationships(entity.id, direction="both")
                for rel in rels:
                    related_entities.append(rel.target_id if rel.source_id == entity.id else rel.source_id)

                # Find documents mentioning this entity
                if hasattr(entity, "attributes") and "source_doc" in entity.attributes:
                    relevant_doc_ids.add(entity.attributes["source_doc"])

        # If we found related entities, get their source docs too
        for entity_name in related_entities:
            entity = self._kg.graph.get_entity(entity_name)
            if entity and hasattr(entity, "attributes") and "source_doc" in entity.attributes:
                relevant_doc_ids.add(entity.attributes["source_doc"])

        # Convert to results
        for doc_id in list(relevant_doc_ids)[:k]:
            if doc_id in self._documents:
                results.append(RetrievalResult(
                    document=self._documents[doc_id],
                    score=1.0,
                    retriever_name=self.name,
                    metadata={
                        "retrieval_method": "graph",
                        "related_entities": related_entities[:10],
                    },
                ))

        # Supplement with vector search if available
        if self._vectorstore and len(results) < k:
            vector_results = await self._vectorstore.search(query, k=k - len(results))

            for vr in vector_results:
                doc_id = vr.document.metadata.get("doc_id")
                if doc_id and doc_id in self._documents and doc_id not in relevant_doc_ids:
                    results.append(RetrievalResult(
                        document=self._documents[doc_id],
                        score=vr.score,
                        retriever_name=self.name,
                        metadata={
                            "retrieval_method": "vector",
                            "entities": vr.document.metadata.get("entities", []),
                        },
                    ))

        return results[:k]
