"""Multi-representation index for diverse query handling.

Stores multiple vector embeddings per document, each capturing different aspects:
- Summary embedding: Captures overall concept (for broad queries)
- Detail embedding: Captures technical specifics (for precise queries)
- Keyword embedding: Captures key terms (for exact matches)
- Question embedding: Captures potential questions answered (for Q&A)

Query routing selects the best representation based on query type.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, FusionStrategy, RetrievalResult
from agenticflow.retriever.utils import fuse_results
from agenticflow.vectorstore import Document

if TYPE_CHECKING:
    from agenticflow.models import Model
    from agenticflow.vectorstore import VectorStore


class RepresentationType(Enum):
    """Types of document representations."""
    ORIGINAL = "original"      # Raw document text
    SUMMARY = "summary"        # Condensed summary
    DETAILED = "detailed"      # Technical details emphasized
    KEYWORDS = "keywords"      # Key terms and concepts
    QUESTIONS = "questions"    # Hypothetical questions answered
    ENTITIES = "entities"      # Named entities and relationships


class QueryType(Enum):
    """Types of queries for routing."""
    BROAD = "broad"            # General, conceptual queries
    SPECIFIC = "specific"      # Technical, detailed queries
    KEYWORD = "keyword"        # Exact term searches
    QUESTION = "question"      # Natural language questions
    ENTITY = "entity"          # Entity-focused queries
    AUTO = "auto"              # Automatic detection


@dataclass
class DocumentRepresentations:
    """Multiple representations of a single document."""
    
    doc_id: str
    original: str
    summary: str | None = None
    detailed: str | None = None
    keywords: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiRepresentationIndex(BaseRetriever):
    """Index with multiple embeddings per document.
    
    Each document is embedded in multiple ways:
    - Original: Raw text embedding
    - Summary: Conceptual summary embedding
    - Detailed: Technical details embedding
    - Keywords: Key term embedding
    - Questions: Hypothetical Q&A embedding
    
    Query routing selects the best representation based on query type,
    or searches all and fuses results.
    
    Benefits:
    - Better coverage for diverse query styles
    - Handles both broad and specific queries
    - Works well for technical/specialized domains
    - Improves recall without sacrificing precision
    
    Example:
        ```python
        from agenticflow.retriever import MultiRepresentationIndex, QueryType
        
        index = MultiRepresentationIndex(
            vectorstore=vs,
            llm=model,
            representations=["summary", "detailed", "questions"],
        )
        
        await index.add_documents(docs)
        
        # Auto-detect query type and route
        results = await index.retrieve("What is machine learning?")
        
        # Force specific representation
        results = await index.retrieve(
            "neural network backpropagation algorithm",
            query_type=QueryType.SPECIFIC,
        )
        
        # Search all representations and fuse
        results = await index.retrieve(
            "AI applications",
            query_type=QueryType.AUTO,
            search_all=True,
        )
        ```
    """
    
    _name: str = "multi_representation"
    
    # Generation prompts
    SUMMARY_PROMPT = '''Create a brief conceptual summary of this document.
Focus on the main ideas and high-level concepts.
Keep it under 100 words.

Document:
{text}

Summary:'''

    DETAILED_PROMPT = '''Extract the key technical details from this document.
Include specific terms, methods, numbers, and precise information.
Focus on details that would answer specific technical questions.

Document:
{text}

Technical Details:'''

    KEYWORDS_PROMPT = '''Extract the most important keywords and key phrases from this document.
Include technical terms, concepts, proper nouns, and important topics.
Return as a comma-separated list.

Document:
{text}

Keywords:'''

    QUESTIONS_PROMPT = '''Generate 3-5 questions that this document answers.
Focus on questions a user might ask that would lead to this document.

Document:
{text}

Questions (one per line):'''

    ENTITIES_PROMPT = '''Extract named entities from this document.
Include people, organizations, technologies, products, and locations.
Return as a comma-separated list.

Document:
{text}

Entities:'''

    QUERY_CLASSIFICATION_PROMPT = '''Classify this search query into one of these categories:
- BROAD: General, conceptual, or exploratory query
- SPECIFIC: Technical, detailed, or precise query
- KEYWORD: Looking for specific terms or exact matches
- QUESTION: Natural language question
- ENTITY: Looking for specific person, organization, or thing

Query: {query}

Category (one word):'''

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: Model,
        *,
        representations: list[str] | None = None,
        auto_route: bool = True,
        fusion_strategy: FusionStrategy | str = FusionStrategy.RRF,
        name: str | None = None,
    ) -> None:
        """Create a multi-representation index.
        
        Args:
            vectorstore: Vector store for embeddings.
            llm: Language model for generating representations.
            representations: Which representations to generate.
                Options: "original", "summary", "detailed", "keywords", "questions", "entities"
                Default: ["original", "summary", "detailed"]
            auto_route: Automatically route queries to best representation.
            fusion_strategy: How to fuse results from multiple representations.
            name: Optional custom name.
        """
        self._vectorstore = vectorstore
        self._llm = llm
        self._representations = representations or ["original", "summary", "detailed"]
        self._auto_route = auto_route
        
        if isinstance(fusion_strategy, str):
            fusion_strategy = FusionStrategy(fusion_strategy)
        self._fusion_strategy = fusion_strategy
        
        # Storage
        self._documents: dict[str, Document] = {}
        self._doc_representations: dict[str, DocumentRepresentations] = {}
        
        if name:
            self._name = name
    
    def _generate_id(self, text: str) -> str:
        """Generate unique document ID."""
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    async def _generate_summary(self, text: str) -> str:
        """Generate conceptual summary."""
        prompt = self.SUMMARY_PROMPT.format(text=text[:4000])
        return await self._llm.generate(prompt)
    
    async def _generate_detailed(self, text: str) -> str:
        """Generate technical details."""
        prompt = self.DETAILED_PROMPT.format(text=text[:4000])
        return await self._llm.generate(prompt)
    
    async def _generate_keywords(self, text: str) -> list[str]:
        """Generate keyword list."""
        prompt = self.KEYWORDS_PROMPT.format(text=text[:4000])
        response = await self._llm.generate(prompt)
        return [kw.strip() for kw in response.split(",") if kw.strip()]
    
    async def _generate_questions(self, text: str) -> list[str]:
        """Generate hypothetical questions."""
        prompt = self.QUESTIONS_PROMPT.format(text=text[:4000])
        response = await self._llm.generate(prompt)
        return [q.strip() for q in response.strip().split("\n") if q.strip()]
    
    async def _generate_entities(self, text: str) -> list[str]:
        """Generate entity list."""
        prompt = self.ENTITIES_PROMPT.format(text=text[:4000])
        response = await self._llm.generate(prompt)
        return [e.strip() for e in response.split(",") if e.strip()]
    
    async def _classify_query(self, query: str) -> QueryType:
        """Classify query to determine best representation."""
        if not self._auto_route:
            return QueryType.AUTO
        
        prompt = self.QUERY_CLASSIFICATION_PROMPT.format(query=query)
        response = await self._llm.generate(prompt)
        
        response = response.strip().upper()
        
        # Map response to QueryType
        if "BROAD" in response:
            return QueryType.BROAD
        elif "SPECIFIC" in response or "TECHNICAL" in response:
            return QueryType.SPECIFIC
        elif "KEYWORD" in response or "TERM" in response:
            return QueryType.KEYWORD
        elif "QUESTION" in response or "?" in query:
            return QueryType.QUESTION
        elif "ENTITY" in response or "PERSON" in response or "ORG" in response:
            return QueryType.ENTITY
        
        return QueryType.AUTO
    
    def _get_representation_for_query_type(self, query_type: QueryType) -> str:
        """Map query type to representation type."""
        mapping = {
            QueryType.BROAD: "summary",
            QueryType.SPECIFIC: "detailed",
            QueryType.KEYWORD: "keywords",
            QueryType.QUESTION: "questions",
            QueryType.ENTITY: "entities",
            QueryType.AUTO: "original",
        }
        rep = mapping.get(query_type, "original")
        
        # Fall back to original if representation not available
        if rep not in self._representations:
            return "original"
        return rep
    
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents and generate all representations.
        
        Args:
            documents: Documents to index.
            
        Returns:
            List of document IDs.
        """
        doc_ids = []
        
        for doc in documents:
            doc_id = doc.id or self._generate_id(doc.text)
            self._documents[doc_id] = doc
            doc_ids.append(doc_id)
            
            # Generate representations
            reps = DocumentRepresentations(
                doc_id=doc_id,
                original=doc.text,
                metadata=doc.metadata,
            )
            
            if "summary" in self._representations:
                reps.summary = await self._generate_summary(doc.text)
            
            if "detailed" in self._representations:
                reps.detailed = await self._generate_detailed(doc.text)
            
            if "keywords" in self._representations:
                reps.keywords = await self._generate_keywords(doc.text)
            
            if "questions" in self._representations:
                reps.questions = await self._generate_questions(doc.text)
            
            if "entities" in self._representations:
                reps.entities = await self._generate_entities(doc.text)
            
            self._doc_representations[doc_id] = reps
            
            # Add each representation to vector store
            for rep_type in self._representations:
                if rep_type == "original":
                    text = reps.original
                elif rep_type == "summary":
                    text = reps.summary or ""
                elif rep_type == "detailed":
                    text = reps.detailed or ""
                elif rep_type == "keywords":
                    text = ", ".join(reps.keywords)
                elif rep_type == "questions":
                    text = "\n".join(reps.questions)
                elif rep_type == "entities":
                    text = ", ".join(reps.entities)
                else:
                    continue
                
                if text:
                    await self._vectorstore.add_texts(
                        [text],
                        metadatas=[{
                            "doc_id": doc_id,
                            "representation": rep_type,
                            **doc.metadata,
                        }],
                    )
        
        return doc_ids
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        query_type: QueryType | None = None,
        search_all: bool = False,
    ) -> list[RetrievalResult]:
        """Retrieve using appropriate representation(s).
        
        Args:
            query: Search query.
            k: Number of documents.
            filter: Additional filters.
            query_type: Force specific query type (or auto-detect).
            search_all: Search all representations and fuse.
            
        Returns:
            Retrieved documents.
        """
        if query_type is None:
            query_type = await self._classify_query(query)
        
        if search_all or query_type == QueryType.AUTO:
            # Search all representations and fuse
            all_results: list[list[RetrievalResult]] = []
            
            for rep_type in self._representations:
                rep_filter = {"representation": rep_type}
                if filter:
                    rep_filter.update(filter)
                
                results = await self._vectorstore.search(query, k=k * 2, filter=rep_filter)
                
                rep_results = []
                for r in results:
                    doc_id = r.document.metadata.get("doc_id")
                    if doc_id and doc_id in self._documents:
                        rep_results.append(RetrievalResult(
                            document=self._documents[doc_id],
                            score=r.score,
                            retriever_name=self.name,
                            metadata={
                                "representation": rep_type,
                                "matched_text": r.document.text[:200],
                            },
                        ))
                
                if rep_results:
                    all_results.append(rep_results)
            
            # Fuse results
            if all_results:
                fused = fuse_results(all_results, strategy=self._fusion_strategy, k=k)
                return fused
            return []
        
        else:
            # Search specific representation
            rep_type = self._get_representation_for_query_type(query_type)
            
            rep_filter = {"representation": rep_type}
            if filter:
                rep_filter.update(filter)
            
            results = await self._vectorstore.search(query, k=k, filter=rep_filter)
            
            retrieval_results = []
            for r in results:
                doc_id = r.document.metadata.get("doc_id")
                if doc_id and doc_id in self._documents:
                    retrieval_results.append(RetrievalResult(
                        document=self._documents[doc_id],
                        score=r.score,
                        retriever_name=self.name,
                        metadata={
                            "query_type": query_type.value,
                            "representation": rep_type,
                            "matched_text": r.document.text[:200],
                        },
                    ))
            
            return retrieval_results
    
    def get_representations(self, doc_id: str) -> DocumentRepresentations | None:
        """Get all representations for a document.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            DocumentRepresentations or None if not found.
        """
        return self._doc_representations.get(doc_id)
    
    def get_representation_stats(self) -> dict[str, Any]:
        """Get statistics about stored representations.
        
        Returns:
            Dictionary with counts and sizes.
        """
        stats = {
            "total_documents": len(self._documents),
            "representations": {},
        }
        
        for rep_type in self._representations:
            count = 0
            total_chars = 0
            
            for doc_reps in self._doc_representations.values():
                if rep_type == "original":
                    if doc_reps.original:
                        count += 1
                        total_chars += len(doc_reps.original)
                elif rep_type == "summary":
                    if doc_reps.summary:
                        count += 1
                        total_chars += len(doc_reps.summary)
                elif rep_type == "detailed":
                    if doc_reps.detailed:
                        count += 1
                        total_chars += len(doc_reps.detailed)
                elif rep_type == "keywords":
                    if doc_reps.keywords:
                        count += 1
                        total_chars += sum(len(k) for k in doc_reps.keywords)
                elif rep_type == "questions":
                    if doc_reps.questions:
                        count += 1
                        total_chars += sum(len(q) for q in doc_reps.questions)
                elif rep_type == "entities":
                    if doc_reps.entities:
                        count += 1
                        total_chars += sum(len(e) for e in doc_reps.entities)
            
            stats["representations"][rep_type] = {
                "count": count,
                "total_chars": total_chars,
                "avg_chars": total_chars / count if count > 0 else 0,
            }
        
        return stats
