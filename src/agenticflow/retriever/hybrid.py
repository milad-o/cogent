"""Hybrid retriever combining metadata and content search.

True hybrid retrieval: search by metadata attributes first,
then by content within matching documents. Can wrap any retriever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, RetrievalResult
from agenticflow.vectorstore import Document

if TYPE_CHECKING:
    from agenticflow.retriever.base import Retriever
    from agenticflow.vectorstore import VectorStore


class MetadataMatchMode(Enum):
    """How to match metadata filters."""
    
    ALL = "all"      # All filters must match (AND)
    ANY = "any"      # Any filter can match (OR)
    BOOST = "boost"  # Metadata matches boost score, don't filter


@dataclass
class MetadataWeight:
    """Weight for a metadata field in scoring."""
    
    field: str
    weight: float = 1.0
    exact_match: bool = True  # False = contains/partial match


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining metadata and content search.
    
    This retriever implements true hybrid search:
    1. Filter/boost by metadata attributes
    2. Search content with the wrapped retriever
    3. Combine scores from both
    
    Unlike ensemble (which combines multiple retrievers), hybrid
    combines metadata-based filtering with content-based search.
    
    Example:
        >>> from agenticflow.retriever import HybridRetriever, DenseRetriever
        >>> 
        >>> # Wrap any retriever
        >>> content_retriever = DenseRetriever(vectorstore)
        >>> 
        >>> hybrid = HybridRetriever(
        ...     retriever=content_retriever,
        ...     metadata_fields=["category", "author", "department"],
        ...     metadata_weight=0.3,  # 30% metadata, 70% content
        ...     mode=MetadataMatchMode.BOOST,
        ... )
        >>> 
        >>> # Query searches both metadata and content
        >>> results = await hybrid.retrieve(
        ...     "machine learning best practices",
        ...     k=5,
        ...     filter={"category": "engineering"},  # Hard filter
        ... )
    
    Scoring modes:
        - BOOST: Metadata matches increase score, no filtering
        - ALL: Only return docs matching ALL metadata criteria
        - ANY: Return docs matching ANY metadata criteria
    """
    
    _name: str = "hybrid"
    
    def __init__(
        self,
        retriever: Retriever,
        *,
        metadata_fields: list[str] | None = None,
        metadata_weights: list[MetadataWeight] | None = None,
        metadata_weight: float = 0.3,
        content_weight: float = 0.7,
        mode: MetadataMatchMode | str = MetadataMatchMode.BOOST,
        name: str | None = None,
    ) -> None:
        """Create a hybrid metadata + content retriever.
        
        Args:
            retriever: The content retriever to wrap (DenseRetriever, etc.).
            metadata_fields: Fields to search in metadata (simple equal-weight).
            metadata_weights: Per-field weights (alternative to metadata_fields).
            metadata_weight: Weight for metadata score (default: 0.3).
            content_weight: Weight for content score (default: 0.7).
            mode: How metadata matches affect results (BOOST, ALL, ANY).
            name: Optional custom name.
        """
        self._retriever = retriever
        self._content_weight = content_weight
        self._metadata_weight = metadata_weight
        
        # Build metadata weights
        if metadata_weights:
            self._metadata_weights = metadata_weights
        elif metadata_fields:
            self._metadata_weights = [
                MetadataWeight(field=f, weight=1.0) for f in metadata_fields
            ]
        else:
            self._metadata_weights = []
        
        if isinstance(mode, str):
            mode = MetadataMatchMode(mode)
        self._mode = mode
        
        if name:
            self._name = name
    
    def _compute_metadata_score(
        self,
        document: Document,
        query: str,
        query_terms: set[str],
    ) -> float:
        """Compute metadata match score for a document.
        
        Args:
            document: Document to score.
            query: Original query string.
            query_terms: Set of lowercase query terms.
            
        Returns:
            Metadata match score (0.0 to 1.0).
        """
        if not self._metadata_weights:
            return 0.0
        
        total_weight = sum(mw.weight for mw in self._metadata_weights)
        if total_weight == 0:
            return 0.0
        
        score = 0.0
        
        for mw in self._metadata_weights:
            field_value = document.metadata.get(mw.field)
            if field_value is None:
                continue
            
            # Convert to string for matching
            field_str = str(field_value).lower()
            
            if mw.exact_match:
                # Check if any query term exactly matches
                if field_str in query_terms or any(
                    term in field_str for term in query_terms
                ):
                    score += mw.weight
            else:
                # Partial match - count matching terms
                matching_terms = sum(
                    1 for term in query_terms if term in field_str
                )
                if matching_terms > 0:
                    score += mw.weight * (matching_terms / len(query_terms))
        
        return score / total_weight
    
    def _matches_metadata_filter(
        self,
        document: Document,
        query_terms: set[str],
    ) -> bool:
        """Check if document matches metadata filter criteria.
        
        Args:
            document: Document to check.
            query_terms: Set of lowercase query terms.
            
        Returns:
            True if document matches according to mode.
        """
        if not self._metadata_weights:
            return True  # No metadata fields = always match
        
        matches = []
        
        for mw in self._metadata_weights:
            field_value = document.metadata.get(mw.field)
            if field_value is None:
                matches.append(False)
                continue
            
            field_str = str(field_value).lower()
            
            if mw.exact_match:
                match = field_str in query_terms or any(
                    term in field_str for term in query_terms
                )
            else:
                match = any(term in field_str for term in query_terms)
            
            matches.append(match)
        
        if self._mode == MetadataMatchMode.ALL:
            return all(matches) if matches else True
        elif self._mode == MetadataMatchMode.ANY:
            return any(matches) if matches else True
        else:  # BOOST mode - always matches
            return True
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve using hybrid metadata + content search.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            filter: Hard metadata filter (passed to underlying retriever).
            **kwargs: Additional arguments for the wrapped retriever.
            
        Returns:
            Results scored by both metadata and content relevance.
        """
        # Get more results to allow for filtering
        fetch_k = k * 3 if self._mode != MetadataMatchMode.BOOST else k * 2
        
        # Get content-based results from wrapped retriever
        content_results = await self._retriever.retrieve(
            query, k=fetch_k, filter=filter, include_scores=True, **kwargs
        )
        
        # Prepare query terms for metadata matching
        query_terms = set(query.lower().split())
        
        # Score and filter results
        hybrid_results: list[RetrievalResult] = []
        
        for result in content_results:
            doc = result.document
            content_score = result.score
            
            # Check metadata filter (for ALL/ANY modes)
            if not self._matches_metadata_filter(doc, query_terms):
                continue
            
            # Compute metadata score
            metadata_score = self._compute_metadata_score(doc, query, query_terms)
            
            # Combine scores
            combined_score = (
                self._content_weight * content_score +
                self._metadata_weight * metadata_score
            )
            
            hybrid_results.append(RetrievalResult(
                document=doc,
                score=combined_score,
                retriever_name=self.name,
                metadata={
                    "content_score": content_score,
                    "metadata_score": metadata_score,
                    "content_weight": self._content_weight,
                    "metadata_weight": self._metadata_weight,
                    "mode": self._mode.value,
                    **result.metadata,
                },
            ))
        
        # Sort by combined score and limit
        hybrid_results.sort(key=lambda r: r.score, reverse=True)
        return hybrid_results[:k]
    
    @property
    def retriever(self) -> Retriever:
        """Access the wrapped content retriever."""
        return self._retriever
    
    @property
    def metadata_fields(self) -> list[str]:
        """List of metadata fields being searched."""
        return [mw.field for mw in self._metadata_weights]
