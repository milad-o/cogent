"""Fusion utilities for combining results from multiple retrievers.

Implements various fusion strategies:
- RRF (Reciprocal Rank Fusion): Robust, parameter-free fusion
- Linear: Weighted score combination
- Max: Take maximum score per document
- Voting: Count appearances across retrievers
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from agenticflow.retriever.base import FusionStrategy, RetrievalResult

if TYPE_CHECKING:
    from agenticflow.vectorstore import Document


def fuse_results(
    result_lists: list[list[RetrievalResult]],
    strategy: FusionStrategy = FusionStrategy.RRF,
    weights: list[float] | None = None,
    k: int | None = None,
    normalize_output: bool = True,
) -> list[RetrievalResult]:
    """Fuse results from multiple retrievers.
    
    Args:
        result_lists: List of result lists from different retrievers.
        strategy: Fusion strategy to use.
        weights: Optional weights for each retriever (for LINEAR strategy).
        k: Number of results to return (None = all).
        normalize_output: Normalize final scores to 0-1 range (default: True).
        
    Returns:
        Fused and sorted results with normalized scores.
        
    Example:
        >>> results1 = await retriever1.retrieve_with_scores(query)
        >>> results2 = await retriever2.retrieve_with_scores(query)
        >>> fused = fuse_results([results1, results2], strategy=FusionStrategy.RRF)
    """
    if not result_lists:
        return []
    
    # Default equal weights
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    if strategy == FusionStrategy.RRF:
        results = _rrf_fusion(result_lists, weights, k)
    elif strategy == FusionStrategy.LINEAR:
        results = _linear_fusion(result_lists, weights, k)
    elif strategy == FusionStrategy.MAX:
        results = _max_fusion(result_lists, k)
    elif strategy == FusionStrategy.VOTING:
        results = _voting_fusion(result_lists, weights, k)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")
    
    # Normalize output scores to 0-1 range for consistency
    if normalize_output and results:
        results = normalize_scores(results)
    
    return results


def _get_doc_key(result: RetrievalResult) -> str:
    """Get a unique key for document deduplication.
    
    Uses metadata 'id' if available, otherwise falls back to text hash.
    """
    # Check for explicit ID in metadata first
    doc_id = result.document.metadata.get("id")
    if doc_id:
        return str(doc_id)
    # Fall back to text-based key
    return result.document.text


def _rrf_fusion(
    result_lists: list[list[RetrievalResult]],
    weights: list[float],
    k: int | None = None,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion.
    
    RRF score = sum(weight_i / (rrf_k + rank_i))
    
    This is robust to different score scales and generally performs well.
    The rrf_k parameter (default 60) controls the impact of rank differences.
    
    Reference: Cormack, Clarke, Buettcher (2009)
    """
    # Track RRF scores per document ID
    doc_scores: dict[str, float] = defaultdict(float)
    doc_objects: dict[str, RetrievalResult] = {}
    
    for i, results in enumerate(result_lists):
        weight = weights[i]
        for rank, result in enumerate(results, start=1):
            doc_key = _get_doc_key(result)
            
            # RRF formula
            rrf_score = weight / (rrf_k + rank)
            doc_scores[doc_key] += rrf_score
            
            # Keep the result object (use first occurrence)
            if doc_key not in doc_objects:
                doc_objects[doc_key] = result
    
    # Sort by RRF score
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    if k:
        sorted_keys = sorted_keys[:k]
    
    # Build results with fused scores
    results = []
    for doc_key in sorted_keys:
        original = doc_objects[doc_key]
        results.append(
            RetrievalResult(
                document=original.document,
                score=doc_scores[doc_key],
                retriever_name="fused_rrf",
                metadata={
                    "fusion_strategy": "rrf",
                    "original_retriever": original.retriever_name,
                },
            )
        )
    
    return results


def _linear_fusion(
    result_lists: list[list[RetrievalResult]],
    weights: list[float],
    k: int | None = None,
) -> list[RetrievalResult]:
    """Linear weighted combination of scores.
    
    Requires normalized scores (0-1) for meaningful results.
    """
    doc_scores: dict[str, float] = defaultdict(float)
    doc_objects: dict[str, RetrievalResult] = {}
    
    for i, results in enumerate(result_lists):
        weight = weights[i]
        for result in results:
            doc_key = _get_doc_key(result)
            
            # Weighted score
            doc_scores[doc_key] += weight * result.score
            
            if doc_key not in doc_objects:
                doc_objects[doc_key] = result
    
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    if k:
        sorted_keys = sorted_keys[:k]
    
    results = []
    for doc_key in sorted_keys:
        original = doc_objects[doc_key]
        results.append(
            RetrievalResult(
                document=original.document,
                score=doc_scores[doc_key],
                retriever_name="fused_linear",
                metadata={
                    "fusion_strategy": "linear",
                    "original_retriever": original.retriever_name,
                },
            )
        )
    
    return results


def _max_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int | None = None,
) -> list[RetrievalResult]:
    """Take maximum score per document across retrievers.
    
    Good when you want the best match from any retriever.
    """
    doc_scores: dict[str, float] = {}
    doc_objects: dict[str, RetrievalResult] = {}
    
    for results in result_lists:
        for result in results:
            doc_key = _get_doc_key(result)
            
            # Max score
            if doc_key not in doc_scores or result.score > doc_scores[doc_key]:
                doc_scores[doc_key] = result.score
                doc_objects[doc_key] = result
    
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    if k:
        sorted_keys = sorted_keys[:k]
    
    results = []
    for doc_key in sorted_keys:
        original = doc_objects[doc_key]
        results.append(
            RetrievalResult(
                document=original.document,
                score=doc_scores[doc_key],
                retriever_name="fused_max",
                metadata={
                    "fusion_strategy": "max",
                    "original_retriever": original.retriever_name,
                },
            )
        )
    
    return results


def _voting_fusion(
    result_lists: list[list[RetrievalResult]],
    weights: list[float],
    k: int | None = None,
) -> list[RetrievalResult]:
    """Voting-based fusion: count weighted appearances.
    
    Good for diverse retriever ensembles where agreement matters.
    """
    doc_votes: dict[str, float] = defaultdict(float)
    doc_objects: dict[str, RetrievalResult] = {}
    doc_best_score: dict[str, float] = {}
    
    for i, results in enumerate(result_lists):
        weight = weights[i]
        for result in results:
            doc_key = _get_doc_key(result)
            
            # Weighted vote
            doc_votes[doc_key] += weight
            
            # Track best score for tiebreaking
            if doc_key not in doc_best_score or result.score > doc_best_score[doc_key]:
                doc_best_score[doc_key] = result.score
                doc_objects[doc_key] = result
    
    # Sort by votes, then by best score
    sorted_keys = sorted(
        doc_votes.keys(),
        key=lambda x: (doc_votes[x], doc_best_score[x]),
        reverse=True,
    )
    
    if k:
        sorted_keys = sorted_keys[:k]
    
    results = []
    for doc_key in sorted_keys:
        original = doc_objects[doc_key]
        results.append(
            RetrievalResult(
                document=original.document,
                score=doc_votes[doc_key],  # Vote count as score
                retriever_name="fused_voting",
                metadata={
                    "fusion_strategy": "voting",
                    "votes": doc_votes[doc_key],
                    "best_score": doc_best_score[doc_key],
                    "original_retriever": original.retriever_name,
                },
            )
        )
    
    return results


def normalize_scores(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Normalize scores to 0-1 range using min-max normalization.
    
    Useful before linear fusion to ensure comparable scales.
    """
    if not results:
        return results
    
    scores = [r.score for r in results]
    min_score = min(scores)
    max_score = max(scores)
    
    # Avoid division by zero
    score_range = max_score - min_score
    if score_range == 0:
        return [
            RetrievalResult(
                document=r.document,
                score=1.0,
                retriever_name=r.retriever_name,
                metadata={**r.metadata, "raw_score": r.score},
            )
            for r in results
        ]
    
    return [
        RetrievalResult(
            document=r.document,
            score=(r.score - min_score) / score_range,
            retriever_name=r.retriever_name,
            metadata={**r.metadata, "raw_score": r.score},
        )
        for r in results
    ]


def deduplicate_results(
    results: list[RetrievalResult],
    by: str = "auto",
) -> list[RetrievalResult]:
    """Remove duplicate documents from results.
    
    Keeps the highest-scored version of each duplicate.
    
    Args:
        results: Results to deduplicate.
        by: How to identify duplicates:
            - "auto": Use document ID if available, else text
            - "id": Use document ID only
            - "text": Use document text
            
    Returns:
        Deduplicated results, sorted by score.
    """
    if not results:
        return results
    
    seen: dict[str, RetrievalResult] = {}
    
    for result in results:
        # Determine document key
        if by == "id":
            key = result.document.metadata.get("id", str(id(result.document)))
        elif by == "text":
            key = result.document.text
        else:  # auto
            key = result.document.metadata.get("id") or result.document.text
        
        # Keep higher score
        if key not in seen or result.score > seen[key].score:
            seen[key] = result
    
    # Sort by score descending
    deduped = list(seen.values())
    deduped.sort(key=lambda r: r.score, reverse=True)
    
    return deduped
