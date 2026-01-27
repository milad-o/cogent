"""Utility functions for retrievers."""

from cogent.retriever.utils.fusion import (
    FusionStrategy,
    deduplicate_results,
    fuse_results,
    normalize_scores,
)
from cogent.retriever.utils.llm_adapter import (
    ChatModelAdapter,
    LLMProtocol,
    adapt_llm,
)
from cogent.retriever.utils.results import (
    add_citations,
    filter_by_score,
    format_citations_reference,
    format_context,
    top_k,
)
from cogent.retriever.utils.tokenizers import (
    BaseTokenizer,
    CompositeTokenizer,
    NGramTokenizer,
    SimpleTokenizer,
    StemmerTokenizer,
    StopwordTokenizer,
    Tokenizer,
    WordPieceTokenizer,
    create_tokenizer,
    tokenize_function,
)

__all__ = [
    # Fusion
    "FusionStrategy",
    "deduplicate_results",
    "fuse_results",
    "normalize_scores",
    # LLM adaptation
    "ChatModelAdapter",
    "LLMProtocol",
    "adapt_llm",
    # Result processing
    "add_citations",
    "filter_by_score",
    "format_citations_reference",
    "format_context",
    "top_k",
    # Tokenizers
    "Tokenizer",
    "BaseTokenizer",
    "SimpleTokenizer",
    "StopwordTokenizer",
    "StemmerTokenizer",
    "NGramTokenizer",
    "WordPieceTokenizer",
    "CompositeTokenizer",
    "create_tokenizer",
    "tokenize_function",
]
