"""Tokenization utilities for sparse retrieval.

Provides various tokenization strategies for BM25 and TF-IDF.
Simple whitespace tokenization by default, with options for
more sophisticated preprocessing.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizers."""
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens."""
        ...


class BaseTokenizer(ABC):
    """Base class for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of tokens.
        """
        ...
    
    def __call__(self, text: str) -> list[str]:
        """Allow using tokenizer as callable."""
        return self.tokenize(text)


class SimpleTokenizer(BaseTokenizer):
    """Simple whitespace tokenizer with basic normalization.
    
    Splits on whitespace and non-alphanumeric characters,
    lowercases tokens, and optionally removes short tokens.
    
    Example:
        >>> tokenizer = SimpleTokenizer()
        >>> tokenizer.tokenize("Hello, World!")
        ['hello', 'world']
    """
    
    def __init__(
        self,
        *,
        lowercase: bool = True,
        min_length: int = 1,
        pattern: str = r'\b\w+\b',
    ) -> None:
        """Create a simple tokenizer.
        
        Args:
            lowercase: Whether to lowercase tokens.
            min_length: Minimum token length to keep.
            pattern: Regex pattern for tokenization.
        """
        self.lowercase = lowercase
        self.min_length = min_length
        self.pattern = re.compile(pattern)
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        if self.lowercase:
            text = text.lower()
        
        tokens = self.pattern.findall(text)
        
        if self.min_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_length]
        
        return tokens


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece-style tokenizer for subword tokenization.
    
    Splits unknown words into subwords based on a vocabulary.
    Useful for handling OOV (out-of-vocabulary) words.
    
    Note: This is a simplified implementation. For production,
    consider using tokenizers from HuggingFace.
    """
    
    def __init__(
        self,
        vocab: set[str] | None = None,
        *,
        lowercase: bool = True,
        unk_token: str = "[UNK]",
        max_word_length: int = 100,
    ) -> None:
        """Create a WordPiece tokenizer.
        
        Args:
            vocab: Set of known tokens. If None, uses simple tokenization.
            lowercase: Whether to lowercase tokens.
            unk_token: Token for unknown words.
            max_word_length: Maximum word length to process.
        """
        self.vocab = vocab or set()
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.max_word_length = max_word_length
        self._word_pattern = re.compile(r'\b\w+\b')
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using WordPiece."""
        if self.lowercase:
            text = text.lower()
        
        words = self._word_pattern.findall(text)
        
        # If no vocab, fall back to simple tokenization
        if not self.vocab:
            return words
        
        tokens = []
        for word in words:
            if len(word) > self.max_word_length:
                tokens.append(self.unk_token)
                continue
            
            subtokens = self._tokenize_word(word)
            tokens.extend(subtokens)
        
        return tokens
    
    def _tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word into subwords."""
        if word in self.vocab:
            return [word]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            found = False
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                
                end -= 1
            
            if not found:
                tokens.append(self.unk_token)
                break
            
            start = end
        
        return tokens


class StopwordTokenizer(BaseTokenizer):
    """Tokenizer with stopword removal.
    
    Removes common words that don't carry much semantic meaning.
    Useful for improving retrieval precision.
    
    Example:
        >>> tokenizer = StopwordTokenizer()
        >>> tokenizer.tokenize("The quick brown fox")
        ['quick', 'brown', 'fox']
    """
    
    # Common English stopwords
    DEFAULT_STOPWORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "been", "being",
        "but", "by", "can", "could", "do", "does", "doing", "done",
        "for", "from", "had", "has", "have", "having", "he", "her",
        "here", "him", "his", "how", "i", "if", "in", "into", "is",
        "it", "its", "just", "me", "more", "most", "my", "no", "not",
        "now", "of", "on", "only", "or", "other", "our", "out", "own",
        "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "them", "then", "there", "these", "they",
        "this", "those", "through", "to", "too", "up", "very", "was",
        "we", "were", "what", "when", "where", "which", "while", "who",
        "will", "with", "would", "you", "your",
    })
    
    def __init__(
        self,
        stopwords: set[str] | None = None,
        *,
        base_tokenizer: BaseTokenizer | None = None,
        lowercase: bool = True,
    ) -> None:
        """Create a stopword tokenizer.
        
        Args:
            stopwords: Custom stopwords. Uses defaults if None.
            base_tokenizer: Underlying tokenizer. Uses SimpleTokenizer if None.
            lowercase: Whether to lowercase (used if no base_tokenizer).
        """
        self.stopwords = stopwords or self.DEFAULT_STOPWORDS
        self.base_tokenizer = base_tokenizer or SimpleTokenizer(lowercase=lowercase)
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and remove stopwords."""
        tokens = self.base_tokenizer.tokenize(text)
        return [t for t in tokens if t.lower() not in self.stopwords]


class StemmerTokenizer(BaseTokenizer):
    """Tokenizer with Porter stemming.
    
    Reduces words to their root form for better matching.
    "running" -> "run", "computers" -> "comput"
    
    Note: For advanced stemming/lemmatization, consider using NLTK or spaCy.
    """
    
    def __init__(
        self,
        *,
        base_tokenizer: BaseTokenizer | None = None,
        lowercase: bool = True,
    ) -> None:
        """Create a stemmer tokenizer.
        
        Args:
            base_tokenizer: Underlying tokenizer.
            lowercase: Whether to lowercase.
        """
        self.base_tokenizer = base_tokenizer or SimpleTokenizer(lowercase=lowercase)
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize and stem."""
        tokens = self.base_tokenizer.tokenize(text)
        return [self._simple_stem(t) for t in tokens]
    
    def _simple_stem(self, word: str) -> str:
        """Simple suffix stripping (not full Porter stemmer).
        
        For proper stemming, use NLTK's PorterStemmer.
        """
        # Common suffixes to strip
        suffixes = [
            "ingly", "edly", "tion", "sion", "ness", "ment",
            "able", "ible", "less", "ious", "eous",
            "ing", "ely", "ful", "ive", "ize",
            "ly", "es", "ed", "er", "en", "s",
        ]
        
        original = word
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 2:
                word = word[:-len(suffix)]
                break
        
        # Avoid overly short stems
        if len(word) < 2:
            return original
        
        return word


class NGramTokenizer(BaseTokenizer):
    """Character n-gram tokenizer.
    
    Generates overlapping character sequences for fuzzy matching.
    Useful for handling typos and partial matches.
    
    Example:
        >>> tokenizer = NGramTokenizer(n=3)
        >>> tokenizer.tokenize("hello")
        ['hel', 'ell', 'llo']
    """
    
    def __init__(
        self,
        n: int = 3,
        *,
        lowercase: bool = True,
        pad: bool = False,
        pad_char: str = " ",
    ) -> None:
        """Create an n-gram tokenizer.
        
        Args:
            n: Size of n-grams.
            lowercase: Whether to lowercase text.
            pad: Whether to pad start/end for edge n-grams.
            pad_char: Character to use for padding.
        """
        self.n = n
        self.lowercase = lowercase
        self.pad = pad
        self.pad_char = pad_char
    
    def tokenize(self, text: str) -> list[str]:
        """Generate character n-grams."""
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        if self.pad:
            padding = self.pad_char * (self.n - 1)
            text = padding + text + padding
        
        ngrams = []
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            if ngram.strip():  # Skip whitespace-only n-grams
                ngrams.append(ngram)
        
        return ngrams


class CompositeTokenizer(BaseTokenizer):
    """Combines multiple tokenizers.
    
    Applies tokenizers in sequence and combines results.
    
    Example:
        >>> tokenizer = CompositeTokenizer([
        ...     SimpleTokenizer(),
        ...     NGramTokenizer(n=3),
        ... ])
    """
    
    def __init__(
        self,
        tokenizers: list[BaseTokenizer],
        *,
        deduplicate: bool = True,
    ) -> None:
        """Create a composite tokenizer.
        
        Args:
            tokenizers: List of tokenizers to apply.
            deduplicate: Whether to remove duplicate tokens.
        """
        self.tokenizers = tokenizers
        self.deduplicate = deduplicate
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize using all tokenizers."""
        all_tokens = []
        for tokenizer in self.tokenizers:
            tokens = tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        
        if self.deduplicate:
            # Preserve order while deduplicating
            seen = set()
            unique = []
            for token in all_tokens:
                if token not in seen:
                    seen.add(token)
                    unique.append(token)
            return unique
        
        return all_tokens


# Convenience factory functions

def create_tokenizer(
    style: str = "simple",
    **kwargs,
) -> BaseTokenizer:
    """Create a tokenizer by style name.
    
    Args:
        style: Tokenizer style ("simple", "stopword", "stemmer", "ngram").
        **kwargs: Additional arguments for the tokenizer.
        
    Returns:
        Tokenizer instance.
        
    Example:
        >>> tokenizer = create_tokenizer("stopword")
        >>> tokenizer.tokenize("The quick brown fox")
        ['quick', 'brown', 'fox']
    """
    styles = {
        "simple": SimpleTokenizer,
        "stopword": StopwordTokenizer,
        "stemmer": StemmerTokenizer,
        "ngram": NGramTokenizer,
        "wordpiece": WordPieceTokenizer,
    }
    
    if style not in styles:
        raise ValueError(f"Unknown tokenizer style: {style}. Available: {list(styles)}")
    
    return styles[style](**kwargs)


def tokenize_function(
    style: str = "simple",
    **kwargs,
) -> Callable[[str], list[str]]:
    """Get a tokenization function.
    
    Convenience wrapper that returns a callable instead of a class.
    
    Args:
        style: Tokenizer style.
        **kwargs: Additional arguments.
        
    Returns:
        Callable that tokenizes text.
        
    Example:
        >>> tokenize = tokenize_function("simple")
        >>> tokenize("Hello World")
        ['hello', 'world']
    """
    tokenizer = create_tokenizer(style, **kwargs)
    return tokenizer.tokenize
