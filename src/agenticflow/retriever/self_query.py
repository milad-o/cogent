"""Self-query retriever with LLM-based filter generation.

Uses an LLM to parse natural language queries into
structured filters and semantic search queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.retriever.base import BaseRetriever, RetrievalResult
from agenticflow.vectorstore import Document

if TYPE_CHECKING:
    from agenticflow.models import Model
    from agenticflow.vectorstore import VectorStore


@dataclass
class AttributeInfo:
    """Description of a filterable document attribute."""
    
    name: str
    description: str
    type: str  # "string", "integer", "float", "boolean", "date"
    
    
@dataclass 
class ParsedQuery:
    """Result of parsing a natural language query."""
    
    semantic_query: str
    filter: dict[str, Any] | None = None
    
    
DEFAULT_PARSE_PROMPT = '''You are a query parser. Given a user query and available attributes, extract:
1. A semantic search query (the main topic/content to search for)
2. Metadata filters based on the attributes

Available attributes:
{attributes}

User query: {query}

Respond in JSON format:
{{
    "semantic_query": "the content search query",
    "filter": {{"attribute_name": "value"}} or null
}}

Rules:
- Only use attributes that are explicitly mentioned or clearly implied
- For numeric comparisons, use: {{"attr": {{"$gt": value}}}} or {{"$lt": value}}
- For string matching, use exact values
- If no filter is needed, set filter to null
- The semantic_query should capture the meaning/topic, not filtering criteria'''


class SelfQueryRetriever(BaseRetriever):
    """Retriever that uses LLM to generate filters from natural language.
    
    Parses natural language queries into structured metadata filters
    and semantic search queries for more precise retrieval.
    
    Example:
        >>> from agenticflow.models import OpenAIModel
        >>> 
        >>> retriever = SelfQueryRetriever(
        ...     vectorstore=vectorstore,
        ...     llm=OpenAIModel(model="gpt-4o-mini"),
        ...     attribute_info=[
        ...         AttributeInfo("category", "Document category", "string"),
        ...         AttributeInfo("year", "Publication year", "integer"),
        ...     ],
        ... )
        >>> 
        >>> # Natural language query with implicit filters
        >>> results = await retriever.retrieve(
        ...     "research papers about AI from 2023"
        ... )
        >>> # LLM extracts: semantic="AI research papers", filter={"year": 2023}
    """
    
    _name: str = "self_query"
    
    def __init__(
        self,
        vectorstore: VectorStore,
        llm: Model,
        attribute_info: list[AttributeInfo],
        *,
        prompt_template: str | None = None,
        k: int = 4,
        enable_filter: bool = True,
        name: str | None = None,
    ) -> None:
        """Create a self-query retriever.
        
        Args:
            vectorstore: Vector store for search.
            llm: LLM for query parsing.
            attribute_info: Descriptions of filterable attributes.
            prompt_template: Custom prompt for query parsing.
            k: Default number of results.
            enable_filter: Whether to apply parsed filters.
            name: Optional custom name.
        """
        self._vectorstore = vectorstore
        self._llm = llm
        self._attribute_info = attribute_info
        self._prompt_template = prompt_template or DEFAULT_PARSE_PROMPT
        self._k = k
        self._enable_filter = enable_filter
        
        if name:
            self._name = name
    
    def _format_attributes(self) -> str:
        """Format attribute info for prompt."""
        lines = []
        for attr in self._attribute_info:
            lines.append(f"- {attr.name} ({attr.type}): {attr.description}")
        return "\n".join(lines)
    
    async def _parse_query(self, query: str) -> ParsedQuery:
        """Use LLM to parse query into semantic query and filters."""
        prompt = self._prompt_template.format(
            attributes=self._format_attributes(),
            query=query,
        )
        
        response = await self._llm.generate(prompt)
        
        # Parse JSON response
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            # Fallback: use original query, no filter
            return ParsedQuery(semantic_query=query, filter=None)
        
        try:
            parsed = json.loads(json_match.group())
            return ParsedQuery(
                semantic_query=parsed.get("semantic_query", query),
                filter=parsed.get("filter"),
            )
        except json.JSONDecodeError:
            return ParsedQuery(semantic_query=query, filter=None)
    
    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[Document]:
        """Retrieve documents using LLM-parsed query.
        
        Args:
            query: Natural language query.
            k: Number of results.
            filter: Additional filter to merge with parsed filter.
            
        Returns:
            Matching documents.
        """
        k = k or self._k
        
        # Parse query
        parsed = await self._parse_query(query)
        
        # Combine filters
        combined_filter = None
        if self._enable_filter:
            if parsed.filter and filter:
                combined_filter = {**parsed.filter, **filter}
            elif parsed.filter:
                combined_filter = parsed.filter
            elif filter:
                combined_filter = filter
        elif filter:
            combined_filter = filter
        
        # Search with parsed query and filter
        results = await self._vectorstore.search(
            query=parsed.semantic_query,
            k=k,
            filter=combined_filter,
        )
        
        return [result.document for result in results]
    
    async def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents with scores using LLM-parsed query.
        
        Args:
            query: Natural language query.
            k: Number of results.
            filter: Additional filter to merge with parsed filter.
            
        Returns:
            Retrieval results with scores.
        """
        k = k or self._k
        
        # Parse query
        parsed = await self._parse_query(query)
        
        # Combine filters
        combined_filter = None
        if self._enable_filter:
            if parsed.filter and filter:
                combined_filter = {**parsed.filter, **filter}
            elif parsed.filter:
                combined_filter = parsed.filter
            elif filter:
                combined_filter = filter
        elif filter:
            combined_filter = filter
        
        # Search
        results = await self._vectorstore.search(
            query=parsed.semantic_query,
            k=k,
            filter=combined_filter,
        )
        
        return [
            RetrievalResult(
                document=result.document,
                score=result.score,
                retriever_name=self.name,
                metadata={
                    "original_query": query,
                    "parsed_query": parsed.semantic_query,
                    "parsed_filter": parsed.filter,
                    "applied_filter": combined_filter,
                },
            )
            for result in results
        ]
    
    async def retrieve_verbose(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> tuple[list[RetrievalResult], ParsedQuery]:
        """Retrieve with full parsing information.
        
        Returns both results and the parsed query for debugging.
        
        Args:
            query: Natural language query.
            k: Number of results.
            filter: Additional filter.
            
        Returns:
            Tuple of (results, parsed_query).
        """
        k = k or self._k
        
        parsed = await self._parse_query(query)
        
        combined_filter = None
        if self._enable_filter:
            if parsed.filter and filter:
                combined_filter = {**parsed.filter, **filter}
            elif parsed.filter:
                combined_filter = parsed.filter
            elif filter:
                combined_filter = filter
        elif filter:
            combined_filter = filter
        
        results = await self._vectorstore.search(
            query=parsed.semantic_query,
            k=k,
            filter=combined_filter,
        )
        
        retrieval_results = [
            RetrievalResult(
                document=result.document,
                score=result.score,
                retriever_name=self.name,
                metadata={
                    "original_query": query,
                    "parsed_query": parsed.semantic_query,
                    "parsed_filter": parsed.filter,
                },
            )
            for result in results
        ]
        
        return retrieval_results, parsed
