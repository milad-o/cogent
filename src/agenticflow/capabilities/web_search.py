"""
WebSearch capability - web search and page fetching.

Provides tools for searching the web and fetching page content,
enabling agents to gather information from the internet.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import WebSearch
    
    agent = Agent(
        name="Researcher",
        model=model,
        capabilities=[WebSearch()],
    )
    
    # Agent can now search and fetch web content
    await agent.run("Search for the latest Python 3.13 features")
    await agent.run("Fetch the content from https://python.org")
    ```
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from agenticflow.tools.base import BaseTool, tool

from agenticflow.capabilities.base import BaseCapability


@dataclass
class SearchResult:
    """A single search result."""
    
    title: str
    url: str
    snippet: str
    source: str = ""  # Search provider
    position: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "position": self.position,
        }


@dataclass
class FetchedPage:
    """Content fetched from a URL."""
    
    url: str
    title: str
    content: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    content_type: str = "text/html"
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content[:1000] + "..." if len(self.content) > 1000 else self.content,
            "fetched_at": self.fetched_at.isoformat(),
            "content_type": self.content_type,
            "error": self.error,
        }


class SearchProvider(ABC):
    """Abstract base for search providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Execute a search query."""
        ...
    
    @abstractmethod
    def news(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search news articles."""
        ...


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider using ddgs package."""
    
    def __init__(self, timeout: int = 10):
        self._timeout = timeout
        self._ddgs = None
    
    @property
    def name(self) -> str:
        return "duckduckgo"
    
    def _get_client(self):
        """Lazy load DDGS client."""
        if self._ddgs is None:
            try:
                from ddgs import DDGS
                self._ddgs = DDGS(timeout=self._timeout)
            except ImportError:
                raise ImportError(
                    "ddgs package required. Install with: uv add ddgs"
                )
        return self._ddgs
    
    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search web using DuckDuckGo."""
        ddgs = self._get_client()
        results = []
        
        try:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                    source=self.name,
                    position=i + 1,
                ))
        except Exception as e:
            # Return empty results on error, let caller handle
            pass
        
        return results
    
    def news(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search news using DuckDuckGo."""
        ddgs = self._get_client()
        results = []
        
        try:
            for i, r in enumerate(ddgs.news(query, max_results=max_results)):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", r.get("link", "")),
                    snippet=r.get("body", r.get("excerpt", "")),
                    source=self.name,
                    position=i + 1,
                ))
        except Exception:
            pass
        
        return results


class WebSearch(BaseCapability):
    """
    WebSearch capability for searching the web and fetching pages.
    
    Provides tools for:
    - Web search (DuckDuckGo by default, free, no API key needed)
    - News search
    - Page content fetching
    - URL validation
    
    Args:
        provider: Search provider to use (default: DuckDuckGo)
        max_results: Default max search results (default: 10)
        fetch_timeout: Timeout for fetching pages in seconds (default: 10)
        max_content_length: Max characters to return from fetched pages (default: 50000)
        user_agent: Custom user agent for fetching (optional)
        name: Capability name (default: "web_search")
    
    Example:
        ```python
        # Basic usage with DuckDuckGo (free, no API key)
        ws = WebSearch()
        
        # Custom settings
        ws = WebSearch(
            max_results=5,
            fetch_timeout=15,
            max_content_length=100000,
        )
        ```
    """
    
    def __init__(
        self,
        provider: SearchProvider | None = None,
        max_results: int = 10,
        fetch_timeout: int = 10,
        max_content_length: int = 50000,
        user_agent: str | None = None,
        name: str = "web_search",
    ):
        self._name = name
        self._provider = provider or DuckDuckGoProvider()
        self._max_results = max_results
        self._fetch_timeout = fetch_timeout
        self._max_content_length = max_content_length
        self._user_agent = user_agent or (
            "Mozilla/5.0 (compatible; AgenticFlow/1.0; +https://github.com/agenticflow)"
        )
        self._tools_cache: list[BaseTool] | None = None
        
        # Simple in-memory cache for fetched pages
        self._page_cache: dict[str, FetchedPage] = {}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return f"Web search and page fetching using {self._provider.name}"
    
    @property
    def tools(self) -> list[BaseTool]:
        if self._tools_cache is None:
            self._tools_cache = [
                self._search_tool(),
                self._news_search_tool(),
                self._fetch_page_tool(),
            ]
        return self._tools_cache
    
    @property
    def provider(self) -> SearchProvider:
        """Get the search provider."""
        return self._provider
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """
        Search the web.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        return self._provider.search(
            query,
            max_results=max_results or self._max_results,
        )
    
    def search_news(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """
        Search news articles.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        return self._provider.news(
            query,
            max_results=max_results or self._max_results,
        )
    
    def fetch(
        self,
        url: str,
        use_cache: bool = True,
    ) -> FetchedPage:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            use_cache: Whether to use cached content
            
        Returns:
            FetchedPage object with content
        """
        # Validate URL
        if not self._is_valid_url(url):
            return FetchedPage(
                url=url,
                title="",
                content="",
                error=f"Invalid URL: {url}",
            )
        
        # Check cache
        if use_cache and url in self._page_cache:
            return self._page_cache[url]
        
        # Fetch the page
        page = self._fetch_url(url)
        
        # Cache successful fetches
        if page.error is None:
            self._page_cache[url] = page
        
        return page
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False
    
    def _fetch_url(self, url: str) -> FetchedPage:
        """Fetch URL content."""
        try:
            import httpx
            
            with httpx.Client(
                timeout=self._fetch_timeout,
                follow_redirects=True,
                headers={"User-Agent": self._user_agent},
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                
                content_type = response.headers.get("content-type", "")
                
                # Get raw content
                raw_content = response.text
                
                # Extract text content
                if "text/html" in content_type:
                    title, content = self._extract_html_content(raw_content)
                else:
                    title = ""
                    content = raw_content
                
                # Truncate if needed
                if len(content) > self._max_content_length:
                    content = content[:self._max_content_length] + "\n\n[Content truncated...]"
                
                return FetchedPage(
                    url=url,
                    title=title,
                    content=content,
                    content_type=content_type,
                )
                
        except ImportError:
            return FetchedPage(
                url=url,
                title="",
                content="",
                error="httpx package required. Install with: uv add httpx",
            )
        except Exception as e:
            return FetchedPage(
                url=url,
                title="",
                content="",
                error=str(e),
            )
    
    def _extract_html_content(self, html: str) -> tuple[str, str]:
        """Extract title and text content from HTML."""
        # Try BeautifulSoup first
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Get title
            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = "\n".join(lines)
            
            return title, content
            
        except ImportError:
            # Fallback to regex-based extraction
            return self._extract_html_regex(html)
    
    def _extract_html_regex(self, html: str) -> tuple[str, str]:
        """Fallback HTML extraction using regex."""
        # Extract title
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        
        # Remove script and style blocks
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Decode common HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        
        return title, text
    
    def clear_cache(self) -> int:
        """Clear the page cache. Returns number of items cleared."""
        count = len(self._page_cache)
        self._page_cache.clear()
        return count
    
    # =========================================================================
    # Tool Generation
    # =========================================================================
    
    def _search_tool(self) -> BaseTool:
        ws = self
        
        @tool
        def web_search(query: str, max_results: int = 10) -> str:
            """
            Search the web for information.
            
            Args:
                query: The search query (be specific for better results)
                max_results: Maximum number of results to return (default: 10)
            
            Returns:
                Search results with titles, URLs, and snippets
            """
            results = ws.search(query, max_results=max_results)
            
            if not results:
                return f"No results found for: {query}"
            
            lines = [f"Search results for '{query}' ({len(results)} results):"]
            for r in results:
                lines.append(f"\n{r.position}. {r.title}")
                lines.append(f"   URL: {r.url}")
                lines.append(f"   {r.snippet[:200]}...")
            
            return "\n".join(lines)
        
        return web_search
    
    def _news_search_tool(self) -> BaseTool:
        ws = self
        
        @tool
        def news_search(query: str, max_results: int = 10) -> str:
            """
            Search for recent news articles.
            
            Args:
                query: The news search query
                max_results: Maximum number of results to return (default: 10)
            
            Returns:
                News articles with titles, URLs, and excerpts
            """
            results = ws.search_news(query, max_results=max_results)
            
            if not results:
                return f"No news found for: {query}"
            
            lines = [f"News results for '{query}' ({len(results)} results):"]
            for r in results:
                lines.append(f"\n{r.position}. {r.title}")
                lines.append(f"   URL: {r.url}")
                lines.append(f"   {r.snippet[:200]}...")
            
            return "\n".join(lines)
        
        return news_search
    
    def _fetch_page_tool(self) -> BaseTool:
        ws = self
        
        @tool
        def fetch_webpage(url: str) -> str:
            """
            Fetch and extract text content from a webpage.
            
            Use this to read the full content of a webpage after finding 
            it via search. Works best with article/content pages.
            
            Args:
                url: The full URL to fetch (must start with http:// or https://)
            
            Returns:
                The extracted text content from the page
            """
            page = ws.fetch(url)
            
            if page.error:
                return f"Error fetching {url}: {page.error}"
            
            lines = [f"Fetched: {page.title or url}"]
            lines.append(f"URL: {page.url}")
            lines.append("-" * 40)
            lines.append(page.content)
            
            return "\n".join(lines)
        
        return fetch_webpage
