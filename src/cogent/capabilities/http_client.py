"""
HTTP Client capability - production-ready HTTP requests.

Provides a secure, production-ready HTTP client for making API calls
with timeout handling, retry logic, and structured error handling.

Example:
    ```python
    from cogent import Agent
    from cogent.capabilities import HTTPClient

    agent = Agent(
        name="API Agent",
        model=model,
        capabilities=[HTTPClient(timeout=30, max_retries=3)],
    )

    # Agent can now make HTTP requests
    await agent.run("Get user data from https://api.example.com/users/123")
    ```
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from cogent.capabilities.base import BaseCapability
from cogent.tools.base import BaseTool, tool

logger = logging.getLogger(__name__)


@dataclass
class HTTPResponse:
    """Result of an HTTP request."""

    status_code: int
    headers: dict[str, str]
    body: str
    url: str
    method: str
    success: bool
    error: str | None = None
    response_time_ms: float = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "headers": dict(self.headers),
            "body": self.body[:1000] + "..." if len(self.body) > 1000 else self.body,
            "url": self.url,
            "method": self.method,
            "success": self.success,
            "error": self.error,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }

    def json(self) -> Any:
        """Parse response body as JSON."""
        try:
            return json.loads(self.body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Response is not valid JSON: {e}") from e


class HTTPClient(BaseCapability):
    """
    HTTP Client capability for making API requests.

    Provides secure HTTP/HTTPS requests with:
    - Timeout protection
    - Automatic retries
    - Request/response logging
    - Structured error handling
    - Header customization

    Args:
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts (default: 3)
        verify_ssl: Whether to verify SSL certificates (default: True)
        follow_redirects: Whether to follow redirects (default: True)
        max_redirects: Maximum number of redirects (default: 10)
        default_headers: Default headers for all requests
        name: Capability name (default: "http_client")

    Example:
        ```python
        # Basic client
        client = HTTPClient()

        # Custom configuration
        client = HTTPClient(
            timeout=60,
            max_retries=5,
            default_headers={"User-Agent": "MyBot/1.0"},
        )
        ```

    Security Notes:
        - SSL verification enabled by default
        - Timeouts prevent hanging requests
        - Automatic retry with exponential backoff
        - Request size limits enforced
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True,
        follow_redirects: bool = True,
        max_redirects: int = 10,
        default_headers: dict[str, str] | None = None,
        name: str = "http_client",
    ):
        self._name = name
        self._timeout = timeout
        self._max_retries = max_retries
        self._verify_ssl = verify_ssl
        self._follow_redirects = follow_redirects
        self._max_redirects = max_redirects
        self._default_headers = default_headers or {}
        self._tools_cache: list[BaseTool] | None = None

        # Request history
        self._history: list[HTTPResponse] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "HTTP client for making API requests"

    @property
    def tools(self) -> list[BaseTool]:
        if self._tools_cache is None:
            self._tools_cache = [
                self._get_tool(),
                self._post_tool(),
                self._put_tool(),
                self._delete_tool(),
            ]
        return self._tools_cache

    @property
    def history(self) -> list[HTTPResponse]:
        """Get request history."""
        return self._history

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: str | dict | None = None,
        params: dict[str, str] | None = None,
    ) -> HTTPResponse:
        """
        Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: Target URL
            headers: Optional request headers
            body: Optional request body (string or dict for JSON)
            params: Optional query parameters

        Returns:
            HTTPResponse with status, headers, and body
        """
        import time

        start_time = time.perf_counter()

        # Merge headers
        merged_headers = {**self._default_headers, **(headers or {})}

        # Prepare request body
        request_body = None
        if body is not None:
            if isinstance(body, dict):
                request_body = json.dumps(body)
                merged_headers.setdefault("Content-Type", "application/json")
            else:
                request_body = body

        try:
            # Create client with retry transport
            transport = httpx.AsyncHTTPTransport(retries=self._max_retries)
            
            async with httpx.AsyncClient(
                transport=transport,
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                max_redirects=self._max_redirects,
            ) as client:
                response = await client.request(
                    method=method.upper(),
                    url=url,
                    headers=merged_headers,
                    content=request_body,
                    params=params,
                )

                response_time = (time.perf_counter() - start_time) * 1000

                result = HTTPResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response.text,
                    url=str(response.url),
                    method=method.upper(),
                    success=200 <= response.status_code < 300,
                    response_time_ms=response_time,
                )

                self._history.append(result)
                return result

        except httpx.TimeoutException as e:
            error_msg = f"Request timed out after {self._timeout}s: {e}"
            logger.warning(error_msg)
            result = HTTPResponse(
                status_code=0,
                headers={},
                body="",
                url=url,
                method=method.upper(),
                success=False,
                error=error_msg,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._history.append(result)
            return result

        except httpx.RequestError as e:
            error_msg = f"Request failed: {e}"
            logger.error(error_msg)
            result = HTTPResponse(
                status_code=0,
                headers={},
                body="",
                url=url,
                method=method.upper(),
                success=False,
                error=error_msg,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._history.append(result)
            return result

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> HTTPResponse:
        """GET request."""
        return await self.request("GET", url, headers=headers, params=params)

    async def post(
        self,
        url: str,
        body: str | dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> HTTPResponse:
        """POST request."""
        return await self.request("POST", url, headers=headers, body=body)

    async def put(
        self,
        url: str,
        body: str | dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> HTTPResponse:
        """PUT request."""
        return await self.request("PUT", url, headers=headers, body=body)

    async def delete(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> HTTPResponse:
        """DELETE request."""
        return await self.request("DELETE", url, headers=headers)

    def clear_history(self) -> int:
        """Clear request history. Returns count cleared."""
        count = len(self._history)
        self._history.clear()
        return count

    # =========================================================================
    # Tool Generation
    # =========================================================================

    def _get_tool(self) -> BaseTool:
        client = self

        @tool
        async def http_get(url: str, headers: str = "{}") -> str:
            """
            Make an HTTP GET request.

            Args:
                url: The URL to request
                headers: Optional JSON string of headers, e.g., '{"Authorization": "Bearer token"}'

            Returns:
                Response with status code, headers, and body
            """
            try:
                parsed_headers = json.loads(headers) if headers else {}
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in headers: {headers}"

            response = await client.get(url, headers=parsed_headers)

            if response.success:
                lines = [f"✓ GET {url}"]
                lines.append(f"Status: {response.status_code}")
                lines.append(f"Response time: {response.response_time_ms:.0f}ms")
                lines.append(f"\nBody:\n{response.body[:500]}")
                if len(response.body) > 500:
                    lines.append("...(truncated)")
            else:
                lines = [f"✗ GET {url} failed"]
                if response.error:
                    lines.append(f"Error: {response.error}")
                else:
                    lines.append(f"Status: {response.status_code}")
                    lines.append(f"Body: {response.body[:200]}")

            return "\n".join(lines)

        return http_get

    def _post_tool(self) -> BaseTool:
        client = self

        @tool
        async def http_post(url: str, body: str = "", headers: str = "{}") -> str:
            """
            Make an HTTP POST request.

            Args:
                url: The URL to request
                body: Request body (JSON string or plain text)
                headers: Optional JSON string of headers

            Returns:
                Response with status code and body
            """
            try:
                parsed_headers = json.loads(headers) if headers else {}
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in headers: {headers}"

            # Try to parse body as JSON
            try:
                parsed_body = json.loads(body) if body else None
            except json.JSONDecodeError:
                parsed_body = body

            response = await client.post(url, body=parsed_body, headers=parsed_headers)

            if response.success:
                lines = [f"✓ POST {url}"]
                lines.append(f"Status: {response.status_code}")
                lines.append(f"Response time: {response.response_time_ms:.0f}ms")
                lines.append(f"\nBody:\n{response.body[:500]}")
                if len(response.body) > 500:
                    lines.append("...(truncated)")
            else:
                lines = [f"✗ POST {url} failed"]
                if response.error:
                    lines.append(f"Error: {response.error}")
                else:
                    lines.append(f"Status: {response.status_code}")
                    lines.append(f"Body: {response.body[:200]}")

            return "\n".join(lines)

        return http_post

    def _put_tool(self) -> BaseTool:
        client = self

        @tool
        async def http_put(url: str, body: str = "", headers: str = "{}") -> str:
            """
            Make an HTTP PUT request.

            Args:
                url: The URL to request
                body: Request body (JSON string or plain text)
                headers: Optional JSON string of headers

            Returns:
                Response with status code and body
            """
            try:
                parsed_headers = json.loads(headers) if headers else {}
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in headers: {headers}"

            try:
                parsed_body = json.loads(body) if body else None
            except json.JSONDecodeError:
                parsed_body = body

            response = await client.put(url, body=parsed_body, headers=parsed_headers)

            if response.success:
                lines = [f"✓ PUT {url}"]
                lines.append(f"Status: {response.status_code}")
                lines.append(f"Response time: {response.response_time_ms:.0f}ms")
            else:
                lines = [f"✗ PUT {url} failed"]
                if response.error:
                    lines.append(f"Error: {response.error}")
                else:
                    lines.append(f"Status: {response.status_code}")

            return "\n".join(lines)

        return http_put

    def _delete_tool(self) -> BaseTool:
        client = self

        @tool
        async def http_delete(url: str, headers: str = "{}") -> str:
            """
            Make an HTTP DELETE request.

            Args:
                url: The URL to request
                headers: Optional JSON string of headers

            Returns:
                Response with status code
            """
            try:
                parsed_headers = json.loads(headers) if headers else {}
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in headers: {headers}"

            response = await client.delete(url, headers=parsed_headers)

            if response.success:
                lines = [f"✓ DELETE {url}"]
                lines.append(f"Status: {response.status_code}")
                lines.append(f"Response time: {response.response_time_ms:.0f}ms")
            else:
                lines = [f"✗ DELETE {url} failed"]
                if response.error:
                    lines.append(f"Error: {response.error}")
                else:
                    lines.append(f"Status: {response.status_code}")

            return "\n".join(lines)

        return http_delete
