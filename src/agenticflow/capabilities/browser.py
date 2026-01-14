"""
Browser capability - headless browser automation.

Provides tools for web scraping, testing, and browser automation using
Playwright, enabling agents to interact with dynamic web pages.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import Browser
    
    agent = Agent(
        name="WebScraper",
        model=model,
        capabilities=[Browser()],
    )
    
    # Agent can now automate browsers
    await agent.run("Go to example.com and screenshot the page")
    await agent.run("Fill out the login form with test credentials")
    ```
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agenticflow.capabilities.base import BaseCapability
from agenticflow.tools.base import tool


@dataclass
class PageInfo:
    """Information about a web page."""
    
    url: str
    title: str
    status: int | None = None
    content_type: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "status": self.status,
            "content_type": self.content_type,
        }


@dataclass
class ScreenshotResult:
    """Result of taking a screenshot."""
    
    path: str | None
    base64_data: str | None = None
    width: int = 0
    height: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "has_base64": self.base64_data is not None,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ExtractResult:
    """Result of extracting content from a page."""
    
    text: str
    html: str | None = None
    links: list[dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:5000] + "..." if len(self.text) > 5000 else self.text,
            "html_length": len(self.html) if self.html else 0,
            "links": self.links[:50],
            "truncated": len(self.text) > 5000,
        }


class Browser(BaseCapability):
    """
    Browser capability for headless browser automation.
    
    Uses Playwright to provide powerful browser automation including:
    - Navigation and page interaction
    - Screenshots and PDF generation
    - Form filling and clicking
    - JavaScript execution
    - Content extraction
    
    Args:
        headless: Run browser in headless mode (default: True)
        browser_type: Browser to use: "chromium", "firefox", "webkit" (default: "chromium")
        timeout_ms: Default timeout for operations in milliseconds (default: 30000)
        allowed_domains: List of allowed domains (empty = all allowed)
        blocked_domains: List of blocked domains
        screenshot_dir: Directory for saving screenshots
        user_agent: Custom user agent string
        
    Tools provided:
        - navigate: Go to a URL
        - screenshot: Take a screenshot
        - extract_text: Extract text content from page
        - extract_links: Extract all links from page
        - click: Click an element
        - fill: Fill a form field
        - select: Select dropdown option
        - evaluate: Execute JavaScript
        - wait_for: Wait for element or condition
        - get_page_info: Get current page info
    
    Note:
        Requires playwright: pip install playwright
        Then run: playwright install
    """
    
    @property
    def name(self) -> str:
        """Unique name for this capability."""
        return "browser"
    
    @property
    def tools(self) -> list:
        """Tools this capability provides to the agent."""
        return self.get_tools()
    
    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        timeout_ms: int = 30000,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        screenshot_dir: str | Path | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.headless = headless
        self.browser_type = browser_type
        self.timeout_ms = timeout_ms
        self.allowed_domains = allowed_domains or []
        self.blocked_domains = blocked_domains or []
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self.user_agent = user_agent
        
        # Browser state (initialized lazily)
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        
        # Check for playwright
        import importlib.util
        self._has_playwright = importlib.util.find_spec("playwright") is not None
    
    def _require_playwright(self) -> None:
        """Ensure playwright is available."""
        if not self._has_playwright:
            msg = "playwright not installed. Install with: pip install playwright && playwright install"
            raise ImportError(msg)
    
    def _validate_url(self, url: str) -> str:
        """Validate URL against allowed/blocked domains."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check blocked domains
        for blocked in self.blocked_domains:
            if blocked.lower() in domain:
                msg = f"Domain {domain} is blocked"
                raise PermissionError(msg)
        
        # Check allowed domains (if specified)
        if self.allowed_domains:
            allowed = False
            for allow in self.allowed_domains:
                if allow.lower() in domain:
                    allowed = True
                    break
            if not allowed:
                msg = f"Domain {domain} is not in allowed list"
                raise PermissionError(msg)
        
        return url
    
    async def _ensure_browser(self) -> Any:
        """Ensure browser is initialized."""
        self._require_playwright()
        
        if self._page is not None:
            return self._page
        
        from playwright.async_api import async_playwright
        
        self._playwright = await async_playwright().start()
        
        # Launch browser
        if self.browser_type == "firefox":
            self._browser = await self._playwright.firefox.launch(headless=self.headless)
        elif self.browser_type == "webkit":
            self._browser = await self._playwright.webkit.launch(headless=self.headless)
        else:
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
        
        # Create context with options
        context_options = {}
        if self.user_agent:
            context_options["user_agent"] = self.user_agent
        
        self._context = await self._browser.new_context(**context_options)
        self._context.set_default_timeout(self.timeout_ms)
        
        self._page = await self._context.new_page()
        return self._page
    
    async def _close_browser(self) -> None:
        """Close browser and cleanup."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
    
    async def __aenter__(self) -> "Browser":
        """Async context manager entry."""
        await self._ensure_browser()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self._close_browser()
    
    # =========================================================================
    # Browser Operations
    # =========================================================================
    
    async def _navigate(self, url: str, wait_until: str = "load") -> PageInfo:
        """Navigate to a URL."""
        url = self._validate_url(url)
        page = await self._ensure_browser()
        
        response = await page.goto(url, wait_until=wait_until)
        
        return PageInfo(
            url=page.url,
            title=await page.title(),
            status=response.status if response else None,
            content_type=response.headers.get("content-type") if response else None,
        )
    
    async def _screenshot(
        self,
        path: str | None = None,
        full_page: bool = False,
        selector: str | None = None,
    ) -> ScreenshotResult:
        """Take a screenshot."""
        page = await self._ensure_browser()
        
        options: dict[str, Any] = {"full_page": full_page}
        
        if path:
            if self.screenshot_dir:
                path = str(self.screenshot_dir / path)
            options["path"] = path
        
        if selector:
            element = await page.query_selector(selector)
            if element:
                data = await element.screenshot(**options)
            else:
                raise ValueError(f"Element not found: {selector}")
        else:
            data = await page.screenshot(**options)
        
        # Get viewport size
        viewport = page.viewport_size or {"width": 0, "height": 0}
        
        return ScreenshotResult(
            path=path,
            base64_data=base64.b64encode(data).decode() if not path else None,
            width=viewport["width"],
            height=viewport["height"],
        )
    
    async def _extract_text(self, selector: str | None = None) -> str:
        """Extract text content from page or element."""
        page = await self._ensure_browser()
        
        if selector:
            element = await page.query_selector(selector)
            if element:
                return await element.inner_text()
            return ""
        
        return await page.inner_text("body")
    
    async def _extract_links(self) -> list[dict[str, str]]:
        """Extract all links from page."""
        page = await self._ensure_browser()
        
        links = await page.eval_on_selector_all(
            "a[href]",
            """elements => elements.map(el => ({
                text: el.innerText.trim().substring(0, 100),
                href: el.href,
                title: el.title || ''
            }))"""
        )
        
        return links
    
    async def _click(self, selector: str) -> bool:
        """Click an element."""
        page = await self._ensure_browser()
        await page.click(selector)
        return True
    
    async def _fill(self, selector: str, value: str) -> bool:
        """Fill a form field."""
        page = await self._ensure_browser()
        await page.fill(selector, value)
        return True
    
    async def _select(self, selector: str, value: str) -> bool:
        """Select a dropdown option."""
        page = await self._ensure_browser()
        await page.select_option(selector, value)
        return True
    
    async def _evaluate(self, script: str) -> Any:
        """Execute JavaScript on the page."""
        page = await self._ensure_browser()
        return await page.evaluate(script)
    
    async def _wait_for(
        self,
        selector: str | None = None,
        state: str = "visible",
        timeout_ms: int | None = None,
    ) -> bool:
        """Wait for element or condition."""
        page = await self._ensure_browser()
        timeout = timeout_ms or self.timeout_ms
        
        if selector:
            await page.wait_for_selector(selector, state=state, timeout=timeout)
        else:
            await page.wait_for_load_state("networkidle", timeout=timeout)
        
        return True
    
    # =========================================================================
    # Tool Methods
    # =========================================================================
    
    def get_tools(self) -> list:
        """Return the tools provided by this capability."""
        
        @tool
        async def navigate(
            url: str,
            wait_until: str = "load",
        ) -> dict[str, Any]:
            """Navigate to a URL in the browser.
            
            Args:
                url: The URL to navigate to
                wait_until: When to consider navigation complete:
                    - "load": Wait for load event (default)
                    - "domcontentloaded": Wait for DOMContentLoaded
                    - "networkidle": Wait for network to be idle
            
            Returns:
                Dictionary with URL, title, status code, and content type
            """
            result = await self._navigate(url, wait_until)
            return result.to_dict()
        
        @tool
        async def screenshot(
            path: str | None = None,
            full_page: bool = False,
            selector: str | None = None,
        ) -> dict[str, Any]:
            """Take a screenshot of the current page or an element.
            
            Args:
                path: File path to save screenshot (optional, returns base64 if not provided)
                full_page: Capture full scrollable page (default: False)
                selector: CSS selector of element to screenshot (optional)
            
            Returns:
                Dictionary with path, dimensions, and optionally base64 data
            """
            result = await self._screenshot(path, full_page, selector)
            return result.to_dict()
        
        @tool
        async def extract_text(
            selector: str | None = None,
        ) -> dict[str, Any]:
            """Extract text content from the page or a specific element.
            
            Args:
                selector: CSS selector of element (optional, extracts body if not provided)
            
            Returns:
                Dictionary with extracted text
            """
            text = await self._extract_text(selector)
            return {
                "text": text[:10000] + "..." if len(text) > 10000 else text,
                "length": len(text),
                "truncated": len(text) > 10000,
            }
        
        @tool
        async def extract_links() -> dict[str, Any]:
            """Extract all links from the current page.
            
            Returns:
                Dictionary with list of links (text, href, title)
            """
            links = await self._extract_links()
            return {
                "links": links[:100],
                "total": len(links),
                "truncated": len(links) > 100,
            }
        
        @tool
        async def click(selector: str) -> dict[str, Any]:
            """Click an element on the page.
            
            Args:
                selector: CSS selector of element to click
            
            Returns:
                Dictionary with success status
            """
            await self._click(selector)
            return {"success": True, "selector": selector}
        
        @tool
        async def fill(
            selector: str,
            value: str,
        ) -> dict[str, Any]:
            """Fill a form field with a value.
            
            Args:
                selector: CSS selector of input/textarea element
                value: Value to fill
            
            Returns:
                Dictionary with success status
            """
            await self._fill(selector, value)
            return {"success": True, "selector": selector}
        
        @tool
        async def select_option(
            selector: str,
            value: str,
        ) -> dict[str, Any]:
            """Select an option from a dropdown.
            
            Args:
                selector: CSS selector of select element
                value: Value or label of option to select
            
            Returns:
                Dictionary with success status
            """
            await self._select(selector, value)
            return {"success": True, "selector": selector, "value": value}
        
        @tool
        async def evaluate_javascript(script: str) -> dict[str, Any]:
            """Execute JavaScript code on the page.
            
            Args:
                script: JavaScript code to execute
            
            Returns:
                Dictionary with the result of execution
            """
            result = await self._evaluate(script)
            return {"result": result}
        
        @tool
        async def wait_for_element(
            selector: str,
            state: str = "visible",
            timeout_ms: int | None = None,
        ) -> dict[str, Any]:
            """Wait for an element to appear or reach a state.
            
            Args:
                selector: CSS selector of element to wait for
                state: State to wait for: "visible", "hidden", "attached", "detached"
                timeout_ms: Maximum time to wait in milliseconds
            
            Returns:
                Dictionary with success status
            """
            await self._wait_for(selector, state, timeout_ms)
            return {"success": True, "selector": selector, "state": state}
        
        @tool
        async def get_page_info() -> dict[str, Any]:
            """Get information about the current page.
            
            Returns:
                Dictionary with URL, title, and viewport size
            """
            page = await self._ensure_browser()
            viewport = page.viewport_size or {"width": 0, "height": 0}
            
            return {
                "url": page.url,
                "title": await page.title(),
                "viewport": viewport,
            }
        
        @tool
        async def go_back() -> dict[str, Any]:
            """Navigate back in browser history.
            
            Returns:
                Dictionary with new URL and title
            """
            page = await self._ensure_browser()
            await page.go_back()
            return {
                "url": page.url,
                "title": await page.title(),
            }
        
        @tool
        async def go_forward() -> dict[str, Any]:
            """Navigate forward in browser history.
            
            Returns:
                Dictionary with new URL and title
            """
            page = await self._ensure_browser()
            await page.go_forward()
            return {
                "url": page.url,
                "title": await page.title(),
            }
        
        @tool
        async def reload() -> dict[str, Any]:
            """Reload the current page.
            
            Returns:
                Dictionary with URL and title
            """
            page = await self._ensure_browser()
            await page.reload()
            return {
                "url": page.url,
                "title": await page.title(),
            }
        
        @tool
        async def get_html(selector: str | None = None) -> dict[str, Any]:
            """Get HTML content of the page or an element.
            
            Args:
                selector: CSS selector of element (optional, gets full page if not provided)
            
            Returns:
                Dictionary with HTML content
            """
            page = await self._ensure_browser()
            
            if selector:
                element = await page.query_selector(selector)
                if element:
                    html = await element.inner_html()
                else:
                    html = ""
            else:
                html = await page.content()
            
            return {
                "html": html[:20000] + "..." if len(html) > 20000 else html,
                "length": len(html),
                "truncated": len(html) > 20000,
            }
        
        @tool
        async def type_text(
            selector: str,
            text: str,
            delay_ms: int = 50,
        ) -> dict[str, Any]:
            """Type text into an element with realistic delays.
            
            Args:
                selector: CSS selector of element
                text: Text to type
                delay_ms: Delay between keystrokes in milliseconds
            
            Returns:
                Dictionary with success status
            """
            page = await self._ensure_browser()
            await page.type(selector, text, delay=delay_ms)
            return {"success": True, "selector": selector, "typed": text}
        
        @tool
        async def press_key(key: str) -> dict[str, Any]:
            """Press a keyboard key.
            
            Args:
                key: Key to press (e.g., "Enter", "Tab", "Escape", "ArrowDown")
            
            Returns:
                Dictionary with success status
            """
            page = await self._ensure_browser()
            await page.keyboard.press(key)
            return {"success": True, "key": key}
        
        @tool
        async def close_browser() -> dict[str, Any]:
            """Close the browser and cleanup resources.
            
            Returns:
                Dictionary with success status
            """
            await self._close_browser()
            return {"success": True, "message": "Browser closed"}
        
        return [
            navigate,
            screenshot,
            extract_text,
            extract_links,
            click,
            fill,
            select_option,
            evaluate_javascript,
            wait_for_element,
            get_page_info,
            go_back,
            go_forward,
            reload,
            get_html,
            type_text,
            press_key,
            close_browser,
        ]
