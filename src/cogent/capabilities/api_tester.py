"""
API Testing capability - automated endpoint validation and performance testing.

Provides comprehensive API testing with:
- Endpoint accessibility validation
- Response schema validation (JSON/Pydantic)
- Performance testing (latency, throughput)
- HTTP status code validation
- Header validation
- Test suite execution

Example:
    ```python
    from cogent import Agent
    from cogent.capabilities import APITester

    agent = Agent(
        name="API Test Agent",
        model=model,
        capabilities=[
            APITester(
                base_url="https://api.example.com",
                timeout=10,
            )
        ],
    )

    # Agent can now test APIs
    await agent.run("Test the /users endpoint and validate the response schema")
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from cogent.capabilities.base import BaseCapability
from cogent.tools.base import BaseTool, tool

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single API test."""

    test_name: str
    endpoint: str
    method: str
    success: bool
    status_code: int | None = None
    response_time_ms: float = 0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "endpoint": self.endpoint,
            "method": self.method,
            "success": self.success,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "error": self.error,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TestSuiteResult:
    """Result of running a test suite."""

    suite_name: str
    total_tests: int
    passed: int
    failed: int
    total_time_ms: float
    tests: list[TestResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.passed / self.total_tests * 100) if self.total_tests > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": f"{self.success_rate:.1f}%",
            "total_time_ms": self.total_time_ms,
            "tests": [t.to_dict() for t in self.tests],
            "timestamp": self.timestamp.isoformat(),
        }


class APITester(BaseCapability):
    """
    API Testing capability for automated endpoint validation.

    Provides comprehensive API testing with:
    - Endpoint accessibility validation
    - Response schema validation (JSON Schema or Pydantic)
    - Performance testing (latency benchmarks)
    - HTTP status code validation
    - Header validation
    - Test suite execution

    Args:
        base_url: Base URL for API endpoints (e.g., "https://api.example.com")
        timeout: Request timeout in seconds (default: 10)
        headers: Default headers to include in all requests
        verify_ssl: Whether to verify SSL certificates (default: True)
        name: Capability name (default: "api_tester")

    Example:
        ```python
        # Basic testing
        tester = APITester("https://api.example.com")
        result = await tester.test_endpoint("/users", expected_status=200)

        # Performance testing
        perf = await tester.test_performance("/users", num_requests=100)

        # Schema validation
        result = await tester.validate_schema(
            "/users/1",
            expected_schema={"id": int, "name": str}
        )
        ```
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: int = 10,
        headers: dict[str, str] | None = None,
        verify_ssl: bool = True,
        name: str = "api_tester",
    ):
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_headers = headers or {}
        self._verify_ssl = verify_ssl
        self._client: httpx.AsyncClient | None = None
        self._test_history: list[TestResult] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"API testing and validation ({self._base_url or 'any endpoint'})"

    @property
    def tools(self) -> list[BaseTool]:
        return [
            self._test_endpoint_tool(),
            self._validate_schema_tool(),
            self._test_performance_tool(),
            self._run_test_suite_tool(),
        ]

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                verify=self._verify_ssl,
                headers=self._default_headers,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from base_url and endpoint."""
        endpoint = endpoint.lstrip("/")
        if self._base_url:
            return f"{self._base_url}/{endpoint}"
        # If no base_url, assume endpoint is full URL
        return endpoint if endpoint.startswith("http") else f"https://{endpoint}"

    async def test_endpoint(
        self,
        endpoint: str,
        method: str = "GET",
        expected_status: int = 200,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> TestResult:
        """
        Test an API endpoint for accessibility and expected status.

        Args:
            endpoint: Endpoint path or full URL
            method: HTTP method (GET, POST, etc.)
            expected_status: Expected HTTP status code
            headers: Additional headers for this request
            json_body: JSON body for POST/PUT requests

        Returns:
            TestResult with success/failure and details
        """
        url = self._build_url(endpoint)
        client = await self._get_client()
        start_time = time.time()

        try:
            request_headers = {**self._default_headers, **(headers or {})}

            response = await client.request(
                method=method,
                url=url,
                headers=request_headers,
                json=json_body,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            status_match = response.status_code == expected_status

            result = TestResult(
                test_name=f"{method} {endpoint}",
                endpoint=endpoint,
                method=method,
                success=status_match,
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
                error=None if status_match else f"Expected {expected_status}, got {response.status_code}",
                details={
                    "url": url,
                    "headers": dict(response.headers),
                    "response_length": len(response.content),
                },
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            result = TestResult(
                test_name=f"{method} {endpoint}",
                endpoint=endpoint,
                method=method,
                success=False,
                status_code=None,
                response_time_ms=elapsed_ms,
                error=str(e),
                details={"url": url},
            )

        self._test_history.append(result)
        return result

    async def validate_schema(
        self,
        endpoint: str,
        schema_model: type[BaseModel] | None = None,
        expected_fields: dict[str, type] | None = None,
        method: str = "GET",
    ) -> TestResult:
        """
        Validate API response against a schema.

        Args:
            endpoint: Endpoint path or full URL
            schema_model: Pydantic model for validation (preferred)
            expected_fields: Dict of field_name: type for simple validation
            method: HTTP method

        Returns:
            TestResult with schema validation details
        """
        url = self._build_url(endpoint)
        client = await self._get_client()
        start_time = time.time()

        try:
            response = await client.request(method=method, url=url)
            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code != 200:
                return TestResult(
                    test_name=f"Schema validation: {endpoint}",
                    endpoint=endpoint,
                    method=method,
                    success=False,
                    status_code=response.status_code,
                    response_time_ms=elapsed_ms,
                    error=f"Non-200 status: {response.status_code}",
                )

            try:
                json_data = response.json()
            except Exception as e:
                return TestResult(
                    test_name=f"Schema validation: {endpoint}",
                    endpoint=endpoint,
                    method=method,
                    success=False,
                    status_code=response.status_code,
                    response_time_ms=elapsed_ms,
                    error=f"Invalid JSON: {e}",
                )

            # Pydantic validation (preferred)
            if schema_model:
                try:
                    validated = schema_model.model_validate(json_data)
                    return TestResult(
                        test_name=f"Schema validation: {endpoint}",
                        endpoint=endpoint,
                        method=method,
                        success=True,
                        status_code=response.status_code,
                        response_time_ms=elapsed_ms,
                        details={"validated_model": validated.model_dump()},
                    )
                except ValidationError as e:
                    return TestResult(
                        test_name=f"Schema validation: {endpoint}",
                        endpoint=endpoint,
                        method=method,
                        success=False,
                        status_code=response.status_code,
                        response_time_ms=elapsed_ms,
                        error=f"Schema validation failed: {e}",
                    )

            # Simple field type validation
            if expected_fields:
                errors = []
                for field_name, expected_type in expected_fields.items():
                    if field_name not in json_data:
                        errors.append(f"Missing field: {field_name}")
                    elif not isinstance(json_data[field_name], expected_type):
                        errors.append(
                            f"Field {field_name}: expected {expected_type.__name__}, "
                            f"got {type(json_data[field_name]).__name__}"
                        )

                if errors:
                    return TestResult(
                        test_name=f"Schema validation: {endpoint}",
                        endpoint=endpoint,
                        method=method,
                        success=False,
                        status_code=response.status_code,
                        response_time_ms=elapsed_ms,
                        error="; ".join(errors),
                    )

                return TestResult(
                    test_name=f"Schema validation: {endpoint}",
                    endpoint=endpoint,
                    method=method,
                    success=True,
                    status_code=response.status_code,
                    response_time_ms=elapsed_ms,
                    details={"validated_fields": list(expected_fields.keys())},
                )

            # No validation specified
            return TestResult(
                test_name=f"Schema validation: {endpoint}",
                endpoint=endpoint,
                method=method,
                success=False,
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
                error="No schema_model or expected_fields provided",
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=f"Schema validation: {endpoint}",
                endpoint=endpoint,
                method=method,
                success=False,
                status_code=None,
                response_time_ms=elapsed_ms,
                error=str(e),
            )

    async def test_performance(
        self,
        endpoint: str,
        num_requests: int = 10,
        method: str = "GET",
        concurrent: bool = False,
    ) -> TestResult:
        """
        Test API endpoint performance with multiple requests.

        Args:
            endpoint: Endpoint path or full URL
            num_requests: Number of requests to make
            method: HTTP method
            concurrent: Whether to run requests concurrently

        Returns:
            TestResult with performance metrics
        """
        start_time = time.time()
        response_times: list[float] = []
        successes = 0
        failures = 0

        async def single_request() -> tuple[bool, float]:
            req_start = time.time()
            try:
                url = self._build_url(endpoint)
                client = await self._get_client()
                response = await client.request(method=method, url=url)
                elapsed = (time.time() - req_start) * 1000
                return (response.status_code == 200, elapsed)
            except Exception:
                elapsed = (time.time() - req_start) * 1000
                return (False, elapsed)

        if concurrent:
            # Run all requests concurrently
            tasks = [single_request() for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)
            for success, elapsed in results:
                response_times.append(elapsed)
                if success:
                    successes += 1
                else:
                    failures += 1
        else:
            # Run requests sequentially
            for _ in range(num_requests):
                success, elapsed = await single_request()
                response_times.append(elapsed)
                if success:
                    successes += 1
                else:
                    failures += 1

        total_elapsed = (time.time() - start_time) * 1000
        avg_response = sum(response_times) / len(response_times) if response_times else 0
        min_response = min(response_times) if response_times else 0
        max_response = max(response_times) if response_times else 0

        return TestResult(
            test_name=f"Performance test: {endpoint}",
            endpoint=endpoint,
            method=method,
            success=failures == 0,
            status_code=200,
            response_time_ms=total_elapsed,
            error=f"{failures} requests failed" if failures > 0 else None,
            details={
                "num_requests": num_requests,
                "successes": successes,
                "failures": failures,
                "concurrent": concurrent,
                "avg_response_ms": round(avg_response, 2),
                "min_response_ms": round(min_response, 2),
                "max_response_ms": round(max_response, 2),
                "requests_per_second": round(num_requests / (total_elapsed / 1000), 2),
            },
        )

    async def run_test_suite(
        self,
        tests: list[dict[str, Any]],
        suite_name: str = "API Test Suite",
    ) -> TestSuiteResult:
        """
        Run a suite of API tests.

        Args:
            tests: List of test configurations, each with 'endpoint', 'method', 'expected_status', etc.
            suite_name: Name for this test suite

        Returns:
            TestSuiteResult with aggregated results
        """
        start_time = time.time()
        results: list[TestResult] = []

        for test_config in tests:
            endpoint = test_config.get("endpoint", "")
            method = test_config.get("method", "GET")
            expected_status = test_config.get("expected_status", 200)

            result = await self.test_endpoint(
                endpoint=endpoint,
                method=method,
                expected_status=expected_status,
            )
            results.append(result)

        total_elapsed = (time.time() - start_time) * 1000
        passed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(results),
            passed=passed,
            failed=failed,
            total_time_ms=total_elapsed,
            tests=results,
        )

    @property
    def history(self) -> list[TestResult]:
        """Get test history."""
        return self._test_history.copy()

    def clear_history(self) -> int:
        """Clear test history. Returns count cleared."""
        count = len(self._test_history)
        self._test_history.clear()
        return count

    # =========================================================================
    # Tool Generation
    # =========================================================================

    def _test_endpoint_tool(self) -> BaseTool:
        tester = self

        @tool
        async def test_api_endpoint(
            endpoint: str,
            method: str = "GET",
            expected_status: int = 200,
        ) -> str:
            """
            Test an API endpoint for accessibility and expected status code.

            Args:
                endpoint: Endpoint path or full URL (e.g., "/users" or "https://api.example.com/users")
                method: HTTP method (GET, POST, PUT, DELETE, etc.)
                expected_status: Expected HTTP status code (default: 200)

            Returns:
                Test result with success/failure, response time, and details

            Example:
                test_api_endpoint("/api/users", "GET", 200)
            """
            result = await tester.test_endpoint(
                endpoint=endpoint,
                method=method.upper(),
                expected_status=expected_status,
            )

            if result.success:
                lines = [f"✓ Test passed: {result.test_name}"]
                lines.append(f"Status: {result.status_code}")
                lines.append(f"Response time: {result.response_time_ms:.0f}ms")
            else:
                lines = [f"✗ Test failed: {result.test_name}"]
                lines.append(f"Error: {result.error}")
                if result.status_code:
                    lines.append(f"Status: {result.status_code}")
                lines.append(f"Response time: {result.response_time_ms:.0f}ms")

            return "\n".join(lines)

        return test_api_endpoint

    def _validate_schema_tool(self) -> BaseTool:
        tester = self

        @tool
        async def validate_api_schema(
            endpoint: str,
            expected_fields: str,
            method: str = "GET",
        ) -> str:
            """
            Validate API response schema against expected fields.

            Args:
                endpoint: Endpoint path or full URL
                expected_fields: JSON string of field:type pairs (e.g., '{"id": "int", "name": "str"}')
                method: HTTP method (default: GET)

            Returns:
                Validation result with success/failure and details

            Example:
                validate_api_schema("/users/1", '{"id": "int", "name": "str", "email": "str"}')
            """
            import json

            try:
                fields_dict = json.loads(expected_fields)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in expected_fields: {expected_fields}"

            # Convert string type names to actual types
            type_mapping = {
                "int": int,
                "str": str,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
            }

            typed_fields = {}
            for field_name, type_name in fields_dict.items():
                if type_name not in type_mapping:
                    return f"Error: Unknown type '{type_name}' for field '{field_name}'"
                typed_fields[field_name] = type_mapping[type_name]

            result = await tester.validate_schema(
                endpoint=endpoint,
                expected_fields=typed_fields,
                method=method.upper(),
            )

            if result.success:
                lines = [f"✓ Schema validation passed: {endpoint}"]
                lines.append(f"Status: {result.status_code}")
                lines.append(f"Response time: {result.response_time_ms:.0f}ms")
                if "validated_fields" in result.details:
                    lines.append(f"Validated fields: {', '.join(result.details['validated_fields'])}")
            else:
                lines = [f"✗ Schema validation failed: {endpoint}"]
                lines.append(f"Error: {result.error}")
                if result.status_code:
                    lines.append(f"Status: {result.status_code}")

            return "\n".join(lines)

        return validate_api_schema

    def _test_performance_tool(self) -> BaseTool:
        tester = self

        @tool
        async def test_api_performance(
            endpoint: str,
            num_requests: int = 10,
            concurrent: bool = False,
        ) -> str:
            """
            Test API endpoint performance with multiple requests.

            Args:
                endpoint: Endpoint path or full URL
                num_requests: Number of requests to make (default: 10, max: 100)
                concurrent: Run requests concurrently (default: False)

            Returns:
                Performance metrics including avg/min/max response times

            Example:
                test_api_performance("/users", 50, True)
            """
            # Limit max requests to prevent abuse
            num_requests = min(num_requests, 100)

            result = await tester.test_performance(
                endpoint=endpoint,
                num_requests=num_requests,
                concurrent=concurrent,
            )

            lines = [f"✓ Performance test: {endpoint}"]
            lines.append(f"Requests: {result.details['num_requests']}")
            lines.append(f"Successes: {result.details['successes']}")
            lines.append(f"Failures: {result.details['failures']}")
            lines.append(f"Mode: {'Concurrent' if concurrent else 'Sequential'}")
            lines.append("\nPerformance:")
            lines.append(f"  Avg response: {result.details['avg_response_ms']}ms")
            lines.append(f"  Min response: {result.details['min_response_ms']}ms")
            lines.append(f"  Max response: {result.details['max_response_ms']}ms")
            lines.append(f"  Throughput: {result.details['requests_per_second']} req/s")
            lines.append(f"\nTotal time: {result.response_time_ms:.0f}ms")

            if not result.success:
                lines.append(f"\n⚠ Warning: {result.error}")

            return "\n".join(lines)

        return test_api_performance

    def _run_test_suite_tool(self) -> BaseTool:
        tester = self

        @tool
        async def run_api_test_suite(
            tests_json: str,
            suite_name: str = "API Test Suite",
        ) -> str:
            """
            Run a suite of API tests.

            Args:
                tests_json: JSON array of test configurations, each with 'endpoint', 'method', 'expected_status'
                suite_name: Name for this test suite

            Returns:
                Suite results with pass/fail summary and individual test details

            Example:
                run_api_test_suite('[{"endpoint": "/users", "method": "GET", "expected_status": 200}]')
            """
            import json

            try:
                tests = json.loads(tests_json)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in tests_json: {tests_json}"

            if not isinstance(tests, list):
                return "Error: tests_json must be a JSON array"

            result = await tester.run_test_suite(tests=tests, suite_name=suite_name)

            lines = [f"{'✓' if result.failed == 0 else '✗'} {result.suite_name}"]
            lines.append(f"Tests: {result.total_tests}")
            lines.append(f"Passed: {result.passed}")
            lines.append(f"Failed: {result.failed}")
            lines.append(f"Success rate: {result.success_rate:.1f}%")
            lines.append(f"Total time: {result.total_time_ms:.0f}ms")

            if result.failed > 0:
                lines.append("\nFailed tests:")
                for test in result.tests:
                    if not test.success:
                        lines.append(f"  - {test.test_name}: {test.error}")

            return "\n".join(lines)

        return run_api_test_suite
