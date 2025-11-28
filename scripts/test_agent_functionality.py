#!/usr/bin/env python3
"""
Comprehensive Agent Functionality Tests - CHALLENGING Edition

These tests are designed to FORCE tool usage and test real agent capabilities.
The LLM cannot answer these questions without using the tools.

Test Categories:
1. Impossible Math - calculations LLM cannot do mentally
2. Hidden Data - information only available via tools
3. Multi-Step Reasoning - requires chaining tool calls
4. Error Recovery - tools that fail and need retry
5. Complex DAG - parallel and dependent operations
6. Plan Execution - dynamic planning with obstacles
"""

import asyncio
import os
import sys
import time
import random
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agenticflow import Agent, AgentConfig
from agenticflow.graphs import ReActExecutor, DAGExecutor, PlanExecutor


# =============================================================================
# CHALLENGING TOOLS - These hold data LLM cannot know
# =============================================================================

# Secret database with random data generated at runtime
SECRET_DATABASE: dict[str, Any] = {}
OPERATION_LOG: list[str] = []  # Track tool calls for verification


def reset_test_state():
    """Reset state between tests."""
    global SECRET_DATABASE, OPERATION_LOG
    SECRET_DATABASE = {}
    OPERATION_LOG = []


@tool
def compute_hash(text: str) -> str:
    """Compute the SHA256 cryptographic hash of the given text.
    
    Use this tool when you need to generate a hash digest of any string.
    The hash is deterministic - same input always produces same output.
    
    Args:
        text: The string to hash. Example: "hello world"
        
    Returns:
        The SHA256 hash as a 64-character hexadecimal string.
        Example: "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    """
    OPERATION_LOG.append(f"compute_hash({text!r})")
    return hashlib.sha256(text.encode()).hexdigest()


@tool
def lookup_secret_value(key: str) -> str:
    """Retrieve a value from the secret key-value database.
    
    Use this to look up previously stored values by their key.
    
    Args:
        key: The key to look up. Example: "api_key", "user_token"
        
    Returns:
        The stored value as a string, or "NOT_FOUND" if the key doesn't exist.
    """
    OPERATION_LOG.append(f"lookup_secret_value({key!r})")
    return str(SECRET_DATABASE.get(key, "NOT_FOUND"))


@tool
def store_secret_value(key: str, value: str) -> str:
    """Store a key-value pair in the secret database.
    
    Use this to save values that can be retrieved later with lookup_secret_value.
    
    Args:
        key: The unique key to store the value under. Example: "session_id"
        value: The string value to store. Example: "abc123"
        
    Returns:
        "STORED" on success.
    """
    OPERATION_LOG.append(f"store_secret_value({key!r}, {value!r})")
    SECRET_DATABASE[key] = value
    return "STORED"


@tool
def compute_large_factorial(n: int) -> str:
    """Calculate the factorial of a number (n!).
    
    Factorial is n × (n-1) × (n-2) × ... × 2 × 1.
    Use this for any factorial calculation - do NOT try to calculate manually.
    
    Args:
        n: A positive integer between 1 and 100 (inclusive).
           Example: 5 (returns "120" because 5! = 5×4×3×2×1 = 120)
        
    Returns:
        The factorial result as a string. Can be very large numbers.
        Example: compute_large_factorial(25) returns "15511210043330985984000000"
    """
    OPERATION_LOG.append(f"compute_large_factorial({n})")
    if n < 1 or n > 100:
        return "ERROR: n must be between 1 and 100"
    result = 1
    for i in range(2, n + 1):
        result *= i
    return str(result)


@tool
def compute_prime_factors(n: int) -> str:
    """Find all prime factors of a number.
    
    Prime factorization breaks down a number into prime numbers that multiply
    to give the original number. Use this tool - do NOT factor manually.
    
    Args:
        n: A positive integer between 2 and 10,000,000.
           Example: 12 (returns "2,2,3" because 12 = 2×2×3)
        
    Returns:
        Comma-separated list of prime factors (may include duplicates).
        If n is prime, returns just n itself.
        Example: 7919 returns "7919" (it's prime - only divisible by 1 and itself)
    """
    OPERATION_LOG.append(f"compute_prime_factors({n})")
    if n < 2 or n > 10000000:
        return "ERROR: n must be between 2 and 10000000"
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return ",".join(map(str, factors))


@tool
def get_employee_data(employee_id: str) -> str:
    """Retrieve an employee's full record from the HR database.
    
    Returns all employee information including their manager's ID,
    which can be used to look up the manager with another call.
    
    Args:
        employee_id: The employee ID in format "EMP001", "EMP002", etc.
        
    Returns:
        JSON string with fields: name, department, salary, manager.
        The "manager" field contains another employee_id (or null for CEO).
        Example: {"name": "Alice", "department": "Engineering", "salary": 125000, "manager": "EMP005"}
    """
    OPERATION_LOG.append(f"get_employee_data({employee_id!r})")
    employees = {
        "EMP001": {"name": "Alice Johnson", "department": "Engineering", "salary": 125000, "manager": "EMP005"},
        "EMP002": {"name": "Bob Smith", "department": "Marketing", "salary": 95000, "manager": "EMP006"},
        "EMP003": {"name": "Carol Williams", "department": "Engineering", "salary": 115000, "manager": "EMP005"},
        "EMP004": {"name": "David Brown", "department": "Sales", "salary": 105000, "manager": "EMP006"},
        "EMP005": {"name": "Eva Martinez", "department": "Engineering", "salary": 175000, "manager": "EMP007"},
        "EMP006": {"name": "Frank Lee", "department": "Marketing", "salary": 165000, "manager": "EMP007"},
        "EMP007": {"name": "Grace Chen", "department": "Executive", "salary": 250000, "manager": None},
    }
    if employee_id not in employees:
        return f"ERROR: Employee {employee_id} not found"
    import json
    return json.dumps(employees[employee_id])


@tool
def get_inventory_count(product_code: str) -> str:
    """Get the current stock quantity for a product.
    
    Args:
        product_code: Product identifier like "PROD-A1", "PROD-B2", etc.
        
    Returns:
        The inventory count as a string (integer value).
        A count of "0" means out of stock.
        Example: "42" means 42 units in stock.
    """
    OPERATION_LOG.append(f"get_inventory_count({product_code!r})")
    inventory = {
        "PROD-A1": 150,
        "PROD-A2": 0,  # Out of stock!
        "PROD-B1": 75,
        "PROD-B2": 230,
        "PROD-C1": 42,
    }
    if product_code not in inventory:
        return f"ERROR: Product {product_code} not found"
    return str(inventory[product_code])


@tool
def get_product_price(product_code: str) -> str:
    """Get the unit price for a product.
    
    Args:
        product_code: Product identifier like "PROD-A1", "PROD-B2", etc.
        
    Returns:
        The price as a decimal string. Example: "29.99"
        To calculate total value: multiply this by inventory count yourself.
    """
    OPERATION_LOG.append(f"get_product_price({product_code!r})")
    prices = {
        "PROD-A1": 29.99,
        "PROD-A2": 49.99,
        "PROD-B1": 15.50,
        "PROD-B2": 8.25,
        "PROD-C1": 199.99,
    }
    if product_code not in prices:
        return f"ERROR: Product {product_code} not found"
    return str(prices[product_code])


# Flaky tool that fails randomly to test error recovery
_flaky_call_count = 0


@tool
def flaky_database_query(query_id: str) -> str:
    """Execute a database query that may fail due to connection issues.
    
    This query sometimes times out. If you get a timeout error, RETRY the 
    same query - it will eventually succeed.
    
    Args:
        query_id: The query identifier. Valid values: "Q1", "Q2", "Q3"
        
    Returns:
        On success: The query result string.
        On timeout: "ERROR: Database connection timeout..." - RETRY if this happens!
    """
    global _flaky_call_count
    _flaky_call_count += 1
    OPERATION_LOG.append(f"flaky_database_query({query_id!r}) attempt #{_flaky_call_count}")
    
    # Fail first 2 attempts, succeed on 3rd
    if _flaky_call_count < 3:
        return f"ERROR: Database connection timeout (attempt {_flaky_call_count}). Please retry."
    
    results = {
        "Q1": "Total users: 15,847",
        "Q2": "Revenue last month: $1,234,567.89",
        "Q3": "Active sessions: 2,341",
    }
    return results.get(query_id, f"ERROR: Unknown query {query_id}")


@tool
def decode_cipher(encoded_text: str, shift: int) -> str:
    """Decode a Caesar cipher by shifting letters back.
    
    A Caesar cipher shifts each letter by a fixed number. This tool reverses it.
    You MUST know the shift value - use get_cipher_shift to look it up first.
    
    Args:
        encoded_text: The encrypted message to decode. Example: "Khoor Zruog"
        shift: The number of positions each letter was shifted (as integer).
               Get this from get_cipher_shift tool. Must be an integer, not string.
               Example: 3 (not "3")
        
    Returns:
        The decoded plaintext message.
        Example: decode_cipher("Khoor Zruog", 3) returns "Hello World"
    """
    OPERATION_LOG.append(f"decode_cipher({encoded_text!r}, {shift})")
    result = []
    for char in encoded_text:
        if char.isalpha():
            base = ord('a') if char.islower() else ord('A')
            decoded = chr((ord(char) - base - shift) % 26 + base)
            result.append(decoded)
        else:
            result.append(char)
    return "".join(result)


@tool
def get_cipher_shift(cipher_id: str) -> str:
    """Look up the shift value for a named cipher.
    
    Different ciphers use different shift values. Use this to find the shift,
    then pass it to decode_cipher (convert the string result to an integer).
    
    Args:
        cipher_id: The cipher identifier. Valid: "CIPHER_ALPHA", "CIPHER_BETA", "CIPHER_GAMMA"
        
    Returns:
        The shift value as a STRING. Convert to int before passing to decode_cipher.
        Example: "3" for CIPHER_ALPHA (use int("3") = 3 as the shift parameter)
    """
    OPERATION_LOG.append(f"get_cipher_shift({cipher_id!r})")
    shifts = {
        "CIPHER_ALPHA": "3",
        "CIPHER_BETA": "7",
        "CIPHER_GAMMA": "13",
    }
    return shifts.get(cipher_id, "ERROR: Unknown cipher ID")


@tool
def validate_checksum(data: str, expected_checksum: str) -> str:
    """Verify that data matches an expected MD5 checksum.
    
    Computes MD5 hash of data and compares to expected value.
    
    Args:
        data: The exact text to verify. Must match exactly (case-sensitive).
        expected_checksum: The expected MD5 hash (32-char hex string from get_expected_checksum).
        
    Returns:
        "VALID" if checksums match, or "INVALID: actual checksum is <hash>" if not.
    """
    OPERATION_LOG.append(f"validate_checksum({data!r}, {expected_checksum!r})")
    actual = hashlib.md5(data.encode()).hexdigest()
    if actual == expected_checksum:
        return "VALID"
    return f"INVALID: actual checksum is {actual}"


@tool
def get_expected_checksum(document_id: str) -> str:
    """Get the expected MD5 checksum for a document by its ID.
    
    Use this to get the checksum, then validate content with validate_checksum.
    
    Args:
        document_id: Document identifier. Valid: "DOC001", "DOC002", "DOC003"
        
    Returns:
        The expected MD5 checksum as a 32-character hex string.
    """
    OPERATION_LOG.append(f"get_expected_checksum({document_id!r})")
    checksums = {
        "DOC001": hashlib.md5(b"Important document content").hexdigest(),
        "DOC002": hashlib.md5(b"Another document here").hexdigest(),
        "DOC003": hashlib.md5(b"Secret plans").hexdigest(),
    }
    return checksums.get(document_id, "ERROR: Document not found")


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert a monetary amount from one currency to another.
    
    Use this for ALL currency conversions. For chained conversions (A→B→C),
    call this tool twice: first A→B, then use that result for B→C.
    
    Args:
        amount: The amount to convert (as a number, e.g., 100.0 or 1000)
        from_currency: Source currency code: "USD", "EUR", "GBP", "JPY", "CAD", "AUD"
        to_currency: Target currency code: "USD", "EUR", "GBP", "JPY", "CAD", "AUD"
        
    Returns:
        The converted amount as a string with 2 decimal places.
        Example: convert_currency(100, "USD", "EUR") returns "92.00"
    """
    OPERATION_LOG.append(f"convert_currency({amount}, {from_currency!r}, {to_currency!r})")
    # Rates relative to USD
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 149.50,
        "CAD": 1.36,
        "AUD": 1.53,
    }
    if from_currency not in rates or to_currency not in rates:
        return "ERROR: Unknown currency"
    usd_amount = amount / rates[from_currency]
    result = usd_amount * rates[to_currency]
    return f"{result:.2f}"


# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    error: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    details: str = ""


class TestRunner:
    """Run tests and collect results."""
    
    def __init__(self):
        self.results: list[TestResult] = []
        self.model = ChatOpenAI(model="gpt-4o-mini")
    
    def create_agent(self, tools: list, description: str = "Test agent") -> Agent:
        """Create a test agent with given tools."""
        return Agent(
            config=AgentConfig(
                name="test_agent",
                description=description,
                model=self.model,
            ),
            tools=tools,
        )
    
    async def run_test(
        self,
        name: str,
        task: str,
        tools: list,
        verify_fn,
        executor_type: str = "react",
        verify_tool_used: list[str] | None = None,
    ) -> TestResult:
        """Run a single test."""
        global _flaky_call_count
        _flaky_call_count = 0
        reset_test_state()
        
        start = time.time()
        agent = self.create_agent(tools)
        
        try:
            if executor_type == "react":
                executor = ReActExecutor(agent)
                executor.max_iterations = 15
            elif executor_type == "dag":
                executor = DAGExecutor(agent)
                executor.max_iterations = 15
            elif executor_type == "plan":
                executor = PlanExecutor(agent)
                executor.max_iterations = 15
            else:
                raise ValueError(f"Unknown executor: {executor_type}")
            
            result = await executor.execute(task)
            duration = time.time() - start
            
            # Verify the result
            passed, details = verify_fn(result, OPERATION_LOG)
            
            # Verify required tools were used
            if verify_tool_used and passed:
                for tool_name in verify_tool_used:
                    if not any(tool_name in log for log in OPERATION_LOG):
                        passed = False
                        details = f"FAILED: Required tool '{tool_name}' was not called. Logs: {OPERATION_LOG}"
                        break
            
            return TestResult(
                name=name,
                passed=passed,
                duration=duration,
                tool_calls=OPERATION_LOG.copy(),
                details=details,
            )
        except Exception as e:
            duration = time.time() - start
            return TestResult(
                name=name,
                passed=False,
                duration=duration,
                error=str(e),
                tool_calls=OPERATION_LOG.copy(),
            )
    
    def print_results(self):
        """Print test results summary."""
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"\n{status} | {r.name} ({r.duration:.2f}s)")
            if r.tool_calls:
                print(f"  Tools called: {len(r.tool_calls)}")
            if r.details:
                print(f"  Details: {r.details[:200]}")
            if r.error:
                print(f"  Error: {r.error}")
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: {passed}/{len(self.results)} tests passed")
        if failed > 0:
            print(f"         {failed} tests FAILED")
        print("=" * 80)
        
        return failed == 0


# =============================================================================
# CHALLENGING TESTS
# =============================================================================

async def test_impossible_factorial(runner: TestRunner) -> TestResult:
    """Test: Compute factorial(25) - LLM cannot calculate this mentally.
    
    25! = 15511210043330985984000000
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        expected = "15511210043330985984000000"
        if expected in result:
            return True, f"Correct factorial found in response"
        return False, f"Expected {expected} not found in: {result[:200]}"
    
    return await runner.run_test(
        name="Impossible Factorial (25!)",
        task="What is the factorial of 25? Use the compute_large_factorial tool to calculate it. Give me the exact number.",
        tools=[compute_large_factorial],
        verify_fn=verify,
        verify_tool_used=["compute_large_factorial"],
    )


async def test_prime_factorization(runner: TestRunner) -> TestResult:
    """Test: Find prime factors of 7919 (it's actually prime).
    
    LLM cannot easily determine if 7919 is prime or factor it.
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        # 7919 is prime, so the only factor is itself
        if "7919" in result and ("prime" in result.lower() or "only factor" in result.lower() or "itself" in result.lower()):
            return True, "Correctly identified 7919 as prime"
        # Check if the tool was called and returned just 7919
        for log in logs:
            if "compute_prime_factors" in log:
                return True, "Tool was called (7919 is prime)"
        return False, f"Expected prime identification for 7919: {result[:200]}"
    
    return await runner.run_test(
        name="Prime Factorization (7919 is prime)",
        task="What are the prime factors of 7919? Use the compute_prime_factors tool.",
        tools=[compute_prime_factors],
        verify_fn=verify,
        verify_tool_used=["compute_prime_factors"],
    )


async def test_hash_computation(runner: TestRunner) -> TestResult:
    """Test: Compute SHA256 hash - LLM cannot compute this.
    
    SHA256("agenticflow") = specific hash
    """
    expected_hash = hashlib.sha256(b"agenticflow").hexdigest()
    
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        if expected_hash in result:
            return True, f"Correct hash found"
        # Check first 16 chars to be lenient
        if expected_hash[:16] in result:
            return True, f"Correct hash prefix found"
        return False, f"Expected hash {expected_hash[:32]}... not found"
    
    return await runner.run_test(
        name="SHA256 Hash Computation",
        task="Compute the SHA256 hash of the text 'agenticflow'. Use the compute_hash tool.",
        tools=[compute_hash],
        verify_fn=verify,
        verify_tool_used=["compute_hash"],
    )


async def test_employee_chain_lookup(runner: TestRunner) -> TestResult:
    """Test: Multi-step - find manager's manager.
    
    EMP001 -> EMP005 -> EMP007 (Grace Chen)
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        # Must have looked up at least 2 employees
        emp_lookups = [l for l in logs if "get_employee_data" in l]
        if len(emp_lookups) < 2:
            return False, f"Expected at least 2 employee lookups, got {len(emp_lookups)}"
        
        if "Grace Chen" in result or "EMP007" in result:
            return True, "Correctly found manager's manager (Grace Chen)"
        return False, f"Expected Grace Chen/EMP007: {result[:200]}"
    
    return await runner.run_test(
        name="Employee Manager Chain (2 lookups)",
        task="Who is the manager of the manager of employee EMP001? First look up EMP001 to find their manager, then look up that manager to find THEIR manager. Tell me the name.",
        tools=[get_employee_data],
        verify_fn=verify,
        verify_tool_used=["get_employee_data"],
    )


async def test_inventory_value_calculation(runner: TestRunner) -> TestResult:
    """Test: Calculate total value = count * price for PROD-C1.
    
    PROD-C1: count=42, price=199.99, total=8399.58
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        # Must have called both tools
        has_count = any("get_inventory_count" in l for l in logs)
        has_price = any("get_product_price" in l for l in logs)
        
        if not has_count or not has_price:
            return False, f"Must call both inventory and price tools"
        
        # Check for correct calculation (42 * 199.99 = 8399.58)
        if "8399.58" in result or "8399.6" in result or "8,399.58" in result:
            return True, "Correct inventory value calculated"
        
        return False, f"Expected 8399.58, got: {result[:200]}"
    
    return await runner.run_test(
        name="Inventory Value Calculation",
        task="Calculate the total inventory value for product PROD-C1. Get the inventory count and price, then multiply them. What is the total value?",
        tools=[get_inventory_count, get_product_price],
        verify_fn=verify,
        verify_tool_used=["get_inventory_count", "get_product_price"],
    )


async def test_cipher_decode(runner: TestRunner) -> TestResult:
    """Test: Two-step cipher decode.
    
    1. Look up shift for CIPHER_ALPHA (3)
    2. Decode "Khoor Zruog" with shift 3 -> "Hello World"
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        has_shift = any("get_cipher_shift" in l for l in logs)
        has_decode = any("decode_cipher" in l for l in logs)
        
        if not has_shift:
            return False, "Must look up cipher shift first"
        if not has_decode:
            return False, "Must decode the cipher"
        
        if "Hello World" in result:
            return True, "Correctly decoded cipher"
        
        return False, f"Expected 'Hello World': {result[:200]}"
    
    return await runner.run_test(
        name="Two-Step Cipher Decode",
        task="Decode the message 'Khoor Zruog' using cipher CIPHER_ALPHA. First look up the shift value for CIPHER_ALPHA, then use that shift to decode the message.",
        tools=[get_cipher_shift, decode_cipher],
        verify_fn=verify,
        verify_tool_used=["get_cipher_shift", "decode_cipher"],
    )


async def test_flaky_retry(runner: TestRunner) -> TestResult:
    """Test: Error recovery - tool fails twice then succeeds.
    
    Agent must retry when tool returns error.
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        # Must have multiple attempts
        attempts = [l for l in logs if "flaky_database_query" in l]
        if len(attempts) < 3:
            return False, f"Expected 3 attempts, got {len(attempts)}"
        
        if "15,847" in result or "15847" in result:
            return True, f"Got correct result after {len(attempts)} retries"
        
        return False, f"Expected 15,847 users: {result[:200]}"
    
    return await runner.run_test(
        name="Flaky API Retry (3 attempts needed)",
        task="Get the total users count using query Q1. The database might timeout, so keep retrying if you get a timeout error. Report the total user count.",
        tools=[flaky_database_query],
        verify_fn=verify,
        verify_tool_used=["flaky_database_query"],
    )


async def test_checksum_validation(runner: TestRunner) -> TestResult:
    """Test: Two-step checksum validation.
    
    1. Get expected checksum for DOC001
    2. Validate "Important document content" against it
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        has_get = any("get_expected_checksum" in l for l in logs)
        has_validate = any("validate_checksum" in l for l in logs)
        
        if not has_get or not has_validate:
            return False, "Must call both checksum tools"
        
        result_lower = result.lower()
        # Accept various ways of saying validation passed
        if any(word in result_lower for word in ["valid", "matches", "correct", "verified", "passed"]):
            return True, "Correctly validated checksum"
        
        return False, f"Expected VALID result: {result[:200]}"
    
    return await runner.run_test(
        name="Checksum Validation (2 steps)",
        task="Validate that the text 'Important document content' has the correct checksum for DOC001. First get the expected checksum for DOC001, then validate the text against it.",
        tools=[get_expected_checksum, validate_checksum],
        verify_fn=verify,
        verify_tool_used=["get_expected_checksum", "validate_checksum"],
    )


async def test_currency_chain(runner: TestRunner) -> TestResult:
    """Test: Currency conversion chain.
    
    Convert 100 USD -> EUR -> GBP
    100 USD = 92 EUR = 78.89 GBP (approximately)
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        conversions = [l for l in logs if "convert_currency" in l]
        if len(conversions) < 2:
            return False, f"Expected 2 conversions, got {len(conversions)}"
        
        # Final result should be around 78-80 GBP
        import re
        numbers = re.findall(r'\d+\.?\d*', result)
        for num in numbers:
            try:
                val = float(num)
                if 78 <= val <= 81:  # Allow some rounding
                    return True, f"Correct final GBP value: {val}"
            except:
                pass
        
        return False, f"Expected ~79 GBP: {result[:200]}"
    
    return await runner.run_test(
        name="Currency Conversion Chain",
        task="Convert 100 USD to EUR, then convert that EUR amount to GBP. What is the final GBP amount?",
        tools=[convert_currency],
        verify_fn=verify,
        verify_tool_used=["convert_currency"],
    )


async def test_secret_store_retrieve(runner: TestRunner) -> TestResult:
    """Test: Store then retrieve a value.
    
    This tests that the agent understands state changes.
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        has_store = any("store_secret_value" in l for l in logs)
        has_lookup = any("lookup_secret_value" in l for l in logs)
        
        if not has_store:
            return False, "Must store the value first"
        if not has_lookup:
            return False, "Must retrieve the value"
        
        # Check the order - store must come before lookup
        store_idx = next(i for i, l in enumerate(logs) if "store_secret_value" in l)
        lookup_idx = next(i for i, l in enumerate(logs) if "lookup_secret_value" in l)
        
        if lookup_idx < store_idx:
            return False, "Must store BEFORE retrieving"
        
        if "my_secret_123" in result:
            return True, "Correctly stored and retrieved value"
        
        return False, f"Expected 'my_secret_123': {result[:200]}"
    
    return await runner.run_test(
        name="Store Then Retrieve Secret",
        task="Store the value 'my_secret_123' with key 'test_key' in the secret database. Then retrieve it back to verify it was stored correctly. What value did you retrieve?",
        tools=[store_secret_value, lookup_secret_value],
        verify_fn=verify,
        verify_tool_used=["store_secret_value", "lookup_secret_value"],
    )


async def test_total_engineering_salary(runner: TestRunner) -> TestResult:
    """Test: Complex aggregation requiring multiple lookups.
    
    Engineering dept: Alice (125k), Carol (115k), Eva (175k)
    Total = 415,000
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        lookups = [l for l in logs if "get_employee_data" in l]
        if len(lookups) < 3:
            return False, f"Expected at least 3 lookups, got {len(lookups)}"
        
        # Check for correct total
        if "415" in result or "415,000" in result or "415000" in result:
            return True, "Correct total salary calculated"
        
        return False, f"Expected 415,000: {result[:200]}"
    
    return await runner.run_test(
        name="Engineering Salary Total (multi-lookup)",
        task="Calculate the total salary of all Engineering department employees. Look up employees EMP001, EMP003, and EMP005 (they are in Engineering). Sum their salaries and tell me the total.",
        tools=[get_employee_data],
        verify_fn=verify,
        verify_tool_used=["get_employee_data"],
    )


async def test_inventory_out_of_stock_check(runner: TestRunner) -> TestResult:
    """Test: Find products with low/zero inventory.
    
    PROD-A2 has 0 inventory (out of stock)
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        lookups = [l for l in logs if "get_inventory_count" in l]
        if len(lookups) < 2:
            return False, f"Expected multiple lookups"
        
        if "PROD-A2" in result and ("0" in result or "out of stock" in result.lower() or "zero" in result.lower()):
            return True, "Correctly identified PROD-A2 as out of stock"
        
        return False, f"Expected PROD-A2 out of stock: {result[:200]}"
    
    return await runner.run_test(
        name="Find Out of Stock Products",
        task="Check the inventory for products PROD-A1 and PROD-A2. Which one is out of stock (has zero inventory)?",
        tools=[get_inventory_count],
        verify_fn=verify,
        verify_tool_used=["get_inventory_count"],
    )


async def test_dag_parallel_execution(runner: TestRunner) -> TestResult:
    """Test: DAG executor with parallel operations.
    
    Get price and count for PROD-B1 in parallel, then compute value.
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        has_count = any("get_inventory_count" in l for l in logs)
        has_price = any("get_product_price" in l for l in logs)
        
        if not has_count or not has_price:
            return False, "Must call both tools"
        
        # PROD-B1: count=75, price=15.50, value=1162.50
        if "1162.5" in result or "1162.50" in result or "1,162.50" in result:
            return True, "Correct value calculated"
        
        return False, f"Expected 1162.50: {result[:200]}"
    
    return await runner.run_test(
        name="DAG Parallel Product Value",
        task="Calculate the inventory value for PROD-B1. Get both the inventory count and the unit price (these can be done in parallel), then multiply them together.",
        tools=[get_inventory_count, get_product_price],
        verify_fn=verify,
        executor_type="dag",
        verify_tool_used=["get_inventory_count", "get_product_price"],
    )


async def test_plan_executor_multi_step(runner: TestRunner) -> TestResult:
    """Test: Plan executor with dynamic planning.
    
    Convert 1000 JPY to USD to CAD.
    1000 JPY = 6.69 USD = 9.10 CAD (approximately)
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        conversions = [l for l in logs if "convert_currency" in l]
        if len(conversions) < 2:
            return False, f"Expected 2 conversions"
        
        # Check for USD step and final CAD
        import re
        numbers = re.findall(r'\d+\.?\d*', result)
        for num in numbers:
            try:
                val = float(num)
                if 8.5 <= val <= 10:  # CAD result range
                    return True, f"Correct CAD value: {val}"
            except:
                pass
        
        return False, f"Expected ~9 CAD: {result[:200]}"
    
    return await runner.run_test(
        name="Plan Executor Currency Chain",
        task="Convert 1000 Japanese Yen (JPY) to US Dollars (USD), then convert that to Canadian Dollars (CAD). What is the final amount in CAD?",
        tools=[convert_currency],
        verify_fn=verify,
        executor_type="plan",
        verify_tool_used=["convert_currency"],
    )


async def test_hash_then_store(runner: TestRunner) -> TestResult:
    """Test: Compute hash and store result.
    
    Tests chaining computation with storage.
    """
    def verify(result: str, logs: list[str]) -> tuple[bool, str]:
        has_hash = any("compute_hash" in l for l in logs)
        has_store = any("store_secret_value" in l for l in logs)
        
        if not has_hash:
            return False, "Must compute hash first"
        if not has_store:
            return False, "Must store the hash"
        
        # Check order
        hash_idx = next(i for i, l in enumerate(logs) if "compute_hash" in l)
        store_idx = next(i for i, l in enumerate(logs) if "store_secret_value" in l)
        
        if store_idx < hash_idx:
            return False, "Must compute hash before storing"
        
        expected_hash = hashlib.sha256(b"password123").hexdigest()
        if expected_hash[:16] in str(SECRET_DATABASE.values()):
            return True, "Hash correctly computed and stored"
        
        return False, f"Hash not stored correctly"
    
    return await runner.run_test(
        name="Compute Hash Then Store",
        task="Compute the SHA256 hash of 'password123', then store that hash value in the secret database with key 'password_hash'. Confirm the hash was stored.",
        tools=[compute_hash, store_secret_value, lookup_secret_value],
        verify_fn=verify,
        verify_tool_used=["compute_hash", "store_secret_value"],
    )


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all challenging tests."""
    print("=" * 80)
    print("AGENTICFLOW - CHALLENGING AGENT FUNCTIONALITY TESTS")
    print("=" * 80)
    print("\nThese tests REQUIRE tool usage - LLM cannot answer without tools.\n")
    
    runner = TestRunner()
    
    # List of all tests
    tests = [
        # Math tests - impossible without tools
        test_impossible_factorial,
        test_prime_factorization,
        test_hash_computation,
        
        # Multi-step reasoning tests
        test_employee_chain_lookup,
        test_inventory_value_calculation,
        test_cipher_decode,
        test_checksum_validation,
        test_currency_chain,
        
        # State management tests
        test_secret_store_retrieve,
        test_hash_then_store,
        
        # Aggregation tests
        test_total_engineering_salary,
        test_inventory_out_of_stock_check,
        
        # Error recovery test
        test_flaky_retry,
        
        # Different executor tests
        test_dag_parallel_execution,
        test_plan_executor_multi_step,
    ]
    
    total_start = time.time()
    
    for test_fn in tests:
        print(f"\n🔄 Running: {test_fn.__name__}")
        result = await test_fn(runner)
        runner.results.append(result)
        
        if result.passed:
            print(f"   ✅ Passed ({result.duration:.2f}s) - {len(result.tool_calls)} tool calls")
        else:
            print(f"   ❌ Failed ({result.duration:.2f}s)")
            if result.error:
                print(f"   Error: {result.error[:100]}")
    
    total_time = time.time() - total_start
    
    all_passed = runner.print_results()
    print(f"\nTotal time: {total_time:.2f}s")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
