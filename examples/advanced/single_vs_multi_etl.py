"""Single Agent vs Multi-Agent: Realistic ETL Comparison.

THE SCENARIO:
We're migrating customer data from a legacy CRM's CSV export to our new database.
The CSV is messy (mixed date formats, nulls, duplicates, encoding issues).
We need to clean it up and load it correctly.

The natural workflow:
    1. "What's in this CSV file? What's broken?"
    2. "How do I fix these issues?"
    3. "Will my fixes actually work?"
    4. "What SQL do I need to load this?"

The comparison:
    Single Agent: Remembers everything naturally (4-layer memory)
    Multi-Agent: Must coordinate and share via RunContext

Run:
    uv run python examples/advanced/single_vs_multi_etl.py
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from cogent import Agent, Observer, RunContext, tool

# ============================================================================
# STRUCTURED OUTPUTS
# ============================================================================

class ColumnInfo(BaseModel):
    """Column schema information."""

    name: str
    data_type: str
    nullable: bool
    issues: list[str] = Field(default_factory=list)


class TransformRule(BaseModel):
    """Transformation rule."""

    source_column: str
    target_column: str
    transform: str
    reason: str


class ValidationIssue(BaseModel):
    """Data validation issue."""

    column: str
    issue_type: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    example: str
    fix: str


class DataMigrationPlan(BaseModel):
    """Plan for migrating the messy CSV data."""

    # What we're dealing with
    source_file: str
    total_rows: int
    columns: list[ColumnInfo]

    # How to clean it
    cleanup_rules: list[TransformRule]

    # What's still broken
    remaining_issues: list[ValidationIssue]

    # How to load it
    target_table: str
    create_table_sql: str
    insert_approach: str

    # Bottom line
    ready_to_migrate: bool
    next_steps: str


# ============================================================================
# MIGRATION CONTEXT
# ============================================================================

@dataclass
class MigrationContext(RunContext):
    """Shared context for multi-agent data migration workflow."""

    # Discovered by analyst
    source_file: str = ""
    total_rows: int = 0
    columns_info: list[dict[str, Any]] = field(default_factory=list)
    problems_found: list[dict[str, str]] = field(default_factory=list)

    # Designed by cleanup expert
    cleanup_fixes: list[dict[str, str]] = field(default_factory=list)

    # Validated by QA tester
    test_results: list[dict[str, str]] = field(default_factory=list)
    remaining_issues: list[dict[str, str]] = field(default_factory=list)

    # Created by database specialist
    target_table: str = "customers"
    create_table_sql: str = ""
    estimated_time: str = ""


# ============================================================================
# SIMULATED DATA SOURCES
# ============================================================================
# NOTE: Tools return hardcoded responses to keep this demo focused on the
# single vs multi-agent comparison, not on building a real ETL pipeline.
# In production, these would actually parse CSVs, validate data, etc.

SOURCE_DATA = {
    "customers.csv": {
        "rows": 1250,
        "preview": [
            "customer_id,name,email,signup_date,status",
            "1,Alice Smith,alice@test.com,2024-01-15,active",
            "2,Bob,bob@INVALID,01/20/2024,Active",  # Issues: date format, email, case
            "3,,null@test.com,2024-02-30,inactive",  # Issues: empty name, invalid date
        ],
        "detected_types": {
            "customer_id": "INTEGER",
            "name": "VARCHAR",
            "email": "VARCHAR",
            "signup_date": "DATE",
            "status": "VARCHAR",
        },
        "issues": {
            "signup_date": [
                "Mixed formats: YYYY-MM-DD and MM/DD/YYYY",
                "Invalid date: 2024-02-30",
            ],
            "email": ["Invalid format: bob@INVALID", "Null email: null@test.com"],
            "name": ["Empty values found"],
            "status": ["Case inconsistency: active vs Active"],
        },
    }
}


# ============================================================================
# TOOLS - Discovery and Analysis
# ============================================================================

@tool
def list_source_files(ctx: MigrationContext | None = None) -> str:
    """List CSV files from the legacy CRM export."""
    result = "Legacy CRM export files:\n- customers.csv (1250 rows)\n- exported on 2025-12-15"

    if ctx and isinstance(ctx, MigrationContext):
        ctx.source_file = "customers.csv"
        ctx.total_rows = 1250

    return result


@tool
def peek_at_csv(filename: str, ctx: MigrationContext | None = None) -> str:
    """Take a quick look at the CSV file structure."""
    if filename not in SOURCE_DATA:
        return f"File {filename} not found"

    data = SOURCE_DATA[filename]
    result = f"Quick peek at {filename}:\n{data['rows']} rows total\n\nColumns found:\n"
    for col, dtype in data["detected_types"].items():
        result += f"  - {col}: looks like {dtype}\n"

    result += "\nFirst few rows:\n"
    for row in data["preview"][:3]:
        result += f"  {row}\n"

    if ctx and isinstance(ctx, MigrationContext):
        ctx.columns_info = [
            {"name": col, "data_type": dtype} for col, dtype in data["detected_types"].items()
        ]

    return result


@tool
def spot_problems(filename: str, ctx: MigrationContext | None = None) -> str:
    """Look for obvious data quality problems."""
    if filename not in SOURCE_DATA:
        return f"File {filename} not found"

    issues = SOURCE_DATA[filename]["issues"]
    result = f"Problems spotted in {filename}:\n\n"
    for col, col_issues in issues.items():
        result += f"Column '{col}':\n"
        for issue in col_issues:
            result += f"  ⚠️  {issue}\n"

    if ctx and isinstance(ctx, MigrationContext):
        ctx.problems_found = [
            {"column": col, "issue": issue}
            for col, col_issues in issues.items()
            for issue in col_issues
        ]

    return result


# ============================================================================
# TOOLS - Figuring Out Fixes
# ============================================================================

@tool
def figure_out_fix(
    column_name: str,
    problem_description: str,
    ctx: MigrationContext | None = None,
) -> str:
    """Figure out how to fix a specific data problem."""
    fixes = {
        "date": "Handle both formats: CASE WHEN date LIKE '%/%' THEN parse as MM/DD/YYYY ELSE parse as YYYY-MM-DD",
        "email": "Lowercase everything, treat 'null@test.com' as actual NULL",
        "name": "Replace empty strings with 'Unknown Customer'",
        "status": "Normalize to lowercase (active, inactive)",
    }

    fix_applied = "No transformation needed"
    for key, fix in fixes.items():
        if key in column_name.lower() or key in problem_description.lower():
            fix_applied = fix
            break

    result = f"Fix for {column_name}:\n{fix_applied}\n\nWhy: {problem_description}"

    if ctx and isinstance(ctx, MigrationContext):
        ctx.cleanup_fixes.append(
            {"column": column_name, "problem": problem_description, "fix": fix_applied}
        )

    return result


# ============================================================================
# TOOLS - Testing Fixes
# ============================================================================

@tool
def test_fix(
    column: str,
    fix_description: str,
    ctx: MigrationContext | None = None,
) -> str:
    """Test if a fix actually works on sample data."""
    test_results = {
        "signup_date": "✓ Both date formats now handled correctly\n⚠️  Invalid date '2024-02-30' becomes NULL (3 rows affected)",
        "email": "✓ All emails lowercase now\n⚠️  'bob@invalid' still invalid - may need manual review (1 row)",
        "name": "✓ Empty names filled with 'Unknown Customer'",
        "status": "✓ All lowercase: 'active' or 'inactive'",
    }

    test_result = "Test passed"
    for key, result in test_results.items():
        if key in column.lower():
            test_result = result
            break

    result_str = f"Test results for {column}:\n{test_result}"

    if ctx and isinstance(ctx, MigrationContext):
        ctx.test_results.append(
            {"column": column, "fix": fix_description, "result": test_result}
        )
        if "⚠️" in test_result or "still invalid" in test_result:
            ctx.remaining_issues.append(
                {"column": column, "issue": "Partial fix - manual review needed"}
            )

    return result_str


# ============================================================================
# TOOLS - Preparing the Database
# ============================================================================

@tool
def write_create_table_sql(
    table_name: str,
    ctx: MigrationContext | None = None,
) -> str:
    """Write the CREATE TABLE SQL for the target database."""
    sql = f"""-- Target table for cleaned customer data
CREATE TABLE {table_name} (
    customer_id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    signup_date DATE,
    status VARCHAR(50) NOT NULL,
    migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_email ON {table_name}(email);
CREATE INDEX idx_status ON {table_name}(status);
CREATE INDEX idx_signup_date ON {table_name}(signup_date);"""

    if ctx and isinstance(ctx, MigrationContext):
        ctx.create_table_sql = sql
        ctx.target_table = table_name

    return sql


@tool
def estimate_migration_time(
    row_count: int,
    has_complex_transforms: bool,
    ctx: MigrationContext | None = None,
) -> str:
    """Estimate how long the migration will take."""
    base_time = row_count / 1000

    if has_complex_transforms:
        base_time *= 2.5

    estimate = f"Estimated migration time: ~{base_time:.1f} seconds for {row_count} rows"

    if ctx and isinstance(ctx, MigrationContext):
        ctx.estimated_time = estimate

    return estimate


# ============================================================================
# ALL TOOLS
# ============================================================================

DISCOVERY_TOOLS = [list_source_files, peek_at_csv, spot_problems]
FIX_TOOLS = [figure_out_fix]
TEST_TOOLS = [test_fix]
DATABASE_TOOLS = [write_create_table_sql, estimate_migration_time]

ALL_TOOLS = DISCOVERY_TOOLS + FIX_TOOLS + TEST_TOOLS + DATABASE_TOOLS


# ============================================================================
# THE TASK
# ============================================================================

TASK = """We need to migrate customer data from our legacy CRM export (customers.csv) into our new database.

The CSV is messy - it came from an old system. We need to:
- Figure out what's in the file and what's broken
- Clean up the data issues
- Make sure our fixes actually work
- Write the SQL to load it into the new database

Deliverable: A complete migration plan that's ready to execute."""


# ============================================================================
# APPROACH 1: SINGLE AGENT
# ============================================================================

async def single_agent_approach() -> dict:
    """Single agent with full 4-layer memory."""
    print("\n" + "=" * 70)
    print("APPROACH 1: Single Agent")
    print("=" * 70)

    observer = Observer(level="progress")

    agent = Agent(
        name="DataEngineer",
        model="gemini:gemini-2.5-pro",
        tools=ALL_TOOLS,
        conversation=True,
        acc=True,
        memory=True,
        cache=True,
        output=DataMigrationPlan,
        instructions="""You're a data engineer handling a messy CSV migration to a clean database.

You have tools for discovery, cleanup design, testing, and database planning. Use them as you see fit to create a complete migration plan.""",
        observer=observer,
    )

    start = time.perf_counter()
    result = await agent.run(TASK)
    duration = time.perf_counter() - start

    output = result.content.data if result.content and hasattr(result.content, "data") else None

    if output:
        print(f"\n📊 Migration Plan: {output.source_file} → {output.target_table}")
        print(f"Rows: {output.total_rows}")
        print(f"\nColumns ({len(output.columns)}):")
        for col in output.columns:
            print(f"  - {col.name} ({col.data_type})")
        print(f"\nCleanup Rules ({len(output.cleanup_rules)}):")
        for rule in output.cleanup_rules:
            print(f"  - {rule.source_column}: {rule.transform}")
        print(f"\nRemaining Issues ({len(output.remaining_issues)}):")
        for issue in output.remaining_issues:
            print(f"  - {issue.column}: {issue.issue_type} ({issue.severity})")
        print(f"\n✅ Ready: {output.ready_to_migrate}")
        print(f"📝 Next: {output.next_steps}")

    tokens = result.metadata.tokens.total_tokens if result.metadata and result.metadata.tokens else 0

    return {
        "approach": "single_agent",
        "duration_seconds": round(duration, 2),
        "tokens": tokens,
        "ready": output.ready_to_migrate if output else False,
        "output": output,
    }


# ============================================================================
# APPROACH 2: MULTI-AGENT
# ============================================================================

async def multi_agent_approach() -> dict:
    """Multi-agent with specialists coordinated via RunContext."""
    print("\n" + "=" * 70)
    print("APPROACH 2: Multi-Agent Specialists + RunContext")
    print("=" * 70)

    observer = Observer(level="progress")

    analyst = Agent(
        name="DataAnalyst",
        model="gemini:gemini-2.5-pro",
        tools=DISCOVERY_TOOLS,
        conversation=True,
        instructions="""You're a data analyst. Your job is to examine data sources and identify quality issues.

Work independently - use whatever tools you need to get a complete picture of the data.""",
    )

    cleaner = Agent(
        name="CleanupExpert",
        model="gemini:gemini-2.5-pro",
        tools=FIX_TOOLS,
        conversation=True,
        instructions="""You're a data cleanup specialist. You design transformations to fix data quality issues.

Work independently - figure out what fixes are needed and design them.""",
    )

    tester = Agent(
        name="QATester",
        model="gemini:gemini-2.5-pro",
        tools=TEST_TOOLS,
        conversation=True,
        instructions="""You're a QA specialist. You validate that proposed fixes actually work.

Work independently - test whatever fixes exist and report what still needs work.""",
    )

    db_specialist = Agent(
        name="DatabaseSpecialist",
        model="gemini:gemini-2.5-pro",
        tools=DATABASE_TOOLS,
        conversation=True,
        instructions="""You're a database specialist. You create schemas and plan database migrations.

Work independently - generate what's needed for the database setup.""",
    )

    coordinator = Agent(
        name="MigrationCoordinator",
        model="gemini:gemini-2.5-pro",
        conversation=True,
        output=DataMigrationPlan,
        instructions="""You're coordinating a data migration. You have access to specialists:

- DataAnalyst: Examines data sources and finds quality issues
- CleanupExpert: Designs transformations to fix data problems
- QATester: Validates that fixes work correctly
- DatabaseSpecialist: Creates database schemas and migration plans

Delegate to specialists as needed to complete the migration plan. You decide who to call, when, and with what information. Work naturally - the workflow should emerge from what you discover.""",
        # Simply pass the agents - their names become the tool names
        subagents=[analyst, cleaner, tester, db_specialist],
        observer=observer,
    )

    migration_ctx = MigrationContext()

    start = time.perf_counter()
    result = await coordinator.run(TASK, context=migration_ctx)
    duration = time.perf_counter() - start

    output = result.content.data if result.content and hasattr(result.content, "data") else None

    if output:
        print(f"\n📊 Migration Plan: {output.source_file} → {output.target_table}")
        print(f"Rows: {output.total_rows}")
        print(f"\nColumns ({len(output.columns)}):")
        for col in output.columns:
            print(f"  - {col.name} ({col.data_type})")
        print(f"\nCleanup Rules ({len(output.cleanup_rules)}):")
        for rule in output.cleanup_rules:
            print(f"  - {rule.source_column}: {rule.transform}")
        print(f"\nRemaining Issues ({len(output.remaining_issues)}):")
        for issue in output.remaining_issues:
            print(f"  - {issue.column}: {issue.issue_type} ({issue.severity})")
        print(f"\n✅ Ready: {output.ready_to_migrate}")
        print(f"📝 Next: {output.next_steps}")

    # Token counting with subagent metadata aggregation
    total_tokens = result.metadata.tokens.total_tokens if result.metadata and result.metadata.tokens else 0

    return {
        "approach": "multi_agent",
        "duration_seconds": round(duration, 2),
        "tokens": total_tokens,
        "ready": output.ready_to_migrate if output else False,
        "output": output,
    }


# ============================================================================
# COMPARISON
# ============================================================================

async def main() -> None:
    print("\n" + "🎯 " * 25)
    print("REALISTIC COMPARISON: Single Agent vs Multi-Agent")
    print("🎯 " * 25)
    print("\n📋 SCENARIO: Migrate messy CSV data to a clean database")
    print("🔗 Natural workflow: Discover issues → Fix them → Test → Load")
    print("💾 Memory matters: Each step builds on what we learned before\n")

    results = []

    print("\n🚀 Running Single Agent approach...")
    results.append(await single_agent_approach())

    print("\n\n🚀 Running Multi-Agent approach...")
    results.append(await multi_agent_approach())

    print("\n" + "=" * 80)
    print("📊 RESULTS COMPARISON")
    print("=" * 80)
    print()

    headers = f"{'Approach':<25} {'Duration':<12} {'Tokens':<12} {'Ready'}"
    print(headers)
    print("-" * 60)

    for r in results:
        approach = r["approach"].replace("_", " ").title()
        duration_str = f"{r['duration_seconds']:.2f}s"
        tokens_str = str(r.get("tokens", "N/A"))
        ready = "✅ Yes" if r["ready"] else "❌ No"
        print(f"{approach:<25} {duration_str:<12} {tokens_str:<12} {ready}")

    print("\n" + "=" * 80)

    single = results[0]
    multi = results[1]

    duration_ratio = (
        multi["duration_seconds"] / single["duration_seconds"]
        if single["duration_seconds"] > 0
        else 1
    )
    token_ratio = (
        multi["tokens"] / single["tokens"] if single["tokens"] > 0 else 1
    )

    print(f"Duration Ratio (Multi/Single): {duration_ratio:.2f}x")
    print(f"Token Ratio (Multi/Single): {token_ratio:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
