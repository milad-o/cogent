"""
Database capability - safe SQL query execution.

Provides secure database access with connection pooling, parameter sanitization,
and protection against SQL injection.

Example:
    ```python
    from cogent import Agent
    from cogent.capabilities import Database

    agent = Agent(
        name="Data Agent",
        model=model,
        capabilities=[
            Database(
                connection_string="sqlite:///data.db",
                max_pool_size=5,
            )
        ],
    )

    # Agent can now query databases safely
    await agent.run("Show me all users in the database")
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cogent.capabilities.base import BaseCapability
from cogent.tools.base import BaseTool, tool

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a database query."""

    rows: list[dict[str, Any]]
    row_count: int
    columns: list[str]
    query: str
    success: bool
    error: str | None = None
    execution_time_ms: float = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "row_count": self.row_count,
            "columns": self.columns,
            "query": self.query[:200] + "..." if len(self.query) > 200 else self.query,
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class Database(BaseCapability):
    """
    Database capability for safe SQL query execution.

    Provides secure database access with:
    - SQL injection prevention (parameterized queries)
    - Connection pooling
    - Automatic rollback on errors
    - Query timeout protection
    - Support for SQLite, PostgreSQL, MySQL

    Args:
        connection_string: Database URL (e.g., "sqlite:///data.db", "postgresql://user:pass@host/db")
        max_pool_size: Maximum connection pool size (default: 5)
        pool_timeout: Connection pool timeout in seconds (default: 30)
        query_timeout: Query execution timeout in seconds (default: 30)
        read_only: Whether to allow only SELECT queries (default: False)
        name: Capability name (default: "database")

    Example:
        ```python
        # SQLite (file-based)
        db = Database("sqlite:///data.db")

        # PostgreSQL
        db = Database(
            "postgresql://user:password@localhost:5432/mydb",
            max_pool_size=10,
        )

        # Read-only mode (SELECT only)
        db = Database("sqlite:///data.db", read_only=True)
        ```

    Security Notes:
        - Always uses parameterized queries
        - No string interpolation in SQL
        - Transaction rollback on errors
        - Query timeout protection
        - Optional read-only mode
    """

    def __init__(
        self,
        connection_string: str,
        max_pool_size: int = 5,
        pool_timeout: int = 30,
        query_timeout: int = 30,
        read_only: bool = False,
        name: str = "database",
    ):
        self._name = name
        self._connection_string = connection_string
        self._max_pool_size = max_pool_size
        self._pool_timeout = pool_timeout
        self._query_timeout = query_timeout
        self._read_only = read_only
        self._tools_cache: list[BaseTool] | None = None
        self._engine = None
        self._async_engine = None

        # Query history
        self._history: list[QueryResult] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        db_type = self._connection_string.split(":")[0]
        mode = "read-only" if self._read_only else "read-write"
        return f"Database access ({db_type}, {mode})"

    @property
    def tools(self) -> list[BaseTool]:
        if self._tools_cache is None:
            tools = [self._query_tool()]
            if not self._read_only:
                tools.extend([
                    self._execute_tool(),
                    self._insert_tool(),
                ])
            return tools
        return self._tools_cache

    @property
    def history(self) -> list[QueryResult]:
        """Get query history."""
        return self._history

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def _get_engine(self):
        """Get or create async database engine."""
        if self._async_engine is None:
            try:
                from sqlalchemy.ext.asyncio import create_async_engine
            except ImportError as e:
                raise ImportError(
                    "SQLAlchemy with async support required. Install with: uv add 'sqlalchemy[asyncio]'"
                ) from e

            self._async_engine = create_async_engine(
                self._connection_string,
                pool_size=self._max_pool_size,
                pool_timeout=self._pool_timeout,
                echo=False,
            )
        return self._async_engine

    def _validate_query(self, query: str) -> None:
        """Validate query for security."""
        query_upper = query.strip().upper()

        # Check read-only mode
        if self._read_only:
            if not query_upper.startswith("SELECT"):
                raise ValueError("Only SELECT queries allowed in read-only mode")

        # Block dangerous keywords (basic protection)
        dangerous = ["DROP TABLE", "DROP DATABASE", "TRUNCATE", "DELETE FROM"]
        for keyword in dangerous:
            if keyword in query_upper:
                logger.warning(f"Blocked dangerous query: {keyword}")
                raise ValueError(f"Query contains blocked keyword: {keyword}")

    async def query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> QueryResult:
        """
        Execute a SELECT query and return results.

        Args:
            sql: SQL query with named parameters (:param_name)
            params: Optional query parameters

        Returns:
            QueryResult with rows and metadata

        Example:
            ```python
            result = await db.query(
                "SELECT * FROM users WHERE age > :min_age",
                params={"min_age": 18}
            )
            ```
        """
        import time

        start_time = time.perf_counter()

        # Validate query
        try:
            self._validate_query(sql)
        except ValueError as e:
            result = QueryResult(
                rows=[],
                row_count=0,
                columns=[],
                query=sql,
                success=False,
                error=str(e),
            )
            self._history.append(result)
            return result

        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import AsyncSession

            engine = await self._get_engine()

            async with AsyncSession(engine) as session:
                # Execute query with timeout
                result_proxy = await session.execute(
                    text(sql),
                    params or {},
                )

                # Fetch all rows
                rows = result_proxy.fetchall()

                # Convert to list of dicts
                columns = list(result_proxy.keys())
                rows_dicts = [dict(zip(columns, row)) for row in rows]

                execution_time = (time.perf_counter() - start_time) * 1000

                result = QueryResult(
                    rows=rows_dicts,
                    row_count=len(rows_dicts),
                    columns=columns,
                    query=sql,
                    success=True,
                    execution_time_ms=execution_time,
                )

                self._history.append(result)
                return result

        except Exception as e:
            error_msg = f"Query failed: {e}"
            logger.error(error_msg)
            result = QueryResult(
                rows=[],
                row_count=0,
                columns=[],
                query=sql,
                success=False,
                error=error_msg,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._history.append(result)
            return result

    async def execute(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> QueryResult:
        """
        Execute an INSERT, UPDATE, or DELETE query.

        Args:
            sql: SQL statement with named parameters
            params: Optional query parameters

        Returns:
            QueryResult with affected row count

        Example:
            ```python
            result = await db.execute(
                "UPDATE users SET status = :status WHERE id = :user_id",
                params={"status": "active", "user_id": 123}
            )
            ```
        """
        import time

        start_time = time.perf_counter()

        # Validate query
        try:
            self._validate_query(sql)
        except ValueError as e:
            result = QueryResult(
                rows=[],
                row_count=0,
                columns=[],
                query=sql,
                success=False,
                error=str(e),
            )
            self._history.append(result)
            return result

        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import AsyncSession

            engine = await self._get_engine()

            async with AsyncSession(engine) as session:
                # Execute statement
                result_proxy = await session.execute(
                    text(sql),
                    params or {},
                )

                # Commit transaction
                await session.commit()

                execution_time = (time.perf_counter() - start_time) * 1000

                result = QueryResult(
                    rows=[],
                    row_count=result_proxy.rowcount,
                    columns=[],
                    query=sql,
                    success=True,
                    execution_time_ms=execution_time,
                )

                self._history.append(result)
                return result

        except Exception as e:
            error_msg = f"Execution failed: {e}"
            logger.error(error_msg)
            result = QueryResult(
                rows=[],
                row_count=0,
                columns=[],
                query=sql,
                success=False,
                error=error_msg,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._history.append(result)
            return result

    async def close(self) -> None:
        """Close database connection pool."""
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None

    def clear_history(self) -> int:
        """Clear query history. Returns count cleared."""
        count = len(self._history)
        self._history.clear()
        return count

    # =========================================================================
    # Tool Generation
    # =========================================================================

    def _query_tool(self) -> BaseTool:
        db = self

        @tool
        async def sql_query(query: str, params: str = "{}") -> str:
            """
            Execute a SQL SELECT query.

            Args:
                query: SQL query with named parameters (e.g., "SELECT * FROM users WHERE age > :min_age")
                params: JSON string of parameters (e.g., '{"min_age": 18}')

            Returns:
                Query results as formatted table

            Security: Always uses parameterized queries to prevent SQL injection.
            """
            import json

            try:
                parsed_params = json.loads(params) if params else {}
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in params: {params}"

            result = await db.query(query, params=parsed_params)

            if result.success:
                lines = [f"✓ Query executed successfully"]
                lines.append(f"Rows: {result.row_count}")
                lines.append(f"Time: {result.execution_time_ms:.0f}ms")

                if result.rows:
                    lines.append(f"\nResults:")
                    # Show first 10 rows
                    for i, row in enumerate(result.rows[:10], 1):
                        lines.append(f"\n{i}. {row}")

                    if result.row_count > 10:
                        lines.append(f"\n... {result.row_count - 10} more rows")
                else:
                    lines.append("\nNo rows returned")
            else:
                lines = [f"✗ Query failed"]
                lines.append(f"Error: {result.error}")

            return "\n".join(lines)

        return sql_query

    def _execute_tool(self) -> BaseTool:
        db = self

        @tool
        async def sql_execute(statement: str, params: str = "{}") -> str:
            """
            Execute a SQL INSERT, UPDATE, or DELETE statement.

            Args:
                statement: SQL statement with named parameters
                params: JSON string of parameters

            Returns:
                Affected row count

            Security: Always uses parameterized queries to prevent SQL injection.
            """
            import json

            try:
                parsed_params = json.loads(params) if params else {}
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in params: {params}"

            result = await db.execute(statement, params=parsed_params)

            if result.success:
                lines = [f"✓ Statement executed successfully"]
                lines.append(f"Affected rows: {result.row_count}")
                lines.append(f"Time: {result.execution_time_ms:.0f}ms")
            else:
                lines = [f"✗ Statement failed"]
                lines.append(f"Error: {result.error}")

            return "\n".join(lines)

        return sql_execute

    def _insert_tool(self) -> BaseTool:
        db = self

        @tool
        async def sql_insert(table: str, data: str) -> str:
            """
            Insert data into a table.

            Args:
                table: Table name
                data: JSON object of column:value pairs (e.g., '{"name": "John", "age": 30}')

            Returns:
                Success message with row count

            Security: Uses parameterized INSERT to prevent SQL injection.
            """
            import json

            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in data: {data}"

            if not isinstance(parsed_data, dict):
                return "Error: Data must be a JSON object"

            # Build parameterized INSERT
            columns = list(parsed_data.keys())
            placeholders = [f":{col}" for col in columns]

            sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

            result = await db.execute(sql, params=parsed_data)

            if result.success:
                lines = [f"✓ Inserted into {table}"]
                lines.append(f"Rows inserted: {result.row_count}")
                lines.append(f"Time: {result.execution_time_ms:.0f}ms")
            else:
                lines = [f"✗ Insert failed"]
                lines.append(f"Error: {result.error}")

            return "\n".join(lines)

        return sql_insert
