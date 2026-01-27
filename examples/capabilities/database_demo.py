"""Database capability demonstration.

This example shows how to use the Database capability for:
- Safe SQL query execution
- Parameterized queries (SQL injection prevention)
- SQLite operations
- Read-only mode
- Query validation

Run with: uv run python examples/capabilities/database_demo.py
"""

import asyncio
import tempfile
from pathlib import Path

from cogent.capabilities.database import Database


async def demo_basic_queries():
    """Demonstrate basic query operations."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Query Operations")
    print("=" * 60)
    
    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    # Connect to SQLite database
    db = Database(connection_string=f"sqlite+aiosqlite:///{db_path}")
    
    try:
        # Create table
        print("\n1. Creating users table...")
        await db.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                age INTEGER
            )
        """)
        print("✓ Table created")
        
        # Insert data
        print("\n2. Inserting users...")
        await db.execute(
            "INSERT INTO users (name, email, age) VALUES (:name, :email, :age)",
            params={"name": "Alice", "email": "alice@example.com", "age": 30}
        )
        await db.execute(
            "INSERT INTO users (name, email, age) VALUES (:name, :email, :age)",
            params={"name": "Bob", "email": "bob@example.com", "age": 25}
        )
        await db.execute(
            "INSERT INTO users (name, email, age) VALUES (:name, :email, :age)",
            params={"name": "Charlie", "email": "charlie@example.com", "age": 35}
        )
        print("✓ 3 users inserted")
        
        # Query all users
        print("\n3. Querying all users...")
        result = await db.query("SELECT * FROM users")
        print(f"✓ Found {result.row_count} users")
        for row in result.rows:
            print(f"  - {dict(row)}")
        
        # Query with parameters
        print("\n4. Querying users with age > 25...")
        result = await db.query(
            "SELECT name, email, age FROM users WHERE age > :min_age",
            params={"min_age": 25}
        )
        print(f"✓ Found {result.row_count} users")
        for row in result.rows:
            print(f"  - {dict(row)}")
        
        print("\n✅ Basic queries demo complete!")
        
    finally:
        await db.close()
        Path(db_path).unlink(missing_ok=True)


async def demo_parameterized_queries():
    """Demonstrate SQL injection prevention with parameterized queries."""
    print("\n" + "=" * 60)
    print("DEMO 2: Parameterized Queries (SQL Injection Prevention)")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    db = Database(connection_string=f"sqlite+aiosqlite:///{db_path}")
    
    try:
        # Setup
        await db.execute("""
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                balance REAL
            )
        """)
        await db.execute(
            "INSERT INTO accounts (username, balance) VALUES (:username, :balance)",
            params={"username": "alice", "balance": 1000.0}
        )
        await db.execute(
            "INSERT INTO accounts (username, balance) VALUES (:username, :balance)",
            params={"username": "bob", "balance": 500.0}
        )
        
        print("\n1. Safe parameterized query...")
        user_input = "alice"  # Simulated user input
        result = await db.query(
            "SELECT * FROM accounts WHERE username = :username",
            params={"username": user_input}
        )
        print(f"✓ Query executed safely, found {result.row_count} account(s)")
        for row in result.rows:
            print(f"  {dict(row)}")
        
        print("\n2. Attempting SQL injection (will be prevented)...")
        malicious_input = "alice' OR '1'='1"  # Classic SQL injection attempt
        result = await db.query(
            "SELECT * FROM accounts WHERE username = :username",
            params={"username": malicious_input}
        )
        # Parameters are escaped, so this searches for literal string "alice' OR '1'='1"
        print(f"✓ Query executed safely, found {result.row_count} account(s)")
        print("  (Malicious input was treated as literal string, not code)")
        
        print("\n✅ Parameterized queries demo complete!")
        
    finally:
        await db.close()
        Path(db_path).unlink(missing_ok=True)


async def demo_read_only_mode():
    """Demonstrate read-only mode."""
    print("\n" + "=" * 60)
    print("DEMO 3: Read-Only Mode")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    # Create database with some data
    db_write = Database(connection_string=f"sqlite+aiosqlite:///{db_path}")
    try:
        await db_write.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL
            )
        """)
        await db_write.execute(
            "INSERT INTO products (name, price) VALUES (:name, :price)",
            params={"name": "Widget", "price": 9.99}
        )
        print("✓ Database created with test data")
    finally:
        await db_write.close()
    
    # Connect in read-only mode
    db_readonly = Database(
        connection_string=f"sqlite+aiosqlite:///{db_path}",
        read_only=True
    )
    
    try:
        print("\n1. Querying in read-only mode...")
        result = await db_readonly.query("SELECT * FROM products")
        print(f"✓ Query successful, found {result.row_count} product(s)")
        for row in result.rows:
            print(f"  {dict(row)}")
        
        print("\n2. Attempting INSERT in read-only mode...")
        try:
            await db_readonly.execute(
                "INSERT INTO products (name, price) VALUES (:name, :price)",
                params={"name": "Gadget", "price": 19.99}
            )
            print("❌ INSERT should have been blocked!")
        except ValueError as e:
            print(f"✓ INSERT blocked: {e}")
        
        print("\n✅ Read-only mode demo complete!")
        
    finally:
        await db_readonly.close()
        Path(db_path).unlink(missing_ok=True)


async def demo_query_validation():
    """Demonstrate dangerous query validation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Dangerous Query Validation")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    db = Database(connection_string=f"sqlite+aiosqlite:///{db_path}")
    
    try:
        # Setup
        await db.execute("""
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT
            )
        """)
        
        print("\n1. Safe query (SELECT)...")
        result = await db.query("SELECT * FROM logs")
        print(f"✓ Query executed, found {result.row_count} log(s)")
        
        print("\n2. Attempting DROP TABLE (will be blocked)...")
        try:
            await db.query("DROP TABLE logs")
            print("❌ DROP should have been blocked!")
        except ValueError as e:
            print(f"✓ DROP blocked: {e}")
        
        print("\n3. Attempting TRUNCATE (will be blocked)...")
        try:
            await db.execute("TRUNCATE TABLE logs")
            print("❌ TRUNCATE should have been blocked!")
        except ValueError as e:
            print(f"✓ TRUNCATE blocked: {e}")
        
        print("\n4. Attempting DELETE FROM (will be blocked)...")
        try:
            await db.execute("DELETE FROM logs")
            print("❌ DELETE FROM should have been blocked!")
        except ValueError as e:
            print(f"✓ DELETE FROM blocked: {e}")
        
        print("\n5. Parameterized DELETE (safe, allowed)...")
        await db.execute(
            "DELETE FROM logs WHERE id = :id",
            params={"id": 999}
        )
        print("✓ Parameterized DELETE allowed (safe with WHERE clause)")
        
        print("\n✅ Query validation demo complete!")
        
    finally:
        await db.close()
        Path(db_path).unlink(missing_ok=True)


async def demo_tool_usage():
    """Demonstrate using Database as agent tools."""
    print("\n" + "=" * 60)
    print("DEMO 5: Database Tools for Agents")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    db = Database(connection_string=f"sqlite+aiosqlite:///{db_path}")
    
    try:
        # Setup
        await db.execute("""
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                completed INTEGER DEFAULT 0
            )
        """)
        
        print("\n1. Getting tool functions...")
        tools = db.tools
        print(f"✓ Registered {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        print("\n2. Using sql_query tool...")
        # Simulate agent calling sql_query
        query_tool = next(t for t in tools if t.name == "sql_query")
        result_str = await query_tool.ainvoke(
            args={
                "query": "SELECT * FROM tasks",
                "params": "{}"
            }
        )
        print(f"✓ Tool result:\n{result_str}")
        
        print("\n3. Using sql_insert tool...")
        # Simulate agent calling sql_insert
        insert_tool = next(t for t in tools if t.name == "sql_insert")
        result_str = await insert_tool.ainvoke(
            args={
                "table": "tasks",
                "data": '{"title": "Write documentation", "completed": 0}'
            }
        )
        print(f"✓ Tool result:\n{result_str}")
        
        print("\n4. Verifying insertion...")
        result_str = await query_tool.ainvoke(
            args={
                "query": "SELECT * FROM tasks",
                "params": "{}"
            }
        )
        print(f"✓ Tool result:\n{result_str}")
        
        print("\n✅ Tool usage demo complete!")
        
    finally:
        await db.close()
        Path(db_path).unlink(missing_ok=True)


async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("DATABASE CAPABILITY DEMONSTRATION")
    print("=" * 60)
    
    await demo_basic_queries()
    await demo_parameterized_queries()
    await demo_read_only_mode()
    await demo_query_validation()
    await demo_tool_usage()
    
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
