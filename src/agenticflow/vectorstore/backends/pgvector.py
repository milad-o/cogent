"""PostgreSQL pgvector backend.

Production-grade vector storage using PostgreSQL with pgvector extension.
Requires: pip install psycopg[binary] pgvector

Best for:
- Existing PostgreSQL infrastructure
- ACID compliance requirements
- SQL-based filtering and joins
- Hybrid search (vector + full-text)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agenticflow.vectorstore.base import SearchResult
from agenticflow.vectorstore.document import Document


@dataclass
class PgVectorBackend:
    """PostgreSQL pgvector backend.
    
    Uses PostgreSQL with pgvector extension for vector similarity search.
    
    Attributes:
        connection_string: PostgreSQL connection string.
        table_name: Name of the table. Default: "documents".
        dimension: Embedding dimension. Default: 1536.
        distance: Distance metric ("cosine", "l2", "inner"). Default: "cosine".
        
    Example:
        backend = PgVectorBackend(
            connection_string="postgresql://user:pass@localhost/db",
            table_name="my_documents",
            dimension=1536
        )
        await backend.add(ids, embeddings, documents)
        results = await backend.search(query_embedding, k=10)
    """
    
    connection_string: str
    table_name: str = "documents"
    dimension: int = 1536
    distance: str = "cosine"
    
    _conn: Any = field(default=None, init=False, repr=False)
    _pool: Any = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize database connection and table."""
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:
            msg = "psycopg or pgvector not installed. Install with: pip install psycopg[binary] pgvector"
            raise ImportError(msg) from e
        
        # Create connection
        self._conn = psycopg.connect(self.connection_string)
        register_vector(self._conn)
        
        # Initialize schema
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Create pgvector extension if not exists
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.dimension}),
                    text TEXT,
                    metadata JSONB
                )
            """)
            
            # Create index based on distance metric
            index_name = f"{self.table_name}_embedding_idx"
            
            if self.distance == "cosine":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            elif self.distance == "l2":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.table_name}
                    USING ivfflat (embedding vector_l2_ops)
                    WITH (lists = 100)
                """)
            else:  # inner product
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.table_name}
                    USING ivfflat (embedding vector_ip_ops)
                    WITH (lists = 100)
                """)
            
            self._conn.commit()
    
    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[Document],
    ) -> None:
        """Add documents with their embeddings."""
        if not ids:
            return
        
        with self._conn.cursor() as cur:
            for doc_id, embedding, doc in zip(ids, embeddings, documents):
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, embedding, text, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata
                    """,
                    (doc_id, embedding, doc.text, json.dumps(doc.metadata)),
                )
            
            self._conn.commit()
    
    async def search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        # Build distance operator based on metric
        if self.distance == "cosine":
            distance_op = "<=>"  # Cosine distance
            score_expr = f"1 - (embedding <=> %s::vector)"
        elif self.distance == "l2":
            distance_op = "<->"  # L2 distance
            score_expr = f"1 / (1 + (embedding <-> %s::vector))"
        else:  # inner product
            distance_op = "<#>"  # Negative inner product
            score_expr = f"-(embedding <#> %s::vector)"
        
        # Build WHERE clause for filter
        where_clause = ""
        filter_params: list[Any] = []
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(f"metadata->>'{key}' = %s")
                filter_params.append(str(value))
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
            SELECT id, text, metadata, {score_expr} as score
            FROM {self.table_name}
            {where_clause}
            ORDER BY embedding {distance_op} %s::vector
            LIMIT %s
        """
        
        with self._conn.cursor() as cur:
            params = [embedding] + filter_params + [embedding, k]
            cur.execute(query, params)
            rows = cur.fetchall()
        
        results = []
        for row in rows:
            doc = Document(
                text=row[1],
                metadata=row[2] or {},
                id=row[0],
            )
            results.append(SearchResult(
                document=doc,
                score=float(row[3]),
                id=row[0],
            ))
        
        return results
    
    async def delete(self, ids: list[str]) -> bool:
        """Delete documents by ID."""
        if not ids:
            return False
        
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
                (ids,),
            )
            deleted = cur.rowcount > 0
            self._conn.commit()
        
        return deleted
    
    async def clear(self) -> None:
        """Remove all documents from the store."""
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table_name}")
            self._conn.commit()
    
    async def get(self, ids: list[str]) -> list[Document]:
        """Get documents by ID."""
        if not ids:
            return []
        
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata FROM {self.table_name} WHERE id = ANY(%s)",
                (ids,),
            )
            rows = cur.fetchall()
        
        return [
            Document(text=row[1], metadata=row[2] or {}, id=row[0])
            for row in rows
        ]
    
    def count(self) -> int:
        """Return the number of documents in the store."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cur.fetchone()[0]
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
