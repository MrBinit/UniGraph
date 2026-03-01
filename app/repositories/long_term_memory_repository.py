import json
from app.core.config import get_settings
from app.infra.postgres_client import get_postgres_pool
from app.schemas.long_term_memory_schema import LongTermMemoryRecord, LongTermMemoryWrite

settings = get_settings()


def _qualified_table() -> str:
    """Return the fully qualified long-term memory table name from config."""
    return f"{settings.postgres.schema_name}.{settings.postgres.memory_table}"


def _vector_literal(embedding: list[float] | None) -> str | None:
    """Convert a Python embedding list into pgvector literal syntax."""
    if not embedding:
        return None
    values = ",".join(f"{float(value):.8f}" for value in embedding)
    return f"[{values}]"


def _row_to_record(row: dict) -> LongTermMemoryRecord:
    """Convert a database row into the typed long-term memory record schema."""
    return LongTermMemoryRecord(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        memory_key=str(row["memory_key"]),
        memory_type=str(row["memory_type"]),
        content=str(row["content"]),
        source=str(row["source"]),
        confidence=float(row["confidence"]),
        metadata=row.get("metadata") or {},
        embedding=row.get("embedding"),
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


def upsert_long_term_memory(
    record: LongTermMemoryWrite,
    *,
    embedding: list[float] | None = None,
) -> LongTermMemoryRecord:
    """Insert or update one long-term memory row using the live table contract."""

    pool = get_postgres_pool()
    sql = f"""
        INSERT INTO {_qualified_table()} (
            user_id,
            memory_key,
            memory_type,
            content,
            source,
            confidence,
            metadata,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
        ON CONFLICT (user_id, memory_key)
        DO UPDATE SET
            memory_type = EXCLUDED.memory_type,
            content = EXCLUDED.content,
            source = EXCLUDED.source,
            confidence = EXCLUDED.confidence,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            updated_at = now()
        RETURNING
            id,
            user_id,
            memory_key,
            memory_type,
            content,
            source,
            confidence,
            metadata,
            embedding,
            created_at,
            updated_at
    """
    params = (
        record.user_id,
        record.memory_key,
        record.memory_type,
        record.content,
        record.source,
        record.confidence,
        json.dumps(record.metadata),
        _vector_literal(embedding),
    )

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        conn.commit()

    return _row_to_record(row or {})


def list_long_term_memories(
    user_id: str,
    *,
    limit: int | None = None,
) -> list[LongTermMemoryRecord]:
    """List the most recent long-term memories for a single user."""

    top_k = limit or settings.postgres.default_top_k
    pool = get_postgres_pool()
    sql = f"""
        SELECT
            id,
            user_id,
            memory_key,
            memory_type,
            content,
            source,
            confidence,
            metadata,
            embedding,
            created_at,
            updated_at
        FROM {_qualified_table()}
        WHERE user_id = %s
        ORDER BY updated_at DESC, created_at DESC
        LIMIT %s
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, top_k))
            rows = cur.fetchall()

    return [_row_to_record(row) for row in rows]


def search_long_term_memories(
    user_id: str,
    *,
    embedding: list[float],
    limit: int | None = None,
    memory_types: list[str] | None = None,
) -> list[LongTermMemoryRecord]:
    """Search a user's vectorized memories ordered by pgvector cosine distance."""

    top_k = limit or settings.postgres.default_top_k
    pool = get_postgres_pool()
    vector_value = _vector_literal(embedding)
    where_parts = ["user_id = %s", "embedding IS NOT NULL"]
    params: list[object] = [user_id]

    if memory_types:
        where_parts.append("memory_type = ANY(%s)")
        params.append(memory_types)

    params.extend([vector_value, top_k])
    where_clause = " AND ".join(where_parts)
    sql = f"""
        SELECT
            id,
            user_id,
            memory_key,
            memory_type,
            content,
            source,
            confidence,
            metadata,
            embedding,
            created_at,
            updated_at
        FROM {_qualified_table()}
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    return [_row_to_record(row) for row in rows]
