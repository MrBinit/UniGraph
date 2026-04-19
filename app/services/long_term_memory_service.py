from datetime import datetime, timezone
from uuid import uuid4
from app.core.config import get_settings
from app.schemas.long_term_memory_schema import LongTermMemoryRecord, LongTermMemoryWrite

settings = get_settings()


def store_long_term_memory(
    record: LongTermMemoryWrite,
    *,
    embedding: list[float] | None = None,
) -> LongTermMemoryRecord:
    """Store one durable long-term memory record through the repository layer."""
    if not settings.postgres.enabled:
        now = datetime.now(timezone.utc).isoformat()
        return LongTermMemoryRecord(
            id=f"memory-disabled-{uuid4()}",
            created_at=now,
            updated_at=now,
            embedding=embedding,
            **record.model_dump(),
        )
    from app.repositories.long_term_memory_repository import upsert_long_term_memory

    return upsert_long_term_memory(record, embedding=embedding)


def get_long_term_memories(
    user_id: str,
    *,
    limit: int | None = None,
) -> list[LongTermMemoryRecord]:
    """Fetch recent long-term memories for the given user."""
    if not settings.postgres.enabled:
        return []
    from app.repositories.long_term_memory_repository import list_long_term_memories

    return list_long_term_memories(user_id, limit=limit)


def find_long_term_memories(
    user_id: str,
    *,
    embedding: list[float],
    limit: int | None = None,
    memory_types: list[str] | None = None,
) -> list[LongTermMemoryRecord]:
    """Search relevant long-term memories for a user by embedding similarity."""
    if not settings.postgres.enabled:
        return []
    from app.repositories.long_term_memory_repository import search_long_term_memories

    return search_long_term_memories(
        user_id,
        embedding=embedding,
        limit=limit,
        memory_types=memory_types,
    )
