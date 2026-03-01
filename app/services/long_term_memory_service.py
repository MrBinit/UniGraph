from app.repositories.long_term_memory_repository import (
    list_long_term_memories,
    search_long_term_memories,
    upsert_long_term_memory,
)
from app.schemas.long_term_memory_schema import LongTermMemoryRecord, LongTermMemoryWrite


def store_long_term_memory(
    record: LongTermMemoryWrite,
    *,
    embedding: list[float] | None = None,
) -> LongTermMemoryRecord:
    """Store one durable long-term memory record through the repository layer."""
    return upsert_long_term_memory(record, embedding=embedding)


def get_long_term_memories(
    user_id: str,
    *,
    limit: int | None = None,
) -> list[LongTermMemoryRecord]:
    """Fetch recent long-term memories for the given user."""
    return list_long_term_memories(user_id, limit=limit)


def find_long_term_memories(
    user_id: str,
    *,
    embedding: list[float],
    limit: int | None = None,
    memory_types: list[str] | None = None,
) -> list[LongTermMemoryRecord]:
    """Search relevant long-term memories for a user by embedding similarity."""
    return search_long_term_memories(
        user_id,
        embedding=embedding,
        limit=limit,
        memory_types=memory_types,
    )
