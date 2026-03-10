import logging
import time
import asyncio

from app.repositories.document_chunk_repository import (
    resolve_document_chunk_search_strategy,
    search_document_chunks,
)
from app.services.embedding_service import embed_text

logger = logging.getLogger(__name__)


def retrieve_document_chunks(
    query: str,
    *,
    top_k: int = 3,
    metadata_filters: dict[str, str] | None = None,
) -> dict:
    """Embed a query, run retrieval, and return results with latency breakdowns."""
    started_at = time.perf_counter()
    retrieval_strategy = resolve_document_chunk_search_strategy(metadata_filters)

    embedding_started_at = time.perf_counter()
    query_embedding = embed_text(query)
    embedding_ms = max(0, int((time.perf_counter() - embedding_started_at) * 1000))

    db_started_at = time.perf_counter()
    results = search_document_chunks(
        embedding=query_embedding,
        limit=top_k,
        metadata_filters=metadata_filters,
    )
    db_ms = max(0, int((time.perf_counter() - db_started_at) * 1000))

    total_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    logger.info(
        "RetrievalLatency | strategy=%s | top_k=%s | results=%s | embedding_ms=%s | db_ms=%s | total_ms=%s | filters=%s",
        retrieval_strategy,
        top_k,
        len(results),
        embedding_ms,
        db_ms,
        total_ms,
        metadata_filters or {},
    )

    return {
        "query": query,
        "top_k": top_k,
        "retrieval_strategy": retrieval_strategy,
        "metadata_filters": metadata_filters or {},
        "timings_ms": {
            "embedding": embedding_ms,
            "database": db_ms,
            "total": total_ms,
        },
        "results": results,
    }


async def aretrieve_document_chunks(
    query: str,
    *,
    top_k: int = 3,
    metadata_filters: dict[str, str] | None = None,
) -> dict:
    """Run retrieval without blocking the event loop."""
    return await asyncio.to_thread(
        retrieve_document_chunks,
        query,
        top_k=top_k,
        metadata_filters=metadata_filters,
    )
