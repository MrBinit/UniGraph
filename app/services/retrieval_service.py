import logging
import time

from app.repositories.document_chunk_repository import (
    resolve_document_chunk_search_strategy,
    search_document_chunks,
    search_document_chunks_async,
)
from app.infra.io_limiters import dependency_limiter
from app.services.embedding_service import aembed_text, embed_text

logger = logging.getLogger(__name__)


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _build_retrieval_response(
    *,
    query: str,
    top_k: int,
    metadata_filters: dict[str, str] | None,
    retrieval_strategy: str,
    embedding_ms: int,
    db_ms: int,
    started_at: float,
    results: list[dict],
) -> dict:
    total_ms = _elapsed_ms(started_at)
    logger.info(
        (
            "RetrievalLatency | strategy=%s | top_k=%s | results=%s "
            "| embedding_ms=%s | db_ms=%s | total_ms=%s | filters=%s"
        ),
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
    embedding_ms = _elapsed_ms(embedding_started_at)

    db_started_at = time.perf_counter()
    results = search_document_chunks(
        embedding=query_embedding,
        limit=top_k,
        metadata_filters=metadata_filters,
    )
    db_ms = _elapsed_ms(db_started_at)
    return _build_retrieval_response(
        query=query,
        top_k=top_k,
        metadata_filters=metadata_filters,
        retrieval_strategy=retrieval_strategy,
        embedding_ms=embedding_ms,
        db_ms=db_ms,
        started_at=started_at,
        results=results,
    )


async def aretrieve_document_chunks(
    query: str,
    *,
    top_k: int = 3,
    metadata_filters: dict[str, str] | None = None,
) -> dict:
    """Embed a query, run async retrieval, and return results with latency breakdowns."""
    started_at = time.perf_counter()
    retrieval_strategy = resolve_document_chunk_search_strategy(metadata_filters)

    embedding_started_at = time.perf_counter()
    query_embedding = await aembed_text(query)
    embedding_ms = _elapsed_ms(embedding_started_at)

    db_started_at = time.perf_counter()
    async with dependency_limiter("retrieval"):
        results = await search_document_chunks_async(
            embedding=query_embedding,
            limit=top_k,
            metadata_filters=metadata_filters,
        )
    db_ms = _elapsed_ms(db_started_at)
    return _build_retrieval_response(
        query=query,
        top_k=top_k,
        metadata_filters=metadata_filters,
        retrieval_strategy=retrieval_strategy,
        embedding_ms=embedding_ms,
        db_ms=db_ms,
        started_at=started_at,
        results=results,
    )
