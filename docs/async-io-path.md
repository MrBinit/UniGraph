# Async I/O Path (Hot Path)

This document describes the async I/O architecture used for chat request handling and retrieval.

## Goals

- Remove `asyncio.to_thread` usage from hot-path LLM/embedding/retrieval/Redis calls.
- Prevent shared default-threadpool saturation under high concurrency.
- Enforce strict concurrency controls per downstream dependency.

## What Changed

### 1) Per-dependency concurrency limiters

`app/infra/io_limiters.py` provides semaphores for:

- `llm`
- `embedding`
- `retrieval`
- `redis`

All limits are configured in `app/config/io_config.yaml` and loaded as `settings.io`.

### 2) Async Redis in hot path

`app/infra/redis_client.py` now exposes async clients:

- `app_async_redis_client`
- `worker_async_redis_client`
- `async_redis_client` (runtime-role selected)

Hot-path services now use async Redis calls with the `redis` limiter:

- `app/services/llm_service.py`
- `app/services/embedding_service.py`
- `app/services/memory_service.py`

### 3) Async retrieval DB queries

`app/infra/postgres_client.py` now exposes `get_async_postgres_pool()`.

`app/repositories/document_chunk_repository.py` adds:

- `search_document_chunks_async(...)`

`app/services/retrieval_service.py::aretrieve_document_chunks(...)` now uses:

- async embedding (`aembed_text`)
- async Postgres search
- `retrieval` limiter

### 4) Bedrock call isolation from shared threadpool

The Python Bedrock SDK is blocking. To avoid default-threadpool contention:

- `app/infra/bedrock_client.py` uses a dedicated `ThreadPoolExecutor`.
- Async entrypoints:
  - `aconverse(...)`
  - `ainvoke_model(...)`
  - `ainvoke_model_json(...)`

`app/infra/bedrock_chat_client.py` and `app/services/embedding_service.py` call these async wrappers and apply limiters (`llm`, `embedding`).

## Configuration

`app/config/io_config.yaml`:

- `io.llm_max_concurrency`
- `io.embedding_max_concurrency`
- `io.retrieval_max_concurrency`
- `io.redis_max_concurrency`
- `io.bedrock_executor_workers`

Environment overrides:

- `IO_LLM_MAX_CONCURRENCY`
- `IO_EMBEDDING_MAX_CONCURRENCY`
- `IO_RETRIEVAL_MAX_CONCURRENCY`
- `IO_REDIS_MAX_CONCURRENCY`
- `IO_BEDROCK_EXECUTOR_WORKERS`

## Tuning Guidance

- Increase `io.llm_max_concurrency` only with model provider throughput validation.
- Keep `io.retrieval_max_concurrency` aligned with Postgres pool capacity.
- Set `io.redis_max_concurrency` based on Redis latency SLOs and command mix.
- `io.bedrock_executor_workers` should be large enough to avoid queueing, but bounded to avoid CPU thrash.

## Verification

The async-path changes are covered with targeted tests in:

- `tests/test_llm_service.py`
- `tests/test_embedding_service.py`
- `tests/test_retrieval_service.py`
- `tests/test_memory_service.py`
