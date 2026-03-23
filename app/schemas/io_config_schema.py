from pydantic import BaseModel, Field


class IOConfig(BaseModel):
    """Per-dependency async I/O concurrency caps for hot-path traffic."""

    llm_max_concurrency: int = Field(default=50, ge=1, le=10000)
    embedding_max_concurrency: int = Field(default=8, ge=1, le=10000)
    retrieval_max_concurrency: int = Field(default=32, ge=1, le=10000)
    reranker_max_concurrency: int = Field(default=8, ge=1, le=10000)
    redis_max_concurrency: int = Field(default=200, ge=1, le=10000)
    bedrock_executor_workers: int = Field(default=64, ge=1, le=10000)
