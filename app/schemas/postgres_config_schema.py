from pydantic import BaseModel, Field


class PostgresConfig(BaseModel):
    enabled: bool = False
    host: str
    port: int = Field(default=5432, ge=1, le=65535)
    database: str
    username: str
    ssl_mode: str = "require"
    min_pool_size: int = Field(default=1, ge=1, le=100)
    max_pool_size: int = Field(default=10, ge=1, le=200)
    connect_timeout_seconds: int = Field(default=10, ge=1, le=120)
    app_name: str = "unigraph"
    schema_name: str = "unigraph"
    memory_table: str = "long_term_memory"
    chunk_table: str = "document_chunks"
    embedding_dimensions: int = Field(default=1536, ge=1, le=16384)
    default_top_k: int = Field(default=3, ge=1, le=100)
    vector_index_type: str = "hnsw"
    ivfflat_lists: int = Field(default=100, ge=1, le=65535)
    hnsw_m: int = Field(default=16, ge=2, le=100)
    hnsw_ef_construction: int = Field(default=64, ge=4, le=1000)
