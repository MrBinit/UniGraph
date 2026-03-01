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
    embedding_dimensions: int = Field(default=1536, ge=1, le=16384)
    default_top_k: int = Field(default=8, ge=1, le=100)
