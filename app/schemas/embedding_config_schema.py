from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configure Bedrock-based embedding generation for chunk manifests."""

    enabled: bool = True
    provider: str = "bedrock"
    region_name: str = "us-east-1"
    model_id: str = "amazon.titan-embed-text-v2:0"
    input_dir: str = "data/chunks"
    output_dir: str = "data/embeddings"
    glob_pattern: str = "*.chunks.json"
    max_text_chars: int = Field(default=20000, ge=100, le=200000)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    cache_ttl_seconds: int = Field(default=86400, ge=60, le=604800)
