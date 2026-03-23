from pydantic import BaseModel, Field


class SerpApiConfig(BaseModel):
    """Configure SerpAPI Google Search integration."""

    enabled: bool = False
    google_search_url: str = "https://serpapi.com/search.json"
    engine: str = "google"
    api_key_env_name: str = "SERPAPI_API_KEY"
    default_gl: str = "us"
    default_hl: str = "en"
    default_num: int = Field(default=10, ge=1, le=100)
    timeout_seconds: float = Field(default=25.0, ge=1.0, le=120.0)
    max_concurrency: int = Field(default=8, ge=1, le=256)
    queue_workers: int = Field(default=4, ge=1, le=256)
    queue_max_size: int = Field(default=200, ge=1, le=100000)
    fallback_enabled: bool = True
    fallback_similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    multi_query_enabled: bool = True
    max_query_variants: int = Field(default=3, ge=1, le=5)
    allowed_domain_suffixes: list[str] = Field(default_factory=list)
    max_context_results: int = Field(default=3, ge=1, le=20)
    fetch_page_content: bool = True
    max_pages_to_fetch: int = Field(default=2, ge=0, le=20)
    page_fetch_timeout_seconds: float = Field(default=6.0, ge=1.0, le=60.0)
    max_page_chars: int = Field(default=3000, ge=200, le=50000)
    strip_boilerplate: bool = True
    min_clean_line_chars: int = Field(default=40, ge=0, le=500)
    page_chunk_chars: int = Field(default=850, ge=120, le=4000)
    page_chunk_overlap_chars: int = Field(default=120, ge=0, le=1000)
    max_chunks_per_page: int = Field(default=1, ge=1, le=20)
    min_chunk_chars: int = Field(default=140, ge=20, le=2000)
    chunk_dedupe_similarity: float = Field(default=0.9, ge=0.5, le=1.0)
