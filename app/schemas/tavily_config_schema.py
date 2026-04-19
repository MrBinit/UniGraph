from pydantic import BaseModel, Field


class TavilyConfig(BaseModel):
    """Configure external web-search integration."""

    enabled: bool = False
    google_search_url: str = "https://api.tavily.com/search"
    engine: str = "tavily"
    search_depth: str = "advanced"
    api_key_env_name: str = "TAVILY_WEB_SEARCH"
    default_gl: str = "us"
    default_hl: str = "en"
    default_num: int = Field(default=10, ge=1, le=100)
    timeout_seconds: float = Field(default=25.0, ge=1.0, le=120.0)
    fast_timeout_seconds: float = Field(default=0.0, ge=0.0, le=120.0)
    deep_timeout_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    max_concurrency: int = Field(default=8, ge=1, le=256)
    queue_workers: int = Field(default=4, ge=1, le=256)
    queue_max_size: int = Field(default=200, ge=1, le=100000)
    response_cache_enabled: bool = True
    always_web_retrieval_enabled: bool = True
    retrieval_fanout_enabled: bool = True
    fallback_enabled: bool = True
    fallback_similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    expansion_similarity_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    query_planner_enabled: bool = True
    query_planner_use_llm: bool = True
    query_planner_model_id: str = ""
    query_planner_acquire_timeout_seconds: float = Field(default=0.75, ge=0.0, le=30.0)
    query_planner_cache_enabled: bool = True
    query_planner_cache_ttl_seconds: int = Field(default=900, ge=0, le=86400)
    query_planner_max_queries: int = Field(default=5, ge=1, le=12)
    query_planner_max_subquestions: int = Field(default=4, ge=0, le=12)
    retrieval_loop_enabled: bool = True
    retrieval_loop_use_llm: bool = True
    retrieval_loop_model_id: str = ""
    retrieval_loop_acquire_timeout_seconds: float = Field(default=0.75, ge=0.0, le=30.0)
    retrieval_loop_cache_enabled: bool = True
    retrieval_loop_cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    retrieval_loop_max_steps: int = Field(default=2, ge=1, le=5)
    retrieval_loop_max_gap_queries: int = Field(default=3, ge=1, le=8)
    deep_required_field_rescue_enabled: bool = True
    deep_required_field_rescue_max_queries: int = Field(default=6, ge=1, le=12)
    agentic_required_field_rescue_max_rounds: int = Field(default=2, ge=0, le=3)
    deep_internal_crawl_enabled: bool = True
    deep_internal_crawl_max_depth: int = Field(default=2, ge=1, le=4)
    deep_internal_crawl_max_pages: int = Field(default=10, ge=1, le=30)
    deep_internal_crawl_links_per_page: int = Field(default=10, ge=1, le=30)
    deep_internal_crawl_per_parent_limit: int = Field(default=4, ge=1, le=12)
    retrieval_min_unique_domains: int = Field(default=2, ge=1, le=8)
    deep_min_unique_domains: int = Field(default=2, ge=1, le=8)
    retrieval_gap_min_token_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    multi_query_enabled: bool = True
    max_query_variants: int = Field(default=3, ge=1, le=5)
    deep_max_query_variants: int = Field(default=5, ge=1, le=8)
    allowed_domain_suffixes: list[str] = Field(default_factory=list)
    official_source_filter_enabled: bool = True
    official_source_allowlist: list[str] = Field(default_factory=lambda: ["daad.de"])
    max_context_results: int = Field(default=3, ge=1, le=20)
    deep_max_context_results: int = Field(default=8, ge=1, le=20)
    deep_default_num: int = Field(default=6, ge=1, le=100)
    fetch_page_content: bool = True
    max_pages_to_fetch: int = Field(default=2, ge=0, le=20)
    deep_max_pages_to_fetch: int = Field(default=4, ge=0, le=20)
    page_fetch_timeout_seconds: float = Field(default=6.0, ge=1.0, le=60.0)
    max_page_chars: int = Field(default=3000, ge=200, le=50000)
    strip_boilerplate: bool = True
    min_clean_line_chars: int = Field(default=40, ge=0, le=500)
    page_chunk_chars: int = Field(default=850, ge=120, le=4000)
    page_chunk_overlap_chars: int = Field(default=120, ge=0, le=1000)
    max_chunks_per_page: int = Field(default=1, ge=1, le=20)
    deep_max_chunks_per_page: int = Field(default=3, ge=1, le=20)
    min_chunk_chars: int = Field(default=140, ge=20, le=2000)
    chunk_dedupe_similarity: float = Field(default=0.9, ge=0.5, le=1.0)
    trust_relevance_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    trust_authority_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    trust_recency_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    trust_agreement_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    deep_cache_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    deep_answer_min_confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    deep_required_field_min_coverage: float = Field(default=0.85, ge=0.5, le=1.0)
    deep_cache_min_required_field_coverage: float = Field(default=0.85, ge=0.0, le=1.0)
    deep_cache_min_source_count: int = Field(default=2, ge=1, le=20)
    cache_max_not_verified_mentions: int = Field(default=3, ge=1, le=20)
    retry_max_attempts: int = Field(default=3, ge=1, le=6)
    retry_base_backoff_seconds: float = Field(default=0.8, ge=0.1, le=5.0)
