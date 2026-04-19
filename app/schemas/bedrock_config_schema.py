from pydantic import BaseModel, Field


class BedrockConfig(BaseModel):
    primary_model_id: str
    fallback_model_id: str
    planner_model_id: str = ""
    planner_fallback_model_id: str = ""
    worker_model_id: str = ""
    worker_fallback_model_id: str = ""
    worker_escalation_model_id: str = ""
    verifier_model_id: str = ""
    verifier_fallback_model_id: str = ""
    finalizer_model_id: str = ""
    finalizer_fallback_model_id: str = ""
    timeout: int = Field(ge=1, le=120)
    max_concurrency: int = Field(ge=1, le=1000)
    web_grounding_enabled: bool = False
    web_grounding_include_sources: bool = True
    throttle_retry_max_attempts: int = Field(default=5, ge=1, le=12)
    throttle_retry_base_backoff_seconds: float = Field(default=0.4, ge=0.0, le=10.0)
    throttle_retry_max_backoff_seconds: float = Field(default=8.0, ge=0.1, le=120.0)
    throttle_retry_jitter_seconds: float = Field(default=0.25, ge=0.0, le=5.0)
    answer_rate_limit_rps: float = Field(default=8.0, ge=0.0, le=500.0)
    answer_rate_limit_burst: int = Field(default=16, ge=0, le=5000)
    planner_rate_limit_rps: float = Field(default=4.0, ge=0.0, le=500.0)
    planner_rate_limit_burst: int = Field(default=8, ge=0, le=5000)
    reranker_enabled: bool = False
    reranker_model_id: str = "cohere.rerank-v3-5:0"
    reranker_api_version: int = Field(default=2, ge=1, le=20)
    reranker_top_n: int = Field(default=4, ge=1, le=20)
    reranker_min_documents: int = Field(default=2, ge=1, le=100)
    reranker_max_documents: int = Field(default=12, ge=1, le=100)
    reranker_max_query_chars: int = Field(default=700, ge=50, le=4000)
    reranker_max_document_chars: int = Field(default=1400, ge=100, le=8000)
