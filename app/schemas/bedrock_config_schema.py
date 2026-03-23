from pydantic import BaseModel, Field


class BedrockConfig(BaseModel):
    primary_model_id: str
    fallback_model_id: str
    timeout: int = Field(ge=1, le=120)
    max_concurrency: int = Field(ge=1, le=1000)
    reranker_enabled: bool = False
    reranker_model_id: str = "cohere.rerank-v3-5:0"
    reranker_top_n: int = Field(default=4, ge=1, le=20)
    reranker_min_documents: int = Field(default=2, ge=1, le=100)
    reranker_max_documents: int = Field(default=12, ge=1, le=100)
    reranker_max_query_chars: int = Field(default=700, ge=50, le=4000)
    reranker_max_document_chars: int = Field(default=1400, ge=100, le=8000)
