from pydantic import BaseModel, Field


class BedrockConfig(BaseModel):
    primary_model_id: str
    fallback_model_id: str
    timeout: int = Field(ge=1, le=120)
    max_concurrency: int = Field(ge=1, le=1000)
