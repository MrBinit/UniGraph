from pydantic import BaseModel, Field


class AzureOpenAIConfig(BaseModel):
    endpoint: str
    api_version: str
    primary_deployment: str
    fallback_deployment: str
    timeout: int = Field(ge=1, le=120)
    max_concurrency: int = Field(ge=1, le=1000)
