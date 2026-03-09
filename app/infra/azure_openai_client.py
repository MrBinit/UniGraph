import os
from openai import AsyncAzureOpenAI
from app.core.config import get_settings

settings = get_settings()

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=settings.azure_openai.endpoint,
    api_version=settings.azure_openai.api_version,
)
