import boto3

from app.core.config import get_settings

settings = get_settings()
_bedrock_runtime_client = None


def get_bedrock_runtime_client():
    """Return a cached Bedrock runtime client for the configured AWS region."""
    global _bedrock_runtime_client
    if _bedrock_runtime_client is None:
        _bedrock_runtime_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.embedding.region_name,
        )
    return _bedrock_runtime_client
