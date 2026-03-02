import boto3

from app.core.config import get_settings

settings = get_settings()


def get_bedrock_runtime_client():
    """Create a Bedrock runtime client using the configured AWS region."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.embedding.region_name,
    )
