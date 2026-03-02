from pydantic import BaseModel

from app.schemas.app_config_schema import AppConfig
from app.schemas.azure_openai_config_schema import AzureOpenAIConfig
from app.schemas.chunking_config_schema import ChunkingConfig
from app.schemas.circuit_config_schema import CircuitConfig
from app.schemas.embedding_config_schema import EmbeddingConfig
from app.schemas.guardrails_config_schema import GuardrailsConfig
from app.schemas.memory_config_schema import MemoryConfig, UserTokenBudgetConfig
from app.schemas.middleware_config_schema import MiddlewareConfig
from app.schemas.postgres_config_schema import PostgresConfig
from app.schemas.redis_config_schema import RedisConfig, RedisRoleConfig
from app.schemas.security_config_schema import SecurityConfig


class Settings(BaseModel):
    app: AppConfig
    redis: RedisConfig
    postgres: PostgresConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    azure_openai: AzureOpenAIConfig
    circuit: CircuitConfig
    memory: MemoryConfig
    guardrails: GuardrailsConfig
    security: SecurityConfig
    middleware: MiddlewareConfig


__all__ = [
    "AppConfig",
    "AzureOpenAIConfig",
    "ChunkingConfig",
    "CircuitConfig",
    "EmbeddingConfig",
    "GuardrailsConfig",
    "MemoryConfig",
    "MiddlewareConfig",
    "PostgresConfig",
    "RedisConfig",
    "RedisRoleConfig",
    "SecurityConfig",
    "Settings",
    "UserTokenBudgetConfig",
]
