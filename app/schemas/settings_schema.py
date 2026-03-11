from pydantic import BaseModel

from app.schemas.app_config_schema import AppConfig
from app.schemas.azure_openai_config_schema import AzureOpenAIConfig
from app.schemas.chunking_config_schema import ChunkingConfig
from app.schemas.circuit_config_schema import CircuitConfig
from app.schemas.embedding_config_schema import EmbeddingConfig
from app.schemas.evaluation_runtime_config_schema import EvaluationRuntimeConfig
from app.schemas.guardrails_config_schema import GuardrailsConfig
from app.schemas.io_config_schema import IOConfig
from app.schemas.memory_config_schema import MemoryConfig, UserTokenBudgetConfig
from app.schemas.middleware_config_schema import MiddlewareConfig
from app.schemas.postgres_config_schema import PostgresConfig
from app.schemas.queue_config_schema import QueueConfig
from app.schemas.redis_config_schema import RedisConfig, RedisRoleConfig
from app.schemas.security_config_schema import SecurityConfig


class Settings(BaseModel):
    app: AppConfig
    redis: RedisConfig
    postgres: PostgresConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    evaluation: EvaluationRuntimeConfig
    azure_openai: AzureOpenAIConfig
    circuit: CircuitConfig
    memory: MemoryConfig
    guardrails: GuardrailsConfig
    security: SecurityConfig
    io: IOConfig
    middleware: MiddlewareConfig
    queue: QueueConfig


__all__ = [
    "AppConfig",
    "AzureOpenAIConfig",
    "ChunkingConfig",
    "CircuitConfig",
    "EmbeddingConfig",
    "EvaluationRuntimeConfig",
    "GuardrailsConfig",
    "IOConfig",
    "MemoryConfig",
    "MiddlewareConfig",
    "PostgresConfig",
    "QueueConfig",
    "RedisConfig",
    "RedisRoleConfig",
    "SecurityConfig",
    "Settings",
    "UserTokenBudgetConfig",
]
