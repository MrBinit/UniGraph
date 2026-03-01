from pydantic import BaseModel, Field


class RedisRoleConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    username: str = ""
    password: str = ""
    namespace: str = Field(min_length=1, default="app")


class RedisConfig(BaseModel):
    app: RedisRoleConfig = Field(default_factory=RedisRoleConfig)
    worker: RedisRoleConfig = Field(
        default_factory=lambda: RedisRoleConfig(namespace="worker")
    )
