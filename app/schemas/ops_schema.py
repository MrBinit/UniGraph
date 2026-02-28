from pydantic import BaseModel, ConfigDict, Field


class MemoryHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    redis_available: bool
    ttl_seconds: int = Field(ge=0)
    encryption_enabled: bool


class QueueHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream_depth: int = Field(ge=0)
    pending_jobs: int = Field(ge=0)
    dlq_depth: int = Field(ge=0)
    consumer_group: str
    last_dlq_error: str = ""


class CompactionHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: int = Field(ge=0)
    removed_messages: int = Field(ge=0)
    removed_tokens: int = Field(ge=0)


class LatencyHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    average_ms: float = Field(ge=0)
    max_ms: int = Field(ge=0)
    last_ms: int = Field(ge=0)
    last_outcome: str = ""


class OpsStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    memory: MemoryHealth
    queue: QueueHealth
    compaction: CompactionHealth
    latency: LatencyHealth
