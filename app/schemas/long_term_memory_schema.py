from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class LongTermMemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)
    memory_key: str = Field(min_length=1, max_length=200)
    memory_type: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=12000)
    source: str = Field(min_length=1, max_length=64)
    confidence: float = Field(default=1.0, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongTermMemoryRecord(LongTermMemoryWrite):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    embedding: list[float] | None = None
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
