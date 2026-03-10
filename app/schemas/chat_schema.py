from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:@\-]+$",
    )
    prompt: str = Field(min_length=1, max_length=8000)


class AsyncChatEnqueueResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=8, max_length=64)
    status: str = Field(min_length=1, max_length=32)
    submitted_at: str = Field(min_length=1, max_length=64)


class AsyncChatStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=8, max_length=64)
    user_id: str = Field(min_length=3, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    status: str = Field(min_length=1, max_length=32)
    submitted_at: str = Field(min_length=1, max_length=64)
    started_at: str = Field(default="", max_length=64)
    completed_at: str = Field(default="", max_length=64)
    response: str = Field(default="", max_length=12000)
    error: str = Field(default="", max_length=2000)
