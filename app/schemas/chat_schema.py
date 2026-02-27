from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:@\-]+$",
    )
    prompt: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1, max_length=12000)
