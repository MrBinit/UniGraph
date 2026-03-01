from pydantic import BaseModel, Field


class CircuitConfig(BaseModel):
    fail_max: int = Field(ge=1, le=1000)
    reset_timeout: int = Field(ge=1, le=86400)
