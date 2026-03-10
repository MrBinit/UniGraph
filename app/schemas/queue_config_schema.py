from pydantic import BaseModel, Field


class QueueConfig(BaseModel):
    llm_async_enabled: bool = False
    llm_queue_url: str = ""
    llm_dlq_url: str = ""
    llm_result_table: str = ""
    llm_result_ttl_days: int = Field(default=7, ge=0, le=3650)
    llm_receive_wait_seconds: int = Field(default=20, ge=0, le=20)
    llm_max_messages_per_poll: int = Field(default=5, ge=1, le=10)
    llm_visibility_timeout_seconds: int = Field(default=300, ge=0, le=43200)
    llm_poll_sleep_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
    metrics_aggregation_queue_enabled: bool = False
    metrics_aggregation_queue_url: str = ""
    metrics_aggregation_receive_wait_seconds: int = Field(default=20, ge=0, le=20)
    metrics_aggregation_max_messages_per_poll: int = Field(default=10, ge=1, le=10)
    metrics_aggregation_visibility_timeout_seconds: int = Field(
        default=300, ge=0, le=43200
    )
    metrics_aggregation_poll_sleep_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
    evaluation_queue_enabled: bool = False
    evaluation_queue_url: str = ""
    evaluation_receive_wait_seconds: int = Field(default=20, ge=0, le=20)
    evaluation_max_messages_per_poll: int = Field(default=10, ge=1, le=10)
    evaluation_visibility_timeout_seconds: int = Field(default=300, ge=0, le=43200)
    evaluation_poll_sleep_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
