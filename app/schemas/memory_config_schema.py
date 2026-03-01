from pydantic import BaseModel, Field


class UserTokenBudgetConfig(BaseModel):
    soft_limit: int = Field(ge=1)
    hard_limit: int = Field(ge=1)
    min_recent_messages_to_keep: int = Field(default=4, ge=1, le=1000)


class MemoryConfig(BaseModel):
    max_tokens: int = Field(ge=1)
    summary_trigger: int = Field(ge=1)
    summary_ratio: float = Field(gt=0, le=1)
    redis_ttl_seconds: int = Field(ge=60)
    default_soft_token_budget: int = Field(default=2800, ge=1)
    default_hard_token_budget: int = Field(default=3600, ge=1)
    min_recent_messages_to_keep: int = Field(default=4, ge=1, le=1000)
    summary_quality_max_ratio: float = Field(default=0.6, gt=0, le=1)
    user_token_budgets: dict[str, UserTokenBudgetConfig] = Field(default_factory=dict)
    summary_queue_stream_key: str = "memory:summary:jobs"
    summary_queue_dlq_stream_key: str = "memory:summary:dlq"
    summary_queue_group: str = "memory-summary-workers"
    summary_queue_read_count: int = Field(default=10, ge=1, le=1000)
    summary_queue_block_ms: int = Field(default=5000, ge=1, le=60000)
    summary_queue_max_attempts: int = Field(default=5, ge=1, le=50)
    summary_queue_dlq_alert_threshold: int = Field(default=1, ge=1, le=100000)
    summary_queue_dlq_alert_cooldown_seconds: int = Field(default=300, ge=1, le=86400)
    summary_queue_dlq_monitor_interval_seconds: int = Field(default=30, ge=1, le=3600)
