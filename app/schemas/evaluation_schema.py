from pydantic import BaseModel, ConfigDict, Field


class EvaluationConversationLabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:@\-]+$",
    )
    expected_answer: str | None = Field(default=None, min_length=1, max_length=12000)
    relevant_chunk_ids: list[str] | None = Field(default=None, max_length=200)
    user_feedback: str | None = Field(default=None, min_length=1, max_length=2000)
    user_feedback_score: int | None = Field(default=None, ge=-1, le=1)


class EvaluationConversationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    created_at: str
    prompt: str
    answer: str
    retrieval_strategy: str = ""
    retrieved_count: int = Field(ge=0)
    labels: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)


class EvaluationConversationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    total_conversations: int = Field(ge=0)
    conversations: list[EvaluationConversationItem] = Field(default_factory=list)


class EvaluationConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation: EvaluationConversationItem


class EvaluationReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    total_conversations: int = Field(ge=0)
    labeled_conversations: int = Field(ge=0)
    retrieval_metrics: dict = Field(default_factory=dict)
    generation_metrics: dict = Field(default_factory=dict)
    web_fallback_metrics: dict = Field(default_factory=dict)
    conversations: list[EvaluationConversationItem] = Field(default_factory=list)


class OfflineEvaluationStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    schedule_enabled: bool
    interval_hours: int = Field(ge=1)
    has_new_requests: bool
    due_by_interval: bool
    should_auto_run: bool
    last_request_timestamp: str = ""
    last_evaluated_timestamp: str = ""
    reason: str = ""


class OfflineEvaluationRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ran: bool
    reason: str
    result: dict = Field(default_factory=dict)
    status: OfflineEvaluationStatusResponse | None = None


class OfflineEvaluationReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: str
    window_hours: int = Field(ge=1)
    evaluated_count: int = Field(ge=0)
    scores: dict = Field(default_factory=dict)
    failure_reasons: dict = Field(default_factory=dict)
    top_bad_examples: list[dict] = Field(default_factory=list)
