from app.core.config import get_settings

settings = get_settings()


def resolve_user_budget(user_id: str) -> tuple[int, int, int]:
    """Resolve the effective token budget limits for a user, including overrides."""
    override = settings.memory.user_token_budgets.get(user_id)

    soft_limit = settings.memory.default_soft_token_budget
    hard_limit = settings.memory.default_hard_token_budget
    min_recent = settings.memory.min_recent_messages_to_keep

    if override:
        soft_limit = override.soft_limit
        hard_limit = override.hard_limit
        min_recent = override.min_recent_messages_to_keep

    soft_limit = max(1, soft_limit)
    hard_limit = max(soft_limit, hard_limit)
    min_recent = max(1, min_recent)
    return soft_limit, hard_limit, min_recent
