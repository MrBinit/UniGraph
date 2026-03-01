import logging

logger = logging.getLogger(__name__)


def compose_context(summary: str, messages: list, new_user_message: str) -> list:
    """Build the chat context from summary, stored messages, and the new user turn."""
    context = []
    if summary:
        context.append({"role": "system", "content": f"Previous summary: {summary}"})
    context.extend(messages)
    if new_user_message:
        context.append({"role": "user", "content": new_user_message})
    return context


def safe_token_count(token_counter, messages: list) -> int:
    """Count tokens defensively and return zero if token estimation fails."""
    try:
        return token_counter(messages)
    except Exception as exc:  # pragma: no cover
        logger.warning("Token counting failed; defaulting to 0. %s", exc)
        return 0


def merge_summaries(existing_summary: str, new_summary: str) -> str:
    """Append a new summary to the existing summary text when both are present."""
    if not existing_summary:
        return new_summary
    if not new_summary:
        return existing_summary
    return f"{existing_summary}\n{new_summary}"


def select_summary_cutoff(messages: list, summary_ratio: float) -> tuple[list, int | None]:
    """Choose the oldest slice of messages that should be summarized next."""
    if not messages:
        return [], None

    split_index = max(1, int(len(messages) * summary_ratio))
    candidates = messages[:split_index]
    if not candidates:
        return [], None

    cutoff_seq = candidates[-1].get("seq")
    if not isinstance(cutoff_seq, int):
        return [], None
    return candidates, cutoff_seq


def truncate_context_without_summary(
    *,
    summary: str,
    messages: list,
    new_user_message: str,
    soft_limit: int,
    hard_limit: int,
    min_recent: int,
    token_counter,
) -> dict:
    """Trim messages and, if required, the summary to keep context within token budgets."""
    events = []
    memory_changed = False

    final_context = compose_context(summary, messages, new_user_message)
    final_tokens = safe_token_count(token_counter, final_context)

    if final_tokens > soft_limit and len(messages) > min_recent:
        removed_messages = 0
        removed_tokens = 0
        before_tokens = final_tokens

        while final_tokens > soft_limit and len(messages) > min_recent:
            removed = messages.pop(0)
            removed_messages += 1
            removed_tokens += safe_token_count(token_counter, [removed])
            final_context = compose_context(summary, messages, new_user_message)
            final_tokens = safe_token_count(token_counter, final_context)

        if removed_messages:
            memory_changed = True
            events.append(
                {
                    "trigger": "soft_budget_truncate",
                    "removed_messages": removed_messages,
                    "removed_tokens": removed_tokens,
                    "before_tokens": before_tokens,
                    "after_tokens": final_tokens,
                    "summary_text": "",
                }
            )

    if final_tokens > hard_limit:
        removed_messages = 0
        removed_tokens = 0
        before_tokens = final_tokens

        while final_tokens > hard_limit and messages:
            removed = messages.pop(0)
            removed_messages += 1
            removed_tokens += safe_token_count(token_counter, [removed])
            final_context = compose_context(summary, messages, new_user_message)
            final_tokens = safe_token_count(token_counter, final_context)

        removed_summary = ""
        if final_tokens > hard_limit and summary:
            removed_summary = summary
            removed_tokens += safe_token_count(token_counter, [{"role": "assistant", "content": summary}])
            summary = ""
            final_context = compose_context(summary, messages, new_user_message)
            final_tokens = safe_token_count(token_counter, final_context)

        if removed_messages or removed_summary:
            memory_changed = True
            events.append(
                {
                    "trigger": "hard_budget_truncate",
                    "removed_messages": removed_messages,
                    "removed_tokens": removed_tokens,
                    "before_tokens": before_tokens,
                    "after_tokens": final_tokens,
                    "summary_text": removed_summary,
                }
            )

    return {
        "summary": summary,
        "messages": messages,
        "final_context": final_context,
        "final_tokens": final_tokens,
        "memory_changed": memory_changed,
        "events": events,
    }
