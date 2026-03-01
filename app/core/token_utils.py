import logging

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

logger = logging.getLogger(__name__)

# Avoid hard failure when tokenizer assets are unavailable (offline/dev envs).
ENCODING = None
if tiktoken is not None:
    try:
        ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialize tiktoken; using fallback token counter. %s", exc)

def count_tokens(messages: list) -> int:
    """Estimate token usage for a list of chat messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if ENCODING is None:
            total += max(1, len(content) // 4)
            continue
        total += len(ENCODING.encode(content))
    return total
