import logging
import re
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-. ]?)?(?:\(?\d{3}\)?[-. ]?)\d{3}[-. ]?\d{4}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_AZURE_KEY_ASSIGN_RE = re.compile(
    r"(?i)\b(?:AZURE_OPENAI_API_KEY|OPENAI_API_KEY)\s*[:=]\s*([^\s\"']+)"
)
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def refusal_response() -> str:
    return settings.guardrails.safe_refusal_message


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, text):
                return True
        except re.error as exc:
            logger.warning("Invalid regex pattern in guardrails config: %s (%s)", pattern, exc)
    return False


def redact_sensitive_content(text: str) -> str:
    if not isinstance(text, str):
        return ""
    redacted = text
    redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = _OPENAI_KEY_RE.sub("[REDACTED_API_KEY]", redacted)
    redacted = _AZURE_KEY_ASSIGN_RE.sub(lambda m: m.group(0).split(m.group(1))[0] + "[REDACTED_API_KEY]", redacted)
    redacted = _CARD_RE.sub("[REDACTED_CARD]", redacted)
    return redacted

def guard_user_input(user_id: str, prompt: str) -> dict:
    if not settings.guardrails.enable_input_guardrails:
        return {"blocked": False, "sanitized_text": prompt, "reason": ""}

    if not isinstance(prompt, str) or not prompt.strip():
        return {"blocked": True, "sanitized_text": "", "reason": "empty_prompt"}

    if len(prompt) > settings.guardrails.max_input_chars:
        logger.info("GuardrailBlockInput | user=%s | reason=max_input_chars", user_id)
        return {"blocked": True, "sanitized_text": "", "reason": "max_input_chars"}

    if _matches_any_pattern(prompt, settings.guardrails.blocked_input_patterns):
        logger.info("GuardrailBlockInput | user=%s | reason=blocked_input_pattern", user_id)
        return {"blocked": True, "sanitized_text": "", "reason": "blocked_input_pattern"}

    sanitized = redact_sensitive_content(prompt)
    return {"blocked": False, "sanitized_text": sanitized, "reason": ""}


def apply_context_guardrails(messages: list[dict]) -> dict:
    if not settings.guardrails.enable_context_guardrails:
        return {"blocked": False, "messages": messages, "reason": ""}

    allowed_roles = {"system", "user", "assistant"}
    cleaned: list[dict] = []
    injection_detected = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in allowed_roles or not isinstance(content, str):
            continue

        content = redact_sensitive_content(content)
        if role != "system" and _matches_any_pattern(content, settings.guardrails.injection_patterns):
            injection_detected = True
            content = "[Potential prompt-injection content removed.]"

        cleaned.append({"role": role, "content": content})

    policy_msg = {
        "role": "system",
        "content": settings.guardrails.policy_system_message,
    }
    cleaned = [policy_msg] + cleaned

    max_msgs = max(1, settings.guardrails.max_context_messages)
    if len(cleaned) > max_msgs:
        keep_tail = max_msgs - 1
        cleaned = [cleaned[0]] + cleaned[-keep_tail:]

    if not cleaned:
        return {"blocked": True, "messages": [], "reason": "empty_context"}

    if injection_detected:
        logger.info("GuardrailContextSanitized | reason=prompt_injection_detected")

    return {"blocked": False, "messages": cleaned, "reason": ""}

def guard_model_output(text: str) -> dict:
    if not settings.guardrails.enable_output_guardrails:
        safe_text = text if isinstance(text, str) else ""
        return {"blocked": False, "text": safe_text, "reason": ""}

    if not isinstance(text, str) or not text.strip():
        return {"blocked": True, "text": refusal_response(), "reason": "empty_output"}

    if _matches_any_pattern(text, settings.guardrails.blocked_output_patterns):
        return {"blocked": True, "text": refusal_response(), "reason": "blocked_output_pattern"}

    redacted = redact_sensitive_content(text)
    if len(redacted) > settings.guardrails.max_output_chars:
        redacted = redacted[: settings.guardrails.max_output_chars].rstrip() + "..."

    if not redacted.strip():
        return {"blocked": True, "text": refusal_response(), "reason": "empty_after_redaction"}

    return {"blocked": False, "text": redacted, "reason": ""}


def sanitize_summary_output(summary_text: str) -> str:
    if not isinstance(summary_text, str):
        return ""

    sanitized = redact_sensitive_content(summary_text)

    if _matches_any_pattern(sanitized, settings.guardrails.blocked_output_patterns):
        return ""

    for pattern in settings.guardrails.injection_patterns:
        try:
            sanitized = re.sub(pattern, "[FILTERED_INJECTION_PATTERN]", sanitized)
        except re.error:
            continue

    return sanitized.strip()
