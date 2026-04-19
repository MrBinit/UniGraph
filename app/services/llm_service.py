import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import os
import re
import time
from types import SimpleNamespace
from typing import AsyncIterator
from urllib.parse import urlparse
from uuid import uuid4
from redis.exceptions import RedisError
from app.core.config import get_prompts, get_settings
from app.infra.bedrock_chat_client import client
from app.infra.io_limiters import dependency_limiter
from app.infra.redis_client import app_scoped_key, async_redis_client, redis_client
from app.services.chat_trace_service import emit_trace_event
from app.services.evaluation_service import store_chat_trace
from app.services.guardrails_service import (
    apply_context_guardrails,
    guard_model_output,
    guard_user_input,
    refusal_response,
)
from app.services.memory_service import build_context, update_memory
from app.services.quality_metrics_service import citation_accuracy_score, generation_metrics
from app.services.reranker_service import arerank_retrieval_results
from app.services.sqs_event_queue_service import enqueue_metrics_record_event
from app.services.retrieval_service import aretrieve_document_chunks
from app.services.web_retrieval_service import aretrieve_web_chunks

settings = get_settings()
prompts = get_prompts()
logger = logging.getLogger(__name__)

_BACKGROUND_TASKS: set[asyncio.Task] = set()
_RETRIEVAL_QUERY_MAX_CHARS = 900
_RETRIEVAL_CONTEXT_MAX_CHARS = 1500
_RETRIEVAL_CHUNK_MAX_CHARS = 360
_RETRIEVAL_MAX_PROMPT_RESULTS = 3
_RETRIEVAL_EVIDENCE_MAX_ITEMS = 3
_RETRIEVAL_EVIDENCE_CONTENT_MAX_CHARS = 700
_CITATION_URL_RE = re.compile(r"https?://[^\s<>\")\]]+")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_STREAM_GUARD_HOLDBACK_CHARS = 120
_LLM_MOCK_MODE_ENV = "LLM_MOCK_MODE"
_LLM_MOCK_TEXT_ENV = "LLM_MOCK_TEXT"
_LLM_MOCK_DELAY_MS_ENV = "LLM_MOCK_DELAY_MS"
_LLM_MOCK_STREAM_CHUNK_CHARS_ENV = "LLM_MOCK_STREAM_CHUNK_CHARS"
_RETRIEVAL_DISABLED_ENV = "RETRIEVAL_DISABLED"
_NO_RELEVANT_INFORMATION_DETAIL = "Sorry, no relevant information is found."
_WEB_RETRIEVAL_TIMEOUT_DETAIL = (
    "Web retrieval timed out while verifying official sources. Please retry."
)
_DEFAULT_EXECUTION_MODE = "deep"
_FAST_MODE = "fast"
_DEEP_MODE = "deep"
_AUTO_MODE = "auto"
_DEADLINE_QUERY_RE = re.compile(
    r"\b(application\s+deadline|deadline|last\s+date|closing\s+date|apply\s+by)\b",
    flags=re.IGNORECASE,
)
_FRESHNESS_QUERY_RE = re.compile(
    r"\b(latest|today|current|recent|news|deadline|updated?)\b",
    flags=re.IGNORECASE,
)
_DEEP_COMPLEXITY_RE = re.compile(
    r"\b(vs|versus|compare|comparison|best|top|rank|ranking|pros and cons|strategy|"
    r"analyz(?:e|ing)|debug|plan|step by step|multi|including|latest|today|deadline)\b",
    flags=re.IGNORECASE,
)
_INTENT_COMPARE_RE = re.compile(r"\b(vs|versus|compare|difference|better)\b", flags=re.IGNORECASE)
_INTENT_DEADLINE_RE = re.compile(
    r"\b(deadline|apply by|last date|closing date|intake|application)\b", flags=re.IGNORECASE
)
_INTENT_REQUIREMENTS_RE = re.compile(
    r"\b(requirement|requirements|eligibility|ielts|toefl|cefr|gpa|ects|credits?|"
    r"language requirement|application portal|where to apply)\b",
    flags=re.IGNORECASE,
)
_INTENT_FACT_RE = re.compile(
    r"\b(what is|how many|tuition|fee|duration|requirement|eligibility|ranking)\b",
    flags=re.IGNORECASE,
)
_INTENT_PLAN_RE = re.compile(
    r"\b(step by step|roadmap|plan|strategy|best way|recommend)\b", flags=re.IGNORECASE
)
_FOLLOW_UP_REFERENCE_RE = re.compile(
    r"\b(it|its|that|those|them|this|these|former|latter|same|above|previous)\b",
    flags=re.IGNORECASE,
)
_FOLLOW_UP_CONTEXT_ENTITY_RE = re.compile(
    r"\b(?:the|this|that|its)\s+"
    r"(?:university|program|course|department|admission|application|requirements?|deadline|intake)\b",
    flags=re.IGNORECASE,
)
_FOLLOW_UP_PREFIXES = (
    "and ",
    "also ",
    "what about",
    "how about",
    "same for",
    "continue",
    "then ",
    "now ",
)
_DATE_LIKE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|"
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b|"
    r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b",
    flags=re.IGNORECASE,
)
_DEADLINE_SENTENCE_HINT_RE = re.compile(
    r"\b(deadline|application period|apply by|closing date|last date|bewerbungsfrist|intake)\b",
    flags=re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAIM_LINE_MIN_CHARS = 42
_CLAIM_CITATION_MIN_COVERAGE = 0.55
_SELECTIVE_RESULT_SCORE_THRESHOLD = 0.52
_TRUST_LOW_CONFIDENCE_THRESHOLD = 0.58
_CLAIM_SNIPPET_MIN_GROUNDING_COVERAGE = 0.6
_WEB_EXPANSION_SIMILARITY_THRESHOLD = 0.45
_WEB_EXPANSION_MIN_DOMAIN_COUNT = 2
_PARTIAL_EVIDENCE_AUTHORITY_MIN_SCORE = 0.75
_WEB_RECOVERY_MAX_QUERIES = 2
_WEB_TIMEOUT_RESCUE_MAX_QUERIES = 3
_WEB_QUERY_MAX_CHARS = 220
_QUERY_NOT_ANSWERED_MARKERS = (
    "does not address",
    "no comparison",
    "no information",
    "not answer",
    "ignores the specific request",
    "generic response",
    "fails to address",
)
_GENERIC_PLACEHOLDER_MARKERS = (
    "i can help with",
    "please ask in this scope",
    "ask in this scope",
    "please ask a specific",
    "could you clarify your question",
    "i need more details",
)
_LEADING_QUERY_FILLER_RE = re.compile(
    r"^(?:tell me|can you|could you|please|i want to know|give me|show me|explain|about)\s+",
    flags=re.IGNORECASE,
)
_TRAILING_QUERY_INSTRUCTION_RE = re.compile(
    r"\b(if any information is missing|if information is missing|search deeper and verify.*|"
    r"also tell me if .*competitive.*|also tell me if .*safe.*)\b.*$",
    flags=re.IGNORECASE,
)
_SPECULATIVE_FACTUAL_CLAIM_RE = re.compile(
    r"\b(likely|probably|possibly|may be|might be|appears to be|seems to be)\b",
    flags=re.IGNORECASE,
)
_SPECULATIVE_FACTUAL_FIELD_RE = re.compile(
    r"\b(language|english|german|ielts|toefl|cefr|deadline|apply|ects|credit|gpa|grade|"
    r"requirements?|eligibility|duration|semesters?|fees?|tuition)\b",
    flags=re.IGNORECASE,
)
_LOW_SIGNAL_SCAFFOLD_LINE_RE = re.compile(
    r"^(evidence and caveats|claim-by-claim citations|uncertainty|missing info|caveats|evidence)\s*:?\s*$",
    flags=re.IGNORECASE,
)
_COMPARISON_ENTITY_STOPWORDS = {
    "for",
    "the",
    "and",
    "with",
    "including",
    "admission",
    "requirements",
    "application",
    "deadlines",
    "deadline",
    "program",
    "programs",
    "master",
    "masters",
    "msc",
    "m.sc",
}
_COMPARISON_ENTITY_DOMAIN_HINTS: tuple[tuple[str, str], ...] = (
    ("tum", "tum.de"),
    ("technical university of munich", "tum.de"),
    ("lmu", "lmu.de"),
    ("ludwig maximilian", "lmu.de"),
    ("rwth", "rwth-aachen.de"),
    ("rwth aachen", "rwth-aachen.de"),
)
_CHAT_CACHE_VERSION = "v3"
_CACHE_LOW_QUALITY_NOT_VERIFIED_RE = re.compile(
    r"\bnot verified from (?:evidence|sources)\b",
    flags=re.IGNORECASE,
)
_REQUIRED_FIELD_LABELS = {
    "comparison_between_requested_entities": "Comparison between requested entities",
    "application_deadline": "Application deadline",
    "application_portal": "Application portal",
    "required_documents": "Required documents",
    "eligibility_requirements": "Eligibility requirements",
    "gpa_or_grade_threshold": "GPA/grade threshold",
    "ects_or_prerequisite_credit_breakdown": "ECTS/prerequisite credit breakdown",
    "tuition_or_fees": "Tuition/fees",
    "language_requirements": "Language requirements",
    "language_test_score_thresholds": "Language test score thresholds",
    "curriculum_focus": "Curriculum focus",
    "career_outcomes": "Career outcomes",
    "scholarship_options": "Scholarship options",
    "visa_or_work_rights": "Visa/work rights",
    "aps_requirement_stage": "APS requirement stage",
    "admission_decision_signal": "Admission decision signal",
}
_ADMISSIONS_CRITICAL_WEB_REQUIRED_FIELDS = {
    "admission_requirements",
    "gpa_threshold",
    "ects_breakdown",
    "language_requirements",
    "language_score_thresholds",
    "application_deadline",
    "application_portal",
}


def _clamp01(value: float | None, *, fallback: float = 0.0) -> float:
    if value is None:
        return max(0.0, min(1.0, fallback))
    return max(0.0, min(1.0, float(value)))


def _chat_cache_key(
    user_id: str,
    prompt: str,
    session_id: str | None = None,
    mode: str | None = None,
) -> str:
    """Build a fixed-length chat cache key without embedding raw prompt text."""
    normalized_session = str(session_id or user_id).strip() or str(user_id).strip()
    normalized_mode = _normalized_request_mode(mode)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if mode is None or normalized_mode == _DEFAULT_EXECUTION_MODE:
        return app_scoped_key(
            "cache",
            "chat",
            _CHAT_CACHE_VERSION,
            user_id,
            normalized_session,
            f"sha256:{prompt_hash}",
        )
    return app_scoped_key(
        "cache",
        "chat",
        _CHAT_CACHE_VERSION,
        user_id,
        normalized_session,
        f"mode:{normalized_mode}",
        f"sha256:{prompt_hash}",
    )


def _resolve_session_id(user_id: str, session_id: str | None) -> str:
    """Resolve an effective session identifier with a safe fallback."""
    candidate = str(session_id or "").strip()
    return candidate or str(user_id).strip()


def _conversation_user_id(user_id: str, session_id: str) -> str:
    """Return memory key space id; isolate when a distinct session id is provided."""
    normalized_user = str(user_id).strip()
    normalized_session = str(session_id).strip()
    if not normalized_session or normalized_session == normalized_user:
        return normalized_user
    return f"{normalized_user}::session::{normalized_session}"


async def _redis_call(method, *args, **kwargs):
    """Execute a Redis operation using the async client and limiter."""
    async with dependency_limiter("redis"):
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed milliseconds from a monotonic start time."""
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _traceable_urls(urls, *, limit: int = 8) -> list[str]:
    if not isinstance(urls, list):
        return []
    collected: list[str] = []
    for item in urls:
        value = _normalized_url(str(item))
        if not value:
            continue
        collected.append(value)
        if len(collected) >= limit:
            break
    return collected


def _safe_int(value) -> int | None:
    """Convert a numeric-like value to int when possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    """Convert a numeric-like value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_retrieval_top_k() -> int:
    postgres_cfg = getattr(settings, "postgres", None)
    value = _safe_int(getattr(postgres_cfg, "default_top_k", None)) if postgres_cfg else None
    if value is None or value <= 0:
        return 3
    return max(1, min(50, value))


def _normalized_model_id(value) -> str:
    return " ".join(str(value or "").split()).strip()


def _model_ids_for_role(role: str, *, attempt: int = 1) -> tuple[str, str]:
    role_key = str(role or "worker").strip().lower()
    default_primary = _normalized_model_id(settings.bedrock.primary_model_id)
    default_fallback = _normalized_model_id(settings.bedrock.fallback_model_id)

    if role_key == "planner":
        primary = _normalized_model_id(getattr(settings.bedrock, "planner_model_id", ""))
        fallback = _normalized_model_id(
            getattr(settings.bedrock, "planner_fallback_model_id", "")
        )
    elif role_key == "verifier":
        primary = _normalized_model_id(getattr(settings.bedrock, "verifier_model_id", ""))
        fallback = _normalized_model_id(
            getattr(settings.bedrock, "verifier_fallback_model_id", "")
        )
    elif role_key == "finalizer":
        primary = _normalized_model_id(getattr(settings.bedrock, "finalizer_model_id", ""))
        fallback = _normalized_model_id(
            getattr(settings.bedrock, "finalizer_fallback_model_id", "")
        )
    else:
        primary = _normalized_model_id(getattr(settings.bedrock, "worker_model_id", ""))
        if int(attempt) > 1:
            escalation = _normalized_model_id(
                getattr(settings.bedrock, "worker_escalation_model_id", "")
            )
            if escalation:
                primary = escalation
        fallback = _normalized_model_id(getattr(settings.bedrock, "worker_fallback_model_id", ""))

    resolved_primary = primary or default_primary
    resolved_fallback = fallback or default_fallback or resolved_primary
    return resolved_primary, resolved_fallback


def _truthy_env(name: str) -> bool:
    """Parse one boolean feature flag from environment variables."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _normalized_request_mode(mode: str | None) -> str:
    candidate = str(mode or "").strip().lower()
    if candidate == "standard":
        return _FAST_MODE
    if candidate in {_FAST_MODE, _DEEP_MODE, _AUTO_MODE}:
        return candidate
    return _AUTO_MODE


def _mode_from_state(state: dict | None) -> str:
    if not isinstance(state, dict):
        return _DEEP_MODE
    for key in ("execution_mode", "requested_mode"):
        candidate = _normalized_request_mode(state.get(key))
        if candidate in {_FAST_MODE, _DEEP_MODE}:
            return candidate
    return _DEEP_MODE


def _max_context_results_for_mode(mode: str | None = None) -> int:
    configured = max(1, int(settings.web_search.max_context_results))
    normalized_mode = _normalized_request_mode(mode)
    if normalized_mode == _AUTO_MODE:
        normalized_mode = _DEEP_MODE
    if normalized_mode != _DEEP_MODE:
        return configured
    deep_override = _safe_int(getattr(settings.web_search, "deep_max_context_results", configured))
    if deep_override is None:
        return configured
    return max(configured, min(20, deep_override))


def _is_complex_query_for_deep(query: str) -> bool:
    compact = " ".join(str(query or "").split())
    if not compact:
        return False
    if len(compact) >= 220:
        return True
    if compact.count("?") >= 2:
        return True
    if _DEEP_COMPLEXITY_RE.search(compact):
        return True
    return False


def _is_admissions_high_precision_query(query: str) -> bool:
    compact = " ".join(str(query or "").split()).strip().lower()
    if not compact:
        return False
    has_program_context = bool(
        re.search(
            r"\b(university|uni|master|masters|m\.sc|msc|program|programme|course|admission)\b",
            compact,
        )
    )
    if not has_program_context:
        return False
    return bool(
        re.search(
            r"\b(requirements?|eligibility|language|international students?|ielts|toefl|cefr|"
            r"deadline|intake|ects|credits?|gpa|grade|tuition|fees?)\b",
            compact,
        )
    )


def _resolve_initial_execution_mode(requested_mode: str, safe_prompt: str) -> str:
    normalized = _normalized_request_mode(requested_mode)
    if normalized == _FAST_MODE:
        return _FAST_MODE
    if normalized == _DEEP_MODE:
        return _DEEP_MODE
    if _is_admissions_high_precision_query(safe_prompt):
        return _DEEP_MODE
    return _DEEP_MODE if _is_complex_query_for_deep(safe_prompt) else _FAST_MODE


def _mode_prompt_config() -> dict:
    config = _chat_config().get("mode_prompts", {})
    return config if isinstance(config, dict) else {}


def _mode_instruction_text(mode: str) -> str:
    config = _mode_prompt_config()
    if mode == _FAST_MODE:
        configured = config.get("fast_system_prompt", "")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return (
            "Execution mode: fast. Prefer one concise pass using strongest available evidence. "
            "Avoid unnecessary planning/verification loops."
        )
    configured = config.get("deep_system_prompt", "")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return (
        "Execution mode: deep. Use multi-hop reasoning over evidence, verify coverage and citations, "
        "and refine once when checks fail. Prioritize completeness on required fields before finalizing."
    )


def _mode_instruction_message(mode: str) -> dict:
    return {"role": "system", "content": _mode_instruction_text(mode)}


def _response_cache_enabled() -> bool:
    return bool(getattr(settings.web_search, "response_cache_enabled", True))


def _admissions_answer_schema_message(state: dict) -> dict | None:
    if not _is_admissions_requirements_query(state):
        return None
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    required_set = {str(item).strip() for item in required_fields if str(item).strip()}

    sections: list[str] = ["Program Snapshot"]
    if required_set & {
        "eligibility_requirements",
        "gpa_or_grade_threshold",
        "ects_or_prerequisite_credit_breakdown",
    }:
        sections.append("Eligibility Requirements")
    if required_set & {"language_requirements", "language_test_score_thresholds"}:
        sections.append("Language Requirements")
    if "application_deadline" in required_set or bool(state.get("deadline_query", False)):
        sections.append("Deadlines")
    if "tuition_or_fees" in required_set:
        sections.append("Fees")
    if "curriculum_focus" in required_set:
        sections.append("Curriculum")
    if "career_outcomes" in required_set:
        sections.append("Career Outcomes")
    include_admission_decision = "admission_decision_signal" in required_set
    if include_admission_decision:
        sections.append("Admission Decision")
    sections.extend(["Missing Information", "Sources"])

    content = (
        "Admissions answer schema (deep):\n"
        "- Use only the sections below, in this order:\n"
    )
    lines = [content.rstrip("\n")]
    for index, section in enumerate(sections, start=1):
        lines.append(f"  {index}) {section}")
    lines.append("- Do not include sections not requested by the query.")
    lines.append("- For requested fields, provide exact numbers/dates when present in evidence.")
    if required_set & {
        "eligibility_requirements",
        "gpa_or_grade_threshold",
        "ects_or_prerequisite_credit_breakdown",
    }:
        lines.append("- Under Eligibility Requirements include explicit lines for:")
        lines.append("  - Degree/background requirement")
        lines.append("  - GPA/grade threshold")
        lines.append("  - ECTS/prerequisite credit breakdown")
    if required_set & {"language_requirements", "language_test_score_thresholds"}:
        lines.append("- Under Language Requirements include explicit lines for:")
        lines.append("  - Accepted tests")
        lines.append("  - Minimum scores")
    if include_admission_decision:
        lines.append("- Under Admission Decision include:")
        lines.append("  - Verdict: Likely / Risky / Unknown")
        lines.append("  - Why (1-2 lines grounded in evidence)")
        lines.append("  - What applicant profile data is still needed")
    else:
        lines.append("- Do not include an Admission Decision section unless explicitly requested.")
    lines.append('- If a field is unavailable, write exactly: "Not verified from sources."')
    return {"role": "system", "content": "\n".join(lines)[:_RETRIEVAL_CONTEXT_MAX_CHARS]}


def _answer_style_instruction_message(mode: str, state: dict) -> dict:
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    required_text = ", ".join(
        _required_field_label(str(item).strip())
        for item in required_fields[:6]
        if str(item).strip()
    )
    if not required_text:
        required_text = "query-specific required fields"

    if str(mode).strip().lower() == _FAST_MODE:
        content = (
            "Answer style policy (fast): concise and readable. Use short paragraphs or short bullets. "
            "No internal scaffolding headings. Keep one final 'Sources' section with unique URLs only."
        )
        return {"role": "system", "content": content}

    content = (
        "Answer style policy (deep):\n"
        "- Start with a 1-2 sentence direct answer.\n"
        "- Then use clear markdown sections only when relevant (for example: Program Overview, "
        "Eligibility Requirements, Language Requirements, Deadlines, Fees, Missing Information, Sources).\n"
        "- Use only sections requested by the query/required fields; do not pad with extra sections.\n"
        "- Ensure required fields are explicitly covered: "
        f"{required_text}.\n"
        "- Keep one fact per bullet and avoid duplicated lines.\n"
        "- Prefer structured field lines over long paragraphs for admissions questions.\n"
        "- Do not include an Admission Decision section unless the user explicitly asks for admission likelihood.\n"
        "- Add one inline URL citation to each factual bullet.\n"
        "- Do not output scaffolding headings like 'Evidence and caveats' or 'Claim-by-Claim Citations'.\n"
        "- End with one 'Sources' section listing unique URLs only."
    )
    return {"role": "system", "content": content}


def _insert_system_message_before_dialog(messages: list, message: dict) -> list:
    if not isinstance(messages, list) or not isinstance(message, dict):
        return messages
    insert_at = 0
    for item in messages:
        if not isinstance(item, dict) or str(item.get("role", "")).lower() != "system":
            break
        insert_at += 1
    return messages[:insert_at] + [message] + messages[insert_at:]


def _execution_policy(mode: str) -> dict:
    normalized = _normalized_request_mode(mode)
    if normalized == _FAST_MODE:
        return {
            "mode": _FAST_MODE,
            "planner_enabled": False,
            "verifier_enabled": False,
            "max_attempts": 1,
            "web_search_mode": _FAST_MODE,
            "auto_requested": False,
        }
    return {
        "mode": _DEEP_MODE,
        "planner_enabled": _agentic_planner_enabled(),
        "verifier_enabled": _agentic_verifier_enabled(),
        # Deep mode allows up to two repair rounds (initial draft + up to 2 revisions).
        "max_attempts": min(3, 1 + (_agentic_max_reflection_rounds() if _agentic_enabled() else 0)),
        "web_search_mode": _DEEP_MODE,
        "auto_requested": normalized == _AUTO_MODE,
    }


def _should_escalate_auto_to_deep(*, result: str, state: dict) -> bool:
    if result == _NO_RELEVANT_INFORMATION_DETAIL:
        return True
    issues = state.get("agent_last_issues")
    if isinstance(issues, list) and issues:
        return True
    if bool(state.get("deadline_query", False)):
        return True
    top_similarity = _safe_float(state.get("retrieval_top_similarity"))
    if top_similarity is not None and top_similarity < 0.5:
        return True
    trust_confidence = _safe_float(state.get("trust_confidence"))
    if trust_confidence is not None and trust_confidence < _TRUST_LOW_CONFIDENCE_THRESHOLD:
        return True
    if bool(state.get("trust_contradiction_flag", False)):
        return True
    if int(state.get("retrieval_source_count", 0) or 0) <= 0:
        return True
    return False


def _is_citation_grounding_required() -> bool:
    chat_config = prompts.get("chat", {})
    if not isinstance(chat_config, dict):
        return True
    raw = chat_config.get("citation_grounded_required", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _chat_config() -> dict:
    config = prompts.get("chat", {})
    return config if isinstance(config, dict) else {}


def _agentic_config() -> dict:
    config = _chat_config().get("agentic", {})
    return config if isinstance(config, dict) else {}


def _agentic_enabled() -> bool:
    raw = _agentic_config().get("enabled", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _agentic_max_reflection_rounds() -> int:
    raw = _agentic_config().get("max_reflection_rounds", 1)
    value = _safe_int(raw)
    if value is None:
        return 1
    return max(0, min(3, value))


def _agentic_instruction_text() -> str:
    configured = _agentic_config().get("system_prompt", "")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return (
        "Internal workflow: (1) plan from provided evidence, (2) draft answer, "
        "(3) verify each factual claim has allowed inline URL citations, "
        "(4) identify uncertainty for weak/conflicting evidence, (5) revise once if needed. "
        "Keep internal reasoning hidden. Do not output chain-of-thought. "
        "Final answer must be clean for users: no internal scaffolding headings, "
        "clear sections, and one unique Sources list."
    )


def _agentic_instruction_message() -> dict:
    return {"role": "system", "content": _agentic_instruction_text()}


def _agentic_planner_enabled() -> bool:
    raw = _agentic_config().get("planner_enabled", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _agentic_verifier_enabled() -> bool:
    raw = _agentic_config().get("verifier_enabled", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _agentic_verifier_min_coverage_score() -> float:
    raw = _agentic_config().get("verifier_min_coverage_score", 0.75)
    value = _safe_float(raw)
    if value is None:
        return 0.75
    return max(0.0, min(1.0, value))


def _deep_answer_min_confidence() -> float:
    raw = getattr(settings.web_search, "deep_answer_min_confidence", 0.72)
    value = _safe_float(raw)
    if value is None:
        return 0.72
    return max(0.0, min(1.0, value))


def _agentic_planner_system_prompt() -> str:
    configured = _agentic_config().get("planner_system_prompt", "")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return (
        "You are an internal answer planner. Think privately and return JSON only with keys: "
        "intent, subquestions, success_criteria."
    )


def _agentic_verifier_system_prompt() -> str:
    configured = _agentic_config().get("verifier_system_prompt", "")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return (
        "You are an internal answer verifier. Return JSON only with keys: "
        "pass, coverage_score, issues, missing_points, revision_guidance."
        " Enforce claim-level citations (inline URL per factual claim) and explicit uncertainty "
        "when evidence is conflicting, stale, or weak."
    )


def _extract_json_object(raw: str) -> dict | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_agentic_text_list(values, *, limit: int = 6) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value).split()).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item[:200])
        if len(normalized) >= limit:
            break
    return normalized


def _is_deadline_query(user_prompt: str) -> bool:
    return bool(_DEADLINE_QUERY_RE.search(str(user_prompt or "")))


def _is_freshness_sensitive_query(user_prompt: str) -> bool:
    return bool(_FRESHNESS_QUERY_RE.search(str(user_prompt or "")))


def _has_date_like_value(text: str) -> bool:
    return bool(_DATE_LIKE_RE.search(str(text or "")))


def _llm_mock_delay_seconds() -> float:
    """Return synthetic LLM latency in seconds for mock mode."""
    raw = os.getenv(_LLM_MOCK_DELAY_MS_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        delay_ms = max(0, int(raw))
    except ValueError:
        return 0.0
    return delay_ms / 1000.0


def _llm_mock_stream_chunk_chars() -> int:
    """Return per-chunk character size used by mock streaming responses."""
    raw = os.getenv(_LLM_MOCK_STREAM_CHUNK_CHARS_ENV, "").strip()
    if not raw:
        return 24
    try:
        return max(1, int(raw))
    except ValueError:
        return 24


def _llm_mock_text(messages: list) -> str:
    """Build deterministic synthetic model output for load testing."""
    configured = os.getenv(_LLM_MOCK_TEXT_ENV, "").strip()
    if configured:
        return configured
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return f"[mock-llm] {content[:240]}"
    return "[mock-llm] synthetic response"


def _mock_completion_response(messages: list):
    """Build one compatibility response object matching Bedrock adapter shape."""
    text = _llm_mock_text(messages)
    prompt_tokens = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", "")).strip()
        if content:
            prompt_tokens += max(1, len(content.split()))
    prompt_tokens = max(1, prompt_tokens)
    completion_tokens = max(1, len(text.split()))
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice], usage=usage)


def _extract_llm_usage(response) -> dict:
    """Normalize response usage information into a JSON-safe dictionary."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", None))
    completion_tokens = _safe_int(getattr(usage, "completion_tokens", None))
    total_tokens = _safe_int(getattr(usage, "total_tokens", None))
    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return {}

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def _record_json_metrics(record: dict) -> None:
    """Enqueue per-request metrics for background persistence."""
    if not settings.queue.metrics_aggregation_queue_enabled:
        return
    if not settings.queue.metrics_aggregation_queue_url.strip():
        logger.warning("Metrics queue URL missing; dropping metrics event.")
        return

    async def _enqueue() -> None:
        try:
            await asyncio.to_thread(enqueue_metrics_record_event, record)
        except Exception:
            logger.warning("Metrics event enqueue failed; continuing.")

    _track_background_task(
        asyncio.create_task(_enqueue()),
        label="Metrics event enqueue",
    )
    # Yield control so this helper actually uses async semantics while remaining fire-and-forget.
    await asyncio.sleep(0)


def _build_json_metrics_record(
    *,
    request_id: str,
    started_at: float,
    user_id: str,
    session_id: str | None,
    user_prompt: str,
    safe_user_prompt: str,
    answer: str,
    outcome: str,
    metrics_state: dict | None = None,
    error_message: str = "",
    **legacy_state,
) -> dict:
    """Build the per-request metrics payload persisted to JSON."""
    state: dict = dict(legacy_state)
    if isinstance(metrics_state, dict):
        state.update(metrics_state)

    return {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": str(session_id or user_id),
        "question": user_prompt,
        "question_sanitized": safe_user_prompt,
        "answer": answer,
        "outcome": outcome,
        "timings_ms": {
            "overall_response_ms": _elapsed_ms(started_at),
            "llm_response_ms": state.get("model_ms"),
            "short_term_memory_ms": state.get("build_context_ms"),
            "long_term_memory_ms": state.get("retrieval_ms"),
            "memory_update_ms": state.get("memory_update_ms"),
            "cache_read_ms": state.get("cache_read_ms"),
            "cache_write_ms": state.get("cache_write_ms"),
            "evaluation_trace_ms": state.get("evaluation_trace_ms"),
        },
        "retrieval": {
            "strategy": str(state.get("retrieval_strategy", "none")),
            "intent": str(state.get("query_intent", "unknown")),
            "query_variants": state.get("retrieval_query_variants") or [],
            "result_count": int(state.get("retrieved_count", 0)),
            "source_count": int(state.get("retrieval_source_count", 0)),
            "top_similarity": state.get("retrieval_top_similarity"),
            "reranker_applied": bool(state.get("retrieval_reranker_applied", False)),
            "reranker_ms": state.get("retrieval_reranker_ms"),
            "selective_before_count": int(state.get("retrieval_selective_before_count", 0) or 0),
            "selective_after_count": int(state.get("retrieval_selective_after_count", 0) or 0),
            "selective_dropped": int(state.get("retrieval_selective_dropped", 0) or 0),
            "avg_quality": state.get("retrieval_avg_quality"),
            "citation_required": bool(state.get("citation_required", False)),
            "citation_min_hosts": int(state.get("citation_min_hosts", 1) or 1),
            "evidence_urls": state.get("evidence_urls") or [],
            "evidence_domain_count": int(state.get("evidence_domain_count", 0) or 0),
            "trust_confidence": state.get("trust_confidence"),
            "trust_freshness": state.get("trust_freshness"),
            "trust_contradiction_flag": bool(state.get("trust_contradiction_flag", False)),
            "trust_authority_score": state.get("trust_authority_score"),
            "trust_agreement_score": state.get("trust_agreement_score"),
            "claim_citation_coverage": state.get("claim_citation_coverage"),
            "claim_snippet_grounding_coverage": state.get("claim_snippet_grounding_coverage"),
            "claim_snippet_conflict_count": int(state.get("claim_snippet_conflict_count", 0) or 0),
            "claim_evidence_map": state.get("claim_evidence_map") or [],
            "evidence": state.get("retrieval_evidence") or [],
        },
        "quality": state.get("quality", {}),
        "hallucination_proxy": (state.get("quality") or {}).get("hallucination_proxy"),
        "llm_usage": state.get("llm_usage", {}),
        "guardrails": {
            "input_reason": str(state.get("input_guard_reason", "")),
            "context_reason": str(state.get("context_guard_reason", "")),
            "output_reason": str(state.get("output_guard_reason", "")),
        },
        "model": {
            "provider": "amazon_bedrock",
            "used_fallback": bool(state.get("used_fallback_model", False)),
            "primary_model_id": settings.bedrock.primary_model_id,
            "fallback_model_id": settings.bedrock.fallback_model_id,
            "requested_mode": str(state.get("requested_mode", _DEFAULT_EXECUTION_MODE)),
            "execution_mode": str(state.get("execution_mode", _DEFAULT_EXECUTION_MODE)),
            "auto_escalated": bool(state.get("auto_escalated", False)),
            "role_model_ids": state.get("role_model_ids") or {},
            "agentic_enabled": bool(state.get("agentic_enabled", False)),
            "agent_rounds": int(state.get("agent_rounds", 0) or 0),
            "agent_last_issues": state.get("agent_last_issues") or [],
        },
        "error": error_message,
    }


def _latency_metrics_key() -> str:
    """Return the Redis key used to store aggregate LLM latency metrics."""
    return app_scoped_key("metrics", "llm", "latency")


async def _record_latency_metrics(started_at: float, outcome: str):
    """Persist request latency metrics for observability and ops reporting."""
    latency_ms = _elapsed_ms(started_at)
    key = _latency_metrics_key()
    try:
        await _redis_call(async_redis_client.hincrby, key, "count", 1)
        await _redis_call(async_redis_client.hincrby, key, "total_ms", latency_ms)
        current_max = await _redis_call(async_redis_client.hget, key, "max_ms")
        if current_max is None or latency_ms > int(current_max):
            await _redis_call(async_redis_client.hset, key, "max_ms", latency_ms)
        await _redis_call(
            async_redis_client.hset,
            key,
            mapping={
                "last_ms": latency_ms,
                "last_outcome": outcome,
            },
        )
    except Exception:
        logger.warning("Latency metrics persistence failed; continuing.")


async def _record_pipeline_stage_metrics(
    *,
    build_context_ms: int,
    retrieval_ms: int,
    model_ms: int,
    retrieval_strategy: str,
    retrieved_count: int,
):
    """Persist stage-level latency metrics for successful full chat pipeline runs."""
    key = _latency_metrics_key()
    try:
        await _redis_call(async_redis_client.hincrby, key, "pipeline_count", 1)
        await _redis_call(
            async_redis_client.hincrby, key, "build_context_total_ms", build_context_ms
        )
        await _redis_call(async_redis_client.hincrby, key, "retrieval_total_ms", retrieval_ms)
        await _redis_call(async_redis_client.hincrby, key, "model_total_ms", model_ms)
        await _redis_call(
            async_redis_client.hset,
            key,
            mapping={
                "last_build_context_ms": build_context_ms,
                "last_retrieval_ms": retrieval_ms,
                "last_model_ms": model_ms,
                "last_retrieval_strategy": retrieval_strategy,
                "last_retrieved_count": retrieved_count,
            },
        )
    except Exception:
        logger.warning("Pipeline stage metrics persistence failed; continuing.")


def _build_retrieval_query(messages: list[dict]) -> str:
    """Build a retrieval query from the latest short-term conversation context."""
    text_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role != "user" or not isinstance(content, str) or not content.strip():
            continue
        text_parts.append(content.strip())
    if not text_parts:
        return ""
    latest = text_parts[-1]
    if len(text_parts) == 1:
        return latest[:_RETRIEVAL_QUERY_MAX_CHARS].strip()

    compact_latest = " ".join(latest.split()).strip()
    lower_latest = compact_latest.lower()
    follow_up = bool(
        any(lower_latest.startswith(prefix) for prefix in _FOLLOW_UP_PREFIXES)
        or (
            len(compact_latest) <= 220
            and _FOLLOW_UP_REFERENCE_RE.search(compact_latest) is not None
        )
        or (
            len(compact_latest) <= 260
            and _FOLLOW_UP_CONTEXT_ENTITY_RE.search(compact_latest) is not None
        )
    )
    if not follow_up:
        return latest[:_RETRIEVAL_QUERY_MAX_CHARS].strip()
    return "\n\n".join(text_parts[-2:])[:_RETRIEVAL_QUERY_MAX_CHARS].strip()


def _classify_query_intent(user_prompt: str) -> str:
    text = " ".join(str(user_prompt or "").split()).strip().lower()
    if not text:
        return "unknown"
    if _INTENT_COMPARE_RE.search(text):
        return "comparison"
    # Mixed admissions questions should not be downgraded to pure deadline intent.
    if _INTENT_DEADLINE_RE.search(text) and _INTENT_REQUIREMENTS_RE.search(text):
        return "fact_lookup"
    if _INTENT_DEADLINE_RE.search(text):
        return "deadline"
    if _INTENT_PLAN_RE.search(text):
        return "strategy"
    if _INTENT_FACT_RE.search(text):
        return "fact_lookup"
    return "exploration"


def _comparison_entities_from_prompt(user_prompt: str) -> list[str]:
    text = " ".join(str(user_prompt or "").split()).strip()
    if not text:
        return []
    match = re.search(
        r"(?:compare\s+)?(.+?)\s+(?:vs|versus)\s+(.+?)(?:,| for | including |$)",
        text,
        re.I,
    )
    if not match:
        return []

    candidates = [match.group(1).strip(), match.group(2).strip()]
    entities: list[str] = []
    for candidate in candidates:
        tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", candidate)
            if len(token) >= 3 and token.lower() not in _COMPARISON_ENTITY_STOPWORDS
        ]
        if not tokens:
            continue
        normalized = " ".join(tokens[:4]).strip()
        if normalized:
            entities.append(normalized)
    deduped: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped[:2]


def _comparison_entity_domain(entity: str) -> str:
    normalized = " ".join(str(entity or "").split()).strip().lower()
    if not normalized:
        return ""
    for hint, domain in _COMPARISON_ENTITY_DOMAIN_HINTS:
        if hint in normalized or normalized in hint:
            return domain
    return ""


def _comparison_entity_tokens(entity: str) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]{3,}", str(entity or "").lower())
        if token not in _COMPARISON_ENTITY_STOPWORDS
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped[:4]


def _comparison_entity_focus_query(entity: str) -> str:
    compact = " ".join(str(entity or "").split()).strip()
    if not compact:
        return ""
    return (
        f"{compact} data science master's program "
        "admission requirements application deadline"
    )


def _result_mentions_entity(result: dict, entity: str) -> bool:
    if not isinstance(result, dict):
        return False
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    text = " ".join(
        str(item)
        for item in (
            metadata.get("url", ""),
            metadata.get("title", ""),
            result.get("source_path", ""),
            result.get("content", ""),
        )
    ).lower()
    if not text:
        return False
    domain = _comparison_entity_domain(entity)
    if domain and domain in text:
        return True
    tokens = _comparison_entity_tokens(entity)
    if not tokens:
        return False
    return any(token in text for token in tokens[:2])


def _missing_comparison_entities(results: list[dict], entities: list[str]) -> list[str]:
    if not isinstance(entities, list) or len(entities) < 2:
        return []
    missing: list[str] = []
    for entity in entities[:2]:
        if not any(_result_mentions_entity(result, str(entity)) for result in results):
            missing.append(str(entity))
    return missing


def _comparison_entity_expansion_queries(
    *, base_query: str, state: dict, current_results: list[dict] | None = None
) -> list[str]:
    entities = state.get("comparison_entities")
    entities = entities if isinstance(entities, list) else []
    if len(entities) < 2:
        entities = _comparison_entities_from_prompt(base_query)
    if len(entities) < 2:
        return []
    missing = (
        _missing_comparison_entities(current_results or [], entities)
        if isinstance(current_results, list) and current_results
        else entities[:2]
    )
    if not missing:
        return []
    queries: list[str] = []
    for entity in missing[:2]:
        focus = _comparison_entity_focus_query(str(entity))
        if not focus:
            continue
        domain = _comparison_entity_domain(str(entity))
        if domain:
            queries.append(f"{focus} site:{domain}")
        queries.append(f"{focus} official")
        queries.append(f"{str(entity).strip()} data science master's program official")
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        compact = " ".join(str(query).split()).strip()
        key = compact.lower()
        if not compact or key in seen:
            continue
        seen.add(key)
        deduped.append(compact[:_RETRIEVAL_QUERY_MAX_CHARS])
        if len(deduped) >= 4:
            break
    return deduped


def _required_answer_fields(user_prompt: str, *, intent: str) -> list[str]:
    text = " ".join(str(user_prompt or "").split()).strip().lower()
    required: list[str] = []
    asks_apply_portal = bool(
        re.search(
            r"\b(application portal|online application|apply online|bewerbungsportal|"
            r"where (?:can i|to) apply|how to apply|where can i apply)\b",
            text,
        )
    )
    has_requirements_scope = bool(
        re.search(
            r"\b(admission requirements?|course requirements?|eligibility|entry criteria|"
            r"required documents?|prerequisites?)\b",
            text,
        )
    )
    has_language_scope = bool(
        re.search(r"\b(language|ielts|toefl|english|german|cefr|international students?)\b", text)
    )
    has_language_score_scope = bool(
        re.search(
            r"\b(ielts|toefl|cefr|minimum score|score threshold|required score|test score|band)\b",
            text,
        )
    )
    if intent == "comparison":
        required.append("comparison_between_requested_entities")
    if "deadline" in text or "apply by" in text or "last date" in text:
        required.append("application_deadline")
    if asks_apply_portal:
        required.append("application_portal")
    if "document" in text or "required document" in text:
        required.append("required_documents")
    if has_requirements_scope:
        required.append("eligibility_requirements")
        required.append("gpa_or_grade_threshold")
        required.append("ects_or_prerequisite_credit_breakdown")
    if "tuition" in text or "fees" in text or "semester contribution" in text:
        required.append("tuition_or_fees")
    if has_language_scope:
        required.append("language_requirements")
    if has_language_score_scope:
        required.append("language_test_score_thresholds")
    if re.search(r"\b(curriculum|focus|modules?|syllabus|course structure|taught|subjects?)\b", text):
        required.append("curriculum_focus")
    if "career" in text or "outcome" in text:
        required.append("career_outcomes")
    if "scholarship" in text:
        required.append("scholarship_options")
    if "visa" in text or "work-right" in text or "work right" in text:
        required.append("visa_or_work_rights")
    if "aps" in text:
        required.append("aps_requirement_stage")
    if (
        "can i get admission" in text
        or "whether i can get admission" in text
        or "chance of admission" in text
        or "am i eligible" in text
        or "admission decision" in text
    ):
        required.append("admission_decision_signal")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in required:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped[:8]


def _required_field_label(field: str) -> str:
    key = str(field or "").strip()
    if not key:
        return "Unknown field"
    label = _REQUIRED_FIELD_LABELS.get(key, "")
    if label:
        return label
    return key.replace("_", " ").strip().title()


def _is_admissions_requirements_query(state: dict) -> bool:
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    required_set = {str(item).strip() for item in required_fields if str(item).strip()}
    admission_markers = {
        "eligibility_requirements",
        "gpa_or_grade_threshold",
        "ects_or_prerequisite_credit_breakdown",
        "language_requirements",
        "language_test_score_thresholds",
        "application_deadline",
        "application_portal",
    }
    return bool(required_set & admission_markers)


def _admissions_missing_web_fields(state: dict) -> list[str]:
    raw = state.get("web_required_fields_missing")
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = " ".join(str(item).split()).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized[:8]


def _admissions_critical_web_fields_missing(state: dict) -> list[str]:
    if not _is_admissions_requirements_query(state):
        return []
    missing = _admissions_missing_web_fields(state)
    return [item for item in missing if item in _ADMISSIONS_CRITICAL_WEB_REQUIRED_FIELDS]


def _required_fields_system_message(state: dict) -> dict | None:
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    if not required_fields:
        return None
    entities = state.get("comparison_entities")
    entities = entities if isinstance(entities, list) else []
    lines = [
        "Answer schema requirements:",
        "- Cover every required field below using evidence-backed facts.",
        '- If a required field cannot be verified, explicitly write: "Not verified from sources."',
        "- Prefer exact numbers/dates over generic wording when present in evidence.",
        "- Do not return a generic disclaimer as the primary answer.",
        "Required fields:",
    ]
    for item in required_fields:
        lines.append(f"- {item}: {_required_field_label(item)}")
    if entities:
        lines.append("Required comparison entities:")
        lines.extend(f"- {entity}" for entity in entities[:2])
    return {"role": "system", "content": "\n".join(lines)[:_RETRIEVAL_CONTEXT_MAX_CHARS]}


def _has_not_verified_marker(answer: str) -> bool:
    lowered = str(answer or "").lower()
    return "not verified from evidence" in lowered or "not verified from sources" in lowered


def _answer_matches_required_field(
    field: str, answer: str, *, comparison_entities: list[str] | None = None
) -> bool:
    lowered = str(answer or "").lower()
    if not lowered:
        return False
    if field == "application_deadline":
        return _has_date_like_value(answer) or bool(
            re.search(r"\bdeadline|apply by|last date\b", lowered)
        )
    if field == "application_portal":
        return bool(
            re.search(
                r"\b(application portal|apply online|online application|bewerbungsportal|application system)\b",
                lowered,
            )
            and ("http://" in lowered or "https://" in lowered or "www." in lowered)
        )
    if field == "required_documents":
        return bool(re.search(r"\bdocument|documents|required\b", lowered))
    if field == "eligibility_requirements":
        return bool(re.search(r"\brequirement|requirements|eligibility|eligible\b", lowered))
    if field == "gpa_or_grade_threshold":
        return bool(
            re.search(r"\b(gpa|grade point|minimum grade|cgpa|grade average)\b", lowered)
            and (
                re.search(r"\b\d(?:[.,]\d+)?\b", lowered)
                or _has_not_verified_marker(answer)
            )
        )
    if field == "ects_or_prerequisite_credit_breakdown":
        return bool(
            re.search(r"\b(ects|credit|credits|cp|prerequisite)\b", lowered)
            and (
                re.search(r"\b\d{1,3}(?:[.,]\d+)?\b", lowered)
                or _has_not_verified_marker(answer)
            )
        )
    if field == "tuition_or_fees":
        return bool(
            re.search(r"\btuition|fee|fees|semester contribution|eur|€|no tuition\b", lowered)
        )
    if field == "language_requirements":
        return bool(re.search(r"\blanguage|english|german|ielts|toefl|c1|c2|b2\b", lowered))
    if field == "language_test_score_thresholds":
        return bool(
            re.search(r"\b(ielts|toefl|cefr|cambridge|duolingo)\b", lowered)
            and (
                re.search(r"\b\d(?:[.,]\d+)?\b", lowered)
                or _has_not_verified_marker(answer)
            )
        )
    if field == "curriculum_focus":
        return bool(re.search(r"\bcurriculum|focus|module|coursework\b", lowered))
    if field == "career_outcomes":
        return bool(re.search(r"\bcareer|employment|outcome|job|industry\b", lowered))
    if field == "scholarship_options":
        return bool(re.search(r"\bscholarship|funding|grant|stipend\b", lowered))
    if field == "visa_or_work_rights":
        return bool(re.search(r"\bvisa|work|hours|residence permit\b", lowered))
    if field == "aps_requirement_stage":
        return bool(
            re.search(r"\baps|certificate|required|before applying|application stage\b", lowered)
        )
    if field == "admission_decision_signal":
        return bool(
            re.search(
                r"\b(admission decision|decision snapshot|verdict|likely|risky|unclear|cannot determine)\b",
                lowered,
            )
        )
    if field == "comparison_between_requested_entities":
        entities = comparison_entities if isinstance(comparison_entities, list) else []
        if len(entities) < 2:
            return bool(re.search(r"\bcompare|comparison|both|whereas|while\b", lowered))
        matched = 0
        for entity in entities[:2]:
            entity_tokens = [
                token
                for token in re.findall(r"[A-Za-z0-9]{3,}", str(entity).lower())
                if token not in _COMPARISON_ENTITY_STOPWORDS
            ]
            if any(token in lowered for token in entity_tokens[:2]):
                matched += 1
        return matched >= 2
    return True


def _missing_required_answer_fields(answer: str, state: dict) -> list[str]:
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    if not required_fields:
        return []
    comparison_entities = state.get("comparison_entities")
    comparison_entities = comparison_entities if isinstance(comparison_entities, list) else []
    missing: list[str] = []
    for field in required_fields:
        if not _answer_matches_required_field(
            str(field), answer, comparison_entities=comparison_entities
        ):
            missing.append(str(field))
    return missing[:8]


def _query_decomposition_limit(state: dict | None = None) -> int:
    configured = _safe_int(getattr(settings.web_search, "max_query_variants", 3))
    if configured is None:
        return 3
    deep_configured = _safe_int(getattr(settings.web_search, "deep_max_query_variants", configured))
    if deep_configured is None:
        deep_configured = configured
    mode = _mode_from_state(state)
    if mode == _FAST_MODE:
        return max(1, min(2, configured))
    return max(2, min(8, max(configured, deep_configured)))


def _decompose_retrieval_queries(*, base_query: str, state: dict) -> list[str]:
    query = " ".join(str(base_query or "").split()).strip()
    if not query:
        return []

    intent = str(state.get("query_intent", "exploration")).strip().lower()
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    required_set = {str(item).strip() for item in required_fields if str(item).strip()}
    variants: list[str] = [query]
    if intent == "deadline":
        variants.extend(
            [
                f"{query} official deadline",
                f"{query} admission requirements official",
            ]
        )
    elif intent == "comparison":
        variants.extend(
            [
                f"{query} official program details",
                f"{query} tuition fees requirements",
            ]
        )
    elif intent == "strategy":
        variants.extend(
            [
                f"{query} official guidance",
                f"{query} risks tradeoffs",
            ]
        )
    elif intent == "fact_lookup":
        variants.extend(
            [
                f"{query} official source",
                f"{query} latest update",
            ]
        )
    else:
        variants.append(f"{query} official source")

    if required_set & {
        "eligibility_requirements",
        "gpa_or_grade_threshold",
        "ects_or_prerequisite_credit_breakdown",
    }:
        variants.extend(
            [
                f"{query} official admission requirements GPA ECTS prerequisites",
                f"{query} official eligibility criteria minimum grade credit requirements",
                f"{query} official examination regulations admission criteria PDF",
            ]
        )
    if required_set & {"language_requirements", "language_test_score_thresholds"}:
        variants.extend(
            [
                f"{query} official language requirements IELTS TOEFL minimum score",
                f"{query} accepted English tests CEFR minimum score official",
            ]
        )
    if "application_deadline" in required_set:
        variants.append(f"{query} official application deadline exact date")

    limit = _query_decomposition_limit(state)
    normalized: list[str] = []
    max_queries = 4 if intent == "comparison" else 3
    seen: set[str] = set()
    for variant in variants:
        compact = " ".join(str(variant).split()).strip()
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(compact[:_RETRIEVAL_QUERY_MAX_CHARS])
        if len(normalized) >= limit:
            break
    return normalized


def _web_expansion_queries(
    *,
    base_query: str,
    state: dict,
    low_similarity: bool,
    insufficient_domains: bool,
    current_results: list[dict] | None = None,
) -> list[str]:
    query = " ".join(str(base_query or "").split()).strip()
    if not query:
        return []
    if not low_similarity and not insufficient_domains:
        return []

    intent = str(state.get("query_intent", "exploration")).strip().lower()
    missing_fields = state.get("web_required_fields_missing")
    missing_fields = missing_fields if isinstance(missing_fields, list) else []
    variants: list[str] = []
    if intent == "comparison":
        variants.extend(
            _comparison_entity_expansion_queries(
                base_query=query,
                state=state,
                current_results=current_results,
            )
        )
    if insufficient_domains:
        variants.extend(
            [
                f"{query} official requirements",
                f"{query} official application details",
            ]
        )
    if low_similarity:
        variants.extend(
            [
                f"{query} exact program page",
                f"{query} official eligibility criteria",
            ]
        )
    for field in [str(item).strip() for item in missing_fields if str(item).strip()][:4]:
        if field in {"admission_requirements", "eligibility_requirements", "gpa_or_grade_threshold"}:
            variants.append(f"{query} official admission requirements minimum grade GPA ECTS")
        elif field in {"language_requirements", "language_test_score_thresholds"}:
            variants.append(f"{query} official IELTS TOEFL CEFR minimum score")
        elif field == "application_deadline":
            variants.append(f"{query} official application deadline exact date")
    if intent == "deadline":
        variants.append(f"{query} official deadline date")
    elif intent == "comparison":
        variants.append(f"{query} side by side official information")
    elif intent == "fact_lookup":
        variants.append(f"{query} official source")

    normalized: list[str] = []
    max_queries = 4 if intent == "comparison" else 3
    seen: set[str] = set()
    for variant in variants:
        compact = " ".join(str(variant).split()).strip()
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(compact[:_RETRIEVAL_QUERY_MAX_CHARS])
        if len(normalized) >= max_queries:
            break
    return normalized


def _retrieval_result_label(metadata, index: int) -> str:
    if not isinstance(metadata, dict):
        return f"Result {index}"
    label_parts: list[str] = []
    university = metadata.get("university")
    section_heading = metadata.get("section_heading")
    if isinstance(university, str) and university.strip():
        label_parts.append(university.strip())
    if isinstance(section_heading, str) and section_heading.strip():
        label_parts.append(section_heading.strip())
    return " | ".join(label_parts) if label_parts else f"Result {index}"


def _retrieval_content_and_metadata(result) -> tuple[str, dict]:
    if not isinstance(result, dict):
        return "", {}
    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        return "", {}
    metadata = result.get("metadata")
    return content, metadata if isinstance(metadata, dict) else {}


def _prompt_retrieval_result_limit() -> int:
    reranker_target = max(3, min(6, int(settings.bedrock.reranker_top_n)))
    return max(_RETRIEVAL_MAX_PROMPT_RESULTS, reranker_target)


def _format_retrieval_context(retrieval_result: dict) -> dict | None:
    """Convert retrieved long-term chunks into a single system-context message."""
    results = retrieval_result.get("results", []) if isinstance(retrieval_result, dict) else []
    if not isinstance(results, list) or not results:
        return None

    lines = [
        "Retrieved long-term knowledge. Use this only when relevant to the user's request.",
    ]
    max_items = _prompt_retrieval_result_limit()
    seen_chunks: set[str] = set()
    used_results = 0
    for result in results:
        content, metadata = _retrieval_content_and_metadata(result)
        if not content:
            continue
        dedupe_key = " ".join(content.lower().split())[:180]
        if dedupe_key in seen_chunks:
            continue
        seen_chunks.add(dedupe_key)
        used_results += 1

        label = _retrieval_result_label(metadata, used_results)
        compact_content = " ".join(content.split())[:_RETRIEVAL_CHUNK_MAX_CHARS]
        lines.append(f"{used_results}. {label}: {compact_content}")
        if used_results >= max_items:
            break

    if len(lines) == 1:
        return None
    joined = "\n".join(lines)
    return {"role": "system", "content": joined[:_RETRIEVAL_CONTEXT_MAX_CHARS]}


def _web_search_key_present() -> bool:
    configured = str(settings.web_search.api_key_env_name).strip()
    candidate_names = [configured, "TAVILY_WEB_SEARCH", "WEB_SEARCH_API_KEY"]
    for env_name in candidate_names:
        if env_name and os.getenv(env_name, "").strip():
            return True
    return False


def _result_similarity(result: dict) -> float | None:
    """Derive a normalized similarity score [0,1] from retrieval metadata."""
    if not isinstance(result, dict):
        return None

    explicit_similarity = _safe_float(result.get("similarity"))
    if explicit_similarity is not None:
        return max(0.0, min(1.0, explicit_similarity))

    score = _safe_float(result.get("score"))
    if score is not None and 0.0 <= score <= 1.0:
        return score

    distance = _safe_float(result.get("distance"))
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - distance))


def _top_retrieval_similarity(results: list[dict]) -> float | None:
    """Return best similarity score among retrieval results when available."""
    if not isinstance(results, list) or not results:
        return None
    best: float | None = None
    for result in results:
        similarity = _result_similarity(result)
        if similarity is None:
            continue
        if best is None or similarity > best:
            best = similarity
    return best


def _result_identity_key(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    chunk_id = str(result.get("chunk_id", "")).strip()
    source_path = str(result.get("source_path", "")).strip()
    content = " ".join(str(result.get("content", "")).split()).lower()[:220]
    return chunk_id or source_path or content


def _result_domain(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    candidate = _normalized_url(str(metadata.get("url", ""))) or _normalized_url(
        str(result.get("source_path", ""))
    )
    return _normalized_host_from_url(candidate)


def _domain_authority_signal(host: str) -> float:
    normalized = str(host or "").strip().lower()
    if not normalized:
        return 0.45
    if (
        normalized.endswith(".gov")
        or normalized.endswith(".gov.de")
        or normalized.endswith(".bund.de")
    ):
        return 0.98
    if ".uni-" in normalized or normalized.endswith(".edu"):
        return 0.92
    if any(
        normalized.endswith(suffix)
        for suffix in (
            "tum.de",
            "lmu.de",
            "rwth-aachen.de",
            "uni-bonn.de",
            "daad.de",
            "aps-india.de",
        )
    ):
        return 0.9
    if normalized.endswith(".de") or normalized.endswith(".eu"):
        return 0.78
    return 0.62


def _result_quality_score(result: dict) -> float:
    if not isinstance(result, dict):
        return 0.0
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    trust_components = metadata.get("trust_components")
    trust_components = trust_components if isinstance(trust_components, dict) else {}

    trust_score = _safe_float(metadata.get("trust_score"))
    if trust_score is None:
        trust_score = _safe_float(result.get("_trust_score"))
    similarity = _result_similarity(result)
    rerank_score = _safe_float(result.get("rerank_score"))
    host = _result_domain(result)
    domain_authority = _domain_authority_signal(host)
    content = " ".join(str(result.get("content", "")).split())
    content_richness = _clamp01(min(len(content), 420) / 420.0, fallback=0.0)
    authority = _safe_float(trust_components.get("authority"))
    if authority is None:
        authority = domain_authority
    else:
        authority = (_clamp01(authority) * 0.7) + (domain_authority * 0.3)
    recency = _safe_float(trust_components.get("recency"))
    agreement = _safe_float(trust_components.get("agreement"))

    weighted_total = 0.0
    weight_sum = 0.0

    if trust_score is not None:
        weighted_total += _clamp01(trust_score) * 0.3
        weight_sum += 0.3
    if similarity is not None:
        weighted_total += _clamp01(similarity) * 0.35
        weight_sum += 0.35
    if rerank_score is not None:
        weighted_total += _clamp01(rerank_score) * 0.2
        weight_sum += 0.2
    if authority is not None:
        weighted_total += _clamp01(authority) * 0.07
        weight_sum += 0.07
    if recency is not None:
        weighted_total += _clamp01(recency) * 0.05
        weight_sum += 0.05
    if agreement is not None:
        weighted_total += _clamp01(agreement) * 0.03
        weight_sum += 0.03
    weighted_total += content_richness * 0.05
    weight_sum += 0.05

    if weight_sum <= 0:
        similarity_fallback = _result_similarity(result)
        if similarity_fallback is not None:
            return _clamp01(similarity_fallback, fallback=0.5)
        return 0.5
    return _clamp01(weighted_total / weight_sum, fallback=0.5)


def _selective_retrieval_results(results: list[dict], state: dict) -> list[dict]:
    if not isinstance(results, list) or not results:
        return []
    if len(results) <= 2:
        return list(results)

    min_unique_domains = max(1, int(getattr(settings.web_search, "retrieval_min_unique_domains", 1)))
    max_keep = max(_max_context_results_for_mode(_mode_from_state(state)), min_unique_domains, 2)
    scored_rows = sorted(
        ((result, _result_quality_score(result), _result_domain(result)) for result in results),
        key=lambda item: item[1],
        reverse=True,
    )
    keep_threshold = _SELECTIVE_RESULT_SCORE_THRESHOLD
    selected: list[dict] = []
    selected_keys: set[str] = set()
    selected_domains: set[str] = set()
    min_keep = 2

    for result, quality_score, domain in scored_rows:
        key = _result_identity_key(result)
        if not key or key in selected_keys:
            continue
        if quality_score < keep_threshold and len(selected) >= min_keep:
            continue
        selected.append(result)
        selected_keys.add(key)
        if domain:
            selected_domains.add(domain)
        if len(selected) >= max_keep:
            break

    if len(selected) < min_keep:
        for result, _quality_score, domain in scored_rows:
            key = _result_identity_key(result)
            if not key or key in selected_keys:
                continue
            selected.append(result)
            selected_keys.add(key)
            if domain:
                selected_domains.add(domain)
            if len(selected) >= min_keep:
                break

    if len(selected_domains) < min_unique_domains:
        for result, _quality_score, domain in scored_rows:
            key = _result_identity_key(result)
            if not key or key in selected_keys:
                continue
            if not domain or domain in selected_domains:
                continue
            selected.append(result)
            selected_keys.add(key)
            selected_domains.add(domain)
            if len(selected_domains) >= min_unique_domains or len(selected) >= max_keep:
                break

    if len(selected) > max_keep:
        selected = selected[:max_keep]

    selected_quality = [_result_quality_score(item) for item in selected]
    selected_domains = {_result_domain(item) for item in selected if _result_domain(item)}
    avg_selected_quality = (
        round(sum(selected_quality) / len(selected_quality), 4) if selected_quality else None
    )
    state["retrieval_selective_before_count"] = len(results)
    state["retrieval_selective_after_count"] = len(selected)
    state["retrieval_selective_dropped"] = max(0, len(results) - len(selected))
    state["retrieval_avg_quality"] = avg_selected_quality
    state["retrieval_single_domain_low_quality"] = bool(
        len(selected_domains) <= 1
        and len(selected) >= 1
        and float(_safe_float(avg_selected_quality) or 0.0) < _SELECTIVE_RESULT_SCORE_THRESHOLD
    )
    emit_trace_event(
        "retrieval_selective_filter",
        {
            "before_count": len(results),
            "after_count": len(selected),
            "dropped_count": max(0, len(results) - len(selected)),
            "required_domains": min_unique_domains,
            "domain_count": len(selected_domains),
            "avg_quality": state.get("retrieval_avg_quality"),
            "single_domain_low_quality": bool(
                state.get("retrieval_single_domain_low_quality", False)
            ),
        },
    )
    return selected or list(results)


def _parse_published_date(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            continue
    return None


def _evidence_date_values(results: list[dict], *, limit: int = 6) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        content = " ".join(str(result.get("content", "")).split())
        if not content:
            continue
        for match in _DATE_LIKE_RE.findall(content):
            date_value = " ".join(str(match).split()).strip()
            if not date_value:
                continue
            lowered = date_value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            values.append(date_value)
            if len(values) >= limit:
                return values
    return values


def _date_sets_conflict(date_sets: list[set[str]]) -> bool:
    normalized_sets = [set(item) for item in date_sets if isinstance(item, set) and item]
    if len(normalized_sets) < 2:
        return False
    for index, first in enumerate(normalized_sets):
        for second in normalized_sets[index + 1 :]:
            if first == second:
                continue
            if first.issubset(second) or second.issubset(first):
                continue
            return True
    return False


def _deadline_date_sets_from_evidence(results: list[dict], *, limit: int = 6) -> list[set[str]]:
    date_sets: list[set[str]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        content = " ".join(str(result.get("content", "")).split())
        if not content:
            continue
        sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(content) if item.strip()]
        for sentence in sentences[:8]:
            lowered = sentence.lower()
            if not _DEADLINE_SENTENCE_HINT_RE.search(lowered):
                continue
            dates = {
                " ".join(str(match).split()).lower()
                for match in _DATE_LIKE_RE.findall(sentence)
                if str(match).strip()
            }
            if not dates:
                continue
            date_sets.append(dates)
            if len(date_sets) >= limit:
                return date_sets
    return date_sets


def _has_deadline_date_conflict(results: list[dict]) -> bool:
    return _date_sets_conflict(_deadline_date_sets_from_evidence(results, limit=8))


def _derive_evidence_trust_signals(results: list[dict], state: dict) -> None:
    if not isinstance(results, list) or not results:
        state["trust_confidence"] = None
        state["trust_freshness"] = "unknown"
        state["trust_contradiction_flag"] = False
        state["trust_authority_score"] = None
        state["trust_agreement_score"] = None
        state["trust_uncertainty_reasons"] = ["Evidence coverage is too thin."]
        return

    quality_scores = [_result_quality_score(item) for item in results]
    authority_scores: list[float] = []
    agreement_scores: list[float] = []
    recency_scores: list[float] = []
    published_ages_days: list[float] = []
    for result in results:
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        trust_components = metadata.get("trust_components")
        trust_components = trust_components if isinstance(trust_components, dict) else {}
        authority = _safe_float(trust_components.get("authority"))
        host = _result_domain(result)
        authority_hint = _domain_authority_signal(host)
        if authority is None:
            authority_scores.append(authority_hint)
        else:
            blended_authority = (_clamp01(authority) * 0.7) + (authority_hint * 0.3)
            authority_scores.append(_clamp01(blended_authority))
        agreement = _safe_float(trust_components.get("agreement"))
        if agreement is not None:
            agreement_scores.append(_clamp01(agreement))
        recency = _safe_float(trust_components.get("recency"))
        if recency is not None:
            recency_scores.append(_clamp01(recency))
        published = _parse_published_date(str(metadata.get("published_date", "")))
        if published is not None:
            published_ages_days.append(
                max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 86400.0)
            )

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    avg_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0.5
    avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.6
    avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
    domain_count = len({_result_domain(item) for item in results if _result_domain(item)})
    diversity_score = _clamp01(domain_count / 3.0, fallback=0.0)

    freshness_label = "unknown"
    freshness_score = 0.45
    if published_ages_days:
        avg_age_days = sum(published_ages_days) / len(published_ages_days)
        if avg_age_days <= 30:
            freshness_label = "fresh"
            freshness_score = 1.0
        elif avg_age_days <= 180:
            freshness_label = "recent"
            freshness_score = 0.75
        else:
            freshness_label = "stale"
            freshness_score = 0.35

    contradiction_flag = bool(avg_agreement <= 0.28 and domain_count >= 3)
    if bool(state.get("deadline_query", False)):
        contradiction_flag = contradiction_flag or (
            avg_agreement <= 0.45 and domain_count >= 2 and _has_deadline_date_conflict(results)
        )

    confidence = (
        (avg_quality * 0.45)
        + (avg_authority * 0.2)
        + (diversity_score * 0.15)
        + (freshness_score * 0.2)
    )
    if contradiction_flag:
        confidence -= 0.2
    if len(results) <= 1:
        confidence -= 0.08
    if _is_freshness_sensitive_query(
        str(state.get("safe_user_prompt", ""))
    ) and freshness_label in {
        "stale",
        "unknown",
    }:
        confidence -= 0.1
    web_required_coverage = _safe_float(state.get("web_required_field_coverage"))
    web_required_coverage = (
        _clamp01(web_required_coverage, fallback=1.0) if web_required_coverage is not None else None
    )
    if web_required_coverage is not None:
        # Penalize confidence when deep retrieval did not close required fields.
        confidence -= (1.0 - web_required_coverage) * 0.25
    admissions_critical_missing = _admissions_critical_web_fields_missing(state)
    if admissions_critical_missing:
        confidence -= min(0.35, 0.1 * len(admissions_critical_missing))
    if _is_admissions_requirements_query(state) and web_required_coverage is None:
        confidence -= 0.12
    confidence = _clamp01(confidence, fallback=0.5)

    state["trust_confidence"] = round(confidence, 4)
    state["trust_freshness"] = freshness_label
    state["trust_contradiction_flag"] = contradiction_flag
    state["trust_authority_score"] = round(avg_authority, 4)
    state["trust_agreement_score"] = round(avg_agreement, 4)
    state["trust_recency_score"] = round(avg_recency, 4)
    state["retrieval_avg_quality"] = round(avg_quality, 4)
    uncertainty_reasons: list[str] = []
    if contradiction_flag:
        uncertainty_reasons.append("Some sources conflict and require manual verification.")
    if confidence < _TRUST_LOW_CONFIDENCE_THRESHOLD:
        uncertainty_reasons.append("Overall evidence confidence is limited.")
    if _is_freshness_sensitive_query(
        str(state.get("safe_user_prompt", ""))
    ) and freshness_label in {
        "stale",
        "unknown",
    }:
        uncertainty_reasons.append("Freshness is not strong for a time-sensitive question.")
    if int(state.get("retrieval_source_count", 0) or 0) < 2:
        uncertainty_reasons.append("Independent source corroboration is limited.")
    if web_required_coverage is not None and web_required_coverage < 0.999:
        uncertainty_reasons.append("Some requested fields are not fully verified from web evidence.")
    if admissions_critical_missing:
        uncertainty_reasons.append(
            "Critical admissions fields are still missing from verified official evidence."
        )
    state["trust_uncertainty_reasons"] = uncertainty_reasons[:3]

    emit_trace_event(
        "evidence_trust_scored",
        {
            "confidence": state["trust_confidence"],
            "freshness": state["trust_freshness"],
            "contradiction_flag": state["trust_contradiction_flag"],
            "authority_score": state["trust_authority_score"],
            "agreement_score": state["trust_agreement_score"],
            "web_required_field_coverage": (
                round(web_required_coverage, 4) if web_required_coverage is not None else None
            ),
            "web_required_fields_missing": (state.get("web_required_fields_missing") or [])[:4],
            "uncertainty_reasons": state["trust_uncertainty_reasons"],
        },
    )


def _merge_retrieval_results(
    primary: list[dict], secondary: list[dict], *, limit: int
) -> list[dict]:
    """Merge retrieval candidates while deduping by stable identity/content keys."""
    merged: list[dict] = []
    seen: set[str] = set()
    for result in list(primary or []) + list(secondary or []):
        if not isinstance(result, dict):
            continue
        chunk_id = str(result.get("chunk_id", "")).strip()
        source_path = str(result.get("source_path", "")).strip()
        content = " ".join(str(result.get("content", "")).split()).lower()[:220]
        key = chunk_id or source_path or content
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= max(1, int(limit)):
            break
    return merged


def _web_expansion_similarity_threshold() -> float:
    configured = _safe_float(
        getattr(
            settings.web_search,
            "expansion_similarity_threshold",
            _WEB_EXPANSION_SIMILARITY_THRESHOLD,
        )
    )
    if configured is None:
        configured = _WEB_EXPANSION_SIMILARITY_THRESHOLD
    return max(0.0, min(1.0, float(configured)))


def _web_expansion_min_domain_count() -> int:
    configured = _safe_int(
        getattr(
            settings.web_search,
            "retrieval_min_unique_domains",
            _WEB_EXPANSION_MIN_DOMAIN_COUNT,
        )
    )
    if configured is None:
        configured = _WEB_EXPANSION_MIN_DOMAIN_COUNT
    return max(1, int(configured))


def _should_use_web_fallback(state: dict, top_similarity: float | None = None) -> bool:
    web_ready, _ = _web_retrieval_ready()
    if not web_ready:
        return False
    if not settings.web_search.fallback_enabled:
        return False
    if int(state.get("retrieved_count", 0) or 0) <= 0:
        return True
    if int(state.get("retrieval_source_count", 0) or 0) < _web_expansion_min_domain_count():
        return True
    if top_similarity is None:
        return False
    threshold = max(0.0, min(1.0, float(settings.web_search.fallback_similarity_threshold)))
    return top_similarity < threshold


def _web_retrieval_ready() -> tuple[bool, str]:
    if not settings.web_search.enabled:
        return False, "web_search_disabled"
    if not _web_search_key_present():
        return False, "web_search_api_key_missing"
    return True, "ready"


def _should_run_web_retrieval() -> bool:
    web_ready, _ = _web_retrieval_ready()
    if not web_ready:
        return False
    return bool(getattr(settings.web_search, "always_web_retrieval_enabled", True))


def _retrieval_fanout_enabled() -> bool:
    return bool(getattr(settings.web_search, "retrieval_fanout_enabled", True))


def _web_retrieval_timeout_seconds(search_mode: str | None = None, *, query: str | None = None) -> float:
    configured = _safe_float(getattr(settings.web_search, "timeout_seconds", 12.0))
    if configured is None:
        configured = 12.0
    normalized_mode = _normalized_request_mode(search_mode)
    if normalized_mode == _AUTO_MODE:
        normalized_mode = _DEEP_MODE
    if normalized_mode == _DEEP_MODE:
        explicit = _safe_float(getattr(settings.web_search, "deep_timeout_seconds", 0.0))
        if explicit is not None and explicit > 0:
            return max(30.0, min(300.0, explicit))
        multiplier = 8.0
        floor_seconds = 45.0
        ceiling_seconds = 180.0
    else:
        explicit = _safe_float(getattr(settings.web_search, "fast_timeout_seconds", 0.0))
        if explicit is not None and explicit > 0:
            return max(10.0, min(120.0, explicit))
        multiplier = 2.5
        floor_seconds = 10.0
        ceiling_seconds = 60.0
    timeout_seconds = configured * multiplier
    compact_query = " ".join(str(query or "").split()).strip()
    if normalized_mode == _DEEP_MODE and compact_query:
        if len(compact_query) >= 220:
            timeout_seconds *= 1.35
        if len(compact_query) >= 420:
            timeout_seconds *= 1.2
    return max(floor_seconds, min(ceiling_seconds, timeout_seconds))


async def _run_one_web_query_with_timeout(
    query: str,
    *,
    top_k: int,
    search_mode: str,
) -> dict | None:
    timeout_seconds = _web_retrieval_timeout_seconds(search_mode, query=query)
    try:
        return await asyncio.wait_for(
            _aretrieve_web_chunks_with_mode(
                query,
                top_k=top_k,
                search_mode=search_mode,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Web retrieval timed out; mode=%s timeout_s=%.2f query=%s",
            str(search_mode),
            timeout_seconds,
            query,
        )
        emit_trace_event(
            "web_query_timeout",
            {
                "mode": str(search_mode),
                "timeout_s": timeout_seconds,
                "query": query[:220],
            },
        )
        return {
            "results": [],
            "_timed_out": True,
            "_timeout_seconds": timeout_seconds,
            "_query": query,
            "_search_mode": str(search_mode),
        }
    except Exception as exc:
        logger.warning("Web retrieval query failed; query=%s error=%s", query, exc)
        return None


def _web_result_timed_out(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get("_timed_out", False))


def _compact_web_retrieval_query(*, base_query: str, state: dict) -> str:
    compact = " ".join(str(base_query or "").replace("\n", " ").split()).strip()
    if not compact:
        return ""
    # Keep search queries narrowly factual; strip metainstructions and user-facing directives.
    compact = re.split(
        r"\b(if any information is missing|if information is missing|search deeper and verify|"
        r"use official sources only|verify using official sources only|also tell me if)\b",
        compact,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .")
    compact = _TRAILING_QUERY_INSTRUCTION_RE.sub("", compact).strip(" .")
    compact = _LEADING_QUERY_FILLER_RE.sub("", compact).strip(" .")
    compact = re.sub(r"\s*[:;]\s*", " ", compact).strip()
    compact = re.sub(r"\s*-\s*", " ", compact).strip()
    compact = re.sub(r"\s{2,}", " ", compact).strip()
    # Use only the first sentence/question to avoid oversized generated search queries.
    first_sentence = re.split(r"(?<=[.?!])\s+", compact, maxsplit=1)[0].strip()
    if first_sentence:
        compact = first_sentence
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    field_hints: list[str] = []
    for field in required_fields:
        key = str(field).strip()
        if key == "application_deadline":
            field_hints.append("application deadline")
        elif key in {"language_requirements", "language_test_score_thresholds"}:
            field_hints.append("IELTS TOEFL language requirement")
        elif key in {"eligibility_requirements", "gpa_or_grade_threshold"}:
            field_hints.append("admission requirement GPA")
        elif key == "ects_or_prerequisite_credit_breakdown":
            field_hints.append("ECTS prerequisite credits")
        elif key == "application_portal":
            field_hints.append("application portal")
    if field_hints:
        suffix = " ".join(field_hints[:4]).strip()
        lowered = compact.lower()
        if suffix and suffix.lower() not in lowered:
            compact = f"{compact} {suffix}".strip()
    if len(compact) <= _WEB_QUERY_MAX_CHARS:
        return compact
    return compact[:_WEB_QUERY_MAX_CHARS].rsplit(" ", 1)[0].strip() or compact[:_WEB_QUERY_MAX_CHARS]


def _timeout_rescue_queries(*, base_query: str, state: dict) -> list[str]:
    compact_base = _compact_web_retrieval_query(base_query=base_query, state=state)
    candidates = _decompose_retrieval_queries(base_query=compact_base, state=state)
    if compact_base:
        candidates = [compact_base] + candidates
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = " ".join(str(candidate).split()).strip()
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(compact[:_WEB_QUERY_MAX_CHARS])
        if len(normalized) >= _WEB_TIMEOUT_RESCUE_MAX_QUERIES:
            break
    return normalized


def _result_dicts(rows) -> list[dict]:
    if not isinstance(rows, list):
        return []
    return [item for item in rows if isinstance(item, dict)]


def _set_retrieval_state(state: dict, results: list[dict]) -> None:
    state["retrieved_results"] = results
    state["retrieved_count"] = len(results)
    state["retrieval_source_count"] = _retrieval_source_count(results)
    state["retrieval_evidence"] = _build_retrieval_evidence(results)


async def _retrieve_vector_candidates(
    retrieval_query: str,
    state: dict,
    prefetched_result: dict | None = None,
    query_variants: list[str] | None = None,
) -> tuple[list[dict], float | None]:
    if not bool(getattr(settings.postgres, "enabled", False)):
        state["retrieval_strategy"] = "vector_disabled"
        state["retrieval_query_variants"] = [retrieval_query]
        _set_retrieval_state(state, [])
        emit_trace_event(
            "retrieval_vector_skipped",
            {
                "reason": "postgres_disabled",
                "query": retrieval_query[:220],
            },
        )
        return [], None

    variants = (
        query_variants if isinstance(query_variants, list) and query_variants else [retrieval_query]
    )
    normalized_variants: list[str] = []
    seen_queries: set[str] = set()
    for variant in variants:
        compact = " ".join(str(variant).split()).strip()
        if not compact:
            continue
        key = compact.lower()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        normalized_variants.append(compact[:_RETRIEVAL_QUERY_MAX_CHARS])
    if not normalized_variants:
        normalized_variants = [retrieval_query]

    emit_trace_event(
        "retrieval_vector_started",
        {
            "query": retrieval_query[:220],
            "query_variants": normalized_variants,
            "query_count": len(normalized_variants),
        },
    )

    async def _run_one_vector_query(query: str) -> dict:
        try:
            return await aretrieve_document_chunks(
                query,
                top_k=_default_retrieval_top_k(),
            )
        except Exception as exc:
            logger.warning("Vector retrieval query failed; continuing. %s", exc)
            return {"retrieval_strategy": "vector_error", "results": []}

    retrieval_results: list[dict] = []
    first_variant = normalized_variants[0]
    if (
        prefetched_result is not None
        and first_variant.strip().lower() == retrieval_query.strip().lower()
    ):
        retrieval_results.append(prefetched_result if isinstance(prefetched_result, dict) else {})
        remaining = normalized_variants[1:]
    else:
        remaining = normalized_variants

    if remaining:
        if len(remaining) == 1:
            retrieval_results.append(await _run_one_vector_query(remaining[0]))
        else:
            tasks = [asyncio.create_task(_run_one_vector_query(query)) for query in remaining]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for item in gathered:
                if isinstance(item, Exception):
                    logger.warning("Vector retrieval query variant failed; continuing. %s", item)
                    continue
                if isinstance(item, dict):
                    retrieval_results.append(item)

    if not retrieval_results:
        retrieval_results = [{"retrieval_strategy": "unknown", "results": []}]

    strategies = {
        str(result.get("retrieval_strategy", "unknown")).strip()
        for result in retrieval_results
        if isinstance(result, dict)
    }
    strategies.discard("")
    if len(strategies) > 1:
        state["retrieval_strategy"] = "vector_multi_query"
    else:
        state["retrieval_strategy"] = next(iter(strategies), "unknown")

    max_context_results = _max_context_results_for_mode(_mode_from_state(state))
    merge_limit = max(
        int(_default_retrieval_top_k()) * max(1, len(normalized_variants)),
        max_context_results,
        int(settings.bedrock.reranker_max_documents),
    )
    vector_results: list[dict] = []
    for retrieval_result in retrieval_results:
        rows = _result_dicts(retrieval_result.get("results", []))
        vector_results = _merge_retrieval_results(vector_results, rows, limit=merge_limit)

    top_similarity = _top_retrieval_similarity(vector_results)
    state["retrieval_top_similarity"] = top_similarity
    state["retrieval_query_variants"] = normalized_variants
    _set_retrieval_state(state, vector_results)
    emit_trace_event(
        "retrieval_vector_completed",
        {
            "result_count": len(vector_results),
            "top_similarity": top_similarity,
            "strategy": state["retrieval_strategy"],
            "query_count": len(normalized_variants),
        },
    )
    return vector_results, top_similarity


async def _retrieve_web_candidates_if_needed(
    retrieval_query: str,
    *,
    vector_results: list[dict],
    vector_has_urls: bool,
    top_similarity: float | None,
    search_mode: str,
    state: dict,
    web_prefetch_task: asyncio.Task | None = None,
) -> tuple[list[dict], bool]:
    web_ready, web_ready_reason = _web_retrieval_ready()
    deep_mode = str(search_mode).strip().lower() == _DEEP_MODE
    min_domain_count = _web_expansion_min_domain_count()
    expansion_similarity_threshold = _web_expansion_similarity_threshold()
    always_web_retrieval = web_ready and (_should_run_web_retrieval() or deep_mode)
    fallback_for_low_confidence = _should_use_web_fallback(state, top_similarity)
    fallback_for_missing_urls = (
        bool(vector_results)
        and not vector_has_urls
        and web_ready
        and settings.web_search.fallback_enabled
    )
    vector_domain_count = len(_allowed_citation_hosts(_evidence_urls(vector_results)))
    low_similarity = (
        top_similarity is not None and top_similarity < expansion_similarity_threshold
    )
    insufficient_domains = vector_domain_count < min_domain_count
    expansion_needed = web_ready and deep_mode and (low_similarity or insufficient_domains)
    state["web_fallback_attempted"] = False
    state["web_result_count"] = 0
    state["web_expansion_used"] = False
    state["web_retrieval_verified"] = None
    state["web_required_field_coverage"] = None
    state["web_required_fields_missing"] = []
    state["web_timeout_count"] = 0
    state["web_timed_out_queries"] = []
    state["web_timeout_rescued"] = False
    web_query = _compact_web_retrieval_query(base_query=retrieval_query, state=state) or retrieval_query

    if not (
        always_web_retrieval
        or fallback_for_low_confidence
        or fallback_for_missing_urls
        or expansion_needed
    ):
        if not web_ready:
            skip_reason = web_ready_reason
        elif not settings.web_search.fallback_enabled and not always_web_retrieval:
            skip_reason = "web_search_fallback_disabled"
        elif vector_results and top_similarity is not None:
            skip_reason = "vector_confident"
        elif vector_results and vector_has_urls:
            skip_reason = "vector_urls_available"
        else:
            skip_reason = "not_needed"
        emit_trace_event(
            "web_retrieval_skipped",
            {
                "reason": skip_reason,
                "query": web_query[:220],
                "planner_available": web_ready,
                "top_similarity": top_similarity,
                "vector_domain_count": vector_domain_count,
            },
        )
        if not web_ready:
            emit_trace_event(
                "query_planner_skipped",
                {
                    "reason": skip_reason,
                },
            )
        return [], False

    reason = "always_web_retrieval"
    if deep_mode and always_web_retrieval:
        reason = "deep_mode_always_web"
    elif not always_web_retrieval:
        reason = "low_confidence" if fallback_for_low_confidence else "missing_urls"
        if expansion_needed and not (fallback_for_low_confidence or fallback_for_missing_urls):
            reason = "query_expansion"
    state["web_fallback_attempted"] = True
    emit_trace_event(
        "web_fallback_started",
        {
            "reason": reason,
            "query": web_query[:220],
            "top_similarity": top_similarity,
            "vector_domain_count": vector_domain_count,
        },
    )
    if always_web_retrieval and vector_results:
        state["retrieval_strategy"] = "hybrid_vector_web"
    elif vector_results and fallback_for_low_confidence:
        state["retrieval_strategy"] = "vector_low_confidence"
    elif vector_results and fallback_for_missing_urls:
        state["retrieval_strategy"] = "vector_missing_urls"

    try:
        emit_trace_event(
            "query_planner_started",
            {
                "query": web_query[:220],
            },
        )
        if web_prefetch_task is not None:
            try:
                web_result = await asyncio.wait_for(
                    web_prefetch_task,
                    timeout=_web_retrieval_timeout_seconds(search_mode, query=web_query),
                )
            except asyncio.TimeoutError:
                if not web_prefetch_task.done():
                    web_prefetch_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await web_prefetch_task
                web_result = (
                    await _run_one_web_query_with_timeout(
                        web_query,
                        top_k=_default_retrieval_top_k(),
                        search_mode=search_mode,
                    )
                    or {}
                )
        else:
            web_result = (
                await _run_one_web_query_with_timeout(
                    web_query,
                    top_k=_default_retrieval_top_k(),
                    search_mode=search_mode,
                )
                or {}
            )
        web_result = web_result if isinstance(web_result, dict) else {}
        web_timeout_count = 1 if _web_result_timed_out(web_result) else 0
        timed_out_queries: list[str] = [web_query] if web_timeout_count else []
        verification = web_result.get("verification", {})
        verification = verification if isinstance(verification, dict) else {}
        web_required_coverage = _safe_float(verification.get("required_field_coverage"))
        if web_required_coverage is not None:
            state["web_required_field_coverage"] = round(
                _clamp01(web_required_coverage, fallback=1.0),
                4,
            )
        missing_required = verification.get("required_fields_missing", [])
        if isinstance(missing_required, list):
            state["web_required_fields_missing"] = [
                " ".join(str(item).split()).strip()
                for item in missing_required
                if " ".join(str(item).split()).strip()
            ][:6]
        verified = verification.get("verified")
        if isinstance(verified, bool):
            state["web_retrieval_verified"] = verified
        query_plan = web_result.get("query_plan", {})
        emit_trace_event(
            "query_planner_completed",
            {
                "planner": str(query_plan.get("planner", "heuristic")),
                "llm_used": bool(query_plan.get("llm_used", False)),
                "subquestions": query_plan.get("subquestions", []),
                "required_fields": query_plan.get("required_fields", []),
                "queries": web_result.get("query_variants", []),
            },
        )
        web_results = _result_dicts(web_result.get("results", []))
        max_web_results = _max_context_results_for_mode(search_mode)
        expansion_queries = _web_expansion_queries(
            base_query=web_query,
            state=state,
            low_similarity=low_similarity,
            insufficient_domains=insufficient_domains,
            current_results=web_results,
        )
        existing_queries = {
            str(item).strip().lower() for item in expansion_queries if str(item).strip()
        }
        current_domains = _allowed_citation_hosts(_evidence_urls(web_results))
        recovery_needed = deep_mode and (
            not web_results or len(current_domains) < min_domain_count
        )
        if recovery_needed:
            recovery_queries = _web_expansion_queries(
                base_query=web_query,
                state=state,
                low_similarity=True,
                insufficient_domains=True,
                current_results=web_results,
            )
            recovery_added = 0
            for query in recovery_queries:
                compact = " ".join(str(query).split()).strip()
                key = compact.lower()
                if not compact or key in existing_queries:
                    continue
                existing_queries.add(key)
                expansion_queries.append(compact)
                recovery_added += 1
                if recovery_added >= _WEB_RECOVERY_MAX_QUERIES:
                    break
        extra_results: list[dict] = []
        if expansion_queries:
            emit_trace_event(
                "web_query_expansion_started",
                {
                    "reasons": [
                        reason
                        for reason, enabled in (
                            ("low_similarity", low_similarity),
                            ("insufficient_domains", insufficient_domains),
                        )
                        if enabled
                    ],
                    "query_variants": expansion_queries,
                    "recovery_mode": recovery_needed,
                },
            )
            expansion_tasks = [
                asyncio.create_task(
                    _run_one_web_query_with_timeout(
                        expansion_query,
                        top_k=_default_retrieval_top_k(),
                        # Keep high-precision admissions expansion in deep mode for better exact-field recall.
                        search_mode=(
                            search_mode
                            if (deep_mode and _is_admissions_requirements_query(state))
                            else (_FAST_MODE if deep_mode else search_mode)
                        ),
                    )
                )
                for expansion_query in expansion_queries
            ]
            gathered_expansions = await asyncio.gather(*expansion_tasks, return_exceptions=True)
            for expansion_result in gathered_expansions:
                if isinstance(expansion_result, Exception) or not isinstance(
                    expansion_result, dict
                ):
                    continue
                if _web_result_timed_out(expansion_result):
                    web_timeout_count += 1
                    timeout_query = " ".join(str(expansion_result.get("_query", "")).split()).strip()
                    if timeout_query:
                        timed_out_queries.append(timeout_query)
                    continue
                rows = _result_dicts(expansion_result.get("results", []))
                extra_results = _merge_retrieval_results(
                    extra_results,
                    rows,
                    limit=max_web_results,
                )
            if extra_results:
                state["web_expansion_used"] = True
                web_results = _merge_retrieval_results(
                    web_results,
                    extra_results,
                    limit=max_web_results,
                )
            emit_trace_event(
                "web_query_expansion_completed",
                {
                    "query_count": len(expansion_queries),
                    "extra_result_count": len(extra_results),
                    "total_result_count": len(web_results),
                },
            )

        if deep_mode and not web_results and web_timeout_count > 0:
            rescue_queries = _timeout_rescue_queries(base_query=web_query, state=state)
            if rescue_queries:
                emit_trace_event(
                    "web_timeout_rescue_started",
                    {
                        "query_variants": rescue_queries,
                        "timeout_count": web_timeout_count,
                    },
                )
                rescue_results: list[dict] = []
                rescue_tasks = [
                    asyncio.create_task(
                        _run_one_web_query_with_timeout(
                            rescue_query,
                            top_k=_default_retrieval_top_k(),
                            search_mode=_FAST_MODE,
                        )
                    )
                    for rescue_query in rescue_queries
                ]
                gathered_rescue = await asyncio.gather(*rescue_tasks, return_exceptions=True)
                for rescue_result in gathered_rescue:
                    if isinstance(rescue_result, Exception) or not isinstance(rescue_result, dict):
                        continue
                    if _web_result_timed_out(rescue_result):
                        web_timeout_count += 1
                        timeout_query = " ".join(str(rescue_result.get("_query", "")).split()).strip()
                        if timeout_query:
                            timed_out_queries.append(timeout_query)
                        continue
                    rows = _result_dicts(rescue_result.get("results", []))
                    rescue_results = _merge_retrieval_results(
                        rescue_results,
                        rows,
                        limit=max_web_results,
                    )
                if rescue_results:
                    state["web_timeout_rescued"] = True
                    state["web_expansion_used"] = True
                    web_results = _merge_retrieval_results(
                        web_results,
                        rescue_results,
                        limit=max_web_results,
                    )
                emit_trace_event(
                    "web_timeout_rescue_completed",
                    {
                        "query_count": len(rescue_queries),
                        "rescued_result_count": len(web_results),
                        "timeout_count": web_timeout_count,
                    },
                )

        if web_timeout_count > 0:
            deduped_timeout_queries: list[str] = []
            seen_timeout_queries: set[str] = set()
            for timeout_query in timed_out_queries:
                compact = " ".join(str(timeout_query).split()).strip()
                if not compact:
                    continue
                key = compact.lower()
                if key in seen_timeout_queries:
                    continue
                seen_timeout_queries.add(key)
                deduped_timeout_queries.append(compact)
            state["web_timeout_count"] = int(state.get("web_timeout_count", 0) or 0) + web_timeout_count
            state["web_timed_out_queries"] = deduped_timeout_queries[:6]

        state["web_result_count"] = len(web_results)
        web_urls = _traceable_urls(_evidence_urls(web_results))
        emit_trace_event(
            "web_fallback_completed",
            {
                "result_count": len(web_results),
                "query_variants": web_result.get("query_variants", []),
                "facts": web_result.get("facts", []),
                "source_urls": web_urls,
                "retrieval_loop": web_result.get("retrieval_loop", {}),
                "query_plan": web_result.get("query_plan", {}),
                "verification": verification,
                "timeout_count": web_timeout_count,
                "timed_out_queries": (state.get("web_timed_out_queries") or [])[:4],
                "timeout_rescued": bool(state.get("web_timeout_rescued", False)),
            },
        )
        if web_results:
            state["retrieval_strategy"] = (
                "web_hybrid" if always_web_retrieval and vector_results else "web_fallback"
            )
        elif vector_results:
            state["retrieval_strategy"] = "vector_only_after_web"
        elif web_timeout_count > 0:
            state["retrieval_strategy"] = "web_timeout_no_results"
        return web_results, True
    except Exception as web_exc:
        state["retrieval_strategy"] = "web_fallback_error"
        state["web_fallback_attempted"] = True
        state["web_result_count"] = 0
        emit_trace_event(
            "web_fallback_error",
            {
                "error": "web_fallback_failed",
                "error_type": type(web_exc).__name__,
                "error_message": " ".join(str(web_exc).split())[:260],
            },
        )
        logger.warning(
            "Web fallback retrieval failed; continuing without web context. %s",
            web_exc,
        )
        return [], True


async def _aretrieve_web_chunks_with_mode(
    retrieval_query: str,
    *,
    top_k: int,
    search_mode: str,
) -> dict:
    try:
        return await aretrieve_web_chunks(
            retrieval_query,
            top_k=top_k,
            search_mode=search_mode,
        )
    except TypeError as exc:
        # Backward compatibility for tests or call-sites patching an older function signature.
        if "search_mode" not in str(exc):
            raise
        return await aretrieve_web_chunks(retrieval_query, top_k=top_k)


def _merge_vector_and_web_results(
    vector_results: list[dict],
    web_results: list[dict],
    *,
    search_mode: str | None = None,
) -> list[dict]:
    if not web_results:
        return vector_results

    merge_limit = max(
        int(_default_retrieval_top_k()),
        _max_context_results_for_mode(search_mode),
        int(settings.bedrock.reranker_max_documents),
    )
    return _merge_retrieval_results(
        vector_results,
        web_results,
        limit=merge_limit,
    )


async def _rerank_if_configured(
    retrieval_query: str,
    merged_results: list[dict],
    state: dict,
) -> list[dict]:
    if not merged_results:
        return merged_results

    original_results = list(merged_results)
    min_unique_domains = max(1, int(getattr(settings.web_search, "retrieval_min_unique_domains", 1)))
    merge_limit = max(
        int(_default_retrieval_top_k()),
        _max_context_results_for_mode(_mode_from_state(state)),
        int(settings.bedrock.reranker_max_documents),
    )

    def _result_identity_key(result: dict) -> str:
        if not isinstance(result, dict):
            return ""
        chunk_id = str(result.get("chunk_id", "")).strip()
        source_path = str(result.get("source_path", "")).strip()
        content = " ".join(str(result.get("content", "")).split()).lower()[:220]
        return chunk_id or source_path or content

    def _result_domain(result: dict) -> str:
        if not isinstance(result, dict):
            return ""
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        candidate = _normalized_url(str(metadata.get("url", ""))) or _normalized_url(
            str(result.get("source_path", ""))
        )
        return _normalized_host_from_url(candidate)

    try:
        rerank_result = await arerank_retrieval_results(retrieval_query, merged_results)
        reranked_rows = _result_dicts(rerank_result.get("results", []))
        if reranked_rows:
            merged_results = reranked_rows
            if min_unique_domains > 1:
                domains = {_result_domain(item) for item in merged_results if _result_domain(item)}
                if len(domains) < min_unique_domains:
                    seen_keys = {
                        _result_identity_key(item)
                        for item in merged_results
                        if _result_identity_key(item)
                    }
                    for candidate in original_results:
                        key = _result_identity_key(candidate)
                        if not key or key in seen_keys:
                            continue
                        candidate_domain = _result_domain(candidate)
                        if not candidate_domain or candidate_domain in domains:
                            continue
                        merged_results.append(candidate)
                        seen_keys.add(key)
                        domains.add(candidate_domain)
                        if len(merged_results) >= merge_limit or len(domains) >= min_unique_domains:
                            break
                if len(merged_results) > merge_limit:
                    merged_results = merged_results[:merge_limit]
                emit_trace_event(
                    "retrieval_rerank_diversity",
                    {
                        "required_domains": min_unique_domains,
                        "result_count": len(merged_results),
                        "domain_count": len(
                            {
                                _result_domain(item)
                                for item in merged_results
                                if _result_domain(item)
                            }
                        ),
                    },
                )
            comparison_entities = state.get("comparison_entities")
            comparison_entities = (
                comparison_entities
                if isinstance(comparison_entities, list)
                else _comparison_entities_from_prompt(str(state.get("safe_user_prompt", "")))
            )
            comparison_entities = [str(item).strip() for item in comparison_entities if str(item).strip()][
                :2
            ]
            if len(comparison_entities) >= 2:
                covered_entities = {
                    entity
                    for entity in comparison_entities
                    if any(_result_mentions_entity(item, entity) for item in merged_results)
                }
                missing_entities = [
                    entity for entity in comparison_entities if entity not in covered_entities
                ]
                if missing_entities:
                    selected_keys = {
                        _result_identity_key(item)
                        for item in merged_results
                        if _result_identity_key(item)
                    }
                    injected: list[dict] = []
                    injected_entities: list[str] = []
                    for entity in missing_entities:
                        for candidate in original_results:
                            key = _result_identity_key(candidate)
                            if not key or key in selected_keys:
                                continue
                            if not _result_mentions_entity(candidate, entity):
                                continue
                            injected.append(candidate)
                            selected_keys.add(key)
                            injected_entities.append(entity)
                            break
                    if injected:
                        merged_results = injected + merged_results
                        if len(merged_results) > merge_limit:
                            merged_results = merged_results[:merge_limit]
                        emit_trace_event(
                            "retrieval_rerank_entity_coverage",
                            {
                                "required_entities": comparison_entities,
                                "injected_entities": injected_entities,
                                "result_count": len(merged_results),
                            },
                        )
        state["retrieval_reranker_applied"] = bool(rerank_result.get("applied", False))
        timings = rerank_result.get("timings_ms", {})
        if isinstance(timings, dict):
            state["retrieval_reranker_ms"] = _safe_int(timings.get("total"))
        if state["retrieval_reranker_applied"]:
            strategy = str(state.get("retrieval_strategy", "")).strip() or "retrieval"
            state["retrieval_strategy"] = f"{strategy}_reranked"
        emit_trace_event(
            "retrieval_reranked",
            {
                "applied": state["retrieval_reranker_applied"],
                "result_count": len(merged_results),
                "reranker_ms": state.get("retrieval_reranker_ms"),
            },
        )
    except Exception as rerank_exc:
        emit_trace_event(
            "retrieval_rerank_error",
            {
                "error": "retrieval_rerank_failed",
                "error_type": type(rerank_exc).__name__,
            },
        )
        logger.warning(
            "Reranker failed; using non-reranked retrieval order. %s",
            rerank_exc,
        )
    return merged_results


def _apply_grounded_retrieval_context(
    *,
    messages: list,
    merged_results: list[dict],
    used_web_results: bool,
    state: dict,
) -> tuple[list | None, str | None]:
    if not merged_results:
        return messages, None

    _set_retrieval_state(state, merged_results)
    _derive_evidence_trust_signals(merged_results, state)
    evidence_urls = _evidence_urls(merged_results)
    evidence_hosts = _allowed_citation_hosts(evidence_urls)
    min_unique_domains = max(1, int(getattr(settings.web_search, "retrieval_min_unique_domains", 1)))
    state["evidence_urls"] = evidence_urls
    state["evidence_domain_count"] = len(evidence_hosts)
    state["citation_min_hosts"] = max(1, min(len(evidence_hosts), min_unique_domains))
    state["citation_required"] = True
    if not evidence_urls:
        if int(state.get("web_timeout_count", 0) or 0) > 0:
            state["context_guard_reason"] = "web_retrieval_timeout"
            return None, _WEB_RETRIEVAL_TIMEOUT_DETAIL
        if str(state.get("query_intent", "")).strip().lower() == "comparison" and merged_results:
            state["citation_required"] = False
            state["allow_uncited_comparison_fallback"] = True
            context_message = _format_retrieval_context({"results": merged_results})
            if context_message is not None:
                messages = messages + [context_message]
            emit_trace_event(
                "citation_grounding_relaxed",
                {
                    "reason": "comparison_no_urls",
                    "result_count": len(merged_results),
                },
            )
            return messages, None
        state["context_guard_reason"] = "weak_evidence_no_urls"
        return None, _NO_RELEVANT_INFORMATION_DETAIL

    emit_trace_event(
        "evidence_selected",
        {
            "result_count": len(merged_results),
            "source_count": len(evidence_urls),
            "source_domain_count": len(evidence_hosts),
            "source_domains": list(sorted(evidence_hosts))[:8],
            "citation_min_hosts": int(state.get("citation_min_hosts", 1) or 1),
            "source_urls": _traceable_urls(evidence_urls),
        },
    )
    emit_trace_event(
        "citation_grounding_ready",
        {
            "required": True,
            "citation_min_hosts": int(state.get("citation_min_hosts", 1) or 1),
            "allowed_source_count": len(evidence_urls),
        },
    )

    context_message = (
        _format_web_retrieval_context({"results": merged_results})
        if used_web_results
        else _format_retrieval_context({"results": merged_results})
    )
    if not context_message:
        state["context_guard_reason"] = "weak_evidence_empty_context"
        return None, _NO_RELEVANT_INFORMATION_DETAIL

    return [
        _citation_grounding_message(evidence_urls),
        context_message,
    ] + messages, None


async def _augment_messages_with_retrieval(
    *,
    messages: list,
    retrieval_query: str,
    search_mode: str,
    state: dict,
    vector_prefetch_result: dict | None = None,
) -> tuple[list | None, str | None]:
    retrieval_started_at = time.perf_counter()
    web_prefetch_task: asyncio.Task | None = None
    web_query = _compact_web_retrieval_query(base_query=retrieval_query, state=state) or retrieval_query
    query_variants = _decompose_retrieval_queries(base_query=web_query, state=state)
    emit_trace_event(
        "retrieval_query_decomposed",
        {
            "intent": str(state.get("query_intent", "unknown")),
            "query": retrieval_query[:220],
            "query_variants": query_variants,
        },
    )
    try:
        web_ready, _ = _web_retrieval_ready()
        deep_mode = str(search_mode).strip().lower() == _DEEP_MODE
        if _retrieval_fanout_enabled() and web_ready and (_should_run_web_retrieval() or deep_mode):
            # Fan-out: prefetch web retrieval while vector retrieval is in-flight.
            web_prefetch_task = asyncio.create_task(
                _run_one_web_query_with_timeout(
                    web_query,
                    top_k=_default_retrieval_top_k(),
                    search_mode=search_mode,
                )
            )
        vector_results, top_similarity = await _retrieve_vector_candidates(
            retrieval_query,
            state,
            prefetched_result=vector_prefetch_result,
            query_variants=query_variants,
        )
        vector_has_urls = bool(_evidence_urls(vector_results))
        web_results, web_fallback_attempted = await _retrieve_web_candidates_if_needed(
            retrieval_query,
            vector_results=vector_results,
            vector_has_urls=vector_has_urls,
            top_similarity=top_similarity,
            search_mode=search_mode,
            state=state,
            web_prefetch_task=web_prefetch_task,
        )
        if web_fallback_attempted and not web_results:
            web_timeout_count = int(state.get("web_timeout_count", 0) or 0)
            freshness_sensitive = _is_freshness_sensitive_query(
                str(state.get("safe_user_prompt", ""))
            )
            if web_timeout_count > 0 and (not vector_results or not vector_has_urls):
                state["context_guard_reason"] = "web_retrieval_timeout"
                state["retrieval_strategy"] = "web_timeout_no_results"
                return None, _WEB_RETRIEVAL_TIMEOUT_DETAIL
            if not vector_results:
                if _is_citation_grounding_required() or freshness_sensitive:
                    state["context_guard_reason"] = "no_relevant_information"
                    return None, _NO_RELEVANT_INFORMATION_DETAIL
                state["retrieval_strategy"] = "web_fallback_empty_no_vector"
                logger.info("Web fallback returned no results and vector was empty; continuing.")
            if vector_results and not vector_has_urls:
                state["context_guard_reason"] = "weak_evidence_no_urls"
                return None, _NO_RELEVANT_INFORMATION_DETAIL
            state["retrieval_strategy"] = "vector_fallback_web_empty"
            logger.info("Web fallback returned no results; using available vector evidence.")

        merged_results = _merge_vector_and_web_results(
            vector_results,
            web_results,
            search_mode=search_mode,
        )
        merged_results = await _rerank_if_configured(retrieval_query, merged_results, state)
        merged_results = _selective_retrieval_results(merged_results, state)
        return _apply_grounded_retrieval_context(
            messages=messages,
            merged_results=merged_results,
            used_web_results=bool(web_results),
            state=state,
        )
    except Exception as exc:
        state["retrieval_strategy"] = "error"
        logger.warning(
            "Long-term retrieval failed; continuing without retrieved context. %s",
            exc,
        )
        return messages, None
    finally:
        if web_prefetch_task is not None:
            if not web_prefetch_task.done():
                web_prefetch_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await web_prefetch_task
        state["retrieval_ms"] = _elapsed_ms(retrieval_started_at)


def _validate_citation_grounding_state(state: dict) -> str | None:
    if not _is_citation_grounding_required():
        return None
    if int(state.get("web_timeout_count", 0) or 0) > 0 and not state.get("evidence_urls"):
        state["context_guard_reason"] = "web_retrieval_timeout"
        return _WEB_RETRIEVAL_TIMEOUT_DETAIL
    if not bool(state.get("citation_required", False)):
        state["context_guard_reason"] = "weak_evidence_missing"
        return _NO_RELEVANT_INFORMATION_DETAIL
    evidence_urls = state.get("evidence_urls", [])
    if not isinstance(evidence_urls, list) or not evidence_urls:
        state["context_guard_reason"] = "weak_evidence_no_urls"
        return _NO_RELEVANT_INFORMATION_DETAIL
    return None


def _web_context_line(result: dict, index: int) -> str | None:
    content = str(result.get("content", "")).strip()
    if not content:
        return None

    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    title = str(metadata.get("university", "")).strip() or f"Web Result {index}"
    url = str(metadata.get("url", "")).strip()
    compact = " ".join(content.split())[:_RETRIEVAL_CHUNK_MAX_CHARS]
    if url:
        return f"{index}. {title} ({url}): {compact}"
    return f"{index}. {title}: {compact}"


def _web_context_result_lines(results: list[dict], *, max_items: int) -> list[str]:
    lines: list[str] = []
    used = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        line = _web_context_line(result, used + 1)
        if not line:
            continue
        lines.append(line)
        used += 1
        if used >= max_items:
            break
    return lines


def _format_web_retrieval_context(web_result: dict) -> dict | None:
    """Convert web fallback results into one system context message with URLs."""
    results = web_result.get("results", []) if isinstance(web_result, dict) else []
    if not isinstance(results, list) or not results:
        return None

    header = [
        "Live web fallback context (Tavily web search). Use only if relevant and cite URLs.",
    ]
    result_lines = _web_context_result_lines(
        results,
        max_items=_prompt_retrieval_result_limit(),
    )
    if not result_lines:
        return None
    joined = "\n".join(header + result_lines)
    return {"role": "system", "content": joined[:_RETRIEVAL_CONTEXT_MAX_CHARS]}


def _build_retrieval_evidence(results: list[dict]) -> list[dict]:
    """Build compact retrieval evidence for grounded hallucination evaluation."""
    if not isinstance(results, list):
        return []

    evidence: list[dict] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        content = str(result.get("content", "")).strip()
        if not content:
            continue
        metadata = result.get("metadata")
        evidence.append(
            {
                "chunk_id": str(result.get("chunk_id", "")),
                "source_path": str(result.get("source_path", "")),
                "distance": result.get("distance"),
                "metadata": metadata if isinstance(metadata, dict) else {},
                "content": " ".join(content.split())[:_RETRIEVAL_EVIDENCE_CONTENT_MAX_CHARS],
            }
        )
        if len(evidence) >= _RETRIEVAL_EVIDENCE_MAX_ITEMS:
            break
    return evidence


def _normalized_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value
    return ""


def _normalized_host_from_url(url: str) -> str:
    host = str(urlparse(url).hostname or "").strip().lower()
    if host.startswith("www."):
        return host[4:]
    return host


def _allowed_citation_hosts(evidence_urls: list[str]) -> set[str]:
    hosts: set[str] = set()
    for url in evidence_urls:
        normalized = _normalized_url(str(url))
        if not normalized:
            continue
        host = _normalized_host_from_url(normalized)
        if host:
            hosts.add(host)
    return hosts


def _evidence_urls(results: list[dict]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        candidates = [
            metadata.get("url", ""),
            result.get("source_path", ""),
        ]
        for candidate in candidates:
            normalized = _normalized_url(str(candidate))
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            urls.append(normalized)
    return urls


def _retrieval_source_count(results: list[dict]) -> int:
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        url = _normalized_url(str(metadata.get("url", "")))
        source_path = str(result.get("source_path", "")).strip()
        key = (url or source_path).strip().lower()
        if not key:
            continue
        seen.add(key)
    return len(seen)


def _citation_grounding_message(evidence_urls: list[str]) -> dict:
    lines = [
        "Citation policy:",
        "- Answer only using provided evidence.",
        "- Cite URLs explicitly in your answer for every factual claim.",
        (
            "- Attach the URL citation inline on each factual sentence or bullet, "
            "not only in a final list."
        ),
        (
            "- If evidence is partial, provide a partial answer with explicit 'Missing info' "
            "items and uncertainty notes."
        ),
        (
            f"- Use exactly '{_NO_RELEVANT_INFORMATION_DETAIL}' only when there is no relevant "
            "evidence at all."
        ),
        "- Keep answer readable: no internal scaffolding labels; use user-facing headings only.",
        "- Include exactly one final Sources section with unique URLs (no duplicates).",
        "Allowed evidence URLs:",
    ]
    for index, url in enumerate(evidence_urls, start=1):
        lines.append(f"{index}. {url}")
        if index >= 12:
            break
    return {"role": "system", "content": "\n".join(lines)[:_RETRIEVAL_CONTEXT_MAX_CHARS]}


def _response_has_allowed_citation(text: str, evidence_urls: list[str]) -> bool:
    if not text or not evidence_urls:
        return False
    allowed_hosts = _allowed_citation_hosts(evidence_urls)
    if not allowed_hosts:
        return False
    return bool(_response_cited_allowed_hosts(text, evidence_urls))


def _response_cited_allowed_hosts(text: str, evidence_urls: list[str]) -> set[str]:
    if not text or not evidence_urls:
        return set()
    allowed_hosts = _allowed_citation_hosts(evidence_urls)
    if not allowed_hosts:
        return set()
    cited_urls = _CITATION_URL_RE.findall(str(text))
    cited_hosts: set[str] = set()
    for cited_url in cited_urls:
        host = _normalized_host_from_url(cited_url)
        if host and host in allowed_hosts:
            cited_hosts.add(host)
    return cited_hosts


def _claim_level_citation_stats(text: str, evidence_urls: list[str]) -> dict[str, float | int]:
    if not text:
        return {"claim_count": 0, "cited_claim_count": 0, "coverage": 1.0}
    lines: list[str] = []
    for raw_line in str(text).splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("sources:"):
            continue
        # Skip pure headings and tiny fragments.
        if line.endswith(":") and "http" not in lowered:
            continue
        normalized = line.lstrip("-*0123456789. ").strip()
        if len(normalized) < _CLAIM_LINE_MIN_CHARS:
            continue
        lines.append(line)

    if not lines:
        return {"claim_count": 0, "cited_claim_count": 0, "coverage": 1.0}

    allowed_hosts = _allowed_citation_hosts(evidence_urls)
    cited_claim_count = 0
    for line in lines:
        cited_urls = _CITATION_URL_RE.findall(line)
        if not cited_urls:
            continue
        if not allowed_hosts:
            cited_claim_count += 1
            continue
        for cited_url in cited_urls:
            host = _normalized_host_from_url(cited_url)
            if host and host in allowed_hosts:
                cited_claim_count += 1
                break
    claim_count = len(lines)
    coverage = float(cited_claim_count) / float(claim_count) if claim_count else 1.0
    return {
        "claim_count": claim_count,
        "cited_claim_count": cited_claim_count,
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
    }


def _claim_lines_for_grounding(text: str, *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in str(text).splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("sources:") or lowered.startswith("evidence and caveats"):
            continue
        if line.endswith(":") and "http" not in lowered:
            continue
        normalized = line.lstrip("-*0123456789. ").strip()
        if len(normalized) < _CLAIM_LINE_MIN_CHARS:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(normalized[:260])
        if len(lines) >= limit:
            break
    return lines


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]{4,}", str(text or ""))}


def _evidence_snippet_candidates(results: list[dict], *, limit: int = 90) -> list[dict]:
    snippets: list[dict] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        raw_url = _normalized_url(str(metadata.get("url", ""))) or _normalized_url(
            str(result.get("source_path", ""))
        )
        if not raw_url:
            continue
        content = " ".join(str(result.get("content", "")).split())
        if not content:
            continue
        sentence_candidates = _SENTENCE_SPLIT_RE.split(content)
        for sentence in sentence_candidates[:5]:
            snippet = " ".join(sentence.split()).strip()
            if len(snippet) < 48:
                continue
            snippets.append(
                {
                    "url": raw_url,
                    "snippet": snippet[:220],
                    "tokens": _token_set(snippet),
                    "dates": _DATE_LIKE_RE.findall(snippet),
                }
            )
            if len(snippets) >= limit:
                return snippets
    return snippets


def _map_claims_to_evidence_snippets(answer: str, state: dict) -> dict:
    retrieved_results = state.get("retrieved_results", [])
    retrieved_results = retrieved_results if isinstance(retrieved_results, list) else []
    claims = _claim_lines_for_grounding(answer, limit=8)
    if not claims:
        return {"claim_count": 0, "grounded_claim_count": 0, "coverage": 1.0, "mappings": []}
    snippets = _evidence_snippet_candidates(retrieved_results, limit=90)
    if not snippets:
        return {
            "claim_count": len(claims),
            "grounded_claim_count": 0,
            "coverage": 0.0,
            "mappings": [{"claim": claim, "snippets": []} for claim in claims],
        }

    mappings: list[dict] = []
    grounded_count = 0
    conflict_count = 0
    for claim in claims:
        claim_tokens = _token_set(claim)
        scored: list[tuple[int, dict]] = []
        for snippet in snippets:
            overlap = len(claim_tokens.intersection(snippet.get("tokens", set())))
            if overlap < 2:
                continue
            scored.append((overlap, snippet))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen_urls: set[str] = set()
        for overlap, snippet in scored:
            url = str(snippet.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            selected.append(
                {
                    "url": url,
                    "snippet": str(snippet.get("snippet", "")).strip(),
                    "overlap": overlap,
                    "dates": list(snippet.get("dates", []))[:2],
                }
            )
            seen_urls.add(url)
            if len(selected) >= 2:
                break
        if selected:
            grounded_count += 1
            date_sets: list[set[str]] = []
            for item in selected:
                item_dates = {
                    " ".join(str(date).split()).lower()
                    for date in item.get("dates", [])
                    if str(date).strip()
                }
                if item_dates:
                    date_sets.append(item_dates)
            if _date_sets_conflict(date_sets):
                conflict_count += 1
        mappings.append({"claim": claim, "snippets": selected})

    claim_count = len(claims)
    coverage = float(grounded_count) / float(claim_count) if claim_count else 1.0
    return {
        "claim_count": claim_count,
        "grounded_claim_count": grounded_count,
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
        "conflict_count": conflict_count,
        "mappings": mappings[:8],
    }


def _has_contradiction_signal(answer: str, state: dict, evidence_map: dict | None = None) -> bool:
    if bool(state.get("trust_contradiction_flag", False)):
        return True
    if isinstance(evidence_map, dict):
        conflict_count = int(evidence_map.get("conflict_count", 0) or 0)
        trust_agreement = _safe_float(state.get("trust_agreement_score"))
        if conflict_count >= 2:
            return True
        if conflict_count >= 1 and trust_agreement is not None and trust_agreement <= 0.4:
            return True
    if bool(state.get("deadline_query", False)):
        answer_dates = {
            " ".join(str(match).split()).lower()
            for match in _DATE_LIKE_RE.findall(str(answer or ""))
            if str(match).strip()
        }
        evidence_dates = {
            " ".join(str(match).split()).lower()
            for match in _evidence_date_values(state.get("retrieved_results", []), limit=10)
            if str(match).strip()
        }
        if answer_dates and evidence_dates and answer_dates.isdisjoint(evidence_dates):
            return True
    return False


def _enforce_citation_grounding(result: str, state: dict) -> str:
    citation_required = bool(state.get("citation_required", False))
    if not citation_required and not _is_citation_grounding_required():
        return result
    if not citation_required:
        if bool(state.get("allow_uncited_comparison_fallback", False)) and str(
            state.get("query_intent", "")
        ).strip().lower() == "comparison":
            return result
        state["output_guard_reason"] = "weak_evidence_missing"
        return _NO_RELEVANT_INFORMATION_DETAIL
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    if not evidence_urls:
        state["output_guard_reason"] = "weak_evidence_no_urls"
        return _NO_RELEVANT_INFORMATION_DETAIL
    if _response_has_allowed_citation(result, evidence_urls):
        return result
    if _has_authoritative_evidence(state):
        state["output_guard_reason"] = "missing_citations_partial"
        return result
    state["output_guard_reason"] = "missing_citations"
    return _NO_RELEVANT_INFORMATION_DETAIL


def _merge_llm_usage(total: dict, current: dict) -> dict:
    if not total:
        return dict(current or {})
    merged = dict(total)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = int(merged.get(key, 0) or 0) + int(current.get(key, 0) or 0)
    return merged


def _required_citation_host_count(state: dict, evidence_urls: list[str]) -> int:
    configured = _safe_int(state.get("citation_min_hosts"))
    if configured is not None and configured > 0:
        return configured
    available_hosts = len(_allowed_citation_hosts(evidence_urls))
    return max(1, available_hosts)


def _agentic_result_issues(result: str, state: dict) -> list[str]:
    issues: list[str] = []
    citation_required = bool(state.get("citation_required", False))
    citation_policy_enabled = _is_citation_grounding_required()
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    admissions_critical_missing = _admissions_critical_web_fields_missing(state)

    if (
        citation_policy_enabled
        and citation_required
        and evidence_urls
        and result != _NO_RELEVANT_INFORMATION_DETAIL
        and not _response_has_allowed_citation(result, evidence_urls)
    ):
        issues.append("missing_allowed_citations")
    if (
        citation_policy_enabled
        and citation_required
        and evidence_urls
        and result != _NO_RELEVANT_INFORMATION_DETAIL
    ):
        required_hosts = _required_citation_host_count(state, evidence_urls)
        cited_hosts = _response_cited_allowed_hosts(result, evidence_urls)
        if len(cited_hosts) < required_hosts:
            issues.append("insufficient_source_diversity")
        claim_stats = _claim_level_citation_stats(result, evidence_urls)
        state["claim_citation_coverage"] = claim_stats.get("coverage")
        state["claim_count"] = int(claim_stats.get("claim_count", 0) or 0)
        state["claim_cited_count"] = int(claim_stats.get("cited_claim_count", 0) or 0)
        if (
            int(claim_stats.get("claim_count", 0) or 0) >= 2
            and float(claim_stats.get("coverage", 1.0) or 1.0) < _CLAIM_CITATION_MIN_COVERAGE
        ):
            issues.append("weak_claim_citation_linkage")
        evidence_map = _map_claims_to_evidence_snippets(result, state)
        state["claim_snippet_grounding_coverage"] = evidence_map.get("coverage")
        state["claim_snippet_conflict_count"] = int(evidence_map.get("conflict_count", 0) or 0)
        state["claim_evidence_map"] = evidence_map.get("mappings", [])
        emit_trace_event(
            "claim_grounding_evaluated",
            {
                "claim_count": int(evidence_map.get("claim_count", 0) or 0),
                "grounded_claim_count": int(evidence_map.get("grounded_claim_count", 0) or 0),
                "coverage": evidence_map.get("coverage"),
                "conflict_count": int(evidence_map.get("conflict_count", 0) or 0),
            },
        )
        if (
            int(evidence_map.get("claim_count", 0) or 0) >= 2
            and float(evidence_map.get("coverage", 1.0) or 1.0)
            < _CLAIM_SNIPPET_MIN_GROUNDING_COVERAGE
        ):
            issues.append("missing_evidence_snippets")
        if _has_contradiction_signal(result, state, evidence_map):
            issues.append("contradiction_detected")
    if result != _NO_RELEVANT_INFORMATION_DETAIL and required_fields:
        missing_fields = _missing_required_answer_fields(result, state)
        total_fields = len(required_fields)
        covered_fields = max(0, total_fields - len(missing_fields))
        coverage = float(covered_fields) / float(total_fields) if total_fields else 1.0
        not_verified_mentions = _cache_not_verified_mentions(result)
        state["not_verified_mentions"] = int(not_verified_mentions)
        state["required_fields_missing"] = missing_fields
        state["required_field_coverage"] = round(max(0.0, min(1.0, coverage)), 4)
        emit_trace_event(
            "required_fields_evaluated",
            {
                "required_count": total_fields,
                "covered_count": covered_fields,
                "coverage": state["required_field_coverage"],
                "missing_fields": missing_fields[:5],
                "not_verified_mentions": int(not_verified_mentions),
            },
        )
        if missing_fields:
            issues.append("missing_required_answer_fields")
            for field in missing_fields[:4]:
                issues.append(f"missing:{field}")
        max_not_verified = max(
            1,
            int(getattr(settings.web_search, "cache_max_not_verified_mentions", 3)),
        )
        if _is_admissions_requirements_query(state):
            max_not_verified = min(max_not_verified, 1)
        if not_verified_mentions > max_not_verified:
            issues.append("too_many_unverified_fields")
        if total_fields and coverage <= 0.34:
            issues.append("query_not_addressed")
    if result != _NO_RELEVANT_INFORMATION_DETAIL and admissions_critical_missing:
        issues.append("web_required_fields_missing")
        for field in admissions_critical_missing[:4]:
            issues.append(f"web_missing:{field}")
    if (
        result != _NO_RELEVANT_INFORMATION_DETAIL
        and _mode_from_state(state) == _DEEP_MODE
        and _is_admissions_requirements_query(state)
    ):
        admissions_required = state.get("required_answer_fields")
        admissions_required = admissions_required if isinstance(admissions_required, list) else []
        if len(admissions_required) >= 3:
            confidence = _safe_float(state.get("trust_confidence"))
            if confidence is not None and confidence < _deep_answer_min_confidence():
                issues.append("confidence_below_target")
    if result != _NO_RELEVANT_INFORMATION_DETAIL and bool(
        state.get("retrieval_single_domain_low_quality", False)
    ):
        issues.append("weak_evidence_single_domain")
    if bool(state.get("deadline_query", False)) and result != _NO_RELEVANT_INFORMATION_DETAIL:
        if not _has_date_like_value(result):
            issues.append("missing_deadline_date")
    if _has_contradiction_signal(result, state):
        issues.append("source_conflict_detected")
    if result != _NO_RELEVANT_INFORMATION_DETAIL and _has_speculative_factual_language(result, state):
        issues.append("speculative_factual_claim")
    if citation_policy_enabled and result == _NO_RELEVANT_INFORMATION_DETAIL and evidence_urls:
        issues.append("returned_no_relevant_information_with_available_evidence")
    if result != _NO_RELEVANT_INFORMATION_DETAIL and _is_generic_placeholder_response(
        result, state
    ):
        issues.append("generic_placeholder_response")
        if "query_not_addressed" not in issues:
            issues.append("query_not_addressed")
    return issues


def _fallback_answer_plan(state: dict) -> dict:
    query = " ".join(str(state.get("safe_user_prompt", "")).split()).strip()[:240]
    success_criteria = [
        "answer the exact user query directly",
        "include only evidence-grounded factual claims",
        "cite allowed URLs for factual claims",
    ]
    if bool(state.get("deadline_query", False)):
        success_criteria.append("include exact date values when available")
    return {
        "intent": query,
        "subquestions": [],
        "success_criteria": success_criteria[:5],
        "planner": "heuristic",
    }


def _normalize_answer_plan_payload(payload: dict, state: dict) -> dict:
    fallback = _fallback_answer_plan(state)
    intent = " ".join(str(payload.get("intent", "")).split()).strip()[:240] or fallback["intent"]
    return {
        "intent": intent,
        "subquestions": _normalize_agentic_text_list(payload.get("subquestions"), limit=6),
        "success_criteria": _normalize_agentic_text_list(payload.get("success_criteria"), limit=8)
        or fallback["success_criteria"],
        "planner": "llm",
    }


def _answer_plan_message(plan: dict) -> dict:
    lines = [
        "Internal execution plan (do not expose):",
        f"- Intent: {str(plan.get('intent', '')).strip()[:220]}",
    ]
    for item in _normalize_agentic_text_list(plan.get("subquestions"), limit=6):
        lines.append(f"- Subquestion: {item}")
    for item in _normalize_agentic_text_list(plan.get("success_criteria"), limit=8):
        lines.append(f"- Success criterion: {item}")
    return {"role": "system", "content": "\n".join(lines)[:_RETRIEVAL_CONTEXT_MAX_CHARS]}


def _build_answer_planner_messages(messages: list, state: dict) -> list[dict]:
    query = str(state.get("safe_user_prompt", "")).strip()[:400]
    evidence_urls = _traceable_urls(state.get("evidence_urls", []), limit=10)
    lines = [
        f"User query: {query}",
        "Allowed evidence URLs:",
    ]
    if evidence_urls:
        lines.extend(f"- {url}" for url in evidence_urls)
    else:
        lines.append("- (none)")
    lines.append(
        "Return strict JSON only with keys: intent, subquestions, success_criteria."
    )
    return messages + [
        {"role": "system", "content": _agentic_planner_system_prompt()},
        {"role": "user", "content": "\n".join(lines)[:_RETRIEVAL_CONTEXT_MAX_CHARS]},
    ]


async def _generate_answer_plan(*, messages: list, state: dict) -> tuple[dict, dict, int]:
    fallback = _fallback_answer_plan(state)
    if not _agentic_planner_enabled():
        emit_trace_event("answer_planning_skipped", {"reason": "planner_disabled"})
        return fallback, {}, 0

    emit_trace_event(
        "answer_planning_started",
        {"query": str(state.get("safe_user_prompt", ""))[:220]},
    )
    try:
        response = await _call_model_with_fallback(
            _build_answer_planner_messages(messages, state),
            state,
            role="planner",
        )
    except Exception:
        emit_trace_event(
            "answer_planning_completed",
            {"planner": "heuristic", "used_fallback": True},
        )
        return fallback, {}, int(state.get("model_ms") or 0)

    usage = _extract_llm_usage(response)
    payload = _extract_json_object(str(response.choices[0].message.content or ""))
    plan = _normalize_answer_plan_payload(payload, state) if payload else fallback
    emit_trace_event(
        "answer_plan_created",
        {
            "planner": str(plan.get("planner", "heuristic")),
            "intent": str(plan.get("intent", ""))[:220],
            "subquestions": plan.get("subquestions", []),
            "success_criteria": plan.get("success_criteria", []),
        },
    )
    emit_trace_event(
        "answer_planning_completed",
        {"planner": str(plan.get("planner", "heuristic")), "used_fallback": not bool(payload)},
    )
    return plan, usage, int(state.get("model_ms") or 0)


def _build_answer_verifier_messages(
    *,
    candidate: str,
    state: dict,
    plan: dict,
    round_number: int,
) -> list[dict]:
    query = str(state.get("safe_user_prompt", "")).strip()[:400]
    evidence_urls = _traceable_urls(state.get("evidence_urls", []), limit=10)
    required_hosts = _required_citation_host_count(state, state.get("evidence_urls", []))
    lines = [
        f"User query: {query}",
        f"Draft answer (round {round_number}): {str(candidate)[:1200]}",
        f"Required minimum cited hosts: {required_hosts}",
        f"Verifier coverage threshold: {_agentic_verifier_min_coverage_score():.2f}",
        f"Plan intent: {str(plan.get('intent', '')).strip()[:220]}",
        "Plan success criteria:",
    ]
    for item in _normalize_agentic_text_list(plan.get("success_criteria"), limit=8):
        lines.append(f"- {item}")
    lines.append("Allowed evidence URLs:")
    if evidence_urls:
        lines.extend(f"- {url}" for url in evidence_urls)
    else:
        lines.append("- (none)")
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    if required_fields:
        lines.append("Required answer fields:")
        lines.extend(f"- {item}" for item in required_fields[:8])
    comparison_entities = state.get("comparison_entities")
    comparison_entities = comparison_entities if isinstance(comparison_entities, list) else []
    if comparison_entities:
        lines.append("Required comparison entities:")
        lines.extend(f"- {item}" for item in comparison_entities[:2])
    lines.append(
        "Return strict JSON only with keys: pass, coverage_score, issues, "
        "missing_points, revision_guidance."
    )
    return [
        {"role": "system", "content": _agentic_verifier_system_prompt()},
        {"role": "user", "content": "\n".join(lines)[:2600]},
    ]


def _normalize_verifier_payload(payload: dict) -> dict:
    raw_pass = payload.get("pass")
    if isinstance(raw_pass, bool):
        passed = raw_pass
    elif isinstance(raw_pass, str):
        passed = raw_pass.strip().lower() in {"1", "true", "yes", "on"}
    else:
        passed = bool(raw_pass)

    coverage = _safe_float(payload.get("coverage_score"))
    if coverage is None:
        coverage = 0.0
    return {
        "pass": passed,
        "coverage_score": max(0.0, min(1.0, coverage)),
        "issues": _normalize_agentic_text_list(payload.get("issues"), limit=8),
        "missing_points": _normalize_agentic_text_list(payload.get("missing_points"), limit=8),
        "revision_guidance": " ".join(str(payload.get("revision_guidance", "")).split()).strip()[
            :260
        ],
    }


async def _verify_answer_with_llm(
    *,
    candidate: str,
    state: dict,
    plan: dict,
    round_number: int,
) -> tuple[dict | None, dict, int]:
    if not _agentic_verifier_enabled():
        return None, {}, 0
    try:
        response = await _call_model_with_fallback(
            _build_answer_verifier_messages(
                candidate=candidate,
                state=state,
                plan=plan,
                round_number=round_number,
            ),
            state,
            role="verifier",
        )
    except Exception:
        return None, {}, int(state.get("model_ms") or 0)
    usage = _extract_llm_usage(response)
    payload = _extract_json_object(str(response.choices[0].message.content or ""))
    if not payload:
        return None, usage, int(state.get("model_ms") or 0)
    return _normalize_verifier_payload(payload), usage, int(state.get("model_ms") or 0)


def _combined_verification_issues(base_issues: list[str], verifier: dict | None) -> list[str]:
    combined: list[str] = []
    seen: set[str] = set()

    def _add(issue: str):
        key = issue.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        combined.append(issue[:140])

    for issue in base_issues:
        _add(str(issue))
    if not isinstance(verifier, dict):
        return combined
    coverage = _safe_float(verifier.get("coverage_score")) or 0.0
    if coverage < _agentic_verifier_min_coverage_score():
        _add("coverage_below_threshold")
    if not bool(verifier.get("pass", False)):
        for issue in _normalize_agentic_text_list(verifier.get("issues"), limit=8):
            _add(f"verifier:{issue}")
        if not verifier.get("issues"):
            _add("verifier_failed")
    for item in _normalize_agentic_text_list(verifier.get("missing_points"), limit=4):
        _add(f"missing:{item}")
    return combined


def _agentic_required_field_rescue_max_rounds() -> int:
    configured = _safe_int(
        getattr(settings.web_search, "agentic_required_field_rescue_max_rounds", 2)
    )
    if configured is None:
        configured = 2
    return max(0, min(3, int(configured)))


def _issue_requests_required_field_rescue(issue: str) -> bool:
    lowered = str(issue or "").strip().lower()
    if not lowered:
        return False
    direct_markers = {
        "missing_required_answer_fields",
        "web_required_fields_missing",
        "too_many_unverified_fields",
        "missing_deadline_date",
    }
    if lowered in direct_markers:
        return True
    if lowered.startswith("missing:") or lowered.startswith("web_missing:"):
        return True
    if lowered.startswith("verifier:"):
        body = lowered.split(":", 1)[1].strip()
        return any(
            marker in body
            for marker in (
                "missing",
                "not address",
                "not covered",
                "deadline",
                "portal",
                "ielts",
                "toefl",
                "language",
                "gpa",
                "ects",
                "credit",
                "eligibility",
                "requirement",
            )
        )
    return False


def _issue_missing_field_hint(issue: str) -> str:
    lowered = str(issue or "").strip().lower()
    if lowered.startswith("missing:") or lowered.startswith("web_missing:"):
        return " ".join(lowered.split(":", 1)[1].split()).strip()
    return ""


def _required_field_focus_terms(field_hint: str) -> list[str]:
    normalized = " ".join(str(field_hint or "").split()).strip().lower()
    if not normalized:
        return []
    if "deadline" in normalized:
        return ["application deadline exact date", "admission timeline date"]
    if "portal" in normalized or "apply" in normalized:
        return ["application portal link", "apply online portal"]
    if "language" in normalized and any(token in normalized for token in ("score", "ielts", "toefl")):
        return ["IELTS TOEFL minimum score", "language test threshold"]
    if "language" in normalized:
        return ["language requirement english german", "instruction language"]
    if any(token in normalized for token in ("gpa", "grade", "cgpa")):
        return ["minimum GPA grade threshold", "admission grade requirement"]
    if any(token in normalized for token in ("ects", "credit", "prerequisite")):
        return ["ECTS prerequisite credits", "credit breakdown requirement"]
    if "eligibility" in normalized or "admission requirement" in normalized:
        return ["eligibility admission requirements", "entry criteria requirements"]
    if "document" in normalized:
        return ["required documents checklist", "application documents official"]
    return [normalized]


def _normalize_domain_slug(text: str) -> str:
    compact = str(text or "").strip().lower()
    if not compact:
        return ""
    compact = (
        compact.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    compact = re.sub(r"[^a-z0-9]+", "-", compact).strip("-")
    compact = re.sub(r"-{2,}", "-", compact)
    return compact


def _inferred_rescue_domains(text: str) -> list[str]:
    compact = " ".join(str(text or "").split()).strip().lower()
    if not compact:
        return []
    inferred: list[str] = []
    seen: set[str] = set()

    def _push(domain: str) -> None:
        candidate = str(domain or "").strip().lower()
        if not candidate or candidate in seen:
            return
        if not re.fullmatch(r"(?:[a-z0-9-]+\.)+(?:de|eu|edu|ac\.uk)", candidate):
            return
        seen.add(candidate)
        inferred.append(candidate)

    for domain in re.findall(r"\b(?:[a-z0-9-]+\.)+(?:de|eu|edu|ac\.uk)\b", compact):
        _push(domain)

    stopwords = {
        "master",
        "masters",
        "m",
        "msc",
        "sc",
        "program",
        "programme",
        "course",
        "admission",
        "requirements",
        "deadline",
        "application",
        "language",
        "ielts",
        "toefl",
        "gpa",
        "ects",
    }
    pattern = re.compile(
        r"\b(?:university|universit[a-z]*|uni)\s+(?:of\s+)?"
        r"([a-z0-9äöüß-]+(?:\s+[a-z0-9äöüß-]+){0,2})",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(compact):
        raw = " ".join(str(match.group(1) or "").split()).strip()
        if not raw:
            continue
        tokens: list[str] = []
        for token in re.findall(r"[a-z0-9äöüß-]+", raw):
            lowered = token.lower()
            if lowered in stopwords:
                break
            tokens.append(lowered)
        if not tokens:
            continue
        slug = _normalize_domain_slug("-".join(tokens[:2]))
        if not slug:
            continue
        _push(f"uni-{slug}.de")
    return inferred


def _rescue_site_hints(state: dict, *, base_query: str = "") -> list[str]:
    hosts: list[str] = []
    seen: set[str] = set()

    def _push(host_or_domain: str) -> None:
        normalized = _normalized_host_from_url(str(host_or_domain))
        if not normalized:
            normalized = str(host_or_domain or "").strip().lower()
            if normalized.startswith("www."):
                normalized = normalized[4:]
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        hosts.append(normalized)

    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    for url in evidence_urls:
        _push(str(url))
        if len(hosts) >= 6:
            break
    if len(hosts) < 6:
        for domain in _inferred_rescue_domains(base_query):
            _push(domain)
            if len(hosts) >= 6:
                break
    if len(hosts) < 6:
        for domain in _inferred_rescue_domains(str(state.get("safe_user_prompt", ""))):
            _push(domain)
            if len(hosts) >= 6:
                break
    return hosts


def _targeted_required_field_rescue_queries(*, base_query: str, state: dict, issues: list[str]) -> list[str]:
    compact_base = _compact_web_retrieval_query(base_query=base_query, state=state) or str(base_query).strip()
    if not compact_base:
        return []
    candidates: list[str] = [compact_base]
    candidates.extend(_decompose_retrieval_queries(base_query=compact_base, state=state))

    missing_hints: list[str] = []
    required_missing = state.get("required_fields_missing", [])
    required_missing = required_missing if isinstance(required_missing, list) else []
    missing_hints.extend(str(item).strip() for item in required_missing if str(item).strip())
    web_missing = state.get("web_required_fields_missing", [])
    web_missing = web_missing if isinstance(web_missing, list) else []
    missing_hints.extend(str(item).strip() for item in web_missing if str(item).strip())
    for issue in issues:
        hint = _issue_missing_field_hint(issue)
        if hint:
            missing_hints.append(hint)
    if not missing_hints:
        required_fields = state.get("required_answer_fields")
        required_fields = required_fields if isinstance(required_fields, list) else []
        missing_hints.extend(str(item).strip() for item in required_fields if str(item).strip())

    focus_terms: list[str] = []
    for hint in missing_hints[:6]:
        focus_terms.extend(_required_field_focus_terms(hint))
    deduped_focus: list[str] = []
    seen_focus: set[str] = set()
    for term in focus_terms:
        compact = " ".join(str(term).split()).strip()
        if not compact:
            continue
        key = compact.lower()
        if key in seen_focus:
            continue
        seen_focus.add(key)
        deduped_focus.append(compact)
        if len(deduped_focus) >= 6:
            break

    for focus in deduped_focus:
        candidates.append(f"{compact_base} official {focus}")
    for host in _rescue_site_hints(state, base_query=compact_base)[:4]:
        candidates.append(f"{compact_base} official site:{host}")
        for focus in deduped_focus[:3]:
            candidates.append(f"{compact_base} {focus} site:{host}")

    max_gap_queries = max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2)))
    limit = max(8, min(18, max_gap_queries * 4))
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = " ".join(str(candidate).split()).strip()
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(compact[:_WEB_QUERY_MAX_CHARS])
        if len(normalized) >= limit:
            break
    return normalized


def _should_attempt_required_field_web_rescue(issues: list[str], state: dict) -> bool:
    if not issues:
        return False
    web_ready, _ = _web_retrieval_ready()
    if not web_ready:
        return False
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    if not required_fields and not _admissions_missing_web_fields(state):
        return False
    return any(_issue_requests_required_field_rescue(issue) for issue in issues)


async def _attempt_required_field_web_rescue(
    *,
    issues: list[str],
    state: dict,
    base_query: str,
    search_mode: str,
) -> tuple[list[dict], bool]:
    queries = _targeted_required_field_rescue_queries(
        base_query=base_query,
        state=state,
        issues=issues,
    )
    if not queries:
        return [], False
    emit_trace_event(
        "agent_required_field_rescue_started",
        {
            "query_count": len(queries),
            "issues": issues[:6],
            "query_variants": queries[:8],
        },
    )

    max_context_results = _max_context_results_for_mode(search_mode)
    merge_limit = max(
        max_context_results,
        int(_default_retrieval_top_k()) * 2,
        int(settings.bedrock.reranker_max_documents),
    )
    timeout_count = 0
    timed_out_queries: list[str] = []
    rescued_rows: list[dict] = []
    best_coverage: float | None = None
    best_missing: list[str] = []
    verified_values: list[bool] = []

    tasks = [
        asyncio.create_task(
            _run_one_web_query_with_timeout(
                query,
                top_k=_default_retrieval_top_k(),
                search_mode=search_mode,
            )
        )
        for query in queries
    ]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    for item in gathered:
        if isinstance(item, Exception) or not isinstance(item, dict):
            continue
        if _web_result_timed_out(item):
            timeout_count += 1
            timeout_query = " ".join(str(item.get("_query", "")).split()).strip()
            if timeout_query:
                timed_out_queries.append(timeout_query)
            continue
        verification = item.get("verification", {})
        verification = verification if isinstance(verification, dict) else {}
        coverage = _safe_float(verification.get("required_field_coverage"))
        if coverage is not None:
            coverage = _clamp01(coverage, fallback=1.0)
            if best_coverage is None or coverage > best_coverage:
                best_coverage = coverage
        missing = verification.get("required_fields_missing", [])
        if isinstance(missing, list):
            normalized_missing = [
                " ".join(str(value).split()).strip()
                for value in missing
                if " ".join(str(value).split()).strip()
            ]
            if not best_missing or len(normalized_missing) < len(best_missing):
                best_missing = normalized_missing[:6]
        verified = verification.get("verified")
        if isinstance(verified, bool):
            verified_values.append(verified)
        rescued_rows = _merge_retrieval_results(
            rescued_rows,
            _result_dicts(item.get("results", [])),
            limit=merge_limit,
        )

    if timeout_count > 0:
        state["web_timeout_count"] = int(state.get("web_timeout_count", 0) or 0) + timeout_count
        existing_timeouts = state.get("web_timed_out_queries", [])
        existing_timeouts = existing_timeouts if isinstance(existing_timeouts, list) else []
        deduped: list[str] = []
        seen_timeout: set[str] = set()
        for timeout_query in existing_timeouts + timed_out_queries:
            compact = " ".join(str(timeout_query).split()).strip()
            if not compact:
                continue
            key = compact.lower()
            if key in seen_timeout:
                continue
            seen_timeout.add(key)
            deduped.append(compact)
            if len(deduped) >= 8:
                break
        state["web_timed_out_queries"] = deduped
    if best_coverage is not None:
        state["web_required_field_coverage"] = round(best_coverage, 4)
    if best_missing:
        state["web_required_fields_missing"] = best_missing
    if verified_values:
        state["web_retrieval_verified"] = any(verified_values)
    state["web_fallback_attempted"] = True

    if not rescued_rows:
        emit_trace_event(
            "agent_required_field_rescue_completed",
            {
                "rescued": False,
                "query_count": len(queries),
                "timeout_count": timeout_count,
            },
        )
        return [], False

    previous_results = state.get("retrieved_results", [])
    previous_results = previous_results if isinstance(previous_results, list) else []
    merged_results = _merge_retrieval_results(previous_results, rescued_rows, limit=merge_limit)
    merged_results = await _rerank_if_configured(base_query, merged_results, state)
    merged_results = _selective_retrieval_results(merged_results, state)
    context_messages, detail = _apply_grounded_retrieval_context(
        messages=[],
        merged_results=merged_results,
        used_web_results=True,
        state=state,
    )
    if detail:
        emit_trace_event(
            "agent_required_field_rescue_completed",
            {
                "rescued": False,
                "reason": "context_detail_guard",
                "detail": str(detail)[:180],
            },
        )
        return [], False
    context_messages = context_messages if isinstance(context_messages, list) else []
    if not context_messages:
        emit_trace_event(
            "agent_required_field_rescue_completed",
            {
                "rescued": False,
                "reason": "empty_context_messages",
            },
        )
        return [], False
    strategy = str(state.get("retrieval_strategy", "")).strip()
    state["retrieval_strategy"] = (
        f"{strategy}_required_field_rescue" if strategy else "required_field_rescue"
    )
    emit_trace_event(
        "agent_required_field_rescue_completed",
        {
            "rescued": True,
            "query_count": len(queries),
            "rescue_result_count": len(rescued_rows),
            "merged_result_count": len(merged_results),
            "source_count": int(state.get("retrieval_source_count", 0) or 0),
            "required_field_coverage": state.get("web_required_field_coverage"),
        },
    )
    return context_messages, True


def _agentic_reflection_message(
    issues: list[str], round_number: int, verifier: dict | None = None
) -> dict:
    compact = ", ".join(issues[:5]) if issues else "quality_check_failed"
    guidance = ""
    if isinstance(verifier, dict):
        guidance = str(verifier.get("revision_guidance", "")).strip()
    extra = f" Guidance: {guidance}" if guidance else ""
    return {
        "role": "system",
        "content": (
            f"Revision round {round_number}: improve the answer. "
            "Fix all missing required answer fields first, then improve claim-level grounding. "
            f"Fix: {compact}. Use only allowed evidence URLs and keep claims verifiable.{extra}"
        ),
    }


def _build_answer_finalizer_messages(*, candidate: str, state: dict, plan: dict) -> list[dict]:
    query = str(state.get("safe_user_prompt", "")).strip()[:400]
    evidence_urls = _traceable_urls(state.get("evidence_urls", []), limit=10)
    lines = [
        f"User query: {query}",
        f"Draft answer to finalize: {str(candidate)[:1500]}",
        f"Plan intent: {str(plan.get('intent', '')).strip()[:220]}",
        "Allowed evidence URLs:",
    ]
    if evidence_urls:
        lines.extend(f"- {url}" for url in evidence_urls)
    else:
        lines.append("- (none)")
    lines.append(
        "Rewrite for clarity and structure only. Keep all factual claims strictly evidence-grounded."
    )
    lines.append("Do not invent facts, dates, thresholds, or links.")
    return [
        {
            "role": "system",
            "content": (
                "You are an internal answer finalizer. Improve readability while preserving factual "
                "correctness and citation grounding. Do not add unsupported claims."
            ),
        },
        {"role": "user", "content": "\n".join(lines)[:2600]},
    ]


async def _finalize_candidate_with_llm(
    *,
    user_id: str,
    candidate: str,
    state: dict,
    plan: dict,
    attempt: int,
) -> tuple[str, dict, int]:
    finalizer_primary, _ = _model_ids_for_role("finalizer")
    worker_primary, _ = _model_ids_for_role("worker", attempt=attempt)
    if (
        not finalizer_primary
        or finalizer_primary == worker_primary
        or candidate == _NO_RELEVANT_INFORMATION_DETAIL
    ):
        return "", {}, 0

    emit_trace_event(
        "answer_finalization_started",
        {"model": finalizer_primary, "answer_preview": str(candidate)[:220]},
    )
    try:
        response = await _call_model_with_fallback(
            _build_answer_finalizer_messages(candidate=candidate, state=state, plan=plan),
            state,
            role="finalizer",
        )
    except Exception:
        emit_trace_event(
            "answer_finalization_completed",
            {"changed": False, "reason": "finalizer_call_failed"},
        )
        return "", {}, int(state.get("model_ms") or 0)

    usage = _extract_llm_usage(response)
    finalized = _extract_guarded_result(
        user_id=user_id,
        raw_result=response.choices[0].message.content,
        state=state,
    )
    if not str(finalized or "").strip():
        emit_trace_event(
            "answer_finalization_completed",
            {"changed": False, "reason": "empty_finalizer_output"},
        )
        return "", usage, int(state.get("model_ms") or 0)
    changed = str(finalized).strip() != str(candidate).strip()
    emit_trace_event(
        "answer_finalization_completed",
        {"changed": changed, "answer_preview": str(finalized)[:220]},
    )
    return finalized, usage, int(state.get("model_ms") or 0)


def _has_authoritative_evidence(state: dict) -> bool:
    if int(state.get("retrieval_source_count", 0) or 0) <= 0:
        return False
    authority_score = _safe_float(state.get("trust_authority_score"))
    if authority_score is not None and authority_score >= _PARTIAL_EVIDENCE_AUTHORITY_MIN_SCORE:
        return True

    retrieved_results = state.get("retrieved_results", [])
    if not isinstance(retrieved_results, list):
        return False
    for result in retrieved_results:
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        trust_components = metadata.get("trust_components")
        trust_components = trust_components if isinstance(trust_components, dict) else {}
        authority = _safe_float(trust_components.get("authority"))
        if authority is not None and authority >= _PARTIAL_EVIDENCE_AUTHORITY_MIN_SCORE:
            return True
    return False


def _is_generic_placeholder_response(answer: str, state: dict) -> bool:
    text = " ".join(str(answer or "").split()).strip()
    if not text:
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in _GENERIC_PLACEHOLDER_MARKERS):
        return True
    return False


def _has_speculative_factual_language(answer: str, state: dict) -> bool:
    if not _is_admissions_requirements_query(state):
        return False
    text = " ".join(str(answer or "").split()).strip()
    if not text:
        return False
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        lowered = sentence.lower().strip()
        if not lowered:
            continue
        if "not verified from sources" in lowered or "not verified from evidence" in lowered:
            continue
        if not _SPECULATIVE_FACTUAL_CLAIM_RE.search(lowered):
            continue
        if _SPECULATIVE_FACTUAL_FIELD_RE.search(lowered):
            return True
    return False


def _issues_include_query_not_answered(issues: list[str]) -> bool:
    for issue in issues:
        lowered = str(issue).strip().lower()
        if not lowered:
            continue
        if lowered == "query_not_addressed":
            return True
        if lowered.startswith("verifier:"):
            body = lowered.split(":", 1)[1].strip()
            if any(marker in body for marker in _QUERY_NOT_ANSWERED_MARKERS):
                return True
    return False


def _allow_partial_admissions_answer(*, issues: list[str], result: str, state: dict) -> bool:
    if not _is_admissions_requirements_query(state):
        return False
    if not str(result or "").strip() or str(result or "").strip() == _NO_RELEVANT_INFORMATION_DETAIL:
        return False
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    if not evidence_urls:
        return False
    if _issues_include_query_not_answered(issues):
        return False
    if _has_speculative_factual_language(result, state):
        return False
    if "missing_allowed_citations" in {str(item).strip().lower() for item in issues}:
        return False
    required_coverage = _safe_float(state.get("required_field_coverage"))
    web_coverage = _safe_float(state.get("web_required_field_coverage"))
    coverages = [value for value in (required_coverage, web_coverage) if value is not None]
    if not coverages:
        return False
    mode = _mode_from_state(state)
    min_coverage = 0.65 if mode == _DEEP_MODE else 0.45
    if max(coverages) < min_coverage:
        return False
    # When deep search exhausted or timed out, return best-evidence partial instead of hard abstaining.
    if mode == _DEEP_MODE:
        timed_out = int(state.get("web_timeout_count", 0) or 0) > 0
        rescue_rounds = int(state.get("agent_required_field_rescue_rounds", 0) or 0)
        rescue_budget = _agentic_required_field_rescue_max_rounds()
        if (
            not timed_out
            and bool(_admissions_critical_web_fields_missing(state))
            and rescue_budget > 0
            and rescue_rounds < rescue_budget
        ):
            return False
        if not timed_out and not bool(state.get("web_fallback_attempted", False)):
            return False
    return True


def _is_hard_verification_failure(issues: list[str], result: str, state: dict) -> bool:
    if result == _NO_RELEVANT_INFORMATION_DETAIL:
        return True
    if _has_speculative_factual_language(result, state):
        return True
    has_authoritative = _has_authoritative_evidence(state)
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    if not evidence_urls and not has_authoritative:
        return True
    if evidence_urls and not _response_has_allowed_citation(result, evidence_urls):
        if not has_authoritative:
            return True
    if _issues_include_query_not_answered(issues):
        return True
    allow_partial_admissions = _allow_partial_admissions_answer(
        issues=issues,
        result=result,
        state=state,
    )
    hard_markers = (
        "weak_evidence",
        "web_required_fields_missing",
        "too_many_unverified_fields",
        "speculative_factual_claim",
    )
    for issue in issues:
        lowered = str(issue).strip().lower()
        if lowered == "missing_allowed_citations" and has_authoritative:
            continue
        if lowered == "insufficient_source_diversity" and has_authoritative:
            continue
        if (
            lowered in {"missing_evidence_snippets", "weak_claim_citation_linkage"}
            and has_authoritative
        ):
            continue
        if allow_partial_admissions and (
            lowered in {"web_required_fields_missing", "too_many_unverified_fields"}
            or lowered.startswith("web_missing:")
            or lowered in {"confidence_below_target", "missing_deadline_date"}
            or lowered.startswith("missing:")
        ):
            continue
        if lowered in {"missing_allowed_citations", "insufficient_source_diversity"}:
            return True
        if lowered in {"missing_evidence_snippets", "weak_claim_citation_linkage"}:
            return True
        if any(marker in lowered for marker in hard_markers):
            return True
    return False


def _candidate_quality_score(candidate: str, issues: list[str], state: dict) -> tuple[int, int]:
    penalty = len(issues)
    if candidate == _NO_RELEVANT_INFORMATION_DETAIL:
        penalty += 200
    if _is_generic_placeholder_response(candidate, state):
        penalty += 140
    if _issues_include_query_not_answered(issues):
        penalty += 120
    if _is_hard_verification_failure(issues, candidate, state):
        penalty += 50
    length_bonus = min(len(str(candidate).strip()), 4000)
    return penalty, -length_bonus


def _derive_abstain_reason(result: str, state: dict) -> str:
    normalized_result = str(result or "").strip()
    if normalized_result == _WEB_RETRIEVAL_TIMEOUT_DETAIL:
        return "web_timeout"
    if normalized_result != _NO_RELEVANT_INFORMATION_DETAIL:
        return ""

    output_guard_reason = str(state.get("output_guard_reason", "")).strip().lower()
    context_guard_reason = str(state.get("context_guard_reason", "")).strip().lower()
    if output_guard_reason == "web_retrieval_timeout" or context_guard_reason == "web_retrieval_timeout":
        return "web_timeout"
    if output_guard_reason in {"agent_verification_failed", "stream_verification_failed"}:
        return "verifier_blocked"
    if output_guard_reason in {"weak_evidence_no_urls", "weak_evidence_missing"}:
        return "insufficient_domains"
    if output_guard_reason == "missing_citations":
        return "verifier_blocked" if _has_authoritative_evidence(state) else "insufficient_domains"

    if (
        bool(state.get("web_fallback_attempted", False))
        and int(state.get("web_result_count", 0) or 0) <= 0
    ):
        return "no_web"

    top_similarity = _safe_float(state.get("retrieval_top_similarity"))
    if top_similarity is not None and top_similarity < _web_expansion_similarity_threshold():
        return "low_similarity"

    domain_count = int(state.get("evidence_domain_count", 0) or 0)
    if domain_count <= 0:
        domain_count = int(state.get("retrieval_source_count", 0) or 0)
    if domain_count < _web_expansion_min_domain_count():
        return "insufficient_domains"

    issues = state.get("agent_last_issues")
    if isinstance(issues, list):
        for issue in issues:
            lowered = str(issue).strip().lower()
            if "verifier:" in lowered or "coverage_below_threshold" in lowered:
                return "verifier_blocked"

    return "verifier_blocked"


async def _generate_agentic_answer(
    *,
    user_id: str,
    messages: list,
    policy: dict,
    state: dict,
) -> tuple[str, dict]:
    max_attempts = max(1, int(policy.get("max_attempts", 1)))
    attempt_limit = max_attempts
    planner_enabled = bool(policy.get("planner_enabled", False))
    verifier_enabled = bool(policy.get("verifier_enabled", False))
    rescue_max_rounds = _agentic_required_field_rescue_max_rounds()
    rescue_rounds = 0
    attempt = 0
    model_ms_total = 0
    llm_usage_total: dict = {}
    working_messages = list(messages)
    plan = _fallback_answer_plan(state)
    final_result = _NO_RELEVANT_INFORMATION_DETAIL
    final_issues: list[str] = []
    best_result = _NO_RELEVANT_INFORMATION_DETAIL
    best_issues: list[str] = []
    best_score = (10**9, 0)
    emit_trace_event(
        "answer_synthesis_started",
        {
            "max_rounds": max_attempts,
        },
    )
    if planner_enabled:
        plan, plan_usage, plan_model_ms = await _generate_answer_plan(
            messages=working_messages, state=state
        )
        model_ms_total += plan_model_ms
        llm_usage_total = _merge_llm_usage(llm_usage_total, plan_usage)
    else:
        emit_trace_event("answer_planning_skipped", {"reason": "mode_policy"})
    working_messages = working_messages + [_answer_plan_message(plan)]

    while attempt < attempt_limit:
        attempt += 1
        emit_trace_event(
            "model_round_started",
            {
                "round": attempt,
                "max_rounds": max_attempts,
            },
        )
        response = await _call_model_with_fallback(
            working_messages,
            state,
            role="worker",
            attempt=attempt,
        )
        model_ms_total += int(state.get("model_ms") or 0)
        llm_usage_total = _merge_llm_usage(llm_usage_total, _extract_llm_usage(response))

        raw_result = response.choices[0].message.content
        candidate = _extract_guarded_result(user_id=user_id, raw_result=raw_result, state=state)
        base_issues = _agentic_result_issues(candidate, state)
        verifier = None
        verifier_usage = {}
        verifier_model_ms = 0
        if verifier_enabled:
            verifier, verifier_usage, verifier_model_ms = await _verify_answer_with_llm(
                candidate=candidate,
                state=state,
                plan=plan,
                round_number=attempt,
            )
        else:
            emit_trace_event(
                "answer_verification_completed",
                {
                    "round": attempt,
                    "verified": not base_issues,
                    "issues": base_issues,
                    "base_issues": base_issues,
                    "verifier": {},
                    "mode": str(policy.get("mode", _DEEP_MODE)),
                    "verifier_skipped": True,
                },
            )
        model_ms_total += verifier_model_ms
        llm_usage_total = _merge_llm_usage(llm_usage_total, verifier_usage)
        issues = _combined_verification_issues(base_issues, verifier)
        final_result = candidate
        final_issues = issues
        candidate_score = _candidate_quality_score(candidate, issues, state)
        if candidate_score < best_score:
            best_score = candidate_score
            best_result = candidate
            best_issues = list(issues)
        emit_trace_event(
            "model_round_completed",
            {
                "round": attempt,
                "issues": issues,
                "answer_preview": str(candidate)[:260],
            },
        )
        if verifier_enabled:
            emit_trace_event(
                "answer_verification_completed",
                {
                    "round": attempt,
                    "verified": not issues,
                    "issues": issues,
                    "base_issues": base_issues,
                    "verifier": verifier or {},
                },
            )
        if not issues:
            break

        rescue_applied = False
        if rescue_rounds < rescue_max_rounds and _should_attempt_required_field_web_rescue(
            issues, state
        ):
            rescue_context_messages, rescue_applied = await _attempt_required_field_web_rescue(
                issues=issues,
                state=state,
                base_query=str(state.get("safe_user_prompt", "")),
                search_mode=_DEEP_MODE,
            )
            rescue_rounds += 1
            state["agent_required_field_rescue_rounds"] = rescue_rounds
            if rescue_applied:
                if attempt_limit <= attempt:
                    attempt_limit = attempt + 1
                working_messages = working_messages + [
                    {"role": "assistant", "content": str(candidate)},
                    _agentic_reflection_message(issues, attempt + 1, verifier),
                ] + rescue_context_messages
                continue
        if attempt >= attempt_limit:
            break

        working_messages = working_messages + [
            {"role": "assistant", "content": str(candidate)},
            _agentic_reflection_message(issues, attempt + 1, verifier),
        ]

    state["model_ms"] = model_ms_total
    state["agent_rounds"] = attempt
    state["agent_required_field_rescue_rounds"] = rescue_rounds
    finalized_result, finalizer_usage, finalizer_model_ms = await _finalize_candidate_with_llm(
        user_id=user_id,
        candidate=best_result if best_result else final_result,
        state=state,
        plan=plan,
        attempt=attempt,
    )
    model_ms_total += finalizer_model_ms
    llm_usage_total = _merge_llm_usage(llm_usage_total, finalizer_usage)
    if finalized_result:
        finalized_issues = _agentic_result_issues(finalized_result, state)
        if _candidate_quality_score(finalized_result, finalized_issues, state) < _candidate_quality_score(
            best_result if best_result else final_result,
            best_issues if best_issues else final_issues,
            state,
        ):
            final_result = finalized_result
            final_issues = finalized_issues
        else:
            final_result = best_result if best_result else final_result
            final_issues = best_issues if best_issues else final_issues
    else:
        final_result = best_result if best_result else final_result
        final_issues = best_issues if best_issues else final_issues
    state["model_ms"] = model_ms_total
    if final_issues and str(state.get("query_intent", "")).strip().lower() == "comparison":
        structured_comparison = _build_structured_comparison_from_evidence(state)
        if structured_comparison:
            structured_issues = _agentic_result_issues(structured_comparison, state)
            structured_score = _candidate_quality_score(structured_comparison, structured_issues, state)
            current_score = _candidate_quality_score(final_result, final_issues, state)
            if structured_score < current_score:
                final_result = structured_comparison
                final_issues = structured_issues
                emit_trace_event(
                    "answer_structured_comparison_recovery",
                    {
                        "issues": final_issues[:8],
                        "answer_preview": str(final_result)[:220],
                    },
                )
    state["agent_last_issues"] = final_issues
    if final_issues and final_result != _NO_RELEVANT_INFORMATION_DETAIL:
        if _is_hard_verification_failure(final_issues, final_result, state):
            state["output_guard_reason"] = "agent_verification_failed"
            final_result = _NO_RELEVANT_INFORMATION_DETAIL
        else:
            state["output_guard_reason"] = "agent_verification_partial"
            emit_trace_event(
                "answer_partial_with_evidence",
                {
                    "issues": final_issues[:8],
                    "answer_preview": str(final_result)[:220],
                },
            )
    state["abstain_reason"] = _derive_abstain_reason(final_result, state)
    emit_trace_event(
        "answer_synthesis_completed",
        {
            "rounds_used": attempt,
            "verified": not final_issues,
            "issues": final_issues,
        },
    )
    return final_result, llm_usage_total


def _track_background_task(task: asyncio.Task, *, label: str) -> None:
    """Track and log fire-and-forget tasks so they are not silently lost."""
    _BACKGROUND_TASKS.add(task)

    def _on_done(completed: asyncio.Task) -> None:
        _BACKGROUND_TASKS.discard(completed)
        try:
            completed.result()
        except Exception as exc:
            logger.warning("%s failed in background. %s", label, exc)

    task.add_done_callback(_on_done)


async def _persist_evaluation_trace(
    *,
    user_id: str,
    prompt: str,
    answer: str,
    retrieved_results: list[dict],
    retrieval_strategy: str,
    build_context_ms: int,
    retrieval_ms: int,
    model_ms: int,
    quality: dict,
    evidence_urls: list[str],
) -> None:
    """Persist evaluation traces outside the request critical path."""
    await asyncio.to_thread(
        store_chat_trace,
        user_id=user_id,
        prompt=prompt,
        answer=answer,
        retrieved_results=retrieved_results,
        retrieval_strategy=retrieval_strategy,
        timings_ms={
            "build_context": build_context_ms,
            "retrieval": retrieval_ms,
            "model": model_ms,
        },
        quality=quality,
        evidence_urls=evidence_urls,
        redis=redis_client,
    )


async def _chat_completion(messages: list, *, model_id: str):
    """Send one non-streaming chat completion request."""
    if _truthy_env(_LLM_MOCK_MODE_ENV):
        delay_seconds = _llm_mock_delay_seconds()
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        return _mock_completion_response(messages)
    web_grounding_enabled = bool(getattr(settings.bedrock, "web_grounding_enabled", False))
    enable_web_grounding = web_grounding_enabled and "nova" in str(model_id).lower()
    return await client.chat.completions.create(
        model=model_id,
        messages=messages,
        limiter_name="llm_answer",
        rate_limit_profile="answer",
        enable_web_grounding=enable_web_grounding,
    )


async def _chat_completion_stream(messages: list, *, model_id: str) -> AsyncIterator[str]:
    """Stream token deltas for one model id."""
    if _truthy_env(_LLM_MOCK_MODE_ENV):
        text = _llm_mock_text(messages)
        chunk_size = _llm_mock_stream_chunk_chars()
        delay_seconds = _llm_mock_delay_seconds()
        for start in range(0, len(text), chunk_size):
            chunk = text[start : start + chunk_size]
            if not chunk:
                continue
            yield chunk
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
        return

    async for delta in client.chat.completions.stream(
        model=model_id,
        messages=messages,
        limiter_name="llm_answer",
        rate_limit_profile="answer",
    ):
        yield delta


async def _call_primary(messages: list):
    """Send the request to the primary Bedrock model."""
    return await _chat_completion(messages, model_id=settings.bedrock.primary_model_id)


async def _call_fallback(messages: list):
    """Send the request to the fallback Bedrock model."""
    return await _chat_completion(messages, model_id=settings.bedrock.fallback_model_id)


async def _stream_primary(messages: list) -> AsyncIterator[str]:
    """Stream token deltas from the primary deployment."""
    async for delta in _chat_completion_stream(
        messages, model_id=settings.bedrock.primary_model_id
    ):
        yield delta


async def _stream_fallback(messages: list) -> AsyncIterator[str]:
    """Stream token deltas from the fallback deployment."""
    async for delta in _chat_completion_stream(
        messages, model_id=settings.bedrock.fallback_model_id
    ):
        yield delta


def _new_metrics_state() -> dict:
    return {
        "build_context_ms": None,
        "retrieval_ms": None,
        "model_ms": None,
        "memory_update_ms": None,
        "cache_read_ms": None,
        "cache_write_ms": None,
        "evaluation_trace_ms": None,
        "retrieval_strategy": "none",
        "query_intent": "unknown",
        "retrieval_query_variants": [],
        "retrieved_count": 0,
        "retrieval_source_count": 0,
        "retrieval_top_similarity": None,
        "web_fallback_attempted": False,
        "web_result_count": 0,
        "web_expansion_used": False,
        "web_retrieval_verified": None,
        "web_required_field_coverage": None,
        "web_required_fields_missing": [],
        "web_timeout_count": 0,
        "web_timed_out_queries": [],
        "web_timeout_rescued": False,
        "retrieval_reranker_applied": False,
        "retrieval_reranker_ms": None,
        "retrieval_selective_before_count": 0,
        "retrieval_selective_after_count": 0,
        "retrieval_selective_dropped": 0,
        "retrieval_avg_quality": None,
        "retrieval_single_domain_low_quality": False,
        "retrieved_results": [],
        "retrieval_evidence": [],
        "citation_required": False,
        "allow_uncited_comparison_fallback": False,
        "citation_min_hosts": 1,
        "evidence_urls": [],
        "evidence_domain_count": 0,
        "comparison_entities": [],
        "required_answer_fields": [],
        "required_field_coverage": None,
        "required_fields_missing": [],
        "trust_confidence": None,
        "trust_freshness": "unknown",
        "trust_contradiction_flag": False,
        "trust_authority_score": None,
        "trust_agreement_score": None,
        "trust_recency_score": None,
        "trust_uncertainty_reasons": [],
        "claim_citation_coverage": None,
        "claim_snippet_grounding_coverage": None,
        "claim_snippet_conflict_count": 0,
        "claim_evidence_map": [],
        "claim_count": 0,
        "claim_cited_count": 0,
        "deadline_query": False,
        "llm_usage": {},
        "quality": {},
        "agentic_enabled": _agentic_enabled(),
        "agent_rounds": 0,
        "agent_required_field_rescue_rounds": 0,
        "agent_last_issues": [],
        "abstain_reason": "",
        "requested_mode": _DEFAULT_EXECUTION_MODE,
        "execution_mode": _DEFAULT_EXECUTION_MODE,
        "auto_escalated": False,
        "input_guard_reason": "",
        "context_guard_reason": "",
        "output_guard_reason": "",
        "safe_user_prompt": "",
        "used_fallback_model": False,
        "role_model_ids": {},
    }


async def _record_request_outcome(
    *,
    request_id: str,
    started_at: float,
    user_id: str,
    session_id: str,
    user_prompt: str,
    safe_user_prompt: str,
    answer: str,
    outcome: str,
    state: dict,
    error_message: str = "",
) -> None:
    await _record_json_metrics(
        _build_json_metrics_record(
            request_id=request_id,
            started_at=started_at,
            user_id=user_id,
            session_id=session_id,
            user_prompt=user_prompt,
            safe_user_prompt=safe_user_prompt,
            answer=answer,
            outcome=outcome,
            metrics_state=state,
            error_message=error_message,
        )
    )
    await _record_latency_metrics(started_at, outcome)


async def _read_cached_response(cache_key: str) -> tuple[str | None, int]:
    started_at = time.perf_counter()
    cached = None
    try:
        cached = await _redis_call(async_redis_client.get, cache_key)
    except RedisError as exc:
        logger.warning("Redis cache read failed. %s", exc)
    return cached, _elapsed_ms(started_at)


async def _prepare_messages_for_model(
    *,
    user_id: str,
    conversation_user_id: str,
    safe_user_prompt: str,
    execution_mode: str,
    policy: dict,
    state: dict,
) -> tuple[list | None, str | None]:
    build_context_started_at = time.perf_counter()
    state["deadline_query"] = _is_deadline_query(safe_user_prompt)
    state["query_intent"] = _classify_query_intent(safe_user_prompt)
    state["comparison_entities"] = _comparison_entities_from_prompt(safe_user_prompt)
    state["required_answer_fields"] = _required_answer_fields(
        safe_user_prompt,
        intent=str(state.get("query_intent", "unknown")),
    )
    emit_trace_event(
        "query_intent_classified",
        {
            "intent": str(state.get("query_intent", "unknown")),
            "query": str(safe_user_prompt)[:220],
            "comparison_entities": state.get("comparison_entities", []),
            "required_answer_fields": state.get("required_answer_fields", []),
        },
    )
    vector_prefetch_task: asyncio.Task | None = None
    prefetched_vector_result: dict | None = None
    if (
        _retrieval_fanout_enabled()
        and bool(getattr(settings.postgres, "enabled", False))
        and not _truthy_env(_RETRIEVAL_DISABLED_ENV)
        and str(safe_user_prompt).strip()
    ):
        # Fan-out: prefetch vector retrieval while short-term context is loading.
        vector_prefetch_task = asyncio.create_task(
            aretrieve_document_chunks(
                str(safe_user_prompt),
                top_k=_default_retrieval_top_k(),
            )
        )
    messages = await build_context(conversation_user_id, safe_user_prompt)
    state["build_context_ms"] = _elapsed_ms(build_context_started_at)

    retrieval_query = _build_retrieval_query(messages)
    if _truthy_env(_RETRIEVAL_DISABLED_ENV):
        state["retrieval_strategy"] = "disabled"
        state["retrieval_ms"] = 0
    elif retrieval_query:
        if vector_prefetch_task is not None:
            if retrieval_query.strip() == str(safe_user_prompt).strip():
                with suppress(Exception):
                    prefetched_vector_result = await vector_prefetch_task
            else:
                if not vector_prefetch_task.done():
                    vector_prefetch_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await vector_prefetch_task
        messages, detail = await _augment_messages_with_retrieval(
            messages=messages,
            retrieval_query=retrieval_query,
            search_mode=str(policy.get("web_search_mode", _DEEP_MODE)),
            state=state,
            vector_prefetch_result=prefetched_vector_result,
        )
        if detail:
            return None, detail
    else:
        if vector_prefetch_task is not None:
            if not vector_prefetch_task.done():
                vector_prefetch_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await vector_prefetch_task
        state["retrieval_ms"] = 0

    citation_detail = _validate_citation_grounding_state(state)
    if citation_detail:
        return None, citation_detail

    chat_system_prompt = prompts.get("chat", {}).get("system_prompt", "")
    system_messages: list[dict] = []
    if state["deadline_query"]:
        system_messages.append(
            {
                "role": "system",
                "content": (
                    "Date-answer policy: when the user asks for deadlines/dates, "
                    "return exact date values from evidence with URL citations. "
                    "If exact dates are absent, explicitly write: "
                    '"Application deadline: Not verified from sources."'
                ),
            }
        )
    if isinstance(chat_system_prompt, str) and chat_system_prompt.strip():
        system_messages.append({"role": "system", "content": chat_system_prompt.strip()})
    if system_messages:
        messages = system_messages + messages
    messages = _insert_system_message_before_dialog(
        messages,
        _mode_instruction_message(execution_mode),
    )
    required_fields_message = _required_fields_system_message(state)
    if required_fields_message:
        messages = _insert_system_message_before_dialog(messages, required_fields_message)
    admissions_schema_message = _admissions_answer_schema_message(state)
    if admissions_schema_message:
        messages = _insert_system_message_before_dialog(messages, admissions_schema_message)
    messages = _insert_system_message_before_dialog(
        messages,
        _answer_style_instruction_message(execution_mode, state),
    )
    if bool(state.get("agentic_enabled", False)):
        messages = messages + [_agentic_instruction_message()]

    context_guard = apply_context_guardrails(messages)
    if not context_guard["blocked"]:
        return context_guard["messages"], None

    state["context_guard_reason"] = str(context_guard.get("reason", "blocked_context"))
    refusal = refusal_response()
    logger.info(
        "GuardrailDecision | stage=context | user=%s | blocked=true | reason=%s",
        user_id,
        state["context_guard_reason"],
    )
    return None, refusal


async def _call_model_by_id(messages: list, *, model_id: str):
    normalized = _normalized_model_id(model_id)
    if not normalized:
        raise RuntimeError("Model id is required.")
    if normalized == _normalized_model_id(settings.bedrock.primary_model_id):
        return await _call_primary(messages)
    if normalized == _normalized_model_id(settings.bedrock.fallback_model_id):
        return await _call_fallback(messages)
    return await _chat_completion(messages, model_id=normalized)


async def _call_model_with_fallback(
    messages: list,
    state: dict,
    *,
    role: str = "worker",
    attempt: int = 1,
):
    primary_model_id, fallback_model_id = _model_ids_for_role(role, attempt=attempt)
    role_model_ids = state.get("role_model_ids")
    if not isinstance(role_model_ids, dict):
        role_model_ids = {}
    role_model_ids[str(role)] = {
        "primary": primary_model_id,
        "fallback": fallback_model_id,
    }
    state["role_model_ids"] = role_model_ids
    model_started_at = time.perf_counter()
    try:
        response = await _call_model_by_id(messages, model_id=primary_model_id)
    except Exception as primary_exc:
        state["used_fallback_model"] = True
        logger.warning(
            "Primary model failed; attempting fallback. role=%s primary=%s fallback=%s error=%s",
            role,
            primary_model_id,
            fallback_model_id,
            primary_exc,
        )
        try:
            response = await _call_model_by_id(messages, model_id=fallback_model_id)
        except Exception:
            logger.exception("Fallback model also failed.")
            state["model_ms"] = _elapsed_ms(model_started_at)
            raise
    state["model_ms"] = _elapsed_ms(model_started_at)
    return response


def _extract_guarded_result(*, user_id: str, raw_result, state: dict) -> str:
    guarded_output = guard_model_output(raw_result)
    result = guarded_output["text"]
    if guarded_output["blocked"]:
        state["output_guard_reason"] = str(guarded_output.get("reason", "blocked_output"))
        logger.info(
            "GuardrailDecision | stage=output | user=%s | blocked=true | reason=%s",
            user_id,
            state["output_guard_reason"],
        )
    grounded_result = _enforce_citation_grounding(str(result), state)
    if grounded_result == _NO_RELEVANT_INFORMATION_DETAIL:
        return grounded_result
    if bool(state.get("deadline_query", False)) and not _has_date_like_value(grounded_result):
        state["output_guard_reason"] = "deadline_missing_date"
        lowered = grounded_result.lower()
        if _has_not_verified_marker(grounded_result) and "application deadline" in lowered:
            return grounded_result
        deadline_line = "Application deadline: Not verified from sources."
        if "application deadline" not in lowered:
            return f"{grounded_result.rstrip()}\n\n{deadline_line}".strip()
    return grounded_result


def _should_refine_stream_draft(*, execution_mode: str, draft: str, state: dict) -> bool:
    text = str(draft or "").strip()
    if not text or text in {_NO_RELEVANT_INFORMATION_DETAIL, refusal_response()}:
        return False
    if str(execution_mode).strip().lower() != _FAST_MODE:
        return False
    if not _is_citation_grounding_required():
        return False
    if not bool(state.get("citation_required", False)):
        return False
    evidence_urls = state.get("evidence_urls", [])
    if not isinstance(evidence_urls, list) or not evidence_urls:
        return False
    issues = _agentic_result_issues(text, state)
    state["stream_draft_issues"] = issues[:8]
    if issues:
        return True
    trust_confidence = _safe_float(state.get("trust_confidence"))
    if trust_confidence is not None and trust_confidence < 0.7:
        return True
    return False


def _stream_refinement_message(state: dict) -> dict:
    issues = state.get("stream_draft_issues")
    issue_list = issues if isinstance(issues, list) else []
    compact_issues = ", ".join(str(item) for item in issue_list[:5]) or "improve grounding quality"
    return {
        "role": "system",
        "content": (
            "Fast refine pass: keep answer concise, verify major claims against evidence, "
            "fix missing or weak citations, resolve contradictions, and return a clean, readable answer "
            "with direct inline citations and a concise Sources section. "
            "Remove low-signal scaffolding labels (for example: Evidence and caveats, "
            "Claim-by-Claim Citations). "
            f"Primary issues to fix: {compact_issues}."
        ),
    }


async def _refine_stream_draft(
    *,
    user_id: str,
    execution_mode: str,
    messages: list,
    draft: str,
    state: dict,
) -> str:
    if not _should_refine_stream_draft(execution_mode=execution_mode, draft=draft, state=state):
        return draft
    emit_trace_event(
        "fast_refine_started",
        {
            "mode": execution_mode,
            "issues": state.get("stream_draft_issues", []),
            "answer_preview": str(draft)[:220],
        },
    )
    try:
        response = await _call_model_with_fallback(
            messages
            + [
                {"role": "assistant", "content": str(draft)},
                _stream_refinement_message(state),
            ],
            state,
            role="finalizer",
        )
        refined = _extract_guarded_result(
            user_id=user_id, raw_result=response.choices[0].message.content, state=state
        )
        if refined in {"", _NO_RELEVANT_INFORMATION_DETAIL}:
            emit_trace_event(
                "fast_refine_completed",
                {"changed": False, "reason": "refine_result_not_usable"},
            )
            return draft
        changed = str(refined).strip() != str(draft).strip()
        emit_trace_event(
            "fast_refine_completed",
            {
                "changed": changed,
                "issues_before": state.get("stream_draft_issues", []),
                "answer_preview": str(refined)[:220],
            },
        )
        return refined if changed else draft
    except Exception as exc:
        emit_trace_event(
            "fast_refine_failed",
            {
                "error_type": type(exc).__name__,
            },
        )
        logger.warning("Fast refine pass failed; keeping streamed draft. %s", exc)
        return draft


def _cache_not_verified_mentions(text: str) -> int:
    return len(_CACHE_LOW_QUALITY_NOT_VERIFIED_RE.findall(str(text or "")))


def _cache_reject_reason_for_cached_text(cached_text: str) -> str:
    text = str(cached_text or "").strip()
    if not text:
        return "empty_cached_text"
    lowered = text.lower()
    if lowered.startswith(_NO_RELEVANT_INFORMATION_DETAIL.lower()):
        return "cached_no_relevant_information"
    if _NO_RELEVANT_INFORMATION_DETAIL.lower() in lowered:
        return "cached_no_relevant_information_variant"
    if "web retrieval timed out while verifying official sources" in lowered:
        return "cached_web_timeout"
    if text == refusal_response():
        return "cached_guardrail_refusal"
    if _is_generic_placeholder_response(text, {}):
        return "cached_generic_placeholder"
    abstain_like_markers = (
        "no relevant information",
        "insufficient evidence",
        "could not verify",
        "cannot verify",
        "unable to provide",
        "not enough information",
    )
    if any(marker in lowered for marker in abstain_like_markers):
        return "cached_abstain_like"
    max_not_verified_mentions = max(
        1,
        int(getattr(settings.web_search, "cache_max_not_verified_mentions", 3)),
    )
    if _cache_not_verified_mentions(text) > max_not_verified_mentions:
        return "cached_excessive_not_verified_fields"
    return ""


def _cache_skip_reason(result: str, state: dict) -> str:
    text = str(result or "").strip()
    if not text:
        return "empty_result"
    normalized = text.lower()
    if normalized.startswith(_NO_RELEVANT_INFORMATION_DETAIL.lower()):
        return "no_relevant_information"
    if _NO_RELEVANT_INFORMATION_DETAIL.lower() in normalized:
        return "abstain_like_no_relevant_information"
    abstain_like_markers = (
        "no relevant information",
        "insufficient evidence",
        "could not verify",
        "cannot verify",
        "unable to provide",
        "not enough information",
    )
    if any(marker in normalized for marker in abstain_like_markers):
        return "abstain_like_variant"
    if text == refusal_response():
        return "guardrail_refusal"
    output_guard_reason = str(state.get("output_guard_reason", "") or "").strip()
    if output_guard_reason:
        return f"output_guard:{output_guard_reason}"
    context_guard_reason = str(state.get("context_guard_reason", "") or "").strip()
    if context_guard_reason:
        return f"context_guard:{context_guard_reason}"
    if _is_generic_placeholder_response(text, state):
        return "generic_placeholder"
    max_not_verified_mentions = max(
        1,
        int(getattr(settings.web_search, "cache_max_not_verified_mentions", 3)),
    )
    if _cache_not_verified_mentions(text) > max_not_verified_mentions:
        return "excessive_not_verified_fields"

    execution_mode = _normalized_request_mode(str(state.get("execution_mode", _DEEP_MODE)))
    if execution_mode == _DEEP_MODE:
        if bool(state.get("trust_contradiction_flag", False)):
            return "deep_conflict_detected"

        web_fallback_attempted = bool(state.get("web_fallback_attempted", False))
        web_verified = state.get("web_retrieval_verified")
        if web_fallback_attempted and web_verified is False:
            return "deep_web_not_verified"

        confidence = _safe_float(state.get("trust_confidence"))
        min_confidence = float(getattr(settings.web_search, "deep_cache_min_confidence", 0.7))
        if web_fallback_attempted and confidence is not None and confidence < min_confidence:
            return "deep_confidence_low"

        required_coverage = _safe_float(state.get("web_required_field_coverage"))
        min_required_coverage = float(
            getattr(settings.web_search, "deep_cache_min_required_field_coverage", 0.85)
        )
        if (
            web_fallback_attempted
            and required_coverage is not None
            and required_coverage < min_required_coverage
        ):
            return "deep_required_field_coverage_low"

        min_sources = max(1, int(getattr(settings.web_search, "deep_cache_min_source_count", 2)))
        source_count = int(state.get("retrieval_source_count", 0) or 0)
        if web_fallback_attempted and source_count < min_sources:
            return "deep_source_count_low"
    return ""


def _uncertainty_reasons(state: dict) -> list[str]:
    reasons = state.get("trust_uncertainty_reasons")
    if isinstance(reasons, list):
        normalized = [" ".join(str(item).split()).strip() for item in reasons]
        return [item for item in normalized if item][:3]
    derived: list[str] = []
    if bool(state.get("trust_contradiction_flag", False)):
        derived.append("Some sources conflict and require manual verification.")
    confidence = _safe_float(state.get("trust_confidence"))
    if confidence is not None and confidence < 0.42:
        derived.append("Overall evidence confidence is limited.")
    freshness = str(state.get("trust_freshness", "unknown")).strip().lower() or "unknown"
    if _is_freshness_sensitive_query(str(state.get("safe_user_prompt", ""))) and freshness in {
        "stale",
        "unknown",
    }:
        derived.append("Freshness is not strong for a time-sensitive question.")
    if (
        confidence is not None
        and confidence < 0.42
        and int(state.get("retrieval_source_count", 0) or 0) < 2
    ):
        derived.append("Independent source corroboration is limited.")
    web_required_coverage = _safe_float(state.get("web_required_field_coverage"))
    if web_required_coverage is not None and web_required_coverage < 0.999:
        derived.append("Some requested fields are not fully verified from web evidence.")
    return derived[:3]


def _apply_answer_policy(result: str, state: dict) -> str:
    text = str(result or "").replace("\r\n", "\n").strip()
    if not text or text in {_NO_RELEVANT_INFORMATION_DETAIL, refusal_response()}:
        return text
    has_sources_heading = bool(
        re.search(r"(?im)^\s*(?:#{1,3}\s*)?sources?\s*:?\s*$", text)
    )
    has_scaffold_heading = any(
        _LOW_SIGNAL_SCAFFOLD_LINE_RE.match(" ".join(line.split()).strip())
        for line in text.splitlines()
    )
    if not has_scaffold_heading and not has_sources_heading:
        return text

    body, source_block = _split_sources_block(text)
    cleaned_body = _clean_answer_body(body)
    rebuilt = _rebuild_sources_section(cleaned_body, source_block, state)
    return rebuilt.strip()


def _split_sources_block(text: str) -> tuple[str, str]:
    marker = re.search(r"(?im)^\s*(?:#{1,3}\s*)?sources?\s*:?\s*$", str(text or ""))
    if not marker:
        return str(text or "").strip(), ""
    body = str(text or "")[: marker.start()].strip()
    sources = str(text or "")[marker.end() :].strip()
    return body, sources


def _clean_answer_body(body: str) -> str:
    lines = str(body or "").splitlines()
    cleaned: list[str] = []
    seen: set[str] = set()
    blank_open = False
    for raw_line in lines:
        line = str(raw_line).rstrip()
        compact = " ".join(line.split()).strip()
        if not compact:
            if blank_open:
                continue
            cleaned.append("")
            blank_open = True
            continue
        blank_open = False
        if _LOW_SIGNAL_SCAFFOLD_LINE_RE.match(compact):
            continue
        dedupe_key = compact.lower()
        # De-duplicate repeated verbose lines while preserving short list labels.
        if len(compact) > 24 and dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _ordered_unique_urls(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        url = _normalized_url(str(value))
        if not url:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(url)
    return normalized


def _rebuild_sources_section(body: str, source_block: str, state: dict) -> str:
    inline_urls = _ordered_unique_urls(_CITATION_URL_RE.findall(body))
    source_urls = _ordered_unique_urls(_CITATION_URL_RE.findall(source_block))
    evidence_urls = _traceable_urls(state.get("evidence_urls", []), limit=12)
    evidence_urls = _ordered_unique_urls(evidence_urls)

    ordered_urls = _ordered_unique_urls(inline_urls + source_urls + evidence_urls)
    allowed_hosts = _allowed_citation_hosts(evidence_urls)
    if allowed_hosts:
        filtered_urls = [
            url
            for url in ordered_urls
            if _normalized_host_from_url(url) in allowed_hosts
        ]
        if filtered_urls:
            ordered_urls = filtered_urls

    if not ordered_urls:
        return body

    mode = str(state.get("execution_mode", _DEEP_MODE)).strip().lower()
    limit = 8 if mode == _DEEP_MODE else 6
    source_lines = ["Sources"] + [f"- {url}" for url in ordered_urls[:limit]]
    if not body:
        return "\n".join(source_lines)
    return f"{body}\n\n" + "\n".join(source_lines)


def _append_uncertainty_section(result: str, state: dict) -> str:
    return result


def _append_missing_info_section(result: str, state: dict) -> str:
    return result


def _build_structured_comparison_from_evidence(state: dict) -> str:
    entities = state.get("comparison_entities")
    entities = entities if isinstance(entities, list) else []
    entities = [str(item).strip() for item in entities if str(item).strip()][:2]
    if len(entities) < 2:
        return ""

    retrieved_results = state.get("retrieved_results")
    retrieved_results = retrieved_results if isinstance(retrieved_results, list) else []
    if not retrieved_results:
        return ""

    evidence_urls = _traceable_urls(state.get("evidence_urls", []), limit=8)
    if len(evidence_urls) < 2:
        return ""

    def _entity_url_and_snippet(entity: str) -> tuple[str, str]:
        for item in retrieved_results:
            if not _result_mentions_entity(item, entity):
                continue
            metadata = item.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            url = _normalized_url(str(metadata.get("url", ""))) or _normalized_url(
                str(item.get("source_path", ""))
            )
            content = " ".join(str(item.get("content", "")).split()).strip()
            snippet = _SENTENCE_SPLIT_RE.split(content)[0].strip() if content else ""
            if len(snippet) > 180:
                snippet = f"{snippet[:177].rstrip()}..."
            if url:
                return url, snippet
        return "", ""

    used_urls: list[str] = []
    lines: list[str] = [
        f"Comparison: {entities[0]} vs {entities[1]} for English-taught data science master's programs.",
    ]
    for index, entity in enumerate(entities, start=1):
        url, snippet = _entity_url_and_snippet(entity)
        if not url:
            fallback_index = min(index - 1, len(evidence_urls) - 1)
            url = evidence_urls[fallback_index]
        if url and url not in used_urls:
            used_urls.append(url)
        evidence_text = snippet or "Program-level evidence is available on this source page."
        lines.extend(
            [
                f"{entity}:",
                f"- Program details: {evidence_text} ({url})",
                f"- Eligibility requirements: Not verified from evidence. ({url})",
                f"- Application deadline: Not verified from evidence. ({url})",
            ]
        )

    if used_urls:
        lines.extend(
            [
                "Comparison Summary:",
                (
                    f"- Both {entities[0]} and {entities[1]} have relevant source pages, "
                    "but exact eligibility and deadline values remain unverified in retrieved snippets. "
                    f"({used_urls[0]})"
                ),
            ]
        )
    lines.append("Sources:")
    for url in used_urls[:4]:
        lines.append(f"- {url}")
    return "\n".join(lines).strip()


async def _update_memory_with_timing(
    *, conversation_user_id: str, safe_user_prompt: str, result: str, state: dict
) -> None:
    started_at = time.perf_counter()
    try:
        await update_memory(conversation_user_id, safe_user_prompt, result)
    except Exception as exc:
        logger.warning("Memory update failed. %s", exc)
    finally:
        state["memory_update_ms"] = _elapsed_ms(started_at)


async def _write_cache_with_timing(*, cache_key: str, result: str, state: dict) -> None:
    if not _response_cache_enabled():
        state["cache_write_ms"] = 0
        emit_trace_event("cache_write_skipped", {"reason": "response_cache_disabled"})
        return
    started_at = time.perf_counter()
    try:
        skip_reason = _cache_skip_reason(result, state)
        if skip_reason:
            emit_trace_event("cache_write_skipped", {"reason": skip_reason})
            return
        await _redis_call(
            async_redis_client.setex,
            cache_key,
            settings.memory.redis_ttl_seconds,
            result,
        )
    except RedisError as exc:
        logger.warning("Redis cache write failed. %s", exc)
    finally:
        state["cache_write_ms"] = _elapsed_ms(started_at)


def _compute_quality_metrics(*, query: str, answer: str, state: dict) -> dict:
    retrieved_results = state.get("retrieved_results", [])
    retrieved_results = retrieved_results if isinstance(retrieved_results, list) else []
    quality = generation_metrics(
        query=query,
        answer=answer,
        retrieved_results=retrieved_results,
    )
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    claim_stats = _claim_level_citation_stats(answer, evidence_urls)
    state["claim_count"] = int(claim_stats.get("claim_count", 0) or 0)
    state["claim_cited_count"] = int(claim_stats.get("cited_claim_count", 0) or 0)
    state["claim_citation_coverage"] = float(claim_stats.get("coverage", 1.0) or 1.0)
    state["trust_confidence"] = float(_safe_float(state.get("trust_confidence")) or 0.0)
    state["trust_authority_score"] = float(_safe_float(state.get("trust_authority_score")) or 0.0)
    state["trust_agreement_score"] = float(_safe_float(state.get("trust_agreement_score")) or 0.0)
    state["claim_snippet_grounding_coverage"] = float(
        _safe_float(state.get("claim_snippet_grounding_coverage")) or 0.0
    )
    state["claim_snippet_conflict_count"] = int(state.get("claim_snippet_conflict_count", 0) or 0)
    quality["citation_accuracy"] = citation_accuracy_score(answer, evidence_urls)
    quality["source_count"] = float(state.get("retrieval_source_count", 0) or 0)
    quality["confidence"] = float(state.get("trust_confidence", 0.0) or 0.0)
    quality["freshness"] = str(state.get("trust_freshness", "unknown"))
    quality["contradiction_flag"] = bool(state.get("trust_contradiction_flag", False))
    quality["claim_citation_coverage"] = float(state.get("claim_citation_coverage", 1.0) or 1.0)
    quality["claim_snippet_grounding_coverage"] = float(
        state.get("claim_snippet_grounding_coverage", 1.0) or 1.0
    )
    quality["claim_snippet_conflict_count"] = float(
        state.get("claim_snippet_conflict_count", 0.0) or 0.0
    )
    return quality


def _schedule_evaluation_trace(*, user_id: str, user_prompt: str, result: str, state: dict) -> None:
    started_at = time.perf_counter()
    _track_background_task(
        asyncio.create_task(
            _persist_evaluation_trace(
                user_id=user_id,
                prompt=user_prompt,
                answer=result,
                retrieved_results=state["retrieved_results"],
                retrieval_strategy=state["retrieval_strategy"],
                build_context_ms=state["build_context_ms"] or 0,
                retrieval_ms=state["retrieval_ms"] or 0,
                model_ms=state["model_ms"] or 0,
                quality=state["quality"],
                evidence_urls=state.get("evidence_urls", []),
            )
        ),
        label="Evaluation trace persistence",
    )
    state["evaluation_trace_ms"] = _elapsed_ms(started_at)


async def _record_success_metrics(
    *,
    request_id: str,
    started_at: float,
    user_id: str,
    session_id: str,
    user_prompt: str,
    safe_user_prompt: str,
    result: str,
    state: dict,
) -> None:
    await _record_pipeline_stage_metrics(
        build_context_ms=state["build_context_ms"] or 0,
        retrieval_ms=state["retrieval_ms"] or 0,
        model_ms=state["model_ms"] or 0,
        retrieval_strategy=state["retrieval_strategy"],
        retrieved_count=state["retrieved_count"],
    )
    logger.info(
        (
            "ChatPipelineLatency | user=%s | build_context_ms=%s | retrieval_ms=%s "
            "| model_ms=%s | retrieval_strategy=%s | retrieved_count=%s"
        ),
        user_id,
        state["build_context_ms"],
        state["retrieval_ms"],
        state["model_ms"],
        state["retrieval_strategy"],
        state["retrieved_count"],
    )
    await _record_request_outcome(
        request_id=request_id,
        started_at=started_at,
        user_id=user_id,
        session_id=session_id,
        user_prompt=user_prompt,
        safe_user_prompt=safe_user_prompt,
        answer=result,
        outcome="success",
        state=state,
    )


def _new_stream_guard_state() -> dict[str, object]:
    return {"blocked": False, "reason": "", "final_text": ""}


def _guard_stream_text(assembled: str, stream_state: dict[str, object]) -> tuple[str, bool]:
    guarded = guard_model_output(assembled)
    guarded_text = str(guarded.get("text", ""))
    blocked = bool(guarded.get("blocked"))
    reason = str(guarded.get("reason", "blocked_output")) if blocked else ""
    stream_state["blocked"] = blocked
    stream_state["reason"] = reason
    stream_state["final_text"] = guarded_text
    return guarded_text, blocked


def _iter_stream_pieces(text: str, size: int):
    for start in range(0, len(text), size):
        piece = text[start : start + size]
        if piece:
            yield piece


def _stable_stream_text(guarded_text: str, emitted: str, holdback_chars: int) -> str | None:
    stable_len = len(guarded_text) - holdback_chars
    if stable_len <= 0:
        return None
    stable = guarded_text[:stable_len]
    if stable and stable != emitted:
        return stable
    return None


async def _yield_blocked_tail(
    *,
    guarded_text: str,
    emitted: str,
) -> AsyncIterator[str]:
    if guarded_text and guarded_text != emitted:
        yield guarded_text


async def _yield_stable_delta(
    *,
    guarded_text: str,
    emitted: str,
    holdback_chars: int,
) -> AsyncIterator[str]:
    stable = _stable_stream_text(guarded_text, emitted, holdback_chars)
    if stable is not None:
        yield stable


async def _iter_guarded_updates(
    *,
    text: str,
    size: int,
    stream_state: dict[str, object],
    emitted: str,
    holdback_chars: int,
    delay_seconds: float,
) -> AsyncIterator[tuple[str, bool]]:
    assembled = str(stream_state.get("_assembled", ""))
    for piece in _iter_stream_pieces(text, size):
        assembled += piece
        stream_state["_assembled"] = assembled
        guarded_text, blocked = _guard_stream_text(assembled, stream_state)
        if blocked:
            async for blocked_text in _yield_blocked_tail(
                guarded_text=guarded_text,
                emitted=emitted,
            ):
                yield blocked_text, True
            yield emitted, True
            return
        async for stable in _yield_stable_delta(
            guarded_text=guarded_text,
            emitted=emitted,
            holdback_chars=holdback_chars,
        ):
            emitted = stable
            yield emitted, False
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)


async def _yield_guarded_stream(
    delta_stream: AsyncIterator[str],
    *,
    stream_state: dict[str, object],
    chunk_size: int,
    chunk_delay_ms: int,
) -> AsyncIterator[str]:
    emitted = ""
    size = max(1, int(chunk_size))
    delay_seconds = max(0.0, float(chunk_delay_ms) / 1000.0)
    holdback_chars = max(0, _STREAM_GUARD_HOLDBACK_CHARS)
    stream_state["_assembled"] = ""

    async for delta in delta_stream:
        text = str(delta or "")
        if not text:
            continue
        async for value, blocked in _iter_guarded_updates(
            text=text,
            size=size,
            stream_state=stream_state,
            emitted=emitted,
            holdback_chars=holdback_chars,
            delay_seconds=delay_seconds,
        ):
            if value and value != emitted:
                emitted = value
                yield emitted
            if blocked:
                return

    guarded_text, _blocked = _guard_stream_text(
        str(stream_state.get("_assembled", "")), stream_state
    )
    if guarded_text and guarded_text != emitted:
        yield guarded_text


async def _emit_guarded_stream(
    delta_stream: AsyncIterator[str],
    *,
    runtime: dict[str, object],
    stream_state: dict[str, object],
    chunk_size: int,
    chunk_delay_ms: int,
) -> AsyncIterator[str]:
    async for partial in _yield_guarded_stream(
        delta_stream,
        stream_state=stream_state,
        chunk_size=chunk_size,
        chunk_delay_ms=chunk_delay_ms,
    ):
        runtime["streamed_text"] = partial
        yield partial


async def _stream_model_with_fallback(
    *,
    messages: list,
    state: dict,
    runtime: dict[str, object],
    chunk_size: int,
    chunk_delay_ms: int,
) -> AsyncIterator[str]:
    model_started_at = time.perf_counter()
    emit_trace_event("model_stream_started", {"mode": "primary"})
    try:
        async for partial in _emit_guarded_stream(
            _stream_primary(messages),
            runtime=runtime,
            stream_state=runtime["stream_guard_state"],
            chunk_size=chunk_size,
            chunk_delay_ms=chunk_delay_ms,
        ):
            yield partial
    except Exception as primary_exc:
        state["used_fallback_model"] = True
        emit_trace_event(
            "model_stream_fallback_started",
            {"error_type": type(primary_exc).__name__},
        )
        logger.warning("Primary model stream failed; attempting fallback. %s", primary_exc)
        if runtime["streamed_text"]:
            state["model_ms"] = _elapsed_ms(model_started_at)
            raise
        runtime["stream_guard_state"] = _new_stream_guard_state()
        try:
            async for partial in _emit_guarded_stream(
                _stream_fallback(messages),
                runtime=runtime,
                stream_state=runtime["stream_guard_state"],
                chunk_size=chunk_size,
                chunk_delay_ms=chunk_delay_ms,
            ):
                yield partial
        except Exception:
            logger.exception("Fallback model stream also failed.")
            state["model_ms"] = _elapsed_ms(model_started_at)
            raise
    state["model_ms"] = _elapsed_ms(model_started_at)
    emit_trace_event(
        "model_stream_completed",
        {
            "model_ms": state["model_ms"],
            "used_fallback": bool(state.get("used_fallback_model", False)),
        },
    )


def _new_request_context(
    user_id: str, user_prompt: str, session_id: str | None, requested_mode: str
) -> dict:
    effective_session_id = _resolve_session_id(user_id, session_id)
    return {
        "started_at": time.perf_counter(),
        "request_id": uuid4().hex,
        "user_id": user_id,
        "user_prompt": user_prompt,
        "effective_session_id": effective_session_id,
        "conversation_user_id": _conversation_user_id(user_id, effective_session_id),
        "safe_user_prompt": user_prompt,
        "requested_mode": _normalized_request_mode(requested_mode),
        "execution_mode": _DEFAULT_EXECUTION_MODE,
        "cache_key": "",
        "state": _new_metrics_state(),
    }


async def _record_context_outcome(
    context: dict,
    *,
    answer: str,
    outcome: str,
    error_message: str = "",
) -> None:
    await _record_request_outcome(
        request_id=str(context["request_id"]),
        started_at=float(context["started_at"]),
        user_id=str(context["user_id"]),
        session_id=str(context["effective_session_id"]),
        user_prompt=str(context["user_prompt"]),
        safe_user_prompt=str(context["safe_user_prompt"]),
        answer=answer,
        outcome=outcome,
        state=context["state"],
        error_message=error_message,
    )


async def _prepare_request(
    user_id: str,
    user_prompt: str,
    session_id: str | None,
    mode: str = _DEFAULT_EXECUTION_MODE,
) -> tuple[dict, list | None, str | None]:
    context = _new_request_context(user_id, user_prompt, session_id, mode)
    state = context["state"]
    state["requested_mode"] = str(context["requested_mode"])
    emit_trace_event(
        "request_received",
        {
            "user_id": user_id,
            "session_id": str(context["effective_session_id"]),
            "prompt_preview": str(user_prompt)[:260],
            "requested_mode": str(context["requested_mode"]),
        },
    )
    input_guard = guard_user_input(user_id, user_prompt)
    if input_guard["blocked"]:
        state["input_guard_reason"] = str(input_guard.get("reason", "blocked_input"))
        refusal = refusal_response()
        logger.info(
            "GuardrailDecision | stage=input | user=%s | blocked=true | reason=%s",
            user_id,
            state["input_guard_reason"],
        )
        await _record_context_outcome(context, answer=refusal, outcome="blocked_input")
        emit_trace_event(
            "request_blocked_input",
            {
                "reason": state["input_guard_reason"],
            },
        )
        return context, None, refusal

    context["safe_user_prompt"] = str(input_guard.get("sanitized_text", user_prompt))
    state["safe_user_prompt"] = str(context["safe_user_prompt"])
    context["execution_mode"] = _resolve_initial_execution_mode(
        str(context["requested_mode"]),
        str(context["safe_user_prompt"]),
    )
    state["execution_mode"] = str(context["execution_mode"])
    state["agentic_enabled"] = bool(str(context["execution_mode"]) == _DEEP_MODE)
    policy = _execution_policy(str(context["execution_mode"]))
    context["cache_key"] = _chat_cache_key(
        user_id,
        str(context["safe_user_prompt"]),
        str(context["effective_session_id"]),
        str(context["requested_mode"]),
    )
    if _response_cache_enabled():
        cached, state["cache_read_ms"] = await _read_cached_response(str(context["cache_key"]))
        if cached:
            cached_text = str(cached)
            reject_reason = _cache_reject_reason_for_cached_text(cached_text)
            if reject_reason:
                emit_trace_event(
                    "cache_hit_rejected",
                    {
                        "reason": reject_reason,
                        "answer_preview": cached_text[:260],
                    },
                )
                try:
                    await _redis_call(async_redis_client.delete, str(context["cache_key"]))
                except RedisError as exc:
                    logger.warning("Redis cache delete failed. %s", exc)
            else:
                await _record_context_outcome(context, answer=cached_text, outcome="cache_hit")
                emit_trace_event(
                    "cache_hit",
                    {
                        "answer_preview": cached_text[:260],
                    },
                )
                return context, None, cached_text
    else:
        state["cache_read_ms"] = 0
        emit_trace_event("cache_bypassed", {"reason": "response_cache_disabled"})

    messages, refusal = await _prepare_messages_for_model(
        user_id=user_id,
        conversation_user_id=str(context["conversation_user_id"]),
        safe_user_prompt=str(context["safe_user_prompt"]),
        execution_mode=str(context["execution_mode"]),
        policy=policy,
        state=state,
    )
    if (
        refusal in {_NO_RELEVANT_INFORMATION_DETAIL, _WEB_RETRIEVAL_TIMEOUT_DETAIL}
        and str(context["requested_mode"]) == _AUTO_MODE
        and str(context["execution_mode"]) == _FAST_MODE
    ):
        emit_trace_event(
            "mode_auto_escalation_started",
            {
                "from_mode": _FAST_MODE,
                "reason": "fast_context_gate",
                "context_reason": str(state.get("context_guard_reason", "")),
            },
        )
        state["auto_escalated"] = True
        context["execution_mode"] = _DEEP_MODE
        state["execution_mode"] = _DEEP_MODE
        state["agentic_enabled"] = True
        state["context_guard_reason"] = ""
        deep_policy = _execution_policy(_DEEP_MODE)
        messages, refusal = await _prepare_messages_for_model(
            user_id=user_id,
            conversation_user_id=str(context["conversation_user_id"]),
            safe_user_prompt=str(context["safe_user_prompt"]),
            execution_mode=_DEEP_MODE,
            policy=deep_policy,
            state=state,
        )
        emit_trace_event(
            "mode_auto_escalation_completed",
            {
                "to_mode": _DEEP_MODE,
                "escalated_for_context": True,
                "success": refusal is None,
                "context_reason": str(state.get("context_guard_reason", "")),
            },
        )
    if refusal is not None:
        await _record_context_outcome(context, answer=refusal, outcome="blocked_context")
        emit_trace_event(
            "request_blocked_context",
            {
                "reason": state["context_guard_reason"],
            },
        )
        return context, None, refusal
    return context, messages, None


async def _finalize_success(context: dict, result: str) -> str:
    state = context["state"]
    result = _apply_answer_policy(result, state)
    result = _append_uncertainty_section(result, state)
    result = _append_missing_info_section(result, state)
    state["abstain_reason"] = _derive_abstain_reason(result, state)
    state["quality"] = _compute_quality_metrics(
        query=str(context["safe_user_prompt"]),
        answer=result,
        state=state,
    )
    await _update_memory_with_timing(
        conversation_user_id=str(context["conversation_user_id"]),
        safe_user_prompt=str(context["safe_user_prompt"]),
        result=result,
        state=state,
    )
    await _write_cache_with_timing(
        cache_key=str(context["cache_key"]),
        result=result,
        state=state,
    )
    _schedule_evaluation_trace(
        user_id=str(context["user_id"]),
        user_prompt=str(context["user_prompt"]),
        result=result,
        state=state,
    )
    await _record_success_metrics(
        request_id=str(context["request_id"]),
        started_at=float(context["started_at"]),
        user_id=str(context["user_id"]),
        session_id=str(context["effective_session_id"]),
        user_prompt=str(context["user_prompt"]),
        safe_user_prompt=str(context["safe_user_prompt"]),
        result=result,
        state=state,
    )
    emit_trace_event(
        "answer_finalized",
        {
            "answer_preview": result[:260],
            "source_urls": _traceable_urls(state.get("evidence_urls", [])),
            "agent_rounds": int(state.get("agent_rounds", 0) or 0),
            "confidence": state.get("trust_confidence"),
            "freshness": state.get("trust_freshness"),
            "contradiction_flag": bool(state.get("trust_contradiction_flag", False)),
            "authority_score": state.get("trust_authority_score"),
            "agreement_score": state.get("trust_agreement_score"),
            "claim_citation_coverage": state.get("claim_citation_coverage"),
            "claim_snippet_grounding_coverage": state.get("claim_snippet_grounding_coverage"),
            "claim_snippet_conflict_count": int(state.get("claim_snippet_conflict_count", 0) or 0),
            "claim_evidence_map": (state.get("claim_evidence_map") or [])[:3],
            "uncertainty_reasons": state.get("trust_uncertainty_reasons", []),
            "web_retrieval_verified": state.get("web_retrieval_verified"),
            "web_required_field_coverage": state.get("web_required_field_coverage"),
            "web_required_fields_missing": (state.get("web_required_fields_missing") or [])[:4],
            "abstain_reason": str(state.get("abstain_reason", "")),
            "required_field_coverage": state.get("required_field_coverage"),
            "required_fields_missing": (state.get("required_fields_missing") or [])[:4],
        },
    )
    return result


async def generate_response(
    user_id: str,
    user_prompt: str,
    session_id: str | None = None,
    mode: str = _DEFAULT_EXECUTION_MODE,
) -> str:
    """Run the full chat pipeline: guardrails, memory, model call, cache, and persistence."""
    context, messages, early_answer = await _prepare_request(
        user_id,
        user_prompt,
        session_id,
        mode=mode,
    )
    if early_answer is not None:
        return early_answer
    state = context["state"]
    policy = _execution_policy(str(context["execution_mode"]))
    policy["auto_requested"] = str(context.get("requested_mode", _DEEP_MODE)) == _AUTO_MODE

    try:
        result, llm_usage = await _generate_agentic_answer(
            user_id=user_id,
            messages=messages,
            policy=policy,
            state=state,
        )
        if (
            bool(policy.get("auto_requested"))
            and not bool(state.get("auto_escalated", False))
            and str(context.get("execution_mode", _DEEP_MODE)) == _FAST_MODE
            and _should_escalate_auto_to_deep(result=result, state=state)
        ):
            emit_trace_event(
                "mode_auto_escalation_started",
                {
                    "from_mode": str(context["execution_mode"]),
                    "reason": "fast_quality_gate",
                    "issues": state.get("agent_last_issues", []),
                },
            )
            state["auto_escalated"] = True
            context["execution_mode"] = _DEEP_MODE
            state["execution_mode"] = _DEEP_MODE
            state["agentic_enabled"] = True
            state["output_guard_reason"] = ""
            state["agent_last_issues"] = []
            deep_policy = _execution_policy(_DEEP_MODE)
            deep_result, deep_usage = await _generate_agentic_answer(
                user_id=user_id,
                messages=messages,
                policy=deep_policy,
                state=state,
            )
            result = deep_result
            llm_usage = _merge_llm_usage(llm_usage, deep_usage)
            emit_trace_event(
                "mode_auto_escalation_completed",
                {
                    "to_mode": _DEEP_MODE,
                    "answer_preview": str(result)[:260],
                },
            )
    except Exception as exc:
        await _record_context_outcome(
            context,
            answer="",
            outcome="model_error",
            error_message=str(exc),
        )
        raise

    state["llm_usage"] = llm_usage
    if state["llm_usage"]:
        logger.info(
            "LLMUsage | user=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
            user_id,
            state["llm_usage"]["prompt_tokens"],
            state["llm_usage"]["completion_tokens"],
            state["llm_usage"]["total_tokens"],
        )
    result = await _finalize_success(context, result)
    return result


async def generate_response_stream(
    user_id: str,
    user_prompt: str,
    session_id: str | None = None,
    mode: str = _DEFAULT_EXECUTION_MODE,
    *,
    chunk_size: int = 120,
    chunk_delay_ms: int = 12,
) -> AsyncIterator[str]:
    """Stream true model output from Bedrock and yield progressively assembled text."""
    context, messages, early_answer = await _prepare_request(
        user_id,
        user_prompt,
        session_id,
        mode=mode,
    )
    if early_answer is not None:
        yield early_answer
        return
    state = context["state"]
    execution_mode = str(context.get("execution_mode", _DEFAULT_EXECUTION_MODE))

    runtime: dict[str, object] = {
        "streamed_text": "",
        "stream_guard_state": _new_stream_guard_state(),
    }
    try:
        async for partial in _stream_model_with_fallback(
            messages=messages,
            state=state,
            runtime=runtime,
            chunk_size=chunk_size,
            chunk_delay_ms=chunk_delay_ms,
        ):
            yield partial
    except Exception as exc:
        await _record_context_outcome(
            context,
            answer=str(runtime["streamed_text"]),
            outcome="model_error",
            error_message=str(exc),
        )
        raise

    stream_guard_state = runtime["stream_guard_state"]
    result = str(
        stream_guard_state.get("final_text", runtime["streamed_text"]) or runtime["streamed_text"]
    )
    if bool(stream_guard_state.get("blocked")):
        state["output_guard_reason"] = str(stream_guard_state.get("reason", "blocked_output"))
        logger.info(
            "GuardrailDecision | stage=output | user=%s | blocked=true | reason=%s",
            user_id,
            state["output_guard_reason"],
        )
    grounded_result = _enforce_citation_grounding(result, state)
    if grounded_result != result:
        result = grounded_result
        if result != str(runtime["streamed_text"]):
            yield result

    refined_result = await _refine_stream_draft(
        user_id=user_id,
        execution_mode=execution_mode,
        messages=messages,
        draft=result,
        state=state,
    )
    if refined_result != result:
        result = refined_result
        if result != str(runtime["streamed_text"]):
            yield result

    stream_issues = _agentic_result_issues(result, state)
    state["agent_last_issues"] = stream_issues
    emit_trace_event(
        "answer_verification_completed",
        {
            "round": 1,
            "verified": not stream_issues,
            "issues": stream_issues,
            "base_issues": stream_issues,
            "verifier": {},
            "mode": execution_mode,
            "verifier_skipped": True,
            "stream_mode": True,
        },
    )
    if stream_issues and result != _NO_RELEVANT_INFORMATION_DETAIL:
        if _is_hard_verification_failure(stream_issues, result, state):
            state["output_guard_reason"] = "stream_verification_failed"
            result = _NO_RELEVANT_INFORMATION_DETAIL
            if result != str(runtime["streamed_text"]):
                yield result
        else:
            state["output_guard_reason"] = "stream_verification_partial"
            emit_trace_event(
                "answer_partial_with_evidence",
                {
                    "issues": stream_issues[:8],
                    "answer_preview": str(result)[:220],
                },
            )

    if int(state.get("agent_rounds", 0) or 0) <= 0:
        state["agent_rounds"] = 1
    if not isinstance(state.get("agent_last_issues"), list):
        state["agent_last_issues"] = []
    finalized_result = await _finalize_success(context, result)
    if finalized_result != result:
        yield finalized_result
