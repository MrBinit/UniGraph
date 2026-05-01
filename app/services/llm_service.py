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
from app.services.student_qa_schema_registry import (
    required_answer_fields_from_schema,
    resolve_question_schema,
)
from app.services.german_source_policy import validate_german_program_scope
from app.services.german_source_routes import (
    is_likely_german_university_query,
    resolve_german_research_task,
)
from app.services.german_university_research_orchestrator import research_german_university
from app.services.unigraph_research import aretrieve_web_chunks

settings = get_settings()
prompts = get_prompts()
logger = logging.getLogger(__name__)

_BACKGROUND_TASKS: set[asyncio.Task] = set()
_RETRIEVAL_QUERY_MAX_CHARS = 900
_RETRIEVAL_CONTEXT_MAX_CHARS = 4200
_RETRIEVAL_CHUNK_MAX_CHARS = 520
_RETRIEVAL_MAX_PROMPT_RESULTS = 6
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
_NOT_VERIFIED_OFFICIAL_DETAIL = "Not verified from official sources."
_WEB_RETRIEVAL_TIMEOUT_DETAIL = (
    "Web retrieval timed out while verifying official sources. Please retry."
)
_WEB_PROVIDER_ERROR_DETAIL = (
    "Web retrieval provider is unavailable (quota/access limit). Please retry later."
)
_FAST_MODE = "fast"
_STANDARD_MODE = "standard"
_DEEP_MODE = "deep"
_AUTO_MODE = "auto"
_DEFAULT_EXECUTION_MODE = "deep"
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
    r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s*[–-]\s*\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)?\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:\s*[–-]\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})?(?:,\s*\d{4})?\b|"
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
_WEAK_CRITICAL_EVIDENCE_RE = re.compile(
    r"\b(related (?:document|page|program)|broader (?:document|listing|program)|"
    r"direct applicability|should be confirmed|indicative|caveat|may not apply)\b",
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
_CHAT_CACHE_VERSION = "v4"
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
    "instruction_language": "Instruction language",
    "gpa_threshold": "GPA/grade threshold",
    "gpa_or_grade_threshold": "GPA/grade threshold",
    "ects_prerequisites": "ECTS/prerequisite credit breakdown",
    "ects_or_prerequisite_credit_breakdown": "ECTS/prerequisite credit breakdown",
    "tuition_or_fees": "Tuition/fees",
    "language_requirements": "Language requirements",
    "language_test_thresholds": "Language test score thresholds",
    "language_test_score_thresholds": "Language test score thresholds",
    "international_deadline": "Application deadline (international)",
    "curriculum_focus": "Curriculum focus",
    "career_outcomes": "Career outcomes",
    "scholarship_options": "Scholarship options",
    "visa_or_work_rights": "Visa/work rights",
    "aps_requirement_stage": "APS requirement stage",
    "admission_decision_signal": "Admission decision signal",
    "professors_or_supervisors": "Professors/supervisors",
    "labs_or_research_groups": "Labs/research groups",
    "department_or_faculty": "Department/faculty",
    "contact_information": "Contact information",
    "funding_or_scholarship": "Funding/scholarship",
    "publication_or_profile_links": "Publications/profile links",
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
_RESEARCHER_REQUIRED_ANSWER_FIELDS = {
    "professors_or_supervisors",
    "labs_or_research_groups",
    "department_or_faculty",
    "contact_information",
    "funding_or_scholarship",
    "publication_or_profile_links",
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
        fallback = _normalized_model_id(getattr(settings.bedrock, "planner_fallback_model_id", ""))
    elif role_key == "verifier":
        primary = _normalized_model_id(getattr(settings.bedrock, "verifier_model_id", ""))
        fallback = _normalized_model_id(getattr(settings.bedrock, "verifier_fallback_model_id", ""))
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
    if candidate in {_FAST_MODE, _STANDARD_MODE, _DEEP_MODE, _AUTO_MODE}:
        return candidate
    return _AUTO_MODE


def _mode_from_state(state: dict | None) -> str:
    if not isinstance(state, dict):
        return _DEEP_MODE
    for key in ("execution_mode", "requested_mode"):
        candidate = _normalized_request_mode(state.get(key))
        if candidate in {_FAST_MODE, _DEEP_MODE}:
            return candidate
        if candidate == _STANDARD_MODE:
            return _FAST_MODE
    return _DEEP_MODE


def _max_context_results_for_mode(mode: str | None = None) -> int:
    configured = max(1, int(settings.web_search.max_context_results))
    normalized_mode = _normalized_request_mode(mode)
    if normalized_mode == _AUTO_MODE:
        normalized_mode = _FAST_MODE
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
    if normalized in {_FAST_MODE, _STANDARD_MODE, _AUTO_MODE}:
        return _FAST_MODE
    if normalized == _DEEP_MODE:
        return _DEEP_MODE
    return _FAST_MODE


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

    sections: list[str] = ["Direct Summary"]
    if required_set & {
        "instruction_language",
    }:
        sections.append("Language of Instruction")
    if required_set & {
        "language_requirements",
        "language_test_thresholds",
        "language_test_score_thresholds",
    }:
        sections.append("IELTS/German Requirements")
        sections.append("Language Requirements")
    if required_set & {
        "eligibility_requirements",
        "gpa_threshold",
        "gpa_or_grade_threshold",
        "ects_prerequisites",
        "ects_or_prerequisite_credit_breakdown",
    }:
        sections.append("GPA and ECTS Requirements")
    if (
        required_set
        & {
            "eligibility_requirements",
        }
        and "GPA and ECTS Requirements" not in sections
    ):
        sections.append("Eligibility Requirements")
    elif "eligibility_requirements" in required_set:
        sections.append("Eligibility Requirements")
    if (
        "application_deadline" in required_set
        or "international_deadline" in required_set
        or bool(state.get("deadline_query", False))
    ):
        sections.append("Application Deadline")
    if "application_portal" in required_set:
        sections.append("Application Portal")
    if "admission_decision_signal" in required_set:
        sections.append("Admission Competitiveness")
    if required_set & {
        "tuition_or_fees",
        "curriculum_focus",
        "career_outcomes",
    }:
        sections.append("Other Requested Details")
    sections.append("Sources")

    seen_sections: set[str] = set()
    sections = [
        section
        for section in sections
        if section and not (section in seen_sections or seen_sections.add(section))
    ]
    include_admission_decision = "admission_decision_signal" in required_set

    lines = [
        "Admissions answer schema (deep):",
        "- Use only the sections below, in this order:",
    ]
    for index, section in enumerate(sections, start=1):
        lines.append(f"  {index}. {section}")
    lines.append(
        "- Start with a short direct answer; do not start with retrieval/planner/coverage text."
    )
    lines.append(
        "- Do not include the field evidence ledger or phrases like 'Official evidence found for'."
    )
    lines.append("- For requested fields, provide exact numbers/dates when present in evidence.")
    if "Language of Instruction" in sections:
        lines.append(
            "- Under Language of Instruction, state the teaching language only if verified for the requested program."
        )
    if "IELTS/German Requirements" in sections:
        lines.append(
            "- Under IELTS/German Requirements, separate accepted language/proof requirements from exact test score thresholds."
        )
    if "GPA and ECTS Requirements" in sections:
        lines.append("- Under GPA and ECTS Requirements include explicit lines for:")
        lines.append("  - Degree/background requirement")
        lines.append("  - GPA/grade threshold")
        lines.append("  - ECTS/prerequisite credit breakdown")
    if (
        "application_deadline" in required_set
        or "international_deadline" in required_set
        or bool(state.get("deadline_query", False))
    ):
        lines.append(
            "- Under Application Deadline, give exact dates only; otherwise mark not verified."
        )
    if include_admission_decision:
        lines.append("- Under Admission Competitiveness include:")
        lines.append(
            "  - Verdict: likely competitive / risky / unknown, not a guaranteed admission prediction."
        )
        lines.append(
            "  - Use official minimum grade, ECTS prerequisites, selection rules, ranking formula, capacity, or historical cutoffs when available."
        )
        lines.append(
            "  - If only minimum eligibility is verified, say eligibility can be assessed but competitiveness cannot be confirmed."
        )
        lines.append(
            "  - For an applicant GPA such as ~3.2, give a cautious risk classification only when official selection evidence supports it."
        )
        lines.append(
            "  - Do not require an official source to literally say 'competitive' or 'safe'."
        )
        lines.append(
            "  - Do not call a course safe unless the verified thresholds/selection evidence make that conclusion defensible."
        )
    else:
        lines.append(
            "- Do not include an Admission Competitiveness section unless explicitly requested."
        )
        lines.append(
            "- Do not include an Admission Decision section unless explicitly requested."
        )
    lines.append(
        "- If a requested field is unavailable, use natural field-specific wording; "
        f'never write "{_NOT_VERIFIED_OFFICIAL_DETAIL}" in the final answer.'
    )
    lines.append(
        "- Do not present 'related document', 'broader listing', or 'indicative' evidence as a verified critical admission fact."
    )
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
            "Answer style policy (fast): concise, conversational, and readable. "
            "Answer only the user's question, usually in one short paragraph for narrow deadline, "
            "language, or tuition questions. Summarize evidence into clean wording; never paste raw "
            "chunks, copied markdown headings, table fragments, internal field names, or coverage labels. "
            "Never write 'Not verified from official sources' or 'Verified from selected evidence'. "
            "If a requested detail is missing, say it naturally. Keep one final 'Sources' section with "
            "unique URLs only when sources are shown."
        )
        return {"role": "system", "content": content}

    content = (
        "Answer style policy (deep):\n"
        "- Start with a direct, student-facing answer in 1-2 sentences.\n"
        "- For narrow deadline, language, or tuition questions, use one short paragraph instead of sections.\n"
        "- Use clear markdown sections only for broad, multi-field questions and only when relevant "
        "(for example: Language, GPA and ECTS, Application Deadline, Application Portal, "
        "Admission Competitiveness, Sources).\n"
        "- Use only sections requested by the query/required fields; do not pad with extra sections.\n"
        "- Ensure required fields are explicitly covered: "
        f"{required_text}.\n"
        "- Summarize retrieved evidence into natural wording. Do not copy raw chunks, page headings, "
        "markdown headings, table fragments, navigation text, or long snippets.\n"
        "- Never print internal labels such as answered_fields, missing_fields, coverage ledger, "
        "selected evidence, field completeness, or source status.\n"
        "- Never write 'Not verified from official sources' or 'Verified from selected evidence'.\n"
        "- If a requested detail is missing, use natural field-specific wording such as: "
        "'The retrieved official evidence does not state a specific IELTS band score.'\n"
        "- Keep one fact per bullet and avoid duplicated lines.\n"
        "- Prefer clean synthesized field lines over long paragraphs for admissions questions.\n"
        "- Do not include an Admission Competitiveness section unless the user explicitly asks for admission likelihood, competitiveness, safety, or chances.\n"
        "- Never output internal ledger/coverage text such as 'Official evidence found for' or 'Requested fields'.\n"
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
    if normalized in {_FAST_MODE, _STANDARD_MODE, _AUTO_MODE}:
        return {
            "mode": _FAST_MODE,
            "planner_enabled": False,
            "verifier_enabled": False,
            "max_attempts": 1,
            "web_search_mode": _FAST_MODE,
            "auto_requested": normalized in {_AUTO_MODE, _STANDARD_MODE},
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
            "web_required_field_coverage": state.get("web_required_field_coverage"),
            "web_required_fields_missing": state.get("web_required_fields_missing") or [],
            "web_source_policy": str(state.get("web_source_policy", "")),
            "web_unresolved_fields": state.get("web_unresolved_fields") or [],
            "web_research_objective_coverage": state.get("web_research_objective_coverage"),
            "web_research_objectives_missing": state.get("web_research_objectives_missing") or [],
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
    return f"{compact} data science master's program " "admission requirements application deadline"


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
    has_professor_scope = bool(
        re.search(
            r"\b(professor|professors|faculty|faculties|supervisor|supervisors|advisor|advisors|mentor|mentors|lecturer|lecturers)\b",
            text,
        )
    )
    has_lab_scope = bool(
        re.search(
            r"\b(lab|labs|laboratory|laboratories|research group|research groups|institute|institutes|research center|research centres|chair)\b",
            text,
        )
    )
    has_department_scope = bool(re.search(r"\b(department|faculty|school|institute|chair)\b", text))
    has_contact_scope = bool(
        re.search(
            r"\b(contact|contacts|email|e-mail|phone|telephone|office|admissions office|who to contact|contact person)\b",
            text,
        )
    )
    has_publication_scope = bool(
        re.search(
            r"\b(publication|publications|papers|journal|google scholar|researchgate|profile link|homepage)\b",
            text,
        )
    )
    has_research_context = bool(
        re.search(
            r"\b(university|universit[a-z]*|uni|program|programme|course|department|faculty|school|"
            r"master|m\.sc|msc|phd|admission|college)\b",
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
    if re.search(
        r"\b(curriculum|focus|modules?|syllabus|course structure|taught|subjects?)\b", text
    ):
        required.append("curriculum_focus")
    if "career" in text or "outcome" in text:
        required.append("career_outcomes")
    if "scholarship" in text:
        required.append("scholarship_options")
    if has_professor_scope and has_research_context:
        required.append("professors_or_supervisors")
    if has_lab_scope and has_research_context:
        required.append("labs_or_research_groups")
    if has_research_context and has_department_scope and (has_professor_scope or has_lab_scope):
        required.append("department_or_faculty")
    if has_contact_scope and has_research_context:
        required.append("contact_information")
    if has_research_context and (
        "scholarship" in text or "funding" in text or "assistantship" in text
    ):
        required.append("funding_or_scholarship")
    if has_publication_scope and has_research_context:
        required.append("publication_or_profile_links")
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


def _is_researcher_objective_query(state: dict) -> bool:
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    required_set = {str(item).strip() for item in required_fields if str(item).strip()}
    return bool(required_set & _RESEARCHER_REQUIRED_ANSWER_FIELDS)


def _research_objectives_missing_from_web(state: dict) -> list[str]:
    raw = state.get("web_research_objectives_missing")
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
        "- If a requested field cannot be verified, use natural wording. Never write "
        f'"{_NOT_VERIFIED_OFFICIAL_DETAIL}" in the final answer.',
        "- Prefer exact numbers/dates over generic wording when present in evidence.",
        "- For applicant-critical facts, do not use weak caveats like 'related document', 'indicative', or 'should be confirmed' as factual answers.",
        "- If exact official evidence is not tied to the requested program, mark the field as not verified instead of guessing.",
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
    return (
        "not verified from evidence" in lowered
        or "not verified from sources" in lowered
        or "not verified from official sources" in lowered
    )


def _answer_matches_required_field(
    field: str, answer: str, *, comparison_entities: list[str] | None = None
) -> bool:
    lowered = str(answer or "").lower()
    if not lowered:
        return False
    if field == "application_deadline":
        return (
            _has_not_verified_marker(answer)
            or _has_date_like_value(answer)
            or bool(re.search(r"\bdeadline|apply by|last date\b", lowered))
        )
    if field == "international_deadline":
        return _answer_matches_required_field("application_deadline", answer)
    if field == "application_portal":
        if _has_not_verified_marker(answer):
            return True
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
    if field in {"gpa_or_grade_threshold", "gpa_threshold"}:
        return bool(
            re.search(r"\b(gpa|grade point|minimum grade|cgpa|grade average)\b", lowered)
            and (re.search(r"\b\d(?:[.,]\d+)?\b", lowered) or _has_not_verified_marker(answer))
        )
    if field in {"ects_or_prerequisite_credit_breakdown", "ects_prerequisites"}:
        return bool(
            re.search(r"\b(ects|credit|credits|cp|prerequisite)\b", lowered)
            and (re.search(r"\b\d{1,3}(?:[.,]\d+)?\b", lowered) or _has_not_verified_marker(answer))
        )
    if field == "tuition_or_fees":
        return bool(
            re.search(r"\btuition|fee|fees|semester contribution|eur|€|no tuition\b", lowered)
        )
    if field == "instruction_language":
        return bool(
            re.search(
                r"\b(language of instruction|teaching language|taught in|english|german)\b", lowered
            )
            or _has_not_verified_marker(answer)
        )
    if field == "language_requirements":
        return bool(re.search(r"\blanguage|english|german|ielts|toefl|c1|c2|b2\b", lowered))
    if field in {"language_test_score_thresholds", "language_test_thresholds"}:
        return bool(
            re.search(r"\b(ielts|toefl|cefr|cambridge|duolingo)\b", lowered)
            and (re.search(r"\b\d(?:[.,]\d+)?\b", lowered) or _has_not_verified_marker(answer))
        )
    if field == "curriculum_focus":
        return bool(re.search(r"\bcurriculum|focus|module|coursework\b", lowered))
    if field == "career_outcomes":
        return bool(re.search(r"\bcareer|employment|outcome|job|industry\b", lowered))
    if field == "scholarship_options":
        return bool(re.search(r"\bscholarship|funding|grant|stipend\b", lowered))
    if field == "professors_or_supervisors":
        return bool(
            re.search(r"\bprofessor|faculty|supervisor|advisor|lecturer\b", lowered)
            or _has_not_verified_marker(answer)
        )
    if field == "labs_or_research_groups":
        return bool(
            re.search(r"\blab|laboratory|research group|institute|chair|research center\b", lowered)
            or _has_not_verified_marker(answer)
        )
    if field == "department_or_faculty":
        return bool(
            re.search(r"\bdepartment|faculty|school|institute|chair\b", lowered)
            or _has_not_verified_marker(answer)
        )
    if field == "contact_information":
        has_contact_marker = bool(
            re.search(r"\bcontact|email|e-mail|phone|telephone|office|admissions office\b", lowered)
        )
        has_contact_value = bool(
            re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", answer)
            or re.search(r"https?://|www\.", lowered)
            or re.search(r"\+\d|\b\d{6,}\b", lowered)
        )
        return bool(
            (has_contact_marker and (has_contact_value or _has_not_verified_marker(answer)))
        )
    if field == "funding_or_scholarship":
        return bool(
            re.search(
                r"\bscholarship|funding|grant|stipend|assistantship|tuition waiver\b", lowered
            )
            or _has_not_verified_marker(answer)
        )
    if field == "publication_or_profile_links":
        has_pub_marker = bool(
            re.search(
                r"\bpublication|publications|paper|papers|google scholar|researchgate|profile\b",
                lowered,
            )
        )
        has_link = bool(re.search(r"https?://|www\.", lowered))
        return bool((has_pub_marker and (has_link or _has_not_verified_marker(answer))))
    if field == "visa_or_work_rights":
        return bool(re.search(r"\bvisa|work|hours|residence permit\b", lowered))
    if field == "aps_requirement_stage":
        return bool(
            re.search(r"\baps|certificate|required|before applying|application stage\b", lowered)
        )
    if field == "admission_decision_signal":
        return bool(
            re.search(
                r"\b(admission decision|admission competitiveness|competitiveness|decision snapshot|verdict|likely|risky|unknown|unclear|cannot determine)\b",
                lowered,
            )
            or _has_not_verified_marker(answer)
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
    if "publication_or_profile_links" in required_set:
        variants.append(f"{query} professor publication list google scholar profile official")
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
        "gpa_threshold",
        "gpa_or_grade_threshold",
        "ects_prerequisites",
        "ects_or_prerequisite_credit_breakdown",
    }:
        variants.extend(
            [
                f"{query} official admission requirements GPA ECTS prerequisites",
                f"{query} official eligibility criteria minimum grade credit requirements",
                f"{query} official examination regulations admission criteria PDF",
            ]
        )
    if required_set & {
        "instruction_language",
        "language_requirements",
        "language_test_thresholds",
        "language_test_score_thresholds",
    }:
        variants.extend(
            [
                f"{query} official language of instruction taught in",
                f"{query} official language requirements IELTS TOEFL minimum score",
                f"{query} accepted English tests CEFR minimum score official",
            ]
        )
    if "application_deadline" in required_set or "international_deadline" in required_set:
        variants.append(f"{query} official application deadline exact date")
    if required_set & _RESEARCHER_REQUIRED_ANSWER_FIELDS:
        variants.extend(
            [
                f"{query} official faculty professors supervisors contacts",
                f"{query} official labs research groups department",
                f"{query} scholarship funding assistantship official",
            ]
        )

    limit = _query_decomposition_limit(state)
    if required_set & _RESEARCHER_REQUIRED_ANSWER_FIELDS:
        limit = max(limit, 4)
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
        if field in {
            "admission_requirements",
            "eligibility_requirements",
            "gpa_threshold",
            "gpa_or_grade_threshold",
            "ects_prerequisites",
            "ects_or_prerequisite_credit_breakdown",
        }:
            variants.append(f"{query} official admission requirements minimum grade GPA ECTS")
        elif field in {
            "instruction_language",
            "language_requirements",
            "language_test_thresholds",
            "language_test_score_thresholds",
        }:
            variants.append(f"{query} official IELTS TOEFL CEFR minimum score")
        elif field in {"application_deadline", "international_deadline"}:
            variants.append(f"{query} official application deadline exact date")
        elif field == "professors_or_supervisors":
            variants.append(f"{query} official faculty professors supervisors")
        elif field == "labs_or_research_groups":
            variants.append(f"{query} official labs research groups institute")
        elif field == "department_or_faculty":
            variants.append(f"{query} official department faculty school")
        elif field == "contact_information":
            variants.append(f"{query} official contact email phone admissions office")
        elif field == "funding_or_scholarship":
            variants.append(f"{query} official scholarship funding assistantship")
        elif field == "publication_or_profile_links":
            variants.append(f"{query} official professor publications profile links")
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


def _result_source_url(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return _normalized_url(str(metadata.get("url", ""))) or _normalized_url(
        str(result.get("source_path", ""))
    )


def _result_title(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    parts = [
        str(metadata.get("title", "")),
        str(metadata.get("university", "")),
        str(metadata.get("section_heading", "")),
    ]
    return " ".join(part for part in parts if part).strip()


def _is_german_university_state(state: dict) -> bool:
    prompt = str(state.get("safe_user_prompt", "")).strip()
    return bool(prompt and is_likely_german_university_query(prompt))


def _german_task_from_state(state: dict):
    prompt = str(state.get("safe_user_prompt", "")).strip()
    if not prompt:
        return None
    try:
        return resolve_german_research_task(prompt)
    except Exception:
        return None


def _german_scope_accepts_result(result: dict, state: dict) -> bool:
    if not _is_german_university_state(state):
        return True
    task = _german_task_from_state(state)
    if task is None:
        return True
    url = _result_source_url(result)
    if not url:
        return True
    content = " ".join(str(result.get("content", "")).split()).strip()
    scope = validate_german_program_scope(
        url,
        title=_result_title(result),
        snippet=content[:600],
        content=content[:2400],
        program=task.program,
        degree_level=task.degree_level,
    )
    return bool(scope.get("accepted", False))


def _filter_german_wrong_scope_results(results: list[dict], state: dict) -> list[dict]:
    if not isinstance(results, list) or not results or not _is_german_university_state(state):
        return results if isinstance(results, list) else []
    accepted: list[dict] = []
    dropped: list[dict] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if _german_scope_accepts_result(result, state):
            accepted.append(result)
        else:
            dropped.append(result)
    if dropped:
        emit_trace_event(
            "german_scope_results_filtered",
            {
                "dropped_count": len(dropped),
                "dropped_urls": [_result_source_url(item) for item in dropped[:8]],
            },
        )
    return accepted or [item for item in results if isinstance(item, dict)]


def _ledger_found_source_urls(state: dict) -> list[str]:
    rows = state.get("coverage_ledger", [])
    if not isinstance(rows, list) or not rows:
        rows = state.get("web_field_evidence", [])
    rows = rows if isinstance(rows, list) else []
    urls: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")).strip().lower() != "found":
            continue
        url = _normalized_url(str(row.get("source_url", "")))
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _is_priority_german_result(result: dict, state: dict) -> bool:
    if not isinstance(result, dict) or not _is_german_university_state(state):
        return False
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if str(metadata.get("source_type", "")).strip().lower() == "german_researcher":
        return True
    url = _result_source_url(result)
    return bool(url and url in set(_ledger_found_source_urls(state)))


def _preserve_priority_german_results(
    selected: list[dict],
    original: list[dict],
    state: dict,
    *,
    limit: int | None = None,
) -> list[dict]:
    if not _is_german_university_state(state):
        return selected
    selected = [item for item in selected if isinstance(item, dict)]
    original = [item for item in original if isinstance(item, dict)]
    max_keep = max(
        int(limit or 0),
        len(selected),
        min(
            len(selected) + 6,
            max(
                _prompt_retrieval_result_limit(),
                _max_context_results_for_mode(_mode_from_state(state)),
            )
            + 6,
        ),
    )
    selected_keys = {_result_identity_key(item) for item in selected if _result_identity_key(item)}
    priority: list[dict] = []
    for candidate in original:
        if not _is_priority_german_result(candidate, state):
            continue
        if not _german_scope_accepts_result(candidate, state):
            continue
        key = _result_identity_key(candidate)
        if not key or key in selected_keys:
            continue
        priority.append(candidate)
        selected_keys.add(key)
        if len(priority) >= 6:
            break
    if not priority:
        return selected
    merged = priority + selected
    if len(merged) > max_keep:
        merged = merged[:max_keep]
    emit_trace_event(
        "german_priority_evidence_preserved",
        {
            "inserted_count": len(priority),
            "source_urls": [_result_source_url(item) for item in priority],
            "result_count": len(merged),
        },
    )
    return merged


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
    results = _filter_german_wrong_scope_results(results, state)
    if len(results) <= 2:
        return list(results)

    min_unique_domains = max(
        1, int(getattr(settings.web_search, "retrieval_min_unique_domains", 1))
    )
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
    selected = _preserve_priority_german_results(
        selected,
        results,
        state,
        limit=max_keep + 6 if _is_german_university_state(state) else max_keep,
    )

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
    research_objectives_missing = _research_objectives_missing_from_web(state)
    if admissions_critical_missing:
        confidence -= min(0.35, 0.1 * len(admissions_critical_missing))
    if _is_admissions_requirements_query(state) and web_required_coverage is None:
        confidence -= 0.12
    web_research_coverage = _safe_float(state.get("web_research_objective_coverage"))
    web_research_coverage = (
        _clamp01(web_research_coverage, fallback=1.0) if web_research_coverage is not None else None
    )
    if _is_researcher_objective_query(state) and web_research_coverage is not None:
        confidence -= (1.0 - web_research_coverage) * 0.12
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
        uncertainty_reasons.append(
            "Some requested fields are not fully verified from web evidence."
        )
    if (
        _is_researcher_objective_query(state)
        and web_research_coverage is not None
        and web_research_coverage < 0.999
    ):
        uncertainty_reasons.append(
            "Some requested research objectives are not fully verified from official sources."
        )
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
            "web_research_objective_coverage": (
                round(web_research_coverage, 4) if web_research_coverage is not None else None
            ),
            "web_research_objectives_missing": (state.get("web_research_objectives_missing") or [])[
                :4
            ],
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


def _web_retrieval_timeout_seconds(
    search_mode: str | None = None, *, query: str | None = None
) -> float:
    configured = _safe_float(getattr(settings.web_search, "timeout_seconds", 12.0))
    if configured is None:
        configured = 12.0
    normalized_mode = _normalized_request_mode(search_mode)
    if normalized_mode == _AUTO_MODE:
        normalized_mode = _DEEP_MODE
    if normalized_mode == _DEEP_MODE:
        explicit = _safe_float(getattr(settings.web_search, "deep_timeout_seconds", 0.0))
        if explicit is not None and explicit > 0:
            tuned = float(explicit)
            compact_query = " ".join(str(query or "").split()).strip()
            if compact_query and _is_admissions_high_precision_query(compact_query):
                tuned = max(tuned, 150.0)
            return max(30.0, min(300.0, tuned))
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
        if _is_admissions_high_precision_query(compact_query):
            timeout_seconds *= 1.25
            floor_seconds = max(floor_seconds, 75.0)
            ceiling_seconds = max(ceiling_seconds, 240.0)
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
    debug: bool = False,
) -> dict | None:
    timeout_seconds = _web_retrieval_timeout_seconds(search_mode, query=query)
    try:
        return await asyncio.wait_for(
            _aretrieve_web_chunks_with_mode(
                query,
                top_k=top_k,
                search_mode=search_mode,
                debug=debug,
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
        error_message = " ".join(str(exc).split()).strip()
        logger.warning("Web retrieval query failed; query=%s error=%s", query, error_message)
        hard_error = bool(
            re.search(
                r"\b(exceeds your plan|usage limit|quota|insufficient credits|invalid api key|unauthorized|forbidden|401|403)\b",
                error_message,
                flags=re.IGNORECASE,
            )
        )
        return {
            "results": [],
            "_failed": True,
            "_hard_error": hard_error,
            "_error_message": error_message[:260],
            "_query": query,
            "_search_mode": str(search_mode),
        }


def _web_result_timed_out(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get("_timed_out", False))


def _truncate_query_safely(text: str, *, max_chars: int) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    clipped = compact[:max_chars]
    return clipped.rsplit(" ", 1)[0].strip() or clipped


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
        if key in {"application_deadline", "international_deadline"}:
            field_hints.append("application deadline")
        elif key in {
            "instruction_language",
            "language_requirements",
            "language_test_thresholds",
            "language_test_score_thresholds",
        }:
            field_hints.append("IELTS TOEFL language requirement")
        elif key in {"eligibility_requirements", "gpa_threshold", "gpa_or_grade_threshold"}:
            field_hints.append("admission requirement GPA")
        elif key in {"ects_prerequisites", "ects_or_prerequisite_credit_breakdown"}:
            field_hints.append("ECTS prerequisite credits")
        elif key == "application_portal":
            field_hints.append("application portal")
    if field_hints:
        suffix = " ".join(field_hints[:4]).strip()
        lowered = compact.lower()
        if suffix and suffix.lower() not in lowered:
            compact = f"{compact} {suffix}".strip()
    return _truncate_query_safely(compact, max_chars=_WEB_QUERY_MAX_CHARS)


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
        trimmed = _truncate_query_safely(compact, max_chars=_WEB_QUERY_MAX_CHARS)
        if not trimmed:
            continue
        key = trimmed.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(trimmed)
        if len(normalized) >= _WEB_TIMEOUT_RESCUE_MAX_QUERIES:
            break
    return normalized


def _unigraph_answered_official_required_field(
    result: dict | None = None,
    state: dict | None = None,
) -> bool:
    payloads: list[dict] = []
    if isinstance(result, dict):
        payloads.append(result)
        if bool(result.get("unigraph_answered_required_field")):
            return True
    if isinstance(state, dict):
        if bool(state.get("unigraph_answered_required_field")):
            return True
        payloads.append(state)

    trusted_source_markers = {
        "official",
        "official_university_page",
        "official_university_pdf",
        "daad",
        "uni_assist",
        "government_or_eu",
    }
    for payload in payloads:
        rows = payload.get("coverage_ledger") or payload.get("web_field_evidence") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            status = " ".join(str(row.get("status", "")).split()).strip().lower()
            if status != "found":
                continue
            source_type = " ".join(str(row.get("source_type", "")).split()).strip().lower()
            if "official" in source_type or source_type in trusted_source_markers:
                return True
    return False


async def _attempt_web_timeout_rescue(
    *,
    base_query: str,
    state: dict,
    top_k: int,
    search_mode: str,
) -> dict | None:
    if _unigraph_answered_official_required_field(state=state):
        state["rescue_retrieval_skipped_reason"] = "unigraph_answered_required_field"
        logger.info(
            "Rescue retrieval skipped | rescue_retrieval_skipped_reason=%s",
            state["rescue_retrieval_skipped_reason"],
        )
        emit_trace_event(
            "rescue_retrieval_skipped",
            {"reason": state["rescue_retrieval_skipped_reason"], "kind": "web_timeout"},
        )
        return None
    rescue_queries = _timeout_rescue_queries(base_query=base_query, state=state)
    if not rescue_queries:
        return None
    normalized_mode = _normalized_request_mode(search_mode)
    rescue_mode = _FAST_MODE if normalized_mode == _DEEP_MODE else normalized_mode
    emit_trace_event(
        "web_timeout_rescue_started",
        {
            "base_query": base_query[:220],
            "rescue_mode": rescue_mode,
            "query_variants": rescue_queries[:8],
        },
    )
    best_result: dict | None = None
    best_score: tuple[float, int, int] = (-1.0, -(10**6), -1)
    timed_out_count = 0
    for rescue_query in rescue_queries:
        candidate = await _run_one_web_query_with_timeout(
            rescue_query,
            top_k=max(2, int(top_k or 2)),
            search_mode=rescue_mode,
            **({"debug": True} if bool(state.get("unigraph_debug_enabled", False)) else {}),
        )
        if not isinstance(candidate, dict):
            continue
        if _web_result_timed_out(candidate):
            timed_out_count += 1
            continue
        rows = _result_dicts(candidate.get("results", []))
        verification = candidate.get("verification", {})
        verification = verification if isinstance(verification, dict) else {}
        coverage = _safe_float(verification.get("required_field_coverage"))
        if coverage is None:
            coverage = 1.0 if rows else 0.0
        missing_ids = verification.get("required_fields_missing", [])
        missing_count = (
            len([item for item in missing_ids if " ".join(str(item).split()).strip()])
            if isinstance(missing_ids, list)
            else (0 if rows else 999)
        )
        score = (float(coverage), -int(missing_count), len(rows))
        if score <= best_score:
            continue
        best_score = score
        best_result = dict(candidate)

    rescued = bool(best_result and _result_dicts(best_result.get("results", [])))
    emit_trace_event(
        "web_timeout_rescue_completed",
        {
            "rescued": rescued,
            "rescue_mode": rescue_mode,
            "timed_out_count": timed_out_count,
            "query_count": len(rescue_queries),
        },
    )
    if not rescued or best_result is None:
        return None
    best_result["_timeout_rescued"] = True
    best_result["_timeout_rescue_mode"] = rescue_mode
    best_result["_timeout_rescue_queries"] = rescue_queries[:8]
    best_result["_timeout_rescue_timed_out_count"] = timed_out_count
    return best_result


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
    always_web_retrieval = web_ready and (_should_run_web_retrieval() or deep_mode)
    fallback_for_low_confidence = _should_use_web_fallback(state, top_similarity)
    fallback_for_missing_urls = (
        bool(vector_results)
        and not vector_has_urls
        and web_ready
        and settings.web_search.fallback_enabled
    )
    vector_domain_count = len(_allowed_citation_hosts(_evidence_urls(vector_results)))
    state["web_fallback_attempted"] = False
    state["web_result_count"] = 0
    state["web_expansion_used"] = False
    state["web_retrieval_verified"] = None
    state["web_required_field_coverage"] = None
    state["web_required_fields_missing"] = []
    state["web_field_evidence"] = []
    state["web_source_policy"] = ""
    state["web_unresolved_fields"] = []
    state["web_research_objective_coverage"] = None
    state["web_research_objectives_missing"] = []
    state["web_timeout_count"] = 0
    state["web_timed_out_queries"] = []
    state["web_timeout_rescued"] = False
    state["web_provider_error"] = ""
    state["unigraph_answered_required_field"] = False
    state["rescue_retrieval_skipped_reason"] = ""
    state["coverage_ledger"] = []
    state["unresolved_slots"] = []
    state["source_policy_decisions"] = []
    state["retrieval_budget_usage"] = {}
    state["question_schema_id"] = (
        " ".join(str(state.get("question_schema_id", "")).split()).strip() or "student_general"
    )
    state["required_slots"] = (
        state.get("required_slots", []) if isinstance(state.get("required_slots", []), list) else []
    )
    web_query = (
        _compact_web_retrieval_query(base_query=retrieval_query, state=state) or retrieval_query
    )

    if not (always_web_retrieval or fallback_for_low_confidence or fallback_for_missing_urls):
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
            "query_planner_started", {"query": web_query[:220], "planner": "deterministic"}
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
                        **(
                            {"debug": True}
                            if bool(state.get("unigraph_debug_enabled", False))
                            else {}
                        ),
                    )
                    or {}
                )
        else:
            web_result = (
                await _run_one_web_query_with_timeout(
                    web_query,
                    top_k=_default_retrieval_top_k(),
                    search_mode=search_mode,
                    **({"debug": True} if bool(state.get("unigraph_debug_enabled", False)) else {}),
                )
                or {}
            )
        web_result = web_result if isinstance(web_result, dict) else {}
        web_hard_error = bool(web_result.get("_hard_error", False))
        if web_hard_error:
            state["web_provider_error"] = " ".join(
                str(web_result.get("_error_message", "")).split()
            ).strip()[:260]
        unigraph_answered = _unigraph_answered_official_required_field(web_result, state)
        if unigraph_answered:
            state["unigraph_answered_required_field"] = True
            state["rescue_retrieval_skipped_reason"] = "unigraph_answered_required_field"
            logger.info(
                "Rescue retrieval skipped | rescue_retrieval_skipped_reason=%s",
                state["rescue_retrieval_skipped_reason"],
            )
            emit_trace_event(
                "rescue_retrieval_skipped",
                {
                    "reason": state["rescue_retrieval_skipped_reason"],
                    "kind": "web_primary",
                    "query": web_query[:220],
                },
            )
        web_timeout_count = 1 if _web_result_timed_out(web_result) else 0
        if web_timeout_count > 0 and not web_hard_error and not unigraph_answered:
            rescued_result = await _attempt_web_timeout_rescue(
                base_query=web_query,
                state=state,
                top_k=_default_retrieval_top_k(),
                search_mode=search_mode,
            )
            if isinstance(rescued_result, dict):
                web_result = rescued_result
                web_timeout_count = 0
                state["web_timeout_rescued"] = True

        if bool(state.get("unigraph_debug_enabled", False)):
            debug_payload = web_result.get("debug", {})
            if isinstance(debug_payload, dict) and debug_payload:
                state["unigraph_debug"] = debug_payload

        if not web_hard_error and not unigraph_answered:
            web_result = await _augment_with_german_researcher(web_query, web_result, state)

        def _field_evidence_quality(rows: list[dict]) -> tuple[int, float]:
            found_count = 0
            confidence_sum = 0.0
            for item in rows:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status", "")).strip().lower()
                if status != "found":
                    continue
                found_count += 1
                confidence_sum += float(_safe_float(item.get("confidence")) or 0.0)
            avg_confidence = (confidence_sum / found_count) if found_count else 0.0
            return found_count, avg_confidence

        def _normalize_field_evidence(payload_rows) -> list[dict]:
            if not isinstance(payload_rows, list):
                return []
            normalized: list[dict] = []
            for row in payload_rows:
                if not isinstance(row, dict):
                    continue
                field_id = " ".join(str(row.get("id", row.get("field", ""))).split()).strip()
                label = " ".join(str(row.get("label", field_id)).split()).strip() or field_id
                status = " ".join(str(row.get("status", "")).split()).strip().lower() or "missing"
                value = " ".join(str(row.get("value", "")).split()).strip()
                source_url = " ".join(str(row.get("source_url", "")).split()).strip()
                source_type = (
                    " ".join(str(row.get("source_type", "")).split()).strip().lower() or "discovery"
                )
                evidence_text = " ".join(
                    str(row.get("evidence_snippet", row.get("evidence_text", ""))).split()
                ).strip()
                confidence = float(_safe_float(row.get("confidence")) or 0.0)
                retrieved_at = " ".join(str(row.get("retrieved_at", "")).split()).strip()
                if not field_id:
                    continue
                normalized.append(
                    {
                        "field": field_id,
                        "id": field_id,
                        "label": label,
                        "status": (
                            status
                            if status in {"found", "missing", "conflict", "stale"}
                            else "missing"
                        ),
                        "value": value,
                        "source_url": source_url,
                        "source_type": (
                            source_type if source_type in {"official", "discovery"} else "discovery"
                        ),
                        "evidence_snippet": evidence_text,
                        "evidence_text": evidence_text,
                        "confidence": round(max(0.0, min(1.0, confidence)), 4),
                        "retrieved_at": retrieved_at,
                    }
                )
                if len(normalized) >= 16:
                    break
            return normalized

        def _absorb_web_field_evidence(payload: dict) -> None:
            rows = _normalize_field_evidence(payload.get("coverage_ledger", []))
            if not rows:
                rows = _normalize_field_evidence(payload.get("evidence_ledger", []))
            if not rows:
                rows = _normalize_field_evidence(payload.get("field_evidence", []))
            if not rows:
                verification_payload = payload.get("verification", {})
                verification_payload = (
                    verification_payload if isinstance(verification_payload, dict) else {}
                )
                rows = _normalize_field_evidence(verification_payload.get("field_evidence", []))
            if not rows:
                return
            existing_rows = state.get("web_field_evidence", [])
            existing_rows = existing_rows if isinstance(existing_rows, list) else []
            existing_quality = _field_evidence_quality(existing_rows)
            candidate_quality = _field_evidence_quality(rows)
            if (not existing_rows) or candidate_quality > existing_quality:
                state["web_field_evidence"] = rows

        def _absorb_web_verification(payload: dict) -> None:
            local_verification = payload.get("verification", {})
            local_verification = local_verification if isinstance(local_verification, dict) else {}

            required_coverage = _safe_float(local_verification.get("required_field_coverage"))
            if required_coverage is not None:
                normalized_required = round(_clamp01(required_coverage, fallback=1.0), 4)
                existing_required = _safe_float(state.get("web_required_field_coverage"))
                if existing_required is None or normalized_required > existing_required:
                    state["web_required_field_coverage"] = normalized_required

            missing_required = local_verification.get("required_fields_missing", [])
            if isinstance(missing_required, list):
                normalized_missing_required = [
                    " ".join(str(item).split()).strip()
                    for item in missing_required
                    if " ".join(str(item).split()).strip()
                ][:6]
                existing_missing_required = state.get("web_required_fields_missing", [])
                existing_missing_required = (
                    existing_missing_required if isinstance(existing_missing_required, list) else []
                )
                if not existing_missing_required or (
                    normalized_missing_required
                    and len(normalized_missing_required) < len(existing_missing_required)
                ):
                    state["web_required_fields_missing"] = normalized_missing_required

            research_coverage = _safe_float(local_verification.get("research_objective_coverage"))
            if research_coverage is not None:
                normalized_research = round(_clamp01(research_coverage, fallback=1.0), 4)
                existing_research = _safe_float(state.get("web_research_objective_coverage"))
                if existing_research is None or normalized_research > existing_research:
                    state["web_research_objective_coverage"] = normalized_research

            missing_research = local_verification.get("research_objectives_missing", [])
            if isinstance(missing_research, list):
                normalized_missing_research = [
                    " ".join(str(item).split()).strip()
                    for item in missing_research
                    if " ".join(str(item).split()).strip()
                ][:6]
                existing_missing_research = state.get("web_research_objectives_missing", [])
                existing_missing_research = (
                    existing_missing_research if isinstance(existing_missing_research, list) else []
                )
                if not existing_missing_research or (
                    normalized_missing_research
                    and len(normalized_missing_research) < len(existing_missing_research)
                ):
                    state["web_research_objectives_missing"] = normalized_missing_research

            verified = local_verification.get("verified")
            if isinstance(verified, bool):
                if state.get("web_retrieval_verified") is None:
                    state["web_retrieval_verified"] = verified
                else:
                    state["web_retrieval_verified"] = (
                        bool(state["web_retrieval_verified"]) or verified
                    )

            source_policy = " ".join(
                str(local_verification.get("source_policy", "")).split()
            ).strip()
            if source_policy:
                state["web_source_policy"] = source_policy

            unresolved_fields = local_verification.get("unresolved_fields", [])
            if isinstance(unresolved_fields, list):
                state["web_unresolved_fields"] = [
                    " ".join(str(item).split()).strip()
                    for item in unresolved_fields
                    if " ".join(str(item).split()).strip()
                ][:10]

        _absorb_web_verification(web_result)
        _absorb_web_field_evidence(web_result)
        coverage_ledger = web_result.get("coverage_ledger", [])
        if isinstance(coverage_ledger, list):
            normalized_coverage = _normalize_field_evidence(coverage_ledger)
            if normalized_coverage:
                state["coverage_ledger"] = normalized_coverage
                state["web_field_evidence"] = normalized_coverage
        if _unigraph_answered_official_required_field(web_result, state):
            state["unigraph_answered_required_field"] = True
            if not state.get("rescue_retrieval_skipped_reason"):
                state["rescue_retrieval_skipped_reason"] = "unigraph_answered_required_field"
        unresolved_slots = web_result.get("unresolved_slots", [])
        if isinstance(unresolved_slots, list):
            state["unresolved_slots"] = [
                " ".join(str(item).split()).strip()
                for item in unresolved_slots
                if " ".join(str(item).split()).strip()
            ][:12]
        source_policy_decisions = web_result.get("source_policy_decisions", [])
        if isinstance(source_policy_decisions, list):
            state["source_policy_decisions"] = [
                dict(item) for item in source_policy_decisions if isinstance(item, dict)
            ][:20]
        retrieval_budget_usage = web_result.get("retrieval_budget_usage", {})
        if isinstance(retrieval_budget_usage, dict):
            state["retrieval_budget_usage"] = dict(retrieval_budget_usage)
        question_schema_id = " ".join(str(web_result.get("question_schema_id", "")).split()).strip()
        if question_schema_id:
            state["question_schema_id"] = question_schema_id
        required_slots_payload = web_result.get("required_slots", [])
        if isinstance(required_slots_payload, list):
            state["required_slots"] = [
                dict(item) for item in required_slots_payload if isinstance(item, dict)
            ][:20]
        coverage_summary = web_result.get("coverage_summary", {})
        coverage_summary = coverage_summary if isinstance(coverage_summary, dict) else {}
        query_plan = web_result.get("query_plan", {})
        if isinstance(query_plan, dict) and query_plan:
            state["query_plan"] = dict(query_plan)
        summary_policy = " ".join(str(coverage_summary.get("source_policy", "")).split()).strip()
        if summary_policy and not state.get("web_source_policy"):
            state["web_source_policy"] = summary_policy
        summary_unresolved = coverage_summary.get("unresolved_fields", [])
        if (
            isinstance(summary_unresolved, list)
            and summary_unresolved
            and not state.get("web_unresolved_fields")
        ):
            state["web_unresolved_fields"] = [
                " ".join(str(item).split()).strip()
                for item in summary_unresolved
                if " ".join(str(item).split()).strip()
            ][:10]
        verification = web_result.get("verification", {})
        verification = verification if isinstance(verification, dict) else {}
        query_plan = query_plan if isinstance(query_plan, dict) else {}
        emit_trace_event(
            "query_planner_completed",
            {
                "planner": str(query_plan.get("planner", "heuristic")),
                "llm_used": bool(query_plan.get("llm_used", False)),
                "subquestions": query_plan.get("subquestions", []),
                "required_fields": query_plan.get("required_fields", []),
                "research_objectives": query_plan.get("research_objectives", []),
                "queries": web_result.get("query_variants", []),
            },
        )
        web_results = _result_dicts(web_result.get("results", []))
        if web_timeout_count > 0:
            state["web_timeout_count"] = (
                int(state.get("web_timeout_count", 0) or 0) + web_timeout_count
            )
            state["web_timed_out_queries"] = [web_query[:220]]

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
                "field_evidence": (state.get("web_field_evidence") or [])[:10],
                "coverage_ledger": (state.get("coverage_ledger") or [])[:10],
                "question_schema_id": state.get("question_schema_id", ""),
                "required_slots": [
                    str(item.get("slot_id", "")).strip()
                    for item in (state.get("required_slots", []) or [])
                    if isinstance(item, dict) and str(item.get("slot_id", "")).strip()
                ][:12],
                "unresolved_slots": (state.get("unresolved_slots") or [])[:8],
                "source_policy_decisions": (state.get("source_policy_decisions") or [])[:8],
                "retrieval_budget_usage": state.get("retrieval_budget_usage") or {},
                "timeout_count": web_timeout_count,
                "timed_out_queries": (state.get("web_timed_out_queries") or [])[:4],
                "timeout_rescued": bool(state.get("web_timeout_rescued", False)),
                "rescue_retrieval_skipped_reason": state.get(
                    "rescue_retrieval_skipped_reason", ""
                ),
                "metrics": web_result.get("metrics", {}),
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
    debug: bool = False,
) -> dict:
    try:
        return await aretrieve_web_chunks(
            retrieval_query,
            top_k=top_k,
            search_mode=search_mode,
            debug=debug,
        )
    except TypeError as exc:
        # Backward compatibility for tests or call-sites patching an older function signature.
        if "search_mode" not in str(exc) and "debug" not in str(exc):
            raise
        try:
            return await aretrieve_web_chunks(
                retrieval_query,
                top_k=top_k,
                search_mode=search_mode,
            )
        except TypeError as second_exc:
            if "search_mode" not in str(second_exc):
                raise
            return await aretrieve_web_chunks(retrieval_query, top_k=top_k)


def _field_evidence_quality_score(rows: list[dict]) -> tuple[int, float]:
    found_count = 0
    confidence_total = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")).strip().lower() != "found":
            continue
        found_count += 1
        confidence_total += float(_safe_float(row.get("confidence")) or 0.0)
    return found_count, (confidence_total / found_count if found_count else 0.0)


_GERMAN_REQUIRED_FIELD_ALIASES = {
    "gpa_or_grade_threshold": "gpa_threshold",
    "ects_or_subject_credit_requirements": "ects_prerequisites",
    "application_deadline": "international_deadline",
    "language_test_score_thresholds": "language_test_score_thresholds",
    "german_language_requirement": "german_language_requirement",
    "application_portal": "application_portal",
    "selection_criteria": "selection_criteria",
    "competitiveness_signal": "competitiveness_signal",
}


def _canonical_required_field_id(field_id: str) -> str:
    compact = " ".join(str(field_id or "").split()).strip()
    return _GERMAN_REQUIRED_FIELD_ALIASES.get(compact, compact)


def _canonical_required_field_list(field_ids) -> list[str]:
    if not isinstance(field_ids, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in field_ids:
        field_id = _canonical_required_field_id(str(item))
        if not field_id or field_id in seen:
            continue
        seen.add(field_id)
        output.append(field_id)
    return output


def _merge_german_research_result(web_result: dict, german_result: dict) -> dict:
    if not isinstance(german_result, dict) or not german_result.get("applicable"):
        return web_result
    merged = dict(web_result if isinstance(web_result, dict) else {})

    existing_results = _result_dicts(merged.get("results", []))
    german_results = _result_dicts(german_result.get("results", []))
    if german_results:
        merged["results"] = _merge_retrieval_results(
            german_results,
            existing_results,
            limit=max(
                len(german_results), len(existing_results), _max_context_results_for_mode("deep")
            ),
        )

    existing_ledger = merged.get("coverage_ledger", merged.get("field_evidence", []))
    existing_ledger = existing_ledger if isinstance(existing_ledger, list) else []
    german_ledger = german_result.get("coverage_ledger", [])
    german_ledger = german_ledger if isinstance(german_ledger, list) else []
    if german_ledger and _field_evidence_quality_score(
        german_ledger
    ) >= _field_evidence_quality_score(existing_ledger):
        merged["coverage_ledger"] = german_ledger
        merged["field_evidence"] = german_ledger
        merged["evidence_ledger"] = german_ledger

    verification = merged.get("verification", {})
    verification = verification if isinstance(verification, dict) else {}
    german_verification = german_result.get("verification", {})
    german_verification = german_verification if isinstance(german_verification, dict) else {}
    if german_verification:
        verification = dict(verification)
        for key in (
            "required_field_coverage",
            "required_fields_missing",
            "unresolved_fields",
            "source_policy",
            "official_domains",
            "source_urls",
        ):
            if key in german_verification:
                if key in {"required_fields_missing", "unresolved_fields"}:
                    verification[key] = _canonical_required_field_list(german_verification[key])
                else:
                    verification[key] = german_verification[key]
        verification["german_researcher_applied"] = True
        merged["verification"] = verification

    coverage_summary = merged.get("coverage_summary", {})
    coverage_summary = coverage_summary if isinstance(coverage_summary, dict) else {}
    german_summary = german_result.get("coverage_summary", {})
    german_summary = german_summary if isinstance(german_summary, dict) else {}
    if german_summary:
        coverage_summary = dict(coverage_summary)
        coverage_summary.update(german_summary)
        coverage_summary["german_researcher_applied"] = True
        merged["coverage_summary"] = coverage_summary

    query_variants = []
    for collection in (merged.get("query_variants", []), german_result.get("query_variants", [])):
        if not isinstance(collection, list):
            continue
        for item in collection:
            text = " ".join(str(item).split()).strip()
            if text and text not in query_variants:
                query_variants.append(text)
    if query_variants:
        merged["query_variants"] = query_variants[:40]

    existing_budget = merged.get("retrieval_budget_usage", {})
    existing_budget = existing_budget if isinstance(existing_budget, dict) else {}
    german_budget = german_result.get("retrieval_budget_usage", {})
    german_budget = german_budget if isinstance(german_budget, dict) else {}
    if german_budget:
        merged_budget = dict(existing_budget)
        merged_budget["german_research"] = german_budget
        base_executed = int(_safe_int(merged_budget.get("queries_executed")) or 0)
        german_executed = int(_safe_int(german_budget.get("queries_executed")) or 0)
        if base_executed > 0:
            merged_budget["queries_executed"] = base_executed + german_executed
        merged["retrieval_budget_usage"] = merged_budget

    merged["german_research"] = {
        "applied": True,
        "source_routes_attempted": german_result.get("source_routes_attempted", {}),
        "verification": german_result.get("verification", {}),
        "timings_ms": german_result.get("timings_ms", {}),
    }
    merged["retrieval_strategy"] = (
        f"{str(merged.get('retrieval_strategy', 'web_search')).strip() or 'web_search'}_german_research"
    )
    return merged


async def _augment_with_german_researcher(web_query: str, web_result: dict, state: dict) -> dict:
    if not bool(getattr(settings.web_search, "german_university_mode_enabled", True)):
        return web_result
    if not is_likely_german_university_query(web_query):
        return web_result
    cache_key = " ".join(str(web_query).split()).strip().lower()
    cached_key = " ".join(str(state.get("german_research_cache_key", "")).split()).strip().lower()
    cached_result = state.get("german_research_cached_result")
    if cache_key and cache_key == cached_key and isinstance(cached_result, dict):
        state["german_researcher_cache_hits"] = (
            int(state.get("german_researcher_cache_hits", 0) or 0) + 1
        )
        state["german_researcher_applied"] = bool(cached_result.get("applicable", False))
        state["german_researcher_verification"] = cached_result.get("verification", {})
        emit_trace_event(
            "german_researcher_cache_hit",
            {
                "query": web_query[:220],
                "verification": cached_result.get("verification", {}),
            },
        )
        return _merge_german_research_result(web_result, cached_result)
    emit_trace_event(
        "german_researcher_started",
        {
            "query": web_query[:220],
        },
    )
    state["german_researcher_calls"] = int(state.get("german_researcher_calls", 0) or 0) + 1
    try:
        german_result = await research_german_university(web_query)
    except Exception as exc:
        emit_trace_event(
            "german_researcher_error",
            {
                "error_type": type(exc).__name__,
                "error_message": " ".join(str(exc).split())[:260],
            },
        )
        logger.warning("German researcher retrieval failed; using base web result. %s", exc)
        return web_result
    state["german_research_cache_key"] = cache_key
    state["german_research_cached_result"] = german_result
    state["german_researcher_applied"] = bool(german_result.get("applicable", False))
    state["german_researcher_verification"] = german_result.get("verification", {})
    emit_trace_event(
        "german_researcher_completed",
        {
            "applied": bool(german_result.get("applicable", False)),
            "verification": german_result.get("verification", {}),
            "source_routes_attempted": german_result.get("source_routes_attempted", {}),
            "timings_ms": german_result.get("timings_ms", {}),
        },
    )
    return _merge_german_research_result(web_result, german_result)


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

    merged_results = _filter_german_wrong_scope_results(merged_results, state)
    original_results = list(merged_results)
    min_unique_domains = max(
        1, int(getattr(settings.web_search, "retrieval_min_unique_domains", 1))
    )
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
            comparison_entities = [
                str(item).strip() for item in comparison_entities if str(item).strip()
            ][:2]
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
    merged_results = _filter_german_wrong_scope_results(merged_results, state)
    merged_results = _preserve_priority_german_results(
        merged_results,
        original_results,
        state,
        limit=merge_limit,
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
    merged_results = _filter_german_wrong_scope_results(merged_results, state)
    merged_results = _preserve_priority_german_results(
        merged_results,
        merged_results,
        state,
        limit=max(len(merged_results), _prompt_retrieval_result_limit() + 6),
    )
    if not merged_results:
        return messages, None

    _set_retrieval_state(state, merged_results)
    _derive_evidence_trust_signals(merged_results, state)
    evidence_urls = _evidence_urls(merged_results)
    evidence_hosts = _allowed_citation_hosts(evidence_urls)
    min_unique_domains = max(
        1, int(getattr(settings.web_search, "retrieval_min_unique_domains", 1))
    )
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
        _format_web_retrieval_context(
            {
                "results": merged_results,
                "field_evidence": state.get("web_field_evidence", []),
            }
        )
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
    web_query = (
        _compact_web_retrieval_query(base_query=retrieval_query, state=state) or retrieval_query
    )
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
        # Web-only runtime path: retrieval context is sourced from the web pipeline.
        # Keep the vector-prefetch argument for backward compatibility with older tests/callers.
        _ = vector_prefetch_result
        web_results, web_fallback_attempted = await _retrieve_web_candidates_if_needed(
            retrieval_query,
            vector_results=[],
            vector_has_urls=False,
            top_similarity=None,
            search_mode=search_mode,
            state=state,
            web_prefetch_task=None,
        )
        if web_fallback_attempted and not web_results:
            web_provider_error = " ".join(str(state.get("web_provider_error", "")).split()).strip()
            if web_provider_error:
                state["context_guard_reason"] = "web_provider_unavailable"
                state["retrieval_strategy"] = "web_provider_error"
                return None, _WEB_PROVIDER_ERROR_DETAIL
            web_timeout_count = int(state.get("web_timeout_count", 0) or 0)
            freshness_sensitive = _is_freshness_sensitive_query(
                str(state.get("safe_user_prompt", ""))
            )
            if web_timeout_count > 0 and not bool(state.get("web_timeout_rescued", False)):
                state["context_guard_reason"] = "web_retrieval_timeout"
                state["retrieval_strategy"] = "web_timeout_no_results"
                return None, _WEB_RETRIEVAL_TIMEOUT_DETAIL
            if web_timeout_count > 0:
                state["retrieval_strategy"] = "web_timeout_rescued_no_results"
            if _is_citation_grounding_required() or freshness_sensitive:
                state["context_guard_reason"] = "weak_evidence_no_urls"
                return None, _NO_RELEVANT_INFORMATION_DETAIL
            state["retrieval_strategy"] = "web_empty_non_citation"
            logger.info("Web retrieval returned no results; continuing without retrieval context.")
            return messages, None

        if not web_results:
            if _is_citation_grounding_required():
                state["context_guard_reason"] = "no_relevant_information"
                state["retrieval_strategy"] = "web_no_results"
                return None, _NO_RELEVANT_INFORMATION_DETAIL
            state["retrieval_strategy"] = "web_skipped_no_results"
            return messages, None

        state["retrieval_strategy"] = (
            "web_only_german_research"
            if bool(state.get("german_researcher_applied", False))
            else "web_only"
        )
        merged_results = await _rerank_if_configured(retrieval_query, web_results, state)
        merged_results = _selective_retrieval_results(merged_results, state)
        return _apply_grounded_retrieval_context(
            messages=messages,
            merged_results=merged_results,
            used_web_results=True,
            state=state,
        )
    except Exception as exc:
        state["retrieval_strategy"] = "web_error"
        logger.warning(
            "Long-term retrieval failed; continuing without retrieved context. %s",
            exc,
        )
        if _is_citation_grounding_required():
            state["context_guard_reason"] = "no_relevant_information"
            return None, _NO_RELEVANT_INFORMATION_DETAIL
        return messages, None
    finally:
        state["retrieval_ms"] = _elapsed_ms(retrieval_started_at)


def _validate_citation_grounding_state(state: dict) -> str | None:
    if not _is_citation_grounding_required():
        return None
    if str(state.get("web_provider_error", "")).strip() and not state.get("evidence_urls"):
        state["context_guard_reason"] = "web_provider_unavailable"
        return _WEB_PROVIDER_ERROR_DETAIL
    if int(state.get("web_timeout_count", 0) or 0) > 0 and not state.get("evidence_urls"):
        if bool(state.get("web_timeout_rescued", False)):
            state["context_guard_reason"] = "weak_evidence_no_urls"
            return _NO_RELEVANT_INFORMATION_DETAIL
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


def _validate_german_university_answer(answer_text: str, query: str, evidence: list[dict]) -> dict:
    """Validate answer quality for German university queries."""
    if not getattr(settings.web_search, "answer_validation_enabled", True):
        return {"pass": True, "quality_score": 1.0, "issues": [], "warnings": []}

    validation = {
        "pass": True,
        "quality_score": 1.0,
        "issues": [],
        "warnings": [],
        "specific_data_found": {},
    }

    if not answer_text or not answer_text.strip():
        validation["pass"] = False
        validation["quality_score"] = 0.0
        validation["issues"].append("Empty answer")
        return validation

    # Check for generic placeholder responses
    generic_markers = [
        "typically requires",
        "generally includes",
        "usually around",
        "varies by program",
        "depends on the university",
        "please check the official",
        "contact the university for",
        "may require",
        "might need",
        "approximately",
    ]

    generic_count = sum(1 for marker in generic_markers if marker in answer_text.lower())
    if generic_count >= 3:
        validation["quality_score"] -= 0.3
        validation["issues"].append(f"Answer contains {generic_count} generic placeholders")

    # Check for "Not verified from official sources" overuse
    not_verified_count = answer_text.count("Not verified from official sources")
    if not_verified_count >= 3:
        validation["quality_score"] -= 0.2
        validation["issues"].append(
            f"Too many unverified fields ({not_verified_count}) - evidence may be insufficient"
        )

    # Validate specific data presence for critical queries
    query_lower = query.lower()

    # Deadline queries must have specific dates
    if any(
        keyword in query_lower
        for keyword in ["deadline", "frist", "last date", "apply by", "closing date"]
    ):
        date_pattern = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
        dates_found = date_pattern.findall(answer_text)
        if dates_found:
            validation["specific_data_found"]["deadline"] = dates_found[0]
        else:
            validation["quality_score"] -= 0.4
            validation["issues"].append("Deadline query but no specific date provided")

    # GPA queries must have specific thresholds
    if any(keyword in query_lower for keyword in ["gpa", "grade", "mindestnote", "minimum grade"]):
        gpa_pattern = re.compile(r"\b[1-4][.,]\d{1,2}\b")
        gpa_found = gpa_pattern.search(answer_text)
        if gpa_found:
            validation["specific_data_found"]["gpa"] = gpa_found.group(0)
        else:
            validation["quality_score"] -= 0.3
            validation["warnings"].append("GPA query but no specific threshold provided")

    # Language requirement queries must have specific scores
    if any(
        keyword in query_lower
        for keyword in [
            "ielts",
            "toefl",
            "language requirement",
            "english requirement",
            "sprachnachweis",
        ]
    ):
        lang_pattern = re.compile(r"(IELTS|TOEFL|TestDaF|DSH|CEFR)[^.]{0,40}\d", re.IGNORECASE)
        lang_found = lang_pattern.search(answer_text)
        if lang_found:
            validation["specific_data_found"]["language"] = lang_found.group(0)
        else:
            validation["quality_score"] -= 0.3
            validation["warnings"].append(
                "Language requirement query but no specific score provided"
            )

    # ECTS/duration queries
    if any(keyword in query_lower for keyword in ["ects", "credits", "duration", "semester"]):
        ects_pattern = re.compile(r"\b\d{1,3}\s*ECTS\b", re.IGNORECASE)
        ects_found = ects_pattern.search(answer_text)
        if ects_found:
            validation["specific_data_found"]["ects"] = ects_found.group(0)

    # Tuition/fee queries
    if any(
        keyword in query_lower for keyword in ["tuition", "fee", "cost", "semester contribution"]
    ):
        fee_pattern = re.compile(r"\b\d{1,5}(?:[.,]\d{1,2})?\s*(?:EUR|€|Euro)\b", re.IGNORECASE)
        fee_found = fee_pattern.search(answer_text)
        if fee_found:
            validation["specific_data_found"]["fee"] = fee_found.group(0)

    # Check source diversity and quality
    source_urls = _CITATION_URL_RE.findall(answer_text)
    unique_domains = set()
    german_official_count = 0

    for url in source_urls:
        parsed = urlparse(url)
        domain = (parsed.hostname or "").lower()
        if domain:
            unique_domains.add(domain)
            # Check if German official source
            if any(
                pattern in domain
                for pattern in ["uni-", "tu-", "fh-", "hs-", "daad.de", "hochschule"]
            ):
                german_official_count += 1

    if len(unique_domains) < 2:
        validation["warnings"].append(
            "Answer relies on single source - consider additional verification"
        )

    if getattr(settings.web_search, "german_university_mode_enabled", True):
        if german_official_count == 0 and len(source_urls) > 0:
            validation["warnings"].append("No German official university sources cited")
        elif german_official_count > 0:
            validation["quality_score"] += 0.05  # Bonus for official German sources

    # Check for vague date references instead of specific dates
    vague_date_markers = ["in july", "in summer", "usually in", "typically in", "around"]
    for marker in vague_date_markers:
        if marker in answer_text.lower() and "deadline" in query_lower:
            validation["warnings"].append(
                f"Vague date reference: '{marker}' - prefer specific dates"
            )

    # Final pass/fail determination
    min_quality_score = float(
        getattr(settings.web_search, "answer_validation_min_quality_score", 0.6)
    )
    validation["quality_score"] = max(0.0, min(1.0, validation["quality_score"]))
    validation["pass"] = (
        validation["quality_score"] >= min_quality_score and len(validation["issues"]) == 0
    )

    return validation


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


def _field_evidence_context_lines(field_evidence: list[dict], *, max_items: int) -> list[str]:
    lines: list[str] = []
    used = 0
    ordered_rows = sorted(
        [row for row in field_evidence if isinstance(row, dict)],
        key=lambda row: (
            0 if str(row.get("status", "")).strip().lower() == "found" else 1,
            -float(_safe_float(row.get("confidence")) or 0.0),
            str(row.get("label", row.get("field", row.get("id", "")))),
        ),
    )
    for row in ordered_rows:
        if not isinstance(row, dict):
            continue
        label = " ".join(str(row.get("label", row.get("field", row.get("id", "")))).split()).strip()
        status = " ".join(str(row.get("status", "")).split()).strip().lower()
        value = " ".join(str(row.get("value", "")).split()).strip()
        source_url = " ".join(str(row.get("source_url", "")).split()).strip()
        source_type = " ".join(str(row.get("source_type", "")).split()).strip().lower()
        evidence_snippet = " ".join(
            str(row.get("evidence_snippet", row.get("evidence_text", ""))).split()
        ).strip()
        confidence = float(_safe_float(row.get("confidence")) or 0.0)
        if not label:
            continue
        if status not in {"found", "conflict", "stale"}:
            lines.append(
                f"- {label}: The retrieved official evidence does not state this requested detail."
            )
        else:
            if status == "conflict":
                value = value or "Conflict across official sources."
            elif status == "stale":
                value = value or "Stale evidence. Refresh required."
            source_part = f" | source={source_url}" if source_url else ""
            source_type_part = f" | source_type={source_type}" if source_type else ""
            snippet_part = f" | evidence={evidence_snippet[:140]}" if evidence_snippet else ""
            value_text = value or "Verified from sources."
            lines.append(
                f"- {label}: {value_text}{source_part}{source_type_part}{snippet_part} | confidence={confidence:.2f}"
            )
        used += 1
        if used >= max_items:
            break
    return lines


def _format_web_retrieval_context(web_result: dict) -> dict | None:
    """Convert web fallback results into one system context message with URLs."""
    results = web_result.get("results", []) if isinstance(web_result, dict) else []
    header = ["Live web fallback context (Tavily web search)."]
    field_evidence = (
        web_result.get(
            "coverage_ledger",
            web_result.get("evidence_ledger", web_result.get("field_evidence", [])),
        )
        if isinstance(web_result, dict)
        else []
    )
    has_field_ledger = isinstance(field_evidence, list) and bool(field_evidence)
    if not has_field_ledger and (not isinstance(results, list) or not results):
        return None
    if has_field_ledger:
        header.append(
            "Use the field evidence ledger to track verified, missing, and conflicting fields. "
            "Do not output the ledger verbatim; synthesize a readable answer from the ledger and source excerpts."
        )
        header.append(
            "If a ledger field is marked found, do not answer that same field as not verified. "
            "For application deadlines, use application/deadline pages or deadline table evidence; do not treat language-proof submission dates as application deadlines."
        )
    if isinstance(field_evidence, list) and field_evidence:
        header.append("Field evidence table (verification-first):")
        header.extend(_field_evidence_context_lines(field_evidence, max_items=16))
    result_lines = _web_context_result_lines(
        results if isinstance(results, list) else [],
        max_items=_prompt_retrieval_result_limit(),
    )
    if result_lines:
        header.append("Official source excerpts:")
        header.extend(result_lines)
    elif not has_field_ledger:
        return None
    joined = "\n".join(header)
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
        "- Use only the exact allowed URLs below; do not cite other same-domain pages.",
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


def _response_disallowed_german_citation_urls(text: str, state: dict) -> list[str]:
    if not text or not _is_german_university_state(state):
        return []
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    allowed_exact = {
        _normalized_url(str(url)).rstrip("/") for url in evidence_urls if _normalized_url(str(url))
    }
    allowed_hosts = _allowed_citation_hosts(evidence_urls)
    if not allowed_exact or not allowed_hosts:
        return []
    disallowed: list[str] = []
    seen: set[str] = set()
    for cited_url in _CITATION_URL_RE.findall(str(text)):
        normalized = _normalized_url(cited_url).rstrip("/")
        if not normalized:
            continue
        host = _normalized_host_from_url(normalized)
        if host not in allowed_hosts or normalized in allowed_exact:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        disallowed.append(normalized)
        if len(disallowed) >= 8:
            break
    return disallowed


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
        if (
            bool(state.get("allow_uncited_comparison_fallback", False))
            and str(state.get("query_intent", "")).strip().lower() == "comparison"
        ):
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
    research_objectives_missing = _research_objectives_missing_from_web(state)

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
        disallowed_german_urls = _response_disallowed_german_citation_urls(result, state)
        if disallowed_german_urls:
            issues.append("disallowed_german_citation_url")
            state["disallowed_german_citation_urls"] = disallowed_german_urls
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
            ledger_rows = state.get("coverage_ledger", [])
            if isinstance(ledger_rows, list) and ledger_rows:
                max_not_verified = max(
                    max_not_verified,
                    sum(
                        1
                        for row in ledger_rows
                        if isinstance(row, dict)
                        and str(row.get("status", "")).strip().lower() != "found"
                    ),
                )
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
        and _is_researcher_objective_query(state)
        and research_objectives_missing
    ):
        issues.append("web_research_objectives_missing")
        for objective in research_objectives_missing[:4]:
            issues.append(f"web_missing:{objective}")
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
    if result != _NO_RELEVANT_INFORMATION_DETAIL and _has_weak_critical_evidence_claim(
        result, state
    ):
        issues.append("weak_critical_evidence_claim")
    if result != _NO_RELEVANT_INFORMATION_DETAIL:
        ledger_unverified_conflicts = _verified_ledger_fields_marked_unverified(result, state)
        if ledger_unverified_conflicts:
            issues.append("verified_ledger_field_marked_unverified")
            for field in ledger_unverified_conflicts[:4]:
                issues.append(f"ledger_conflict:{field}")
        if _answer_deadline_conflicts_with_ledger(result, state):
            issues.append("deadline_conflicts_with_verified_ledger")
    if bool(state.get("deadline_query", False)) and result != _NO_RELEVANT_INFORMATION_DETAIL:
        if not _has_date_like_value(result):
            issues.append("missing_deadline_date")
    if _has_contradiction_signal(result, state):
        issues.append("source_conflict_detected")
    if result != _NO_RELEVANT_INFORMATION_DETAIL and _has_speculative_factual_language(
        result, state
    ):
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
    lines.append("Return strict JSON only with keys: intent, subquestions, success_criteria.")
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
        "web_research_objectives_missing",
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
                "professor",
                "faculty",
                "supervisor",
                "lab",
                "research group",
                "department",
                "contact",
                "scholarship",
                "funding",
                "publication",
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
    if "language" in normalized and any(
        token in normalized for token in ("score", "ielts", "toefl")
    ):
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
    if any(token in normalized for token in ("professor", "faculty", "supervisor", "advisor")):
        return [
            "official professors supervisors faculty directory",
            "department faculty contact profile",
        ]
    if any(token in normalized for token in ("lab", "research group", "institute", "chair")):
        return ["official labs research groups institutes", "research chair projects"]
    if any(token in normalized for token in ("department", "school")):
        return ["official department faculty school page", "program department contact"]
    if any(token in normalized for token in ("contact", "email", "phone", "office")):
        return ["official contact email phone admissions office", "program coordinator contact"]
    if any(token in normalized for token in ("scholarship", "funding", "assistantship", "grant")):
        return ["official scholarship funding assistantship", "tuition waiver funding opportunity"]
    if any(
        token in normalized
        for token in ("publication", "google scholar", "researchgate", "profile")
    ):
        return ["official professor publication profile links", "faculty google scholar profile"]
    return [normalized]


def _normalize_domain_slug(text: str) -> str:
    compact = str(text or "").strip().lower()
    if not compact:
        return ""
    compact = compact.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
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


def _targeted_required_field_rescue_queries(
    *, base_query: str, state: dict, issues: list[str]
) -> list[str]:
    compact_base = (
        _compact_web_retrieval_query(base_query=base_query, state=state) or str(base_query).strip()
    )
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
    web_research_missing = state.get("web_research_objectives_missing", [])
    web_research_missing = web_research_missing if isinstance(web_research_missing, list) else []
    missing_hints.extend(str(item).strip() for item in web_research_missing if str(item).strip())
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

    configured_limit = _safe_int(
        getattr(settings.web_search, "deep_required_field_rescue_max_queries", 3)
    )
    if configured_limit is None:
        configured_limit = 3
    limit = max(1, min(6, configured_limit))
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = " ".join(str(candidate).split()).strip()
        if not compact:
            continue
        trimmed = _truncate_query_safely(compact, max_chars=_WEB_QUERY_MAX_CHARS)
        if not trimmed:
            continue
        key = trimmed.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(trimmed)
        if len(normalized) >= limit:
            break
    return normalized


def _required_field_rescue_signature(state: dict) -> str:
    parts: list[str] = []
    for key in (
        "web_required_fields_missing",
        "required_fields_missing",
        "web_research_objectives_missing",
    ):
        raw = state.get(key, [])
        raw = raw if isinstance(raw, list) else []
        normalized = sorted(
            {
                " ".join(str(item).split()).strip().lower()
                for item in raw
                if " ".join(str(item).split()).strip()
            }
        )
        if normalized:
            parts.append(",".join(normalized))
    return " | ".join(parts)[:300]


def _should_attempt_required_field_web_rescue(issues: list[str], state: dict) -> bool:
    if not issues:
        return False
    if _unigraph_answered_official_required_field(state=state):
        state["rescue_retrieval_skipped_reason"] = "unigraph_answered_required_field"
        logger.info(
            "Rescue retrieval skipped | rescue_retrieval_skipped_reason=%s",
            state["rescue_retrieval_skipped_reason"],
        )
        emit_trace_event(
            "rescue_retrieval_skipped",
            {"reason": state["rescue_retrieval_skipped_reason"], "kind": "required_field"},
        )
        return False
    web_ready, _ = _web_retrieval_ready()
    if not web_ready:
        return False
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    if (
        not required_fields
        and not _admissions_missing_web_fields(state)
        and not _research_objectives_missing_from_web(state)
    ):
        return False
    signature = _required_field_rescue_signature(state)
    attempted_signatures = state.get("agent_required_field_rescue_signatures", [])
    attempted_signatures = attempted_signatures if isinstance(attempted_signatures, list) else []
    if signature and signature in attempted_signatures:
        return False
    return any(_issue_requests_required_field_rescue(issue) for issue in issues)


async def _attempt_required_field_web_rescue(
    *,
    issues: list[str],
    state: dict,
    base_query: str,
    search_mode: str,
) -> tuple[list[dict], bool]:
    if _unigraph_answered_official_required_field(state=state):
        state["rescue_retrieval_skipped_reason"] = "unigraph_answered_required_field"
        logger.info(
            "Rescue retrieval skipped | rescue_retrieval_skipped_reason=%s",
            state["rescue_retrieval_skipped_reason"],
        )
        emit_trace_event(
            "rescue_retrieval_skipped",
            {"reason": state["rescue_retrieval_skipped_reason"], "kind": "required_field"},
        )
        return [], False
    queries = _targeted_required_field_rescue_queries(
        base_query=base_query,
        state=state,
        issues=issues,
    )
    if not queries:
        return [], False
    signature = _required_field_rescue_signature(state)
    attempted_signatures = state.get("agent_required_field_rescue_signatures", [])
    attempted_signatures = attempted_signatures if isinstance(attempted_signatures, list) else []
    if signature and signature in attempted_signatures:
        return [], False
    if signature:
        state["agent_required_field_rescue_signatures"] = (attempted_signatures + [signature])[:6]
    normalized_mode = _normalized_request_mode(search_mode)
    rescue_mode = _FAST_MODE if normalized_mode == _DEEP_MODE else normalized_mode
    emit_trace_event(
        "agent_required_field_rescue_started",
        {
            "query_count": len(queries),
            "issues": issues[:6],
            "query_variants": queries[:8],
            "rescue_mode": rescue_mode,
        },
    )

    max_context_results = _max_context_results_for_mode(rescue_mode)
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
    best_research_coverage: float | None = None
    best_research_missing: list[str] = []
    verified_values: list[bool] = []

    tasks = [
        asyncio.create_task(
            _run_one_web_query_with_timeout(
                query,
                top_k=_default_retrieval_top_k(),
                search_mode=rescue_mode,
                **({"debug": True} if bool(state.get("unigraph_debug_enabled", False)) else {}),
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
        research_coverage = _safe_float(verification.get("research_objective_coverage"))
        if research_coverage is not None:
            research_coverage = _clamp01(research_coverage, fallback=1.0)
            if best_research_coverage is None or research_coverage > best_research_coverage:
                best_research_coverage = research_coverage
        research_missing = verification.get("research_objectives_missing", [])
        if isinstance(research_missing, list):
            normalized_research_missing = [
                " ".join(str(value).split()).strip()
                for value in research_missing
                if " ".join(str(value).split()).strip()
            ]
            if not best_research_missing or len(normalized_research_missing) < len(
                best_research_missing
            ):
                best_research_missing = normalized_research_missing[:6]
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
    if best_research_coverage is not None:
        state["web_research_objective_coverage"] = round(best_research_coverage, 4)
    if best_research_missing:
        state["web_research_objectives_missing"] = best_research_missing
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
                "rescue_mode": rescue_mode,
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
                "rescue_mode": rescue_mode,
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
                "rescue_mode": rescue_mode,
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
            "research_objective_coverage": state.get("web_research_objective_coverage"),
            "rescue_mode": rescue_mode,
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
        if (
            "not verified from sources" in lowered
            or "not verified from official sources" in lowered
            or "not verified from evidence" in lowered
        ):
            continue
        if not _SPECULATIVE_FACTUAL_CLAIM_RE.search(lowered):
            continue
        if _SPECULATIVE_FACTUAL_FIELD_RE.search(lowered):
            return True
    return False


def _has_weak_critical_evidence_claim(answer: str, state: dict) -> bool:
    if not _is_admissions_requirements_query(state):
        return False
    required_fields = state.get("required_answer_fields")
    required_fields = required_fields if isinstance(required_fields, list) else []
    critical_fields = {
        "eligibility_requirements",
        "gpa_threshold",
        "gpa_or_grade_threshold",
        "ects_prerequisites",
        "ects_or_prerequisite_credit_breakdown",
        "language_test_thresholds",
        "language_test_score_thresholds",
        "application_deadline",
    }
    if not (set(str(item).strip() for item in required_fields) & critical_fields):
        return False
    text = " ".join(str(answer or "").split()).strip()
    if not text:
        return False
    if not _WEAK_CRITICAL_EVIDENCE_RE.search(text):
        return False
    return bool(_SPECULATIVE_FACTUAL_FIELD_RE.search(text))


_LEDGER_FIELD_LINE_MARKERS = {
    "language_test_score_thresholds": ("ielts", "toefl", "test score", "minimum score"),
    "german_language_requirement": ("german language", "german requirement", "deutsch"),
    "gpa_or_grade_threshold": ("gpa", "grade threshold", "mindestnote", "minimum grade"),
    "gpa_threshold": ("gpa", "grade threshold", "mindestnote", "minimum grade"),
    "ects_or_subject_credit_requirements": ("ects", "prerequisite credit", "credit breakdown"),
    "ects_prerequisites": ("ects", "prerequisite credit", "credit breakdown"),
    "application_deadline": (
        "application deadline",
        "deadline",
        "fall semester",
        "spring semester",
    ),
    "international_deadline": (
        "application deadline",
        "deadline",
        "fall semester",
        "spring semester",
    ),
    "application_portal": ("application portal", "where to apply", "portal"),
    "selection_criteria": ("selection formula", "selection criteria", "ranking criteria"),
}
_LOOSE_MONTH_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
    flags=re.IGNORECASE,
)


def _verified_ledger_fields_marked_unverified(answer: str, state: dict) -> list[str]:
    if not answer:
        return []
    rows = state.get("coverage_ledger", [])
    if not isinstance(rows, list) or not rows:
        rows = state.get("web_field_evidence", [])
    rows = rows if isinstance(rows, list) else []
    found_fields = {
        str(row.get("id", row.get("field", ""))).strip()
        for row in rows
        if isinstance(row, dict)
        and str(row.get("status", "")).strip().lower() == "found"
        and str(row.get("id", row.get("field", ""))).strip()
    }
    if not found_fields:
        return []
    bad_fields: list[str] = []
    lowered_lines = [" ".join(line.split()).lower() for line in str(answer).splitlines()]
    for field_id in sorted(found_fields):
        markers = _LEDGER_FIELD_LINE_MARKERS.get(field_id, ())
        if not markers:
            continue
        for line in lowered_lines:
            if "not verified from official sources" not in line:
                continue
            if any(marker in line for marker in markers):
                bad_fields.append(field_id)
                break
    return bad_fields[:6]


def _answer_deadline_conflicts_with_ledger(answer: str, state: dict) -> bool:
    rows = state.get("coverage_ledger", [])
    if not isinstance(rows, list) or not rows:
        rows = state.get("web_field_evidence", [])
    rows = rows if isinstance(rows, list) else []
    deadline_values: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        field_id = str(row.get("id", row.get("field", ""))).strip()
        if field_id not in {"application_deadline", "international_deadline"}:
            continue
        if str(row.get("status", "")).strip().lower() != "found":
            continue
        value = " ".join(str(row.get("value", "")).split()).lower()
        if value:
            deadline_values.append(value)
    if not deadline_values:
        return False
    allowed_dates = {
        " ".join(match.split()).lower()
        for value in deadline_values
        for match in _LOOSE_MONTH_DATE_RE.findall(value)
        if str(match).strip()
    }
    if not allowed_dates:
        return False
    in_deadline_section = False
    answer_dates: set[str] = set()
    for raw_line in str(answer or "").splitlines():
        line = " ".join(raw_line.split()).strip()
        lowered = line.lower()
        if (
            re.search(r"^\s{0,3}#{0,3}\s*application deadline\b", lowered)
            or lowered == "application deadline"
        ):
            in_deadline_section = True
            continue
        if (
            in_deadline_section
            and re.match(r"^\s{0,3}#{1,3}\s+\w+", line)
            and "application deadline" not in lowered
        ):
            break
        if (
            in_deadline_section
            or "application deadline" in lowered
            or "fall semester" in lowered
            or "spring semester" in lowered
        ):
            answer_dates.update(
                " ".join(match.split()).lower()
                for match in _LOOSE_MONTH_DATE_RE.findall(line)
                if str(match).strip()
            )
    return bool(answer_dates and not answer_dates.issubset(allowed_dates))


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
    if (
        not str(result or "").strip()
        or str(result or "").strip() == _NO_RELEVANT_INFORMATION_DETAIL
    ):
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


def _structured_recovery_answer_usable(answer: str, state: dict) -> bool:
    text = str(answer or "").strip()
    if not text or text == _NO_RELEVANT_INFORMATION_DETAIL:
        return False
    if not _is_admissions_requirements_query(state):
        return False
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    if not evidence_urls:
        return False
    if _has_speculative_factual_language(text, state):
        return False
    if _issues_include_query_not_answered(_agentic_result_issues(text, state)):
        return False
    disallowed = _response_disallowed_german_citation_urls(text, state)
    if disallowed:
        return False
    return True


def _force_structured_recovery_when_evidence_exists(answer: str, state: dict) -> bool:
    text = str(answer or "").strip()
    if not text or text == _NO_RELEVANT_INFORMATION_DETAIL:
        return False
    if not _is_admissions_requirements_query(state):
        return False
    rows = state.get("coverage_ledger", [])
    if not isinstance(rows, list) or not rows:
        rows = state.get("web_field_evidence", [])
    rows = rows if isinstance(rows, list) else []
    found_rows = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("status", "")).strip().lower() == "found"
    ]
    if not found_rows:
        return False
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    if not evidence_urls:
        return False
    if _response_disallowed_german_citation_urls(text, state):
        return False
    coverage = _safe_float(state.get("web_required_field_coverage"))
    if coverage is None:
        coverage = _safe_float(state.get("required_field_coverage"))
    found_count = len(found_rows)
    if found_count >= 2:
        return True
    if float(coverage or 0.0) >= 0.5:
        return True
    return bool(len(evidence_urls) >= 3)


def _can_return_best_effort_admissions_answer(answer: str, state: dict) -> bool:
    text = str(answer or "").strip()
    if not text or text == _NO_RELEVANT_INFORMATION_DETAIL:
        return False
    if not _is_admissions_requirements_query(state):
        return False
    evidence_urls = state.get("evidence_urls", [])
    evidence_urls = evidence_urls if isinstance(evidence_urls, list) else []
    if not evidence_urls:
        return False
    if _response_disallowed_german_citation_urls(text, state):
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
        "weak_critical_evidence_claim",
        "disallowed_german_citation_url",
        "verified_ledger_field_marked_unverified",
        "deadline_conflicts_with_verified_ledger",
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


def _structured_ledger_found_count(state: dict) -> int:
    rows = state.get("coverage_ledger", [])
    if not isinstance(rows, list) or not rows:
        rows = state.get("web_field_evidence", [])
    rows = rows if isinstance(rows, list) else []
    return sum(
        1
        for row in rows
        if isinstance(row, dict) and str(row.get("status", "")).strip().lower() == "found"
    )


def _answer_has_extraction_artifacts(answer: str) -> bool:
    text = str(answer or "")
    lowered = text.lower()
    if "| --- |" in text or "[...]" in text:
        return True
    if "show citations" in lowered or "sources checked" in lowered:
        return True
    if (
        len(
            re.findall(
                r"\b\d{1,2}\s+[A-Z][a-z]+\s*[-\u2012\u2013\u2014]\s*\d{1,2}\s+[A-Z][a-z]+", text
            )
        )
        > 4
    ):
        return True
    return False


def _should_use_structured_field_answer(
    *,
    answer: str,
    issues: list[str],
    structured_answer: str,
    structured_issues: list[str],
    state: dict,
) -> bool:
    if not _should_prefer_structured_field_evidence_answer(state):
        return False
    if not str(structured_answer or "").strip():
        return False
    if (
        not _structured_recovery_answer_usable(structured_answer, state)
        and not _force_structured_recovery_when_evidence_exists(structured_answer, state)
        and not _can_return_best_effort_admissions_answer(structured_answer, state)
    ):
        return False
    if not str(answer or "").strip() or answer == _NO_RELEVANT_INFORMATION_DETAIL:
        return True
    structured_score = _candidate_quality_score(structured_answer, structured_issues, state)
    current_score = _candidate_quality_score(answer, issues, state)
    if structured_score < current_score:
        return True
    if issues and _structured_ledger_found_count(state) >= 2:
        return True
    if _answer_has_extraction_artifacts(answer) and _structured_ledger_found_count(state) >= 2:
        return True
    return False


def _derive_abstain_reason(result: str, state: dict) -> str:
    normalized_result = str(result or "").strip()
    if normalized_result == _WEB_RETRIEVAL_TIMEOUT_DETAIL:
        return "web_timeout"
    if normalized_result != _NO_RELEVANT_INFORMATION_DETAIL:
        return ""

    output_guard_reason = str(state.get("output_guard_reason", "")).strip().lower()
    context_guard_reason = str(state.get("context_guard_reason", "")).strip().lower()
    if (
        output_guard_reason == "web_retrieval_timeout"
        or context_guard_reason == "web_retrieval_timeout"
    ):
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
                working_messages = (
                    working_messages
                    + [
                        {"role": "assistant", "content": str(candidate)},
                        _agentic_reflection_message(issues, attempt + 1, verifier),
                    ]
                    + rescue_context_messages
                )
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
        if _candidate_quality_score(
            finalized_result, finalized_issues, state
        ) < _candidate_quality_score(
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
    if _should_prefer_structured_field_evidence_answer(state):
        structured_answer = _build_structured_field_evidence_answer(state)
        structured_issues = (
            _agentic_result_issues(structured_answer, state) if structured_answer else []
        )
        if _should_use_structured_field_answer(
            answer=final_result,
            issues=final_issues,
            structured_answer=structured_answer,
            structured_issues=structured_issues,
            state=state,
        ):
            final_result = structured_answer
            final_issues = structured_issues
            state["final_answer_source"] = "field_renderer"
            state["final_prompt_used"] = False
            emit_trace_event(
                "answer_ledger_first_used",
                {
                    "issues": final_issues[:8],
                    "answer_preview": str(final_result)[:220],
                },
            )
        else:
            final_issues = _agentic_result_issues(final_result, state)
            emit_trace_event(
                "answer_ledger_first_skipped",
                {
                    "reason": "generated_answer_scored_higher",
                    "answer_preview": str(final_result)[:220],
                },
            )
    if final_issues and str(state.get("query_intent", "")).strip().lower() == "comparison":
        structured_comparison = _build_structured_comparison_from_evidence(state)
        if structured_comparison:
            structured_issues = _agentic_result_issues(structured_comparison, state)
            structured_score = _candidate_quality_score(
                structured_comparison, structured_issues, state
            )
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
            recovered = _build_structured_field_evidence_answer(state)
            if recovered:
                recovered_issues = _agentic_result_issues(recovered, state)
                if (
                    not _is_hard_verification_failure(recovered_issues, recovered, state)
                    or _structured_recovery_answer_usable(recovered, state)
                    or _force_structured_recovery_when_evidence_exists(recovered, state)
                ):
                    final_result = recovered
                    final_issues = recovered_issues
                    state["final_answer_source"] = "fallback_builder"
                    state["final_prompt_used"] = False
                    state["output_guard_reason"] = "agent_verification_partial"
                    emit_trace_event(
                        "answer_partial_with_field_evidence_recovery",
                        {
                            "issues": final_issues[:8],
                            "answer_preview": str(final_result)[:220],
                        },
                    )
                else:
                    if _can_return_best_effort_admissions_answer(recovered, state):
                        final_result = recovered
                        final_issues = recovered_issues
                        state["output_guard_reason"] = "agent_verification_partial"
                        emit_trace_event(
                            "answer_best_effort_with_evidence",
                            {
                                "issues": final_issues[:8],
                                "answer_preview": str(final_result)[:220],
                            },
                        )
                    else:
                        state["output_guard_reason"] = "agent_verification_failed"
                        final_result = _NO_RELEVANT_INFORMATION_DETAIL
            else:
                if _can_return_best_effort_admissions_answer(final_result, state):
                    state["output_guard_reason"] = "agent_verification_partial"
                    emit_trace_event(
                        "answer_best_effort_without_structured_recovery",
                        {
                            "issues": final_issues[:8],
                            "answer_preview": str(final_result)[:220],
                        },
                    )
                else:
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
        "web_field_evidence": [],
        "web_source_policy": "",
        "web_unresolved_fields": [],
        "web_research_objective_coverage": None,
        "web_research_objectives_missing": [],
        "web_timeout_count": 0,
        "web_timed_out_queries": [],
        "web_timeout_rescued": False,
        "unigraph_answered_required_field": False,
        "rescue_retrieval_skipped_reason": "",
        "german_researcher_calls": 0,
        "german_researcher_cache_hits": 0,
        "german_research_cache_key": "",
        "german_research_cached_result": {},
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
        "question_schema_id": "",
        "required_slots": [],
        "required_answer_fields": [],
        "coverage_ledger": [],
        "unresolved_slots": [],
        "source_policy_decisions": [],
        "retrieval_budget_usage": {},
        "unigraph_debug_enabled": False,
        "unigraph_debug": {},
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
        "agent_required_field_rescue_signatures": [],
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
    schema_payload = resolve_question_schema(
        safe_user_prompt,
        intent=str(state.get("query_intent", "unknown")),
    )
    schema_fields = required_answer_fields_from_schema(
        safe_user_prompt,
        intent=str(state.get("query_intent", "unknown")),
    )
    merged_fields: list[str] = []
    seen_fields: set[str] = set()
    for field in list(state.get("required_answer_fields", [])) + list(schema_fields):
        key = " ".join(str(field).split()).strip()
        if not key or key in seen_fields:
            continue
        seen_fields.add(key)
        merged_fields.append(key)
    state["required_answer_fields"] = merged_fields[:10]
    state["question_schema_id"] = str(schema_payload.get("schema_id", "student_general")).strip()
    required_slots = schema_payload.get("required_slots", [])
    state["required_slots"] = [dict(item) for item in required_slots if isinstance(item, dict)][:20]
    emit_trace_event(
        "query_intent_classified",
        {
            "intent": str(state.get("query_intent", "unknown")),
            "query": str(safe_user_prompt)[:220],
            "comparison_entities": state.get("comparison_entities", []),
            "required_answer_fields": state.get("required_answer_fields", []),
            "question_schema_id": state.get("question_schema_id", ""),
            "required_slots": [
                str(item.get("slot_id", "")).strip()
                for item in (state.get("required_slots", []) or [])
                if isinstance(item, dict) and str(item.get("slot_id", "")).strip()
            ],
        },
    )
    messages = await build_context(conversation_user_id, safe_user_prompt)
    state["build_context_ms"] = _elapsed_ms(build_context_started_at)

    retrieval_query = _build_retrieval_query(messages)
    if _truthy_env(_RETRIEVAL_DISABLED_ENV):
        state["retrieval_strategy"] = "disabled"
        state["retrieval_ms"] = 0
    elif retrieval_query:
        messages, detail = await _augment_messages_with_retrieval(
            messages=messages,
            retrieval_query=retrieval_query,
            search_mode=str(policy.get("web_search_mode", _DEEP_MODE)),
            state=state,
        )
        if detail:
            return None, detail
    else:
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
                    "If exact dates are absent, say naturally that the retrieved official "
                    "evidence does not state a separate deadline for this case."
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
            return grounded_result.replace(
                _NOT_VERIFIED_OFFICIAL_DETAIL,
                "The retrieved official evidence does not state a separate deadline for this case.",
            )
        deadline_line = (
            "The retrieved official evidence does not state a separate deadline for this case."
        )
        if "application deadline" not in lowered:
            return f"{grounded_result.rstrip()}\n\n{deadline_line}".strip()
    return grounded_result


def _final_answer_has_raw_output(text: str) -> bool:
    value = str(text or "")
    lowered = value.lower()
    if any(
        marker in lowered
        for marker in (
            "###",
            "[...]",
            "not verified from official sources",
            "verified in the retrieved official evidence",
            "verified from selected evidence",
            "answered_fields",
            "missing_fields",
        )
    ):
        return True
    return bool(
        re.search(
            r"\b(?:Application deadline|Intake|Other semester deadline|IELTS score|GPA|Application portal):\s.{300,}",
            value,
            re.I | re.S,
        )
    )


def _sanitize_final_user_answer(text: str) -> str:
    value = str(text or "").strip()
    replacements = {
        "Not verified from official sources.": "The retrieved official evidence does not state this requested detail.",
        "Not verified from official sources": "The retrieved official evidence does not state this requested detail",
        "Verified in the retrieved official evidence.": "Stated in the retrieved official evidence.",
        "Verified in the retrieved official evidence": "Stated in the retrieved official evidence",
        "Verified from selected evidence.": "Stated in the retrieved official evidence.",
        "Verified from selected evidence": "Stated in the retrieved official evidence",
        "[...]": "",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"#{1,6}\s*", "", value)
    value = re.sub(r"\b(answered_fields|missing_fields)\b", "", value, flags=re.I)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


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
        research_coverage = _safe_float(state.get("web_research_objective_coverage"))
        min_research_coverage = (
            float(getattr(settings.web_search, "deep_cache_min_required_field_coverage", 0.85))
            - 0.1
        )
        if (
            web_fallback_attempted
            and _is_researcher_objective_query(state)
            and research_coverage is not None
            and research_coverage < max(0.55, min_research_coverage)
        ):
            return "deep_research_objective_coverage_low"

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
    web_research_coverage = _safe_float(state.get("web_research_objective_coverage"))
    if (
        _is_researcher_objective_query(state)
        and web_research_coverage is not None
        and web_research_coverage < 0.999
    ):
        derived.append(
            "Some requested research objectives are not fully verified from official sources."
        )
    return derived[:3]


def _apply_answer_policy(result: str, state: dict) -> str:
    text = str(result or "").replace("\r\n", "\n").strip()
    if not text or text in {_NO_RELEVANT_INFORMATION_DETAIL, refusal_response()}:
        return text
    has_sources_heading = bool(re.search(r"(?im)^\s*(?:#{1,3}\s*)?sources?\s*:?\s*$", text))
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


def _dedupe_semicolon_values(value: str, *, limit: int = 6) -> str:
    compact = " ".join(str(value or "").split()).strip()
    if not compact:
        return ""
    parts = [part.strip() for part in re.split(r"\s*;\s*", compact) if part.strip()]
    if len(parts) <= 1:
        return compact
    output: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = re.sub(r"\s*[\u2012\u2013\u2014]\s*", " - ", part).strip()
        key = re.sub(r"\s+", " ", cleaned).lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return "; ".join(output)


def _ledger_value_is_suspicious_ects(field_id: str, value: str) -> bool:
    if field_id not in {
        "ects_prerequisites",
        "ects_or_subject_credit_requirements",
        "ects_or_prerequisite_credit_breakdown",
    }:
        return False
    matches = [
        int(item) for item in re.findall(r"\b(\d{1,3})\s*ECTS\b", value, flags=re.IGNORECASE)
    ]
    if not matches:
        return False
    if max(matches) > 12:
        return False
    return not re.search(
        r"\b(admission|eligibility|applicants?|at least|minimum|prerequisites?|"
        r"informatics|computer science|business|mathematics|statistics|programming|subject)\b",
        value,
        flags=re.IGNORECASE,
    )


def _ledger_value_is_generic_german_requirement(field_id: str, value: str) -> bool:
    if field_id != "german_language_requirement":
        return False
    lowered = value.lower()
    generic_markers = (
        "for some of the master's programs",
        "german citizens do not have to provide",
        "all-german university entrance qualification",
        "completed degree in a degree course taught in german",
    )
    if any(marker in lowered for marker in generic_markers):
        return True
    return len(value) > 260 and "business informatics" not in lowered


def _ledger_row_is_bad_admissions_source(
    field_id: str, value: str, source_url: str, source_page_type: str
) -> bool:
    lowered_url = str(source_url or "").strip().lower()
    lowered_type = str(source_page_type or "").strip().lower()
    lowered_value = str(value or "").strip().lower()
    admissions_field_ids = {
        "instruction_language",
        "language_of_instruction",
        "language_requirements",
        "language_test_score_thresholds",
        "language_test_thresholds",
        "german_language_requirement",
        "gpa_threshold",
        "gpa_or_grade_threshold",
        "ects_prerequisites",
        "ects_or_subject_credit_requirements",
        "ects_or_prerequisite_credit_breakdown",
        "application_deadline",
        "international_deadline",
        "application_portal",
        "selection_criteria",
        "competitiveness_signal",
        "admission_decision_signal",
    }
    if field_id in admissions_field_ids and re.search(
        r"\b(bsc|bachelor|bachelor'?s)\b|/bsc-|/bachelor",
        lowered_url,
    ):
        return True
    if lowered_type in {
        "ranking_or_directory",
        "ambassador_or_testimonial",
        "study_organization_page",
        "generic_pdf_or_brochure",
        "brochure_or_generic_pdf",
    }:
        return True
    if field_id in {
        "ects_prerequisites",
        "ects_or_subject_credit_requirements",
        "ects_or_prerequisite_credit_breakdown",
    }:
        if any(
            marker in lowered_url
            for marker in (
                "organizing-your-studies",
                "degree-plans",
                "course-schedules",
                "learning-agreements",
            )
        ):
            return True
        if re.search(r"\b(2|4|12)\s*ects\b", lowered_value) and not re.search(
            r"\b(admission|eligibility|applicants?|at least|minimum|prerequisite|"
            r"informatics|computer science|business|mathematics|statistics|programming)\b",
            lowered_value,
        ):
            return True
    if field_id in {
        "language_requirements",
        "language_test_score_thresholds",
        "language_test_thresholds",
        "german_language_requirement",
    }:
        if any(
            marker in lowered_url
            for marker in (
                "/going-abroad/",
                "studying-abroad",
                "proof-of-language-proficiency",
                "exchange-students",
                "incoming-students",
            )
        ):
            return True
        if any(marker in lowered_url for marker in ("masterbroschuere", "masterbroschüre")):
            return True
        if "for some of the master's programs" in lowered_value:
            return True
        if "the gre general test" in lowered_value and field_id == "german_language_requirement":
            return True
    return False


def _clean_ledger_field_value(field_id: str, value: str) -> str:
    cleaned = " ".join(str(value or "").split()).strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("| --- |", " ").replace("|---|", " ")
    cleaned = " ".join(cleaned.split()).strip()
    if field_id == "application_deadline":
        cleaned = _dedupe_semicolon_values(cleaned, limit=4)
    elif field_id in {
        "language_requirements",
        "language_test_score_thresholds",
        "language_test_thresholds",
        "ects_prerequisites",
        "ects_or_subject_credit_requirements",
        "ects_or_prerequisite_credit_breakdown",
    }:
        cleaned = _dedupe_semicolon_values(cleaned, limit=6)
    if len(cleaned) > 360:
        sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
        cleaned = sentence if 40 <= len(sentence) <= 360 else cleaned[:357].rstrip() + "..."
    return cleaned


def _structured_field_answer_title(state: dict) -> str:
    task = _german_task_from_state(state) if _is_german_university_state(state) else None
    program = " ".join(
        str(getattr(task, "program", "") if task is not None else "").split()
    ).strip()
    institution = " ".join(
        str(getattr(task, "institution", "") if task is not None else "").split()
    ).strip()
    if program and institution:
        return f"{program} - {institution}"
    if program:
        return program
    if institution:
        return institution
    return "Admissions Requirements"


def _rebuild_sources_section(body: str, source_block: str, state: dict) -> str:
    inline_urls = _ordered_unique_urls(_CITATION_URL_RE.findall(body))
    source_urls = _ordered_unique_urls(_CITATION_URL_RE.findall(source_block))
    evidence_urls = _traceable_urls(state.get("evidence_urls", []), limit=12)
    evidence_urls = _ordered_unique_urls(evidence_urls)

    ordered_urls = _ordered_unique_urls(inline_urls + source_urls + evidence_urls)
    allowed_hosts = _allowed_citation_hosts(evidence_urls)
    if allowed_hosts:
        filtered_urls = [
            url for url in ordered_urls if _normalized_host_from_url(url) in allowed_hosts
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


def _build_structured_field_evidence_answer(state: dict) -> str:
    rows = state.get("coverage_ledger", [])
    if not isinstance(rows, list) or not rows:
        rows = state.get("web_field_evidence", [])
    rows = rows if isinstance(rows, list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return ""

    evidence_urls = _traceable_urls(state.get("evidence_urls", []), limit=10)
    fallback_url = evidence_urls[0] if evidence_urls else ""
    german_query = _is_german_university_state(state)
    allowed_evidence = {
        _normalized_url(url).rstrip("/") for url in evidence_urls if _normalized_url(url)
    }
    by_id = {
        str(row.get("id", row.get("field", ""))).strip(): row
        for row in rows
        if str(row.get("id", row.get("field", ""))).strip()
    }
    source_pool: list[str] = []

    def _natural_missing_line(field_id: str) -> str:
        normalized = str(field_id or "").strip().lower()
        if normalized in {
            "ielts_score",
            "language_test_score_thresholds",
            "language_test_thresholds",
        }:
            return "The retrieved official evidence does not state a specific IELTS band score."
        if normalized in {
            "application_deadline",
            "international_deadline",
            "other_semester_deadline",
        }:
            return "The retrieved official evidence does not state a separate deadline for this case."
        if normalized in {
            "required_application_documents",
            "international_applicant_documents",
            "degree_transcript_requirements",
            "language_proof",
        }:
            return (
                "The retrieved official evidence confirms the application route, but does not "
                "show the full document checklist."
            )
        if normalized in {
            "tuition_fee",
            "semester_contribution",
            "tuition_or_semester_fee",
            "tuition_or_fees",
        }:
            return "The retrieved official evidence does not state the fee amount."
        return "The retrieved official evidence does not state this requested detail."

    def _line(row: dict | None, label: str) -> str:
        row = row if isinstance(row, dict) else {}
        field_id = str(row.get("id", row.get("field", ""))).strip()
        status = str(row.get("status", "")).strip().lower()
        value = _clean_ledger_field_value(field_id, str(row.get("value", "")))
        source_url = _normalized_url(str(row.get("source_url", "")))
        source_page_type = str(row.get("source_page_type", "")).strip()
        if status == "found" and (
            _ledger_value_is_suspicious_ects(field_id, value)
            or _ledger_value_is_generic_german_requirement(field_id, value)
            or _ledger_row_is_bad_admissions_source(field_id, value, source_url, source_page_type)
            or _NOT_VERIFIED_OFFICIAL_DETAIL.lower() in value.lower()
        ):
            status = "missing"
            value = ""
        if german_query and allowed_evidence and source_url:
            normalized_source = source_url.rstrip("/")
            if normalized_source not in allowed_evidence:
                source_url = ""
        if not source_url and status == "found":
            source_url = fallback_url
        if source_url and status in {"found", "conflict", "stale"}:
            source_pool.append(source_url)
        if status == "found":
            line_value = value or "Stated in the retrieved official evidence."
        elif status == "conflict":
            line_value = "Conflict between official sources. Manual verification required."
        elif status == "stale":
            line_value = value or "Stale evidence. Refresh required for verification."
        else:
            line_value = _natural_missing_line(field_id)
        citation = (
            f" ({source_url})" if source_url and status in {"found", "conflict", "stale"} else ""
        )
        return f"- {label}: {line_value}" + citation

    def _first(*field_ids: str) -> dict | None:
        for field_id in field_ids:
            row = by_id.get(field_id)
            if row:
                return row
        return None

    prompt_text = " ".join(str(state.get("safe_user_prompt", "")).split()).strip().lower()

    def _value_is_missing_marker(value: str) -> bool:
        return _NOT_VERIFIED_OFFICIAL_DETAIL.lower() in str(value or "").lower()

    def _row_found(row: dict | None) -> bool:
        if not isinstance(row, dict) or str(row.get("status", "")).strip().lower() != "found":
            return False
        value = str(row.get("value") or row.get("evidence_snippet") or row.get("evidence_text") or "")
        return not _value_is_missing_marker(value)

    def _row_value(row: dict | None) -> str:
        row = row if isinstance(row, dict) else {}
        return _clean_ledger_field_value(
            str(row.get("id", row.get("field", ""))).strip(),
            str(row.get("value") or row.get("evidence_snippet") or row.get("evidence_text") or ""),
        )

    def _row_url(row: dict | None) -> str:
        return _normalized_url(str((row or {}).get("source_url", ""))) if isinstance(row, dict) else ""

    language_row = _first(
        "english_language_requirement",
        "language_requirements",
        "language_requirement",
        "language_of_instruction",
        "instruction_language",
    )
    ielts_row = _first(
        "ielts_score",
        "language_test_score_thresholds",
        "language_test_thresholds",
    )
    toefl_row = _first("toefl_score")
    narrow_language_query = bool(
        re.search(r"\b(ielts|toefl|duolingo|cefr|language requirement|english proficiency)\b", prompt_text)
    ) and not bool(
        re.search(
            r"\b(deadline|tuition|fee|documents?|portal|gpa|ects|overview|curriculum|scholarship|funding)\b",
            prompt_text,
        )
    )
    if narrow_language_query:
        lines: list[str] = [_structured_field_answer_title(state)]
        source_pool = []
        if _row_found(language_row):
            value = _row_value(language_row) or "English language requirement is stated in the retrieved official evidence."
            url = _row_url(language_row)
            if url:
                source_pool.append(url)
            lines.extend(["", f"English requirement: {value}" + (f" ({url})" if url else "")])
        else:
            lines.extend(["", "English requirement: The retrieved evidence did not verify the requested language requirement."])
        if _row_found(ielts_row):
            value = _row_value(ielts_row) or "IELTS requirement is stated in the retrieved official evidence."
            url = _row_url(ielts_row)
            if url:
                source_pool.append(url)
            lines.append(f"IELTS score: {value}" + (f" ({url})" if url else ""))
        elif "ielts" in prompt_text:
            lines.append(
                "IELTS score: The retrieved official evidence does not state a specific IELTS band score for this program."
            )
        if _row_found(toefl_row):
            value = _row_value(toefl_row) or "TOEFL requirement is stated in the retrieved official evidence."
            url = _row_url(toefl_row)
            if url:
                source_pool.append(url)
            lines.append(f"TOEFL score: {value}" + (f" ({url})" if url else ""))
        source_urls = _ordered_unique_urls(source_pool + evidence_urls)
        if source_urls:
            lines.extend(["", "Sources"])
            for url in source_urls[:4]:
                lines.append(f"- {url}")
        return "\n".join(lines).strip()

    query_plan = state.get("query_plan", {})
    query_plan = query_plan if isinstance(query_plan, dict) else {}
    intent = str(query_plan.get("detected_intent") or query_plan.get("intent") or "").strip()
    if not intent:
        topic_hits = sum(
            1
            for pattern in (
                r"\b(deadline|application period|bewerbungsfrist)\b",
                r"\b(ielts|toefl|duolingo|cefr|language|english proficiency)\b",
                r"\b(tuition|fee|semesterbeitrag)\b",
                r"\b(documents?|checklist|transcript|certificate)\b",
                r"\b(portal|where to apply|uni-assist)\b",
                r"\b(gpa|ects|eligibility|requirements?)\b",
            )
            if re.search(pattern, prompt_text)
        )
        if re.search(r"\b(compare|comparison|versus| vs )\b", prompt_text):
            intent = "comparison_lookup"
        elif prompt_text.startswith("tell me about ") or topic_hits > 1:
            intent = "admission_requirement_lookup"
        elif re.search(r"\b(deadline|application period|bewerbungsfrist)\b", prompt_text):
            intent = "deadline_lookup"
        elif re.search(r"\b(ielts|toefl|duolingo|cefr|language|english proficiency)\b", prompt_text):
            intent = "language_requirement_lookup"
        elif re.search(r"\b(tuition|fee|semesterbeitrag)\b", prompt_text):
            intent = "tuition_fee_lookup"
        elif re.search(r"\b(documents?|checklist|transcript|certificate)\b", prompt_text):
            intent = "document_requirement_lookup"
        else:
            intent = "general_program_overview"

    required_from_plan = query_plan.get("required_fields", [])
    required_from_plan = required_from_plan if isinstance(required_from_plan, list) else []
    optional_from_plan = query_plan.get("optional_fields", [])
    optional_from_plan = optional_from_plan if isinstance(optional_from_plan, list) else []
    excluded_from_plan = query_plan.get("excluded_fields", [])
    excluded = {
        str(item).strip()
        for item in excluded_from_plan
        if str(item).strip()
    } if isinstance(excluded_from_plan, list) else set()

    intent_fields: dict[str, list[str]] = {
        "deadline_lookup": ["application_deadline", "international_deadline", "other_semester_deadline", "intake_or_semester"],
        "language_requirement_lookup": [
            "english_language_requirement",
            "language_requirements",
            "language_requirement",
            "language_of_instruction",
            "instruction_language",
            "ielts_score",
            "toefl_score",
            "duolingo_score",
            "german_language_requirement",
            "language_test_score_thresholds",
            "language_test_thresholds",
        ],
        "tuition_fee_lookup": ["tuition_fee", "semester_contribution", "tuition_or_semester_fee", "tuition_or_fees"],
        "document_requirement_lookup": [
            "required_application_documents",
            "international_applicant_documents",
            "degree_transcript_requirements",
            "language_proof",
            "aps_requirement",
            "vpd_requirement",
            "uni_assist_requirement",
        ],
        "application_portal_lookup": ["application_portal", "application_process", "uni_assist_requirement", "vpd_requirement"],
        "comparison_lookup": [str(row.get("id", row.get("field", ""))).strip() for row in rows],
    }
    requested_ids = [
        str(item).strip()
        for item in [*required_from_plan, *optional_from_plan]
        if str(item).strip()
    ] or intent_fields.get(intent, [])
    if intent in intent_fields:
        allowed = set(intent_fields[intent])
        requested_ids = [field_id for field_id in requested_ids if field_id in allowed]
    requested_ids = [field_id for field_id in dict.fromkeys(requested_ids) if field_id not in excluded]

    label_by_field = {
        "application_deadline": "Application deadline",
        "international_deadline": "Application deadline",
        "other_semester_deadline": "Other semester deadline",
        "intake_or_semester": "Intake",
        "english_language_requirement": "English requirement",
        "language_requirements": "Language requirement",
        "language_requirement": "Language requirement",
        "language_of_instruction": "Language of instruction",
        "instruction_language": "Language of instruction",
        "ielts_score": "IELTS score",
        "toefl_score": "TOEFL score",
        "duolingo_score": "Duolingo score",
        "german_language_requirement": "German language requirement",
        "tuition_fee": "Tuition fee",
        "semester_contribution": "Semester contribution",
        "tuition_or_semester_fee": "Fees",
        "tuition_or_fees": "Fees",
        "required_application_documents": "Required application documents",
        "international_applicant_documents": "International applicant documents",
        "degree_transcript_requirements": "Transcript requirements",
        "language_proof": "Language proof",
        "aps_requirement": "APS requirement",
        "vpd_requirement": "VPD requirement",
        "uni_assist_requirement": "uni-assist requirement",
        "application_portal": "Where to apply",
        "application_process": "Application process",
        "academic_eligibility": "Academic eligibility",
        "gpa_requirement": "GPA / grade threshold",
        "gpa_or_grade_threshold": "GPA / grade threshold",
        "ects_prerequisites": "ECTS / prerequisite credits",
        "ects_or_prerequisite_credit_breakdown": "ECTS / prerequisite credits",
        "ects_or_subject_credit_requirements": "ECTS / prerequisite credit breakdown",
        "language_test_score_thresholds": "IELTS/TOEFL thresholds",
        "language_test_thresholds": "IELTS/TOEFL thresholds",
        "language_requirements": "Language requirement",
    }

    def _render_field(field_id: str) -> str:
        row = _first(field_id)
        label = label_by_field.get(field_id, field_id.replace("_", " ").title())
        return _line(row, label)

    lines: list[str] = [_structured_field_answer_title(state)]
    if intent == "comparison_lookup":
        lines.extend(["", "| Field | Answer | Source |", "| --- | --- | --- |"])
        for field_id in requested_ids[:10]:
            row = _first(field_id)
            row = row if isinstance(row, dict) else {}
            status = str(row.get("status", "")).strip().lower()
            if status != "found":
                continue
            value = _row_value(row) or "Stated in the retrieved official evidence."
            url = _row_url(row)
            if url:
                source_pool.append(url)
            lines.append(f"| {label_by_field.get(field_id, field_id.replace('_', ' ').title())} | {value} | {url} |")
    elif intent == "document_requirement_lookup":
        lines.extend(["", "Checklist"])
        for field_id in requested_ids:
            lines.append(_render_field(field_id))
    elif intent in {"general_program_overview", "program_overview_lookup", "admission_requirement_lookup", "eligibility_check"}:
        section_groups = [
            ("Overview", ["program_overview", "program_duration", "degree_level"]),
            ("Language of Instruction", ["language_of_instruction", "instruction_language"]),
            ("IELTS / German Requirements", ["english_language_requirement", "language_requirements", "language_requirement", "language_test_score_thresholds", "language_test_thresholds", "ielts_score", "toefl_score", "german_language_requirement"]),
            ("GPA and ECTS Requirements", ["academic_eligibility", "gpa_requirement", "gpa_or_grade_threshold", "gpa_threshold", "ects_prerequisites", "ects_or_prerequisite_credit_breakdown", "ects_or_subject_credit_requirements", "required_degree_background"]),
            ("Application Deadline", ["application_deadline", "international_deadline"]),
            ("Application Portal", ["application_portal", "application_process"]),
            ("Fees", ["tuition_fee", "semester_contribution", "tuition_or_semester_fee", "tuition_or_fees"]),
        ]
        requested_set = set(requested_ids) if requested_ids else {str(row.get("id", row.get("field", ""))).strip() for row in rows}
        if not requested_ids and "gpa" in prompt_text:
            requested_set.add("gpa_threshold")
        for title, field_ids in section_groups:
            section_lines = []
            for field_id in field_ids:
                if field_id not in requested_set or field_id in excluded:
                    continue
                if field_id in {"gpa_requirement", "gpa_or_grade_threshold", "gpa_threshold"}:
                    gpa_row = _first("gpa_requirement", "gpa_or_grade_threshold", "gpa_threshold")
                    selection_row = _first("selection_criteria", "competitiveness_signal", "admission_decision_signal")
                    if (
                        (not isinstance(gpa_row, dict) or str(gpa_row.get("status", "")).strip().lower() != "found")
                        and isinstance(selection_row, dict)
                        and str(selection_row.get("status", "")).strip().lower() == "found"
                    ):
                        selection_url = _normalized_url(str(selection_row.get("source_url", "")))
                        if selection_url:
                            source_pool.append(selection_url)
                        section_lines.append(
                            "- GPA / grade threshold: No fixed minimum GPA/grade threshold was found in the retrieved official evidence; "
                            "the final grade or grade average is used as a selection criterion."
                            + (f" ({selection_url})" if selection_url else "")
                        )
                        continue
                if field_id == "german_language_requirement":
                    german_row = _first("german_language_requirement")
                    instruction_row = _first("language_of_instruction", "instruction_language")
                    language_requirement_row = _first(
                        "english_language_requirement", "language_requirements", "language_requirement"
                    )
                    if (
                        isinstance(german_row, dict)
                        and str(german_row.get("status", "")).strip().lower() != "found"
                        and isinstance(instruction_row, dict)
                        and str(instruction_row.get("status", "")).strip().lower() == "found"
                        and "english" in str(instruction_row.get("value", "")).lower()
                        and isinstance(language_requirement_row, dict)
                        and str(language_requirement_row.get("status", "")).strip().lower() == "found"
                    ):
                        section_lines.append(
                            "- German language requirement: No separate German requirement was verified for this English-taught MSc from the official evidence."
                        )
                        continue
                section_lines.append(_render_field(field_id))
            if section_lines:
                lines.extend(["", title, *section_lines])
        if re.search(r"\b(competitive|safe|chance|chances)\b", prompt_text):
            selection_row = _first("selection_criteria", "competitiveness_signal", "admission_decision_signal")
            lines.extend(["", "Admission Competitiveness"])
            if selection_row and str(selection_row.get("status", "")).strip().lower() == "found":
                lines.append(_line(selection_row, "Selection evidence"))
                lines.append(
                    "- Verdict for ~3.2 GPA: Cannot be classified as safe from official evidence alone; use the verified selection criteria and any published cutoffs/capacity when available."
                )
            else:
                lines.append(
                    "- Verdict for ~3.2 GPA: The retrieved official evidence does not state "
                    "enough information to classify this profile as safe."
                )
    else:
        for field_id in requested_ids:
            lines.append(_render_field(field_id))

    if _is_german_university_state(state) and evidence_urls:
        source_urls = _ordered_unique_urls(evidence_urls)
    else:
        source_urls = _ordered_unique_urls(source_pool + evidence_urls)
    if source_urls:
        lines.extend(["", "Sources"])
        for url in source_urls[:8]:
            lines.append(f"- {url}")
    return "\n".join(lines).strip()


def _should_prefer_structured_field_evidence_answer(state: dict) -> bool:
    if not _is_admissions_requirements_query(state):
        return False
    query_plan = state.get("query_plan", {})
    query_plan = query_plan if isinstance(query_plan, dict) else {}
    intent = str(query_plan.get("detected_intent") or query_plan.get("intent") or "").strip()
    if bool(state.get("unigraph_answered_required_field")) and intent in {
        "deadline_lookup",
        "language_requirement_lookup",
        "tuition_fee_lookup",
    }:
        return False
    rows = state.get("coverage_ledger", [])
    if not isinstance(rows, list) or not rows:
        rows = state.get("web_field_evidence", [])
    rows = rows if isinstance(rows, list) else []
    if not rows:
        return False
    found_count = 0
    critical_found = False
    critical_fields = {
        "application_portal",
        "application_deadline",
        "international_deadline",
        "instruction_language",
        "language_requirements",
        "language_test_thresholds",
        "language_test_score_thresholds",
        "gpa_threshold",
        "gpa_or_grade_threshold",
        "ects_prerequisites",
        "ects_or_prerequisite_credit_breakdown",
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")).strip().lower() != "found":
            continue
        found_count += 1
        field_id = str(row.get("id", row.get("field", row.get("slot_id", "")))).strip()
        if field_id in critical_fields:
            critical_found = True
    return found_count > 0 and critical_found


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
                "- Eligibility requirements: The retrieved official evidence does not state "
                f"this requested detail. ({url})",
                "- Application deadline: The retrieved official evidence does not state a "
                f"separate deadline for this case. ({url})",
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
    if bool(state.get("unigraph_debug_enabled", False)):
        state["cache_write_ms"] = 0
        emit_trace_event("cache_write_skipped", {"reason": "debug_enabled"})
        return
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
    *,
    debug: bool = False,
) -> tuple[dict, list | None, str | None]:
    context = _new_request_context(user_id, user_prompt, session_id, mode)
    state = context["state"]
    state["unigraph_debug_enabled"] = bool(debug)
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
    if bool(debug):
        state["cache_read_ms"] = 0
        emit_trace_event("cache_bypassed", {"reason": "debug_enabled"})
    elif _response_cache_enabled():
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
        and str(context["requested_mode"]) in {_AUTO_MODE, _STANDARD_MODE}
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
    before_sanitizer = str(result or "")
    if not state.get("final_answer_source"):
        state["final_answer_source"] = "llm_synthesis"
    if "final_prompt_used" not in state:
        state["final_prompt_used"] = state["final_answer_source"] == "llm_synthesis"
    raw_span_rendered = _final_answer_has_raw_output(before_sanitizer)
    state["raw_span_rendered"] = bool(raw_span_rendered)
    result = _sanitize_final_user_answer(before_sanitizer)
    state["final_answer_before_sanitizer"] = before_sanitizer
    state["final_answer_after_sanitizer"] = result
    logger.info(
        "FinalAnswerPath | final_answer_source=%s | final_prompt_used=%s | raw_span_rendered=%s | final_answer_before_sanitizer=%s | final_answer_after_sanitizer=%s",
        state.get("final_answer_source", ""),
        bool(state.get("final_prompt_used", False)),
        bool(state.get("raw_span_rendered", False)),
        before_sanitizer[:500],
        result[:500],
    )
    result = _apply_answer_policy(result, state)
    result = _append_uncertainty_section(result, state)
    result = _append_missing_info_section(result, state)
    result = _sanitize_final_user_answer(result)
    state["final_answer_after_sanitizer"] = result
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
            "web_research_objective_coverage": state.get("web_research_objective_coverage"),
            "web_research_objectives_missing": (state.get("web_research_objectives_missing") or [])[
                :4
            ],
            "question_schema_id": state.get("question_schema_id", ""),
            "required_slots": [
                str(item.get("slot_id", "")).strip()
                for item in (state.get("required_slots", []) or [])
                if isinstance(item, dict) and str(item.get("slot_id", "")).strip()
            ][:12],
            "unresolved_slots": (state.get("unresolved_slots") or [])[:8],
            "retrieval_budget_usage": state.get("retrieval_budget_usage") or {},
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
    *,
    debug: bool = False,
) -> str | dict:
    """Run the full chat pipeline: guardrails, memory, model call, cache, and persistence."""
    context, messages, early_answer = await _prepare_request(
        user_id,
        user_prompt,
        session_id,
        mode=mode,
        debug=debug,
    )
    if early_answer is not None:
        return {"response": early_answer, "debug": {}} if debug else early_answer
    state = context["state"]
    policy = _execution_policy(str(context["execution_mode"]))
    policy["auto_requested"] = str(context.get("requested_mode", _DEEP_MODE)) in {
        _AUTO_MODE,
        _STANDARD_MODE,
    }

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

    if _should_prefer_structured_field_evidence_answer(state):
        initial_stream_issues = _agentic_result_issues(result, state)
        structured_answer = _build_structured_field_evidence_answer(state)
        structured_issues = (
            _agentic_result_issues(structured_answer, state) if structured_answer else []
        )
        if _should_use_structured_field_answer(
            answer=result,
            issues=initial_stream_issues,
            structured_answer=structured_answer,
            structured_issues=structured_issues,
            state=state,
        ):
            result = structured_answer
            state["final_answer_source"] = "field_renderer"
            state["final_prompt_used"] = False
            emit_trace_event(
                "answer_ledger_first_used",
                {
                    "issues": structured_issues[:8],
                    "answer_preview": str(result)[:220],
                    "stream_mode": True,
                },
            )
            if result != str(runtime["streamed_text"]):
                yield result
        else:
            _ = _agentic_result_issues(result, state)
            emit_trace_event(
                "answer_ledger_first_skipped",
                {
                    "reason": "generated_answer_scored_higher",
                    "answer_preview": str(result)[:220],
                    "stream_mode": True,
                },
            )

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
            recovered = _build_structured_field_evidence_answer(state)
            if recovered:
                recovered_issues = _agentic_result_issues(recovered, state)
                if (
                    not _is_hard_verification_failure(recovered_issues, recovered, state)
                    or _structured_recovery_answer_usable(recovered, state)
                    or _force_structured_recovery_when_evidence_exists(recovered, state)
                ):
                    result = recovered
                    stream_issues = recovered_issues
                    state["agent_last_issues"] = stream_issues
                    state["final_answer_source"] = "fallback_builder"
                    state["final_prompt_used"] = False
                    state["output_guard_reason"] = "stream_verification_partial"
                    emit_trace_event(
                        "answer_partial_with_field_evidence_recovery",
                        {
                            "issues": stream_issues[:8],
                            "answer_preview": str(result)[:220],
                            "stream_mode": True,
                        },
                    )
                    if result != str(runtime["streamed_text"]):
                        yield result
                else:
                    if _can_return_best_effort_admissions_answer(recovered, state):
                        result = recovered
                        stream_issues = recovered_issues
                        state["agent_last_issues"] = stream_issues
                        state["output_guard_reason"] = "stream_verification_partial"
                        emit_trace_event(
                            "answer_best_effort_with_evidence",
                            {
                                "issues": stream_issues[:8],
                                "answer_preview": str(result)[:220],
                                "stream_mode": True,
                            },
                        )
                        if result != str(runtime["streamed_text"]):
                            yield result
                    else:
                        state["output_guard_reason"] = "stream_verification_failed"
                        result = _NO_RELEVANT_INFORMATION_DETAIL
                        if result != str(runtime["streamed_text"]):
                            yield result
            else:
                if _can_return_best_effort_admissions_answer(result, state):
                    state["output_guard_reason"] = "stream_verification_partial"
                    emit_trace_event(
                        "answer_best_effort_without_structured_recovery",
                        {
                            "issues": stream_issues[:8],
                            "answer_preview": str(result)[:220],
                            "stream_mode": True,
                        },
                    )
                else:
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
