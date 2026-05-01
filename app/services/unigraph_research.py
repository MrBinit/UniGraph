"""
UniGraph Phase 1: fast official-source university research.

This module deliberately avoids deep research, vector storage, persistent cache,
multi-agent orchestration, and verification loops. It plans once, fans out to a
small number of targeted searches, extracts bounded evidence from official pages
and PDFs, then answers only from selected evidence chunks.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency in minimal test envs
    pdfplumber = None

from app.core.config import get_settings
from app.services.tavily_search_service import aextract_urls, asearch_google

settings = get_settings()
logger = logging.getLogger(__name__)

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "fbclid",
    "gclid",
}
LOW_QUALITY_DOMAINS = (
    "reddit.",
    "quora.",
    "facebook.",
    "instagram.",
    "youtube.",
    "medium.com",
    "blogspot.",
)
URL_PATH_PENALTY_TERMS = (
    "outgoing",
    "exchange",
    "news",
    "event",
    "events",
    "press",
    "language-center",
    "language-centre",
    "sprachenzentrum",
    "alumni",
    "cris.",
    "/projects/",
    "library",
    "/searching/",
    "theses",
    "publications/theses",
    "pattern-recognition-blog",
    "faudir",
    "written-assignments",
)
URL_PATH_BOOST_TERMS = (
    "admission",
    "application",
    "apply",
    "requirements",
    "degree-program",
    "degree-programme",
    "study-program",
    "studies",
    "prospective-students",
    "language-proficiency",
    "application-faq",
    "faq",
    "program",
    "master",
    "msc",
)
STUDENT_PAGE_SIGNALS = (
    "admission",
    "application",
    "apply",
    "requirements",
    "degree-program",
    "degree-programme",
    "study-program",
    "studies/degree",
    "prospective-students",
    "language-proficiency",
    "application-faq",
    "faq",
    "master-ai",
    "master programme",
    "master-programme",
    "supporting documents",
    "beizufügende unterlagen",
    "beizufuegende unterlagen",
    "studium",
    "international-applicants",
    "language requirement",
    "language requirements",
    "language proficiency",
    "english certificate",
    "cefr",
    "qualification",
    "tuition",
    "fees",
    "semesterbeitrag",
    "deadline",
    "deadlines",
    "application deadline",
    "application period",
    "dates and deadlines",
    "bewerbungsfrist",
    "bewerbungszeitraum",
    "bewerbungsportal",
    "application platform",
    )
NON_STUDENT_PAGE_SIGNALS = (
    "cris.",
    "/projects/",
    "/publications/",
    "publications/theses",
    "/theses/",
    "library",
    "/searching/",
    "ai-tools",
    "pattern-recognition-blog",
    "faudir",
    "written-assignments",
    "module handbook",
    "modulhandbuch",
)
INTENT_CHUNK_GATES: dict[str, dict[str, tuple[str, ...]]] = {
    "language_requirement_lookup": {
        "allow": (
            "ielts",
            "toefl",
            "duolingo",
            "cefr",
            "english proficiency",
            "language requirement",
            "language of instruction",
            "sprachnachweis",
            "sprachkenntnisse",
            "b2",
            "c1",
        ),
        "exclude": (
            "gre",
            "gmat",
            "motivation letter",
            "gpa",
            "tuition",
            "deadline",
            "application period",
            "documents",
            "transcript",
            "duration",
            "module handbook",
            "curriculum",
        ),
    },
    "deadline_lookup": {
        "allow": (
            "deadline",
            "application period",
            "bewerbungsfrist",
            "winter semester",
            "summer semester",
            "intake",
            "eu",
            "non-eu",
            "international applicant",
        ),
        "exclude": (
            "ielts",
            "toefl",
            "gpa",
            "tuition",
            "documents",
            "module handbook",
            "curriculum",
        ),
    },
    "application_portal_lookup": {
        "allow": (
            "application portal",
            "apply online",
            "uni-assist",
            "tumonline",
            "campo",
            "direct application",
            "vpd",
            "application process",
        ),
        "exclude": ("ielts", "toefl", "gpa", "tuition", "curriculum"),
    },
    "tuition_fee_lookup": {
        "allow": (
            "tuition",
            "tuition fee",
            "semester contribution",
            "semesterbeitrag",
            "studiengebühren",
            "non-eu",
            "eu",
            "fee exemption",
            "fees",
        ),
        "exclude": (
            "curriculum",
            "deadline",
            "language requirement",
            "application portal",
            "scholarship",
            "funding",
            "gpa",
            "documents",
        ),
    },
}
GERMAN_UNIVERSITY_FOCUS = True
AMBIGUOUS_GERMAN_UNIVERSITIES = {
    "fau": {
        "preferred_name": "Friedrich-Alexander-Universität Erlangen-Nürnberg",
        "preferred_domain": "fau.de",
        "secondary_name": "Florida Atlantic University",
        "secondary_domains": ("fau.edu",),
    }
}
LANGUAGE_TERMS = (
    "ielts",
    "toefl",
    "duolingo",
    "english",
    "language",
    "proficiency",
    "b2",
    "c1",
    "cefr",
    "sprachnachweis",
    "sprachkenntnisse",
    "englisch",
)
LANGUAGE_UNRELATED_TERMS = (
    "tuition",
    "fee",
    "semesterbeitrag",
    "studiengebühren",
    "gpa",
    "grade",
    "duration",
    "semester",
    "transcript",
    "document",
    "documents",
    "gre",
    "curriculum",
    "module",
)
VALUE_REQUIRED_FIELDS = {
    "ielts_score",
    "toefl_score",
    "duolingo_score",
    "gpa_requirement",
    "tuition_fee",
    "semester_contribution",
    "application_deadline",
    "program_duration",
}
KNOWN_PROGRAM_MARKERS = (
    "artificial intelligence",
    "data science",
    "informatics",
    "computer science",
    "management",
    "business informatics",
    "electrical engineering",
    "mechanical engineering",
)
OFFICIAL_KEYWORDS = (
    "admission",
    "admissions",
    "application",
    "apply",
    "deadline",
    "requirements",
    "bewerbung",
    "bewerbungsfrist",
    "zulassung",
    "zulassungsvoraussetzungen",
    "sprachnachweis",
    "unterlagen",
    "modulhandbuch",
    "pruefungsordnung",
    "prüfungsordnung",
    "semesterbeitrag",
    "studiengebuehren",
    "studiengebühren",
)
GERMAN_SEARCH_TERMS = [
    "Zulassungsvoraussetzungen",
    "Bewerbungsfrist",
    "Sprachnachweis",
    "erforderliche Unterlagen",
    "Pruefungsordnung",
    "Prüfungsordnung",
    "Modulhandbuch",
    "Semesterbeitrag",
    "Studiengebühren",
]

FIELD_KEYWORDS: dict[str, list[str]] = {
    "english_language_requirement": [
        "english",
        "language",
        "language proof",
        "proficiency",
        "cefr",
        "b2",
        "c1",
        "sprachnachweis",
        "sprachkenntnisse",
        "englisch",
    ],
    "ielts_score": ["ielts", "band", "overall", "minimum score"],
    "toefl_score": ["toefl", "internet-based", "ibt"],
    "duolingo_score": ["duolingo", "det"],
    "german_language_requirement": ["german language", "deutsch", "dsh", "testdaf"],
    "application_deadline": [
        "deadline",
        "application period",
        "apply by",
        "bewerbungsfrist",
        "bewerbungszeitraum",
        "winter semester",
        "summer semester",
    ],
    "other_semester_deadline": [
        "summer semester",
        "winter semester",
        "application period",
        "deadline",
        "bewerbungsfrist",
    ],
    "intake_or_semester": ["intake", "winter semester", "summer semester", "semester"],
    "applicant_category": ["international", "eu", "non-eu", "applicant", "bewerber"],
    "academic_eligibility": [
        "eligibility",
        "admission requirement",
        "academic requirement",
        "zulassungsvoraussetzungen",
        "qualification",
    ],
    "gpa_requirement": ["gpa", "grade", "minimum grade", "final grade", "average grade", "note"],
    "required_degree_background": [
        "bachelor",
        "degree",
        "subject",
        "background",
        "credits",
        "ects",
    ],
    "admission_restrictions": ["restricted admission", "selection", "aptitude", "nc"],
    "required_application_documents": [
        "documents",
        "checklist",
        "required documents",
        "application documents",
        "unterlagen",
        "certificate",
        "transcript",
    ],
    "international_applicant_documents": ["international", "visa", "passport", "foreign"],
    "language_proof": ["language proof", "ielts", "toefl", "duolingo", "sprachnachweis"],
    "degree_transcript_requirements": ["degree certificate", "transcript", "diploma"],
    "aps_requirement": ["aps", "academic evaluation centre", "akademische prüfstelle"],
    "vpd_requirement": ["vpd", "preliminary review documentation", "vorprüfungsdokumentation"],
    "uni_assist_requirement": ["uni-assist", "uni assist"],
    "tuition_fee": ["tuition", "fees", "studiengebühren", "tuition fee"],
    "semester_contribution": ["semester contribution", "semesterbeitrag", "student services fee"],
    "gre_gmat_requirement": ["gre", "gmat"],
    "program_shortlist": [
        "master",
        "msc",
        "program",
        "degree programme",
        "artificial intelligence",
        "data science",
    ],
    "teaching_language": [
        "language of instruction",
        "teaching language",
        "english-taught",
        "german-taught",
    ],
    "program_duration": ["duration", "standard period", "semesters", "regelstudienzeit"],
    "curriculum_modules": ["curriculum", "module", "module handbook", "modulhandbuch", "courses"],
    "scholarship_funding": ["scholarship", "funding", "financial aid", "daad scholarship"],
    "application_process": [
        "how to apply",
        "application portal",
        "apply online",
        "application process",
    ],
    "general_information": ["program", "degree", "study", "university"],
}

INTENT_EXTRACT_TERMS: dict[str, list[str]] = {
    "deadline_lookup": [
        "Application Period",
        "Application deadline",
        "Important Dates and Deadlines",
        "Winter semester",
        "Summer semester",
        "February",
        "May",
        "October",
        "November",
        "01.02",
        "31.05",
        "01.10",
        "30.11",
        "Bewerbungsfrist",
        "Bewerbungszeitraum",
    ],
    "language_requirement_lookup": [
        "language requirement",
        "language proficiency",
        "English",
        "German",
        "IELTS",
        "TOEFL",
        "CEFR",
        "B1",
        "B2",
        "C1",
        "Sprachnachweis",
        "Englischkenntnisse",
        "Deutschkenntnisse",
    ],
    "tuition_fee_lookup": [
        "tuition fee",
        "semester fee",
        "semester contribution",
        "non-EU",
        "international students",
        "Studiengebühren",
        "Semesterbeitrag",
    ],
    "document_requirement_lookup": [
        "required documents",
        "application documents",
        "transcript",
        "degree certificate",
        "CV",
        "motivation letter",
        "language certificate",
        "APS",
        "VPD",
        "Bewerbungsunterlagen",
        "beizufügende Unterlagen",
        "beizufuegende Unterlagen",
    ],
    "application_portal_lookup": [
        "apply online",
        "application portal",
        "TUMonline",
        "campo",
        "uni-assist",
        "application platform",
        "Bewerbungsportal",
    ],
    "curriculum_lookup": [
        "modules",
        "curriculum",
        "study plan",
        "module handbook",
        "credits",
        "ECTS",
        "Modulhandbuch",
        "Studienplan",
    ],
}

INTENT_PROFILES: dict[str, dict[str, list[str]]] = {
    "language_requirement_lookup": {
        "required": ["english_language_requirement"],
        "optional": ["ielts_score", "toefl_score", "duolingo_score", "german_language_requirement"],
        "excluded": [
            "tuition_fee",
            "semester_contribution",
            "gpa_requirement",
            "required_application_documents",
            "application_deadline",
            "program_duration",
            "curriculum_modules",
            "gre_gmat_requirement",
        ],
    },
    "deadline_lookup": {
        "required": ["application_deadline"],
        "optional": ["intake_or_semester", "applicant_category", "other_semester_deadline"],
        "excluded": [
            "english_language_requirement",
            "ielts_score",
            "toefl_score",
            "duolingo_score",
            "german_language_requirement",
            "gpa_requirement",
            "required_degree_background",
            "tuition_fee",
            "semester_contribution",
            "required_application_documents",
            "curriculum_modules",
            "application_process",
        ],
    },
    "eligibility_check": {
        "required": [
            "english_language_requirement",
            "academic_eligibility",
            "gpa_requirement",
            "required_degree_background",
        ],
        "optional": ["admission_restrictions", "application_deadline", "ielts_score"],
        "excluded": ["tuition_fee", "curriculum_modules", "scholarship_funding"],
    },
    "document_requirement_lookup": {
        "required": [
            "required_application_documents",
            "international_applicant_documents",
            "language_proof",
            "degree_transcript_requirements",
        ],
        "optional": ["aps_requirement", "vpd_requirement", "uni_assist_requirement"],
        "excluded": ["tuition_fee", "curriculum_modules", "scholarship_funding"],
    },
    "tuition_fee_lookup": {
        "required": ["tuition_fee"],
        "optional": ["semester_contribution", "applicant_category"],
        "excluded": ["ielts_score", "gpa_requirement", "curriculum_modules"],
    },
    "admission_requirement_lookup": {
        "required": ["academic_eligibility", "required_degree_background"],
        "optional": ["gpa_requirement", "english_language_requirement", "admission_restrictions"],
        "excluded": ["tuition_fee", "curriculum_modules", "scholarship_funding"],
    },
    "program_overview_lookup": {
        "required": ["teaching_language", "program_duration"],
        "optional": ["tuition_fee", "application_deadline"],
        "excluded": ["required_application_documents"],
    },
    "curriculum_lookup": {
        "required": ["curriculum_modules"],
        "optional": ["program_duration", "teaching_language"],
        "excluded": ["tuition_fee", "application_deadline", "ielts_score"],
    },
    "application_process_lookup": {
        "required": ["application_process"],
        "optional": [
            "application_deadline",
            "uni_assist_requirement",
            "aps_requirement",
            "vpd_requirement",
        ],
        "excluded": ["tuition_fee", "curriculum_modules"],
    },
    "application_portal_lookup": {
        "required": ["application_process"],
        "optional": ["uni_assist_requirement", "vpd_requirement", "aps_requirement"],
        "excluded": [
            "english_language_requirement",
            "ielts_score",
            "gpa_requirement",
            "tuition_fee",
            "curriculum_modules",
            "application_deadline",
        ],
    },
    "multi_program_discovery": {
        "required": ["program_shortlist", "english_language_requirement"],
        "optional": ["ielts_score", "toefl_score", "duolingo_score", "teaching_language"],
        "excluded": [
            "tuition_fee",
            "application_deadline",
            "required_application_documents",
            "gpa_requirement",
            "curriculum_modules",
        ],
    },
    "scholarship_funding_lookup": {
        "required": ["scholarship_funding"],
        "optional": ["tuition_fee"],
        "excluded": ["ielts_score", "gpa_requirement", "curriculum_modules"],
    },
    "general_university_question": {
        "required": ["general_information"],
        "optional": [],
        "excluded": [],
    },
}

KNOWN_UNIVERSITIES: dict[str, dict[str, Any]] = {
    "university of mannheim": {
        "name": "University of Mannheim",
        "short": "Mannheim",
        "domains": ["uni-mannheim.de"],
        "aliases": ["university of mannheim", "uni mannheim", "mannheim"],
    },
    "technical university of munich": {
        "name": "Technical University of Munich",
        "short": "TUM",
        "domains": ["tum.de", "cit.tum.de", "campus.tum.de"],
        "aliases": ["technical university of munich", "tu munich", "tum", "t.u. munich"],
    },
    "friedrich-alexander-universität erlangen-nürnberg": {
        "name": "Friedrich-Alexander-Universität Erlangen-Nürnberg",
        "short": "FAU",
        "domains": [
            "fau.de",
            "fau.eu",
            "ai.study.fau.eu",
            "www.ai.study.fau.eu",
            "studium.fau.de",
            "informatik.studium.fau.de",
            "tf.fau.de",
        ],
        "aliases": ["fau", "fau erlangen", "erlangen nürnberg", "erlangen-nürnberg"],
    },
    "universität hamburg": {
        "name": "Universität Hamburg",
        "short": "UHH",
        "domains": ["uni-hamburg.de"],
        "aliases": ["university of hamburg", "universität hamburg", "uni hamburg", "uhh"],
    },
    "ludwig-maximilians-universität münchen": {
        "name": "Ludwig-Maximilians-Universität München",
        "short": "LMU",
        "domains": ["lmu.de"],
        "aliases": ["lmu", "lmu munich", "ludwig maximilians"],
    },
    "rwth aachen university": {
        "name": "RWTH Aachen University",
        "short": "RWTH",
        "domains": ["rwth-aachen.de"],
        "aliases": ["rwth", "rwth aachen"],
    },
    "karlsruhe institute of technology": {
        "name": "Karlsruhe Institute of Technology",
        "short": "KIT",
        "domains": ["kit.edu"],
        "aliases": ["kit", "karlsruhe institute of technology"],
    },
}
OFFICIAL_SECONDARY_DOMAINS = ("daad.de", "www2.daad.de", "uni-assist.de")
UNIVERSITY_DOMAIN_CONFIG: dict[str, dict[str, Any]] = {
    str(record["short"]): {
        "aliases": record.get("aliases", []),
        "root_domains": [
            domain for domain in record.get("domains", []) if len(str(domain).split(".")) <= 2
        ],
        "known_subdomains": [
            domain for domain in record.get("domains", []) if len(str(domain).split(".")) > 2
        ],
        "preferred_program_admission_patterns": (
            "degree-program",
            "study",
            "studium",
            "admission",
            "application",
            "bewerbung",
            "zulassung",
        ),
    }
    for record in KNOWN_UNIVERSITIES.values()
}


def _cfg_int(name: str, default: int, *, minimum: int = 1, maximum: int = 100) -> int:
    value = getattr(settings.web_search, name, default)
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


MAX_QUERIES = _cfg_int("phase1_max_queries", 5, maximum=8)
MAX_RESULTS_PER_QUERY = _cfg_int("phase1_max_results_per_query", 3, maximum=8)
MAX_TOTAL_URLS_TO_FETCH = _cfg_int("phase1_max_total_urls_to_fetch", 8, maximum=16)
MAX_PDFS_TO_READ = _cfg_int("phase1_max_pdfs_to_read", 3, maximum=6)
MAX_PDF_SIZE_MB = _cfg_int("phase1_max_pdf_size_mb", 15, maximum=50)
MAX_PDF_PAGES = _cfg_int("phase1_max_pdf_pages", 40, maximum=100)
MAX_EVIDENCE_CHUNKS = _cfg_int("phase1_max_evidence_chunks", 12, maximum=30)
CHUNK_CHARS = _cfg_int("page_chunk_chars", 850, minimum=250, maximum=4000)
CHUNK_OVERLAP = _cfg_int("page_chunk_overlap_chars", 120, minimum=0, maximum=1000)
DEBUG_ARTIFACT_DIR = os.getenv("UNIGRAPH_DEBUG_DIR", "data/debug/unigraph")

QUERY_MODE_LIMITS: dict[str, dict[str, int]] = {
    "fast_lookup": {
        "max_queries": 3,
        "max_urls": 3,
        "max_pdfs": 1,
        "max_pdf_pages": 12,
        "max_programs": 1,
    },
    "research_lookup": {
        "max_queries": 5,
        "max_urls": 6,
        "max_pdfs": 2,
        "max_pdf_pages": 24,
        "max_programs": 1,
    },
    "discovery_lookup": {
        "max_queries": 8,
        "max_urls": 10,
        "max_pdfs": 3,
        "max_pdf_pages": 30,
        "max_programs": 5,
    },
}


@dataclass
class QueryPlan:
    university: str = ""
    university_short: str = ""
    program: str = ""
    country: str = ""
    degree_level: str = ""
    user_intent: str = ""
    intent: str = "general_university_question"
    required_info: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    optional_fields: list[str] = field(default_factory=list)
    excluded_fields: list[str] = field(default_factory=list)
    user_profile_details: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    german_keywords: list[str] = field(default_factory=list)
    search_queries: list[dict[str, Any]] = field(default_factory=list)
    priority_sources: list[str] = field(default_factory=list)
    ambiguity_note: str = ""
    decomposition_fallback_used: bool = False
    planner_type: str = "llm"
    answer_shape: str = "short_paragraph"
    query_mode: str = "fast_lookup"


@dataclass
class ExtractedPage:
    text: str
    page_number: int | None = None


@dataclass
class ExtractedContent:
    url: str
    title: str
    domain: str
    source_type: str
    document_type: str
    source_quality: float
    retrieved_at: str
    query: str
    pages: list[ExtractedPage]


@dataclass
class EvidenceChunk:
    text: str
    url: str
    title: str
    domain: str
    source_type: str
    document_type: str
    retrieved_at: str
    query: str
    score: float
    section: str
    page_number: int | None = None
    scoring: dict[str, float] = field(default_factory=dict)
    field: str = ""
    support_level: str = "weak"
    selection_reason: str = ""
    evidence_scope: str = "unrelated"
    row_or_section: str = ""


@dataclass
class ResearchResult:
    query: str
    answer: str
    evidence_chunks: list[EvidenceChunk]
    query_plan: QueryPlan
    debug_info: dict[str, Any]


def canonicalize_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return str(url or "").strip()
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            urlencode(query, doseq=True),
            "",
        )
    )


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _debug_artifact_dir() -> Path:
    return Path(os.getenv("UNIGRAPH_DEBUG_DIR", DEBUG_ARTIFACT_DIR)).expanduser()


def save_debug_artifact(debug_info: dict[str, Any]) -> str:
    request_id = _compact(debug_info.get("request_id")) or f"unigraph-{uuid.uuid4().hex[:12]}"
    safe_request_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", request_id).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    debug_dir = _debug_artifact_dir()
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{timestamp}-{safe_request_id}.json"
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(debug_info, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    logger.info("UniGraph debug artifact written | path=%s", path)
    return str(path)


SEARCH_HINT_PATTERNS = (
    r"(?:IELTS TOEFL language requirement\s*)+",
    r"(?:official IELTS TOEFL CEFR minimum score\s*)+",
    r"(?:application deadline\s*)+",
    r"(?:application portal\s*)+",
)


def _clean_original_question(query: str) -> str:
    cleaned = _compact(query)
    for pattern in SEARCH_HINT_PATTERNS:
        cleaned = re.sub(rf"\s+{pattern}$", "", cleaned, flags=re.I).strip()
    return cleaned


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _is_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.lower().endswith(".pdf")


def _safe_json_loads(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _strict_json_loads(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
    if not match:
        raise ValueError("Planner response did not contain a JSON object.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Planner response JSON was not an object.")
    return payload


def _is_germany_plan(plan: QueryPlan) -> bool:
    haystack = " ".join([plan.country, plan.university, plan.program]).lower()
    return "germany" in haystack or "german" in haystack or ".de" in " ".join(plan.priority_sources)


def _query_mentions_language_requirement(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(term in lowered for term in LANGUAGE_TERMS)


def _normalize_field_name(value: Any) -> str:
    normalized = _compact(value).lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    if any(term in normalized for term in ("deadline", "bewerbungsfrist", "bewerbungszeitraum")):
        if "summer" in normalized:
            return "other_semester_deadline"
        return "application_deadline"
    if normalized in {"applicable_semester", "requested_semester", "winter_semester"}:
        return "intake_or_semester"
    if any(term in normalized for term in ("portal", "platform", "tumonline", "campo")):
        return "application_portal"
    if any(term in normalized for term in ("language", "english", "german", "ielts", "toefl", "cefr")):
        if "ielts" in normalized:
            return "ielts_score"
        if "toefl" in normalized:
            return "toefl_score"
        if "german" in normalized or "deutsch" in normalized:
            return "german_language_requirement"
        return "english_language_requirement"
    if any(term in normalized for term in ("tuition", "fee", "semesterbeitrag", "studiengeb")):
        return "tuition_fee"
    if any(term in normalized for term in ("document", "transcript", "certificate", "unterlagen")):
        return "required_application_documents"
    if any(term in normalized for term in ("module", "curriculum", "study_plan", "ects")):
        return "curriculum_modules"
    return normalized


def _normalize_answer_shape(value: Any, intent: str) -> str:
    normalized = _compact(value).lower().replace("-", "_").replace(" ", "_")
    if normalized in {"short_paragraph", "checklist", "table", "comparison", "overview"}:
        return normalized
    if intent == "document_requirement_lookup":
        return "checklist"
    if intent == "comparison_lookup":
        return "comparison"
    if intent in {"program_overview_lookup", "general_program_overview"}:
        return "overview"
    return "short_paragraph"


def _field_terms(field_name: str, plan: QueryPlan | None = None) -> list[str]:
    normalized = field_name.replace("_", " ").lower()
    terms = [normalized, field_name.lower(), *FIELD_KEYWORDS.get(field_name, [])]
    if field_name in {"application_deadline", "other_semester_deadline", "intake_or_semester"}:
        terms += INTENT_EXTRACT_TERMS["deadline_lookup"]
    elif field_name in {
        "english_language_requirement",
        "german_language_requirement",
        "ielts_score",
        "toefl_score",
        "duolingo_score",
        "language_proof",
        "teaching_language",
    }:
        terms += INTENT_EXTRACT_TERMS["language_requirement_lookup"]
    elif field_name in {"tuition_fee", "semester_contribution"}:
        terms += INTENT_EXTRACT_TERMS["tuition_fee_lookup"]
    elif field_name in {
        "required_application_documents",
        "international_applicant_documents",
        "degree_transcript_requirements",
        "aps_requirement",
        "vpd_requirement",
        "uni_assist_requirement",
    }:
        terms += INTENT_EXTRACT_TERMS["document_requirement_lookup"]
    elif field_name in {"application_process", "application_portal"}:
        terms += INTENT_EXTRACT_TERMS["application_portal_lookup"]
    elif field_name == "curriculum_modules":
        terms += INTENT_EXTRACT_TERMS["curriculum_lookup"]
    if plan is not None:
        terms += plan.keywords + plan.german_keywords
    return list(dict.fromkeys([term.lower() for term in terms if _compact(term)]))


def _detect_university_from_text(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    matches: list[tuple[int, dict[str, Any]]] = []
    for record in KNOWN_UNIVERSITIES.values():
        aliases = [str(record["name"]).lower(), str(record["short"]).lower(), *record["aliases"]]
        for alias in aliases:
            if alias and re.search(rf"\b{re.escape(alias.lower())}\b", lowered):
                matches.append((len(alias), record))
                break
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _detect_program_from_question(query: str) -> str:
    patterns = [
        r"\b(?:msc|m\.sc\.|master(?:'s)?(?:\s+of\s+science)?)\s+([^?.,;]+?)\s+(?:at|in|from)\b",
        r"\b(?:msc|m\.sc\.|master(?:'s)?(?:\s+of\s+science)?)\s+([^?.,;]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.I)
        if match:
            program = _compact(match.group(1))
            program = re.sub(r"\b(university|tu|technical university|at)$", "", program, flags=re.I)
            return _compact(program)
    if re.search(r"\bmaster[-\s]+informatik\b|\binformatik[-\s]+master\b", query, re.I):
        return "Informatics"
    return ""


def _detect_degree_level(query: str) -> str:
    lowered = str(query or "").lower()
    if re.search(r"\b(msc|m\.sc\.|master(?:'s)?|master of science)\b", lowered):
        return "master"
    if re.search(r"\b(bsc|b\.sc\.|bachelor(?:'s)?|bachelor of science)\b", lowered):
        return "bachelor"
    if re.search(r"\b(phd|doctorate|doctoral)\b", lowered):
        return "doctoral"
    return ""


def _infer_query_mode(query: str, intent: str, required_fields: list[str]) -> str:
    lowered = str(query or "").lower()
    if intent == "multi_program_discovery" or re.search(
        r"\b(suggest|recommend|shortlist|compare|which programs?|which universities?)\b",
        lowered,
    ):
        return "discovery_lookup"
    if intent in {
        "eligibility_check",
        "document_requirement_lookup",
        "admission_requirement_lookup",
        "program_overview_lookup",
    }:
        return "research_lookup"
    if len(required_fields or []) >= 4:
        return "research_lookup"
    return "fast_lookup"


def _mode_limit(plan: QueryPlan, key: str, default: int) -> int:
    return QUERY_MODE_LIMITS.get(plan.query_mode, QUERY_MODE_LIMITS["fast_lookup"]).get(
        key, default
    )


def _current_question_context(query: str) -> dict[str, Any]:
    university = _detect_university_from_text(query)
    return {
        "university": university,
        "program": _detect_program_from_question(query),
        "degree_level": _detect_degree_level(query),
        "question_tokens": set(re.findall(r"[a-z0-9äöüß]{3,}", query.lower())),
    }


def _apply_current_question_context(plan: QueryPlan, query: str) -> bool:
    context = _current_question_context(query)
    changed = False
    university = context["university"]
    if university:
        if plan.university and plan.university.lower() != str(university["name"]).lower():
            changed = True
        plan.university = str(university["name"])
        plan.university_short = str(university["short"])
        plan.priority_sources = list(
            dict.fromkeys(
                [
                    *university.get("domains", []),
                    *plan.priority_sources,
                    *OFFICIAL_SECONDARY_DOMAINS,
                ]
            )
        )
    program = str(context["program"])
    if program:
        if plan.program and program.lower() not in plan.program.lower():
            changed = True
        plan.program = program
    degree_level = str(context.get("degree_level") or "")
    if degree_level and not plan.degree_level:
        plan.degree_level = degree_level
    return changed


def resolve_official_domains(plan: QueryPlan, query: str = "") -> list[str]:
    university = _detect_university_from_text(
        " ".join([query, plan.university, plan.university_short])
    )
    if university:
        return list(dict.fromkeys(str(domain).lower() for domain in university.get("domains", [])))
    official = [
        domain
        for domain in plan.priority_sources
        if domain and domain not in OFFICIAL_SECONDARY_DOMAINS
    ]
    return list(dict.fromkeys(official))


def _split_official_domain_tiers(domains: list[str]) -> tuple[list[str], list[str]]:
    roots = [domain for domain in domains if len(str(domain).split(".")) <= 2]
    subdomains = [domain for domain in domains if len(str(domain).split(".")) > 2]
    return list(dict.fromkeys(roots)), list(dict.fromkeys(subdomains))


def _official_pdf_domains(domains: list[str]) -> list[str]:
    roots, subdomains = _split_official_domain_tiers(domains)
    preferred = [
        domain
        for domain in [*subdomains, *roots]
        if any(term in domain for term in ("study", "studium", "tf.", "fau", "cit.", "tum"))
    ]
    return list(dict.fromkeys([*preferred, *subdomains, *roots]))


def _domain_allowed(domain: str, allowed_domains: list[str] | tuple[str, ...]) -> bool:
    normalized_domain = str(domain or "").lower().removeprefix("www.")
    normalized_allowed = {
        str(item or "").lower().removeprefix("www.") for item in allowed_domains if item
    }
    return any(
        normalized_domain == allowed or normalized_domain.endswith("." + allowed)
        for allowed in normalized_allowed
    )


def _domain_exactly_configured(domain: str, allowed_domains: list[str] | tuple[str, ...]) -> bool:
    normalized_domain = str(domain or "").lower().removeprefix("www.")
    normalized_allowed = {
        str(item or "").lower().removeprefix("www.") for item in allowed_domains if item
    }
    return normalized_domain in normalized_allowed


def _unknown_program_subdomain_reason(url: str, title: str, snippet: str, plan: QueryPlan) -> str:
    domain = _domain(url)
    configured_domains = resolve_official_domains(plan)
    if _domain_exactly_configured(domain, configured_domains):
        return ""
    if not any(marker in domain for marker in (".studium.", ".study.")):
        return ""
    first_label = domain.split(".", 1)[0].lower()
    generic_labels = {"www", "studium", "study", "informatik", "tf", "cit", "campus"}
    if first_label in generic_labels:
        return ""
    program_tokens = _target_program_tokens(plan)
    if any(token and (token == first_label or token in first_label) for token in program_tokens):
        return ""
    combined = f"{url} {title} {snippet}".lower()
    if program_tokens and program_tokens.intersection(
        set(re.findall(r"[a-z0-9äöüß]{2,}", combined))
    ):
        return ""
    return "unknown_program_specific_subdomain"


def _weak_unconfigured_subdomain_reason(
    url: str, title: str, snippet: str, plan: QueryPlan
) -> str:
    domain = _domain(url)
    configured_domains = resolve_official_domains(plan)
    if _domain_exactly_configured(domain, configured_domains):
        return ""
    if _domain_allowed(domain, OFFICIAL_SECONDARY_DOMAINS):
        return ""
    if not _domain_allowed(domain, configured_domains):
        return ""
    combined = f"{url} {title} {snippet}".lower()
    if any(
        term in combined
        for term in ("general", "university-wide", "all applicants", "international applicants")
    ):
        return ""
    if _program_match_score(combined, plan) >= 0.60:
        return ""
    return "weak_program_match_on_unconfigured_official_subdomain"


def _infer_intent(query: str) -> str:
    lowered = query.lower()
    if re.search(r"\b(suggest|recommend|find|shortlist|which)\b", lowered) and re.search(
        r"\b(programs?|masters?|master's|msc)\b", lowered
    ):
        return "multi_program_discovery"
    if re.search(r"\b(can i apply|eligible|eligibility|am i eligible|profile|gpa)\b", lowered):
        return "eligibility_check"
    if re.search(r"\b(deadline|application period|intake|bewerbungsfrist)\b", lowered):
        return "deadline_lookup"
    if re.search(r"\b(documents?|checklist|unterlagen|transcript|certificate)\b", lowered):
        return "document_requirement_lookup"
    if re.search(r"\b(tuition|fees?|semester contribution|semesterbeitrag|studiengeb)", lowered):
        return "tuition_fee_lookup"
    if re.search(r"\b(curriculum|modules?|module handbook|modulhandbuch|courses?)\b", lowered):
        return "curriculum_lookup"
    if re.search(r"\b(scholarship|funding|financial aid)\b", lowered):
        return "scholarship_funding_lookup"
    if re.search(
        r"\b(portal|uni-assist|tumonline|campo|application platform|where (?:do|to) i apply|where to apply)\b",
        lowered,
    ):
        return "application_portal_lookup"
    if re.search(r"\b(how (?:do|to) apply|application process|vpd|aps)\b", lowered):
        return "application_process_lookup"
    if _query_mentions_language_requirement(query) or re.search(
        r"\b(english-taught|german-taught|teaching language|language of instruction)\b", lowered
    ):
        return "language_requirement_lookup"
    if re.search(r"\b(admission requirements?|requirements?|zulassungsvoraussetzungen)\b", lowered):
        return "admission_requirement_lookup"
    if re.search(r"\b(duration|overview|how long|semesters?)\b", lowered):
        return "program_overview_lookup"
    return "general_university_question"


def _intent_profile(query: str) -> dict[str, Any]:
    intent = _infer_intent(query)
    profile = INTENT_PROFILES.get(intent, INTENT_PROFILES["general_university_question"])
    required = list(profile["required"])
    optional = list(profile["optional"])
    lowered = query.lower()
    if "ielts" in lowered and "ielts_score" not in required:
        required.append("ielts_score")
    if "toefl" in lowered and "toefl_score" not in optional:
        optional.append("toefl_score")
    if "duolingo" in lowered and "duolingo_score" not in optional:
        optional.append("duolingo_score")
    if "aps" in lowered and "aps_requirement" not in required + optional:
        required.append("aps_requirement")
    if "uni-assist" in lowered and "uni_assist_requirement" not in required + optional:
        required.append("uni_assist_requirement")
    return {
        "intent": intent,
        "required_fields": list(dict.fromkeys(required)),
        "optional_fields": list(dict.fromkeys(optional)),
        "excluded_fields": list(dict.fromkeys(profile["excluded"])),
    }


def _explicit_non_german_fau(query: str) -> bool:
    lowered = str(query or "").lower()
    return bool(re.search(r"\b(florida|atlantic|usa|united states|america|boca raton)\b", lowered))


def _prefers_german_fau(query: str, plan: QueryPlan | None = None) -> bool:
    haystack = f"{query} {getattr(plan, 'university', '')} {getattr(plan, 'country', '')}".lower()
    return bool(re.search(r"\bfau\b", haystack)) and not _explicit_non_german_fau(query)


def _requested_sections_from_query(query: str, fallback: list[str]) -> list[str]:
    profile = _intent_profile(query)
    return profile["required_fields"] or fallback or ["general_information"]


def _with_german_fau_focus(plan: QueryPlan, query: str) -> QueryPlan:
    if not (GERMAN_UNIVERSITY_FOCUS and _prefers_german_fau(query, plan)):
        return plan
    fau = AMBIGUOUS_GERMAN_UNIVERSITIES["fau"]
    plan.university = plan.university or str(fau["preferred_name"])
    plan.university_short = plan.university_short or "FAU"
    plan.country = plan.country or "Germany"
    plan.priority_sources = list(
        dict.fromkeys([str(fau["preferred_domain"]), *plan.priority_sources, "daad.de"])
    )
    if _query_mentions_language_requirement(query):
        profile = _intent_profile(query)
        plan.required_info = profile["required_fields"]
        plan.required_fields = profile["required_fields"]
        plan.optional_fields = profile["optional_fields"]
        plan.excluded_fields = profile["excluded_fields"]
        plan.intent = profile["intent"]
        plan.keywords = list(dict.fromkeys([*plan.keywords, "IELTS", "English proficiency"]))
    plan.ambiguity_note = (
        "FAU is ambiguous; German university focus prefers Friedrich-Alexander-Universität "
        "Erlangen-Nürnberg (fau.de) over Florida Atlantic University unless Florida/USA is explicit."
    )
    query_limit = _mode_limit(plan, "max_queries", MAX_QUERIES)
    plan.search_queries = [item for item in plan.search_queries if isinstance(item, dict)][
        :query_limit
    ]
    return plan


def _build_search_queries(plan: QueryPlan, query: str) -> list[dict[str, Any]]:
    target = " ".join(
        item
        for item in [plan.university_short or plan.university, plan.university, plan.program]
        if item
    )
    if not target:
        target = query
    field_phrase = " ".join(
        field.replace("_", " ")
        for field in (plan.required_fields or plan.required_info or ["admission requirements"])[:3]
    )
    queries = [
        {
            "query": f"{target} {field_phrase} official",
            "type": "official_page",
            "priority": 1.0,
        },
        {
            "query": f"{target} {field_phrase} admission application requirements",
            "type": plan.intent,
            "priority": 0.92,
        },
        {
            "query": (
                f"{target} {field_phrase} site:{plan.priority_sources[0]}"
                if plan.priority_sources
                else f"{target} {field_phrase} university"
            ),
            "type": "official_page",
            "priority": 0.9,
        },
        {"query": f"{target} {field_phrase} filetype:pdf", "type": "pdf", "priority": 0.82},
        {"query": f"{target} {field_phrase} DAAD", "type": "daad", "priority": 0.7},
    ]
    if plan.intent == "multi_program_discovery":
        queries = [
            {
                "query": f"Germany AI master's IELTS 6.5 English language requirements official university",
                "type": "official_page",
                "priority": 1.0,
            },
            {
                "query": "site:daad.de Germany artificial intelligence master IELTS English requirements",
                "type": "daad",
                "priority": 0.85,
            },
            {
                "query": "German university MSc Artificial Intelligence IELTS 6.5 English B2",
                "type": "official_page",
                "priority": 0.8,
            },
        ]
    return queries[: _mode_limit(plan, "max_queries", MAX_QUERIES)]


def _field_query_keywords(plan: QueryPlan) -> list[str]:
    fields = plan.required_fields or plan.required_info or ["admission requirements"]
    terms: list[str] = []
    if plan.intent == "deadline_lookup":
        terms.extend(["application period", "winter semester", "Bewerbungsfrist"])
    elif plan.intent == "language_requirement_lookup":
        terms.extend(
            [
                "IELTS",
                "English language requirements",
                "Sprachnachweis",
                "TOEFL",
                "B2",
                "Zulassungsvoraussetzungen",
            ]
        )
    elif plan.intent == "tuition_fee_lookup":
        terms.extend(["tuition fees", "non-EU", "semester fee", "Semesterbeitrag"])
    elif plan.intent == "application_portal_lookup":
        terms.extend(["application portal", "apply online", "uni-assist", "VPD"])
    for field_name in fields[:3]:
        field_terms = FIELD_KEYWORDS.get(field_name, [field_name.replace("_", " ")])
        terms.extend(field_terms[:3])
    return list(dict.fromkeys(_compact(term) for term in terms if _compact(term)))[:8]


def _broadened_field_keywords(plan: QueryPlan, *, pdf: bool = False) -> list[str]:
    if plan.intent == "language_requirement_lookup":
        terms = [
            "IELTS",
            "English language requirement",
            "CEFR B2",
            "language proficiency",
            "Sprachnachweis",
            "Englischkenntnisse",
            "Application FAQ",
            "English certificate",
            "supporting documents",
            "Beizufuegende Unterlagen",
            "Beizufügende Unterlagen",
        ]
    elif plan.intent == "deadline_lookup":
        terms = [
            "application period",
            "application deadline",
            "Bewerbungsfrist",
            "Bewerbungszeitraum",
        ]
    elif plan.intent == "tuition_fee_lookup":
        terms = ["tuition fee", "semester contribution", "Studiengebühren", "Semesterbeitrag"]
    else:
        terms = [*_field_query_keywords(plan), "FAQ", "admission", "application"]
    if pdf:
        terms.extend(["supporting documents", "Beizufuegende Unterlagen", "PDF"])
    return list(dict.fromkeys(_compact(term) for term in terms if _compact(term)))


def _official_site_queries(
    plan: QueryPlan,
    query: str,
    domains: list[str],
    *,
    retry: bool = False,
    pdf: bool = False,
    query_limit: int | None = None,
) -> list[dict[str, Any]]:
    query_limit = query_limit or _mode_limit(plan, "max_queries", MAX_QUERIES)
    degree_terms = []
    if plan.degree_level == "master" or re.search(
        r"\b(msc|m\.sc|master)\b", f"{query} {plan.program}", re.I
    ):
        degree_terms = ["MSc", "Master"]
    elif plan.degree_level == "bachelor":
        degree_terms = ["BSc", "Bachelor"]
    target_names = list(
        dict.fromkeys(
            [
                *[
                    f"{degree} {plan.program}"
                    for degree in degree_terms
                    if plan.program and not plan.program.lower().startswith(degree.lower())
                ],
                plan.program,
            ]
        )
    )
    if not any(target_names):
        target_names = [query]
    keywords = (
        _broadened_field_keywords(plan, pdf=pdf) if retry or pdf else _field_query_keywords(plan)
    )
    if retry and not pdf:
        keywords = list(
            dict.fromkeys(
                [
                    *keywords,
                    "Bewerbungszeitraum",
                    "Zulassungsvoraussetzungen",
                    "Studiengebühren",
                    "Bewerbungsunterlagen",
                ]
            )
        )
    queries: list[dict[str, Any]] = []
    for domain in domains:
        for target in target_names[:3]:
            for keyword in keywords[:3]:
                queries.append(
                    {
                        "query": (
                            f'site:{domain} "{target}" "{keyword}" filetype:pdf'
                            if pdf
                            else f'site:{domain} "{target}" "{keyword}"'
                        ),
                        "type": (
                            "tier1_pdf" if pdf else ("tier1_retry" if retry else "tier1_official")
                        ),
                        "priority": 1.0 if not retry else 0.95,
                        "include_domains": [domain],
                    }
                )
                if len(queries) >= query_limit:
                    return queries
    return queries[:query_limit]


def _secondary_source_queries(plan: QueryPlan, query: str) -> list[dict[str, Any]]:
    targets = list(
        dict.fromkeys(
            [
                plan.program,
                plan.university_short,
                plan.university,
                " ".join(part for part in [plan.university_short, plan.program] if part),
                " ".join(part for part in [plan.university, plan.program] if part),
            ]
        )
    )
    domains = ["www2.daad.de", "daad.de"]
    if plan.intent in {"application_portal_lookup", "document_requirement_lookup"}:
        domains.append("uni-assist.de")
    queries: list[dict[str, Any]] = []
    for domain in domains:
        for target in [item for item in targets if item][:3]:
            queries.append(
                {
                    "query": f'site:{domain} "{target}"',
                    "type": "tier2_secondary",
                    "priority": 0.82,
                    "include_domains": [domain],
                }
            )
            if len(queries) >= _mode_limit(plan, "max_queries", MAX_QUERIES):
                return queries
    return queries


def _broad_fallback_queries(plan: QueryPlan, query: str) -> list[dict[str, Any]]:
    target = " ".join(part for part in [plan.university, plan.program] if part) or query
    keywords = " ".join(_field_query_keywords(plan)[:4])
    return [
        {"query": f'"{target}" {keywords}', "type": "tier3_broad", "priority": 0.55},
        {"query": f"{target} official {keywords}", "type": "tier3_broad", "priority": 0.50},
    ][: _mode_limit(plan, "max_queries", MAX_QUERIES)]


def _query_validation_status(plan: QueryPlan, query: str) -> dict[str, Any]:
    current_university = _detect_university_from_text(
        " ".join([query, plan.university, plan.university_short])
    )
    rejected: list[dict[str, str]] = []
    accepted: list[dict[str, Any]] = []
    for item in plan.search_queries:
        query_text = _compact(item.get("query")) if isinstance(item, dict) else ""
        if not query_text:
            continue
        mentioned_university = _detect_university_from_text(query_text)
        if (
            current_university
            and mentioned_university
            and str(mentioned_university["name"]).lower() != str(current_university["name"]).lower()
        ):
            rejected.append(
                {
                    "query": query_text,
                    "reason": "query_mentions_different_university",
                    "detected_university": str(mentioned_university["name"]),
                }
            )
            continue
        if plan.program:
            program_tokens = {
                token
                for token in re.findall(r"[a-z0-9äöüß]{3,}", plan.program.lower())
                if token not in {"msc", "master", "science"}
            }
            if program_tokens and not program_tokens.intersection(
                set(re.findall(r"[a-z0-9äöüß]{3,}", query_text.lower()))
            ):
                rejected.append(
                    {
                        "query": query_text,
                        "reason": "query_missing_current_program_terms",
                    }
                )
                continue
        accepted.append(item)
    return {
        "valid": bool(accepted) and not rejected,
        "accepted_queries": accepted,
        "rejected_queries": rejected,
    }


def validate_and_repair_search_queries(plan: QueryPlan, query: str) -> dict[str, Any]:
    context_changed = _apply_current_question_context(plan, query)
    initial = _query_validation_status(plan, query)
    if initial["rejected_queries"] or not initial["accepted_queries"]:
        repaired_queries = _build_search_queries(plan, query)
        plan.search_queries = repaired_queries
        repaired = _query_validation_status(plan, query)
        if not repaired["accepted_queries"]:
            plan.search_queries = _fallback_plan(query).search_queries
            repaired = _query_validation_status(plan, query)
        status = {
            "valid": bool(repaired["accepted_queries"]) and not repaired["rejected_queries"],
            "context_changed": context_changed,
            "regenerated": True,
            "initial_rejected_queries": initial["rejected_queries"],
            "accepted_queries": repaired["accepted_queries"],
            "rejected_queries": repaired["rejected_queries"],
        }
    else:
        plan.search_queries = initial["accepted_queries"][
            : _mode_limit(plan, "max_queries", MAX_QUERIES)
        ]
        status = {
            "valid": True,
            "context_changed": context_changed,
            "regenerated": False,
            "initial_rejected_queries": [],
            "accepted_queries": initial["accepted_queries"],
            "rejected_queries": [],
        }
    return status


def _fallback_plan(query: str) -> QueryPlan:
    keywords = [token for token in re.findall(r"[A-Za-zÄÖÜäöüß0-9][\wÄÖÜäöüß-]{2,}", query)[:12]]
    profile = _intent_profile(query)
    required_info = profile["required_fields"]
    query_mode = _infer_query_mode(query, profile["intent"], required_info)
    field_phrase = " ".join(field.replace("_", " ") for field in required_info[:3])
    queries = [
        {
            "query": f"{query} {field_phrase} official university",
            "type": "official_page",
            "priority": 1.0,
        },
        {
            "query": f"{query} official admissions {field_phrase}",
            "type": profile["intent"],
            "priority": 0.9,
        },
        {"query": f"{query} {field_phrase} filetype:pdf", "type": "pdf", "priority": 0.85},
        {"query": f"{query} DAAD", "type": "daad", "priority": 0.7},
    ]
    plan = QueryPlan(
        country="Germany" if re.search(r"\b(germany|german|deutschland)\b", query, re.I) else "",
        user_intent=query,
        intent=profile["intent"],
        required_info=required_info,
        required_fields=required_info,
        optional_fields=profile["optional_fields"],
        excluded_fields=profile["excluded_fields"],
        keywords=keywords,
        german_keywords=(
            GERMAN_SEARCH_TERMS[:4]
            if re.search(r"\b(germany|german|deutschland)\b", query, re.I)
            else []
        ),
        search_queries=queries[: QUERY_MODE_LIMITS[query_mode]["max_queries"]],
        priority_sources=["daad.de"],
        degree_level=_detect_degree_level(query),
        query_mode=query_mode,
        planner_type="heuristic",
        answer_shape=_normalize_answer_shape("", profile["intent"]),
    )
    _apply_current_question_context(plan, query)
    return _with_german_fau_focus(plan, query)


def _normalize_plan(payload: dict[str, Any], query: str) -> QueryPlan:
    fallback = _fallback_plan(query)
    profile = _intent_profile(query)
    raw_queries = payload.get("search_queries")
    queries: list[dict[str, Any]] = []
    if isinstance(raw_queries, list):
        for item in raw_queries:
            if not isinstance(item, dict):
                continue
            text = _compact(item.get("query"))
            if not text:
                continue
            queries.append(
                {
                    "query": text,
                    "type": _compact(item.get("type")) or "official_page",
                    "priority": float(item.get("priority") or 0.8),
                }
            )
            if len(queries) >= MAX_QUERIES:
                break
    if not queries:
        queries = fallback.search_queries

    required_fields = [
        _normalize_field_name(item)
        for item in payload.get(
            "required_fields", payload.get("required_info", fallback.required_fields)
        )
        if _compact(item)
    ] or profile["required_fields"]
    required_fields = list(dict.fromkeys(required_fields))
    optional_fields = [
        _normalize_field_name(item)
        for item in payload.get("optional_fields", fallback.optional_fields)
        if _compact(item)
    ] or profile["optional_fields"]
    optional_fields = list(dict.fromkeys(optional_fields))
    excluded_fields = [
        _normalize_field_name(item)
        for item in payload.get("excluded_fields", fallback.excluded_fields)
        if _compact(item)
    ] or profile["excluded_fields"]
    excluded_fields = list(dict.fromkeys(excluded_fields))
    deterministic_intent = profile["intent"]
    llm_intent = _compact(
        payload.get("detected_intent") or payload.get("intent") or payload.get("user_intent")
    ).lower()
    intent = (
        deterministic_intent
        if deterministic_intent != "general_university_question"
        else llm_intent
    )
    if intent not in INTENT_PROFILES:
        intent = deterministic_intent

    plan = QueryPlan(
        university=_compact(payload.get("university")) or fallback.university,
        university_short=_compact(payload.get("university_short")),
        program=_compact(payload.get("program")) or fallback.program,
        country=_compact(payload.get("country")) or fallback.country,
        degree_level=_compact(payload.get("degree_level")) or fallback.degree_level,
        user_intent=_compact(payload.get("user_intent")) or query,
        intent=intent,
        required_info=required_fields,
        required_fields=required_fields,
        optional_fields=optional_fields,
        excluded_fields=excluded_fields,
        user_profile_details=(
            payload.get("user_profile_details", {})
            if isinstance(payload.get("user_profile_details"), dict)
            else {}
        ),
        keywords=[
            _compact(item) for item in payload.get("keywords", fallback.keywords) if _compact(item)
        ],
        german_keywords=[
            _compact(item)
            for item in payload.get("german_keywords", fallback.german_keywords)
            if _compact(item)
        ],
        search_queries=queries,
        priority_sources=[
            _compact(item).lower().removeprefix("www.")
            for item in payload.get("priority_sources", fallback.priority_sources)
            if _compact(item)
        ],
        query_mode=_infer_query_mode(query, intent, required_fields),
        planner_type="llm",
        answer_shape=_normalize_answer_shape(payload.get("answer_shape"), intent),
    )
    _apply_current_question_context(plan, query)
    if _is_germany_plan(plan):
        plan.german_keywords = list(dict.fromkeys([*plan.german_keywords, *GERMAN_SEARCH_TERMS]))[
            :12
        ]
    return _with_german_fau_focus(plan, query)


async def analyze_query(query: str) -> QueryPlan:
    from app.infra.bedrock_chat_client import client as bedrock_client

    system_prompt = """
You plan fast official-source university research. Return only strict JSON.
The JSON must include this answer-planning contract:
{
  "detected_intent": "...",
  "university": "...",
  "program": "...",
  "degree_level": "...",
  "required_fields": [],
  "optional_fields": [],
  "excluded_fields": [],
  "answer_shape": "short_paragraph | checklist | table | comparison | overview"
}

You may also include university_short, country, user_intent, intent,
user_profile_details, keywords, german_keywords, search_queries, and
priority_sources for retrieval.

Classify intent as one of: language_requirement_lookup, deadline_lookup,
eligibility_check, document_requirement_lookup, tuition_fee_lookup,
admission_requirement_lookup, program_overview_lookup, curriculum_lookup,
application_process_lookup, application_portal_lookup, multi_program_discovery,
general_university_question. Put the same value in detected_intent and intent.

Required fields are facts that must be answered. Optional fields are useful only
when directly relevant. Excluded fields must not appear in the final answer.
For a narrow IELTS question, exclude tuition, GPA, documents, deadlines, GRE, and
curriculum. For a deadline question, exclude IELTS, GPA, tuition, and documents
unless the deadline explicitly depends on them.

Create at most 5 targeted search queries. Include official program/admission pages,
deadlines, language requirements, documents, tuition/fees if relevant, PDFs, DAAD,
and German terms for German universities. Prefer official university domains, DAAD,
.de/.eu, official PDFs, admissions, faculty/program, international office, and
uni-assist where relevant. Avoid blogs, forums, consultants, and unsourced portals.
"""
    user_prompt = f"Question: {query}"
    logger.info("UniGraph query decomposition started | question=%s", query)
    try:
        response = await bedrock_client.chat.completions.create(
            model=settings.bedrock.primary_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        plan = _normalize_plan(_strict_json_loads(content), query)
        plan.decomposition_fallback_used = False
        plan.planner_type = "llm"
        plan.user_profile_details.pop("fallback_error", None)
    except Exception as exc:
        logger.exception("UniGraph LLM planner failed | error=%s", exc)
        plan = _fallback_plan(query)
        plan.decomposition_fallback_used = True
        plan.planner_type = "llm_failed"
        plan.user_profile_details["fallback_error"] = str(exc)
    logger.info(
        "UniGraph query decomposition complete | university=%s | program=%s | sections=%s",
        plan.university,
        plan.program,
        plan.required_fields,
    )
    return plan


async def execute_search_queries(plan: QueryPlan) -> tuple[list[dict[str, Any]], int]:
    logger.info("UniGraph generated search queries | queries=%s", plan.search_queries)

    async def _search(query_obj: dict[str, Any]) -> dict[str, Any]:
        query_text = str(query_obj.get("query", "")).strip()
        include_domains = query_obj.get("include_domains")
        if not isinstance(include_domains, list):
            site_domains = re.findall(r"\bsite:([A-Za-z0-9_.-]+\.[A-Za-z]{2,})", query_text)
            include_domains = [domain.lower().removeprefix("www.") for domain in site_domains]
        include_domains = [str(domain).strip() for domain in include_domains if str(domain).strip()]
        try:
            payload = await asearch_google(
                query=query_text,
                num=MAX_RESULTS_PER_QUERY,
                search_depth="basic",
                include_raw_content=False,
                include_answer=False,
                include_domains=include_domains or None,
                exclude_domains=list(LOW_QUALITY_DOMAINS),
            )
            rows = payload.get("organic_results", [])
            logger.info("UniGraph fan-out results | query=%s | count=%s", query_text, len(rows))
            return {
                "query": query_text,
                "type": query_obj.get("type", "official_page"),
                "priority": float(query_obj.get("priority") or 0.8),
                "include_domains": include_domains,
                "results": rows if isinstance(rows, list) else [],
            }
        except Exception as exc:
            logger.warning("UniGraph search failed | query=%s | error=%s", query_text, exc)
            return {
                "query": query_text,
                "type": query_obj.get("type", "official_page"),
                "priority": float(query_obj.get("priority") or 0.8),
                "results": [],
                "error": str(exc),
            }

    queries = plan.search_queries[: _mode_limit(plan, "max_queries", MAX_QUERIES)]
    return list(await asyncio.gather(*[_search(item) for item in queries])), len(queries)


async def execute_tiered_retrieval(
    plan: QueryPlan,
    query: str,
    *,
    debug_collector: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any]]:
    target_domains = resolve_official_domains(plan, query)
    root_domains, subdomains = _split_official_domain_tiers(target_domains)
    all_results: list[dict[str, Any]] = []
    total_calls = 0
    max_search_calls = 6 if plan.query_mode == "fast_lookup" else 20
    min_official_calls_before_daad = 5 if plan.query_mode == "fast_lookup" else 0
    tier_debug: dict[str, Any] = {
        "resolved_official_domains": target_domains,
        "root_official_domains": root_domains,
        "official_subdomains": subdomains,
        "tiers_attempted": [],
        "fallback_tier_used": "",
        "zero_candidate_recovery": [],
        "weak_candidates_deferred": [],
    }
    deferred_candidates: list[dict[str, Any]] = []

    async def _run_tier(
        tier_name: str,
        queries: list[dict[str, Any]],
        *,
        allow_secondary: bool,
        allow_third_party: bool,
    ) -> list[dict[str, Any]]:
        nonlocal total_calls, all_results
        remaining_calls = max(0, max_search_calls - total_calls)
        if remaining_calls <= 0:
            tier_debug["tiers_attempted"].append(
                {"tier": tier_name, "queries": [], "raw_result_urls": [], "accepted_urls": []}
            )
            return []
        plan.search_queries = queries[
            : min(_mode_limit(plan, "max_queries", MAX_QUERIES), remaining_calls)
        ]
        tier_results, calls = await execute_search_queries(plan)
        total_calls += calls
        all_results.extend(tier_results)
        selected = select_and_deduplicate_urls(
            tier_results,
            plan,
            debug_collector=debug_collector,
            tier=tier_name,
            allowed_domains=target_domains,
            allow_secondary=allow_secondary,
            allow_third_party=allow_third_party,
        )
        tier_debug["tiers_attempted"].append(
            {
                "tier": tier_name,
                "queries": queries,
                "raw_result_urls": [
                    canonicalize_url(str(row.get("link") or row.get("url") or ""))
                    for result in tier_results
                    for row in result.get("results", [])
                    if isinstance(row, dict)
                ],
                "accepted_urls": [item["url"] for item in selected],
                "accepted_candidates": len(selected),
                "top_rejection_reasons": _top_rejection_reasons(debug_collector),
            }
        )
        return selected

    tier1_queries = _official_site_queries(
        plan, query, root_domains or target_domains
    ) or _build_search_queries(plan, query)
    selected_urls = await _run_tier(
        "tier1a_root", tier1_queries, allow_secondary=False, allow_third_party=False
    )
    if selected_urls:
        if _selected_has_strong_retrieval_signal(plan, selected_urls):
            tier_debug["tier_used"] = "tier1a_root"
            return all_results, selected_urls, total_calls, tier_debug
        deferred_candidates.extend(selected_urls)
        tier_debug["weak_candidates_deferred"].extend(
            {"tier": "tier1a_root", "url": item["url"], "reason": "weak_retrieval_signal"}
            for item in selected_urls
        )
        tier_debug["zero_candidate_recovery"].append(
            {
                "after_tier": "tier1a_root",
                "action": "continue_to_subdomains_because_root_candidates_are_weak",
                "top_rejection_reasons": _top_rejection_reasons(debug_collector),
            }
        )

    if subdomains:
        tier_debug["zero_candidate_recovery"].append(
            {
                "after_tier": "tier1a_root",
                "action": "expand_to_known_official_subdomains",
                "top_rejection_reasons": _top_rejection_reasons(debug_collector),
            }
        )
        subdomain_queries = _official_site_queries(
            plan, query, subdomains, retry=True, query_limit=max(1, max_search_calls - total_calls)
        )
        selected_urls = await _run_tier(
            "tier1b_subdomains",
            subdomain_queries,
            allow_secondary=False,
            allow_third_party=False,
        )
        if selected_urls:
            if _selected_has_strong_retrieval_signal(plan, selected_urls):
                tier_debug["tier_used"] = "tier1b_subdomains"
                tier_debug["fallback_tier_used"] = "tier1b_subdomains"
                return all_results, selected_urls, total_calls, tier_debug
            deferred_candidates.extend(selected_urls)
            tier_debug["weak_candidates_deferred"].extend(
                {"tier": "tier1b_subdomains", "url": item["url"], "reason": "weak_retrieval_signal"}
                for item in selected_urls
            )

    retry_queries = _official_site_queries(
        plan,
        query,
        target_domains,
        retry=True,
        query_limit=max(1, max_search_calls - total_calls),
    )
    if retry_queries and total_calls < max_search_calls:
        tier_debug["zero_candidate_recovery"].append(
            {
                "after_tier": "tier1b_subdomains" if subdomains else "tier1a_root",
                "action": "broaden_field_terms_on_official_domains",
                "top_rejection_reasons": _top_rejection_reasons(debug_collector),
            }
        )
        selected_urls = await _run_tier(
            "tier1_retry_terms", retry_queries, allow_secondary=False, allow_third_party=False
        )
        if selected_urls:
            if _selected_has_strong_retrieval_signal(plan, selected_urls):
                tier_debug["tier_used"] = "tier1_retry_terms"
                tier_debug["fallback_tier_used"] = "tier1_retry_terms"
                return all_results, selected_urls, total_calls, tier_debug
            deferred_candidates.extend(selected_urls)
            tier_debug["weak_candidates_deferred"].extend(
                {"tier": "tier1_retry_terms", "url": item["url"], "reason": "weak_retrieval_signal"}
                for item in selected_urls
            )

    pdf_queries = _official_site_queries(
        plan,
        query,
        _official_pdf_domains(target_domains),
        retry=True,
        pdf=True,
        query_limit=max(1, max_search_calls - total_calls),
    )
    if pdf_queries and total_calls < max_search_calls:
        tier_debug["zero_candidate_recovery"].append(
            {
                "after_tier": "tier1_retry_terms",
                "action": "search_official_pdfs_supporting_documents",
                "top_rejection_reasons": _top_rejection_reasons(debug_collector),
            }
        )
        selected_urls = await _run_tier(
            "tier1c_official_pdfs", pdf_queries, allow_secondary=False, allow_third_party=False
        )
        if selected_urls:
            if _selected_has_strong_retrieval_signal(plan, selected_urls):
                tier_debug["tier_used"] = "tier1c_official_pdfs"
                tier_debug["fallback_tier_used"] = "tier1c_official_pdfs"
                return all_results, selected_urls, total_calls, tier_debug
            deferred_candidates.extend(selected_urls)
            tier_debug["weak_candidates_deferred"].extend(
                {"tier": "tier1c_official_pdfs", "url": item["url"], "reason": "weak_retrieval_signal"}
                for item in selected_urls
            )

    tier2_queries = _secondary_source_queries(plan, query)
    if plan.query_mode != "fast_lookup" or total_calls >= min_official_calls_before_daad:
        selected_urls = await _run_tier(
            "tier2", tier2_queries, allow_secondary=True, allow_third_party=False
        )
        if selected_urls:
            tier_debug["tier_used"] = "tier2"
            tier_debug["fallback_tier_used"] = "tier2"
            return all_results, selected_urls, total_calls, tier_debug
    else:
        tier_debug["zero_candidate_recovery"].append(
            {
                "after_tier": "tier1c_official_pdfs",
                "action": "skip_daad_until_official_recovery_budget_used",
                "official_calls_used": total_calls,
            }
        )

    tier3_queries = _broad_fallback_queries(plan, query)
    selected_urls = await _run_tier(
        "tier3", tier3_queries, allow_secondary=True, allow_third_party=True
    )
    if selected_urls:
        tier_debug["tier_used"] = "tier3"
        tier_debug["fallback_tier_used"] = "tier3"
        return all_results, selected_urls, total_calls, tier_debug
    if deferred_candidates:
        max_urls = _mode_limit(plan, "max_urls", MAX_TOTAL_URLS_TO_FETCH)
        tier_debug["tier_used"] = "deferred_weak_official"
        tier_debug["fallback_tier_used"] = "deferred_weak_official"
        return all_results, _diversify_candidates(deferred_candidates, max_urls), total_calls, tier_debug
    tier_debug["tier_used"] = "none"
    tier_debug["fallback_tier_used"] = "none"
    return all_results, [], total_calls, tier_debug


def calculate_source_quality(url: str, *, document_type: str = "html") -> tuple[float, str]:
    domain = _domain(url)
    known_official_university_domains = {
        domain for record in KNOWN_UNIVERSITIES.values() for domain in record.get("domains", [])
    } | {
        "tum.de",
        "lmu.de",
        "rwth-aachen.de",
        "kit.edu",
        "uni-mannheim.de",
    }
    university_like = (
        domain in known_official_university_domains
        or any(domain.endswith("." + item) for item in known_official_university_domains)
        or domain.endswith(".edu")
        or domain.endswith(".de")
        and any(part in domain for part in ("uni-", "tu-", "tum.", "lmu.", "rwth-", "kit.", "fu-"))
        or any(part in domain for part in ("university", "hochschule"))
    )
    if university_like:
        return 0.95, (
            "official_university_pdf" if document_type == "pdf" else "official_university_page"
        )
    if domain.endswith("daad.de") or domain == "daad.de" or domain.endswith("study-in-germany.de"):
        return 0.85, "daad"
    if "uni-assist" in domain:
        return 0.75, "uni_assist"
    if domain.endswith(".eu") or ".gov" in domain or domain.endswith(".bund.de"):
        return 0.75, "government_or_eu"
    if any(item in domain for item in ("education", "study", "studieren", "mastersportal")):
        return 0.40, "third_party_education_site"
    if any(item in domain for item in LOW_QUALITY_DOMAINS) or "forum" in domain or "blog" in domain:
        return 0.20, "blog_or_forum"
    return 0.50, "other"


def _accepted_search_result(url: str, title: str, snippet: str, plan: QueryPlan) -> bool:
    domain = _domain(url)
    combined = f"{url} {title} {snippet}".lower()
    if any(bad in domain for bad in LOW_QUALITY_DOMAINS):
        return False
    relevance = _url_relevance_score(url, title, snippet, plan)
    if relevance <= -0.45:
        return False
    if any(
        source and (domain == source or domain.endswith("." + source))
        for source in plan.priority_sources
    ):
        return True
    quality, source_type = calculate_source_quality(
        url, document_type="pdf" if _is_pdf_url(url) else "html"
    )
    if quality >= 0.75:
        return True
    if (domain.endswith(".de") or domain.endswith(".eu")) and any(
        term in combined for term in OFFICIAL_KEYWORDS
    ):
        return True
    return source_type not in {"blog_or_forum", "third_party_education_site"} and quality >= 0.50


def _url_relevance_score(url: str, title: str, snippet: str, plan: QueryPlan) -> float:
    parsed = urlparse(url)
    combined = f"{parsed.path} {title} {snippet}".lower()
    score = 0.0
    target_terms = [
        plan.university,
        plan.university_short,
        plan.program,
        *[field.replace("_", " ") for field in (plan.required_fields or plan.required_info)[:3]],
    ]
    for term in target_terms:
        normalized = _compact(term).lower()
        if normalized and normalized in combined:
            score += 0.15
    if any(term in combined for term in URL_PATH_BOOST_TERMS):
        score += 0.20
    penalty_hits = [term for term in URL_PATH_PENALTY_TERMS if term in combined]
    score -= min(0.70, len(penalty_hits) * 0.25)
    if plan.program:
        program_tokens = {
            token
            for token in re.findall(r"[a-z0-9äöüß]{3,}", plan.program.lower())
            if token not in {"msc", "master", "science"}
        }
        if program_tokens and not program_tokens.intersection(
            set(re.findall(r"[a-z0-9äöüß]{3,}", combined))
        ):
            score -= 0.15
    return max(-1.0, min(1.0, score))


def _program_match_score(text: str, plan: QueryPlan) -> float:
    return _program_match_details(text, plan)["score"]


def _program_match_details(text: str, plan: QueryPlan) -> dict[str, Any]:
    if not plan.program:
        return {
            "score": 0.0,
            "program_name_match": 0.0,
            "degree_level_match": False,
            "path_degree_signal": "unknown",
            "path_degree_match": True,
            "intent_context_match": False,
        }
    tokens = _target_program_tokens(plan)
    haystack = text.lower()
    path_signal = _degree_signal(urlparse(haystack.split()[0]).path if haystack.split() else "")
    path_degree_match = _degree_signal_matches_target(path_signal, plan)
    degree_level_match = bool(
        re.search(
            r"\b(msc|m\.sc|master|master's|degree programme|degree program|degree-programme|degree-program)\b",
            haystack,
        )
    )
    if plan.degree_level == "master" and path_signal == "master":
        degree_level_match = True
    if plan.degree_level == "master" and path_signal == "bachelor":
        degree_level_match = False
    if plan.degree_level == "bachelor" and path_signal == "bachelor":
        degree_level_match = True
    if plan.degree_level == "bachelor" and path_signal == "master":
        degree_level_match = False
    intent_context_match = any(signal in haystack for signal in STUDENT_PAGE_SIGNALS)
    if plan.program.lower() in haystack and degree_level_match:
        return {
            "score": 1.0,
            "program_name_match": 1.0,
            "degree_level_match": True,
            "path_degree_signal": path_signal,
            "path_degree_match": path_degree_match,
            "intent_context_match": intent_context_match,
        }
    if plan.program.lower() in haystack and re.search(
        r"\b(msc|m\.sc|master|master's|degree programme|degree program|degree-programme|degree-program)\b",
        haystack,
    ):
        return {
            "score": 1.0,
            "program_name_match": 1.0,
            "degree_level_match": True,
            "path_degree_signal": path_signal,
            "path_degree_match": path_degree_match,
            "intent_context_match": intent_context_match,
        }
    if not tokens:
        return {
            "score": 0.0,
            "program_name_match": 0.0,
            "degree_level_match": degree_level_match,
            "path_degree_signal": path_signal,
            "path_degree_match": path_degree_match,
            "intent_context_match": intent_context_match,
        }
    found = sum(1 for token in tokens if token in haystack)
    base = found / len(tokens)
    if base >= 1.0 and not re.search(
        r"\b(msc|m\.sc|master|master's|degree programme|degree program|degree-programme|degree-program)\b",
        haystack,
    ):
        base = 0.45
    score = min(
        1.0,
        (base * 0.60)
        + (0.25 if degree_level_match else 0.0)
        + (0.15 if intent_context_match else 0.0),
    )
    if plan.degree_level == "master" and path_signal == "master":
        score = min(1.0, score + 0.12)
    if plan.degree_level == "master" and path_signal == "bachelor":
        score = max(0.0, score - 0.50)
    if plan.degree_level == "bachelor" and path_signal == "bachelor":
        score = min(1.0, score + 0.12)
    if plan.degree_level == "bachelor" and path_signal == "master":
        score = max(0.0, score - 0.50)
    return {
        "score": score,
        "program_name_match": base,
        "degree_level_match": degree_level_match,
        "path_degree_signal": path_signal,
        "path_degree_match": path_degree_match,
        "intent_context_match": intent_context_match,
    }


def _field_relevance_score(text: str, plan: QueryPlan) -> float:
    keywords: list[str] = []
    for field_name in plan.required_fields or plan.required_info:
        keywords.extend(FIELD_KEYWORDS.get(field_name, [field_name.replace("_", " ")]))
    return _keyword_match(text, keywords)


PAGE_TYPES_BY_INTENT: dict[str, set[str]] = {
    "language_requirement_lookup": {
        "program_page",
        "admissions_page",
        "application_page",
        "language_requirement_page",
        "faq_page",
        "program_faq_page",
        "program_application_page",
        "document_checklist_pdf",
        "official_policy_pdf",
        "exact_daad_listing",
    },
    "deadline_lookup": {
        "program_page",
        "admissions_page",
        "application_page",
        "program_application_page",
        "dates_deadlines_page",
        "exact_daad_listing",
    },
    "tuition_fee_lookup": {
        "program_page",
        "fee_page",
        "official_policy_page",
        "official_policy_pdf",
        "exact_daad_listing",
    },
    "application_portal_lookup": {
        "program_page",
        "admissions_page",
        "application_page",
        "program_faq_page",
        "program_application_page",
        "portal_instruction_page",
        "application_portal_page",
        "exact_daad_listing",
    },
    "application_process_lookup": {
        "program_page",
        "admissions_page",
        "program_faq_page",
        "program_application_page",
        "application_portal_page",
        "document_checklist_pdf",
        "exact_daad_listing",
    },
    "document_requirement_lookup": {
        "program_page",
        "admissions_page",
        "application_page",
        "program_faq_page",
        "program_application_page",
        "document_checklist_page",
        "document_checklist_pdf",
        "official_policy_pdf",
        "exact_daad_listing",
    },
    "curriculum_lookup": {"program_page", "curriculum_page", "module_handbook"},
}

HARD_REJECT_PAGE_TYPES = {
    "research_page",
    "thesis_pdf",
    "blog",
    "profile_page",
    "news_event",
    "outgoing_exchange_page",
}


def classify_page_type(url: str, title: str = "", snippet: str = "") -> str:
    domain = _domain(url)
    combined = f"{url} {title} {snippet}".lower()
    is_pdf = _is_pdf_url(url)
    if "daad.de" in domain and (
        "international-programmes" in combined
        or any(term in combined for term in ("degree programme", "degree program", "master", "msc"))
    ):
        return "exact_daad_listing"
    if any(
        term in combined
        for term in (
            "outgoing",
            "exchange",
            "fauexchange",
            "partnerhochschulen",
            "studieren-im-ausland",
            "direktaustausch",
        )
    ):
        return "outgoing_exchange_page"
    if any(term in combined for term in ("cris.", "/projects/", "research project")):
        return "research_page"
    if is_pdf and any(term in combined for term in ("/theses/", "thesis", "dissertation")):
        return "thesis_pdf"
    if any(term in combined for term in ("/blog", "blog/", "pattern-recognition-blog")):
        return "blog"
    if any(term in combined for term in ("faudir", "/people/", "/profiles/", "professor")):
        return "profile_page"
    if any(term in combined for term in ("/news", "/event", "events/", "press release")):
        return "news_event"
    if any(term in combined for term in ("module handbook", "modulhandbuch", "module-handbook")):
        return "module_handbook"
    if any(
        term in combined
        for term in (
            "degree-programs/master-",
            "degree-programs/detail/",
            "studienangebot/detail/",
            "studiengaenge/master-",
            "studiengänge/master-",
            "degree programme",
            "degree program",
            "study program",
            "studiengang",
        )
    ) and any(term in combined for term in ("master", "msc", "m.sc", "informatik", "informatics")):
        return "program_page"
    if re.search(r"/master-[^/]+/faq\b|/msc[^/]*/faq\b|/faq\b", combined) and any(
        term in combined for term in ("master", "msc", "application", "admission", "language")
    ):
        return "program_faq_page"
    if "/faq" in combined or "frequently asked questions" in combined:
        return "faq_page"
    if re.search(r"/master-[^/]+/application|/application-master\b|/msc[^/]*/application", combined):
        return "program_application_page"
    if any(
        term in combined
        for term in (
            "important dates and deadlines",
            "dates and deadlines",
            "application deadlines",
            "application deadline",
            "bewerbungsfristen",
            "bewerbungsfrist",
            "bewerbungszeitraum",
        )
    ):
        return "dates_deadlines_page"
    if "language-proficiency" in combined or "language proficiency" in combined:
        return "language_requirement_page"
    if any(
        term in combined
        for term in ("curriculum", "modules", "course catalogue", "courses")
    ) and not any(term in combined for term in ("application period", "application deadline", "bewerbungsfrist")):
        return "curriculum_page"
    if any(
        term in combined
        for term in (
            "beizufügende unterlagen",
            "beizufuegende unterlagen",
            "required documents",
            "document checklist",
            "bewerbungsunterlagen",
        )
    ):
        return "document_checklist_pdf" if is_pdf else "document_checklist_page"
    if any(
        term in combined
        for term in (
            "sprachnachweis",
            "language requirement",
            "language requirements",
            "english language requirements",
            "language proof",
            "proof of language",
            "language proficiency",
            "language-proficiency",
            "english certificate",
            "cefr",
        )
    ):
        return "language_requirement_page"
    if any(term in combined for term in ("tuition", "fees", "semesterbeitrag", "studiengeb")):
        return "fee_page"
    if any(
        term in combined
        for term in (
            "application portal",
            "apply online",
            "tumonline",
            "campo",
            "uni-assist",
            "vpd",
            "application platform",
            "bewerbungsportal",
        )
    ):
        return "portal_instruction_page"
    if any(
        term in combined for term in ("admission", "application", "apply", "zulassung", "bewerbung")
    ):
        if any(term in combined for term in ("apply", "application", "bewerbung")):
            return "application_page"
        return "admissions_page"
    if any(
        term in combined
        for term in (
            "degree-program",
            "degree programme",
            "degree program",
            "study program",
            "studiengang",
        )
    ):
        return "program_page"
    if any(term in combined for term in ("regulation", "ordnung", "policy")):
        return "official_policy_pdf" if is_pdf else "official_policy_page"
    return "unknown"


def _page_type_match_score(page_type: str, plan: QueryPlan) -> float:
    allowed = PAGE_TYPES_BY_INTENT.get(plan.intent)
    if allowed is None:
        allowed = {
            "program_page",
            "admissions_page",
            "language_requirement_page",
            "fee_page",
            "application_portal_page",
            "document_checklist_pdf",
            "official_policy_pdf",
            "exact_daad_listing",
        }
    if page_type in allowed:
        return 1.0
    if page_type == "unknown":
        return 0.35
    if page_type == "program_page":
        return 0.65
    if page_type in {
        "program_faq_page",
        "program_application_page",
        "application_page",
        "faq_page",
        "portal_instruction_page",
        "dates_deadlines_page",
        "document_checklist_page",
        "official_policy_page",
    }:
        return 0.75
    return 0.0


def _page_type_rejection_reason(page_type: str, plan: QueryPlan) -> str:
    if page_type in HARD_REJECT_PAGE_TYPES:
        return f"rejected_page_type:{page_type}"
    if page_type == "module_handbook" and plan.intent != "curriculum_lookup":
        return "module_handbook_not_requested"
    if page_type == "curriculum_page" and plan.intent not in {
        "curriculum_lookup",
        "program_overview_lookup",
    }:
        return "curriculum_page_not_requested"
    allowed = PAGE_TYPES_BY_INTENT.get(plan.intent)
    if allowed and page_type not in allowed and page_type != "unknown":
        return f"page_type_not_relevant:{page_type}"
    return ""


def _degree_level_match_score(text: str, plan: QueryPlan) -> float:
    if not plan.degree_level:
        return 0.5
    lowered = text.lower()
    if plan.degree_level == "master":
        if re.search(r"\b(bsc|b\.sc|bachelor)\b", lowered) and not re.search(
            r"\b(msc|m\.sc|master)\b", lowered
        ):
            return 0.0
        return 1.0 if re.search(r"\b(msc|m\.sc|master)\b", lowered) else 0.45
    if plan.degree_level == "bachelor":
        if re.search(r"\b(msc|m\.sc|master)\b", lowered) and not re.search(
            r"\b(bsc|b\.sc|bachelor)\b", lowered
        ):
            return 0.0
        return 1.0 if re.search(r"\b(bsc|b\.sc|bachelor)\b", lowered) else 0.45
    return 0.5


def _degree_level_path_rejection_reason(url: str, plan: QueryPlan) -> str:
    if not plan.degree_level:
        return ""
    path = urlparse(url).path.lower()
    if plan.degree_level == "master" and re.search(
        r"(^|/|-)(bachelor|bsc|b\.sc|ba|b\.a)(/|-|$)", path
    ):
        return "wrong_degree_level"
    if plan.degree_level == "bachelor" and re.search(
        r"(^|/|-)(master|msc|m\.sc|ma|m\.a)(/|-|$)", path
    ):
        return "wrong_degree_level"
    return ""


def _degree_signal(value: str) -> str:
    lowered = str(value or "").lower()
    has_master = bool(re.search(r"\b(msc|m\.sc|master|master's)\b|/master[-_/]|-m-sc\b|msc", lowered))
    has_bachelor = bool(
        re.search(r"\b(bsc|b\.sc|bachelor|bachelor's)\b|/bachelor[-_/]|-b-sc\b|/bsc\b", lowered)
    )
    if has_master and not has_bachelor:
        return "master"
    if has_bachelor and not has_master:
        return "bachelor"
    if has_master and has_bachelor:
        return "mixed"
    return "unknown"


def _degree_signal_matches_target(signal: str, plan: QueryPlan) -> bool:
    if not plan.degree_level or signal in {"unknown", "mixed"}:
        return True
    return signal == plan.degree_level


def _degree_signal_debug(url: str, title: str, snippet: str, plan: QueryPlan) -> dict[str, Any]:
    path_signal = _degree_signal(urlparse(url).path)
    title_signal = _degree_signal(title)
    snippet_signal = _degree_signal(snippet)
    final_signal = "unknown"
    for signal in (path_signal, title_signal, snippet_signal):
        if signal in {"master", "bachelor"}:
            final_signal = signal
            break
    if final_signal == "unknown" and "mixed" in {path_signal, title_signal, snippet_signal}:
        final_signal = "mixed"
    return {
        "path_degree_signal": path_signal,
        "title_degree_signal": title_signal,
        "snippet_degree_signal": snippet_signal,
        "final_degree_signal": final_signal,
        "final_degree_match": _degree_signal_matches_target(final_signal, plan),
    }


def _span_degree_rejection_reason(text: str, plan: QueryPlan) -> str:
    signal = _degree_signal(text)
    if plan.degree_level == "master" and signal == "bachelor":
        return "wrong_degree_level_span"
    if plan.degree_level == "bachelor" and signal == "master":
        return "wrong_degree_level_span"
    return ""


def _path_family(url: str) -> str:
    parsed = urlparse(url)
    parts = [
        part
        for part in parsed.path.lower().split("/")
        if part and not re.fullmatch(r"\d{4}|\d{2}|files?|en|de", part)
    ]
    return "/".join([_domain(url), *parts[:3]])


def _candidate_has_strong_retrieval_signal(plan: QueryPlan, candidate: dict[str, Any]) -> bool:
    if bool(candidate.get("strong_official_candidate", False)):
        return True
    page_type = str(candidate.get("page_type") or "")
    if page_type in HARD_REJECT_PAGE_TYPES:
        return False
    if page_type == "module_handbook" and plan.intent != "curriculum_lookup":
        return False
    domain = _domain(str(candidate.get("url", "")))
    program_match = float(candidate.get("program_match_score") or 0.0)
    field_relevance = float(candidate.get("field_relevance_score") or 0.0)
    degree_match = float(
        (candidate.get("url_score_components") or {}).get("degree_level_match", 0.0) or 0.0
    )
    allowed_page_type = _page_type_match_score(page_type, plan) >= 0.65
    exact_configured_domain = _domain_exactly_configured(domain, resolve_official_domains(plan))
    combined = (
        f"{candidate.get('url', '')} {candidate.get('title', '')} {candidate.get('snippet', '')}"
        .lower()
    )
    general_policy = any(
        term in combined
        for term in ("general", "university-wide", "all applicants", "international applicants")
    )
    program_path_hint = any(
        term in combined
        for term in ("master-ai", "artificial-intelligence-m-sc", "master-informatics")
    )
    if program_match >= 0.60 and allowed_page_type and degree_match > 0.0:
        return True
    if exact_configured_domain and allowed_page_type and (program_path_hint or field_relevance >= 0.18):
        return True
    if general_policy and allowed_page_type and field_relevance >= 0.18:
        return True
    return False


def _is_strong_official_candidate(
    *,
    url: str,
    title: str,
    snippet: str,
    plan: QueryPlan,
    tier: str,
    target_domains: list[str],
    page_type: str,
    program_match: float,
    score_components: dict[str, float],
) -> tuple[bool, str]:
    if plan.query_mode != "fast_lookup":
        return False, "not_fast_lookup"
    domain = _domain(url)
    if not _domain_allowed(domain, target_domains or resolve_official_domains(plan)):
        return False, "not_official_university_domain"
    if tier == "tier3" and not _domain_allowed(domain, target_domains or resolve_official_domains(plan)):
        return False, "third_party_tier"
    if page_type in HARD_REJECT_PAGE_TYPES:
        return False, f"rejected_page_type:{page_type}"
    page_type_reason = _page_type_rejection_reason(page_type, plan)
    if page_type_reason:
        return False, page_type_reason
    if _degree_level_path_rejection_reason(url, plan):
        return False, "wrong_degree_level"
    degree_score = float(score_components.get("degree_level_match", 0.0) or 0.0)
    if degree_score <= 0.0:
        return False, "wrong_degree_level"
    combined = f"{url} {title} {snippet}".lower()
    program_tokens = _target_program_tokens(plan)
    program_alias_match = bool(program_tokens and program_tokens.intersection(set(re.findall(r"[a-z0-9äöüß]{3,}", combined))))
    if plan.program and program_match < 0.55 and not program_alias_match:
        return False, "weak_exact_program_match"
    if "biomedical engineering" in combined and "biomedical engineering" not in plan.program.lower():
        return False, "different_named_program"
    if "part-time" in combined or "part time" in combined:
        return False, "part_time_variant_not_requested"
    if _result_mentions_wrong_program(url, title, snippet, plan):
        return False, "different_named_program"
    if _page_type_match_score(page_type, plan) < 0.65:
        return False, f"page_type_not_relevant:{page_type}"
    return True, (
        "strong_official_candidate: official_domain + target_program_or_alias + "
        "degree_level + intent_allowed_page_type"
    )


def _language_candidate_priority(plan: QueryPlan, candidate: dict[str, Any]) -> float:
    if plan.intent != "language_requirement_lookup":
        return 0.0
    url = str(candidate.get("url", "")).lower()
    page_type = str(candidate.get("page_type") or "")
    if "language-proficiency" in url:
        return 0.75
    if page_type == "language_requirement_page":
        return 0.50
    if "/master-" in url and page_type == "program_faq_page":
        return 0.32
    if "/master-" in url and page_type == "program_application_page":
        return 0.28
    if page_type == "document_checklist_pdf":
        return 0.20
    if page_type == "exact_daad_listing":
        return 0.10
    if "/bachelor-" in url:
        return -0.60
    return 0.0


def _selected_has_strong_retrieval_signal(plan: QueryPlan, selected: list[dict[str, Any]]) -> bool:
    return any(_candidate_has_strong_retrieval_signal(plan, item) for item in selected)


def _diversify_candidates(candidates: list[dict[str, Any]], max_urls: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for item in candidates:
        family = _path_family(str(item.get("url", "")))
        if family in seen_families:
            continue
        seen_families.add(family)
        selected.append(item)
        if len(selected) >= max_urls:
            return selected
    if len(selected) < max_urls:
        selected_urls = {str(item.get("url", "")) for item in selected}
        for item in candidates:
            if str(item.get("url", "")) in selected_urls:
                continue
            selected.append(item)
            if len(selected) >= max_urls:
                break
    return selected


def _url_score_components(
    url: str,
    title: str,
    snippet: str,
    plan: QueryPlan,
    *,
    tier: str,
    target_domains: list[str],
    page_type: str,
    program_match: float,
    field_relevance: float,
) -> dict[str, float]:
    domain = _domain(url)
    combined = f"{url} {title} {snippet}"
    domain_match = 1.0 if _domain_allowed(domain, target_domains) else 0.0
    if tier == "tier2" and _domain_allowed(domain, OFFICIAL_SECONDARY_DOMAINS):
        domain_match = 0.85
    elif tier == "tier3" and not domain_match:
        domain_match = 0.35
    page_type_match = _page_type_match_score(page_type, plan)
    degree_level_match = _degree_level_match_score(combined, plan)
    return {
        "domain_match": domain_match,
        "program_match": max(0.0, min(1.0, program_match)),
        "intent_field_match": max(0.0, min(1.0, field_relevance)),
        "page_type_match": page_type_match,
        "degree_level_match": degree_level_match,
        "url_score": (
            (0.30 * domain_match)
            + (0.25 * max(0.0, min(1.0, program_match)))
            + (0.20 * max(0.0, min(1.0, field_relevance)))
            + (0.15 * page_type_match)
            + (0.10 * degree_level_match)
        ),
    }


def _result_mentions_wrong_program(url: str, title: str, snippet: str, plan: QueryPlan) -> bool:
    return _mentions_wrong_program(f"{url} {title} {snippet}", plan)


def _student_page_rejection_reason(url: str, title: str, snippet: str, plan: QueryPlan) -> str:
    combined = f"{url} {title} {snippet}".lower()
    domain = _domain(url)
    if _domain_allowed(domain, OFFICIAL_SECONDARY_DOMAINS):
        return ""
    degree_path_reason = _degree_level_path_rejection_reason(url, plan)
    if degree_path_reason:
        return degree_path_reason
    if any(signal in combined for signal in NON_STUDENT_PAGE_SIGNALS):
        return "non_student_or_research_page"
    narrow_student_intents = {
        "language_requirement_lookup",
        "deadline_lookup",
        "tuition_fee_lookup",
        "application_portal_lookup",
        "document_requirement_lookup",
        "admission_requirement_lookup",
        "eligibility_check",
    }
    if plan.intent in narrow_student_intents:
        if any(
            term in combined
            for term in (
                "outgoing",
                "exchange",
                "fauexchange",
                "partnerhochschulen",
                "studieren-im-ausland",
                "direktaustausch",
            )
        ):
            return "outgoing_exchange_page_not_relevant"
        if "degree b.sc" in combined or "bachelor of science" in combined:
            return "wrong_degree_level"
        if (
            _is_pdf_url(url)
            and plan.intent != "curriculum_lookup"
            and any(
                term in combined
                for term in ("module", "po-version", "examination language", "workload")
            )
        ):
            return "module_or_curriculum_pdf_not_requested"
        if not any(signal in combined for signal in STUDENT_PAGE_SIGNALS):
            return "missing_student_admission_page_signal"
        general_policy_match = any(
            term in combined
            for term in (
                "general",
                "university-wide",
                "all applicants",
                "international applicants",
            )
        )
        if (
            plan.intent == "language_requirement_lookup"
            and _domain_exactly_configured(domain, resolve_official_domains(plan))
            and (
                classify_page_type(url, title, snippet)
                in {"language_requirement_page", "admissions_page", "application_portal_page"}
                or any(
                    term in combined
                    for term in (
                        "language-proficiency",
                        "language proficiency",
                        "english certificate",
                        "cefr",
                        "sprachnachweis",
                        "master-ai",
                        "application-faq",
                    )
                )
            )
        ):
            general_policy_match = True
        if (
            plan.intent == "language_requirement_lookup"
            and not _domain_exactly_configured(domain, resolve_official_domains(plan))
            and any(
                term in combined
                for term in (
                    "outgoing",
                    "exchange",
                    "fauexchange",
                    "partnerhochschulen",
                    "studieren-im-ausland",
                    "direktaustausch",
                )
            )
        ):
            return "outgoing_exchange_page_not_relevant"
        if (
            plan.program
            and _program_match_score(combined, plan) < 0.60
            and not general_policy_match
        ):
            return "weak_exact_program_match"
        if (
            "part-time" in combined or "part time" in combined
        ) and not re.search(r"\b(part[-\s]?time|teilzeit)\b", plan.user_intent.lower()):
            return "part_time_variant_not_requested"
    return ""


def _skip_reason_for_search_result(url: str, title: str, snippet: str, plan: QueryPlan) -> str:
    domain = _domain(url)
    combined = f"{url} {title} {snippet}".lower()
    if plan.ambiguity_note and "fau.edu" in domain:
        return "ambiguous_secondary_institution_florida_atlantic"
    if any(bad in domain for bad in LOW_QUALITY_DOMAINS):
        return "low_quality_domain"
    if any(term in combined for term in ("consultant", "forum", "reddit", "quora")):
        return "low_quality_or_untrusted_source"
    subdomain_reason = _unknown_program_subdomain_reason(url, title, snippet, plan)
    if subdomain_reason:
        return subdomain_reason
    weak_subdomain_reason = _weak_unconfigured_subdomain_reason(url, title, snippet, plan)
    if weak_subdomain_reason:
        return weak_subdomain_reason
    if _url_relevance_score(url, title, snippet, plan) <= -0.45:
        return "url_path_or_title_irrelevant_to_target_program"
    student_page_reason = _student_page_rejection_reason(url, title, snippet, plan)
    if student_page_reason:
        return student_page_reason
    page_type = classify_page_type(url, title, snippet)
    page_type_reason = _page_type_rejection_reason(page_type, plan)
    if page_type_reason:
        return page_type_reason
    if _result_mentions_wrong_program(url, title, snippet, plan):
        return "different_named_program"
    if plan.intent == "deadline_lookup" and _is_pdf_url(url):
        combined = f"{url} {title} {snippet}".lower()
        if "daad" in _domain(url) and _program_match_score(combined, plan) < 0.65:
            return "generic_deadline_pdf_without_exact_program"
    if not _accepted_search_result(url, title, snippet, plan):
        return "low_quality_or_irrelevant_source"
    return ""


def _tier_domain_rejection_reason(
    url: str,
    *,
    tier: str,
    target_domains: list[str],
    allow_secondary: bool,
    allow_third_party: bool,
) -> str:
    domain = _domain(url)
    if not target_domains:
        return ""
    if tier.startswith("tier1") and not _domain_allowed(domain, target_domains):
        return "outside_target_university_domains"
    if tier == "tier2" and not (
        _domain_allowed(domain, target_domains)
        or (allow_secondary and _domain_allowed(domain, OFFICIAL_SECONDARY_DOMAINS))
    ):
        return "outside_official_secondary_domains"
    if tier == "tier3":
        if _domain_allowed(domain, target_domains) or _domain_allowed(
            domain, OFFICIAL_SECONDARY_DOMAINS
        ):
            return ""
        if not allow_third_party:
            return "third_party_not_allowed_when_official_evidence_exists"
    return ""


def _merge_debug_lists(existing: list[Any], new_items: list[Any]) -> list[Any]:
    merged = list(existing or [])
    seen = {
        (
            str(item.get("url", "")),
            str(item.get("reason", "")),
            str(item.get("query", "")),
            str(item.get("tier", "")),
        )
        for item in merged
        if isinstance(item, dict)
    }
    for item in new_items or []:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("url", "")),
            str(item.get("reason", "")),
            str(item.get("query", "")),
            str(item.get("tier", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _top_rejection_reasons(debug_collector: dict[str, Any] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not debug_collector:
        return counts
    for item in debug_collector.get("skipped_urls", []):
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason", "")).strip() or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def select_and_deduplicate_urls(
    search_results: list[dict[str, Any]],
    plan: QueryPlan,
    *,
    debug_collector: dict[str, Any] | None = None,
    tier: str = "legacy",
    allowed_domains: list[str] | None = None,
    allow_secondary: bool = True,
    allow_third_party: bool = True,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped: list[dict[str, str]] = []
    for search_result in search_results:
        query = str(search_result.get("query", ""))
        query_priority = float(search_result.get("priority") or 0.8)
        query_type = str(search_result.get("type", "official_page"))
        for row in search_result.get("results", []):
            if not isinstance(row, dict):
                continue
            raw_url = row.get("link") or row.get("url") or ""
            url = canonicalize_url(str(raw_url))
            if not url or url.lower() in seen:
                if url:
                    skipped.append({"url": url, "reason": "duplicate_url", "query": query})
                continue
            title = _compact(row.get("title"))
            snippet = _compact(row.get("snippet") or row.get("content"))
            target_domains = allowed_domains or resolve_official_domains(plan)
            tier_rejection = (
                _tier_domain_rejection_reason(
                    url,
                    tier=tier,
                    target_domains=target_domains,
                    allow_secondary=allow_secondary,
                    allow_third_party=allow_third_party,
                )
                if tier != "legacy"
                else ""
            )
            if tier_rejection:
                logger.info(
                    "UniGraph candidate rejected | url=%s | rejected_candidate_reason=%s | strong_official_candidate=false",
                    url,
                    tier_rejection,
                )
                skipped.append(
                    {
                        "url": url,
                        "reason": tier_rejection,
                        "rejected_candidate_reason": tier_rejection,
                        "strong_official_candidate": "false",
                        "query": query,
                        "tier": tier,
                    }
                )
                continue
            skip_reason = _skip_reason_for_search_result(url, title, snippet, plan)
            if skip_reason:
                logger.info(
                    "UniGraph candidate rejected | url=%s | rejected_candidate_reason=%s | strong_official_candidate=false",
                    url,
                    skip_reason,
                )
                skipped.append(
                    {
                        "url": url,
                        "reason": skip_reason,
                        "rejected_candidate_reason": skip_reason,
                        "strong_official_candidate": "false",
                        "query": query,
                        "tier": tier,
                    }
                )
                continue
            seen.add(url.lower())
            document_type = "pdf" if _is_pdf_url(url) or query_type == "pdf" else "html"
            source_quality, source_type = calculate_source_quality(url, document_type=document_type)
            official_boost = 0.25 if source_quality >= 0.75 else 0.0
            pdf_boost = 0.10 if document_type == "pdf" else 0.0
            keyword_boost = (
                0.10
                if any(term in f"{url} {title} {snippet}".lower() for term in OFFICIAL_KEYWORDS)
                else 0.0
            )
            path_relevance = _url_relevance_score(url, title, snippet, plan)
            program_details = _program_match_details(f"{url} {title} {snippet}", plan)
            program_match = float(program_details["score"])
            field_relevance = _field_relevance_score(f"{url} {title} {snippet}", plan)
            page_type = classify_page_type(url, title, snippet)
            page_type_before_override = page_type
            degree_debug = _degree_signal_debug(url, title, snippet, plan)
            score_components = _url_score_components(
                url,
                title,
                snippet,
                plan,
                tier=tier,
                target_domains=target_domains,
                page_type=page_type,
                program_match=program_match,
                field_relevance=field_relevance,
            )
            strong_official_candidate, strong_reason = _is_strong_official_candidate(
                url=url,
                title=title,
                snippet=snippet,
                plan=plan,
                tier=tier,
                target_domains=target_domains,
                page_type=page_type,
                program_match=program_match,
                score_components=score_components,
            )
            logger.info(
                "UniGraph candidate evaluated | url=%s | strong_official_candidate=%s | reason=%s",
                url,
                strong_official_candidate,
                strong_reason,
            )
            if plan.intent in {
                "language_requirement_lookup",
                "deadline_lookup",
                "tuition_fee_lookup",
                "document_requirement_lookup",
                "application_portal_lookup",
                "curriculum_lookup",
            }:
                likely_answer_page = (
                    plan.intent == "language_requirement_lookup"
                    and page_type
                    in {
                        "language_requirement_page",
                        "admissions_page",
                        "application_portal_page",
                        "program_page",
                    }
                    and (
                        program_match >= 0.45
                        or any(
                            term in f"{url} {title} {snippet}".lower()
                            for term in (
                                "master-ai",
                                "language-proficiency",
                                "application-faq",
                                "international applicants",
                            )
                        )
                    )
                )
                if field_relevance < 0.08 and not likely_answer_page and not strong_official_candidate:
                    logger.info(
                        "UniGraph candidate rejected | url=%s | rejected_candidate_reason=weak_field_relevance | strong_official_candidate=false",
                        url,
                    )
                    skipped.append(
                        {
                            "url": url,
                            "reason": "weak_field_relevance",
                            "strong_official_candidate": "false",
                            "rejected_candidate_reason": "weak_field_relevance",
                            "query": query,
                            "tier": tier,
                        }
                    )
                    continue
            if float(score_components["degree_level_match"]) <= 0.0:
                logger.info(
                    "UniGraph candidate rejected | url=%s | rejected_candidate_reason=wrong_degree_level | strong_official_candidate=false",
                    url,
                )
                skipped.append(
                    {
                        "url": url,
                        "reason": "wrong_degree_level",
                        "strong_official_candidate": "false",
                        "rejected_candidate_reason": "wrong_degree_level",
                        "query": query,
                        "tier": tier,
                        "page_type": page_type,
                    }
                )
                continue
            min_url_score = 0.46 if tier.startswith("tier1") or tier == "legacy" else 0.50
            if tier == "tier3":
                min_url_score = 0.58
            if float(score_components["url_score"]) < min_url_score:
                logger.info(
                    "UniGraph candidate rejected | url=%s | rejected_candidate_reason=low_url_score | strong_official_candidate=false",
                    url,
                )
                skipped.append(
                    {
                        "url": url,
                        "reason": "low_url_score",
                        "strong_official_candidate": "false",
                        "rejected_candidate_reason": "low_url_score",
                        "query": query,
                        "tier": tier,
                        "page_type": page_type,
                        "url_score": str(round(float(score_components["url_score"]), 4)),
                    }
                )
                continue
            logger.info(
                "UniGraph candidate accepted | url=%s | strong_official_candidate=%s | accepted_candidate_reason=%s | fetched_before_field_match=%s",
                url,
                strong_official_candidate,
                strong_reason if strong_official_candidate else "score_threshold_met",
                strong_official_candidate and field_relevance < 0.08,
            )
            candidates.append(
                {
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "query": query,
                    "query_type": query_type,
                    "document_type": document_type,
                    "source_quality": source_quality,
                    "source_type": source_type,
                    "url_relevance_score": path_relevance,
                    "program_match_score": program_match,
                    "program_match_components": program_details,
                    "field_relevance_score": field_relevance,
                    "degree_level_match": bool(program_details.get("degree_level_match")),
                    "intent_context_match": bool(program_details.get("intent_context_match")),
                    "field_match": field_relevance >= 0.08,
                    "page_type": page_type,
                    "page_type_before_override": page_type_before_override,
                    "page_type_after_override": page_type,
                    "strong_official_candidate": strong_official_candidate,
                    "accepted_candidate_reason": (
                        strong_reason
                        if strong_official_candidate
                        else (
                            f"url_score={float(score_components['url_score']):.2f}; "
                            f"field_relevance={field_relevance:.2f}"
                        )
                    ),
                    "fetched_before_field_match": strong_official_candidate
                    and field_relevance < 0.08,
                    **degree_debug,
                    "url_score": score_components["url_score"],
                    "url_score_components": score_components,
                    "retrieval_tier": tier,
                    "selection_reason": (
                        f"page_type={page_type}; program_match={program_match:.2f}; "
                        f"field_relevance={field_relevance:.2f}; "
                        f"path_degree_signal={degree_debug['path_degree_signal']}; "
                        f"strong_official_candidate={strong_official_candidate}"
                    ),
                    "score": (0.50 * float(score_components["url_score"]))
                    + (0.25 * query_priority)
                    + (0.15 * source_quality)
                    + official_boost
                    + pdf_boost
                    + keyword_boost
                    + (0.10 * path_relevance)
                    + _language_candidate_priority(plan, {
                        "url": url,
                        "page_type": page_type,
                    }),
                }
            )
    candidates.sort(
        key=lambda item: (float(item["score"]), float(item["source_quality"])), reverse=True
    )
    max_urls = _mode_limit(plan, "max_urls", MAX_TOTAL_URLS_TO_FETCH)
    selected = _diversify_candidates(candidates, max_urls)
    logger.info(
        "UniGraph deduplicated URLs | candidates=%s | selected=%s", len(candidates), selected
    )
    if debug_collector is not None:
        debug_collector["skipped_urls"] = _merge_debug_lists(
            debug_collector.get("skipped_urls", []), skipped
        )
        debug_collector["rejected_pdfs"] = _merge_debug_lists(
            debug_collector.get("rejected_pdfs", []),
            [item for item in skipped if _is_pdf_url(str(item.get("url", "")))],
        )
        new_source_scores = [
            {
                "url": item["url"],
                "domain": _domain(str(item["url"])),
                "source_type": item["source_type"],
                "source_quality": item["source_quality"],
                "selection_score": item["score"],
                "url_relevance_score": item.get("url_relevance_score", 0.0),
                "program_match_score": item.get("program_match_score", 0.0),
                "program_match_components": item.get("program_match_components", {}),
                "field_relevance_score": item.get("field_relevance_score", 0.0),
                "degree_level_match": item.get("degree_level_match", False),
                "intent_context_match": item.get("intent_context_match", False),
                "field_match": item.get("field_match", False),
                "strong_official_candidate": item.get("strong_official_candidate", False),
                "accepted_candidate_reason": item.get("accepted_candidate_reason", ""),
                "fetched_before_field_match": item.get("fetched_before_field_match", False),
                "page_type": item.get("page_type", "unknown"),
                "page_type_before_override": item.get("page_type_before_override", "unknown"),
                "page_type_after_override": item.get("page_type_after_override", "unknown"),
                "path_degree_signal": item.get("path_degree_signal", "unknown"),
                "title_degree_signal": item.get("title_degree_signal", "unknown"),
                "snippet_degree_signal": item.get("snippet_degree_signal", "unknown"),
                "final_degree_signal": item.get("final_degree_signal", "unknown"),
                "final_degree_match": item.get("final_degree_match", False),
                "url_score": item.get("url_score", 0.0),
                "url_score_components": item.get("url_score_components", {}),
                "selection_reason": item.get("selection_reason", ""),
                "retrieval_tier": item.get("retrieval_tier", tier),
                "query": item["query"],
            }
            for item in candidates
        ]
        debug_collector["source_scores"] = [
            *debug_collector.get("source_scores", []),
            *new_source_scores,
        ]
    return selected


def _html_to_text(raw_html: str) -> str:
    text = re.sub(
        r"<(script|style|nav|header|footer|aside)\b[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S
    )
    text = re.sub(
        r"</?(p|br|li|tr|td|th|div|section|article|h[1-6])\b[^>]*>", "\n", text, flags=re.I
    )
    text = re.sub(r"<[^>]+>", " ", text)
    return _compact(unescape(text))


async def extract_html_content(url_info: dict[str, Any]) -> ExtractedContent | None:
    url = str(url_info["url"])
    fallback_text = _compact(
        " ".join(
            str(url_info.get(key, "") or "")
            for key in ("title", "snippet")
            if str(url_info.get(key, "") or "").strip()
        )
    )
    try:
        payload = await aextract_urls(
            [url], extract_depth="advanced", query=str(url_info.get("query", ""))
        )
    except Exception as exc:
        logger.warning("UniGraph Tavily extract failed | url=%s | error=%s", url, exc)
        text = fallback_text
        if len(text) < 80:
            return None
        return ExtractedContent(
            url=url,
            title=str(url_info.get("title", "")),
            domain=_domain(url),
            source_type=str(url_info.get("source_type", "other")),
            document_type="html",
            source_quality=float(url_info.get("source_quality") or 0.5),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            query=str(url_info.get("query", "")),
            pages=[ExtractedPage(text=text)],
        )
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not results:
        text = fallback_text
        if len(text) < 80:
            return None
        return ExtractedContent(
            url=url,
            title=str(url_info.get("title", "")),
            domain=_domain(url),
            source_type=str(url_info.get("source_type", "other")),
            document_type="html",
            source_quality=float(url_info.get("source_quality") or 0.5),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            query=str(url_info.get("query", "")),
            pages=[ExtractedPage(text=text)],
        )
    row = results[0] if isinstance(results[0], dict) else {}
    text = _compact(row.get("raw_content") or row.get("content") or "")
    if "<" in text and ">" in text:
        text = _html_to_text(text)
    if len(text) < 120:
        text = fallback_text if len(fallback_text) > len(text) else text
    elif fallback_text and fallback_text.lower() not in text.lower():
        # Tavily extraction can miss the search-result snippet even when the
        # snippet contains the answer-bearing sentence. Keep it as a compact
        # fallback span so field mapping can still use selected official results.
        text = _compact(f"{text} {fallback_text}")
    if len(text) < 80:
        return None
    return ExtractedContent(
        url=url,
        title=str(url_info.get("title", "")),
        domain=_domain(url),
        source_type=str(url_info.get("source_type", "other")),
        document_type="html",
        source_quality=float(url_info.get("source_quality") or 0.5),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        query=str(url_info.get("query", "")),
        pages=[ExtractedPage(text=text)],
    )


def _download_pdf_to_temp(url: str) -> str | None:
    max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    request = urllib.request.Request(url, headers={"User-Agent": "unigraph-phase1-research/1.0"})
    with urllib.request.urlopen(
        request, timeout=float(settings.web_search.page_fetch_timeout_seconds)
    ) as response:
        content_type = str(response.headers.get("content-type", "")).lower()
        content_length = int(response.headers.get("content-length") or 0)
        if content_length and content_length > max_bytes:
            logger.info("UniGraph skipped oversized PDF | url=%s | size=%s", url, content_length)
            return None
        if "pdf" not in content_type and not _is_pdf_url(url):
            return None
        fd, path = tempfile.mkstemp(prefix="unigraph_pdf_", suffix=".pdf")
        read_total = 0
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                read_total += len(chunk)
                if read_total > max_bytes:
                    handle.close()
                    os.unlink(path)
                    logger.info("UniGraph skipped oversized PDF while reading | url=%s", url)
                    return None
                handle.write(chunk)
    return path


def _pdf_page_has_signal(text: str, plan: QueryPlan | None) -> bool:
    if plan is None:
        return True
    lowered = text.lower()
    terms = [term.lower() for term in _field_query_keywords(plan)]
    terms.extend(token.lower() for token in _target_program_tokens(plan))
    if plan.degree_level == "master":
        terms.extend(["master", "msc", "m.sc"])
    elif plan.degree_level == "bachelor":
        terms.extend(["bachelor", "bsc", "b.sc"])
    terms = [term for term in dict.fromkeys(terms) if term]
    return any(term in lowered for term in terms)


def _extract_pdf_pages(
    path: str, url: str, plan: QueryPlan | None = None, max_pages: int | None = None
) -> list[ExtractedPage]:
    if pdfplumber is None:
        logger.warning("UniGraph PDF extraction skipped; pdfplumber is unavailable.")
        return []
    pages: list[ExtractedPage] = []
    with pdfplumber.open(path) as pdf:
        preview_text_parts: list[str] = []
        for page in pdf.pages[:2]:
            preview_text_parts.append(page.extract_text() or "")
        if plan is not None and not _pdf_page_has_signal(" ".join(preview_text_parts), plan):
            logger.info("UniGraph PDF rejected after preview | url=%s", url)
            return []
        page_limit = max_pages or MAX_PDF_PAGES
        for index, page in enumerate(pdf.pages[:page_limit], start=1):
            parts: list[str] = []
            text = page.extract_text() or ""
            if text:
                parts.append(text)
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for table_index, table in enumerate(tables, start=1):
                if not table:
                    continue
                for row_index, row in enumerate(table, start=1):
                    row_text = " | ".join(_compact(cell) for cell in (row or []))
                    if row_text:
                        parts.append(f"[Table {table_index} Row {row_index}] {row_text}")
            page_text = _compact("\n".join(parts))
            if page_text:
                if plan is not None and index > 2 and not _pdf_page_has_signal(page_text, plan):
                    continue
                pages.append(ExtractedPage(text=page_text, page_number=index))
    logger.info("UniGraph PDF read | url=%s | pages=%s", url, len(pages))
    return pages


def _pdf_prefetch_rejection_reason(url_info: dict[str, Any], plan: QueryPlan | None) -> str:
    if plan is None:
        return ""
    page_type = str(
        url_info.get("page_type")
        or classify_page_type(
            str(url_info.get("url", "")),
            str(url_info.get("title", "")),
            str(url_info.get("snippet", "")),
        )
    )
    if _page_type_rejection_reason(page_type, plan):
        return _page_type_rejection_reason(page_type, plan)
    combined = f"{url_info.get('url', '')} {url_info.get('title', '')} {url_info.get('snippet', '')}".lower()
    if any(term in combined for term in ("thesis", "/theses/", "dissertation")):
        return "thesis_pdf_not_relevant"
    if (
        any(term in combined for term in ("module handbook", "modulhandbuch", "po-version"))
        and plan.intent != "curriculum_lookup"
    ):
        return "module_handbook_not_requested"
    return ""


async def extract_pdf_content(
    url_info: dict[str, Any],
    plan: QueryPlan | None = None,
    debug_collector: dict[str, Any] | None = None,
) -> ExtractedContent | None:
    url = str(url_info["url"])
    rejection_reason = _pdf_prefetch_rejection_reason(url_info, plan)
    if rejection_reason:
        if debug_collector is not None:
            debug_collector.setdefault("rejected_pdfs", []).append(
                {"url": url, "reason": rejection_reason, "stage": "pre_download"}
            )
        return None

    def _read_pdf() -> list[ExtractedPage]:
        path = _download_pdf_to_temp(url)
        if not path:
            return []
        try:
            max_pages = _mode_limit(plan, "max_pdf_pages", MAX_PDF_PAGES) if plan else MAX_PDF_PAGES
            return _extract_pdf_pages(path, url, plan=plan, max_pages=max_pages)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    try:
        pages = await asyncio.to_thread(_read_pdf)
    except Exception as exc:
        logger.warning("UniGraph PDF extraction failed | url=%s | error=%s", url, exc)
        return None
    if not pages:
        if debug_collector is not None:
            debug_collector.setdefault("rejected_pdfs", []).append(
                {"url": url, "reason": "no_relevant_pdf_pages_extracted", "stage": "post_download"}
            )
        return None
    return ExtractedContent(
        url=url,
        title=str(url_info.get("title", "")),
        domain=_domain(url),
        source_type=str(url_info.get("source_type", "other")),
        document_type="pdf",
        source_quality=float(url_info.get("source_quality") or 0.5),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        query=str(url_info.get("query", "")),
        pages=pages,
    )


def _has_direct_required_evidence(plan: QueryPlan, grouped: dict[str, list[EvidenceChunk]]) -> bool:
    required = plan.required_fields or plan.required_info or ["general_information"]
    for field_name in required:
        rows = grouped.get(field_name, [])
        if not any(
            row.support_level == "direct"
            and float(row.scoring.get("source_quality", 0.0) or 0.0) >= 0.75
            for row in rows[:3]
        ):
            return False
    return True


async def extract_all_contents(
    selected_urls: list[dict[str, Any]],
    plan: QueryPlan | None = None,
    debug_collector: dict[str, Any] | None = None,
) -> list[ExtractedContent]:
    pdf_seen = 0
    bounded: list[dict[str, Any]] = []
    max_pdfs = _mode_limit(plan, "max_pdfs", MAX_PDFS_TO_READ) if plan else MAX_PDFS_TO_READ
    max_urls = (
        _mode_limit(plan, "max_urls", MAX_TOTAL_URLS_TO_FETCH) if plan else MAX_TOTAL_URLS_TO_FETCH
    )
    for item in selected_urls:
        if item.get("document_type") == "pdf":
            if pdf_seen >= max_pdfs:
                if debug_collector is not None:
                    debug_collector.setdefault("rejected_pdfs", []).append(
                        {
                            "url": item.get("url", ""),
                            "reason": "query_mode_pdf_limit_reached",
                            "stage": "pre_download",
                        }
                    )
                continue
            pdf_seen += 1
        bounded.append(item)
        if len(bounded) >= max_urls:
            break
    logger.info("UniGraph URLs fetched | urls=%s", [item["url"] for item in bounded])
    extracted: list[ExtractedContent] = []
    early_stop_triggered = False
    for item in bounded:
        result = (
            await extract_pdf_content(item, plan=plan, debug_collector=debug_collector)
            if item.get("document_type") == "pdf"
            else await extract_html_content(item)
        )
        if result is None:
            continue
        extracted.append(result)
        if plan is not None and plan.query_mode == "fast_lookup":
            grouped = group_and_rank_evidence(extracted, plan, debug_collector=debug_collector)
            if _has_direct_required_evidence(plan, grouped):
                early_stop_triggered = True
                break
    if debug_collector is not None:
        debug_collector["early_stop_triggered"] = early_stop_triggered
        debug_collector["fetched_url_candidates"] = [item["url"] for item in bounded]
        debug_collector["pdfs_downloaded"] = [
            item.url for item in extracted if item.document_type == "pdf"
        ]
        debug_collector["pdf_pages_read"] = sum(
            len(item.pages) for item in extracted if item.document_type == "pdf"
        )
        debug_collector["tavily_extract_calls"] = sum(
            1 for item in extracted if item.document_type == "html"
        )
    logger.info(
        "UniGraph extraction complete | docs=%s | pdfs=%s | html=%s",
        len(extracted),
        sum(1 for item in extracted if item.document_type == "pdf"),
        sum(1 for item in extracted if item.document_type == "html"),
    )
    return extracted


def _chunk_debug_payload(chunk: EvidenceChunk) -> dict[str, Any]:
    scoring: dict[str, Any] = {}
    for key, value in chunk.scoring.items():
        if isinstance(value, bool):
            scoring[key] = value
        elif isinstance(value, (int, float)):
            scoring[key] = round(float(value), 4)
        else:
            scoring[key] = value
    return {
        "section": chunk.section,
        "field": chunk.field or chunk.section,
        "evidence_scope": chunk.evidence_scope,
        "support_level": chunk.support_level,
        "selection_reason": chunk.selection_reason,
        "row_or_section": chunk.row_or_section,
        "url": chunk.url,
        "title": chunk.title,
        "domain": chunk.domain,
        "source_type": chunk.source_type,
        "document_type": chunk.document_type,
        "page_number": chunk.page_number,
        "retrieved_at": chunk.retrieved_at,
        "query": chunk.query,
        "score": round(float(chunk.score), 4),
        "scoring": scoring,
        "allowed_for_current_question": bool(
            chunk.scoring.get("allowed_for_current_question", 0.0)
        ),
        "span_selected_from_chunk": bool(chunk.scoring.get("span_selected_from_chunk", 0.0)),
        "selected_span_degree_signal": chunk.scoring.get("selected_span_degree_signal", "unknown"),
        "selected_span_degree_match": bool(
            chunk.scoring.get("selected_span_degree_match", True)
        ),
        "target_program_row_found": bool(chunk.scoring.get("target_program_row_found", 0.0)),
        "neighboring_rows_ignored": bool(chunk.scoring.get("neighboring_rows_ignored", 0.0)),
        "row_confidence": round(float(chunk.scoring.get("row_confidence", 0.0) or 0.0), 4),
        "text": chunk.text[:1200],
    }


def chunk_text(
    text: str, *, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    text = _compact(text)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end]
        if end < len(text):
            split_at = max(chunk.rfind(". "), chunk.rfind("; "), chunk.rfind(" "))
            if split_at > int(chunk_chars * 0.65):
                end = start + split_at + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _section_keywords(section: str, plan: QueryPlan) -> list[str]:
    return _field_terms(section, plan)


def _keyword_match(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.30
    haystack = text.lower()
    matches = sum(1 for keyword in keywords if keyword and keyword.lower() in haystack)
    return min(1.0, matches / max(4, min(len(keywords), 12)))


def _split_evidence_sentences(text: str) -> list[str]:
    compacted = _compact(text)
    if not compacted:
        return []
    row_matches = list(re.finditer(r"\[Table\s+\d+\s+Row\s+\d+\]", compacted))
    if row_matches:
        rows: list[str] = []
        for index, match in enumerate(row_matches):
            end = row_matches[index + 1].start() if index + 1 < len(row_matches) else len(compacted)
            rows.append(_compact(compacted[match.start() : end]))
        return [row for row in rows if row]
    parts = re.split(r"(?=\[Table\s+\d+\s+Row\s+\d+\])|(?<=[.!?])\s+|\s+;\s+|\n+", compacted)
    return [_compact(part) for part in parts if _compact(part)]


def _excluded_terms_for_plan(plan: QueryPlan) -> set[str]:
    terms: set[str] = set()
    for field_name in plan.excluded_fields:
        terms.update(_field_terms(field_name, None))
    gate = INTENT_CHUNK_GATES.get(plan.intent)
    if gate:
        terms.update(str(term).lower() for term in gate.get("exclude", ()))
    return {term for term in terms if len(term) >= 3}


def _sentence_has_excluded_field(sentence: str, plan: QueryPlan, field_name: str) -> bool:
    lowered = sentence.lower()
    field_terms = set(_field_terms(field_name, None))
    excluded_hits = [
        term
        for term in _excluded_terms_for_plan(plan)
        if term not in field_terms and term in lowered
    ]
    if not excluded_hits:
        return False
    allowed_hits = [
        term for term in _field_terms(field_name, None) if len(term) >= 3 and term in lowered
    ]
    return len(excluded_hits) >= max(1, len(allowed_hits))


def _row_or_section_label(span: str) -> str:
    match = re.search(r"\[Table\s+\d+\s+Row\s+\d+\]", span)
    if match:
        return match.group(0).strip("[]")
    return "sentence_or_paragraph"


def _row_program_confidence(span: str, plan: QueryPlan) -> float:
    program_tokens = _target_program_tokens(plan)
    if not program_tokens:
        return 0.0
    row_tokens = set(re.findall(r"[a-z0-9äöüß]{3,}", span.lower()))
    hits = len(program_tokens.intersection(row_tokens))
    return hits / max(1, len(program_tokens))


def _select_relevant_span(chunk: str, field_name: str, plan: QueryPlan) -> tuple[str, str, str]:
    sentences = _split_evidence_sentences(chunk)
    if not sentences:
        return "", "no_sentences", ""
    table_rows = [
        sentence for sentence in sentences if re.search(r"\[Table\s+\d+\s+Row\s+\d+\]", sentence)
    ]
    if table_rows and plan.program:
        matching_rows = [
            row
            for row in table_rows
            if _row_program_confidence(row, plan) >= 0.55
            and _chunk_matches_field(row, field_name, plan)
        ]
        for row in matching_rows:
            if not _sentence_has_excluded_field(row, plan, field_name):
                return row, "selected_target_program_table_row", _row_or_section_label(row)
        if matching_rows:
            return "", "target_program_row_contained_excluded_fields", ""
        if any(_chunk_matches_field(row, field_name, plan) for row in table_rows):
            return "", "neighboring_program_rows_ignored", ""
    selected: list[str] = []
    rejected_excluded = 0
    for sentence in sentences:
        if not _chunk_matches_field(sentence, field_name, plan):
            continue
        if _sentence_has_excluded_field(sentence, plan, field_name):
            rejected_excluded += 1
            continue
        selected.append(sentence)
        if len(selected) >= 3:
            break
    if selected:
        span = " ".join(selected)
        return span, "selected_field_matching_sentences", _row_or_section_label(span)

    # Last resort: split mixed sentences into smaller clauses and keep only the useful clause.
    clauses: list[str] = []
    for sentence in sentences:
        clauses.extend(re.split(r",\s+and\s+|,\s+|\s+and\s+|\s*/\s*", sentence))
    for clause in clauses:
        clause = _compact(clause)
        if not clause:
            continue
        if _chunk_matches_field(clause, field_name, plan) and not _sentence_has_excluded_field(
            clause, plan, field_name
        ):
            selected.append(clause)
            if len(selected) >= 3:
                break
    if selected:
        span = " ".join(selected)
        return span, "selected_field_matching_clauses", _row_or_section_label(span)
    if rejected_excluded:
        return "", "field_sentences_contained_excluded_fields", ""
    return "", "no_field_specific_span", ""


def _is_language_section(section: str) -> bool:
    lowered = section.replace("_", " ").lower()
    return "language" in lowered or "ielts" in lowered or "english" in lowered or "spra" in lowered


def _chunk_matches_requested_section(chunk: str, section: str, plan: QueryPlan) -> bool:
    lowered = chunk.lower()
    if _is_language_section(section):
        if not any(term in lowered for term in LANGUAGE_TERMS):
            return False
        language_hits = sum(1 for term in LANGUAGE_TERMS if term in lowered)
        unrelated_hits = sum(1 for term in LANGUAGE_UNRELATED_TERMS if term in lowered)
        if unrelated_hits > language_hits + 1:
            return False
    return _keyword_match(chunk, _section_keywords(section, plan)) >= 0.18


def _chunk_passes_intent_gate(chunk: str, field_name: str, plan: QueryPlan) -> bool:
    gate = INTENT_CHUNK_GATES.get(plan.intent)
    if not gate:
        return True
    lowered = chunk.lower()
    allow = tuple(
        dict.fromkeys(
            [
                *gate.get("allow", ()),
                *(term.lower() for term in INTENT_EXTRACT_TERMS.get(plan.intent, [])),
            ]
        )
    )
    exclude = gate.get("exclude", ())
    if allow and not any(term in lowered for term in allow):
        return False
    excluded_hit_count = sum(1 for term in exclude if term in lowered)
    allowed_hit_count = sum(1 for term in allow if term in lowered)
    if excluded_hit_count and excluded_hit_count >= allowed_hit_count + 1:
        return False
    if plan.intent == "application_portal_lookup" and field_name == "application_deadline":
        return "deadline" in lowered and any(
            term in lowered for term in ("portal", "apply", "application")
        )
    return True


def _chunk_matches_field(chunk: str, field_name: str, plan: QueryPlan) -> bool:
    if not _chunk_passes_intent_gate(chunk, field_name, plan):
        return False
    lowered = chunk.lower()
    if field_name == "english_language_requirement" and _direct_english_requirement_span(chunk):
        return True
    if _is_language_section(field_name):
        if not any(term in lowered for term in LANGUAGE_TERMS):
            return False
        language_hits = sum(1 for term in LANGUAGE_TERMS if term in lowered)
        unrelated_hits = sum(1 for term in LANGUAGE_UNRELATED_TERMS if term in lowered)
        if unrelated_hits > language_hits + 1:
            return False
    return _keyword_match(chunk, _field_terms(field_name, None)) >= 0.18


def _direct_english_requirement_span(text: str) -> bool:
    lowered = text.lower()
    has_english = any(
        term in lowered
        for term in (
            "english language proficiency",
            "english proficiency",
            "english certificate",
            "english language certificate",
            "englischkenntnisse",
            "sprachnachweis",
        )
    )
    has_cefr = "cefr" in lowered or "common european framework" in lowered
    has_level = bool(re.search(r"\b(b2|c1|c2)\b", lowered))
    has_master = bool(re.search(r"\b(master|m\.sc|msc|master's degree)\b", lowered))
    return has_english and has_level and (has_cefr or has_master)


def _source_matches_university(source: ExtractedContent, plan: QueryPlan) -> bool:
    if not plan.university and not plan.university_short:
        return True
    expected = _detect_university_from_text(" ".join([plan.university, plan.university_short]))
    if not expected:
        return True
    expected_domains = set(expected.get("domains", []))
    if source.domain in expected_domains or any(
        source.domain.endswith("." + domain) for domain in expected_domains
    ):
        return True
    source_university = _detect_university_from_text(
        " ".join([source.domain, source.url, source.title])
    )
    if source_university and str(source_university["name"]) != str(expected["name"]):
        return False
    if source.source_type in {"daad", "uni_assist", "government_or_eu"}:
        return True
    return not expected_domains or source.source_quality < 0.75


def _target_program_tokens(plan: QueryPlan) -> set[str]:
    base_tokens = {
        token
        for token in re.findall(r"[a-z0-9äöüß]{3,}", plan.program.lower())
        if token not in {"msc", "master", "science", "degree", "program", "programme"}
    }
    words = [
        token
        for token in re.findall(r"[a-zäöüß]+", plan.program.lower())
        if token not in {"msc", "master", "science", "degree", "program", "programme"}
    ]
    if len(words) >= 2:
        base_tokens.add("".join(word[0] for word in words))
    if "informatics" in base_tokens:
        base_tokens.add("informatik")
    if "informatik" in base_tokens:
        base_tokens.add("informatics")
    return base_tokens


def _mentions_wrong_program(text: str, plan: QueryPlan) -> bool:
    if not plan.program:
        return False
    lowered = text.lower()
    target = plan.program.lower()
    if "informatics" in target:
        lowered = lowered.replace("informatik", "informatics")
    for marker in KNOWN_PROGRAM_MARKERS:
        if marker in lowered and marker not in target and target not in marker:
            return True
    return False


def _evidence_scope(source: ExtractedContent, chunk: str, plan: QueryPlan) -> str:
    combined = f"{source.url} {source.title} {chunk}".lower()
    program_tokens = _target_program_tokens(plan)
    if program_tokens and program_tokens.intersection(
        set(re.findall(r"[a-z0-9äöüß]{3,}", combined))
    ):
        return "program_specific"
    if source.source_type == "daad":
        return "DAAD"
    if source.source_type in {"third_party_education_site", "blog_or_forum", "other"}:
        return "third_party"
    if any(term in combined for term in ("faculty", "school of", "department", "department of")):
        return "faculty_general"
    if source.source_quality >= 0.75:
        return "university_general"
    return "unrelated"


def _is_general_policy_field(field_name: str) -> bool:
    return field_name in {
        "tuition_fee",
        "semester_contribution",
        "application_process",
        "application_deadline",
        "intake_or_semester",
        "applicant_category",
        "uni_assist_requirement",
        "vpd_requirement",
        "aps_requirement",
        "english_language_requirement",
    }


def _deadline_chunk_has_valid_date(chunk: str) -> bool:
    lowered = chunk.lower()
    if re.search(r"\bwise\s*\d{4}/\d{2}\b", lowered):
        return False
    if any(term in lowered for term in ("module handbook", "curriculum", "examination regulation")):
        return False
    if re.search(r"\b20(?:0[0-9]|1[0-9])\b", lowered):
        return False
    has_deadline_term = any(
        term in lowered
        for term in (
            "deadline",
            "application period",
            "bewerbungsfrist",
            "bewerbungszeitraum",
            "apply by",
        )
    )
    has_date = bool(
        re.search(
            r"\b\d{1,2}\.?\s*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b|\b\d{1,2}\.\d{1,2}\.",
            lowered,
        )
    )
    return has_deadline_term and (
        has_date or "winter semester" in lowered or "summer semester" in lowered
    )


def _evidence_rejection_reason(
    source: ExtractedContent, chunk: str, field_name: str, plan: QueryPlan
) -> str:
    combined = f"{source.url} {source.title} {chunk}".lower()
    if not _source_matches_university(source, plan):
        return "wrong_university"
    degree_path_reason = _degree_level_path_rejection_reason(source.url, plan)
    if degree_path_reason:
        span_signal = _degree_signal(chunk)
        if span_signal != plan.degree_level:
            return degree_path_reason
    span_degree_reason = _span_degree_rejection_reason(chunk, plan)
    if span_degree_reason:
        return span_degree_reason
    scope = _evidence_scope(source, chunk, plan)
    if scope == "unrelated":
        return "unrelated_source_scope"
    if _mentions_wrong_program(combined, plan):
        return "wrong_program"
    if (
        plan.program
        and scope in {"faculty_general", "university_general", "third_party"}
        and not _is_general_policy_field(field_name)
        and source.source_type != "daad"
    ):
        return "not_program_specific"
    if any(term in combined for term in URL_PATH_PENALTY_TERMS):
        return "irrelevant_page_type"
    if "module handbook" in combined and plan.intent != "curriculum_lookup":
        return "module_handbook_not_requested"
    if field_name in {
        "application_deadline",
        "intake_or_semester",
    } and not _deadline_chunk_has_valid_date(chunk):
        return "deadline_date_invalid_or_irrelevant"
    return ""


def _field_support_level(text: str, field_name: str, score: float) -> str:
    lowered = text.lower()
    if field_name == "english_language_requirement" and _direct_english_requirement_span(text):
        return "direct"
    direct_patterns = {
        "english_language_requirement": r"\b(english|englisch|cefr|language|sprachnachweis)\b.{0,120}\b(b2|c1|ielts|toefl|duolingo)\b|\b(b2|c1)\b.{0,120}\b(english|englisch|cefr|language|sprachnachweis)\b",
        "german_language_requirement": r"\b(german|deutsch|testdaf|dsh)\b.{0,120}\b(b1|b2|c1|c2|testdaf|dsh)\b",
        "application_deadline": r"\b\d{1,2}\.?\s*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)|\b\d{1,2}\.\d{1,2}\.",
        "ielts_score": r"\bielts\b.{0,90}\b[4-9](?:\.\d)?\b",
        "toefl_score": r"\btoefl\b.{0,90}\b\d{2,3}\b",
        "duolingo_score": r"\bduolingo\b.{0,90}\b\d{2,3}\b",
        "gpa_requirement": r"\b(gpa|grade|note)\b.{0,90}\b\d(?:\.\d+)?\b",
        "tuition_fee": r"\b(tuition|fee|studiengeb).{0,90}(€|eur|euro|usd|\d)",
        "semester_contribution": r"\b(semester contribution|semesterbeitrag).{0,90}(€|eur|euro|\d)",
        "program_duration": r"\b(duration|semesters|regelstudienzeit).{0,90}\b\d+\b",
    }
    if re.search(direct_patterns.get(field_name, r"$^"), lowered, re.I):
        return "direct"
    if field_name in VALUE_REQUIRED_FIELDS:
        return "indirect" if score >= 0.22 else "weak"
    if score >= 0.42:
        return "direct"
    if score >= 0.22:
        return "indirect"
    return "weak"


def _map_chunk_to_fields(chunk: str, plan: QueryPlan) -> list[dict[str, Any]]:
    candidate_fields = list(dict.fromkeys([*plan.required_fields, *plan.optional_fields]))
    excluded = set(plan.excluded_fields)
    mappings: list[dict[str, Any]] = []
    for field_name in candidate_fields:
        if field_name in excluded:
            continue
        if not _chunk_matches_field(chunk, field_name, plan):
            continue
        keyword_score = _keyword_match(chunk, _field_terms(field_name, None))
        mappings.append(
            {
                "field": field_name,
                "keyword_match": keyword_score,
                "support_level": _field_support_level(chunk, field_name, keyword_score),
                "reason": f"matched keywords for {field_name}",
            }
        )
    mappings.sort(key=lambda item: item["keyword_match"], reverse=True)
    return mappings[:2]


def _excluded_chunk_reason(chunk: str, plan: QueryPlan) -> str:
    lowered = chunk.lower()
    gate = INTENT_CHUNK_GATES.get(plan.intent)
    if gate:
        allow = tuple(
            dict.fromkeys(
                [
                    *gate.get("allow", ()),
                    *(term.lower() for term in INTENT_EXTRACT_TERMS.get(plan.intent, [])),
                ]
            )
        )
        exclude = gate.get("exclude", ())
        if allow and not any(term in lowered for term in allow):
            return "intent_gate_no_allowed_terms"
        excluded = [term for term in exclude if term in lowered]
        allowed_hits = sum(1 for term in allow if term in lowered)
        if len(excluded) >= allowed_hits + 1:
            return "intent_gate_excluded_terms:" + ",".join(excluded[:3])
    matched_excluded = [
        field_name
        for field_name in plan.excluded_fields
        if _keyword_match(lowered, _field_terms(field_name, None)) >= 0.18
    ]
    if matched_excluded:
        return "matched_excluded_field:" + ",".join(matched_excluded[:3])
    return "no_required_or_optional_field_match"


def _official_rows_or_all(rows: list[EvidenceChunk]) -> list[EvidenceChunk]:
    priority = {
        "program_specific": 5,
        "faculty_general": 4,
        "university_general": 3,
        "DAAD": 2,
        "third_party": 1,
        "unrelated": 0,
    }
    official = [row for row in rows if float(row.scoring.get("source_quality", 0.0) or 0.0) >= 0.75]
    scoped_rows = official or rows
    scoped_rows.sort(
        key=lambda row: (priority.get(row.evidence_scope, 0), row.score),
        reverse=True,
    )
    return scoped_rows


def _query_relevance(query: str, plan: QueryPlan) -> float:
    for item in plan.search_queries:
        if str(item.get("query")) == query:
            return max(0.0, min(1.0, float(item.get("priority") or 0.5)))
    return 0.5


def group_and_rank_evidence(
    extracted: list[ExtractedContent],
    plan: QueryPlan,
    *,
    debug_collector: dict[str, Any] | None = None,
) -> dict[str, list[EvidenceChunk]]:
    if not plan.required_fields:
        plan.required_fields = plan.required_info or ["general_information"]
    sections = list(dict.fromkeys([*plan.required_fields, *plan.optional_fields]))
    if not sections:
        sections = ["general_information"]
    grouped: dict[str, list[EvidenceChunk]] = {section: [] for section in sections}
    total_chunks = 0
    excluded_chunks: list[dict[str, Any]] = []
    selected_answer_spans: list[dict[str, Any]] = []
    field_mapping_results: list[dict[str, Any]] = []
    for source in extracted:
        for page in source.pages:
            for chunk in chunk_text(page.text):
                total_chunks += 1
                mappings = _map_chunk_to_fields(chunk, plan)
                if not mappings:
                    excluded_chunks.append(
                        {
                            "url": source.url,
                            "title": source.title,
                            "domain": source.domain,
                            "page_number": page.page_number,
                            "support_level": "reject",
                            "allowed_for_current_question": False,
                            "reason": _excluded_chunk_reason(chunk, plan),
                            "text": chunk[:700],
                        }
                    )
                    continue
                for mapping in mappings:
                    section = str(mapping["field"])
                    field_mapping_results.append(
                        {
                            "url": source.url,
                            "title": source.title,
                            "page_number": page.page_number,
                            "field": section,
                            "support_level": mapping.get("support_level", "weak"),
                            "keyword_match": round(float(mapping.get("keyword_match", 0.0)), 4),
                            "reason": mapping.get("reason", ""),
                            "chunk_preview": chunk[:500],
                        }
                    )
                    span, span_reason, row_or_section = _select_relevant_span(chunk, section, plan)
                    if not span:
                        excluded_chunks.append(
                            {
                                "url": source.url,
                                "title": source.title,
                                "domain": source.domain,
                                "page_number": page.page_number,
                                "field": section,
                                "evidence_scope": _evidence_scope(source, chunk, plan),
                                "support_level": "reject",
                                "allowed_for_current_question": False,
                                "row_or_section": row_or_section,
                                "reason": span_reason,
                                "text": chunk[:700],
                            }
                        )
                        continue
                    rejection_reason = _evidence_rejection_reason(source, span, section, plan)
                    if rejection_reason:
                        excluded_chunks.append(
                            {
                                "url": source.url,
                                "title": source.title,
                                "domain": source.domain,
                                "page_number": page.page_number,
                                "field": section,
                                "evidence_scope": _evidence_scope(source, chunk, plan),
                                "support_level": "reject",
                                "allowed_for_current_question": False,
                                "row_or_section": row_or_section,
                                "reason": rejection_reason,
                                "text": chunk[:700],
                            }
                        )
                        continue
                    keyword_match = _keyword_match(span, _field_terms(section, None))
                    source_quality = source.source_quality
                    query_relevance = _query_relevance(source.query, plan)
                    scope = _evidence_scope(source, span, plan)
                    scope_boost = {
                        "program_specific": 0.12,
                        "faculty_general": 0.06,
                        "university_general": 0.03,
                        "DAAD": 0.02,
                        "third_party": -0.08,
                    }.get(scope, -0.20)
                    final_score = (
                        (0.50 * keyword_match)
                        + (0.30 * source_quality)
                        + (0.20 * query_relevance)
                        + scope_boost
                    )
                    row_confidence = _row_program_confidence(span, plan)
                    support_level = _field_support_level(span, section, keyword_match)
                    selected_answer_spans.append(
                        {
                            "url": source.url,
                            "title": source.title,
                            "page_number": page.page_number,
                            "field": section,
                            "support_level": support_level,
                            "evidence_scope": scope,
                            "row_or_section": row_or_section,
                            "span_reason": span_reason,
                            "selected_span_degree_signal": _degree_signal(span),
                            "selected_span_degree_match": _degree_signal_matches_target(
                                _degree_signal(span), plan
                            ),
                            "text": span[:900],
                        }
                    )
                    grouped[section].append(
                        EvidenceChunk(
                            text=span,
                            url=source.url,
                            title=source.title,
                            domain=source.domain,
                            source_type=source.source_type,
                            document_type=source.document_type,
                            page_number=page.page_number,
                            retrieved_at=source.retrieved_at,
                            query=source.query,
                            section=section,
                            score=final_score,
                            scoring={
                                "keyword_match": keyword_match,
                                "source_quality": source_quality,
                                "query_relevance": query_relevance,
                                "allowed_for_current_question": 1.0,
                                "span_selected_from_chunk": 1.0,
                                "selected_span_degree_signal": _degree_signal(span),
                                "selected_span_degree_match": _degree_signal_matches_target(
                                    _degree_signal(span), plan
                                ),
                                "target_program_row_found": (
                                    1.0
                                    if span_reason == "selected_target_program_table_row"
                                    else 0.0
                                ),
                                "neighboring_rows_ignored": (
                                    1.0
                                    if span_reason == "selected_target_program_table_row"
                                    else 0.0
                                ),
                                "row_confidence": row_confidence,
                            },
                            field=section,
                            support_level=support_level,
                            selection_reason=f"{mapping['reason']}; {span_reason}",
                            evidence_scope=scope,
                            row_or_section=row_or_section,
                        )
                    )
    for section, rows in list(grouped.items()):
        rows.sort(key=lambda item: item.score, reverse=True)
        grouped[section] = _official_rows_or_all(rows)
    logger.info(
        "UniGraph chunks created and grouped | chunks=%s | grouped=%s",
        total_chunks,
        {section: len(rows) for section, rows in grouped.items()},
    )
    if debug_collector is not None:
        debug_collector["excluded_evidence_chunks"] = excluded_chunks[:80]
        debug_collector["selected_answer_spans"] = selected_answer_spans[:80]
        debug_collector["extracted_field_spans"] = selected_answer_spans[:80]
        debug_collector["field_mapping_results"] = field_mapping_results[:120]
    return grouped


def fan_in_evidence(grouped: dict[str, list[EvidenceChunk]]) -> list[EvidenceChunk]:
    selected: list[EvidenceChunk] = []
    seen: set[tuple[str, str]] = set()
    per_section = max(1, MAX_EVIDENCE_CHUNKS // max(1, len(grouped)))
    for section, rows in grouped.items():
        usable_rows = [row for row in rows if row.support_level in {"direct", "indirect"}]
        if not usable_rows:
            usable_rows = rows[:1]
        used = 0
        for chunk in usable_rows:
            key = (chunk.url, chunk.text[:180].lower())
            if key in seen:
                continue
            seen.add(key)
            selected.append(chunk)
            used += 1
            if used >= per_section:
                break
    if len(selected) < MAX_EVIDENCE_CHUNKS:
        leftovers = [
            chunk
            for rows in grouped.values()
            for chunk in rows
            if chunk.support_level in {"direct", "indirect"}
        ]
        if not leftovers:
            leftovers = [chunk for rows in grouped.values() for chunk in rows[:1]]
        leftovers.sort(key=lambda item: item.score, reverse=True)
        for chunk in leftovers:
            key = (chunk.url, chunk.text[:180].lower())
            if key in seen:
                continue
            selected.append(chunk)
            seen.add(key)
            if len(selected) >= MAX_EVIDENCE_CHUNKS:
                break
    selected.sort(key=lambda item: item.score, reverse=True)
    logger.info("UniGraph selected evidence chunks | selected=%s", [c.__dict__ for c in selected])
    return selected[:MAX_EVIDENCE_CHUNKS]


SOURCE_SCOPE_PRIORITY = {
    "program_specific_direct": 7,
    "program_specific_indirect": 6,
    "admissions_general": 5,
    "faculty_general": 4,
    "university_general": 3,
    "DAAD_exact_program": 2,
    "DAAD": 3,
    "third_party": 1,
    "unrelated": 0,
}


def _source_scope_key(chunk: EvidenceChunk) -> str:
    if chunk.evidence_scope == "program_specific":
        return (
            "program_specific_direct"
            if chunk.support_level == "direct"
            else "program_specific_indirect"
        )
    if chunk.evidence_scope == "DAAD":
        return "DAAD_exact_program" if _target_program_tokens_from_text(chunk.text) else "DAAD"
    lowered = f"{chunk.url} {chunk.title} {chunk.text}".lower()
    if chunk.evidence_scope in {"faculty_general", "university_general"} and any(
        term in lowered
        for term in (
            "admission",
            "application",
            "apply",
            "bewerbung",
            "zulassung",
            "required documents",
            "application period",
            "deadline",
        )
    ):
        return "admissions_general"
    return chunk.evidence_scope or "unrelated"


def _target_program_tokens_from_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9äöüß]{3,}", str(text or "").lower())
        if token not in {"msc", "master", "science", "degree", "program", "programme"}
    }


def _field_priority_chunk(rows: list[EvidenceChunk]) -> EvidenceChunk | None:
    usable = [row for row in rows if row.support_level in {"direct", "indirect"}]
    if not usable:
        return None
    usable.sort(
        key=lambda row: (
            SOURCE_SCOPE_PRIORITY.get(_source_scope_key(row), 0),
            1 if row.support_level == "direct" else 0,
            float(row.scoring.get("source_quality", 0.0) or 0.0),
            row.score,
        ),
        reverse=True,
    )
    return usable[0]


def _clean_evidence_value(text: str) -> str:
    value = _compact(text)
    value = re.sub(r"#{1,6}\s*", "", value)
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -:;")
    return value


def _normalize_date_token(value: str) -> str:
    value = _compact(value).strip(" .")
    month_map = {
        "1": "January",
        "01": "January",
        "2": "February",
        "02": "February",
        "3": "March",
        "03": "March",
        "4": "April",
        "04": "April",
        "5": "May",
        "05": "May",
        "6": "June",
        "06": "June",
        "7": "July",
        "07": "July",
        "8": "August",
        "08": "August",
        "9": "September",
        "09": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }
    match = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.?", value)
    if match:
        day = str(int(match.group(1))).zfill(2)
        return f"{day} {month_map.get(match.group(2), match.group(2))}"
    match = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)", value)
    if match:
        return f"{str(int(match.group(1))).zfill(2)} {match.group(2)}"
    return value


def _normalize_period_value(value: str) -> str:
    parts = re.split(r"\s*[-–]\s*", _compact(value), maxsplit=1)
    if len(parts) == 2:
        return f"{_normalize_date_token(parts[0])} - {_normalize_date_token(parts[1])}"
    return _normalize_date_token(value)


def _extract_deadline_values(text: str) -> dict[str, str]:
    cleaned = _clean_evidence_value(text)
    values: dict[str, str] = {}
    for semester, key in (
        ("winter semester", "winter_semester_application_period"),
        ("summer semester", "summer_semester_application_period"),
    ):
        period = _extract_semester_period(cleaned, semester)
        if period:
            values[key] = _normalize_period_value(period)
    visa_match = re.search(
        r"february\s+1\s*[-–]\s*march\s+31.{0,120}?(?:visa|recommended)|(?:visa|recommended).{0,120}?february\s+1\s*[-–]\s*march\s+31",
        cleaned,
        flags=re.I,
    )
    if visa_match:
        values["visa_recommended_deadline"] = "31 March"
    compulsory_match = re.search(
        r"february\s+1\s*[-–]\s*may\s+31.{0,120}?(?:compulsory|required|final)|(?:compulsory|required|final).{0,120}?february\s+1\s*[-–]\s*may\s+31",
        cleaned,
        flags=re.I,
    )
    if compulsory_match:
        values["compulsory_deadline"] = "31 May"
    return values


def _extract_document_items(text: str) -> list[str]:
    cleaned = _clean_evidence_value(text)
    candidates = [
        "online application",
        "STiNE",
        "transcript",
        "degree certificate",
        "CV",
        "motivation letter",
        "language certificate",
        "language proof",
        "APS",
        "VPD",
    ]
    items: list[str] = []
    for item in candidates:
        if re.search(rf"\b{re.escape(item)}\b", cleaned, re.I):
            items.append(item)
    return list(dict.fromkeys(items))


def _normalize_evidence_value(field_name: str, chunk: EvidenceChunk) -> dict[str, Any]:
    cleaned = _clean_evidence_value(chunk.text)
    normalized: dict[str, Any] = {"value": cleaned[:420]}
    if field_name in {"application_deadline", "other_semester_deadline", "intake_or_semester"}:
        deadline_values = _extract_deadline_values(chunk.text)
        if deadline_values:
            normalized["normalized_values"] = deadline_values
            wants_summer = field_name == "other_semester_deadline"
            key = (
                "summer_semester_application_period"
                if wants_summer
                else "winter_semester_application_period"
            )
            normalized["value"] = deadline_values.get(key) or next(iter(deadline_values.values()))
    elif field_name in {
        "required_application_documents",
        "international_applicant_documents",
        "degree_transcript_requirements",
        "language_proof",
    }:
        items = _extract_document_items(chunk.text)
        checklist_items = [
            item for item in items if item.lower() not in {"online application", "stine"}
        ]
        if checklist_items:
            normalized["checklist_items"] = items
            normalized["value"] = "; ".join(items)
        elif re.search(r"\b(upload|online application|application is online|portal)\b", cleaned, re.I):
            normalized["confirmed_process"] = "application is online and documents must be uploaded"
            normalized["value"] = normalized["confirmed_process"]
    elif field_name in {"tuition_fee", "semester_contribution"}:
        match = re.search(
            r"(?:tuition fee|semester fee|semester contribution|studiengebühren|semesterbeitrag).{0,120}?(?:€|eur|euro)?\s?\d[\d.,]*",
            cleaned,
            flags=re.I,
        )
        if match:
            normalized["value"] = _compact(match.group(0))
    return normalized


def _evidence_packet_entry(chunk: EvidenceChunk) -> dict[str, Any]:
    field_name = chunk.field or chunk.section
    normalized = _normalize_evidence_value(field_name, chunk)
    return {
        "value": normalized.get("value", _clean_evidence_value(chunk.text)[:420]),
        "normalized_values": normalized.get("normalized_values", {}),
        "checklist_items": normalized.get("checklist_items", []),
        "confirmed_process": normalized.get("confirmed_process", ""),
        "support_level": chunk.support_level,
        "scope": _source_scope_key(chunk),
        "source_url": chunk.url,
        "source_type": chunk.source_type,
        "document_type": chunk.document_type,
        "page_number": chunk.page_number,
        "evidence_span": _clean_evidence_value(chunk.text)[:900],
        "row_or_section": chunk.row_or_section,
        "reason": chunk.selection_reason,
        "score": round(float(chunk.score), 4),
    }


def _evidence_value_signature(text: str) -> str:
    text = _compact(text).lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    numbers = " ".join(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))
    levels = " ".join(re.findall(r"\b(?:a1|a2|b1|b2|c1|c2|ielts|toefl|duolingo)\b", text))
    dates = " ".join(
        re.findall(
            r"\b\d{1,2}\.?\s*(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b|\b\d{1,2}\.\d{1,2}\.?\b",
            text,
        )
    )
    signature = _compact(" ".join(part for part in (levels, numbers, dates) if part))
    return signature or text[:180]


def _field_conflict(rows: list[EvidenceChunk]) -> dict[str, Any] | None:
    direct_rows = [
        row
        for row in rows
        if row.support_level == "direct"
        and row.source_type
        in {
            "official_university_page",
            "official_university_pdf",
            "daad",
            "uni_assist",
            "government_or_eu",
        }
    ][:4]
    if len(direct_rows) < 2:
        return None
    signatures = {_evidence_value_signature(row.text) for row in direct_rows}
    if len(signatures) <= 1:
        return None
    return {
        "status": "conflicting",
        "reason": "trusted_sources_disagree_for_requested_field",
        "sources": [_evidence_packet_entry(row) for row in direct_rows[:3]],
    }


def build_evidence_packet(
    grouped: dict[str, list[EvidenceChunk]],
    plan: QueryPlan,
) -> tuple[dict[str, Any], list[EvidenceChunk]]:
    required_fields = list(dict.fromkeys(plan.required_fields or plan.required_info or []))
    optional_fields = list(dict.fromkeys(plan.optional_fields or []))
    excluded_fields = list(dict.fromkeys(plan.excluded_fields or []))
    answered_fields: dict[str, dict[str, Any]] = {}
    deduplicated_fields: dict[str, list[str]] = {}
    missing_fields: dict[str, str] = {}
    conflicting_fields: dict[str, dict[str, Any]] = {}
    field_statuses: dict[str, str] = {field_name: "excluded" for field_name in excluded_fields}
    selected_chunks: list[EvidenceChunk] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def _add_selected(field_name: str, chunk: EvidenceChunk) -> None:
        key = (field_name, chunk.url, _evidence_value_signature(chunk.text))
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected_chunks.append(chunk)

    for field_name in required_fields:
        conflict = _field_conflict(grouped.get(field_name, []))
        if conflict:
            conflicting_fields[field_name] = conflict
            field_statuses[field_name] = "conflicting"
        best = _field_priority_chunk(grouped.get(field_name, []))
        if best is None:
            missing_fields[field_name] = _missing_reason_for_field(field_name, grouped)
            field_statuses.setdefault(field_name, "missing")
            continue
        signature = (best.url, _evidence_value_signature(best.text))
        existing_field = next(
            (
                existing
                for existing, entry in answered_fields.items()
                if (entry.get("source_url"), str(entry.get("value_signature", ""))) == signature
            ),
            "",
        )
        if existing_field and plan.intent == "deadline_lookup":
            field_statuses[field_name] = f"deduplicated_to:{existing_field}"
            deduplicated_fields.setdefault(existing_field, []).append(field_name)
            continue
        answered_fields[field_name] = _evidence_packet_entry(best)
        answered_fields[field_name]["value_signature"] = signature[1]
        if not conflict:
            field_statuses[field_name] = (
                "answered_direct" if best.support_level == "direct" else "answered_indirect"
            )
        _add_selected(field_name, best)

    for field_name in optional_fields:
        if field_name in excluded_fields or field_name in answered_fields:
            continue
        best = _field_priority_chunk(grouped.get(field_name, []))
        if best is None:
            continue
        signature = (best.url, _evidence_value_signature(best.text))
        existing_field = next(
            (
                existing
                for existing, entry in answered_fields.items()
                if (entry.get("source_url"), str(entry.get("value_signature", ""))) == signature
            ),
            "",
        )
        if existing_field:
            field_statuses[field_name] = f"deduplicated_to:{existing_field}"
            deduplicated_fields.setdefault(existing_field, []).append(field_name)
            continue
        answered_fields[field_name] = _evidence_packet_entry(best)
        answered_fields[field_name]["value_signature"] = signature[1]
        field_statuses[field_name] = (
            "answered_direct" if best.support_level == "direct" else "answered_indirect"
        )
        _add_selected(field_name, best)

    source_set: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for chunk in selected_chunks:
        if chunk.url in seen_urls:
            continue
        seen_urls.add(chunk.url)
        source_set.append(
            {
                "url": chunk.url,
                "title": chunk.title,
                "source_type": chunk.source_type,
                "document_type": chunk.document_type,
                "scope": _source_scope_key(chunk),
                "fields_supported": [
                    field_name
                    for field_name, entry in answered_fields.items()
                    if entry.get("source_url") == chunk.url
                ],
            }
        )

    packet = {
        "intent": plan.intent,
        "query_mode": plan.query_mode,
        "university": plan.university,
        "program": plan.program,
        "degree_level": plan.degree_level,
        "required_fields": {field_name: field_statuses.get(field_name, "missing") for field_name in required_fields},
        "optional_fields": {
            field_name: field_statuses.get(field_name, "not_used") for field_name in optional_fields
        },
        "answered_fields": answered_fields,
        "deduplicated_fields": deduplicated_fields,
        "missing_fields": missing_fields,
        "excluded_fields": excluded_fields,
        "conflicting_fields": conflicting_fields,
        "field_completeness_status": (
            "complete"
            if required_fields
            and all(
                field_statuses.get(field_name, "").startswith(("answered", "deduplicated_to:"))
                for field_name in required_fields
            )
            else "partial"
            if answered_fields
            else "missing"
        ),
        "source_set_selected": source_set,
    }
    packet["normalized_evidence_packet"] = {
        "answered_fields": answered_fields,
        "missing_fields": {
            field: reason
            for field, reason in missing_fields.items()
            if field in required_fields
        },
        "deduplicated_fields": deduplicated_fields,
        "excluded_fields": excluded_fields,
    }
    return packet, selected_chunks[:MAX_EVIDENCE_CHUNKS]


def _evidence_context(chunks: list[EvidenceChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        page = f", page {chunk.page_number}" if chunk.page_number else ""
        lines.append(
            f"[E{index}] field={chunk.field or chunk.section}; support_level={chunk.support_level}; "
            f"evidence_scope={chunk.evidence_scope}; "
            f"source_type={chunk.source_type}; "
            f"document_type={chunk.document_type}{page}; title={chunk.title or chunk.url}; "
            f"row_or_section={chunk.row_or_section}; "
            f"url={chunk.url}; retrieved_at={chunk.retrieved_at}"
        )
        lines.append(chunk.text[:1200])
        lines.append("")
    return "\n".join(lines)


def _extract_semester_period(text: str, semester: str) -> str:
    escaped = re.escape(semester)
    patterns = [
        rf"{escaped}\s*(?:application\s*period|deadline)?\s*[:\-–]\s*(\d{{1,2}}\s+[A-Za-z]+\s*[-–]\s*\d{{1,2}}\s+[A-Za-z]+)",
        rf"{escaped}.{0,120}?(\d{{1,2}}\s+[A-Za-z]+\s*[-–]\s*\d{{1,2}}\s+[A-Za-z]+)",
        rf"{escaped}.{0,120}?(\d{{1,2}}\.\d{{1,2}}\.?\s*[-–]\s*\d{{1,2}}\.\d{{1,2}}\.?)",
        rf"application\s+period\s+for\s+the\s+{escaped}\s+is\s+(\d{{1,2}}\s+[A-Za-z]+\s*[-–]\s*\d{{1,2}}\s+[A-Za-z]+)",
        rf"application\s+period\s+for\s+the\s+{escaped}\s+is\s+(\d{{1,2}}\.\d{{1,2}}\.?\s*[-–]\s*\d{{1,2}}\.\d{{1,2}}\.?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def _deadline_answer_from_evidence(query: str, evidence_chunks: list[EvidenceChunk]) -> str:
    if not evidence_chunks:
        return ""
    joined = " ".join(chunk.text for chunk in evidence_chunks[:5])
    wants_winter = "winter" in query.lower()
    wants_summer = "summer" in query.lower()
    requested_semester = (
        "winter semester" if wants_winter or not wants_summer else "summer semester"
    )
    requested_period = _extract_semester_period(joined, requested_semester)
    other_semester = (
        "summer semester" if requested_semester == "winter semester" else "winter semester"
    )
    other_period = _extract_semester_period(joined, other_semester)
    source = evidence_chunks[0].url
    program_label = "the requested program"
    if "informatics" in query.lower() and ("tum" in query.lower() or "munich" in query.lower()):
        program_label = "MSc Informatics at TUM"
    if not requested_period:
        return ""
    first = (
        f"For {program_label}, the official program page lists the {requested_semester} "
        f"application period as {requested_period} [E1]."
    )
    if other_period:
        first += f" The same page lists the {other_semester} period as {other_period} [E1]."
    return f"{first}\n\nSource: {source}"


def _program_label_from_plan(plan: QueryPlan) -> str:
    program = plan.program.strip()
    university = (plan.university_short or plan.university).strip()
    if program and university:
        return f"{program} at {university}"
    return program or university or "the requested program"


def _citation_for_chunk(chunk: EvidenceChunk, index: int) -> str:
    return f"[E{index}]"


def _language_answer_from_evidence(
    evidence_chunks: list[EvidenceChunk],
    plan: QueryPlan,
    field_missing_reasons: dict[str, str] | None = None,
) -> str:
    language_chunks = [
        chunk
        for chunk in evidence_chunks
        if (chunk.field or chunk.section) == "english_language_requirement"
        and chunk.support_level in {"direct", "indirect"}
    ]
    if not language_chunks:
        return ""
    language_chunk = language_chunks[0]
    ielts_chunks = [
        chunk
        for chunk in evidence_chunks
        if (chunk.field or chunk.section) == "ielts_score" and chunk.support_level == "direct"
    ]
    toefl_chunks = [
        chunk
        for chunk in evidence_chunks
        if (chunk.field or chunk.section) == "toefl_score" and chunk.support_level == "direct"
    ]

    program_label = _program_label_from_plan(plan)
    citation = _citation_for_chunk(language_chunk, evidence_chunks.index(language_chunk) + 1)
    requirement_text = _compact(language_chunk.text)
    first = (
        f"For {program_label}, the retrieved official evidence states: "
        f"{requirement_text} {citation}."
    )

    details: list[str] = []
    if ielts_chunks:
        chunk = ielts_chunks[0]
        details.append(
            f"The retrieved evidence states this IELTS requirement: {_compact(chunk.text)} "
            f"{_citation_for_chunk(chunk, evidence_chunks.index(chunk) + 1)}."
        )
    else:
        missing = (field_missing_reasons or {}).get(
            "ielts_score",
            "The retrieved official evidence does not state a specific IELTS band score for this program.",
        )
        details.append(missing)
    if toefl_chunks:
        chunk = toefl_chunks[0]
        details.append(
            f"The retrieved evidence also states this TOEFL requirement: {_compact(chunk.text)} "
            f"{_citation_for_chunk(chunk, evidence_chunks.index(chunk) + 1)}."
        )

    source_urls = list(dict.fromkeys([language_chunk.url, *(chunk.url for chunk in ielts_chunks[:1])]))
    source_line = "Sources: " + ", ".join(source_urls[:3])
    return f"{first}\n\n{' '.join(details)}\n\n{source_line}".strip()


def _language_answer_from_packet(packet: dict[str, Any], plan: QueryPlan) -> str:
    answered = packet.get("answered_fields", {})
    missing = packet.get("missing_fields", {})
    answered = answered if isinstance(answered, dict) else {}
    missing = missing if isinstance(missing, dict) else {}
    language = answered.get("english_language_requirement") or answered.get(
        "german_language_requirement"
    )
    if not isinstance(language, dict):
        return ""
    program_label = _program_label_from_plan(plan)
    evidence = _compact(language.get("value") or language.get("evidence_span"))
    citation = language.get("source_url", "")
    first = f"For {program_label}, the retrieved official evidence states: {evidence}."
    if citation:
        first += f" Source: {citation}"

    details: list[str] = []
    ielts = answered.get("ielts_score")
    if isinstance(ielts, dict):
        details.append(
            "The retrieved official evidence states this IELTS requirement: "
            + _compact(ielts.get("value") or ielts.get("evidence_span"))
            + (f" Source: {ielts.get('source_url')}" if ielts.get("source_url") else "")
        )
    elif "ielts_score" in missing:
        details.append(_natural_missing_answer("ielts_score"))
    toefl = answered.get("toefl_score")
    if isinstance(toefl, dict):
        details.append(
            "The retrieved official evidence states this TOEFL requirement: "
            + _compact(toefl.get("value") or toefl.get("evidence_span"))
            + (f" Source: {toefl.get('source_url')}" if toefl.get("source_url") else "")
        )
    return f"{first} {' '.join(details)}".strip()


def _deadline_answer_from_packet(query: str, packet: dict[str, Any], plan: QueryPlan) -> str:
    answered = packet.get("answered_fields", {})
    answered = answered if isinstance(answered, dict) else {}
    deadline = answered.get("application_deadline")
    if not isinstance(deadline, dict):
        return ""
    normalized_values = deadline.get("normalized_values", {})
    normalized_values = normalized_values if isinstance(normalized_values, dict) else {}
    text = _compact(deadline.get("value") or deadline.get("evidence_span"))
    wants_winter = "winter" in query.lower()
    wants_summer = "summer" in query.lower()
    requested_semester = (
        "winter semester" if wants_winter or not wants_summer else "summer semester"
    )
    requested_key = (
        "winter_semester_application_period"
        if requested_semester == "winter semester"
        else "summer_semester_application_period"
    )
    requested_period = str(normalized_values.get(requested_key) or "").strip()
    if not requested_period:
        requested_period = _extract_semester_period(text, requested_semester)
    if requested_period:
        answer = (
            f"For {_program_label_from_plan(plan)}, the {requested_semester} application "
            f"period is {requested_period.replace(' - ', ' to ')}."
        )
    else:
        answer = f"For {_program_label_from_plan(plan)}, the retrieved official evidence states: {text}."
    visa_deadline = str(normalized_values.get("visa_recommended_deadline") or "").strip()
    compulsory_deadline = str(normalized_values.get("compulsory_deadline") or "").strip()
    if visa_deadline and compulsory_deadline:
        answer += (
            f" The official page recommends applicants who need a visa apply by "
            f"{visa_deadline}, but {compulsory_deadline} is the compulsory deadline."
        )
    if deadline.get("source_url"):
        answer += f" Source: {deadline['source_url']}"
    return answer


def _tuition_answer_from_packet(packet: dict[str, Any], plan: QueryPlan) -> str:
    answered = packet.get("answered_fields", {})
    answered = answered if isinstance(answered, dict) else {}
    fee = answered.get("tuition_fee") or answered.get("semester_contribution")
    if not isinstance(fee, dict):
        return ""
    answer = f"For {_program_label_from_plan(plan)}, the retrieved official evidence states: {_compact(fee.get('value') or fee.get('evidence_span'))}."
    if fee.get("source_url"):
        answer += f" Source: {fee['source_url']}"
    return answer


def _document_answer_from_packet(packet: dict[str, Any], plan: QueryPlan) -> str:
    answered = packet.get("answered_fields", {})
    missing = packet.get("missing_fields", {})
    answered = answered if isinstance(answered, dict) else {}
    missing = missing if isinstance(missing, dict) else {}
    document_entries = [
        entry
        for field in (
            "required_application_documents",
            "international_applicant_documents",
            "degree_transcript_requirements",
            "language_proof",
        )
        if isinstance((entry := answered.get(field)), dict)
    ]
    items: list[str] = []
    sources: list[str] = []
    confirmed_process = ""
    for entry in document_entries:
        items.extend(str(item) for item in entry.get("checklist_items", []) if str(item).strip())
        if entry.get("confirmed_process") and not confirmed_process:
            confirmed_process = str(entry["confirmed_process"])
        if entry.get("source_url"):
            sources.append(str(entry["source_url"]))
    items = list(dict.fromkeys(items))
    sources = list(dict.fromkeys(sources))
    if items:
        lines = [f"For {_program_label_from_plan(plan)}, the retrieved official evidence includes this document checklist:"]
        lines.extend(f"- {item}" for item in items)
        if sources:
            lines.append(f"Source: {sources[0]}")
        return "\n".join(lines)
    if confirmed_process:
        source_text = f" Source checked: {sources[0]}" if sources else ""
        return (
            "The official page confirms the application is online and documents must be "
            f"uploaded, but the retrieved evidence does not show the full checklist.{source_text}"
        )
    if missing:
        checked = ", ".join(sources[:2]) if sources else "the retrieved official sources"
        return f"I checked {checked}, but the retrieved evidence does not show the full checklist."
    return ""


_BAD_FINAL_ANSWER_PATTERNS = (
    "Not verified from official sources",
    "Verified from selected evidence",
    "Verified in the retrieved official evidence",
    "###",
    "[...]",
    "answered_fields",
    "missing_fields",
    "selected evidence",
    "field completeness",
    "source status",
)


def _natural_missing_answer(field_name: str) -> str:
    if field_name in {"ielts_score", "language_test_score_thresholds", "language_test_thresholds"}:
        return "The retrieved official evidence does not state a specific IELTS band score."
    if field_name in {"application_deadline", "international_deadline", "other_semester_deadline"}:
        return "The retrieved official evidence does not state a separate deadline for this case."
    if field_name in {
        "required_application_documents",
        "international_applicant_documents",
        "degree_transcript_requirements",
    }:
        return (
            "The retrieved official evidence confirms the application route, but does not show "
            "the full document checklist."
        )
    if field_name in {"tuition_fee", "semester_contribution", "tuition_or_semester_fee"}:
        return "The retrieved official evidence does not state the fee amount."
    return "The retrieved official evidence does not state this requested detail."


def _has_bad_final_answer_output(answer: str) -> bool:
    text = str(answer or "")
    lowered = text.lower()
    if any(pattern.lower() in lowered for pattern in _BAD_FINAL_ANSWER_PATTERNS):
        return True
    if re.search(r"Application deadline:\s*.{0,80}(?:###|Application Periods|Winter semester:)", text, re.I | re.S):
        return True
    if re.search(
        r"\b(?:Application deadline|Intake|Other semester deadline|IELTS score|GPA|Application portal):\s.{300,}",
        text,
        re.I | re.S,
    ):
        return True
    if re.search(r"^\s*#{1,6}\s+", text, re.M):
        return True
    return False


def _clean_final_answer_text(answer: str) -> str:
    text = str(answer or "").strip()
    replacements = {
        "Not verified from official sources.": "The retrieved official evidence does not state this requested detail.",
        "Not verified from official sources": "The retrieved official evidence does not state this requested detail",
        "Verified from selected evidence.": "Stated in the retrieved official evidence.",
        "Verified from selected evidence": "Stated in the retrieved official evidence",
        "Verified in the retrieved official evidence.": "Stated in the retrieved official evidence.",
        "Verified in the retrieved official evidence": "Stated in the retrieved official evidence",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = text.replace("[...]", "")
    text = re.sub(
        r"\b(answered_fields|missing_fields|selected evidence|field completeness|source status)\b",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _raw_span_rendered(answer: str) -> bool:
    return _has_bad_final_answer_output(answer)


def _clean_final_answer_packet(packet: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(packet, dict):
        return {}

    def _clean_entry(entry: dict[str, Any]) -> dict[str, Any]:
        normalized_values = entry.get("normalized_values", {})
        normalized_values = normalized_values if isinstance(normalized_values, dict) else {}
        checklist_items = entry.get("checklist_items", [])
        checklist_items = checklist_items if isinstance(checklist_items, list) else []
        value = ""
        if normalized_values:
            value = "; ".join(f"{key}: {value}" for key, value in normalized_values.items())
        elif checklist_items:
            value = "; ".join(str(item) for item in checklist_items if str(item).strip())
        else:
            value = _clean_evidence_value(str(entry.get("value", "")))[:240]
        return {
            "value": value,
            "normalized_values": normalized_values,
            "checklist_items": checklist_items,
            "confirmed_process": str(entry.get("confirmed_process", "")),
            "source_url": str(entry.get("source_url", "")),
            "source_type": str(entry.get("source_type", "")),
            "support_level": str(entry.get("support_level", "")),
            "scope": str(entry.get("scope", "")),
        }

    answered = packet.get("answered_fields", {})
    answered = answered if isinstance(answered, dict) else {}
    missing = packet.get("missing_fields", {})
    missing = missing if isinstance(missing, dict) else {}
    required = packet.get("required_fields", {})
    required = required if isinstance(required, dict) else {}
    return {
        "intent": packet.get("intent", ""),
        "university": packet.get("university", ""),
        "program": packet.get("program", ""),
        "degree_level": packet.get("degree_level", ""),
        "required_fields": required,
        "answered_fields": {
            field: _clean_entry(entry)
            for field, entry in answered.items()
            if isinstance(entry, dict)
        },
        "missing_fields": {
            field: _natural_missing_answer(field)
            for field in missing.keys()
        },
        "excluded_fields": packet.get("excluded_fields", []),
        "deduplicated_fields": packet.get("deduplicated_fields", {}),
        "source_set_selected": packet.get("source_set_selected", []),
    }


def _safe_fallback_answer_from_packet(
    query: str,
    packet: dict[str, Any],
    plan: QueryPlan,
    evidence_chunks: list[EvidenceChunk],
    field_missing_reasons: dict[str, str] | None = None,
) -> str:
    if plan.intent == "deadline_lookup":
        answer = _deadline_answer_from_packet(query, packet, plan) or _deadline_answer_from_evidence(
            query, evidence_chunks
        )
    elif plan.intent == "language_requirement_lookup":
        answer = _language_answer_from_packet(packet, plan) or _language_answer_from_evidence(
            evidence_chunks, plan, field_missing_reasons
        )
    elif plan.intent == "tuition_fee_lookup":
        answer = _tuition_answer_from_packet(packet, plan)
    elif plan.intent == "document_requirement_lookup":
        answer = _document_answer_from_packet(packet, plan)
    else:
        answer = ""
    if not answer:
        requested_fields = plan.required_fields or plan.required_info or []
        answer = " ".join(_natural_missing_answer(field) for field in requested_fields)
    return _clean_final_answer_text(answer or "The retrieved official evidence does not answer the requested field.")


async def generate_answer(
    query: str,
    evidence_chunks: list[EvidenceChunk],
    plan: QueryPlan,
    field_confidence: dict[str, str] | None = None,
    field_missing_reasons: dict[str, str] | None = None,
    evidence_packet: dict[str, Any] | None = None,
    answer_metadata: dict[str, Any] | None = None,
) -> str:
    from app.infra.bedrock_chat_client import client as bedrock_client

    packet = evidence_packet or {}
    if not packet and evidence_chunks:
        grouped_from_chunks: dict[str, list[EvidenceChunk]] = {}
        for chunk in evidence_chunks:
            grouped_from_chunks.setdefault(chunk.field or chunk.section, []).append(chunk)
        packet, _packet_chunks = build_evidence_packet(grouped_from_chunks, plan)
    if not evidence_chunks and not packet.get("answered_fields"):
        missing = field_missing_reasons or {
            field: _missing_reason_for_field(field, {})
            for field in (plan.required_fields or plan.required_info)
        }
        requested_fields = plan.required_fields or plan.required_info or []
        natural = " ".join(_natural_missing_answer(field) for field in requested_fields)
        answer = natural or "The retrieved official evidence does not answer the requested field."
        if answer_metadata is not None:
            answer_metadata.update(
                {
                    "final_answer_source": "fallback_builder",
                    "final_prompt_used": False,
                    "raw_span_rendered": _raw_span_rendered(answer),
                    "final_answer_before_sanitizer": answer,
                    "final_answer_after_sanitizer": _clean_final_answer_text(answer),
                }
            )
        return _clean_final_answer_text(answer)

    system_prompt = """
You are UniGraph's final answer writer.

You receive:
- user_question
- detected_intent
- answered_fields
- missing_fields
- excluded_fields
- selected evidence with citations

Write the final answer for the student.

Rules:
1. Answer only the user's actual question.
2. Use only the selected evidence.
3. Do not copy raw evidence chunks.
4. Do not copy webpage markdown such as ###, [...], headings, table fragments, or long snippets.
5. Do not show internal field labels like "Application deadline:", "Intake:", "answered_fields", "missing_fields", or "verified from selected evidence".
6. Never write "Not verified from official sources".
7. If a requested field is missing, say it naturally, e.g. "The retrieved official evidence does not state a specific IELTS band score."
8. Do not mention fields the user did not ask about.
9. For narrow factual questions, write one short paragraph.
10. Use checklist only for document questions.
11. Use table only for comparison questions.
12. Cite the evidence beside factual claims.

Intent formatting:
- deadline_lookup: one short paragraph with only the deadline/application period.
- language_requirement_lookup: one short paragraph with only language/IELTS/TOEFL/CEFR info.
- tuition_fee_lookup: one short paragraph with only tuition/semester fee.
- document_requirement_lookup: checklist.
- comparison_lookup: table.

Example:
Question: When is the winter semester application deadline for MSc Informatics at TU Munich?
Good answer:
For MSc Informatics at TUM, the official CIT page lists the winter semester application period as 01 February to 31 May. For applicants who need a visa, 31 March is recommended, while 31 May is the compulsory deadline. [citation]

Question: What are the IELTS requirements for MSc Artificial Intelligence at FAU?
Good answer:
For MSc Artificial Intelligence at FAU, the official FAU AI page states that Master's students need English proficiency at minimum CEFR B2. The retrieved official evidence does not state a specific IELTS band score. [citation]
"""
    user_prompt = (
        f"Question: {query}\n\n"
        f"Detected intent: {plan.intent}\n"
        f"Required fields: {json.dumps(plan.required_fields or plan.required_info, ensure_ascii=False)}\n"
        f"Optional fields: {json.dumps(plan.optional_fields, ensure_ascii=False)}\n"
        f"Excluded fields: {json.dumps(plan.excluded_fields, ensure_ascii=False)}\n"
        f"Field-level confidence: {json.dumps(field_confidence or {}, ensure_ascii=False)}\n\n"
        f"Field-specific missing reasons: {json.dumps(field_missing_reasons or {}, ensure_ascii=False)}\n\n"
        f"Answer shape: {_answer_shape(plan)}\n\n"
        f"Normalized evidence packet:\n{json.dumps(_clean_final_answer_packet(packet), ensure_ascii=False)[:8000]}"
    )
    fallback_answer = _safe_fallback_answer_from_packet(
        query, packet, plan, evidence_chunks, field_missing_reasons
    )
    metadata = {
        "final_answer_source": "llm_synthesis",
        "final_prompt_used": True,
        "raw_span_rendered": False,
        "final_answer_before_sanitizer": "",
        "final_answer_after_sanitizer": "",
    }
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await bedrock_client.chat.completions.create(
            model=settings.bedrock.primary_model_id,
            messages=messages,
        )
        answer = str(response.choices[0].message.content or "").strip()
        metadata["final_answer_before_sanitizer"] = answer
        if _has_bad_final_answer_output(answer):
            metadata["raw_span_rendered"] = True
            stricter_prompt = (
                user_prompt
                + "\n\nPrompt-level check failed. The drafted answer contained one of: "
                "\"###\", \"[...]\", \"Not verified from official sources\", "
                "\"Verified from selected evidence\", or copied raw chunks. Regenerate once. "
                "Return only a clean student-facing answer using the rules in the system prompt."
            )
            response = await bedrock_client.chat.completions.create(
                model=settings.bedrock.primary_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": stricter_prompt},
                ],
            )
            answer = str(response.choices[0].message.content or "").strip()
            metadata["final_answer_before_sanitizer"] = answer
        if _has_bad_final_answer_output(answer):
            metadata["final_answer_source"] = "fallback_builder"
            metadata["raw_span_rendered"] = True
            answer = fallback_answer
        else:
            answer = _clean_final_answer_text(answer)
        metadata["final_answer_after_sanitizer"] = answer
        if answer_metadata is not None:
            answer_metadata.update(metadata)
        return answer
    except Exception as exc:
        logger.warning("UniGraph answer generation failed | error=%s", exc)
        metadata["final_answer_source"] = "fallback_builder"
        metadata["final_prompt_used"] = False
        metadata["raw_span_rendered"] = _raw_span_rendered(fallback_answer)
        metadata["final_answer_before_sanitizer"] = fallback_answer
        metadata["final_answer_after_sanitizer"] = _clean_final_answer_text(fallback_answer)
        if answer_metadata is not None:
            answer_metadata.update(metadata)
        return metadata["final_answer_after_sanitizer"]


def _fields_not_verified(plan: QueryPlan, grouped: dict[str, list[EvidenceChunk]]) -> list[str]:
    missing: list[str] = []
    for section in plan.required_fields or plan.required_info or ["general_information"]:
        rows = grouped.get(section, [])
        if section in VALUE_REQUIRED_FIELDS:
            if not any(row.support_level == "direct" for row in rows[:3]):
                missing.append(section)
            continue
        if not rows or rows[0].score < 0.48:
            missing.append(section)
    return missing


def _missing_reason_for_field(field_name: str, grouped: dict[str, list[EvidenceChunk]]) -> str:
    rows = grouped.get(field_name, [])
    if field_name == "ielts_score":
        language_text = " ".join(
            row.text for row in grouped.get("english_language_requirement", [])[:5]
        )
        if language_text:
            return "Retrieved evidence discusses English proficiency, but does not state a specific IELTS band score for this program."
        return "No retrieved evidence states a specific IELTS band score for this program."
    if field_name == "application_deadline":
        return "The retrieved evidence does not clearly state the application deadline for the requested program."
    if field_name == "gpa_requirement":
        return "The retrieved official sources do not state a fixed minimum GPA; eligibility may depend on formal academic assessment."
    if field_name == "required_application_documents":
        return "I found no retrieved program-specific application document checklist."
    if field_name == "tuition_fee":
        return "The retrieved evidence does not clearly state the tuition fee for the requested applicant context."
    if field_name in {"aps_requirement", "vpd_requirement", "uni_assist_requirement"}:
        return f"The retrieved evidence does not clearly verify the {field_name.replace('_', ' ')}."
    if rows:
        return f"Retrieved evidence for {field_name.replace('_', ' ')} is indirect or weak."
    return f"No retrieved evidence directly verifies {field_name.replace('_', ' ')}."


def _answer_shape(plan: QueryPlan) -> dict[str, Any]:
    if plan.intent == "language_requirement_lookup":
        return {
            "style": "brief_1_to_3_paragraphs",
            "include_only": [
                "English/German language requirement",
                "IELTS/TOEFL/Duolingo/CEFR evidence when present",
                "exact missing test score reason when absent",
            ],
            "avoid_sections": True,
        }
    if plan.intent == "deadline_lookup":
        return {
            "style": "brief_deadline_answer",
            "include_only": ["deadline/application period", "intake", "applicant category"],
            "avoid_sections": True,
        }
    if plan.intent == "tuition_fee_lookup":
        return {
            "style": "brief_fee_answer",
            "include_only": ["tuition fee", "semester contribution", "fee category", "exemptions"],
            "avoid_sections": True,
        }
    if plan.intent == "application_portal_lookup":
        return {
            "style": "brief_application_method_answer",
            "include_only": [
                "portal",
                "uni-assist/VPD if verified",
                "direct application instructions",
            ],
            "avoid_sections": True,
        }
    if plan.intent == "document_requirement_lookup":
        return {
            "style": "checklist",
            "include_only": [
                "required documents",
                "international applicant documents",
                "language proof",
                "APS/VPD/uni-assist if relevant",
            ],
            "avoid_sections": False,
        }
    if plan.intent == "eligibility_check":
        return {
            "style": "cautious_profile_comparison",
            "include_only": [
                "academic eligibility",
                "GPA/minimum grade if available",
                "ECTS/degree background",
                "language requirements",
                "GRE/GMAT only if relevant",
            ],
            "avoid_guaranteeing_admission": True,
        }
    if plan.intent == "multi_program_discovery":
        return {
            "style": "shortlist",
            "include_phrase": "based on the programs verified in this search",
            "max_programs": 5,
        }
    return {"style": "adaptive", "include_only": plan.required_fields or plan.required_info}


def _field_statuses(
    plan: QueryPlan, grouped: dict[str, list[EvidenceChunk]]
) -> tuple[list[str], list[str], dict[str, str]]:
    answered: list[str] = []
    partial: list[str] = []
    missing: dict[str, str] = {}
    for field_name in plan.required_fields or plan.required_info or ["general_information"]:
        rows = grouped.get(field_name, [])
        if not rows:
            missing[field_name] = _missing_reason_for_field(field_name, grouped)
            continue
        best = rows[0]
        if field_name in VALUE_REQUIRED_FIELDS and best.support_level != "direct":
            partial.append(field_name)
            missing[field_name] = _missing_reason_for_field(field_name, grouped)
        elif best.score >= 0.58 and best.support_level in {"direct", "indirect"}:
            answered.append(field_name)
        elif best.score >= 0.40:
            partial.append(field_name)
            missing[field_name] = _missing_reason_for_field(field_name, grouped)
        else:
            missing[field_name] = _missing_reason_for_field(field_name, grouped)
    return answered, partial, missing


def _confidence(chunks: list[EvidenceChunk], fields_not_verified: list[str]) -> float:
    if not chunks:
        return 0.0
    avg = sum(chunk.score for chunk in chunks[:5]) / min(5, len(chunks))
    penalty = min(0.35, len(fields_not_verified) * 0.08)
    return max(0.0, min(1.0, avg - penalty))


def _confidence_label(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.52:
        return "medium"
    if score > 0.0:
        return "low"
    return "not verified"


def _third_party_usage(
    evidence: list[EvidenceChunk], grouped: dict[str, list[EvidenceChunk]]
) -> list[dict[str, Any]]:
    usage: list[dict[str, Any]] = []
    for chunk in evidence:
        if chunk.source_type not in {"third_party_education_site", "blog_or_forum", "other"}:
            continue
        field_name = chunk.field or chunk.section
        official_available = any(
            row.source_type
            in {
                "official_university_page",
                "official_university_pdf",
                "daad",
                "uni_assist",
                "government_or_eu",
            }
            for row in grouped.get(field_name, [])
        )
        usage.append(
            {
                "field": field_name,
                "url": chunk.url,
                "source_type": chunk.source_type,
                "reason": (
                    "used because no official or trusted source was selected for this field"
                    if not official_available
                    else "selected as supplementary evidence despite trusted evidence being available"
                ),
            }
        )
    return usage


def _field_level_confidence(
    query: str,
    plan: QueryPlan,
    grouped: dict[str, list[EvidenceChunk]],
) -> dict[str, str]:
    confidence: dict[str, str] = {}
    for section in plan.required_fields or plan.required_info or ["general_information"]:
        rows = grouped.get(section, [])
        best_score = rows[0].score if rows else 0.0
        confidence[section] = _confidence_label(best_score)
    if _query_mentions_language_requirement(query):
        language_rows = grouped.get("english_language_requirement", [])
        language_text = " ".join(row.text for row in language_rows[:6]).lower()
        official_language = any(
            float(row.scoring.get("source_quality", 0.0) or 0.0) >= 0.75 for row in language_rows
        )
        confidence["English B2 requirement"] = (
            "high" if official_language and re.search(r"\bb2\b", language_text) else "not verified"
        )
        confidence["numeric IELTS score"] = (
            "high"
            if official_language and re.search(r"\bielts\b.{0,80}\b[4-9](?:\.\d)?\b", language_text)
            else "not verified"
        )
        confidence["per-section IELTS score"] = (
            "high"
            if official_language
            and re.search(
                r"\b(reading|writing|speaking|listening)\b.{0,80}\b[4-9](?:\.\d)?\b", language_text
            )
            else "not verified"
        )
    return confidence


async def research_university_question(query: str) -> ResearchResult:
    started = time.perf_counter()
    original_question = _clean_original_question(query)
    request_id = f"unigraph-{uuid.uuid4().hex[:12]}"
    logger.info(
        "UniGraph Phase 1 started | request_id=%s | user_question=%s",
        request_id,
        original_question,
    )
    previous_state_cleared = True
    debug_collector: dict[str, Any] = {
        "request_id": request_id,
        "current_question": original_question,
        "previous_state_cleared": previous_state_cleared,
    }
    search_results: list[dict[str, Any]] = []
    selected_urls: list[dict[str, Any]] = []
    extracted: list[ExtractedContent] = []
    grouped: dict[str, list[EvidenceChunk]] = {}
    evidence: list[EvidenceChunk] = []
    evidence_packet: dict[str, Any] = {}
    plan = await analyze_query(original_question)
    query_validation = validate_and_repair_search_queries(plan, original_question)
    retrieval_started = time.perf_counter()
    search_results, selected_urls, tavily_calls, tier_debug = await execute_tiered_retrieval(
        plan,
        original_question,
        debug_collector=debug_collector,
    )
    extracted = await extract_all_contents(
        selected_urls, plan=plan, debug_collector=debug_collector
    )
    retrieval_duration = time.perf_counter() - retrieval_started
    grouped = group_and_rank_evidence(extracted, plan, debug_collector=debug_collector)
    evidence_packet, evidence = build_evidence_packet(grouped, plan)
    field_completeness = evidence_packet.get("field_completeness_status", "missing")
    packet_required = evidence_packet.get("required_fields", {})
    packet_required = packet_required if isinstance(packet_required, dict) else {}
    packet_missing = evidence_packet.get("missing_fields", {})
    packet_missing = packet_missing if isinstance(packet_missing, dict) else {}
    packet_conflicts = evidence_packet.get("conflicting_fields", {})
    packet_conflicts = packet_conflicts if isinstance(packet_conflicts, dict) else {}
    fields_not_verified = [
        field_name
        for field_name, status in packet_required.items()
        if status in {"missing", "conflicting"}
        or not str(status).startswith(("answered", "deduplicated_to:"))
    ]
    fields_answered = [
        field_name
        for field_name, status in packet_required.items()
        if status == "answered_direct"
    ]
    fields_partially_answered = [
        field_name
        for field_name, status in packet_required.items()
        if status == "answered_indirect"
    ]
    fields_missing_with_reason = dict(packet_missing)
    for field_name, conflict in packet_conflicts.items():
        fields_missing_with_reason[field_name] = str(
            conflict.get("reason", "Trusted sources conflict for this requested field.")
        )
    if not fields_answered and not fields_partially_answered:
        fields_answered, fields_partially_answered, fields_missing_with_reason = _field_statuses(
            plan, grouped
        )
        fields_not_verified = _fields_not_verified(plan, grouped)
    logger.info(
        "UniGraph field completeness | status=%s | answered=%s | missing=%s",
        field_completeness,
        fields_answered + fields_partially_answered,
        fields_not_verified,
    )
    logger.info(
        "UniGraph normalized evidence packet | normalized_evidence_packet=%s | deduplicated_fields=%s",
        evidence_packet.get("normalized_evidence_packet", {}),
        evidence_packet.get("deduplicated_fields", {}),
    )
    confidence = _confidence(evidence, fields_not_verified)
    field_confidence = _field_level_confidence(query, plan, grouped)
    answer_metadata: dict[str, Any] = {}
    answer = await generate_answer(
        original_question,
        evidence,
        plan,
        field_confidence,
        fields_missing_with_reason,
        evidence_packet=evidence_packet,
        answer_metadata=answer_metadata,
    )
    if plan.intent == "deadline_lookup":
        answer_formatter_used = "deadline_formatter"
    elif plan.intent == "language_requirement_lookup":
        answer_formatter_used = "language_requirement_formatter"
    elif plan.intent == "tuition_fee_lookup":
        answer_formatter_used = "tuition_fee_formatter"
    elif plan.intent == "document_requirement_lookup":
        answer_formatter_used = "document_checklist_formatter"
    else:
        answer_formatter_used = "llm_dynamic_formatter"
    final_answer_input = {
        "intent": plan.intent,
        "required_fields": plan.required_fields or plan.required_info,
        "answered_fields": evidence_packet.get("answered_fields", {}),
        "missing_fields": evidence_packet.get("missing_fields", {}),
        "excluded_fields": plan.excluded_fields,
    }
    logger.info(
        "UniGraph final answer synthesis | answer_formatter_used=%s | final_answer_source=%s | final_prompt_used=%s | raw_span_rendered=%s | final_answer_before_sanitizer=%s | final_answer_after_sanitizer=%s | final_answer_input=%s",
        answer_formatter_used,
        answer_metadata.get("final_answer_source", ""),
        answer_metadata.get("final_prompt_used", False),
        answer_metadata.get("raw_span_rendered", False),
        str(answer_metadata.get("final_answer_before_sanitizer", ""))[:500],
        str(answer_metadata.get("final_answer_after_sanitizer", ""))[:500],
        final_answer_input,
    )
    duration = time.perf_counter() - started
    chunks_created_detail = [
        {
            "url": item.url,
            "title": item.title,
            "domain": item.domain,
            "source_type": item.source_type,
            "document_type": item.document_type,
            "page_number": page.page_number,
            "retrieved_at": item.retrieved_at,
            "query": item.query,
            "chunks": chunk_text(page.text),
        }
        for item in extracted
        for page in item.pages
    ]
    debug_info = {
        "request_id": request_id,
        "current_question": original_question,
        "previous_state_cleared": previous_state_cleared,
        "original_question": original_question,
        "user_question": original_question,
        "search_keywords": _field_query_keywords(plan),
        "decomposition_fallback_used": plan.decomposition_fallback_used,
        "planner_type": plan.planner_type,
        "fallback_used": plan.decomposition_fallback_used,
        "fallback_error": plan.user_profile_details.get("fallback_error", ""),
        "query_mode": plan.query_mode,
        "query_decomposition": plan.__dict__,
        "detected_intent": plan.intent,
        "detected_university": plan.university,
        "detected_program": plan.program,
        "resolved_official_domains": tier_debug.get("resolved_official_domains", []),
        "required_fields": plan.required_fields or plan.required_info,
        "optional_fields": plan.optional_fields,
        "excluded_fields": plan.excluded_fields,
        "query_validation_status": query_validation,
        "retrieval_tier_used": tier_debug.get("tier_used", ""),
        "fallback_tier_used": tier_debug.get("fallback_tier_used", ""),
        "tiered_retrieval": tier_debug,
        "generated_search_strategy": {
            "intent": plan.intent,
            "query_mode": plan.query_mode,
            "required_information_sections": plan.required_fields or plan.required_info,
            "optional_fields": plan.optional_fields,
            "excluded_fields": plan.excluded_fields,
            "priority_sources": plan.priority_sources,
            "keywords": plan.keywords,
            "german_keywords": plan.german_keywords,
            "limits": {
                "max_queries": _mode_limit(plan, "max_queries", MAX_QUERIES),
                "max_results_per_query": MAX_RESULTS_PER_QUERY,
                "max_total_urls_to_fetch": _mode_limit(plan, "max_urls", MAX_TOTAL_URLS_TO_FETCH),
                "max_pdfs_to_read": _mode_limit(plan, "max_pdfs", MAX_PDFS_TO_READ),
                "max_pdf_size_mb": MAX_PDF_SIZE_MB,
                "max_pdf_pages": _mode_limit(plan, "max_pdf_pages", MAX_PDF_PAGES),
                "max_evidence_chunks": MAX_EVIDENCE_CHUNKS,
            },
        },
        "generated_search_queries": plan.search_queries,
        "generated_queries": plan.search_queries,
        "tavily_calls_used": tavily_calls
        + sum(1 for item in extracted if item.document_type == "html"),
        "tavily_search_calls": tavily_calls,
        "tavily_extract_calls": debug_collector.get("tavily_extract_calls", 0),
        "fan_out_search_results": search_results,
        "raw_search_results": search_results,
        "filtered_urls": selected_urls,
        "selected_urls": selected_urls,
        "accepted_urls": selected_urls,
        "deduplicated_urls": [item["url"] for item in selected_urls],
        "skipped_urls": debug_collector.get("skipped_urls", []),
        "rejected_urls_with_reasons": debug_collector.get("skipped_urls", []),
        "rejected_pdfs": debug_collector.get("rejected_pdfs", []),
        "early_stop_triggered": debug_collector.get("early_stop_triggered", False),
        "urls_fetched": [item.url for item in extracted],
        "fetched_urls": [item.url for item in extracted],
        "pdfs_read": [item.url for item in extracted if item.document_type == "pdf"],
        "PDFs_downloaded": debug_collector.get("pdfs_downloaded", []),
        "PDF_pages_read": debug_collector.get("pdf_pages_read", 0),
        "pdf_pages_extracted": {
            item.url: [page.page_number for page in item.pages if page.page_number is not None]
            for item in extracted
            if item.document_type == "pdf"
        },
        "chunks_created": sum(
            len(chunk_text(page.text)) for item in extracted for page in item.pages
        ),
        "chunks_created_detail": chunks_created_detail,
        "grouped_evidence_by_requested_section": {
            section: {
                "chunk_count": len(rows),
                "top_chunks": [_chunk_debug_payload(chunk) for chunk in rows[:5]],
            }
            for section, rows in grouped.items()
        },
        "selected_evidence_chunks": [_chunk_debug_payload(chunk) for chunk in evidence],
        "selected_answer_spans": debug_collector.get("selected_answer_spans", []),
        "extracted_field_spans": debug_collector.get("extracted_field_spans", []),
        "field_mapping_results": debug_collector.get("field_mapping_results", []),
        "selected_chunks": [_chunk_debug_payload(chunk) for chunk in evidence],
        "evidence_selected_count": len(evidence),
        "source_set_selected": evidence_packet.get("source_set_selected", []),
        "field_completeness_status": field_completeness,
        "answered_fields": evidence_packet.get("answered_fields", {}),
        "missing_fields": evidence_packet.get("missing_fields", {}),
        "normalized_evidence_packet": evidence_packet.get("normalized_evidence_packet", {}),
        "deduplicated_fields": evidence_packet.get("deduplicated_fields", {}),
        "conflicting_fields": evidence_packet.get("conflicting_fields", {}),
        "complementary_sources_used": [
            source
            for source in evidence_packet.get("source_set_selected", [])[1:]
            if isinstance(source, dict)
        ],
        "evidence_passed_to_final_answer": [_chunk_debug_payload(chunk) for chunk in evidence],
        "field_mapped_evidence": [_chunk_debug_payload(chunk) for chunk in evidence],
        "excluded_evidence_chunks": debug_collector.get("excluded_evidence_chunks", []),
        "excluded_chunks": debug_collector.get("excluded_evidence_chunks", []),
        "source_scores": debug_collector.get("source_scores", []),
        "fan_in_evidence": [
            {"section": chunk.section, "url": chunk.url, "score": chunk.score} for chunk in evidence
        ],
        "total_tavily_calls": tavily_calls
        + sum(1 for item in extracted if item.document_type == "html"),
        "final_confidence": confidence,
        "field_level_confidence": field_confidence,
        "fields_not_verified": fields_not_verified,
        "fields_missing": fields_not_verified,
        "fields_excluded": plan.excluded_fields,
        "final_required_fields": plan.required_fields or plan.required_info,
        "final_optional_fields": plan.optional_fields,
        "final_excluded_fields": plan.excluded_fields,
        "fields_answered": fields_answered,
        "fields_partially_answered": fields_partially_answered,
        "fields_missing_with_reason": fields_missing_with_reason,
        "excluded_chunks_with_reasons": debug_collector.get("excluded_evidence_chunks", []),
        "evidence_passed_to_final_llm": [_chunk_debug_payload(chunk) for chunk in evidence],
        "evidence_packet_sent_to_final_llm": evidence_packet,
        "answer_template_used": (
            "deadline_dynamic"
            if plan.intent == "deadline_lookup"
            else "language_requirement_dynamic"
            if plan.intent == "language_requirement_lookup"
            else "llm_dynamic"
        ),
        "fallback_template_reason": "",
        "row_level_extraction_metadata": [
            {
                "url": chunk.url,
                "page_number": chunk.page_number,
                "row_or_section": chunk.row_or_section,
                "field": chunk.field,
                "target_program_row_found": bool(
                    chunk.scoring.get("target_program_row_found", 0.0)
                ),
                "neighboring_rows_ignored": bool(
                    chunk.scoring.get("neighboring_rows_ignored", 0.0)
                ),
                "row_confidence": chunk.scoring.get("row_confidence", 0.0),
            }
            for chunk in evidence
        ],
        "final_answer_shape": _answer_shape(plan),
        "answer_formatter_used": answer_formatter_used,
        "final_answer_source": answer_metadata.get("final_answer_source", ""),
        "final_prompt_used": bool(answer_metadata.get("final_prompt_used", False)),
        "raw_span_rendered": bool(answer_metadata.get("raw_span_rendered", False)),
        "final_answer_before_sanitizer": answer_metadata.get("final_answer_before_sanitizer", ""),
        "final_answer_after_sanitizer": answer_metadata.get("final_answer_after_sanitizer", ""),
        "final_answer_input": final_answer_input,
        "third_party_sources_used": _third_party_usage(evidence, grouped),
        "final_context_chars": len(_evidence_context(evidence)),
        "final_context_tokens_estimate": (
            max(1, len(_evidence_context(evidence)) // 4) if evidence else 0
        ),
        "total_retrieval_time_seconds": retrieval_duration,
        "duration_seconds": duration,
    }
    logger.info(
        "UniGraph Phase 1 complete | confidence=%.3f | fields_not_verified=%s | tavily_calls=%s",
        confidence,
        fields_not_verified,
        debug_info["total_tavily_calls"],
    )
    return ResearchResult(
        query=original_question,
        answer=answer,
        evidence_chunks=evidence,
        query_plan=plan,
        debug_info=debug_info,
    )


def _chunk_to_retrieval_row(index: int, chunk: EvidenceChunk) -> dict[str, Any]:
    return {
        "chunk_id": f"unigraph:phase1:evidence:{index}",
        "content": chunk.text,
        "distance": max(0.0, 1.0 - chunk.score),
        "metadata": {
            "url": chunk.url,
            "title": chunk.title,
            "domain": chunk.domain,
            "source_type": chunk.source_type,
            "document_type": chunk.document_type,
            "page_number": chunk.page_number,
            "retrieved_at": chunk.retrieved_at,
            "query": chunk.query,
            "section": chunk.section,
            "field": chunk.field or chunk.section,
            "evidence_scope": chunk.evidence_scope,
            "support_level": chunk.support_level,
            "selection_reason": chunk.selection_reason,
            "row_or_section": chunk.row_or_section,
            "score": chunk.score,
            **chunk.scoring,
        },
    }


def _coverage_row_value(
    section: str,
    rows: list[dict[str, Any]],
    missing: bool,
    debug_info: dict[str, Any] | None = None,
) -> str:
    if missing:
        return ""
    debug_info = debug_info if isinstance(debug_info, dict) else {}
    answered_fields = debug_info.get("answered_fields", {})
    answered_fields = answered_fields if isinstance(answered_fields, dict) else {}
    entry = answered_fields.get(section)
    if isinstance(entry, dict):
        normalized_values = entry.get("normalized_values", {})
        normalized_values = normalized_values if isinstance(normalized_values, dict) else {}
        if normalized_values:
            return "; ".join(f"{key}: {value}" for key, value in normalized_values.items())
        value = _clean_evidence_value(str(entry.get("value", "")))
        if value:
            return value[:260]
    for row in rows:
        metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
        if not isinstance(metadata, dict) or metadata.get("section") != section:
            continue
        cleaned = _clean_evidence_value(str(row.get("content", "")))
        if cleaned:
            return cleaned[:260]
    return "Stated in the retrieved official evidence."


async def aretrieve_web_chunks(query: str, **kwargs) -> dict[str, Any]:
    debug_enabled = bool(kwargs.get("debug", False))
    result = await research_university_question(query)
    rows = [
        _chunk_to_retrieval_row(index, chunk)
        for index, chunk in enumerate(result.evidence_chunks, start=1)
    ]
    payload = {
        "query": result.query,
        "results": rows,
        "answer": result.answer,
        "retrieval_strategy": "unigraph_phase1_official_source_research",
        "web_retrieval_verified": bool(result.evidence_chunks)
        and not result.debug_info.get("fields_not_verified"),
        "unigraph_answered_required_field": bool(result.debug_info.get("fields_answered"))
        or bool(result.debug_info.get("fields_partially_answered")),
        "confidence": result.debug_info.get("final_confidence", 0.0),
        "field_level_confidence": result.debug_info.get("field_level_confidence", {}),
        "fields_not_verified": result.debug_info.get("fields_not_verified", []),
        "query_plan": {
            "planner": result.debug_info.get("planner_type", result.query_plan.planner_type),
            "planner_type": result.debug_info.get("planner_type", result.query_plan.planner_type),
            "llm_used": result.debug_info.get("planner_type") == "llm",
            "fallback_error": result.debug_info.get("fallback_error", ""),
            "detected_intent": result.query_plan.intent,
            "intent": result.query_plan.intent,
            "university": result.query_plan.university,
            "program": result.query_plan.program,
            "degree_level": result.query_plan.degree_level,
            "required_fields": result.query_plan.required_fields
            or result.query_plan.required_info,
            "optional_fields": result.query_plan.optional_fields,
            "excluded_fields": result.query_plan.excluded_fields,
            "answer_shape": result.query_plan.answer_shape,
            "queries": result.query_plan.search_queries,
        },
        "coverage_ledger": [
            {
                "field": section,
                "label": section.replace("_", " ").title(),
                "status": (
                    "not_verified"
                    if section in result.debug_info.get("fields_not_verified", [])
                    else "found"
                ),
                "value": _coverage_row_value(
                    section,
                    rows,
                    section in result.debug_info.get("fields_not_verified", []),
                    result.debug_info,
                ),
                "evidence_snippet": _coverage_row_value(
                    section,
                    rows,
                    section in result.debug_info.get("fields_not_verified", []),
                    result.debug_info,
                ),
                "source_url": next(
                    (
                        row["metadata"]["url"]
                        for row in rows
                        if row["metadata"]["section"] == section
                    ),
                    "",
                ),
                "source_type": next(
                    (
                        row["metadata"]["source_type"]
                        for row in rows
                        if row["metadata"]["section"] == section
                    ),
                    "",
                ),
                "confidence": result.debug_info.get("final_confidence", 0.0),
                "confidence_label": result.debug_info.get("field_level_confidence", {}).get(
                    section, ""
                ),
            }
            for section in (
                result.query_plan.required_fields
                or result.query_plan.required_info
                or ["general_information"]
            )
        ],
    }
    if debug_enabled:
        save_debug_artifact(result.debug_info)
    return payload
