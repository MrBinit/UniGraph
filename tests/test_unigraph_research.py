import pytest

from app.services import unigraph_research as service
from app.services import llm_service
from app.schemas.chat_schema import ChatRequest


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def test_canonicalize_url_removes_tracking_params_and_fragment():
    assert (
        service.canonicalize_url("https://www.tum.de/en/studies?utm_source=x&fbclid=y&id=42#apply")
        == "https://www.tum.de/en/studies?id=42"
    )


def test_chat_request_debug_defaults_to_true():
    request = ChatRequest(user_id="user-1", prompt="What are the IELTS requirements at FAU?")

    assert request.debug is True


def test_chat_request_debug_can_be_disabled_explicitly():
    request = ChatRequest(
        user_id="user-1",
        prompt="What are the IELTS requirements at FAU?",
        debug=False,
    )

    assert request.debug is False


def test_select_urls_filters_low_quality_and_prefers_official_pdf():
    plan = service.QueryPlan(
        priority_sources=["tum.de", "daad.de"],
        search_queries=[{"query": "tum data engineering deadline", "priority": 1.0}],
    )
    rows = [
        {
            "query": "tum data engineering deadline",
            "type": "pdf",
            "priority": 1.0,
            "results": [
                {
                    "title": "Admission requirements",
                    "link": "https://www.tum.de/file.pdf?utm_campaign=x",
                    "snippet": "application requirements",
                },
                {
                    "title": "Forum thread",
                    "link": "https://www.reddit.com/r/germany/comments/1",
                    "snippet": "deadline rumor",
                },
            ],
        }
    ]

    selected = service.select_and_deduplicate_urls(rows, plan)

    assert [item["url"] for item in selected] == ["https://www.tum.de/file.pdf"]
    assert selected[0]["source_quality"] == 0.95
    assert selected[0]["document_type"] == "pdf"


def test_fau_german_context_filters_florida_atlantic(monkeypatch):
    plan = service._with_german_fau_focus(
        service.QueryPlan(
            university_short="FAU",
            program="MSc Artificial Intelligence",
            required_info=["english_language_requirement"],
            required_fields=["english_language_requirement"],
            priority_sources=["fau.de"],
            search_queries=[{"query": "FAU MSc Artificial Intelligence IELTS", "priority": 1.0}],
        ),
        "What is the IELTS requirement for MSc Artificial Intelligence at FAU?",
    )
    rows = [
        {
            "query": "FAU MSc Artificial Intelligence IELTS",
            "type": "official_page",
            "priority": 1.0,
            "results": [
                {
                    "title": "FAU Erlangen AI language requirements",
                    "link": "https://www.fau.de/education/degree-programme/artificial-intelligence",
                    "snippet": "English language requirements IELTS",
                },
                {
                    "title": "Florida Atlantic University Artificial Intelligence",
                    "link": "https://www.fau.edu/engineering/artificial-intelligence",
                    "snippet": "Florida Atlantic University IELTS",
                },
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(rows, plan, debug_collector=debug)

    assert [item["url"] for item in selected] == [
        "https://www.fau.de/education/degree-programme/artificial-intelligence"
    ]
    assert debug["skipped_urls"][0]["reason"] == "ambiguous_secondary_institution_florida_atlantic"


def test_group_and_rank_evidence_preserves_metadata_and_sections():
    plan = service.QueryPlan(
        required_info=["application_deadline", "english_language_requirement"],
        required_fields=["application_deadline", "english_language_requirement"],
        keywords=["application", "deadline", "language"],
        german_keywords=["bewerbungsfrist", "sprachnachweis"],
        search_queries=[{"query": "official deadline", "priority": 1.0}],
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.tum.de/admissions",
            title="Admissions",
            domain="tum.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-28T00:00:00+00:00",
            query="official deadline",
            pages=[
                service.ExtractedPage(
                    text="The application deadline is 31 May. IELTS is accepted as language proof."
                )
            ],
        )
    ]

    grouped = service.group_and_rank_evidence(extracted, plan)
    selected = service.fan_in_evidence(grouped)

    assert "application_deadline" in grouped
    assert "english_language_requirement" in grouped
    assert selected
    assert selected[0].url == "https://www.tum.de/admissions"
    assert selected[0].source_type == "official_university_page"


def test_language_query_fan_in_excludes_unrelated_admissions_chunks():
    plan = service.QueryPlan(
        required_info=["english_language_requirement"],
        required_fields=["english_language_requirement"],
        keywords=["IELTS", "English proficiency"],
        search_queries=[{"query": "official IELTS", "priority": 1.0}],
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.fau.de/ai",
            title="AI",
            domain="fau.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-28T00:00:00+00:00",
            query="official IELTS",
            pages=[
                service.ExtractedPage(
                    text=(
                        "English proficiency must be proven at CEFR level B2. IELTS and TOEFL "
                        "can be used as English language proof. Tuition fees are listed elsewhere. "
                        "The programme duration is four semesters and transcripts are required."
                    )
                ),
                service.ExtractedPage(
                    text=(
                        "Tuition fee is 1500 EUR. GPA, transcript evaluation, GRE, documents, "
                        "and programme duration are described on this page."
                    )
                ),
            ],
        )
    ]

    grouped = service.group_and_rank_evidence(extracted, plan)
    selected = service.fan_in_evidence(grouped)

    assert selected
    assert all("tuition fee is 1500" not in chunk.text.lower() for chunk in selected)
    assert any("english proficiency" in chunk.text.lower() for chunk in selected)


def test_language_evidence_span_removes_gre_and_document_sentences():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.fau.de/education/degree-programme/artificial-intelligence",
            title="MSc Artificial Intelligence admission requirements",
            domain="fau.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text=(
                        "MSc Artificial Intelligence requires English proficiency at CEFR level B2. "
                        "GRE is required for some applicants. A motivation letter and transcripts "
                        "must be submitted."
                    )
                )
            ],
        )
    ]
    debug = {}

    grouped = service.group_and_rank_evidence(extracted, plan, debug_collector=debug)
    selected = service.fan_in_evidence(grouped)

    assert selected
    evidence_text = " ".join(chunk.text for chunk in selected).lower()
    assert "english proficiency" in evidence_text
    assert "cefr level b2" in evidence_text
    assert "gre" not in evidence_text
    assert "motivation letter" not in evidence_text
    assert "transcripts" not in evidence_text
    assert service._chunk_debug_payload(selected[0])["span_selected_from_chunk"] is True


def test_numeric_ielts_field_remains_missing_without_band_score():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.fau.de/education/degree-programme/artificial-intelligence",
            title="MSc Artificial Intelligence language requirements",
            domain="fau.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text="MSc Artificial Intelligence accepts IELTS as proof of English proficiency."
                )
            ],
        )
    ]

    grouped = service.group_and_rank_evidence(extracted, plan)

    assert "ielts_score" in plan.required_fields
    assert "ielts_score" in service._fields_not_verified(plan, grouped)


def test_table_row_isolation_ignores_neighboring_program_scores():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.fau.de/language-requirements.pdf",
            title="Language requirements",
            domain="fau.de",
            source_type="official_university_pdf",
            document_type="pdf",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    page_number=3,
                    text=(
                        "[Table 1 Row 1] Autonomy Technologies M.Sc. | IELTS 6.0 | TOEFL 80 "
                        "[Table 1 Row 2] Artificial Intelligence M.Sc. | English CEFR B2 "
                        "[Table 1 Row 3] Advanced Optical Technologies M.Sc. | IELTS 5.0"
                    ),
                )
            ],
        )
    ]

    grouped = service.group_and_rank_evidence(extracted, plan)
    selected = service.fan_in_evidence(grouped)
    evidence_text = " ".join(chunk.text for chunk in selected).lower()

    assert "artificial intelligence" in evidence_text
    assert "cefr b2" in evidence_text
    assert "ielts 6.0" not in evidence_text
    assert "ielts 5.0" not in evidence_text
    debug_payload = [service._chunk_debug_payload(chunk) for chunk in selected]
    assert any(item["target_program_row_found"] for item in debug_payload)
    assert any(item["neighboring_rows_ignored"] for item in debug_payload)


def test_field_level_confidence_reports_missing_numeric_ielts():
    plan = service.QueryPlan(
        required_info=["english_language_requirement"],
        required_fields=["english_language_requirement"],
    )
    grouped = {
        "english_language_requirement": [
            service.EvidenceChunk(
                text="English proficiency must be proven at CEFR level B2.",
                url="https://www.fau.de/ai",
                title="AI",
                domain="fau.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-04-28T00:00:00+00:00",
                query="q",
                score=0.9,
                section="english_language_requirement",
                scoring={"source_quality": 0.95},
            )
        ]
    }

    confidence = service._field_level_confidence("IELTS requirement for FAU AI", plan, grouped)

    assert confidence["English B2 requirement"] == "high"
    assert confidence["numeric IELTS score"] == "not verified"
    assert confidence["per-section IELTS score"] == "not verified"


@pytest.mark.asyncio
async def test_analyze_query_uses_bedrock_without_unsupported_max_tokens(monkeypatch):
    captured = {}

    class _Message:
        content = (
            '{"university":"FAU","university_short":"FAU","program":"MSc Artificial Intelligence",'
            '"country":"Germany","intent":"language_requirement_lookup",'
            '"required_fields":["english_language_requirement","ielts_score"],'
            '"optional_fields":["toefl_score"],"excluded_fields":["tuition_fee"],'
            '"search_queries":[{"query":"site:fau.de \\"MSc Artificial Intelligence\\" IELTS","type":"official_page","priority":1.0}],'
            '"priority_sources":["fau.de"]}'
        )

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            if "max_tokens" in kwargs:
                raise TypeError("unexpected max_tokens")
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    import app.infra.bedrock_chat_client as bedrock_chat_client

    monkeypatch.setattr(bedrock_chat_client, "client", _Client())

    plan = await service.analyze_query(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )

    assert "max_tokens" not in captured
    assert plan.decomposition_fallback_used is False
    assert plan.university_short == "FAU"
    assert plan.required_fields == ["english_language_requirement", "ielts_score"]


@pytest.mark.asyncio
async def test_research_uses_clean_original_question_for_polluted_keyword_suffix(monkeypatch):
    async def _fake_analyze(query):
        plan = service._fallback_plan(query)
        plan.decomposition_fallback_used = False
        return plan

    async def _fake_retrieve(plan, query, debug_collector=None):
        return [], [], 0, {"tier_used": "none", "resolved_official_domains": []}

    monkeypatch.setattr(service, "analyze_query", _fake_analyze)
    monkeypatch.setattr(service, "execute_tiered_retrieval", _fake_retrieve)

    result = await service.research_university_question(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU? "
        "IELTS TOEFL language requirement IELTS TOEFL language requirement"
    )

    assert result.query == "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    assert result.debug_info["original_question"] == result.query


def test_fallback_plan_detects_deadline_intent_fields_and_exclusions():
    plan = service._fallback_plan(
        "What is the application deadline for MSc Data Science at University of Mannheim?"
    )

    assert plan.intent == "deadline_lookup"
    assert "application_deadline" in plan.required_fields
    assert "intake_or_semester" in plan.optional_fields
    assert "application_process" in plan.excluded_fields
    assert "ielts_score" in plan.excluded_fields
    assert "tuition_fee" in plan.excluded_fields


def test_fallback_plan_detects_document_intent_fields():
    plan = service._fallback_plan(
        "What documents are required for international students applying to TU Munich MSc Informatics?"
    )

    assert plan.intent == "document_requirement_lookup"
    assert "required_application_documents" in plan.required_fields
    assert "language_proof" in plan.required_fields
    assert "degree_transcript_requirements" in plan.required_fields
    assert "aps_requirement" in plan.optional_fields
    assert "tuition_fee" in plan.excluded_fields


def test_query_validation_repairs_stale_university_queries():
    question = "What is the application deadline for MSc Data Science at University of Mannheim?"
    plan = service._normalize_plan(
        {
            "university": "Technical University of Munich",
            "university_short": "TUM",
            "program": "MSc Informatics",
            "intent": "deadline_lookup",
            "required_fields": ["application_deadline"],
            "search_queries": [
                {
                    "query": "TU Munich MSc Informatics application deadline site:tum.de",
                    "type": "official_page",
                    "priority": 1.0,
                }
            ],
            "priority_sources": ["tum.de"],
        },
        question,
    )

    status = service.validate_and_repair_search_queries(plan, question)

    assert plan.university == "University of Mannheim"
    assert plan.program == "Data Science"
    assert status["regenerated"] is True
    assert status["initial_rejected_queries"][0]["reason"] == "query_mentions_different_university"
    assert all("tum" not in item["query"].lower() for item in plan.search_queries)
    assert any("mannheim" in item["query"].lower() for item in plan.search_queries)


def test_resolves_tum_and_mannheim_official_domains():
    tum = service._fallback_plan(
        "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    )
    mannheim = service._fallback_plan(
        "Does MSc Data Science at University of Mannheim accept TOEFL instead of IELTS?"
    )

    assert service.resolve_official_domains(tum) == ["tum.de", "cit.tum.de", "campus.tum.de"]
    assert service.resolve_official_domains(mannheim) == ["uni-mannheim.de"]


def test_resolves_fau_official_root_and_subdomains():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )

    domains = service.resolve_official_domains(plan)

    assert "fau.de" in domains
    assert "fau.eu" in domains
    assert "ai.study.fau.eu" in domains
    assert "www.ai.study.fau.eu" in domains
    assert "informatik.studium.fau.de" in domains


def test_official_subdomain_queries_use_broadened_language_terms():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )

    queries = service._official_site_queries(
        plan,
        "q",
        ["ai.study.fau.eu"],
        retry=True,
        query_limit=6,
    )
    query_text = " ".join(item["query"] for item in queries)

    assert "site:ai.study.fau.eu" in query_text
    assert any(
        term in query_text for term in ["language proficiency", "CEFR B2", "Application FAQ"]
    )
    assert all(item["include_domains"] == ["ai.study.fau.eu"] for item in queries)


def test_domain_filter_allows_official_subdomains_but_rejects_research_pages():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    rows = [
        {
            "query": 'site:fau.de "Artificial Intelligence" "language"',
            "type": "tier1_official",
            "priority": 1.0,
            "results": [
                {
                    "title": "Reading comprehension with AI",
                    "link": "https://cris.fau.de/projects/331545191?lang=en_GB",
                    "snippet": "language learning with AI",
                },
                {
                    "title": "MSc Artificial Intelligence admissions",
                    "link": "https://www.ai.study.fau.eu/admissions/",
                    "snippet": "English language requirements B2",
                },
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )

    assert [service._domain(item["url"]) for item in selected] == ["ai.study.fau.eu"]
    assert any(
        item["reason"]
        in {
            "non_student_or_research_page",
            "rejected_page_type:research_page",
            "weak_program_match_on_unconfigured_official_subdomain",
        }
        for item in debug["skipped_urls"]
    )


def test_fau_language_retrieval_rejects_bachelor_module_pdf_and_research_pages():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    rows = [
        {
            "query": 'site:fau.de "MSc Artificial Intelligence" "language"',
            "type": "tier1_official",
            "priority": 1.0,
            "results": [
                {
                    "title": "[PDF] Bachelor of Science Artificial Intelligence",
                    "link": "https://tf.fau.de/AI_BA/module-handbook.pdf",
                    "snippet": "Teaching and examination language english PO-Version workload",
                },
                {
                    "title": "AI-Tools",
                    "link": "https://fau.de/en/searching/ai-tools",
                    "snippet": "AI tools for literature searching",
                },
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )

    assert selected == []
    reasons = {item["reason"] for item in debug["skipped_urls"]}
    assert "wrong_degree_level" in reasons or "module_or_curriculum_pdf_not_requested" in reasons
    assert "non_student_or_research_page" in reasons


def test_fau_language_retrieval_accepts_master_ai_program_language_pages():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    rows = [
        {
            "query": 'site:ai.study.fau.eu "MSc Artificial Intelligence" "CEFR B2"',
            "type": "tier1_retry",
            "priority": 1.0,
            "results": [
                {
                    "title": "Language Proficiency",
                    "link": (
                        "https://www.ai.study.fau.eu/prospective-students/"
                        "living-studying-in-germany/language-proficiency"
                    ),
                    "snippet": "English language proficiency CEFR B2 for admission.",
                },
                {
                    "title": "Application Master AI",
                    "link": "https://www.ai.study.fau.eu/prospective-students/master-ai/application-master",
                    "snippet": "Master AI application information and English certificate requirements.",
                },
                {
                    "title": "Master Programme AI",
                    "link": "https://www.ai.study.fau.eu/prospective-students/master-ai/master-programme-ai",
                    "snippet": "Profile of the Artificial Intelligence Master programme.",
                },
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1b_subdomains",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )
    selected_urls = {item["url"] for item in selected}

    assert (
        "https://www.ai.study.fau.eu/prospective-students/living-studying-in-germany/language-proficiency"
        in selected_urls
    )
    assert "https://www.ai.study.fau.eu/prospective-students/master-ai/application-master" in selected_urls
    assert all(item["page_type"] != "profile_page" for item in selected)


def test_fau_language_retrieval_rejects_bachelor_ai_path_for_master_query():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    rows = [
        {
            "query": 'site:ai.study.fau.eu "MSc Artificial Intelligence" "IELTS"',
            "type": "tier1_retry",
            "priority": 1.0,
            "results": [
                {
                    "title": "Application FAQs - Artificial Intelligence (B.Sc./M.Sc.)",
                    "link": "https://www.ai.study.fau.eu/prospective-students/bachelor-ai/application-faq",
                    "snippet": "Questions regarding language proficiency. English B2 CEFR.",
                },
                {
                    "title": "Application - Artificial Intelligence (B.Sc./M.Sc.)",
                    "link": "https://www.ai.study.fau.eu/prospective-students/master-ai/application-master",
                    "snippet": "Master AI application information and English certificate requirements.",
                },
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1b_subdomains",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )

    assert [item["url"] for item in selected] == [
        "https://www.ai.study.fau.eu/prospective-students/master-ai/application-master"
    ]
    assert any(
        item["url"].endswith("/bachelor-ai/application-faq")
        and item["reason"] == "wrong_degree_level"
        for item in debug["skipped_urls"]
    )


def test_fau_ai_page_types_are_specific_for_language_lookup():
    assert (
        service.classify_page_type(
            "https://www.ai.study.fau.eu/prospective-students/living-studying-in-germany/language-proficiency",
            "Language Proficiency",
            "English language proficiency CEFR B2",
        )
        == "language_requirement_page"
    )
    assert (
        service.classify_page_type(
            "https://www.ai.study.fau.eu/prospective-students/master-ai/faq",
            "FAQ - Artificial Intelligence",
            "Questions regarding language proficiency and admission",
        )
        == "program_faq_page"
    )
    assert (
        service.classify_page_type(
            "https://www.ai.study.fau.eu/prospective-students/master-ai/application-master",
            "Application - Artificial Intelligence",
            "Master AI application information and English certificate requirements",
        )
        == "program_application_page"
    )


def test_language_candidate_ranking_prefers_master_ai_pages_over_bachelor_faq():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    rows = [
        {
            "query": 'site:ai.study.fau.eu "MSc Artificial Intelligence" "IELTS"',
            "type": "tier1_retry",
            "priority": 1.0,
            "results": [
                {
                    "title": "Application FAQs - Artificial Intelligence (B.Sc./M.Sc.)",
                    "link": "https://www.ai.study.fau.eu/prospective-students/bachelor-ai/application-faq",
                    "snippet": "Questions regarding language proficiency. English B2 CEFR.",
                },
                {
                    "title": "FAQ - Artificial Intelligence",
                    "link": "https://www.ai.study.fau.eu/prospective-students/master-ai/faq",
                    "snippet": "Master AI FAQ. IELTS or TOEFL and English language proficiency.",
                },
                {
                    "title": "Application - Artificial Intelligence",
                    "link": "https://www.ai.study.fau.eu/prospective-students/master-ai/application-master",
                    "snippet": "Master AI application information and English certificate requirements.",
                },
                {
                    "title": "Language Proficiency",
                    "link": (
                        "https://www.ai.study.fau.eu/prospective-students/"
                        "living-studying-in-germany/language-proficiency"
                    ),
                    "snippet": "Master students need English language proficiency at least CEFR B2.",
                },
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1b_subdomains",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )
    urls = [item["url"] for item in selected]

    assert "https://www.ai.study.fau.eu/prospective-students/bachelor-ai/application-faq" not in urls
    assert urls[0].endswith("/language-proficiency")
    assert any(url.endswith("/master-ai/faq") for url in urls)
    assert any(url.endswith("/master-ai/application-master") for url in urls)
    assert debug["source_scores"]
    assert all("path_degree_signal" in item for item in debug["source_scores"])


def test_evidence_filter_rejects_bachelor_path_for_master_query():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.ai.study.fau.eu/prospective-students/bachelor-ai/application-faq",
            title="Application FAQs - Artificial Intelligence (B.Sc./M.Sc.)",
            domain="ai.study.fau.eu",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text="Questions regarding language proficiency require English at CEFR B2."
                )
            ],
        )
    ]
    debug = {}

    grouped = service.group_and_rank_evidence(extracted, plan, debug_collector=debug)
    selected = service.fan_in_evidence(grouped)

    assert selected == []
    assert any(
        item["reason"] == "wrong_degree_level" for item in debug["excluded_evidence_chunks"]
    )


def test_evidence_filter_allows_bachelor_path_only_for_explicit_master_span():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.ai.study.fau.eu/prospective-students/bachelor-ai/application-faq",
            title="Application FAQs - Artificial Intelligence (B.Sc./M.Sc.)",
            domain="ai.study.fau.eu",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text="For Master students, English language proficiency at CEFR B2 is required."
                )
            ],
        )
    ]

    grouped = service.group_and_rank_evidence(extracted, plan)
    selected = service.fan_in_evidence(grouped)

    assert selected
    assert selected[0].scoring["selected_span_degree_signal"] == "master"


def test_fau_language_retrieval_rejects_other_program_studium_subdomain():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    rows = [
        {
            "query": 'site:fau.de "MSc Artificial Intelligence" "English language requirement"',
            "type": "tier1_official",
            "priority": 1.0,
            "results": [
                {
                    "title": "Standards of Decision-Making Across Cultures FAQ",
                    "link": "https://www.sdac.studium.fau.de/faqs",
                    "snippet": "English language proficiency B2 for the master's program.",
                }
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1a_root",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )

    assert selected == []
    assert any(
        item["reason"] == "unknown_program_specific_subdomain"
        for item in debug["skipped_urls"]
    )


def test_fau_language_retrieval_rejects_weak_other_program_official_subdomain():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    rows = [
        {
            "query": 'site:fau.de "Master Artificial Intelligence" "English language requirements"',
            "type": "tier1_official",
            "priority": 1.0,
            "results": [
                {
                    "title": "Application and Admission - Advanced Materials and Processes",
                    "link": "https://www.map.tf.fau.de/prospective-students/application-admission",
                    "snippet": "Master's Program Advanced Materials and Processes. English language tests.",
                }
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1a_root",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )

    assert selected == []
    assert any(
        item["reason"] == "weak_program_match_on_unconfigured_official_subdomain"
        for item in debug["skipped_urls"]
    )


def test_fau_language_retrieval_rejects_outgoing_exchange_page_before_subdomain_retry():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    rows = [
        {
            "query": 'site:fau.de "MSc Artificial Intelligence" "Sprachnachweis"',
            "type": "tier1_official",
            "priority": 1.0,
            "results": [
                {
                    "title": "FAUexchange - Partnerhochschulen | FAU",
                    "link": (
                        "https://www.fau.de/studium/studienorganisation/wege-ins-ausland/"
                        "studieren-im-ausland/direktaustausch-fauexchange/"
                        "fauexchange-partnerhochschulen"
                    ),
                    "snippet": (
                        "Bei der Bewerbung an der Partnerhochschule muss als "
                        "Sprachnachweis der TOEFL eingereicht werden. Artificial "
                        "Intelligence in Biomedical Engineering."
                    ),
                }
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1a_root",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )

    assert selected == []
    assert any(
        item["reason"] == "outgoing_exchange_page_not_relevant"
        for item in debug["skipped_urls"]
    )


def test_outgoing_exchange_page_type_is_hard_rejected():
    page_type = service.classify_page_type(
        "https://www.fau.de/studium/studieren-im-ausland/fauexchange-partnerhochschulen",
        "FAUexchange - Partnerhochschulen",
        "Sprachnachweis TOEFL for partner university applications",
    )

    assert page_type == "outgoing_exchange_page"
    assert page_type in service.HARD_REJECT_PAGE_TYPES


@pytest.mark.asyncio
async def test_tiered_retrieval_continues_from_weak_root_candidate_to_subdomain(monkeypatch):
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    calls = []

    async def _fake_execute_search_queries(current_plan):
        tier_queries = list(current_plan.search_queries)
        calls.extend(tier_queries)
        if any("site:ai.study.fau.eu" in item["query"] for item in tier_queries):
            return (
                [
                    {
                        "query": tier_queries[0]["query"],
                        "type": "tier1_retry",
                        "priority": 1.0,
                        "results": [
                            {
                                "title": "Application Master AI",
                                "link": (
                                    "https://www.ai.study.fau.eu/prospective-students/"
                                    "master-ai/application-master"
                                ),
                                "snippet": (
                                    "Master AI application information and English "
                                    "certificate requirements CEFR B2."
                                ),
                            }
                        ],
                    }
                ],
                len(tier_queries),
            )
        return (
            [
                {
                    "query": tier_queries[0]["query"] if tier_queries else "",
                    "type": "tier1_official",
                    "priority": 1.0,
                    "results": [
                        {
                            "title": "Language requirements",
                            "link": "https://www.fau.de/studium/sprachnachweis",
                            "snippet": (
                                "Sprachnachweis TOEFL. Artificial Intelligence in "
                                "Biomedical Engineering."
                            ),
                        }
                    ],
                }
            ],
            len(tier_queries),
        )

    monkeypatch.setattr(service, "execute_search_queries", _fake_execute_search_queries)
    debug = {}

    _raw, selected, _calls, tier_debug = await service.execute_tiered_retrieval(
        plan, plan.user_intent, debug_collector=debug
    )

    assert selected
    assert selected[0]["url"].endswith("/master-ai/application-master")
    assert tier_debug["tier_used"] == "tier1b_subdomains"
    assert tier_debug["weak_candidates_deferred"]
    assert any("site:ai.study.fau.eu" in item["query"] for item in calls)


def test_tier1_hard_domain_filter_rejects_random_us_universities_for_tum():
    plan = service._fallback_plan(
        "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    )
    rows = [
        {
            "query": 'site:tum.de "Informatics" "application period"',
            "type": "tier1_official",
            "priority": 1.0,
            "results": [
                {
                    "title": "Informatics deadlines",
                    "link": "https://www.ncsu.edu/graduate/catalog/informatics",
                    "snippet": "application deadline",
                },
                {
                    "title": "Master Informatics application period",
                    "link": "https://cit.tum.de/en/cit/studies/degree-programs/master-informatics/",
                    "snippet": "winter semester application period",
                },
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )

    assert [
        item["domain"] if "domain" in item else service._domain(item["url"]) for item in selected
    ] == ["cit.tum.de"]
    assert any(
        item["reason"] == "outside_target_university_domains" for item in debug["skipped_urls"]
    )


def test_tier1_rejects_daad_but_tier2_allows_it():
    plan = service._fallback_plan(
        "Does MSc Data Science at University of Mannheim accept TOEFL instead of IELTS?"
    )
    rows = [
        {
            "query": 'site:uni-mannheim.de "Data Science" TOEFL',
            "type": "tier1_official",
            "priority": 1.0,
            "results": [
                {
                    "title": "University of Mannheim Data Science TOEFL",
                    "link": "https://www2.daad.de/deutschland/studienangebote/international-programmes/data-science",
                    "snippet": "TOEFL IELTS University of Mannheim Data Science",
                }
            ],
        }
    ]

    tier1 = service.select_and_deduplicate_urls(
        rows,
        plan,
        tier="tier1",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=False,
        allow_third_party=False,
    )
    tier2 = service.select_and_deduplicate_urls(
        rows,
        plan,
        tier="tier2",
        allowed_domains=service.resolve_official_domains(plan),
        allow_secondary=True,
        allow_third_party=False,
    )

    assert tier1 == []
    assert tier2 and tier2[0]["source_type"] == "daad"


@pytest.mark.asyncio
async def test_zero_candidate_recovery_searches_official_subdomains_before_daad(monkeypatch):
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    calls = []

    async def _fake_execute_search_queries(current_plan):
        tier_queries = list(current_plan.search_queries)
        calls.extend(tier_queries)
        if any("site:ai.study.fau.eu" in item["query"] for item in tier_queries):
            return (
                [
                    {
                        "query": tier_queries[0]["query"],
                        "type": "tier1_retry",
                        "priority": 1.0,
                        "results": [
                            {
                                "title": "MSc Artificial Intelligence application FAQ",
                                "link": "https://ai.study.fau.eu/application-faq/",
                                "snippet": "Artificial Intelligence Master English language proficiency CEFR B2",
                            }
                        ],
                    }
                ],
                len(tier_queries),
            )
        return (
            [
                {
                    "query": tier_queries[0]["query"] if tier_queries else "",
                    "type": "tier1_official",
                    "priority": 1.0,
                    "results": [
                        {
                            "title": "AI tools",
                            "link": "https://fau.de/en/searching/ai-tools",
                            "snippet": "library AI tools",
                        }
                    ],
                }
            ],
            len(tier_queries),
        )

    monkeypatch.setattr(service, "execute_search_queries", _fake_execute_search_queries)
    debug = {}

    _raw, selected, _calls, tier_debug = await service.execute_tiered_retrieval(
        plan, plan.user_intent, debug_collector=debug
    )

    assert selected
    assert selected[0]["url"] == "https://ai.study.fau.eu/application-faq"
    assert tier_debug["tier_used"] == "tier1b_subdomains"
    assert not any("daad.de" in item["query"] for item in calls)
    assert tier_debug["zero_candidate_recovery"]


def test_official_site_queries_are_site_restricted_and_field_specific():
    plan = service._fallback_plan(
        "Does University of Mannheim charge tuition fees for non-EU students in MSc Data Science?"
    )

    queries = service._official_site_queries(plan, "q", service.resolve_official_domains(plan))

    assert queries
    assert all("site:uni-mannheim.de" in item["query"] for item in queries)
    assert any(
        "tuition" in item["query"].lower() or "non-eu" in item["query"].lower() for item in queries
    )


def test_fau_msc_ai_language_queries_are_exact_not_weak_generic_terms():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )

    queries = service._official_site_queries(plan, "q", service.resolve_official_domains(plan))
    query_texts = [item["query"] for item in queries]

    assert 'site:fau.de "MSc Artificial Intelligence" "IELTS"' in query_texts
    assert any("Sprachnachweis" in item for item in query_texts)
    assert all('"artificial intelligence" "english"' not in item.lower() for item in query_texts)
    assert all('"artificial intelligence" "language"' not in item.lower() for item in query_texts)


def test_application_portal_intent_uses_portal_fields_and_exclusions():
    plan = service._fallback_plan("Where do I apply for TU Munich MSc Informatics?")

    assert plan.intent in {"application_portal_lookup", "application_process_lookup"}
    assert "application_process" in plan.required_fields
    assert "tuition_fee" in plan.excluded_fields
    assert "curriculum_modules" in plan.excluded_fields


def test_multi_program_discovery_intent_uses_limited_shortlist_fields():
    plan = service._fallback_plan(
        "Suggest German AI master's programs where IELTS 6.5 is likely accepted."
    )

    assert plan.intent == "multi_program_discovery"
    assert plan.query_mode == "discovery_lookup"
    assert "program_shortlist" in plan.required_fields
    assert "english_language_requirement" in plan.required_fields
    assert "tuition_fee" in plan.excluded_fields
    assert len(plan.search_queries) <= service.QUERY_MODE_LIMITS["discovery_lookup"]["max_queries"]


def test_fast_lookup_uses_tight_query_and_url_limits():
    plan = service._fallback_plan(
        "Does MSc Data Science at University of Mannheim accept TOEFL instead of IELTS?"
    )

    assert plan.query_mode == "fast_lookup"
    queries = service._official_site_queries(plan, "q", service.resolve_official_domains(plan))
    assert len(queries) <= service.QUERY_MODE_LIMITS["fast_lookup"]["max_queries"]


def test_page_type_classification_and_url_score_reject_research_pages():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    accepted_type = service.classify_page_type(
        "https://www.ai.study.fau.eu/admissions/",
        "MSc Artificial Intelligence admissions",
        "English language requirements B2",
    )
    rejected_type = service.classify_page_type(
        "https://cris.fau.de/projects/331545191",
        "Reading comprehension of the 17 UN goals with AI",
        "language learning with artificial intelligence",
    )

    assert accepted_type in {"admissions_page", "language_requirement_page"}
    assert rejected_type == "research_page"
    assert service._page_type_rejection_reason(rejected_type, plan) == (
        "rejected_page_type:research_page"
    )


def test_deadline_evidence_filtering_excludes_unrelated_language_and_tuition():
    plan = service._fallback_plan(
        "What is the application deadline for MSc Data Science at University of Mannheim?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.uni-mannheim.de/apply",
            title="Apply",
            domain="uni-mannheim.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-28T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text=(
                        "The application deadline for the winter semester is 31 May "
                        "for international applicants."
                    )
                ),
                service.ExtractedPage(
                    text=(
                        "IELTS 6.5 is accepted. Tuition fee information and GPA are "
                        "described elsewhere."
                    )
                ),
            ],
        )
    ]
    debug = {}

    grouped = service.group_and_rank_evidence(extracted, plan, debug_collector=debug)
    selected = service.fan_in_evidence(grouped)

    assert selected
    assert all(
        chunk.field
        in {
            "application_deadline",
            "intake_or_semester",
            "applicant_category",
            "application_process",
        }
        for chunk in selected
    )
    assert all("ielts 6.5" not in chunk.text.lower() for chunk in selected)
    assert debug["excluded_evidence_chunks"]


def test_application_portal_gate_excludes_language_and_tuition_chunks():
    plan = service._fallback_plan("Where do I apply for TU Munich MSc Informatics?")
    extracted = [
        service.ExtractedContent(
            url="https://www.tum.de/apply",
            title="Apply",
            domain="tum.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text="Applications are submitted through TUMonline using the official application portal."
                ),
                service.ExtractedPage(
                    text="IELTS, GPA, tuition fees, and curriculum modules are described elsewhere."
                ),
            ],
        )
    ]

    grouped = service.group_and_rank_evidence(extracted, plan)
    selected = service.fan_in_evidence(grouped)

    assert selected
    assert all(chunk.field == "application_process" for chunk in selected)
    assert all("tuition" not in chunk.text.lower() for chunk in selected)


def test_url_path_relevance_penalizes_unrelated_exchange_and_news_pages():
    plan = service._fallback_plan(
        "What is the application deadline for MSc Data Science at University of Mannheim?"
    )
    rows = [
        {
            "query": plan.search_queries[0]["query"],
            "type": "official_page",
            "priority": 1.0,
            "results": [
                {
                    "title": "Outgoing exchange news",
                    "link": "https://www.uni-mannheim.de/news/outgoing-exchange-event",
                    "snippet": "student exchange event",
                },
                {
                    "title": "MSc Data Science application requirements",
                    "link": "https://www.uni-mannheim.de/en/academics/programs/msc-data-science/application",
                    "snippet": "application deadline and admission requirements",
                },
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(rows, plan, debug_collector=debug)

    assert selected[0]["url"].endswith("/msc-data-science/application")
    assert any(
        item["reason"] == "url_path_or_title_irrelevant_to_target_program"
        for item in debug["skipped_urls"]
    )


def test_selected_evidence_debug_includes_field_mapping_metadata():
    chunk = service.EvidenceChunk(
        text="The application deadline is 31 May.",
        url="https://www.uni-mannheim.de/apply",
        title="Apply",
        domain="uni-mannheim.de",
        source_type="official_university_page",
        document_type="html",
        retrieved_at="2026-04-28T00:00:00+00:00",
        query="deadline",
        score=0.9,
        section="application_deadline",
        field="application_deadline",
        support_level="direct",
        selection_reason="matched keywords for application_deadline",
    )

    payload = service._chunk_debug_payload(chunk)

    assert payload["field"] == "application_deadline"
    assert payload["support_level"] == "direct"
    assert payload["selection_reason"] == "matched keywords for application_deadline"


def test_strong_official_program_page_is_accepted_before_snippet_field_match():
    plan = service._fallback_plan(
        "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    )
    rows = [
        {
            "query": "site:cit.tum.de MSc Informatics deadline",
            "type": "official_page",
            "priority": 1.0,
            "results": [
                {
                    "title": "Master Informatics",
                    "link": "https://cit.tum.de/en/cit/studies/degree-programs/master-informatics/",
                    "snippet": "Information about the degree program, structure, and study profile.",
                }
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1a_root",
        allowed_domains=["tum.de", "cit.tum.de"],
        allow_secondary=False,
        allow_third_party=False,
    )

    assert selected
    assert selected[0]["strong_official_candidate"] is True
    assert selected[0]["fetched_before_field_match"] is True
    assert selected[0]["page_type"] == "program_page"
    assert not any(item["reason"] == "weak_field_relevance" for item in debug["skipped_urls"])


def test_tum_master_informatics_program_page_is_not_rejected_as_curriculum():
    plan = service._normalize_plan(
        {
            "detected_intent": "deadline_lookup",
            "university": "Technical University of Munich",
            "program": "Informatics",
            "degree_level": "master",
            "required_fields": [
                "winter_semester_application_deadline",
                "deadline_date",
                "applicable_semester",
            ],
            "optional_fields": [],
            "excluded_fields": ["ielts_score"],
            "answer_shape": "short_paragraph",
            "search_queries": [
                {
                    "query": 'site:cit.tum.de "MSc Informatics" "application period"',
                    "type": "tier1_official",
                    "priority": 1.0,
                }
            ],
            "priority_sources": ["tum.de", "cit.tum.de"],
        },
        "When is the winter semester application deadline for MSc Informatics at TU Munich?",
    )
    rows = [
        {
            "query": 'site:cit.tum.de "MSc Informatics" "application period"',
            "type": "tier1_official",
            "priority": 1.0,
            "results": [
                {
                    "title": "Master Informatics",
                    "link": "https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
                    "snippet": "Information about the degree program, application, curriculum, and study profile.",
                }
            ],
        }
    ]
    debug = {}

    selected = service.select_and_deduplicate_urls(
        rows,
        plan,
        debug_collector=debug,
        tier="tier1b_subdomains",
        allowed_domains=["tum.de", "cit.tum.de"],
        allow_secondary=False,
        allow_third_party=False,
    )

    assert plan.required_fields == ["application_deadline", "intake_or_semester"]
    assert service.classify_page_type(
        "https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
        "Master Informatics",
        "Information about the degree program, application, curriculum, and study profile.",
    ) == "program_page"
    assert selected
    assert selected[0]["url"].endswith("/master-informatics")
    assert selected[0]["strong_official_candidate"] is True


@pytest.mark.asyncio
async def test_deadline_answer_shape_is_concise_and_excludes_unrelated_fields():
    plan = service._fallback_plan(
        "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    )
    chunk = service.EvidenceChunk(
        text=(
            "For the master's program Informatics, the application period for the winter "
            "semester is 01 February - 31 May. The application period for the summer "
            "semester is 01 October - 30 November."
        ),
        url="https://cit.tum.de/en/cit/studies/degree-programs/master-informatics/",
        title="Master Informatics",
        domain="cit.tum.de",
        source_type="official_university_page",
        document_type="html",
        retrieved_at="2026-04-29T00:00:00+00:00",
        query="q",
        score=0.95,
        section="application_deadline",
        field="application_deadline",
        support_level="direct",
        evidence_scope="program_specific",
        scoring={"source_quality": 0.95},
    )

    answer = await service.generate_answer(
        "When is the winter semester application deadline for MSc Informatics at TU Munich?",
        [chunk],
        plan,
    )

    assert "01 February to 31 May" in answer
    assert "cit.tum.de" in answer
    forbidden = [
        "IELTS",
        "TOEFL",
        "GPA",
        "ECTS",
        "language requirement",
        "tuition",
        "documents",
        "not verified",
    ]
    assert all(term.lower() not in answer.lower() for term in forbidden)


@pytest.mark.asyncio
async def test_deadline_evidence_is_normalized_and_deduplicated_before_answer():
    query = "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    plan = service._fallback_plan(query)
    raw = (
        "### Application Periods Winter semester: 01 February - 31 May "
        "Summer semester: 01 October - 30 November ### Application Portal"
    )
    grouped = {
        "application_deadline": [
            service.EvidenceChunk(
                text=raw,
                url="https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
                title="Master Informatics",
                domain="cit.tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.95,
                section="application_deadline",
                field="application_deadline",
                support_level="direct",
                evidence_scope="program_specific",
                scoring={"source_quality": 0.95},
            )
        ],
        "intake_or_semester": [
            service.EvidenceChunk(
                text=raw,
                url="https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
                title="Master Informatics",
                domain="cit.tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.9,
                section="intake_or_semester",
                field="intake_or_semester",
                support_level="direct",
                evidence_scope="program_specific",
                scoring={"source_quality": 0.95},
            )
        ],
    }

    packet, selected = service.build_evidence_packet(grouped, plan)
    answer = await service.generate_answer(query, selected, plan, evidence_packet=packet)

    assert packet["field_completeness_status"] == "complete"
    assert packet["deduplicated_fields"] == {"application_deadline": ["intake_or_semester"]}
    assert "01 February to 31 May" in answer
    assert "###" not in answer
    assert answer.count("01 February") == 1
    assert "IELTS" not in answer
    assert "not verified" not in answer.lower()


@pytest.mark.asyncio
async def test_deadline_answer_formats_visa_recommended_and_compulsory_dates():
    query = "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    plan = service._fallback_plan(query)
    text = (
        "For the winter semester: February 1 – March 31: recommended application deadline "
        "for applicants who need a visa. February 1 – May 31: compulsory application deadline."
    )
    grouped = {
        "application_deadline": [
            service.EvidenceChunk(
                text=text,
                url="https://www.tum.de/en/studies/degree-programs/detail/informatics-master-of-science-msc",
                title="Informatics Master",
                domain="tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.95,
                section="application_deadline",
                field="application_deadline",
                support_level="direct",
                evidence_scope="program_specific",
                scoring={"source_quality": 0.95},
            )
        ]
    }

    packet, selected = service.build_evidence_packet(grouped, plan)
    answer = await service.generate_answer(query, selected, plan, evidence_packet=packet)

    normalized = packet["answered_fields"]["application_deadline"]["normalized_values"]
    assert normalized["visa_recommended_deadline"] == "31 March"
    assert normalized["compulsory_deadline"] == "31 May"
    assert "31 March" in answer
    assert "31 May" in answer
    assert "compulsory deadline" in answer


@pytest.mark.asyncio
async def test_document_answer_reports_online_upload_when_checklist_missing():
    query = "What documents are required for international students applying to TU Munich MSc Informatics?"
    plan = service._fallback_plan(query)
    grouped = {
        "required_application_documents": [
            service.EvidenceChunk(
                text=(
                    "Applications are submitted through the online application portal. "
                    "Please upload your documents in the online application."
                ),
                url="https://www.cit.tum.de/en/cit/studies/prospective-students/application",
                title="Application",
                domain="cit.tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.84,
                section="required_application_documents",
                field="required_application_documents",
                support_level="indirect",
                evidence_scope="admissions_general",
                scoring={"source_quality": 0.95},
            )
        ]
    }

    packet, selected = service.build_evidence_packet(grouped, plan)
    answer = await service.generate_answer(query, selected, plan, evidence_packet=packet)

    assert "official page confirms the application is online" in answer
    assert "does not show the full checklist" in answer
    assert "https://www.cit.tum.de/en/cit/studies/prospective-students/application" in answer
    assert "not verified" not in answer.lower()


def test_program_specific_indirect_source_beats_faculty_general_direct_source():
    program_chunk = service.EvidenceChunk(
        text="MSc Data Science applicants need English language proficiency.",
        url="https://www.uni-mannheim.de/en/academics/programs/msc-data-science/",
        title="MSc Data Science",
        domain="uni-mannheim.de",
        source_type="official_university_page",
        document_type="html",
        retrieved_at="2026-05-01T00:00:00+00:00",
        query="q",
        score=0.7,
        section="english_language_requirement",
        field="english_language_requirement",
        support_level="indirect",
        evidence_scope="program_specific",
        scoring={"source_quality": 0.95},
    )
    general_chunk = service.EvidenceChunk(
        text="Faculty language requirements for teacher education mention TOEFL.",
        url="https://www.uni-mannheim.de/en/academics/language-requirements/",
        title="Faculty Language Requirements",
        domain="uni-mannheim.de",
        source_type="official_university_page",
        document_type="html",
        retrieved_at="2026-05-01T00:00:00+00:00",
        query="q",
        score=0.95,
        section="english_language_requirement",
        field="english_language_requirement",
        support_level="direct",
        evidence_scope="faculty_general",
        scoring={"source_quality": 0.95},
    )

    assert service._field_priority_chunk([general_chunk, program_chunk]) is program_chunk


def test_evidence_scope_rejects_wrong_program_chunks():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.fau.de/education/degree-programme/informatics",
            title="MSc Informatics language requirements",
            domain="fau.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text="For MSc Informatics, English proficiency B2 and IELTS 6.5 are accepted."
                )
            ],
        )
    ]
    debug = {}

    grouped = service.group_and_rank_evidence(extracted, plan, debug_collector=debug)
    selected = service.fan_in_evidence(grouped)

    assert not selected
    assert any(item["reason"] == "wrong_program" for item in debug["excluded_evidence_chunks"])


def test_program_specific_language_evidence_carries_scope():
    plan = service._fallback_plan(
        "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.fau.de/education/degree-programme/artificial-intelligence",
            title="MSc Artificial Intelligence",
            domain="fau.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text="MSc Artificial Intelligence requires English proficiency at CEFR level B2."
                )
            ],
        )
    ]

    grouped = service.group_and_rank_evidence(extracted, plan)
    selected = service.fan_in_evidence(grouped)

    assert selected
    assert selected[0].evidence_scope == "program_specific"
    assert service._chunk_debug_payload(selected[0])["evidence_scope"] == "program_specific"


def test_deadline_filter_rejects_malformed_old_pdf_dates():
    plan = service._fallback_plan(
        "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.tum.de/module-handbook.pdf",
            title="Module Handbook MSc Informatics",
            domain="tum.de",
            source_type="official_university_pdf",
            document_type="pdf",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text="WiSe 2004/20 examination regulation version date. Module handbook curriculum table."
                )
            ],
        )
    ]
    debug = {}

    grouped = service.group_and_rank_evidence(extracted, plan, debug_collector=debug)
    selected = service.fan_in_evidence(grouped)

    assert not selected
    assert debug["excluded_evidence_chunks"]


def test_tuition_gate_excludes_deadline_and_language_chunks():
    plan = service._fallback_plan(
        "Does University of Mannheim charge tuition fees for non-EU students in MSc Data Science?"
    )
    extracted = [
        service.ExtractedContent(
            url="https://www.uni-mannheim.de/en/studies/fees",
            title="Tuition fees",
            domain="uni-mannheim.de",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-29T00:00:00+00:00",
            query=plan.search_queries[0]["query"],
            pages=[
                service.ExtractedPage(
                    text="Non-EU students are charged tuition fees of 1,500 EUR per semester."
                ),
                service.ExtractedPage(
                    text="The application deadline is 31 May. IELTS and language requirements are elsewhere."
                ),
            ],
        )
    ]

    grouped = service.group_and_rank_evidence(extracted, plan)
    selected = service.fan_in_evidence(grouped)

    assert selected
    assert any(chunk.field == "tuition_fee" for chunk in selected)
    assert all("application deadline" not in chunk.text.lower() for chunk in selected)


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_returns_compatibility_dict(monkeypatch):
    result = service.ResearchResult(
        query="q",
        answer="answer [E1]",
        evidence_chunks=[
            service.EvidenceChunk(
                text="Official application deadline evidence.",
                url="https://www.tum.de/deadline",
                title="Deadline",
                domain="tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-04-28T00:00:00+00:00",
                query="deadline",
                score=0.9,
                section="application_deadline",
            )
        ],
        query_plan=service.QueryPlan(required_info=["application_deadline"]),
        debug_info={"final_confidence": 0.9, "fields_not_verified": []},
    )

    async def _fake_research(_query):
        return result

    monkeypatch.setattr(service, "research_university_question", _fake_research)

    payload = await service.aretrieve_web_chunks("q")

    assert payload["results"][0]["metadata"]["source_type"] == "official_university_page"
    assert payload["coverage_ledger"][0]["status"] == "found"
    assert payload["web_retrieval_verified"] is True
    assert "debug" not in payload


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_saves_debug_artifact_without_returning_it(
    monkeypatch, tmp_path
):
    result = service.ResearchResult(
        query="q",
        answer="answer [E1]",
        evidence_chunks=[],
        query_plan=service.QueryPlan(required_info=["application_deadline"]),
        debug_info={
            "query_decomposition": {"university": "TUM"},
            "generated_search_queries": [{"query": "tum deadline"}],
            "raw_search_results": [],
            "skipped_urls": [],
            "final_confidence": 0.2,
            "fields_not_verified": ["application_deadline"],
        },
    )

    async def _fake_research(_query):
        return result

    monkeypatch.setenv("UNIGRAPH_DEBUG_DIR", str(tmp_path))
    monkeypatch.setattr(service, "research_university_question", _fake_research)

    normal_payload = await service.aretrieve_web_chunks("q")
    debug_payload = await service.aretrieve_web_chunks("q", debug=True)

    assert "debug" not in normal_payload
    assert "debug" not in debug_payload
    artifacts = list(tmp_path.glob("*.json"))
    assert len(artifacts) == 1
    assert '"university": "TUM"' in artifacts[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_language_requirement_span_maps_to_answered_field_and_concise_answer():
    query = "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    plan = service._fallback_plan(query)
    extracted = [
        service.ExtractedContent(
            url="https://www.ai.study.fau.eu/prospective-students/living-studying-in-germany/language-proficiency/",
            title="Language Proficiency - FAU AI",
            domain="ai.study.fau.eu",
            source_type="official_university_page",
            document_type="html",
            source_quality=0.95,
            retrieved_at="2026-04-30T00:00:00+00:00",
            query="site:ai.study.fau.eu Master Artificial Intelligence language proficiency",
            pages=[
                service.ExtractedPage(
                    text=(
                        "For successful participation in the Master's degree program, an "
                        "English language proficiency certificate of at least B2 level "
                        "(CEFR) is required."
                    )
                )
            ],
        )
    ]

    debug = {}
    grouped = service.group_and_rank_evidence(extracted, plan, debug_collector=debug)
    selected = service.fan_in_evidence(grouped)
    answered, _partial, missing = service._field_statuses(plan, grouped)
    answer = await service.generate_answer(query, selected, plan, {}, missing)

    assert "english_language_requirement" in answered
    assert "ielts_score" in missing
    assert "CEFR" in answer
    assert "B2" in answer
    assert "specific IELTS band score" in answer
    forbidden = ["GPA", "ECTS", "Application Deadline", "Application Portal", "tuition"]
    assert all(term.lower() not in answer.lower() for term in forbidden)
    assert debug["selected_answer_spans"][0]["field"] == "english_language_requirement"
    assert debug["selected_answer_spans"][0]["support_level"] == "direct"


@pytest.mark.asyncio
async def test_unigraph_coverage_ledger_uses_evidence_text_for_language(monkeypatch):
    result = service.ResearchResult(
        query="What are the IELTS requirements for MSc Artificial Intelligence at FAU?",
        answer="answer",
        evidence_chunks=[
            service.EvidenceChunk(
                text="English language proficiency certificate of at least B2 level (CEFR) is required.",
                url="https://www.ai.study.fau.eu/prospective-students/living-studying-in-germany/language-proficiency/",
                title="Language Proficiency",
                domain="ai.study.fau.eu",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-04-30T00:00:00+00:00",
                query="q",
                score=0.9,
                section="english_language_requirement",
                field="english_language_requirement",
                support_level="direct",
            )
        ],
        query_plan=service.QueryPlan(
            required_info=["english_language_requirement", "ielts_score"],
            required_fields=["english_language_requirement", "ielts_score"],
        ),
        debug_info={
            "final_confidence": 0.8,
            "fields_not_verified": ["ielts_score"],
            "field_level_confidence": {"english_language_requirement": "high"},
        },
    )

    async def _fake_research(_query):
        return result

    monkeypatch.setattr(service, "research_university_question", _fake_research)

    payload = await service.aretrieve_web_chunks(result.query)

    found_row = payload["coverage_ledger"][0]
    assert found_row["field"] == "english_language_requirement"
    assert found_row["status"] == "found"
    assert "B2 level" in found_row["value"]
    assert "B2 level" in found_row["evidence_snippet"]


@pytest.mark.asyncio
async def test_unigraph_coverage_ledger_uses_normalized_values_not_raw_chunks(monkeypatch):
    result = service.ResearchResult(
        query="When is the winter semester application deadline for MSc Informatics at TU Munich?",
        answer="For MSc Informatics at TUM, the winter semester application period is 01 February to 31 May.",
        evidence_chunks=[
            service.EvidenceChunk(
                text="### Application Periods Winter semester: 01 February - 31 May Summer semester: 01 October - 30 November ### [...]",
                url="https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
                title="Master Informatics",
                domain="cit.tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.95,
                section="application_deadline",
                field="application_deadline",
                support_level="direct",
            )
        ],
        query_plan=service.QueryPlan(
            intent="deadline_lookup",
            required_info=["application_deadline"],
            required_fields=["application_deadline"],
        ),
        debug_info={
            "final_confidence": 0.9,
            "fields_not_verified": [],
            "answered_fields": {
                "application_deadline": {
                    "value": "01 February - 31 May",
                    "normalized_values": {
                        "winter_semester_application_period": "01 February - 31 May"
                    },
                    "source_url": "https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
                }
            },
        },
    )

    async def _fake_research(_query):
        return result

    monkeypatch.setattr(service, "research_university_question", _fake_research)

    payload = await service.aretrieve_web_chunks(result.query)

    value = payload["coverage_ledger"][0]["value"]
    assert "winter_semester_application_period: 01 February - 31 May" == value
    assert "###" not in value
    assert "[...]" not in value


def test_structured_renderer_for_narrow_language_question_avoids_universal_template():
    answer = llm_service._build_structured_field_evidence_answer(
        {
            "safe_user_prompt": "What are the IELTS requirements for MSc Artificial Intelligence at FAU?",
            "coverage_ledger": [
                {
                    "id": "english_language_requirement",
                    "field": "english_language_requirement",
                    "label": "English Language Requirement",
                    "status": "found",
                    "value": "English language proficiency certificate of at least B2 level (CEFR) is required.",
                    "source_url": "https://www.ai.study.fau.eu/prospective-students/living-studying-in-germany/language-proficiency/",
                },
                {
                    "id": "ielts_score",
                    "field": "ielts_score",
                    "label": "IELTS Score",
                    "status": "missing",
                    "value": "",
                    "source_url": "",
                },
            ],
            "evidence_urls": [
                "https://www.ai.study.fau.eu/prospective-students/living-studying-in-germany/language-proficiency/"
            ],
        }
    )

    assert "English requirement:" in answer
    assert "B2 level" in answer
    assert "specific IELTS band score" in answer
    forbidden = [
        "GPA and ECTS",
        "Application Deadline",
        "Application Portal",
        "Missing or Uncertain",
        "Not verified from official sources",
        "Verified from selected evidence",
        "###",
    ]
    assert all(term not in answer for term in forbidden)


@pytest.mark.asyncio
async def test_final_deadline_answer_guardrails_block_raw_and_internal_wording():
    query = "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    plan = service._fallback_plan(query)
    raw = "### Application Periods Winter semester: 01 February - 31 May"
    grouped = {
        "application_deadline": [
            service.EvidenceChunk(
                text=raw,
                url="https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
                title="Master Informatics",
                domain="cit.tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.95,
                section="application_deadline",
                field="application_deadline",
                support_level="direct",
                evidence_scope="program_specific",
                scoring={"source_quality": 0.95},
            )
        ]
    }

    packet, selected = service.build_evidence_packet(grouped, plan)
    answer = await service.generate_answer(query, selected, plan, evidence_packet=packet)

    assert "\n" not in answer
    assert "01 February to 31 May" in answer
    forbidden = [
        "Not verified from official sources",
        "Verified from selected evidence",
        "###",
        "answered_fields",
        "missing_fields",
        "Intake:",
    ]
    assert all(term not in answer for term in forbidden)


@pytest.mark.asyncio
async def test_final_language_answer_uses_natural_missing_ielts_wording_only():
    query = "What are the IELTS requirements for MSc Artificial Intelligence at FAU?"
    plan = service._fallback_plan(query)
    grouped = {
        "english_language_requirement": [
            service.EvidenceChunk(
                text="Master's students need English proficiency at minimum CEFR B2.",
                url="https://www.ai.study.fau.eu/prospective-students/language-proficiency/",
                title="Language Proficiency",
                domain="ai.study.fau.eu",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.95,
                section="english_language_requirement",
                field="english_language_requirement",
                support_level="direct",
                evidence_scope="program_specific",
                scoring={"source_quality": 0.95},
            )
        ]
    }

    packet, selected = service.build_evidence_packet(grouped, plan)
    answer = await service.generate_answer(query, selected, plan, evidence_packet=packet)

    assert "CEFR B2" in answer
    assert "specific IELTS band score" in answer
    forbidden = [
        "Not verified from official sources",
        "Verified from selected evidence",
        "deadline",
        "GPA",
        "tuition",
        "documents",
        "portal",
    ]
    assert all(term.lower() not in answer.lower() for term in forbidden)


@pytest.mark.asyncio
async def test_final_llm_answer_regenerates_once_when_bad_output_detected(monkeypatch):
    class _Completions:
        def __init__(self):
            self.calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _FakeResponse(
                    "Application deadline: ### Application Periods Winter semester: 01 February - 31 May"
                )
            return _FakeResponse("For MSc Informatics at TUM, the winter semester application period is 01 February to 31 May.")

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    fake_client = _Client()
    monkeypatch.setattr(service.settings.bedrock, "primary_model_id", "test-model")
    monkeypatch.setattr("app.infra.bedrock_chat_client.client", fake_client)
    plan = service.QueryPlan(
        intent="general_program_overview",
        university_short="TUM",
        program="MSc Informatics",
        required_fields=["program_overview"],
        answer_shape="overview",
    )
    chunk = service.EvidenceChunk(
        text="MSc Informatics is offered at TUM.",
        url="https://www.tum.de/en/studies/degree-programs/detail/informatics-master-of-science-msc",
        title="Informatics",
        domain="tum.de",
        source_type="official_university_page",
        document_type="html",
        retrieved_at="2026-05-01T00:00:00+00:00",
        query="q",
        score=0.9,
        section="program_overview",
        field="program_overview",
        support_level="direct",
    )

    answer = await service.generate_answer(
        "Give me a brief overview of MSc Informatics at TUM.",
        [chunk],
        plan,
    )

    assert fake_client.chat.completions.calls == 2
    assert "01 February to 31 May" in answer
    assert "###" not in answer
    assert "Application deadline:" not in answer


@pytest.mark.asyncio
async def test_narrow_deadline_uses_llm_synthesis_and_clean_packet(monkeypatch):
    query = "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    plan = service._fallback_plan(query)
    raw = (
        "### Application Periods Winter semester: 01 February - 31 May "
        "Summer semester: 01 October - 30 November ### [...]"
    )
    grouped = {
        "application_deadline": [
            service.EvidenceChunk(
                text=raw,
                url="https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
                title="Master Informatics",
                domain="cit.tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.95,
                section="application_deadline",
                field="application_deadline",
                support_level="direct",
                evidence_scope="program_specific",
                scoring={"source_quality": 0.95},
            )
        ]
    }
    packet, selected = service.build_evidence_packet(grouped, plan)
    captured_prompts: list[str] = []

    class _Completions:
        async def create(self, **kwargs):
            captured_prompts.append(str(kwargs["messages"][1]["content"]))
            return _FakeResponse(
                "For MSc Informatics at TUM, the official CIT page lists the winter semester application period as 01 February to 31 May. Source: https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics"
            )

    class _Client:
        def __init__(self):
            self.chat = type("_Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(service.settings.bedrock, "primary_model_id", "test-model")
    monkeypatch.setattr("app.infra.bedrock_chat_client.client", _Client())
    metadata: dict[str, object] = {}

    answer = await service.generate_answer(
        query,
        selected,
        plan,
        evidence_packet=packet,
        answer_metadata=metadata,
    )

    assert metadata["final_answer_source"] == "llm_synthesis"
    assert metadata["final_prompt_used"] is True
    assert metadata["raw_span_rendered"] is False
    assert "01 February to 31 May" in answer
    assert captured_prompts
    assert "###" not in captured_prompts[0]
    assert "[...]" not in captured_prompts[0]
    assert "evidence_span" not in captured_prompts[0]


@pytest.mark.asyncio
async def test_bad_regenerated_answer_uses_safe_normalized_fallback(monkeypatch):
    query = "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    plan = service._fallback_plan(query)
    raw = "### Application Periods Winter semester: 01 February - 31 May"
    grouped = {
        "application_deadline": [
            service.EvidenceChunk(
                text=raw,
                url="https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
                title="Master Informatics",
                domain="cit.tum.de",
                source_type="official_university_page",
                document_type="html",
                retrieved_at="2026-05-01T00:00:00+00:00",
                query="q",
                score=0.95,
                section="application_deadline",
                field="application_deadline",
                support_level="direct",
                evidence_scope="program_specific",
                scoring={"source_quality": 0.95},
            )
        ]
    }
    packet, selected = service.build_evidence_packet(grouped, plan)

    class _Completions:
        async def create(self, **_kwargs):
            return _FakeResponse(
                "Application deadline: ### Application Periods Winter semester: 01 February - 31 May"
            )

    class _Client:
        def __init__(self):
            self.chat = type("_Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(service.settings.bedrock, "primary_model_id", "test-model")
    monkeypatch.setattr("app.infra.bedrock_chat_client.client", _Client())
    metadata: dict[str, object] = {}

    answer = await service.generate_answer(
        query,
        selected,
        plan,
        evidence_packet=packet,
        answer_metadata=metadata,
    )

    assert metadata["final_answer_source"] == "fallback_builder"
    assert metadata["final_prompt_used"] is True
    assert metadata["raw_span_rendered"] is True
    assert "01 February to 31 May" in answer
    assert "###" not in answer
    assert "Application deadline:" not in answer


@pytest.mark.asyncio
async def test_research_debug_records_final_answer_source_flags(monkeypatch):
    query = "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    plan = service._fallback_plan(query)
    chunk = service.EvidenceChunk(
        text="Winter semester: 01 February - 31 May",
        url="https://www.cit.tum.de/en/cit/studies/degree-programs/master-informatics",
        title="Master Informatics",
        domain="cit.tum.de",
        source_type="official_university_page",
        document_type="html",
        retrieved_at="2026-05-01T00:00:00+00:00",
        query="q",
        score=0.95,
        section="application_deadline",
        field="application_deadline",
        support_level="direct",
        evidence_scope="program_specific",
        scoring={"source_quality": 0.95},
    )

    async def _fake_analyze(_query):
        return plan

    async def _fake_execute(*_args, **_kwargs):
        return [], [], 0, {}

    async def _fake_extract(*_args, **_kwargs):
        return []

    def _fake_group(*_args, **_kwargs):
        return {"application_deadline": [chunk]}

    async def _fake_generate(*_args, answer_metadata=None, **_kwargs):
        answer_metadata.update(
            {
                "final_answer_source": "llm_synthesis",
                "final_prompt_used": True,
                "raw_span_rendered": False,
            }
        )
        return "For MSc Informatics at TUM, the winter semester application period is 01 February to 31 May."

    monkeypatch.setattr(service, "analyze_query", _fake_analyze)
    monkeypatch.setattr(service, "execute_tiered_retrieval", _fake_execute)
    monkeypatch.setattr(service, "extract_all_contents", _fake_extract)
    monkeypatch.setattr(service, "group_and_rank_evidence", _fake_group)
    monkeypatch.setattr(service, "generate_answer", _fake_generate)

    result = await service.research_university_question(query)

    assert result.debug_info["final_answer_source"] == "llm_synthesis"
    assert result.debug_info["final_prompt_used"] is True
    assert result.debug_info["raw_span_rendered"] is False
