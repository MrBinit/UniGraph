import json
from app.core.config import get_settings
from app.infra.postgres_client import get_postgres_pool
from app.schemas.university_metadata_schema import UniversityMetadataIngestionPayload

settings = get_settings()


def _schema_name() -> str:
    schema = str(settings.postgres.schema_name).strip()
    return schema or "unigraph"


def _table(table_name: str) -> str:
    return f"{_schema_name()}.{table_name}"


def _as_json(value) -> str:
    return json.dumps(value if isinstance(value, dict) else {})


def _id_map(cur, table_name: str, key_column: str, keys: list[str]) -> dict[str, str]:
    normalized = sorted({str(key).strip() for key in keys if str(key).strip()})
    if not normalized:
        return {}
    cur.execute(
        f"SELECT {key_column}, id FROM {_table(table_name)} WHERE {key_column} = ANY(%s)",
        (normalized,),
    )
    rows = cur.fetchall() or []
    result: dict[str, str] = {}
    for row in rows:
        key = str(row.get(key_column, "")).strip()
        value = str(row.get("id", "")).strip()
        if key and value:
            result[key] = value
    return result


def ensure_university_metadata_tables() -> None:
    """Create normalized university metadata tables and indexes."""
    sql = f"""
        CREATE SCHEMA IF NOT EXISTS {_schema_name()};

        CREATE TABLE IF NOT EXISTS {_table("universities")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            university_key TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            name_local TEXT NOT NULL DEFAULT '',
            country TEXT NOT NULL,
            city TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            website TEXT NOT NULL DEFAULT '',
            established_year INTEGER NULL,
            university_type TEXT NOT NULL DEFAULT '',
            campus_type TEXT NOT NULL DEFAULT '',
            application_portal TEXT NOT NULL DEFAULT '',
            default_language TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {_table("departments")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            department_key TEXT NOT NULL UNIQUE,
            university_id UUID NOT NULL REFERENCES {_table("universities")}(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            website TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {_table("programs")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            program_key TEXT NOT NULL UNIQUE,
            university_id UUID NOT NULL REFERENCES {_table("universities")}(id) ON DELETE CASCADE,
            department_id UUID NULL REFERENCES {_table("departments")}(id) ON DELETE SET NULL,
            program_name TEXT NOT NULL,
            name_local TEXT NOT NULL DEFAULT '',
            degree_level TEXT NOT NULL,
            duration_months INTEGER NULL,
            ects_credits INTEGER NULL,
            tuition_fee DOUBLE PRECISION NULL,
            tuition_currency TEXT NOT NULL DEFAULT '',
            language_primary TEXT NOT NULL DEFAULT '',
            program_url TEXT NOT NULL DEFAULT '',
            admission_type TEXT NOT NULL DEFAULT '',
            study_mode TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_programs_degree_department
        ON {_table("programs")} (degree_level, department_id);

        CREATE TABLE IF NOT EXISTS {_table("program_intakes")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            program_id UUID NOT NULL REFERENCES {_table("programs")}(id) ON DELETE CASCADE,
            intake_term TEXT NOT NULL,
            intake_year INTEGER NULL,
            application_open_date DATE NULL,
            application_deadline DATE NULL,
            priority_deadline DATE NULL,
            document_deadline DATE NULL,
            program_start DATE NULL,
            is_rolling BOOLEAN NOT NULL DEFAULT FALSE,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (program_id, intake_term, COALESCE(intake_year, -1))
        );

        CREATE TABLE IF NOT EXISTS {_table("application_routes")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            program_id UUID NOT NULL REFERENCES {_table("programs")}(id) ON DELETE CASCADE,
            applicant_type TEXT NOT NULL,
            portal_url TEXT NOT NULL DEFAULT '',
            application_fee DOUBLE PRECISION NULL,
            fee_currency TEXT NOT NULL DEFAULT '',
            admission_type TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (program_id, applicant_type)
        );

        CREATE TABLE IF NOT EXISTS {_table("program_requirements")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            program_id UUID NOT NULL REFERENCES {_table("programs")}(id) ON DELETE CASCADE,
            applicant_type TEXT NOT NULL DEFAULT '',
            requirement_type TEXT NOT NULL,
            requirement_value TEXT NOT NULL,
            is_mandatory BOOLEAN NOT NULL DEFAULT TRUE,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {_table("language_requirements")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            program_id UUID NOT NULL REFERENCES {_table("programs")}(id) ON DELETE CASCADE,
            applicant_type TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL,
            test_type TEXT NOT NULL,
            min_score TEXT NOT NULL,
            score_scale TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {_table("professors")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            professor_key TEXT NOT NULL UNIQUE,
            university_id UUID NOT NULL REFERENCES {_table("universities")}(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            department TEXT NOT NULL DEFAULT '',
            research_interests TEXT NOT NULL DEFAULT '',
            email TEXT NOT NULL DEFAULT '',
            website TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {_table("labs")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            lab_key TEXT NOT NULL UNIQUE,
            university_id UUID NOT NULL REFERENCES {_table("universities")}(id) ON DELETE CASCADE,
            lab_name TEXT NOT NULL,
            research_focus TEXT NOT NULL DEFAULT '',
            lab_website TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {_table("courses")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            course_key TEXT NOT NULL UNIQUE,
            university_id UUID NOT NULL REFERENCES {_table("universities")}(id) ON DELETE CASCADE,
            department_id UUID NULL REFERENCES {_table("departments")}(id) ON DELETE SET NULL,
            course_name TEXT NOT NULL,
            course_code TEXT NOT NULL DEFAULT '',
            ects_credits INTEGER NULL,
            language TEXT NOT NULL DEFAULT '',
            level TEXT NOT NULL DEFAULT '',
            course_url TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS {_table("program_courses")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            program_id UUID NOT NULL REFERENCES {_table("programs")}(id) ON DELETE CASCADE,
            course_id UUID NOT NULL REFERENCES {_table("courses")}(id) ON DELETE CASCADE,
            course_type TEXT NOT NULL DEFAULT 'optional',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (program_id, course_id)
        );

        CREATE TABLE IF NOT EXISTS {_table("program_labs")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            program_id UUID NOT NULL REFERENCES {_table("programs")}(id) ON DELETE CASCADE,
            lab_id UUID NOT NULL REFERENCES {_table("labs")}(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (program_id, lab_id)
        );

        CREATE TABLE IF NOT EXISTS {_table("program_professors")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            program_id UUID NOT NULL REFERENCES {_table("programs")}(id) ON DELETE CASCADE,
            professor_id UUID NOT NULL REFERENCES {_table("professors")}(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (program_id, professor_id)
        );

        CREATE TABLE IF NOT EXISTS {_table("professor_labs")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            professor_id UUID NOT NULL REFERENCES {_table("professors")}(id) ON DELETE CASCADE,
            lab_id UUID NOT NULL REFERENCES {_table("labs")}(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (professor_id, lab_id)
        );

        CREATE TABLE IF NOT EXISTS {_table("source_records")} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_type TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_title TEXT NOT NULL DEFAULT '',
            retrieved_at TIMESTAMPTZ NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            extractor_version TEXT NOT NULL DEFAULT '',
            confidence DOUBLE PRECISION NULL,
            raw_snippet TEXT NOT NULL DEFAULT '',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def _new_ingestion_counts() -> dict[str, int]:
    return {
        "universities": 0,
        "departments": 0,
        "programs": 0,
        "program_intakes": 0,
        "application_routes": 0,
        "program_requirements": 0,
        "language_requirements": 0,
        "professors": 0,
        "labs": 0,
        "courses": 0,
        "program_courses": 0,
        "program_labs": 0,
        "program_professors": 0,
        "professor_labs": 0,
        "source_records": 0,
    }


def _university_keys(payload: UniversityMetadataIngestionPayload) -> list[str]:
    return (
        [item.university_key for item in payload.universities]
        + [item.university_key for item in payload.departments]
        + [item.university_key for item in payload.programs]
        + [item.university_key for item in payload.professors]
        + [item.university_key for item in payload.labs]
        + [item.university_key for item in payload.courses]
    )


def _department_keys(payload: UniversityMetadataIngestionPayload) -> list[str]:
    keys = [item.department_key for item in payload.departments]
    keys.extend(item.department_key for item in payload.programs if item.department_key)
    keys.extend(item.department_key for item in payload.courses if item.department_key)
    return keys


def _program_keys(payload: UniversityMetadataIngestionPayload) -> list[str]:
    return (
        [item.program_key for item in payload.programs]
        + [item.program_key for item in payload.program_intakes]
        + [item.program_key for item in payload.application_routes]
        + [item.program_key for item in payload.program_requirements]
        + [item.program_key for item in payload.language_requirements]
        + [item.program_key for item in payload.program_courses]
        + [item.program_key for item in payload.program_labs]
        + [item.program_key for item in payload.program_professors]
    )


def _professor_keys(payload: UniversityMetadataIngestionPayload) -> list[str]:
    return (
        [item.professor_key for item in payload.professors]
        + [item.professor_key for item in payload.program_professors]
        + [item.professor_key for item in payload.professor_labs]
    )


def _lab_keys(payload: UniversityMetadataIngestionPayload) -> list[str]:
    return (
        [item.lab_key for item in payload.labs]
        + [item.lab_key for item in payload.program_labs]
        + [item.lab_key for item in payload.professor_labs]
    )


def _course_keys(payload: UniversityMetadataIngestionPayload) -> list[str]:
    return [item.course_key for item in payload.courses] + [
        item.course_key for item in payload.program_courses
    ]


def _upsert_universities(cur, payload: UniversityMetadataIngestionPayload, counts: dict[str, int]):
    for university in payload.universities:
        cur.execute(
            f"""
            INSERT INTO {_table("universities")} (
                university_key, name, name_local, country, city, location, website,
                established_year, university_type, campus_type, application_portal,
                default_language, description, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (university_key) DO UPDATE SET
                name = EXCLUDED.name,
                name_local = EXCLUDED.name_local,
                country = EXCLUDED.country,
                city = EXCLUDED.city,
                location = EXCLUDED.location,
                website = EXCLUDED.website,
                established_year = EXCLUDED.established_year,
                university_type = EXCLUDED.university_type,
                campus_type = EXCLUDED.campus_type,
                application_portal = EXCLUDED.application_portal,
                default_language = EXCLUDED.default_language,
                description = EXCLUDED.description,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                university.university_key,
                university.name,
                university.name_local,
                university.country,
                university.city,
                university.location,
                university.website,
                university.established_year,
                university.university_type,
                university.campus_type,
                university.application_portal,
                university.default_language,
                university.description,
                _as_json(university.metadata),
            ),
        )
        counts["universities"] += 1


def _upsert_departments(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    university_ids: dict[str, str],
):
    for department in payload.departments:
        university_id = university_ids.get(department.university_key)
        if not university_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("departments")} (
                department_key, university_id, name, website, metadata
            )
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (department_key) DO UPDATE SET
                university_id = EXCLUDED.university_id,
                name = EXCLUDED.name,
                website = EXCLUDED.website,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                department.department_key,
                university_id,
                department.name,
                department.website,
                _as_json(department.metadata),
            ),
        )
        counts["departments"] += 1


def _upsert_programs(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    university_ids: dict[str, str],
    department_ids: dict[str, str],
):
    for program in payload.programs:
        university_id = university_ids.get(program.university_key)
        if not university_id:
            continue
        department_id = (
            department_ids.get(program.department_key) if program.department_key else None
        )
        cur.execute(
            f"""
            INSERT INTO {_table("programs")} (
                program_key, university_id, department_id, program_name, name_local,
                degree_level, duration_months, ects_credits, tuition_fee,
                tuition_currency, language_primary, program_url, admission_type,
                study_mode, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (program_key) DO UPDATE SET
                university_id = EXCLUDED.university_id,
                department_id = EXCLUDED.department_id,
                program_name = EXCLUDED.program_name,
                name_local = EXCLUDED.name_local,
                degree_level = EXCLUDED.degree_level,
                duration_months = EXCLUDED.duration_months,
                ects_credits = EXCLUDED.ects_credits,
                tuition_fee = EXCLUDED.tuition_fee,
                tuition_currency = EXCLUDED.tuition_currency,
                language_primary = EXCLUDED.language_primary,
                program_url = EXCLUDED.program_url,
                admission_type = EXCLUDED.admission_type,
                study_mode = EXCLUDED.study_mode,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                program.program_key,
                university_id,
                department_id,
                program.program_name,
                program.name_local,
                program.degree_level,
                program.duration_months,
                program.ects_credits,
                program.tuition_fee,
                program.tuition_currency,
                program.language_primary,
                program.program_url,
                program.admission_type,
                program.study_mode,
                _as_json(program.metadata),
            ),
        )
        counts["programs"] += 1


def _upsert_professors(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    university_ids: dict[str, str],
):
    for professor in payload.professors:
        university_id = university_ids.get(professor.university_key)
        if not university_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("professors")} (
                professor_key, university_id, name, title, department,
                research_interests, email, website, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (professor_key) DO UPDATE SET
                university_id = EXCLUDED.university_id,
                name = EXCLUDED.name,
                title = EXCLUDED.title,
                department = EXCLUDED.department,
                research_interests = EXCLUDED.research_interests,
                email = EXCLUDED.email,
                website = EXCLUDED.website,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                professor.professor_key,
                university_id,
                professor.name,
                professor.title,
                professor.department,
                professor.research_interests,
                professor.email,
                professor.website,
                _as_json(professor.metadata),
            ),
        )
        counts["professors"] += 1


def _upsert_labs(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    university_ids: dict[str, str],
):
    for lab in payload.labs:
        university_id = university_ids.get(lab.university_key)
        if not university_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("labs")} (
                lab_key, university_id, lab_name, research_focus, lab_website, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (lab_key) DO UPDATE SET
                university_id = EXCLUDED.university_id,
                lab_name = EXCLUDED.lab_name,
                research_focus = EXCLUDED.research_focus,
                lab_website = EXCLUDED.lab_website,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                lab.lab_key,
                university_id,
                lab.lab_name,
                lab.research_focus,
                lab.lab_website,
                _as_json(lab.metadata),
            ),
        )
        counts["labs"] += 1


def _upsert_courses(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    university_ids: dict[str, str],
    department_ids: dict[str, str],
):
    for course in payload.courses:
        university_id = university_ids.get(course.university_key)
        if not university_id:
            continue
        department_id = department_ids.get(course.department_key) if course.department_key else None
        cur.execute(
            f"""
            INSERT INTO {_table("courses")} (
                course_key, university_id, department_id, course_name, course_code,
                ects_credits, language, level, course_url, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (course_key) DO UPDATE SET
                university_id = EXCLUDED.university_id,
                department_id = EXCLUDED.department_id,
                course_name = EXCLUDED.course_name,
                course_code = EXCLUDED.course_code,
                ects_credits = EXCLUDED.ects_credits,
                language = EXCLUDED.language,
                level = EXCLUDED.level,
                course_url = EXCLUDED.course_url,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                course.course_key,
                university_id,
                department_id,
                course.course_name,
                course.course_code,
                course.ects_credits,
                course.language,
                course.level,
                course.course_url,
                _as_json(course.metadata),
            ),
        )
        counts["courses"] += 1


def _insert_program_intakes(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    program_ids: dict[str, str],
):
    for intake in payload.program_intakes:
        program_id = program_ids.get(intake.program_key)
        if not program_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("program_intakes")} (
                program_id, intake_term, intake_year, application_open_date,
                application_deadline, priority_deadline, document_deadline,
                program_start, is_rolling, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT DO NOTHING
            """,
            (
                program_id,
                intake.intake_term,
                intake.intake_year,
                intake.application_open_date,
                intake.application_deadline,
                intake.priority_deadline,
                intake.document_deadline,
                intake.program_start,
                intake.is_rolling,
                _as_json(intake.metadata),
            ),
        )
        counts["program_intakes"] += 1


def _insert_application_routes(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    program_ids: dict[str, str],
):
    for route in payload.application_routes:
        program_id = program_ids.get(route.program_key)
        if not program_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("application_routes")} (
                program_id, applicant_type, portal_url, application_fee,
                fee_currency, admission_type, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT DO NOTHING
            """,
            (
                program_id,
                route.applicant_type,
                route.portal_url,
                route.application_fee,
                route.fee_currency,
                route.admission_type,
                _as_json(route.metadata),
            ),
        )
        counts["application_routes"] += 1


def _insert_program_requirements(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    program_ids: dict[str, str],
):
    for requirement in payload.program_requirements:
        program_id = program_ids.get(requirement.program_key)
        if not program_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("program_requirements")} (
                program_id, applicant_type, requirement_type, requirement_value,
                is_mandatory, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                program_id,
                requirement.applicant_type,
                requirement.requirement_type,
                requirement.requirement_value,
                requirement.is_mandatory,
                _as_json(requirement.metadata),
            ),
        )
        counts["program_requirements"] += 1


def _insert_language_requirements(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    program_ids: dict[str, str],
):
    for language in payload.language_requirements:
        program_id = program_ids.get(language.program_key)
        if not program_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("language_requirements")} (
                program_id, applicant_type, language, test_type,
                min_score, score_scale, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                program_id,
                language.applicant_type,
                language.language,
                language.test_type,
                language.min_score,
                language.score_scale,
                _as_json(language.metadata),
            ),
        )
        counts["language_requirements"] += 1


def _insert_program_courses(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    program_ids: dict[str, str],
    course_ids: dict[str, str],
):
    for link in payload.program_courses:
        program_id = program_ids.get(link.program_key)
        course_id = course_ids.get(link.course_key)
        if not program_id or not course_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("program_courses")} (
                program_id, course_id, course_type
            )
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (program_id, course_id, link.course_type),
        )
        counts["program_courses"] += 1


def _insert_program_labs(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    program_ids: dict[str, str],
    lab_ids: dict[str, str],
):
    for link in payload.program_labs:
        program_id = program_ids.get(link.program_key)
        lab_id = lab_ids.get(link.lab_key)
        if not program_id or not lab_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("program_labs")} (
                program_id, lab_id
            )
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (program_id, lab_id),
        )
        counts["program_labs"] += 1


def _insert_program_professors(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    program_ids: dict[str, str],
    professor_ids: dict[str, str],
):
    for link in payload.program_professors:
        program_id = program_ids.get(link.program_key)
        professor_id = professor_ids.get(link.professor_key)
        if not program_id or not professor_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("program_professors")} (
                program_id, professor_id
            )
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (program_id, professor_id),
        )
        counts["program_professors"] += 1


def _insert_professor_labs(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
    professor_ids: dict[str, str],
    lab_ids: dict[str, str],
):
    for link in payload.professor_labs:
        professor_id = professor_ids.get(link.professor_key)
        lab_id = lab_ids.get(link.lab_key)
        if not professor_id or not lab_id:
            continue
        cur.execute(
            f"""
            INSERT INTO {_table("professor_labs")} (
                professor_id, lab_id
            )
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (professor_id, lab_id),
        )
        counts["professor_labs"] += 1


def _insert_source_records(
    cur,
    payload: UniversityMetadataIngestionPayload,
    counts: dict[str, int],
):
    for record in payload.source_records:
        cur.execute(
            f"""
            INSERT INTO {_table("source_records")} (
                entity_type, entity_key, source_url, source_title, retrieved_at,
                content_hash, extractor_version, confidence, raw_snippet, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                record.entity_type,
                record.entity_key,
                record.source_url,
                record.source_title,
                record.retrieved_at,
                record.content_hash,
                record.extractor_version,
                record.confidence,
                record.raw_snippet,
                _as_json(record.metadata),
            ),
        )
        counts["source_records"] += 1


def ingest_university_metadata_payload(
    payload: UniversityMetadataIngestionPayload,
) -> dict[str, int]:
    """Upsert one normalized metadata payload into Postgres and return row counts."""
    ensure_university_metadata_tables()
    counts = _new_ingestion_counts()

    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            _upsert_universities(cur, payload, counts)
            university_ids = _id_map(
                cur, "universities", "university_key", _university_keys(payload)
            )

            _upsert_departments(cur, payload, counts, university_ids)
            department_ids = _id_map(
                cur, "departments", "department_key", _department_keys(payload)
            )

            _upsert_programs(cur, payload, counts, university_ids, department_ids)
            _upsert_professors(cur, payload, counts, university_ids)
            _upsert_labs(cur, payload, counts, university_ids)
            _upsert_courses(cur, payload, counts, university_ids, department_ids)

            program_ids = _id_map(cur, "programs", "program_key", _program_keys(payload))
            professor_ids = _id_map(cur, "professors", "professor_key", _professor_keys(payload))
            lab_ids = _id_map(cur, "labs", "lab_key", _lab_keys(payload))
            course_ids = _id_map(cur, "courses", "course_key", _course_keys(payload))

            _insert_program_intakes(cur, payload, counts, program_ids)
            _insert_application_routes(cur, payload, counts, program_ids)
            _insert_program_requirements(cur, payload, counts, program_ids)
            _insert_language_requirements(cur, payload, counts, program_ids)
            _insert_program_courses(cur, payload, counts, program_ids, course_ids)
            _insert_program_labs(cur, payload, counts, program_ids, lab_ids)
            _insert_program_professors(cur, payload, counts, program_ids, professor_ids)
            _insert_professor_labs(cur, payload, counts, professor_ids, lab_ids)
            _insert_source_records(cur, payload, counts)

        conn.commit()

    return counts
