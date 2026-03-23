from app.repositories import university_metadata_repository as repo
from app.schemas.university_metadata_schema import UniversityMetadataIngestionPayload


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        normalized = " ".join(sql.split())
        mappings = self.conn.mappings

        if "INSERT INTO unigraph.universities" in normalized:
            mappings["universities"][params[0]] = "u-1"
            self._rows = []
            return
        if "INSERT INTO unigraph.departments" in normalized:
            mappings["departments"][params[0]] = "d-1"
            self._rows = []
            return
        if "INSERT INTO unigraph.programs" in normalized:
            mappings["programs"][params[0]] = "p-1"
            self._rows = []
            return
        if "INSERT INTO unigraph.professors" in normalized:
            mappings["professors"][params[0]] = "pr-1"
            self._rows = []
            return
        if "INSERT INTO unigraph.labs" in normalized:
            mappings["labs"][params[0]] = "l-1"
            self._rows = []
            return
        if "INSERT INTO unigraph.courses" in normalized:
            mappings["courses"][params[0]] = "c-1"
            self._rows = []
            return

        if "FROM unigraph.universities" in normalized and "university_key" in normalized:
            self._rows = [
                {"university_key": key, "id": mappings["universities"][key]}
                for key in params[0]
                if key in mappings["universities"]
            ]
            return
        if "FROM unigraph.departments" in normalized and "department_key" in normalized:
            self._rows = [
                {"department_key": key, "id": mappings["departments"][key]}
                for key in params[0]
                if key in mappings["departments"]
            ]
            return
        if "FROM unigraph.programs" in normalized and "program_key" in normalized:
            self._rows = [
                {"program_key": key, "id": mappings["programs"][key]}
                for key in params[0]
                if key in mappings["programs"]
            ]
            return
        if "FROM unigraph.professors" in normalized and "professor_key" in normalized:
            self._rows = [
                {"professor_key": key, "id": mappings["professors"][key]}
                for key in params[0]
                if key in mappings["professors"]
            ]
            return
        if "FROM unigraph.labs" in normalized and "lab_key" in normalized:
            self._rows = [
                {"lab_key": key, "id": mappings["labs"][key]}
                for key in params[0]
                if key in mappings["labs"]
            ]
            return
        if "FROM unigraph.courses" in normalized and "course_key" in normalized:
            self._rows = [
                {"course_key": key, "id": mappings["courses"][key]}
                for key in params[0]
                if key in mappings["courses"]
            ]
            return

        self._rows = []

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, pool):
        self.pool = pool
        self.executed = []
        self.committed = False
        self.mappings = {
            "universities": {},
            "departments": {},
            "programs": {},
            "professors": {},
            "labs": {},
            "courses": {},
        }

    def __enter__(self):
        self.pool.connections.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.committed = True


class _FakePool:
    def __init__(self):
        self.connections = []

    def connection(self):
        return _FakeConnection(self)


def test_ensure_university_metadata_tables_creates_expected_tables(monkeypatch):
    fake_pool = _FakePool()
    monkeypatch.setattr(repo, "get_postgres_pool", lambda: fake_pool)

    repo.ensure_university_metadata_tables()
    sql = fake_pool.connections[0].executed[0][0]

    assert "CREATE TABLE IF NOT EXISTS unigraph.universities" in sql
    assert "CREATE TABLE IF NOT EXISTS unigraph.programs" in sql
    assert "CREATE TABLE IF NOT EXISTS unigraph.source_records" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_programs_degree_department" in sql
    assert fake_pool.connections[0].committed is True


def test_ingest_university_metadata_payload_processes_all_entities(monkeypatch):
    fake_pool = _FakePool()
    monkeypatch.setattr(repo, "get_postgres_pool", lambda: fake_pool)
    monkeypatch.setattr(repo, "ensure_university_metadata_tables", lambda: None)

    payload = UniversityMetadataIngestionPayload.model_validate(
        {
            "universities": [
                {
                    "university_key": "uni:rtu",
                    "name": "Rheinberg Technical University",
                    "country": "Germany",
                    "city": "Munich",
                }
            ],
            "departments": [
                {
                    "department_key": "dept:cs",
                    "university_key": "uni:rtu",
                    "name": "Computer Science",
                }
            ],
            "programs": [
                {
                    "program_key": "prog:ai-ms",
                    "university_key": "uni:rtu",
                    "department_key": "dept:cs",
                    "program_name": "MSc Artificial Intelligence",
                    "degree_level": "masters",
                }
            ],
            "program_intakes": [{"program_key": "prog:ai-ms", "intake_term": "winter"}],
            "application_routes": [
                {"program_key": "prog:ai-ms", "applicant_type": "international"}
            ],
            "program_requirements": [
                {
                    "program_key": "prog:ai-ms",
                    "requirement_type": "gpa",
                    "requirement_value": "2.5",
                }
            ],
            "language_requirements": [
                {
                    "program_key": "prog:ai-ms",
                    "language": "english",
                    "test_type": "ielts",
                    "min_score": "6.5",
                }
            ],
            "professors": [
                {
                    "professor_key": "prof:max",
                    "university_key": "uni:rtu",
                    "name": "Max Mustermann",
                }
            ],
            "labs": [
                {
                    "lab_key": "lab:ai",
                    "university_key": "uni:rtu",
                    "lab_name": "AI Systems Lab",
                }
            ],
            "courses": [
                {
                    "course_key": "course:ml",
                    "university_key": "uni:rtu",
                    "department_key": "dept:cs",
                    "course_name": "Machine Learning",
                }
            ],
            "program_courses": [{"program_key": "prog:ai-ms", "course_key": "course:ml"}],
            "program_labs": [{"program_key": "prog:ai-ms", "lab_key": "lab:ai"}],
            "program_professors": [{"program_key": "prog:ai-ms", "professor_key": "prof:max"}],
            "professor_labs": [{"professor_key": "prof:max", "lab_key": "lab:ai"}],
            "source_records": [
                {
                    "entity_type": "program",
                    "entity_key": "prog:ai-ms",
                    "source_url": "https://example.edu/programs/ai-ms",
                }
            ],
        }
    )

    result = repo.ingest_university_metadata_payload(payload)
    conn = fake_pool.connections[0]
    sql_blob = "\n".join(entry[0] for entry in conn.executed)

    assert "INSERT INTO unigraph.universities" in sql_blob
    assert "INSERT INTO unigraph.programs" in sql_blob
    assert "INSERT INTO unigraph.program_intakes" in sql_blob
    assert "INSERT INTO unigraph.program_courses" in sql_blob
    assert "INSERT INTO unigraph.source_records" in sql_blob
    assert conn.committed is True
    assert result["universities"] == 1
    assert result["programs"] == 1
    assert result["program_courses"] == 1
