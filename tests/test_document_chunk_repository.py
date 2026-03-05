from app.repositories import document_chunk_repository


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, cursor):
        self._cursor = cursor

    def connection(self):
        return _FakeConnection(self._cursor)


def test_vector_index_sql_uses_hnsw_config():
    cfg = document_chunk_repository.settings.postgres
    original = (cfg.vector_index_type, cfg.hnsw_m, cfg.hnsw_ef_construction)
    try:
        cfg.vector_index_type = "hnsw"
        cfg.hnsw_m = 24
        cfg.hnsw_ef_construction = 96
        sql = document_chunk_repository._vector_index_sql(if_not_exists=False)
    finally:
        cfg.vector_index_type, cfg.hnsw_m, cfg.hnsw_ef_construction = original

    assert "CREATE INDEX idx_doc_chunks_embedding" in sql
    assert "USING hnsw" in sql
    assert "m = 24" in sql
    assert "ef_construction = 96" in sql


def test_ingest_embedding_manifest_upserts_chunks(monkeypatch):
    cursor = _FakeCursor()
    monkeypatch.setattr(document_chunk_repository, "get_postgres_pool", lambda: _FakePool(cursor))

    count = document_chunk_repository.ingest_embedding_manifest(
        {
            "chunks": [
                {
                    "chunk_id": "university_1:0000",
                    "chunk_index": 0,
                    "source_file": "university_1.md",
                    "source_path": "/tmp/university_1.md",
                    "content": "Sample embedded chunk",
                    "char_count": 21,
                    "metadata": {"document_id": "university_1", "country": "Germany"},
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        }
    )

    assert count == 1
    assert len(cursor.calls) == 3
    assert "CREATE TABLE IF NOT EXISTS" in cursor.calls[0][0]
    assert "CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding" in cursor.calls[1][0]
    assert "INSERT INTO" in cursor.calls[2][0]
    params = cursor.calls[2][1]
    assert params[0] == "university_1"
    assert params[1] == "university_1:0000"
    assert params[-1] == "[0.10000000,0.20000000,0.30000000]"


def test_resolve_document_chunk_search_strategy_uses_hnsw_without_filters():
    cfg = document_chunk_repository.settings.postgres
    original = cfg.vector_index_type
    try:
        cfg.vector_index_type = "hnsw"
        strategy = document_chunk_repository.resolve_document_chunk_search_strategy(None)
    finally:
        cfg.vector_index_type = original

    assert strategy == "hnsw"


def test_search_document_chunks_uses_ann_query_without_filters(monkeypatch):
    class _SearchCursor(_FakeCursor):
        def fetchall(self):
            return [
                {
                    "chunk_id": "university_1:0002",
                    "document_id": "university_1",
                    "chunk_index": 2,
                    "source_file": "university_1.md",
                    "source_path": "/tmp/university_1.md",
                    "content": "Program Overview for the master's program.",
                    "char_count": 41,
                    "metadata": {"country": "Germany", "entity_type": "program"},
                    "distance": 0.123,
                }
            ]

    cursor = _SearchCursor()
    monkeypatch.setattr(document_chunk_repository, "get_postgres_pool", lambda: _FakePool(cursor))

    results = document_chunk_repository.search_document_chunks(
        embedding=[0.1, 0.2, 0.3],
        limit=3,
    )

    assert len(results) == 1
    assert "ORDER BY embedding <=> %s::vector" in cursor.calls[0][0]
    assert "metadata @> %s::jsonb" not in cursor.calls[0][0]
    params = cursor.calls[0][1]
    assert params[0] == "[0.10000000,0.20000000,0.30000000]"
    assert params[1] == "[0.10000000,0.20000000,0.30000000]"
    assert params[2] == 3
    assert results[0]["distance"] == 0.123


def test_search_document_chunks_uses_filtered_exact_strategy(monkeypatch):
    class _SearchCursor(_FakeCursor):
        def fetchall(self):
            return [
                {
                    "chunk_id": "university_1:0002",
                    "document_id": "university_1",
                    "chunk_index": 2,
                    "source_file": "university_1.md",
                    "source_path": "/tmp/university_1.md",
                    "content": "Program Overview for the master's program.",
                    "char_count": 41,
                    "metadata": {"country": "Germany", "entity_type": "program"},
                    "distance": 0.123,
                }
            ]

    cursor = _SearchCursor()
    monkeypatch.setattr(document_chunk_repository, "get_postgres_pool", lambda: _FakePool(cursor))

    results = document_chunk_repository.search_document_chunks(
        embedding=[0.1, 0.2, 0.3],
        limit=3,
        metadata_filters={"country": "Germany", "entity_type": "program"},
    )

    assert len(results) == 1
    assert "WITH filtered_chunks AS MATERIALIZED" in cursor.calls[0][0]
    assert "metadata @> %s::jsonb" in cursor.calls[0][0]
    assert "ORDER BY embedding <=> %s::vector" in cursor.calls[0][0]
    params = cursor.calls[0][1]
    assert params[0] == '{"country": "Germany", "entity_type": "program"}'
    assert params[1] == "[0.10000000,0.20000000,0.30000000]"
    assert params[2] == "[0.10000000,0.20000000,0.30000000]"
    assert params[3] == 3
    assert results[0]["distance"] == 0.123


def test_rebuild_document_chunk_vector_index_uses_configured_type(monkeypatch):
    cursor = _FakeCursor()
    monkeypatch.setattr(document_chunk_repository, "get_postgres_pool", lambda: _FakePool(cursor))

    cfg = document_chunk_repository.settings.postgres
    original = cfg.vector_index_type
    try:
        cfg.vector_index_type = "hnsw"
        index_type = document_chunk_repository.rebuild_document_chunk_vector_index()
    finally:
        cfg.vector_index_type = original

    assert index_type == "hnsw"
    assert "DROP INDEX IF EXISTS" in cursor.calls[0][0]
    assert "USING hnsw" in cursor.calls[0][0]
