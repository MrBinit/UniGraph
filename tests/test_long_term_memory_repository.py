from datetime import datetime, timezone

from app.repositories import long_term_memory_repository as repo
from app.schemas.long_term_memory_schema import LongTermMemoryWrite


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        if "RETURNING" in sql:
            self._result = {
                "id": "abc-123",
                "user_id": "user-1",
                "memory_key": "pref:country",
                "memory_type": "user_preference",
                "content": "Prefers US universities",
                "source": "summary",
                "confidence": 0.9,
                "metadata": {"country": "US"},
                "embedding": None,
                "created_at": datetime(2026, 2, 28, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 2, 28, tzinfo=timezone.utc),
            }
        else:
            self._result = [
                {
                    "id": "abc-123",
                    "user_id": "user-1",
                    "memory_key": "pref:country",
                    "memory_type": "user_preference",
                    "content": "Prefers US universities",
                    "source": "summary",
                    "confidence": 0.9,
                    "metadata": {"country": "US"},
                    "embedding": None,
                    "created_at": datetime(2026, 2, 28, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 2, 28, tzinfo=timezone.utc),
                }
            ]

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._result or []


class FakeConnection:
    def __init__(self, pool):
        self.pool = pool
        self.executed = []
        self.committed = False

    def __enter__(self):
        self.pool.connections.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.committed = True


class FakePool:
    def __init__(self):
        self.connections = []

    def connection(self):
        return FakeConnection(self)


def test_upsert_long_term_memory_matches_real_table(monkeypatch):
    fake_pool = FakePool()
    monkeypatch.setattr(repo, "get_postgres_pool", lambda: fake_pool)

    record = LongTermMemoryWrite(
        user_id="user-1",
        memory_key="pref:country",
        memory_type="user_preference",
        content="Prefers US universities",
        source="summary",
        confidence=0.9,
        metadata={"country": "US"},
    )

    saved = repo.upsert_long_term_memory(record, embedding=[0.1, 0.2])
    conn = fake_pool.connections[0]
    sql, params = conn.executed[0]

    assert "INSERT INTO" in sql
    assert "user_id" in sql
    assert "memory_key" in sql
    assert "ON CONFLICT (user_id, memory_key)" in sql
    assert params[0] == "user-1"
    assert params[1] == "pref:country"
    assert params[7] == "[0.10000000,0.20000000]"
    assert conn.committed is True
    assert saved.user_id == "user-1"


def test_list_long_term_memories_filters_by_user(monkeypatch):
    fake_pool = FakePool()
    monkeypatch.setattr(repo, "get_postgres_pool", lambda: fake_pool)

    records = repo.list_long_term_memories("user-1", limit=5)
    conn = fake_pool.connections[0]
    sql, params = conn.executed[0]

    assert "WHERE user_id = %s" in sql
    assert params == ("user-1", 5)
    assert len(records) == 1
    assert records[0].memory_key == "pref:country"


def test_search_long_term_memories_orders_by_vector_distance(monkeypatch):
    fake_pool = FakePool()
    monkeypatch.setattr(repo, "get_postgres_pool", lambda: fake_pool)

    records = repo.search_long_term_memories(
        "user-1",
        embedding=[0.3, 0.4],
        limit=3,
        memory_types=["user_preference"],
    )
    conn = fake_pool.connections[0]
    sql, params = conn.executed[0]

    assert "embedding IS NOT NULL" in sql
    assert "memory_type = ANY(%s)" in sql
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert params == ("user-1", ["user_preference"], "[0.30000000,0.40000000]", 3)
    assert len(records) == 1
