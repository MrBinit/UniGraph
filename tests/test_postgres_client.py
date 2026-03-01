from app.infra import postgres_client


def test_build_postgres_conninfo_uses_yaml_and_env(monkeypatch):
    """Verify conninfo is built from YAML config and an env-sourced password."""
    cfg = postgres_client.settings.postgres
    original = (
        cfg.enabled,
        cfg.host,
        cfg.port,
        cfg.database,
        cfg.username,
        cfg.ssl_mode,
        cfg.connect_timeout_seconds,
        cfg.app_name,
    )

    try:
        cfg.enabled = True
        cfg.host = "db.example.test"
        cfg.port = 5432
        cfg.database = "appdb"
        cfg.username = "appuser"
        cfg.ssl_mode = "require"
        cfg.connect_timeout_seconds = 10
        cfg.app_name = "unigraph"
        monkeypatch.setenv("POSTGRES_PASSWORD", "Test!Pass#1$X^")

        conninfo = postgres_client.build_postgres_conninfo()

        assert "db.example.test:5432/appdb" in conninfo
        assert "sslmode=require" in conninfo
        assert "application_name=unigraph" in conninfo
        assert "Test%21Pass%231%24X%5E" in conninfo
    finally:
        (
            cfg.enabled,
            cfg.host,
            cfg.port,
            cfg.database,
            cfg.username,
            cfg.ssl_mode,
            cfg.connect_timeout_seconds,
            cfg.app_name,
        ) = original


def test_postgres_password_is_required(monkeypatch):
    """Verify building a Postgres connection requires a password in the environment."""
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    try:
        postgres_client._postgres_password()
        raise AssertionError("expected missing password to raise")
    except RuntimeError as exc:
        assert "POSTGRES_PASSWORD" in str(exc)
