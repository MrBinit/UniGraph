from app.infra import postgres_client


def test_build_postgres_conninfo_uses_yaml_and_env(monkeypatch):
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
        cfg.host = "unigraph.cwjweimqmbri.us-east-1.rds.amazonaws.com"
        cfg.port = 5432
        cfg.database = "unigraph"
        cfg.username = "unigraph"
        cfg.ssl_mode = "require"
        cfg.connect_timeout_seconds = 10
        cfg.app_name = "unigraph"
        monkeypatch.setenv("POSTGRES_PASSWORD", "G7k!vP9#rT2$Lm8^")

        conninfo = postgres_client.build_postgres_conninfo()

        assert "unigraph.cwjweimqmbri.us-east-1.rds.amazonaws.com:5432/unigraph" in conninfo
        assert "sslmode=require" in conninfo
        assert "application_name=unigraph" in conninfo
        assert "G7k%21vP9%23rT2%24Lm8%5E" in conninfo
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
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    try:
        postgres_client._postgres_password()
        raise AssertionError("expected missing password to raise")
    except RuntimeError as exc:
        assert "POSTGRES_PASSWORD" in str(exc)
