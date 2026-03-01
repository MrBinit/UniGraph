import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from app.core.config import get_settings

load_dotenv()

settings = get_settings()
_POOL = None


def _postgres_password() -> str:
    """Return the Postgres password from the environment or fail fast if missing."""
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is required in the environment for Postgres access.")
    return password


def build_postgres_conninfo() -> str:
    """Build a psycopg-compatible connection string from YAML config and env password."""
    cfg = settings.postgres
    username = quote_plus(cfg.username)
    password = quote_plus(_postgres_password())
    host = cfg.host.strip()
    database = cfg.database.strip()
    ssl_mode = cfg.ssl_mode.strip()
    app_name = quote_plus(cfg.app_name)

    return (
        f"postgresql://{username}:{password}@{host}:{cfg.port}/{database}"
        f"?sslmode={ssl_mode}"
        f"&connect_timeout={cfg.connect_timeout_seconds}"
        f"&application_name={app_name}"
    )


def get_postgres_pool():
    """Return the shared Postgres connection pool, creating it on first use."""
    global _POOL

    if not settings.postgres.enabled:
        raise RuntimeError("Postgres is disabled in config.")

    if _POOL is None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "psycopg and psycopg_pool are required for Postgres. Install requirements first."
            ) from exc
        _POOL = ConnectionPool(
            conninfo=build_postgres_conninfo(),
            min_size=settings.postgres.min_pool_size,
            max_size=settings.postgres.max_pool_size,
            kwargs={"row_factory": dict_row},
        )
    return _POOL


def close_postgres_pool():
    """Close and clear the shared Postgres pool when the process shuts down."""
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def verify_postgres_connection() -> dict:
    """Run a lightweight query to confirm the app can connect to Postgres."""
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    current_database() AS database_name,
                    current_schema() AS schema_name,
                    current_user AS user_name,
                    version() AS server_version
                """
            )
            row = cur.fetchone()
    return row or {}
