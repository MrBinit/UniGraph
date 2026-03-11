import json
import logging
import os
import yaml
from functools import lru_cache
from pathlib import Path
import boto3

from app.core.paths import APP_CONFIG_DIR
from app.schemas.settings_schema import Settings

logger = logging.getLogger(__name__)


def _load_yaml_file(file_path: Path) -> dict:
    """Load one YAML file and enforce a top-level mapping structure."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top-level: {file_path}")
    return data


def _load_aws_secrets_manager_env():
    """Load environment variables from one AWS Secrets Manager JSON secret."""
    secret_id = os.getenv("AWS_SECRETS_MANAGER_SECRET_ID", "").strip()
    if not secret_id:
        return

    region = (
        os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
    )
    client_kwargs = {"region_name": region} if region else {}

    try:
        client = boto3.client("secretsmanager", **client_kwargs)
        response = client.get_secret_value(SecretId=secret_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load AWS Secrets Manager secret '{secret_id}'. {exc}"
        ) from exc

    secret_string = response.get("SecretString", "")
    if not isinstance(secret_string, str) or not secret_string.strip():
        raise RuntimeError(f"AWS secret '{secret_id}' is empty or not a SecretString payload.")

    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"AWS secret '{secret_id}' is not valid JSON. {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"AWS secret '{secret_id}' must be a JSON object.")

    loaded_keys = 0
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip() or value is None:
            continue
        key = key.strip()
        existing = os.getenv(key)
        if existing is not None and existing != "":
            continue
        if isinstance(value, (str, int, float, bool)):
            os.environ[key] = str(value)
            loaded_keys += 1

    logger.info("AWSSecretLoaded | secret_id=%s | loaded_keys=%s", secret_id, loaded_keys)


def _apply_env_overrides(data: dict) -> dict:
    """Apply environment overrides on top of YAML configuration for deployments."""
    config = data

    def _set(path: list[str], env_name: str, cast=str):
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            return

        value = raw
        if cast is bool:
            value = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif cast is int:
            value = int(raw)
        else:
            value = cast(raw)

        node = config
        for key in path[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[path[-1]] = value

    _set(["app", "log_level"], "APP_LOG_LEVEL")
    _set(["app", "docs_enabled"], "APP_DOCS_ENABLED", bool)
    _set(["app", "metrics_json_enabled"], "APP_METRICS_JSON_ENABLED", bool)
    _set(["app", "metrics_json_dir"], "APP_METRICS_JSON_DIR")
    _set(["app", "metrics_dynamodb_enabled"], "APP_METRICS_DYNAMODB_ENABLED", bool)
    _set(["app", "metrics_dynamodb_requests_table"], "APP_METRICS_DYNAMODB_REQUESTS_TABLE")
    _set(["app", "metrics_dynamodb_aggregate_table"], "APP_METRICS_DYNAMODB_AGGREGATE_TABLE")
    _set(["app", "metrics_dynamodb_ttl_days"], "APP_METRICS_DYNAMODB_TTL_DAYS", int)
    _set(["evaluation", "enabled"], "EVALUATION_ENABLED", bool)
    _set(["evaluation", "dynamodb_table"], "EVALUATION_DYNAMODB_TABLE")
    _set(["evaluation", "judge_model_id"], "EVALUATION_JUDGE_MODEL_ID")
    _set(["evaluation", "batch_size"], "EVALUATION_BATCH_SIZE", int)
    _set(["evaluation", "max_items_per_run"], "EVALUATION_MAX_ITEMS_PER_RUN", int)
    _set(["evaluation", "lookback_hours"], "EVALUATION_LOOKBACK_HOURS", int)
    _set(["evaluation", "ttl_days"], "EVALUATION_TTL_DAYS", int)
    _set(["evaluation", "schedule_enabled"], "EVALUATION_SCHEDULE_ENABLED", bool)
    _set(["evaluation", "schedule_interval_hours"], "EVALUATION_SCHEDULE_INTERVAL_HOURS", int)

    _set(["security", "auth_enabled"], "SECURITY_AUTH_ENABLED", bool)
    _set(["security", "jwt_secret"], "SECURITY_JWT_SECRET")
    _set(["security", "jwt_issuer"], "SECURITY_JWT_ISSUER")
    _set(["security", "jwt_exp_minutes"], "SECURITY_JWT_EXP_MINUTES", int)

    _set(["azure_openai", "endpoint"], "AZURE_OPENAI_ENDPOINT")
    _set(["azure_openai", "api_version"], "AZURE_OPENAI_API_VERSION")
    _set(["azure_openai", "primary_deployment"], "AZURE_OPENAI_PRIMARY_DEPLOYMENT")
    _set(["azure_openai", "fallback_deployment"], "AZURE_OPENAI_FALLBACK_DEPLOYMENT")
    _set(["azure_openai", "timeout"], "AZURE_OPENAI_TIMEOUT", int)
    _set(["azure_openai", "max_concurrency"], "AZURE_OPENAI_MAX_CONCURRENCY", int)

    _set(["redis", "app", "host"], "REDIS_APP_HOST")
    _set(["redis", "app", "port"], "REDIS_APP_PORT", int)
    _set(["redis", "app", "db"], "REDIS_APP_DB", int)
    _set(["redis", "app", "username"], "REDIS_APP_USERNAME")
    _set(["redis", "app", "password"], "REDIS_APP_PASSWORD")
    _set(["redis", "app", "tls"], "REDIS_APP_TLS", bool)
    _set(["redis", "app", "ssl_cert_reqs"], "REDIS_APP_SSL_CERT_REQS")
    _set(["redis", "app", "ssl_ca_certs"], "REDIS_APP_SSL_CA_CERTS")
    _set(["redis", "app", "namespace"], "REDIS_APP_NAMESPACE")

    _set(["redis", "worker", "host"], "REDIS_WORKER_HOST")
    _set(["redis", "worker", "port"], "REDIS_WORKER_PORT", int)
    _set(["redis", "worker", "db"], "REDIS_WORKER_DB", int)
    _set(["redis", "worker", "username"], "REDIS_WORKER_USERNAME")
    _set(["redis", "worker", "password"], "REDIS_WORKER_PASSWORD")
    _set(["redis", "worker", "tls"], "REDIS_WORKER_TLS", bool)
    _set(["redis", "worker", "ssl_cert_reqs"], "REDIS_WORKER_SSL_CERT_REQS")
    _set(["redis", "worker", "ssl_ca_certs"], "REDIS_WORKER_SSL_CA_CERTS")
    _set(["redis", "worker", "namespace"], "REDIS_WORKER_NAMESPACE")

    _set(["postgres", "enabled"], "POSTGRES_ENABLED", bool)
    _set(["postgres", "host"], "POSTGRES_HOST")
    _set(["postgres", "port"], "POSTGRES_PORT", int)
    _set(["postgres", "database"], "POSTGRES_DATABASE")
    _set(["postgres", "username"], "POSTGRES_USERNAME")
    _set(["postgres", "ssl_mode"], "POSTGRES_SSL_MODE")
    _set(["postgres", "min_pool_size"], "POSTGRES_MIN_POOL_SIZE", int)
    _set(["postgres", "max_pool_size"], "POSTGRES_MAX_POOL_SIZE", int)
    _set(["postgres", "connect_timeout_seconds"], "POSTGRES_CONNECT_TIMEOUT_SECONDS", int)
    _set(["postgres", "app_name"], "POSTGRES_APP_NAME")
    _set(["postgres", "schema_name"], "POSTGRES_SCHEMA_NAME")
    _set(["postgres", "memory_table"], "POSTGRES_MEMORY_TABLE")
    _set(["postgres", "chunk_table"], "POSTGRES_CHUNK_TABLE")
    _set(["postgres", "evaluation_table"], "POSTGRES_EVALUATION_TABLE")
    _set(["postgres", "default_top_k"], "POSTGRES_DEFAULT_TOP_K", int)
    _set(["postgres", "vector_index_type"], "POSTGRES_VECTOR_INDEX_TYPE")
    _set(["postgres", "embedding_dimensions"], "POSTGRES_EMBEDDING_DIMENSIONS", int)

    _set(["memory", "summary_queue_claim_idle_ms"], "MEMORY_SUMMARY_QUEUE_CLAIM_IDLE_MS", int)
    _set(["memory", "summary_queue_claim_batch_size"], "MEMORY_SUMMARY_QUEUE_CLAIM_BATCH_SIZE", int)

    _set(["middleware", "timeout_seconds"], "MIDDLEWARE_TIMEOUT_SECONDS", int)
    _set(["middleware", "max_in_flight_requests"], "MIDDLEWARE_MAX_IN_FLIGHT_REQUESTS", int)
    _set(["middleware", "rate_limit_requests"], "MIDDLEWARE_RATE_LIMIT_REQUESTS", int)
    _set(["middleware", "rate_limit_window_seconds"], "MIDDLEWARE_RATE_LIMIT_WINDOW_SECONDS", int)
    _set(
        ["middleware", "enable_distributed_rate_limit"],
        "MIDDLEWARE_ENABLE_DISTRIBUTED_RATE_LIMIT",
        bool,
    )
    _set(
        ["middleware", "distributed_rate_limit_prefix"], "MIDDLEWARE_DISTRIBUTED_RATE_LIMIT_PREFIX"
    )
    _set(
        ["middleware", "trusted_proxy_cidrs"],
        "MIDDLEWARE_TRUSTED_PROXY_CIDRS",
        lambda raw: [entry.strip() for entry in raw.split(",") if entry.strip()],
    )
    _set(
        ["middleware", "enable_distributed_backpressure"],
        "MIDDLEWARE_ENABLE_DISTRIBUTED_BACKPRESSURE",
        bool,
    )
    _set(["middleware", "distributed_backpressure_key"], "MIDDLEWARE_DISTRIBUTED_BACKPRESSURE_KEY")
    _set(
        ["middleware", "distributed_backpressure_lease_seconds"],
        "MIDDLEWARE_DISTRIBUTED_BACKPRESSURE_LEASE_SECONDS",
        int,
    )
    _set(["middleware", "enable_request_logging"], "MIDDLEWARE_ENABLE_REQUEST_LOGGING", bool)
    _set(["middleware", "enable_rate_limit"], "MIDDLEWARE_ENABLE_RATE_LIMIT", bool)
    _set(["middleware", "enable_timeout"], "MIDDLEWARE_ENABLE_TIMEOUT", bool)
    _set(["middleware", "enable_backpressure"], "MIDDLEWARE_ENABLE_BACKPRESSURE", bool)
    _set(["middleware", "enable_route_matching"], "MIDDLEWARE_ENABLE_ROUTE_MATCHING", bool)

    _set(["queue", "llm_async_enabled"], "LLM_ASYNC_ENABLED", bool)
    _set(["queue", "llm_queue_url"], "LLM_QUEUE_URL")
    _set(["queue", "llm_dlq_url"], "LLM_DLQ_URL")
    _set(["queue", "llm_result_table"], "LLM_RESULT_TABLE")
    _set(["queue", "llm_result_ttl_days"], "LLM_RESULT_TTL_DAYS", int)
    _set(["queue", "llm_receive_wait_seconds"], "LLM_RECEIVE_WAIT_SECONDS", int)
    _set(["queue", "llm_max_messages_per_poll"], "LLM_MAX_MESSAGES_PER_POLL", int)
    _set(
        ["queue", "llm_visibility_timeout_seconds"],
        "LLM_VISIBILITY_TIMEOUT_SECONDS",
        int,
    )
    _set(["queue", "llm_poll_sleep_seconds"], "LLM_POLL_SLEEP_SECONDS", float)
    _set(
        ["queue", "metrics_aggregation_queue_enabled"],
        "METRICS_AGGREGATION_QUEUE_ENABLED",
        bool,
    )
    _set(["queue", "metrics_aggregation_queue_url"], "METRICS_AGGREGATION_QUEUE_URL")
    _set(
        ["queue", "metrics_aggregation_receive_wait_seconds"],
        "METRICS_AGGREGATION_RECEIVE_WAIT_SECONDS",
        int,
    )
    _set(
        ["queue", "metrics_aggregation_max_messages_per_poll"],
        "METRICS_AGGREGATION_MAX_MESSAGES_PER_POLL",
        int,
    )
    _set(
        ["queue", "metrics_aggregation_visibility_timeout_seconds"],
        "METRICS_AGGREGATION_VISIBILITY_TIMEOUT_SECONDS",
        int,
    )
    _set(
        ["queue", "metrics_aggregation_poll_sleep_seconds"],
        "METRICS_AGGREGATION_POLL_SLEEP_SECONDS",
        float,
    )
    _set(["queue", "evaluation_queue_enabled"], "EVALUATION_QUEUE_ENABLED", bool)
    _set(["queue", "evaluation_queue_url"], "EVALUATION_QUEUE_URL")
    _set(["queue", "evaluation_receive_wait_seconds"], "EVALUATION_RECEIVE_WAIT_SECONDS", int)
    _set(
        ["queue", "evaluation_max_messages_per_poll"],
        "EVALUATION_MAX_MESSAGES_PER_POLL",
        int,
    )
    _set(
        ["queue", "evaluation_visibility_timeout_seconds"],
        "EVALUATION_VISIBILITY_TIMEOUT_SECONDS",
        int,
    )
    _set(
        ["queue", "evaluation_poll_sleep_seconds"],
        "EVALUATION_POLL_SLEEP_SECONDS",
        float,
    )
    _set(["io", "llm_max_concurrency"], "IO_LLM_MAX_CONCURRENCY", int)
    _set(["io", "embedding_max_concurrency"], "IO_EMBEDDING_MAX_CONCURRENCY", int)
    _set(["io", "retrieval_max_concurrency"], "IO_RETRIEVAL_MAX_CONCURRENCY", int)
    _set(["io", "redis_max_concurrency"], "IO_REDIS_MAX_CONCURRENCY", int)
    _set(["io", "bedrock_executor_workers"], "IO_BEDROCK_EXECUTOR_WORKERS", int)

    return config


@lru_cache()
def get_settings() -> Settings:
    """Load and cache merged application settings from the config directory."""
    config_files = sorted(APP_CONFIG_DIR.glob("*_config.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No config files found in {APP_CONFIG_DIR}")

    data = {}
    for config_file in config_files:
        data.update(_load_yaml_file(config_file))

    _load_aws_secrets_manager_env()
    data = _apply_env_overrides(data)
    return Settings(**data)


@lru_cache()
def get_prompts() -> dict:
    """Load and cache prompt definitions from the prompt config file."""
    prompt_path = APP_CONFIG_DIR / "prompt.yaml"
    return _load_yaml_file(prompt_path)


@lru_cache()
def get_evaluation_prompts() -> dict:
    """Load and cache offline evaluator (LLM-as-judge) prompts."""
    prompt_path = APP_CONFIG_DIR / "evaluation_prompt.yaml"
    return _load_yaml_file(prompt_path)
